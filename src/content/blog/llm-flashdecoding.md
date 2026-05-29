---
title: 'FlashDecoding：长上下文 Decode 阶段的并行化思路'
description: '解释为什么 Decode 阶段在长上下文下容易变成 Memory-bound，以及 FlashDecoding 如何把单请求 KV 读取拆分到多个 block 并行。'
category: '推理优化'
pubDate: '2026-05-29'
updatedDate: '2026-05-29'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Decode 阶段的形状](#二decode-阶段的形状)
3. [为什么长上下文 Decode 慢](#三为什么长上下文-decode-慢)
4. [FlashDecoding 的核心思路](#四flashdecoding-的核心思路)
5. [分块 Softmax 合并](#五分块-softmax-合并)
6. [伪代码](#六伪代码)
7. [适用场景](#七适用场景)
8. [面试表达](#八面试表达)
9. [总结](#九总结)

## 一、核心结论

FlashDecoding 主要面向自回归推理的 Decode 阶段。

- Prefill 一次处理多个 query token，矩阵较大，并行度高。
- Decode 每次通常只有一个 query token，但要 attend 到很长的历史 KV。
- 当 batch 较小、上下文很长时，单请求可用并行度不足。
- FlashDecoding 把 KV 序列维度切成多个 split，让多个 CUDA block 并行处理同一个请求。
- 最后把各 split 的局部 softmax 结果合并成全局结果。

## 二、Decode 阶段的形状

Prefill 阶段：

```text
Q: [prompt_len, head_dim]
K: [prompt_len, head_dim]
V: [prompt_len, head_dim]
```

Decode 阶段每次生成一个 token：

```text
Q: [1, head_dim]
K_cache: [context_len, head_dim]
V_cache: [context_len, head_dim]
```

计算：

```text
score = Q K_cache^T       # [1, context_len]
out = softmax(score) V    # [1, head_dim]
```

注意：`Q` 很小，`K/V` 很长。

## 三、为什么长上下文 Decode 慢

Decode 阶段每生成一个 token，都要读取历史 KV Cache。

如果上下文长度为 `n`，每层每个 head 都要读：

```text
K_cache: n x d
V_cache: n x d
```

这通常是 Memory-bound：

- 计算量不大。
- 读取 KV 的数据量随上下文长度线性增长。
- batch 小时，GPU 并行度不够。
- 每步生成之间有串行依赖，不能像训练那样一次并行所有 token。

## 四、FlashDecoding 的核心思路

普通 decode attention 可能让一个 block 处理一个 query 的完整 KV。

当 `context_len` 很长时，一个 block 干太多活，其他 SM 可能不够忙。

FlashDecoding 做法：

```text
把 K/V 序列切成多个 split
每个 split 由一个 block 计算局部 attention
最后把多个 split 的结果合并
```

例如：

```text
context_len = 8192
split size = 1024
split 数 = 8
```

同一个请求的一个 head 可以拆成 8 个 block 并行处理。

## 五、分块 Softmax 合并

每个 split 只能看到一部分 KV，因此只能得到局部：

```text
m_i = max(score_i)
l_i = sum(exp(score_i - m_i))
o_i = sum(exp(score_i - m_i) * V_i)
```

全局合并时：

```text
m = max_i(m_i)
l = sum_i(l_i * exp(m_i - m))
o = sum_i(o_i * exp(m_i - m)) / l
```

这和 Online Softmax 的思想一致。关键是不同 split 的 softmax 基准不同，需要用 `exp(m_i - m)` 重新缩放。

## 六、伪代码

```python
def flash_decoding(q, K_cache, V_cache, split_size):
    partial = []

    for K_i, V_i in split(K_cache, V_cache, split_size):
        # 每个 split 可以由不同 CUDA block 并行处理。
        s_i = q @ K_i.T
        m_i = max(s_i)
        p_i = exp(s_i - m_i)
        l_i = sum(p_i)
        o_i = p_i @ V_i
        partial.append((m_i, l_i, o_i))

    # 合并所有 split 的局部结果。
    m = max(m_i for m_i, _, _ in partial)
    l = sum(l_i * exp(m_i - m) for m_i, l_i, _ in partial)
    o = sum(o_i * exp(m_i - m) for m_i, _, o_i in partial) / l
    return o
```

实际实现会把 partial 结果写到临时 buffer，再由第二个 kernel 或同一 kernel 的后续阶段合并。

## 七、适用场景

FlashDecoding 更适合：

- batch size 较小。
- context length 很长。
- Decode 阶段成为瓶颈。
- 单个请求内部需要更多并行度。

不一定明显受益的场景：

- batch 已经很大，GPU 并行度足够。
- 上下文较短，拆分带来的额外合并开销超过收益。
- KV Cache 访问本身受分页、布局或带宽限制严重。

## 八、面试表达

FlashDecoding 可以这样回答：

1. Decode 阶段每次只有一个 query token，但要读取整个历史 KV Cache。
2. 当 batch 小、上下文长时，单请求 attention 并行度不足，容易 Memory-bound。
3. FlashDecoding 沿 sequence 维度把 KV Cache 切成多个 split，让多个 block 并行处理同一个 query。
4. 每个 split 计算局部 max、sum 和局部输出，最后用 Online Softmax 的缩放公式合并。
5. 它主要优化长上下文、小 batch Decode 的吞吐和延迟。

## 九、总结

FlashAttention 更常用于 Prefill 这类多 query 场景；FlashDecoding 针对 Decode 中单 query、长 KV 的并行度不足问题。两者都依赖分块和 Online Softmax，但优化目标不同。
