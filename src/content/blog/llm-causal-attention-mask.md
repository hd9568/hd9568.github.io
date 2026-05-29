---
title: 'Causal Attention：自回归推理中的 Mask 机制'
description: '解释 Causal Mask 为什么存在，如何保证 Decode 阶段只能看历史 token，并用简洁代码展示 mask 的构造方式。'
category: '推理优化'
pubDate: '2026-06-26'
updatedDate: '2026-06-26'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [自回归生成的约束](#二自回归生成的约束)
3. [Causal Mask 的形状](#三causal-mask-的形状)
4. [训练阶段如何使用](#四训练阶段如何使用)
5. [Prefill 与 Decode 的差异](#五prefill-与-decode-的差异)
6. [代码示例](#六代码示例)
7. [实现细节](#七实现细节)
8. [面试表达](#八面试表达)
9. [总结](#九总结)

## 一、核心结论

Causal Attention 的目的很简单：第 `i` 个 token 不能看到第 `i+1` 及之后的 token。

- 自回归模型按从左到右生成文本。
- 训练时虽然整段文本同时输入，但每个位置只能用历史 token 预测下一个 token。
- Causal Mask 把未来位置的 attention score 置为负无穷。
- Softmax 后，被 mask 的位置概率接近 0。
- Decode 阶段每次只生成一个新 token，本质上天然只访问历史 KV。

## 二、自回归生成的约束

语言模型生成序列：

```text
P(x1, x2, ..., xn) = P(x1) P(x2|x1) P(x3|x1,x2) ... P(xn|x1...x_{n-1})
```

第 `t` 个 token 只能依赖之前 token。

如果训练时不加 Causal Mask，第 `t` 个位置就可能偷看未来答案，训练目标会被破坏。

## 三、Causal Mask 的形状

序列长度为 4 时，允许关注的位置如下：

```text
query\key   0  1  2  3
0           ✓  x  x  x
1           ✓  ✓  x  x
2           ✓  ✓  ✓  x
3           ✓  ✓  ✓  ✓
```

矩阵形式：

```text
[[0, -inf, -inf, -inf],
 [0,    0, -inf, -inf],
 [0,    0,    0, -inf],
 [0,    0,    0,    0]]
```

把它加到 attention scores 上：

```text
masked_scores = scores + causal_mask
```

未来位置变成 `-inf`，Softmax 后概率为 0。

## 四、训练阶段如何使用

训练时通常一次输入完整序列：

```text
input:  [BOS, I, like, GPU]
target: [I, like, GPU, EOS]
```

虽然 `I like GPU` 同时进入模型，但位置 `I` 不能看 `like` 和 `GPU`。Causal Mask 保证每个位置只使用左侧上下文。

## 五、Prefill 与 Decode 的差异

### Prefill

Prefill 阶段一次处理 prompt 中的所有 token，需要完整的 causal attention。

```text
prompt tokens: x0, x1, x2, ..., xn
```

每个 token 都可以看自己和之前 token。此时 causal mask 是一个下三角结构。

### Decode

Decode 阶段每次只处理一个新 token。

```text
step t: new query q_t attends to K/V of x0...x_t
```

由于 KV Cache 中只有历史 token 和当前 token，不存在未来 token，因此通常不需要构造完整 `n x n` mask。实现中更多是处理有效长度、分页地址和 batch 内不同序列长度。

## 六、代码示例

PyTorch 中构造 Causal Mask：

```python
import torch


def causal_mask(seq_len, device='cuda'):
    # 上三角为 True，表示这些位置需要被 mask。
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    return mask


def causal_attention(q, k, v):
    # q, k, v: [batch, heads, seq_len, head_dim]
    seq_len = q.shape[-2]
    scores = q @ k.transpose(-2, -1)

    mask = causal_mask(seq_len, q.device)

    # 被 mask 的位置设为很小的数，softmax 后接近 0。
    scores = scores.masked_fill(mask, float('-inf'))

    probs = torch.softmax(scores, dim=-1)
    return probs @ v
```

这段代码表达清楚，但实际高性能 kernel 不一定真的生成完整 mask。很多实现会在 kernel 内根据 `query_idx < key_idx` 判断是否需要屏蔽。

## 七、实现细节

### 1. `-inf` 的数值处理

FP16 下直接使用 `-inf` 通常可以，但某些 kernel 会用一个足够小的常数，例如 `-65504`。

### 2. Padding Mask 与 Causal Mask

批量输入中还有 padding：

```text
[hello, world, PAD, PAD]
```

Padding Mask 防止关注无效 token；Causal Mask 防止关注未来 token。二者可以合并。

### 3. 不同请求长度

Serving 场景中 batch 内请求长度不同，Decode kernel 需要知道每个请求的当前有效长度，避免读到未初始化 KV。

## 八、面试表达

Causal Attention 可以这样回答：

1. 自回归语言模型从左到右生成，第 `t` 个 token 只能依赖 `0..t` 的上下文。
2. 训练时整段序列并行输入，所以需要 Causal Mask 防止当前位置看到未来 token。
3. Causal Mask 通常是上三角 mask，把未来位置的 score 置为 `-inf`。
4. Softmax 后这些位置概率为 0。
5. Decode 阶段每次只处理新 token，依赖历史 KV Cache，因此不需要完整的 `n x n` mask，但要处理有效长度和分页 KV。

## 九、总结

Causal Mask 是自回归模型成立的基础。训练阶段它保证不偷看未来；推理阶段它体现为每个新 token 只访问历史 KV。理解这一点，才能继续理解 Prefill、Decode、KV Cache 和 PagedAttention。
