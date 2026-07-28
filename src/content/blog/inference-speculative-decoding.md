---
title: 'Speculative Decoding：用草稿模型减少串行 Decode 次数'
description: '推导投机解码的 Draft、Verify、Accept/Reject 流程，解释无损采样公式、加速上限、接受率，并结合 vLLM 的配置、N-gram Proposer 和 Rejection Sampler 说明工程实现。'
category: '推理优化'
pubDate: '2026-07-28T12:40:00+08:00'
updatedDate: '2026-07-28T12:40:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [串行 Decode 的瓶颈](#一串行-decode-的瓶颈)
2. [投机解码的基本流程](#二投机解码的基本流程)
3. [Greedy 模式如何验证](#三greedy-模式如何验证)
4. [Sampling 模式如何保持分布不变](#四sampling-模式如何保持分布不变)
5. [接受率和加速模型](#五接受率和加速模型)
6. [最小实现](#六最小实现)
7. [Draft 方法的选择](#七draft-方法的选择)
8. [vLLM 中的实现结构](#八vllm-中的实现结构)
9. [KV Cache 与调度细节](#九kv-cache-与调度细节)
10. [什么时候会变慢](#十什么时候会变慢)
11. [评估和调参](#十一评估和调参)
12. [总结](#十二总结)

## 一、串行 Decode 的瓶颈

标准自回归生成一次只能得到一个新 token：

```text
x1 = target(prompt)
x2 = target(prompt, x1)
x3 = target(prompt, x1, x2)
...
```

生成 `K` 个 token 需要 `K` 次目标模型前向。Decode 通常是小 batch、低算术强度的访存密集过程，每次前向都要读取大量权重和历史 KV Cache。

关键观察是：

> 目标模型虽然必须逐 token 定义概率分布，但给定一串候选 token 后，可以在一次并行前向中验证多个位置。

投机解码用更便宜的 Proposer 先猜 `K` 个 token，再让目标模型一次验证这些猜测，从而减少目标模型的串行调用次数。

## 二、投机解码的基本流程

设：

- `q`：Draft Model 的分布。
- `p`：Target Model 的分布。
- `K`：一次提出的 Draft token 数。

流程如下：

```text
1. Draft:
   d1, d2, ..., dK = proposer(prefix)

2. Verify:
   target 一次计算 K 个位置的 logits

3. Accept/Reject:
   从左到右判断 d1 ... dK

4. Commit:
   接受连续前缀；遇到第一个拒绝位置后停止

5. Continue:
   使用已提交 token 进入下一轮
```

为什么只接受连续前缀？因为一旦 `d_i` 被拒绝，后面的 `d_{i+1}` 是在错误历史上由 Draft 生成的，条件分布已经失效。

### 2.1 全部接受时的 Bonus Token

若 `K` 个 Draft token 全部通过，目标模型验证时通常还计算了下一个位置的分布，因此可以额外采样一个 Bonus Token：

```text
一次 Target Verify 最多提交 K + 1 个 token
```

## 三、Greedy 模式如何验证

Greedy Decoding 每个位置选择：

```text
argmax p(x | prefix)
```

验证非常直接：

```python
for i in range(K):
    target_token = target_argmax[i]
    if draft_tokens[i] != target_token:
        output.append(target_token)
        break
    output.append(draft_tokens[i])
else:
    output.append(bonus_target_token)
```

只要第一个不一致位置改用 Target 的 token，最终结果与逐 token 运行 Target Greedy 完全相同。

例子：

```text
Draft:  [A, B, X, Y]
Target: [A, B, C, D]

提交: [A, B, C]
```

`X` 被拒绝后，`Y` 也不能使用。下一轮从 `[A, B, C]` 继续。

## 四、Sampling 模式如何保持分布不变

直接比较 Draft token 与 Target 采样 token 会改变输出分布。标准投机采样使用拒绝采样校正。

Draft 在位置 `i` 提出 token `x`，接受概率为：

```text
a(x) = min(1, p(x) / q(x))
```

若均匀随机数 `u <= a(x)`，接受 `x`。

若拒绝，不能简单地从 `p` 重新采样，因为高概率区域已经通过接受步骤获得了一部分质量。应从残差分布采样：

```text
p_residual(x) =
    max(p(x) - q(x), 0)
    / sum_y max(p(y) - q(y), 0)
```

这样最终 token 的边缘分布仍等于 `p`。

### 4.1 为什么这个公式无损

对任意 token `x`，通过 Draft 提出并接受它的概率是：

```text
q(x) * min(1, p(x) / q(x))
= min(q(x), p(x))
```

Target 中尚未覆盖的概率质量是：

```text
p(x) - min(q(x), p(x))
= max(p(x) - q(x), 0)
```

拒绝后的残差采样补回这部分质量，所以总分布恢复为 `p(x)`。

“无损”指采样分布不变，不代表浮点实现逐 bit 相同。并行计算、随机数消费顺序和数值精度仍可能影响可复现性。

## 五、接受率和加速模型

设每个 Draft token 的条件接受率近似为 `alpha`，一次提出 `K` 个 token。一次 Verify 期望提交的 token 数为：

```text
E[L] = 1 + alpha + alpha^2 + ... + alpha^K
     = (1 - alpha^(K + 1)) / (1 - alpha)
```

其中最后一项包含全部接受后的 Bonus Token。

例如 `K = 4`：

| 接受率 `alpha` | `E[L]` |
| ---: | ---: |
| 0.5 | 1.94 |
| 0.7 | 2.77 |
| 0.9 | 4.10 |
| 1.0 | 5.00 |

但 `E[L]` 不是实际加速比。还要计算：

```text
T_round =
    T_draft(K)
  + T_target_verify(K)
  + T_accept
  + T_scheduler
```

近似加速比：

```text
speedup ≈ E[L] * T_target_decode(1) / T_round
```

Target Verify 处理 `K` 个位置，成本高于单 token Decode，但通常低于 `K` 次串行 Decode；Draft 和验证开销必须足够小，投机才有收益。

### 5.1 `K` 不是越大越好

增大 `K`：

- 提高单轮最多提交 token 数。
- 增加 Draft 计算。
- 增加 Verify token 数和临时内存。
- 使后部 token 被执行却因前部拒绝而浪费的概率上升。

最佳 `K` 由接受率和硬件执行效率共同决定。

## 六、最小实现

下面展示 Greedy 投机解码的控制逻辑：

```python
import torch


@torch.inference_mode()
def speculative_greedy_step(
    target,
    draft,
    input_ids: torch.Tensor,
    num_draft_tokens: int,
) -> torch.Tensor:
    prefix = input_ids
    proposals = []

    # Draft 仍是自回归的，但模型更小或 Proposer 更便宜。
    for _ in range(num_draft_tokens):
        draft_logits = draft(prefix).logits[:, -1]
        token = draft_logits.argmax(dim=-1, keepdim=True)
        proposals.append(token)
        prefix = torch.cat([prefix, token], dim=1)

    proposed = torch.cat(proposals, dim=1)

    # 一次 Target 前向验证所有候选位置。
    verify_input = torch.cat([input_ids, proposed], dim=1)
    logits = target(verify_input).logits

    start = input_ids.shape[1] - 1
    target_tokens = logits[:, start : start + num_draft_tokens + 1].argmax(
        dim=-1
    )

    accepted = []
    for i in range(num_draft_tokens):
        if proposed[0, i] != target_tokens[0, i]:
            accepted.append(target_tokens[:, i : i + 1])
            break
        accepted.append(proposed[:, i : i + 1])
    else:
        # K 个候选全通过，提交额外的 Target token。
        accepted.append(
            target_tokens[:, num_draft_tokens : num_draft_tokens + 1]
        )

    return torch.cat(accepted, dim=1)
```

这个示例为了清晰重新计算了完整前缀。实际框架会复用 Target 和 Draft 的 KV Cache，并只执行新增位置。

## 七、Draft 方法的选择

### 7.1 独立小模型

使用与 Target 词表兼容的小模型：

```text
优点：实现直观，候选质量通常较稳定
缺点：额外权重、显存、KV Cache 和调度成本
```

Draft 越大，接受率通常越高，但自身成本也越高。

### 7.2 N-gram Prompt Lookup

在已有 token 序列中查找相同 N-gram，并把其后 token 当作候选：

```text
已有上下文：... A B C D E ...
当前后缀：      A B C
候选：              D E
```

它不需要 Draft Model，适合代码、结构化文本和存在重复片段的输入。自然语言无重复时接受率可能较低。

### 7.3 Medusa / MTP

在主模型上增加预测未来多个 token 的 Head，或训练 Multi-Token Prediction 模块：

- 不需要完整独立 Draft Model。
- 可共享主模型 Hidden State。
- 需要专门训练的权重。
- 候选可能是树结构而不是单链。

### 7.4 EAGLE

EAGLE 类方法在特征空间预测后续状态，再生成候选 token。相比只看 token ID，它可利用更丰富的 Target 特征，但实现和模型适配更复杂。

### 7.5 Suffix/历史响应匹配

把历史请求或当前 Prompt 的后缀结构建立索引，根据最长匹配提出候选。它没有额外模型计算，但收益高度依赖工作负载重复性。

## 八、vLLM 中的实现结构

vLLM 用 `SpeculativeConfig` 统一描述：

```text
method                  Draft 方法
model                   Draft Model 或特殊 Proposer
num_speculative_tokens  每轮最大候选数
draft_tensor_parallel_size
prompt_lookup_min/max   N-gram 匹配范围
rejection_sample_method 验证采样方法
```

框架可根据配置选择 Draft Model、N-gram、MTP、EAGLE、Medusa 等路径。

### 8.1 N-gram Proposer

`NgramProposer` 的核心输入是：

```text
每个请求已生成的 token
当前有效 token 数
每轮最多候选数 K
N-gram 最小/最大窗口
```

它在 CPU 侧批量搜索匹配片段，并返回每个请求的候选 token 列表。实现中会预分配输出缓冲区，避免每轮重复分配；只有 token 总量足够大时才值得启用更多 CPU 线程。

简化搜索：

```python
def ngram_propose(tokens, min_n, max_n, k):
    for n in range(max_n, min_n - 1, -1):
        suffix = tokens[-n:]
        for start in range(len(tokens) - n - 1, -1, -1):
            if tokens[start : start + n] == suffix:
                candidate_start = start + n
                return tokens[candidate_start : candidate_start + k]
    return []
```

### 8.2 Rejection Sampler

Target Forward 产生验证 logits 后，`RejectionSampler` 会：

1. 应用 Temperature、Penalty 等采样参数。
2. 读取 Draft token 和可选 Draft logits。
3. 执行接受/拒绝采样。
4. 返回每个请求实际提交的 token 数。
5. 按需要计算 Logprobs。

这部分通常使用 GPU Kernel 批量处理，避免把每个位置的概率搬回 CPU。

## 九、KV Cache 与调度细节

### 9.1 Lookahead Slot

候选 token 可能需要临时 KV slot。调度器要预留 Lookahead 空间：

```text
普通 Decode：当前新 token 的 slot
投机 Decode：当前 token + 多个候选位置的 slot
```

若 KV Cache 很紧张，投机解码会降低可容纳的并发数。

### 9.2 拒绝后的 KV 处理

Target 已为候选位置执行前向，但第 `j` 个候选被拒绝时：

- `[0, j)` 的已接受 KV 可以提交。
- 第 `j` 个位置使用校正 token 的状态。
- 后续候选位置的 KV 不能保留为有效序列状态。

框架需要调整序列长度、Block Table 和 slot 元数据，不能只丢弃 token ID。

### 9.3 与 Continuous Batching

不同请求接受长度不同：

```text
请求 A：提交 5 token
请求 B：提交 2 token
请求 C：提交 1 token
```

下一轮 batch 的逻辑位置和 KV 长度随请求变化。调度器必须按请求记录 `num_scheduled_tokens` 和实际 `num_sampled`。

### 9.4 Draft 与 Target 的并行配置

Target 可能采用多卡 Tensor Parallel，而 Draft 很小。若 Draft 也跟随相同 TP：

- 每轮增加通信。
- 小矩阵在多卡上效率可能更差。

因此框架通常允许 Draft 使用不同的 TP 大小，但这会引入设备放置和数据同步问题。

## 十、什么时候会变慢

- Draft 接受率低。
- Draft Model 过大。
- `K` 过大，后部验证经常浪费。
- 原始 Decode 已有很大 batch，Target 计算利用率较高。
- KV Cache 紧张，Lookahead 降低并发。
- 采样参数复杂，验证 Kernel 开销高。
- Target Verify 没有高效的多 token Kernel。
- Draft 与 Target 跨设备传输。
- 输出很短，初始化和 Warmup 开销无法摊薄。

投机解码更适合低到中等并发、单请求延迟敏感、Target Decode 明显 Memory-bound 的场景。高吞吐大 batch 下，收益常会下降。

## 十一、评估和调参

需要记录：

```text
acceptance rate：候选 token 接受比例
accepted length：每轮连续接受长度
emitted tokens / target verify
draft latency
target verify latency
rejection sampling latency
end-to-end TPOT
token throughput
额外 KV Cache 占用
```

调参建议：

1. 先测无投机的 Target 基线。
2. 从较小 `K` 开始，例如 2 或 3。
3. 分开测 Draft、Verify 和 Accept 时间。
4. 按 Prompt 类型和采样参数统计接受率，不只看全局平均。
5. 比较不同并发度；单请求加速不能代表在线吞吐收益。
6. 确认输出质量或采样分布保持预期。

接受率高但不加速，说明 Draft/Verify 开销过大；接受率低但仍加速，可能是 Proposer 极其便宜。最终判断依据必须是端到端指标。

## 十二、总结

Speculative Decoding 用便宜方法先提出多个候选，再让 Target 一次并行验证：

```text
Draft K 个 token
-> Target 一次 Verify
-> 接受连续前缀
-> 拒绝时用残差分布校正
-> 最多提交 K + 1 个 token
```

它不近似 Target 的最终分布，而是通过接受/拒绝规则保持 Greedy 结果或采样分布不变。实际收益由接受率、Draft 成本、Target Verify 效率、KV Lookahead 和并发度共同决定，不能用候选数量直接推断加速比。
