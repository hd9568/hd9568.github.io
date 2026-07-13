---
title: '大模型采样优化：从排序版 Top-K/Top-P 到 FlashInfer Dual-Pivot Rejection Sampling'
description: '结合 FlashInfer 采样优化思路，讲解 LLM sampling 为什么会成为瓶颈，Top-K/Top-P/Min-P 的语义，排序路径的问题，以及 sorting-free GPU kernel 的核心算法。'
category: 'Research & Work'
pubDate: '2026-07-13T16:52:00+08:00'
updatedDate: '2026-07-13T16:52:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Sampling 在推理链路中的位置](#二sampling-在推理链路中的位置)
3. [Top-K、Top-P、Min-P 分别是什么](#三top-ktop-pmin-p-分别是什么)
4. [朴素 PyTorch 实现为什么慢](#四朴素-pytorch-实现为什么慢)
5. [Inverse Transform Sampling](#五inverse-transform-sampling)
6. [Rejection Sampling](#六rejection-sampling)
7. [Dual-Pivot Rejection Sampling](#七dual-pivot-rejection-sampling)
8. [GPU kernel 视角](#八gpu-kernel-视角)
9. [简化代码示例](#九简化代码示例)
10. [工程注意点](#十工程注意点)
11. [面试表达](#十一面试表达)
12. [总结](#十二总结)

## 一、核心结论

大模型 sampling 是从 logits / probabilities 中选下一个 token。

在大 vocabulary 模型中，sampling 可能成为推理瓶颈。原因是常见 Top-K / Top-P 实现依赖：

```text
sort
softmax
cumsum
mask
scatter / gather
```

当 vocab size 是 32K、100K 甚至更大时，每步 decode 都排序会非常贵。

FlashInfer 文章的核心思路是：

```text
不要为了 Top-K / Top-P 先完整排序；
用 sorting-free 的 rejection sampling 在 GPU 上直接采样。
```

它提出 Dual-Pivot Rejection Sampling，让过滤采样在理论上有对数轮数收敛保证，并通过 fused sampling kernel 减少多次 kernel launch。

## 二、Sampling 在推理链路中的位置

自回归推理每一步大致是：

```text
hidden state -> LM head -> logits -> sampling -> next token
```

其中：

```text
logits: [batch_size, vocab_size]
```

例如：

```text
batch_size = 128
vocab_size = 151936
```

sampling 每步都要处理 `128 * 151936` 个 logits。

如果采样阶段排序，会产生：

```text
O(V log V)
```

其中 `V` 是 vocabulary size。Decode 每生成一个 token 都要做一次，所以它会影响 inter-token latency。

## 三、Top-K、Top-P、Min-P 分别是什么

### Top-K

Top-K 保留概率最高的 K 个 token。

例如：

```text
K = 50
```

只在概率最大的 50 个 token 中采样，其他 token 置为不可选。

### Top-P

Top-P 也叫 nucleus sampling。

它先按概率从大到小排序，然后保留最小集合，使累计概率超过阈值 `p`。

例如：

```text
p = 0.9
```

保留累计概率刚好超过 0.9 的那些 token。

### Min-P

Min-P 根据最大概率设置阈值：

```text
threshold = p_base * p_max
```

保留概率不低于这个阈值的 token。

它可以过滤掉相对最大 token 太小的尾部概率。

### 常见组合

线上常见：

```text
Top-K + Top-P
Top-P + temperature
Min-P + Top-P
```

不同策略的目标都是控制生成质量和随机性。

## 四、朴素 PyTorch 实现为什么慢

一个典型 Top-K + Top-P 写法：

```python
def apply_top_k_top_p(logits, top_p, top_k):
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    # Top-K
    top_k_mask_value = logits_sort.gather(
        1, (logits_sort.size(1) - top_k).unsqueeze(1)
    )
    top_k_mask = logits_sort < top_k_mask_value
    logits_sort.masked_fill_(top_k_mask, -float("inf"))

    # Top-P
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = probs_sum <= 1 - top_p.unsqueeze(1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # scatter 回原顺序
    src = torch.arange(logits_idx.shape[-1], device=logits_idx.device).expand_as(logits_idx)
    logits_idx_inv = torch.empty_like(logits_idx).scatter_(dim=-1, index=logits_idx, src=src)
    return torch.gather(logits_sort, dim=-1, index=logits_idx_inv)
```

问题：

- `sort` 昂贵。
- `softmax` 和 `cumsum` 需要全 vocab 扫描。
- mask / gather / scatter 多次访存。
- 多个 PyTorch op 可能对应多个 kernel launch。
- batch 越大、vocab 越大，开销越明显。

FlashInfer 的文章指出，在 vLLM 1xH100 配置中，优化 sampling kernel 可以让整体 sampling time 降低超过 50%。

## 五、Inverse Transform Sampling

最基础的 categorical sampling 是 inverse transform sampling。

给定概率：

```text
p = [0.1, 0.2, 0.3, 0.4]
```

先做前缀和 CDF：

```text
cdf = [0.1, 0.3, 0.6, 1.0]
```

随机采样：

```text
u ~ Uniform(0, 1)
```

如果：

```text
u = 0.52
```

找到第一个 `cdf[i] >= u`：

```text
i = 2
```

所以采样 token 2。

GPU 上可以用 block-level reduce / scan 计算局部 prefix sum。

当 vocab 很大时，一个 block 处理不了全部 token，可以分块：

```text
block 0: token 0-4095
block 1: token 4096-8191
...
```

先计算每块概率和，找到随机数落在哪个 block，再在该 block 内做 prefix sum 找 token。

## 六、Rejection Sampling

Top-P/Top-K 不是简单从全量概率采样，而是要先过滤 token。

FlashInfer 文章用 rejection sampling 避免完整排序。

以 Top-P 为例，简化思想：

```text
维护一个 pivot。
只考虑概率大于 pivot 的 token。
每轮采样一个 token，根据它的概率更新 pivot。
如果当前保留集合已经满足 Top-P 条件，就接受采样结果。
否则继续提高 pivot，拒绝更多低概率 token。
```

为什么这能避免排序？

因为它不需要知道完整从大到小顺序，只需要不断用 pivot 缩小候选集合。

普通 rejection sampling 的问题是：

```text
轮数没有严格保证，不同分布下延迟可能波动。
```

这会影响线上服务的 inter-token latency 稳定性。

## 七、Dual-Pivot Rejection Sampling

Dual-Pivot Rejection Sampling 用两个 pivot 改善收敛。

维护区间：

```text
low
high
```

其中：

```text
f(low) = 0
f(high) = 1
```

`f(x)` 表示当前阈值是否已经满足过滤条件。

每轮：

1. 在 `(low, +∞)` 的概率范围中采样一个 token。
2. 设 `pivot1 = p_j`。
3. 设 `pivot2 = (pivot1 + high) / 2`。
4. 根据 `f(pivot1)` 和 `f(pivot2)` 更新区间。

三种情况：

```text
f(pivot1) = 1
  接受 token，返回。

f(pivot1) = 0 且 f(pivot2) = 1
  low = pivot1
  high = pivot2

f(pivot1) = 0 且 f(pivot2) = 0
  low = pivot2
```

关键性质：

```text
如果没有接受 token，区间至少缩小一半。
```

因此轮数有对数级上界：

```text
O(log(1 / epsilon))
```

其中 `epsilon` 和浮点最小可区分值有关。

## 八、GPU kernel 视角

高性能 sampling kernel 需要做到：

- 一次 kernel 内完成过滤和采样。
- 避免全排序。
- 尽量减少全 vocab 多次读写。
- 使用 block-level reduce / scan。
- batch 内每个 request 并行。
- 对大 vocab 分块处理。
- 尽量 early stop。

一个 block 可能处理一个 request 的一个 vocab tile：

```text
BLOCK_SIZE = NUM_THREADS * ITEMS_PER_THREAD
```

例如：

```text
NUM_THREADS = 1024
ITEMS_PER_THREAD = 4
BLOCK_SIZE = 4096
```

对 vocab > 4096 的模型，就分多个 tile。

FlashInfer 提到使用 CUB / CCCL 的：

- `BlockReduce`
- `BlockScan`
- `BlockAdjacentDifference`

这些都是 GPU block 内并行原语。

## 九、简化代码示例

下面是 CPU 伪代码，用来理解 Top-K rejection sampling。它不是高性能实现。

```python
import random

def sample_with_pivot(probs, pivot):
    total = sum(p for p in probs if p > pivot)
    u = random.random() * total
    acc = 0.0
    for i, p in enumerate(probs):
        if p <= pivot:
            continue
        acc += p
        if acc >= u:
            return i
    return len(probs) - 1

def topk_valid(probs, pivot, k):
    cnt = sum(1 for p in probs if p > pivot)
    return cnt <= k

def topk_rejection_sample(probs, k):
    pivot = 0.0
    while True:
        idx = sample_with_pivot(probs, pivot)
        p = probs[idx]
        if topk_valid(probs, p, k):
            return idx
        pivot = p
```

Dual-pivot 简化版：

```python
def dual_pivot_topk_sample(probs, k):
    low = 0.0
    high = max(probs)

    while True:
        idx = sample_with_pivot(probs, low)
        p1 = probs[idx]
        p2 = (p1 + high) / 2.0

        if topk_valid(probs, p1, k):
            return idx

        if topk_valid(probs, p2, k):
            low = p1
            high = p2
        else:
            low = p2
```

真实 GPU 实现还要处理：

- logits 到概率的 online softmax。
- Top-P / Min-P 的 validity function。
- 随机数状态。
- 数值稳定性。
- 多 batch 并行。
- prefix sum 的非严格单调问题。

## 十、工程注意点

### 1. prefix sum 的数值问题

FlashInfer 文章特别提到：浮点加法非结合、非交换，parallel prefix sum 即使输入非负，也不一定严格单调。

这可能导致采样定位错误。

工程上需要做额外保护：

- 避免因为浮点误差找不到 token。
- 对边界情况兜底。
- 保证概率归一化和随机数范围稳定。

### 2. 排序路径不一定永远差

如果 vocab 很小，或者 batch 很小，排序路径可能足够快。

优化要看 profiling。

### 3. sampling 和 CUDA Graph

Sampling 可能包含随机性和动态过滤路径，放进 CUDA Graph 前要确认：

- 随机状态如何更新。
- shape 是否稳定。
- kernel 是否 graph-safe。

### 4. 多策略组合

线上通常不是单独 Top-K 或 Top-P，而是组合：

```text
temperature + repetition penalty + top-k + top-p + min-p
```

最优实现需要尽量融合这些步骤，减少多次全 vocab 扫描。

## 十一、面试表达

可以这样讲 sampling 优化：

```text
LLM sampling 在大 vocabulary 下会成为瓶颈，朴素 Top-K/Top-P 通常需要 sort、softmax、cumsum 和 mask。
排序复杂度高，而且会引入多次显存访问和多 kernel launch。
FlashInfer 的思路是用 sorting-free sampling，通过 inverse transform sampling 和 rejection sampling 直接在过滤后的分布中采样。
Dual-Pivot Rejection Sampling 用 low/high 两个 pivot 缩小候选概率区间，失败时至少把区间缩小一半，从而给出对数轮数收敛保证。
```

如果问 GPU 实现：

```text
GPU kernel 通常让每个 request 的 vocab 分块处理，用 block reduce 求局部概率和，用 block scan 找采样位置。
大 vocab 下按 tile 顺序累加，发现随机数落入某个 tile 后再在 tile 内定位 token。
关键是避免全排序和多次全量读写。
```

## 十二、总结

采样优化的核心不是改变采样语义，而是避免不必要排序。

```text
朴素路径：sort -> softmax -> cumsum -> mask -> sample
FlashInfer 路径：sorting-free rejection sampling -> fused GPU kernel
```

它适合：

- 大 vocab。
- 高 batch。
- 高并发推理。
- 对 inter-token latency 稳定性要求高的服务。

需要注意：

- 数值稳定。
- 随机数正确性。
- 多策略组合。
- graph capture 兼容性。
- fallback 路径。
