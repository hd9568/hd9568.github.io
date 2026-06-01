---
title: 'FlashAttention-2 / 3：减少非矩阵乘开销与提升 Tensor Core 利用率'
description: '解释 FlashAttention-2 和 FlashAttention-3 相比 FlashAttention-1 的优化方向：更好的并行划分、更少 non-matmul 开销和更高硬件利用率。'
category: '推理优化'
pubDate: '2026-05-29T14:10:00+08:00'
updatedDate: '2026-05-29T14:10:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [FlashAttention-1 的限制](#二flashattention-1-的限制)
3. [Non-matmul 开销](#三non-matmul-开销)
4. [FlashAttention-2 的改进](#四flashattention-2-的改进)
5. [FlashAttention-3 的方向](#五flashattention-3-的方向)
6. [对推理的意义](#六对推理的意义)
7. [面试表达](#七面试表达)
8. [总结](#八总结)

## 一、核心结论

FlashAttention-2 / 3 仍然是精确 Attention，不是近似算法。

- FlashAttention-1 已经解决了大量 HBM IO 问题。
- FlashAttention-2 重点是改进并行划分，减少非矩阵乘操作开销，提高 GPU 利用率。
- FlashAttention-3 面向更新硬件，进一步利用 Hopper 架构特性和异步流水。
- 优化目标从“少访问 HBM”进一步推进到“让 Tensor Core 更忙、让调度更高效”。

## 二、FlashAttention-1 的限制

FlashAttention-1 通过 tiling 避免保存完整 attention matrix，但仍存在一些问题：

- 单个 thread block 负责的工作划分不一定充分利用 SM。
- Softmax、rescale、mask、边界判断等 non-matmul 操作占比仍然可观。
- Q/K/V block 的切分方式影响并行度和数据复用。
- 对不同 sequence length、head dimension 的适配需要更细的调度策略。

矩阵乘可以用 Tensor Core 很快完成，但 softmax 和归一化这类操作通常走普通 CUDA core 或 scalar/vector 指令，吞吐远低于 Tensor Core。

## 三、Non-matmul 开销

Attention 中并不只有 GEMM。

```text
QK^T        -> matmul
scale       -> non-matmul
mask        -> non-matmul
rowmax      -> non-matmul reduction
exp/sum     -> non-matmul reduction
rescale     -> non-matmul
PV          -> matmul
```

如果 matmul 已经很快，non-matmul 部分就会成为新的瓶颈。

优化方向：

- 减少 rescale 次数。
- 减少 shared memory 读写。
- 减少同步。
- 改善 warp/thread block 的工作分配。
- 让更多时间花在 Tensor Core 可处理的矩阵乘上。

## 四、FlashAttention-2 的改进

FlashAttention-2 的核心方向是更好的并行化和更少额外开销。

### 1. 更好的 work partitioning

把一个 attention 计算拆分给更多 thread block 或 warp，提升并行度。

尤其在 batch/head 数较少、sequence length 较长时，如果并行粒度不够，SM 可能吃不满。

### 2. 减少 non-matmul FLOPs

Online Softmax 中存在最大值更新、分母更新、输出 rescale。FlashAttention-2 尽量减少这些非矩阵乘相关操作。

### 3. 改善 warp 级切分

不同 warp 负责不同数据块，减少不必要的数据交换和同步。

简化理解：

```text
FlashAttention-1: 已经 IO-aware
FlashAttention-2: 更好的 parallelism-aware 和 work-partition-aware
```

## 五、FlashAttention-3 的方向

FlashAttention-3 进一步面向 Hopper 这类新架构优化。

常见关键词：

- Tensor Core 更高利用率。
- 异步数据搬运。
- Warpgroup 级别矩阵乘。
- 更好的 producer-consumer pipeline。
- FP8 等低精度路径。

可以把它理解成：

```text
用更贴近硬件的流水线，让数据搬运、矩阵乘和 softmax 更好重叠。
```

在新硬件上，单纯减少 HBM IO 还不够，还需要减少 pipeline bubble，让 Tensor Core 尽量持续工作。

## 六、对推理的意义

推理中 Attention 有两个阶段：

- Prefill：一次处理完整 prompt，类似训练 forward，FlashAttention 非常关键。
- Decode：每次一个新 token，瓶颈更多在 KV Cache 读取和小 batch 调度，FlashDecoding/PagedAttention 更关键。

因此：

- 长 prompt 的 TTFT 受 Prefill Attention 性能影响明显。
- 长上下文 Decode 的 TPOT 更受 KV 读取、batching、PagedAttention 影响。
- FlashAttention-2/3 对 Prefill 优化更直接，对 Decode 也有相关思想迁移。

## 七、面试表达

FlashAttention-2/3 可以这样回答：

1. FlashAttention-1 主要解决标准 Attention 的 HBM IO 问题，避免保存完整 `n x n` 矩阵。
2. FlashAttention-2 在此基础上优化并行划分，减少 non-matmul 操作，提高 SM 和 Tensor Core 利用率。
3. Attention 中除了 `QK^T` 和 `PV`，还有 mask、softmax、rescale、reduction 等非矩阵乘操作，这些会成为瓶颈。
4. FlashAttention-3 面向 Hopper 等新架构，进一步利用异步流水、warpgroup 和低精度能力。
5. 推理中 Prefill 更直接受 FlashAttention 影响，Decode 则还需要关注 KV Cache 和调度。

## 八、总结

FlashAttention-1 解决“不要把巨大中间矩阵写回 HBM”；FlashAttention-2/3 继续解决“如何让硬件更充分工作”。面试中不要只背名字，要能说出 IO、non-matmul、并行切分和 Tensor Core 利用这几层逻辑。
