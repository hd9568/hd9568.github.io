---
title: '梯度累积详解：用 Micro-Batch 模拟大 Batch 的正确方式'
description: '从数学等价性出发讲解梯度累积、global batch 计算、PyTorch/AMP/DDP 实现、no_sync 通信优化、变长序列归一化、梯度裁剪和学习率调度。'
category: '分布式训练'
pubDate: '2026-07-20T12:31:00+08:00'
updatedDate: '2026-07-20T12:31:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [梯度累积解决什么问题](#一梯度累积解决什么问题)
2. [Micro-Batch、Mini-Batch 与 Global Batch](#二micro-batchmini-batch-与-global-batch)
3. [数学上为什么可以累积](#三数学上为什么可以累积)
4. [最小 PyTorch 实现](#四最小-pytorch-实现)
5. [为什么 loss 通常要除以累积步数](#五为什么-loss-通常要除以累积步数)
6. [什么时候不严格等价于大 Batch](#六什么时候不严格等价于大-batch)
7. [分布式训练中的梯度累积](#七分布式训练中的梯度累积)
8. [DDP no_sync：避免重复通信](#八ddp-no_sync避免重复通信)
9. [混合精度训练中的正确顺序](#九混合精度训练中的正确顺序)
10. [梯度裁剪、学习率和 Scheduler](#十梯度裁剪学习率和-scheduler)
11. [变长序列与 Token 级归一化](#十一变长序列与-token-级归一化)
12. [BatchNorm、Dropout 和随机性](#十二batchnormdropout-和随机性)
13. [最后不足一个累积窗口如何处理](#十三最后不足一个累积窗口如何处理)
14. [显存、吞吐与通信分析](#十四显存吞吐与通信分析)
15. [TensorFlow 实现](#十五tensorflow-实现)
16. [常见错误检查表](#十六常见错误检查表)
17. [总结](#十七总结)

## 一、梯度累积解决什么问题

训练大模型时，单卡显存通常无法直接容纳目标 batch。

假设希望使用：

```text
batch size = 128
```

但每张 GPU 一次最多只能处理：

```text
micro-batch size = 8
```

梯度累积的做法是：

```text
连续处理多个 micro-batch；
每次只做 forward 和 backward；
先不更新参数；
把梯度累加到 parameter.grad；
积累到指定次数后再 optimizer.step()。
```

例如：

```text
8 samples/micro-batch * 16 accumulation steps = 128 samples/update
```

梯度累积主要解决 activation memory 问题。每次 forward 只保留一个 micro-batch 的 activation，backward 后该 activation 可释放，而梯度继续保留。

它不会减少：

- 模型参数显存。
- 优化器状态显存。
- 单个样本本身的 activation。

如果单个样本都放不下，还需要模型并行、序列并行、activation checkpointing 等方法。

## 二、Micro-Batch、Mini-Batch 与 Global Batch

这些术语在不同框架中可能略有差异，本文统一定义。

### Micro-Batch Size

一次 forward/backward 在单个设备上处理的样本数：

```text
B_micro
```

### Gradient Accumulation Steps

一次 optimizer update 前，每个 rank 执行的 micro-batch 次数：

```text
K
```

### Data Parallel World Size

参与数据并行的 rank 数：

```text
N
```

### Global Batch Size

每次参数更新实际使用的总样本数：

```text
B_global = B_micro * K * N
```

例如：

```text
B_micro = 4
K = 8
N = 16
```

则：

```text
B_global = 4 * 8 * 16 = 512
```

需要注意：

- Tensor Parallel 不增加独立数据份数，通常不乘入 global batch。
- Pipeline Parallel 的 micro-batch 数和 gradient accumulation 常相关，但不能重复计算。
- Sequence Parallel 是特征维/序列维切分，也不直接增加 global batch。

## 三、数学上为什么可以累积

设一个大 batch 被拆成 `K` 个大小相同的 micro-batch：

```text
B = B_1 ∪ B_2 ∪ ... ∪ B_K
```

对每个 micro-batch，平均 loss：

```text
L_k(θ) = (1 / |B_k|) * sum_{x in B_k} l(x; θ)
```

整个大 batch 的平均 loss：

```text
L(θ) = (1/K) * sum_{k=1}^{K} L_k(θ)
```

梯度线性：

```text
∇L(θ) = (1/K) * sum_{k=1}^{K} ∇L_k(θ)
```

因此可以对每个 micro-batch 计算：

```text
(1/K) * ∇L_k
```

并累加，最后得到大 batch 的平均梯度。

### 关键条件：累积期间参数不能更新

所有 micro-batch 的梯度必须针对同一组参数 `θ_t`：

```text
g = sum_k ∇L_k(θ_t)
θ_{t+1} = optimizer(θ_t, g)
```

如果每个 micro-batch 都调用 `optimizer.step()`，后续梯度基于不同参数计算，就不再是梯度累积。

## 四、最小 PyTorch 实现

```python
import torch


accum_steps = 8
optimizer.zero_grad(set_to_none=True)

for micro_step, batch in enumerate(loader):
    output = model(batch["input"])
    loss = loss_fn(output, batch["target"])

    # 将 K 个 micro-batch 的平均梯度累加成一个大 batch 平均梯度。
    loss = loss / accum_steps
    loss.backward()

    if (micro_step + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

核心规则：

```text
每个 micro-batch: backward()
每个 accumulation window: step() + zero_grad()
```

`backward()` 默认把新梯度加到已有 `.grad`，而不是覆盖：

```text
param.grad <- param.grad + new_grad
```

这正是梯度累积的基础。

### 为什么用 set_to_none=True

```python
optimizer.zero_grad(set_to_none=True)
```

把 `.grad` 设为 `None`，而不是填零，通常能：

- 减少一次清零写入。
- 让后续 backward 直接分配/写入梯度。
- 区分“没有梯度”和“梯度为零”。

## 五、为什么 loss 通常要除以累积步数

如果每个 micro-batch loss 都是 mean reduction：

```python
loss = loss_fn(..., reduction="mean")
```

直接累积 `K` 次得到：

```text
sum_k ∇L_k
```

而大 batch mean loss 的梯度是：

```text
(1/K) * sum_k ∇L_k
```

所以通常要：

```python
(loss / K).backward()
```

### 不除会怎样

不除时梯度放大 `K` 倍，近似相当于把学习率放大 `K` 倍：

```text
θ <- θ - η * K * g_mean
```

这可能造成：

- 训练不稳定。
- 梯度裁剪频繁触发。
- 不同 accumulation 配置下结果不可比。

### 如果 loss 是 sum reduction

若 loss 返回样本 loss 之和：

```python
loss = loss_fn(..., reduction="sum")
```

要得到每样本平均梯度，应除以 accumulation window 的总样本数，而不是简单除以 `K`。

## 六、什么时候不严格等价于大 Batch

梯度累积常被描述为“等价于大 batch”，但严格等价需要条件。

### 1. BatchNorm

真正大 batch 的 BatchNorm 统计量基于全部样本：

```text
mean(B), var(B)
```

梯度累积中，每个 micro-batch 单独计算：

```text
mean(B_k), var(B_k)
```

因此 forward 输出已经不同，梯度也不同。

LayerNorm、RMSNorm 不跨 batch 聚合统计量，受此问题影响较小。

### 2. Dropout

不同 micro-batch 使用不同随机 mask。大 batch 也会为不同元素生成不同 mask，所以从统计分布看通常合理，但逐位结果不会完全相同。

### 3. 参数相关随机操作

如果每个 forward 修改状态、随机路由或更新 running statistics，micro-batch 切分可能改变行为。

### 4. Loss 归一化方式

若每个 micro-batch 样本数不同、token 数不同，却简单除以固定 `K`，结果不等价于全局按样本或 token 求平均。

### 5. Optimizer 每 micro-step 执行副作用

以下操作应只在 accumulation boundary 执行：

- optimizer step。
- weight decay update。
- EMA update。
- learning-rate scheduler step。
- global step 自增。

### 6. 梯度裁剪时机

每个 micro-batch 单独裁剪再相加：

```text
sum_k clip(g_k)
```

通常不等于先相加再裁剪：

```text
clip(sum_k g_k)
```

正确的大 batch 语义通常是后者。

## 七、分布式训练中的梯度累积

数据并行中，每个 rank 累积本地 micro-batch。

设：

```text
rank r 的第 k 个梯度 = g_{r,k}
```

目标全局平均梯度：

```text
g = 1/(N*K) * sum_r sum_k g_{r,k}
```

如果：

- 每个 micro-batch loss 除以 `K`。
- DDP AllReduce 对 rank 求平均。

则最终得到正确的全局平均梯度。

### World Size 由 DDP 处理

PyTorch DDP 默认把不同 rank 的梯度聚合并按 world size 归一化。通常只需要手动除以 accumulation steps，不需要再除以 world size。

重复除以 world size 会让梯度过小。

### Global Batch 改变会影响优化

如果增加：

- `B_micro`。
- `K`。
- `N`。

都会增大 global batch。

Global batch 增大后，梯度噪声减小，但单位 epoch 的 optimizer steps 减少。可能需要调整：

- learning rate。
- warmup steps。
- total scheduler steps。
- weight decay 解释。

## 八、DDP no_sync：避免重复通信

PyTorch DDP 默认在每次 backward 时同步梯度。

如果累积 `K=8`：

```text
默认行为：每个 micro-batch 都 AllReduce，共 8 次
```

但前 7 次只需要本地累积，不必同步。可以使用 `no_sync()`：

```python
from contextlib import nullcontext


accum_steps = 8
optimizer.zero_grad(set_to_none=True)

for micro_step, batch in enumerate(loader):
    should_step = (micro_step + 1) % accum_steps == 0
    sync_context = nullcontext() if should_step else model.no_sync()

    with sync_context:
        output = model(batch["input"])
        loss = loss_fn(output, batch["target"]) / accum_steps
        loss.backward()

    if should_step:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

通信从：

```text
K 次梯度同步 / optimizer step
```

降为：

```text
1 次梯度同步 / optimizer step
```

### no_sync 的边界

最后一个 micro-batch 必须在正常 DDP context 中 backward，触发同步。

如果整个窗口都在 `no_sync()` 中，rank 之间不会聚合，模型副本会逐渐分叉。

### 内存影响

延迟同步可能让更多梯度 bucket 暂时保留，峰值内存和普通每步同步略有差异，需要实测。

## 九、混合精度训练中的正确顺序

FP16 训练常使用 `GradScaler`：

```python
scaler = torch.amp.GradScaler("cuda")
optimizer.zero_grad(set_to_none=True)

for micro_step, batch in enumerate(loader):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(batch["input"])
        loss = loss_fn(output, batch["target"]) / accum_steps

    scaler.scale(loss).backward()

    if (micro_step + 1) % accum_steps == 0:
        # 如果需要梯度裁剪，先反缩放。
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

正确顺序：

```text
每个 micro-batch:
  scale(loss)
  backward()

accumulation boundary:
  unscale gradients
  clip gradients
  scaler.step()
  scaler.update()
  zero_grad()
```

### 为什么 scaler.update 不能每个 micro-step 调用

同一个 accumulation window 中，所有梯度属于同一次 optimizer update。Scale 应保持一致，只有真正尝试 optimizer step 后才更新。

### BF16

BF16 指数范围接近 FP32，通常不需要 loss scaling，但仍要注意：

- 梯度累积精度。
- Reduction dtype。
- optimizer state dtype。

## 十、梯度裁剪、学习率和 Scheduler

### 梯度裁剪

应在所有 micro-batch 梯度累积完成后执行：

```python
loss.backward()  # 重复 K 次

clip_grad_norm_(...)
optimizer.step()
```

AMP 下先：

```python
scaler.unscale_(optimizer)
```

再裁剪。

### Scheduler 按 optimizer step 更新

错误：

```python
每个 micro-batch scheduler.step()
```

正确：

```python
每次 optimizer.step() 后 scheduler.step()
```

设原来无累积时总 step 为：

```text
num_batches * epochs
```

使用 `K` 次累积后：

```text
optimizer_steps ≈ ceil(num_batches / K) * epochs
```

Warmup 和 decay 应按 optimizer steps 重新计算。

### Weight Decay

AdamW 的 weight decay 在 `optimizer.step()` 时应用。梯度累积后每 `K` 个 micro-batch 执行一次，语义对应一个大 batch update。

如果错误地每 micro-step 执行 optimizer，weight decay 也会多应用 `K` 次。

### 梯度范数日志

日志应在：

```text
全部累积完成
且 AMP unscale 后
```

记录，才能反映真实 optimizer update 使用的梯度范数。

## 十一、变长序列与 Token 级归一化

大语言模型 batch 常包含不同数量的有效 token。

假设两个 micro-batch：

```text
micro-batch 1: 100 valid tokens
micro-batch 2: 900 valid tokens
```

如果分别算 token mean loss，再对两个 loss 等权平均：

```text
(mean_100 + mean_900) / 2
```

第一个 micro-batch 的每个 token 权重是第二个的 9 倍。这不等价于 1000 个 token 的整体平均：

```text
(sum_100 + sum_900) / 1000
```

### 正确的 Token 加权

方法一：使用 sum loss，并除以整个 accumulation window 的有效 token 总数。

```python
total_tokens = sum(batch["loss_mask"].sum() for batch in window)

for batch in window:
    token_loss = loss_fn(..., reduction="none")
    loss_sum = (token_loss * batch["loss_mask"]).sum()
    (loss_sum / total_tokens).backward()
```

但这要求提前知道窗口总 token 数。

方法二：先累积未归一化梯度，最后按 token 数缩放梯度：

```python
window_tokens = 0

for batch in window:
    loss_sum = masked_token_loss(...).sum()
    loss_sum.backward()
    window_tokens += batch["loss_mask"].sum()

for p in model.parameters():
    if p.grad is not None:
        p.grad.div_(window_tokens)
```

分布式训练还要 AllReduce 全局 token count，确保所有 rank 使用同一分母。

### 样本权重同理

如果 loss 有 sample weight，应按：

```text
sum(weight_i * loss_i) / sum(weight_i)
```

而不是简单平均每个 micro-batch 的 mean loss。

## 十二、BatchNorm、Dropout 和随机性

### BatchNorm

Gradient accumulation 不会扩大 BatchNorm 的统计 batch。

解决方向：

- 使用 SyncBatchNorm，只能同步同一 micro-step 的 ranks，不能跨时间累积。
- 增大 micro-batch。
- 冻结 BatchNorm。
- 改用 GroupNorm、LayerNorm 或 RMSNorm。

### Dropout

不同 micro-batch 会使用不同 mask。只要随机状态管理正确，通常不影响训练目标的统计合理性。

Activation checkpointing 会重算 forward，框架必须保存/恢复 RNG state，确保 backward 重算使用相同 Dropout mask。

### 数据增强

每个 micro-batch 的随机数据增强也会改变逐位结果，但这和正常大 batch 中每个样本独立增强一致。

## 十三、最后不足一个累积窗口如何处理

假设：

```text
num_micro_batches = 10
accum_steps = 4
```

窗口是：

```text
[4, 4, 2]
```

最后只剩 2 个 micro-batch。

如果仍把每个 loss 除以 4，最后一次梯度会缩小一半。

### 方法一：动态窗口大小

```python
num_batches = len(loader)

for step, batch in enumerate(loader):
    window_start = (step // accum_steps) * accum_steps
    window_end = min(window_start + accum_steps, num_batches)
    current_accum = window_end - window_start

    loss = loss_fn(model(batch["x"]), batch["y"])
    (loss / current_accum).backward()

    if step + 1 == window_end:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

### 方法二：drop_last

丢弃最后不完整窗口。实现简单，但浪费数据。

### 方法三：跨 epoch 累积

把梯度保留到下一 epoch。此方法会让 epoch boundary 语义复杂，数据 sampler 和 scheduler 也要协调。

## 十四、显存、吞吐与通信分析

### 显存收益

Activation memory 通常随 micro-batch size 近似增长：

```text
M_activation ∝ B_micro
```

梯度累积允许保持较小 `B_micro`，因此降低每次 forward 的峰值 activation。

但梯度 buffer 在第一个 backward 后会一直保留到 optimizer step。

### 吞吐不一定提高

把一个大 batch 拆成多个 micro-batch 会增加：

- Kernel launch 次数。
- Python/runtime overhead。
- 小 GEMM 比例。
- 重复读取参数。

因此梯度累积主要是容量技术，不一定是吞吐优化。

### 太小的 Micro-Batch

GPU 矩阵乘需要足够大的 M 维才能提高利用率。`B_micro=1` 时，某些模型会出现：

- Tensor Core 利用率低。
- Kernel 太碎。
- 通信无法被计算隐藏。

应选择“显存能容纳的最大高效 micro-batch”，而不是盲目设为 1。

### 通信收益

若 DDP 使用 `no_sync()`，梯度通信频率降低 `K` 倍。但每次同步的总梯度大小不变。

这可能：

- 降低通信启动次数。
- 提高通信计算比。
- 让单次 step 更长。

## 十五、TensorFlow 实现

下面给出自定义 TensorFlow 训练循环：

```python
import tensorflow as tf


accum_steps = 8
optimizer = tf.keras.optimizers.Adam(1e-4)

accum_grads = [
    tf.Variable(tf.zeros_like(v), trainable=False)
    for v in model.trainable_variables
]


@tf.function
def micro_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred) / tf.cast(accum_steps, tf.float32)

    grads = tape.gradient(loss, model.trainable_variables)

    for buffer, grad in zip(accum_grads, grads):
        if grad is not None:
            buffer.assign_add(grad)

    return loss


@tf.function
def apply_step():
    optimizer.apply_gradients(zip(accum_grads, model.trainable_variables))

    for buffer in accum_grads:
        buffer.assign(tf.zeros_like(buffer))
```

训练循环：

```python
for step, (x, y) in enumerate(dataset):
    micro_step(x, y)

    if (step + 1) % accum_steps == 0:
        apply_step()
```

若梯度是 `IndexedSlices`，直接累加到 dense buffer 可能造成大规模 Embedding 梯度稠密化。稀疏参数需要专门的按 index 合并或框架原生累积实现。

## 十六、常见错误检查表

### Loss

- Mean loss 是否除以 accumulation steps？
- 不等长 micro-batch 是否按样本数加权？
- 变长序列是否按有效 token 数加权？

### Optimizer

- 是否只在 accumulation boundary 调用 `step()`？
- `zero_grad()` 是否误放在每个 micro-step？
- Weight decay 和 EMA 是否只更新一次？

### Distributed

- Global batch 是否正确计算？
- DDP 是否用 `no_sync()` 避免重复通信？
- 最后一个 micro-step 是否触发同步？
- 是否错误地又除了一次 world size？

### AMP 与裁剪

- `GradScaler.update()` 是否只在 optimizer step 后调用？
- 梯度裁剪前是否 `unscale_()`？
- 是否对累积后的完整梯度裁剪？

### Scheduler

- Scheduler 是否按 optimizer step 而非 micro-step 前进？
- Warmup/total steps 是否考虑 accumulation？

### 边界

- 最后不完整窗口是否使用正确分母？
- NaN/Inf 导致 step 跳过后，梯度 buffer 是否正确清理？
- Checkpoint 是否保存 accumulation progress 和 gradient buffer？

## 十七、总结

梯度累积的核心是：

```text
多个 micro-batch 共享同一组参数完成 backward；
把梯度累加；
最后只执行一次 optimizer update。
```

Global batch：

```text
B_global = B_micro * accumulation_steps * data_parallel_world_size
```

严格模拟大 batch 的关键条件：

1. 累积期间不更新参数。
2. Loss 使用正确归一化。
3. 梯度在完整累积后再裁剪。
4. Scheduler、EMA、weight decay 按 optimizer step 更新。
5. 分布式训练避免不必要的每 micro-step 同步。
6. 变长序列按有效 token 总数归一化。
7. 正确认识 BatchNorm 等状态算子带来的差异。

梯度累积主要用较小 activation 峰值换取更大的有效 batch。它可能降低通信频率，但也会增加小 kernel 和运行时开销，因此应同时观察显存、samples/s、tokens/s 和 optimizer-step latency，而不是只看能否运行。
