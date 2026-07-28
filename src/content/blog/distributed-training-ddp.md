---
title: 'DDP 分布式训练详解：进程模型、梯度同步与通信重叠'
description: '系统讲解 PyTorch DistributedDataParallel 的数学语义、单卡单进程架构、DistributedSampler、梯度 Bucket、AllReduce、通信重叠、完整训练模板与性能调优。'
category: '分布式训练'
pubDate: '2026-07-28T15:21:00+08:00'
updatedDate: '2026-07-28T15:21:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [DDP 解决什么问题](#一ddp-解决什么问题)
2. [数据并行的数学语义](#二数据并行的数学语义)
3. [单卡单进程模型](#三单卡单进程模型)
4. [一次 DDP 训练迭代](#四一次-ddp-训练迭代)
5. [梯度 Bucket 与通信重叠](#五梯度-bucket-与通信重叠)
6. [AllReduce 做了什么](#六allreduce-做了什么)
7. [可运行的 PyTorch DDP 模板](#七可运行的-pytorch-ddp-模板)
8. [DistributedSampler 为什么必不可少](#八distributedsampler-为什么必不可少)
9. [Global Batch、学习率与梯度累积](#九global-batch学习率与梯度累积)
10. [常用 DDP 参数](#十常用-ddp-参数)
11. [正确性与常见错误](#十一正确性与常见错误)
12. [性能模型和调优](#十二性能模型和调优)
13. [DDP、DataParallel、FSDP 与 ZeRO](#十三ddpdataparallelfsdp-与-zero)
14. [Checkpoint 与故障恢复](#十四checkpoint-与故障恢复)
15. [排查清单](#十五排查清单)
16. [总结](#十六总结)

## 一、DDP 解决什么问题

`DistributedDataParallel`，简称 DDP，是 PyTorch 中最常用的同步数据并行训练方式。

它的基本结构是：

```text
每个 GPU 对应一个独立进程
每个进程保存一份完整模型
每个进程读取不同的数据
反向传播时同步梯度
每个进程独立执行相同的参数更新
```

假设有 4 张 GPU：

```text
Rank 0: model replica 0 + batch 0
Rank 1: model replica 1 + batch 1
Rank 2: model replica 2 + batch 2
Rank 3: model replica 3 + batch 3
```

每张卡计算本地梯度，再通过 Collective Communication 得到相同的全局平均梯度。

DDP 主要解决：

- 用多张 GPU 并行处理更大的 Global Batch。
- 缩短固定训练数据量所需时间。
- 支持单机多卡和多机多卡。
- 通过 NCCL 高效同步 GPU 梯度。

DDP 不解决：

- 单卡放不下完整模型。
- 单个样本的 Activation 放不下。
- 优化器状态过大。

因为每个 Rank 仍保存完整参数、完整梯度和完整优化器状态。模型状态显存问题需要 ZeRO、FSDP、Tensor Parallel 等方案。

## 二、数据并行的数学语义

设有 `N` 个 DDP Rank，每个 Rank 的本地 Batch 大小为 `B`。

Rank `r` 的平均 Loss：

```text
L_r(θ) = (1/B) * sum_{x in batch_r} l(x; θ)
```

本地梯度：

```text
g_r = ∇L_r(θ)
```

DDP 默认同步后，每个 Rank 得到：

```text
g = (1/N) * sum_{r=0}^{N-1} g_r
```

代入本地梯度：

```text
g =
  (1/N) * sum_r [
    (1/B) * sum_{x in batch_r} ∇l(x; θ)
  ]
```

整理后：

```text
g =
  1/(N*B) * sum_{x in global_batch} ∇l(x; θ)
```

这正是大小为：

```text
B_global = N * B
```

的 Global Batch 平均梯度。

### 2.1 等价成立的条件

上述推导依赖每个 Rank：

- 使用相同的模型参数。
- 本地 Batch 样本数相同。
- Loss 的归一化方式相同。
- 都参与同一轮 Collective。
- 都执行相同次数的 `optimizer.step()`。

若不同 Rank 的有效样本数不同，简单平均 Rank 梯度不等于按样本数加权的全局平均。

例如：

```text
Rank 0: 8 个有效样本
Rank 1: 2 个有效样本
```

DDP 默认会让两个 Rank 各占 50% 权重，而不是按 `8:2` 加权。变长序列、过滤样本和最后一个不完整 Batch 都要注意这一点。

## 三、单卡单进程模型

推荐映射：

```text
1 process <-> 1 GPU
```

进程需要区分三个 Rank 概念。

### 3.1 Global Rank

整个 Process Group 中的进程编号：

```text
rank ∈ [0, world_size)
```

它用于：

- Collective Communication。
- 只在 Rank 0 保存 Checkpoint。
- 区分日志进程。

### 3.2 Local Rank

当前节点内的 GPU 编号：

```text
local_rank ∈ [0, num_gpus_per_node)
```

用于：

```python
torch.cuda.set_device(local_rank)
```

### 3.3 World Size

参与训练的总进程数：

```text
world_size = num_nodes * processes_per_node
```

例如 2 台机器，每台 8 张 GPU：

```text
world_size = 2 * 8 = 16
```

### 3.4 Process Group

所有 Rank 先建立通信组：

```python
dist.init_process_group(backend="nccl")
```

GPU 训练通常使用 NCCL；CPU Collective 常用 Gloo。

`torchrun` 会为每个进程设置：

```text
RANK
LOCAL_RANK
WORLD_SIZE
MASTER_ADDR
MASTER_PORT
```

训练脚本从环境变量读取即可。

## 四、一次 DDP 训练迭代

DDP 生命周期可以分成初始化和迭代两部分。

### 4.1 初始化

```text
启动多个进程
-> 初始化 Process Group
-> 每个进程绑定一张 GPU
-> 每个进程构造完整模型
-> DDP 同步初始模型状态
-> 构造分片 DataLoader
```

默认初始化同步会确保各 Rank 参数一致。训练期间参数本身通常不在每步广播；DDP 同步的是梯度。

若 `broadcast_buffers=True`，BatchNorm Running Mean 等 Buffer 会在 Forward 开始时从权威 Rank 广播。

### 4.2 Forward

每个 Rank 独立计算：

```python
output_r = model(input_r)
loss_r = loss_fn(output_r, target_r)
```

Forward 通常没有梯度通信。不同 Rank 的 Activation 也不会互相传输。

### 4.3 Backward

```python
loss_r.backward()
```

Autograd 计算每个参数的本地梯度。DDP 注册的 Hook 在梯度就绪时通知 `Reducer`。

梯度不会等整个 Backward 完成后再一次性同步，而是按 Bucket 分批启动 AllReduce。

### 4.4 Optimizer Step

所有 Bucket 同步完成后，每个 Rank 的参数梯度相同：

```python
optimizer.step()
```

每个 Rank 都必须调用 `optimizer.step()`。

因为：

```text
初始参数相同
同步梯度相同
优化器状态相同
更新规则相同
```

更新后的参数仍相同。

完整迭代：

```text
Rank r local batch
-> Forward
-> Backward
-> Gradient AllReduce
-> 所有 Rank 获得相同平均梯度
-> 所有 Rank 执行相同 Optimizer Step
```

## 五、梯度 Bucket 与通信重叠

如果每个 Parameter 都单独 AllReduce，会产生大量小通信：

```text
每个 Collective 都有启动延迟
小消息难以利用链路带宽
```

DDP 把多个参数梯度拼成 Bucket：

```text
Bucket 0: parameter gradients ...
Bucket 1: parameter gradients ...
Bucket 2: parameter gradients ...
```

当某个 Bucket 内所有梯度都 Ready 时，立即异步启动 AllReduce。

### 5.1 为什么能与 Backward 重叠

Backward 大致按网络反方向执行：

```text
输出层梯度先 Ready
中间层梯度随后 Ready
输入层梯度最后 Ready
```

时间线：

```text
Backward: [layer L] [layer L-1] [layer L-2] ... [layer 0]
Comm:              [bucket 0 AR]
                              [bucket 1 AR]
                                           [bucket 2 AR]
```

当 GPU 继续计算前面层的梯度时，通信 Stream 可以同步已经完成的后面层 Bucket。

理想情况下：

```text
大部分通信被 Backward 计算隐藏
```

最后 Ready 的 Bucket 无法再与后续 Backward 重叠，通常形成暴露在迭代尾部的 Communication Tail。

### 5.2 Bucket 大小的权衡

小 Bucket：

- 更早 Ready，重叠机会更大。
- Collective 次数更多，启动开销更高。

大 Bucket：

- Collective 次数少，带宽效率高。
- 必须等待更多梯度，通信启动更晚。

`bucket_cap_mb` 没有对所有模型都最优的固定值，应结合 Timeline 调整。

### 5.3 参数顺序为什么重要

DDP 通常按接近反向梯度 Ready 的顺序组织 Bucket。若模型 Forward/Backward 的实际 Ready 顺序与参数注册顺序差异很大，Bucket 会等待最后一个梯度，降低重叠。

所有 Rank 必须以相同顺序注册参数和触发 Collective，否则可能出现错误或死锁。

### 5.4 `gradient_as_bucket_view`

开启后，`parameter.grad` 可以直接成为 Bucket Buffer 的 View：

```text
Autograd 写 grad
-> 直接写入通信 Bucket
-> 不再额外复制到 Bucket
```

它可以减少一次梯度大小的额外 Buffer 和复制开销。启用后不应直接对 Grad 调用不兼容的 `detach_()`。

## 六、AllReduce 做了什么

从结果看，AllReduce：

```text
输入：每个 Rank 一份本地梯度
输出：每个 Rank 都得到所有梯度之和或平均值
```

概念上：

```text
AllReduce = ReduceScatter + AllGather
```

### 6.1 Ring AllReduce

设：

```text
N = Rank 数
M = 梯度 Bucket 字节数
```

Ring AllReduce 分为：

```text
ReduceScatter: N - 1 步
AllGather:     N - 1 步
```

每一步传输约 `M/N` 字节，因此每个 Rank 的总通信量：

```text
V_ring =
  2 * (N - 1) / N * M
```

当 `N` 较大：

```text
V_ring ≈ 2M
```

例如 4 个 Rank、1 GiB 梯度：

```text
V_ring =
  2 * 3/4 * 1 GiB
  = 1.5 GiB / Rank
```

### 6.2 简单时间模型

```text
T_comm ≈
    2 * (N - 1) * α
  + 2 * (N - 1) / N * M / BW
```

其中：

- `α`：一次通信步骤的延迟。
- `BW`：有效链路带宽。

小 Bucket 更受 `α` 影响，大 Bucket 更受带宽影响。

### 6.3 为什么不是 Rank 0 收集后广播

集中式 Reduce + Broadcast 会让 Rank 0 成为带宽瓶颈。Ring、Tree 等 Collective 会利用多条链路并行传输，更适合 GPU 集群。

## 七、可运行的 PyTorch DDP 模板

下面给出单机多卡和多机多卡都可使用的训练骨架。

```python
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup_distributed():
    # torchrun 会注入这些环境变量。
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    return rank, local_rank, world_size


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model_factory, dataset, epochs: int):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)

    # 必须在模型构造前设置 Seed。
    seed_everything(2026)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=8,       # 每个 Rank 的 Batch Size
        sampler=sampler,
        shuffle=False,      # 已由 DistributedSampler 负责
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = model_factory().to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=25,
        gradient_as_bucket_view=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )
    scaler = torch.amp.GradScaler("cuda")

    try:
        for epoch in range(epochs):
            # 各 Rank 使用相同 epoch，但得到不同且可复现的 Shuffle 分片。
            sampler.set_epoch(epoch)
            model.train()

            for inputs, targets in loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    outputs = model(inputs)
                    loss = torch.nn.functional.cross_entropy(
                        outputs,
                        targets,
                    )

                # DDP 的梯度 Hook 在 Backward 中自动触发 AllReduce。
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(checkpoint, "checkpoint.pt")
    finally:
        dist.destroy_process_group()
```

单机 8 卡启动：

```bash
torchrun --standalone --nproc-per-node=8 train.py
```

两台机器、每台 8 卡：

Node 0：

```bash
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank=0 \
  --master-addr=<node-0-address> \
  --master-port=29500 \
  train.py
```

Node 1 把 `--node-rank` 改为 `1`，其他 Rendezvous 参数保持一致。

### 7.1 BF16

支持 BF16 的 GPU 通常可以改为：

```python
with torch.amp.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    loss = ...
```

BF16 指数范围与 FP32 相同，通常不需要 Loss Scaling；是否使用 `GradScaler` 应按实际精度方案决定。

## 八、DistributedSampler 为什么必不可少

DDP 不会自动切分 DataLoader。

如果每个 Rank 都直接使用同一个普通 DataLoader：

```text
Rank 0 读取样本 0...B
Rank 1 也读取样本 0...B
...
```

这会重复计算相同数据，Global Batch 并没有真正扩大。

`DistributedSampler` 根据：

```text
rank
world_size
epoch
```

把全局样本索引切成不同分片。

### 8.1 `set_epoch(epoch)`

每个 Epoch 必须调用：

```python
sampler.set_epoch(epoch)
```

它让所有 Rank 基于同一个 Epoch Seed 生成全局 Shuffle，然后各自取不同分片。

若不调用，每个 Epoch 的 Shuffle 顺序可能完全相同。

### 8.2 `shuffle=False`

使用 `sampler` 时，DataLoader 本身应设置：

```python
shuffle=False
```

Shuffle 由 Sampler 统一负责。

### 8.3 数据量不能整除 World Size

默认情况下，Sampler 可能补齐索引，使各 Rank 样本数一致；这会重复少量样本。

`drop_last=True` 会丢弃尾部，保证各 Rank 数量一致但损失少量数据。

训练语义、评估准确率和可复现性需要明确选择。

### 8.4 IterableDataset

流式数据集不能只依赖普通 `DistributedSampler`。每个 Worker 必须按 Rank 和 DataLoader Worker ID 共同分片，否则仍可能重复读取。

## 九、Global Batch、学习率与梯度累积

设：

```text
B_local = 每个 Rank 每次 Forward 的样本数
K       = Gradient Accumulation Steps
N       = DDP World Size
```

Global Batch：

```text
B_global = B_local * K * N
```

例如：

```text
B_local = 4
K = 8
N = 16
```

则：

```text
B_global = 4 * 8 * 16 = 512
```

### 9.1 DDP 已经除以 World Size

默认 DDP 会把不同 Rank 的梯度归一化。Loss 若已经是本地 Batch Mean，通常不需要再手动除以 `world_size`。

梯度累积时，通常只需：

```python
loss = loss / accumulation_steps
```

### 9.2 `no_sync()`

若每个 Micro-Batch 都正常 Backward，DDP 每次都会同步梯度。

前 `K-1` 次可以：

```python
with model.no_sync():
    outputs = model(inputs)
    loss = loss_fn(outputs, targets) / K
    loss.backward()
```

最后一次在正常 DDP Context 中 Backward，触发所有累积梯度同步。

注意 Forward 也应放进 `no_sync()` Context。

梯度累积的边界条件、AMP、变长 Token 归一化见独立的梯度累积专题。

### 9.3 学习率

扩大 Global Batch 后，梯度噪声和每个 Epoch 的更新次数都会变化。

线性缩放：

```text
lr_new ≈ lr_base * B_new / B_base
```

只是常见启发式，不是普遍定律。通常还要重新调整 Warmup、Weight Decay 和 Scheduler。

## 十、常用 DDP 参数

### 10.1 `bucket_cap_mb`

控制梯度 Bucket 容量上限。

调优目标：

```text
足够大以获得高通信带宽
足够小以尽早启动通信
```

### 10.2 `find_unused_parameters`

动态图中某些参数可能未参与当前 Loss。开启后，DDP 从 Autograd Graph 查找未使用参数，并提前把它们标记为 Ready。

代价：

- 每步额外遍历 Autograd Graph。
- 降低性能。
- 控制流复杂时更难推理。

若模型始终使用全部参数，应保持 `False`。

如果确实有未使用参数却关闭该选项，Reducer 可能报错，因为某些 Bucket 永远等不到对应梯度。

### 10.3 `static_graph`

适用于：

- 每轮使用的参数集合不变。
- 训练图结构不变。
- 不存在随迭代变化的控制流。

它能减少未使用参数检查等开销，并支持部分复杂的重入反向场景。

### 10.4 `broadcast_buffers`

默认在 Forward 前同步 Module Buffer，例如 BatchNorm Running Statistics。

它不会把普通 Parameter 每步广播。

### 10.5 `gradient_as_bucket_view`

让 Grad 直接复用 Bucket Buffer，减少显存和复制，通常值得开启。

## 十一、正确性与常见错误

### 11.1 只在 Rank 0 调用 `optimizer.step()`

错误。每个 Rank 都有独立模型副本和优化器，所有 Rank 都要更新。

只有日志和 Checkpoint 通常限制在 Rank 0。

### 11.2 不使用 DistributedSampler

所有 Rank 重复读取相同数据，浪费计算并改变训练语义。

### 11.3 每个 Rank 进入不同 Collective

Collective 必须由通信组内所有 Rank 以一致顺序调用。

危险模式：

```python
if rank == 0:
    loss.backward()  # 只有 Rank 0 触发 DDP AllReduce
```

其他 Rank 未进入对应 Collective，会导致等待或超时。

### 11.4 Rank 间控制流不同

输入驱动的条件分支可能让不同 Rank 使用不同参数集合：

```text
Rank 0 使用 branch A
Rank 1 使用 branch B
```

需要确保 DDP 配置能处理未使用参数，且所有 Rank 的 Collective 顺序一致。

### 11.5 不同 Rank Batch 数不同

某些 Rank 提前耗尽 DataLoader，其他 Rank 仍在 Backward，会导致 Collective 无法配对。

优先让 Sampler 保证相同步数；无法保证时可使用 DDP Join 机制处理不均匀输入。

### 11.6 BatchNorm 语义

普通 BatchNorm 的均值和方差只基于每个 Rank 的本地 Batch。

若需要跨 Rank 统计，可转换为：

```python
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
```

SyncBatchNorm 会增加通信。LLM 常用 LayerNorm/RMSNorm，不受此问题影响。

### 11.7 Loss 日志不代表全局 Loss

Rank 0 的 `loss.item()` 只是 Rank 0 本地 Batch 的 Loss。

若需要全局平均日志：

```python
loss_for_log = loss.detach()
dist.all_reduce(loss_for_log, op=dist.ReduceOp.SUM)
loss_for_log /= dist.get_world_size()
```

该通信只用于日志，不参与梯度。

### 11.8 随机种子

模型初始化必须保持一致；数据增强和 Dropout 是否需要各 Rank 不同随机流，要按任务设计。

简单地让所有 Rank 使用完全相同随机状态，可能让数据增强高度相关。常见做法是：

```text
公共 Seed 保证初始化一致
数据采样由 Rank/Epoch 派生 Seed
```

### 11.9 原地修改参数或梯度

DDP 在构造时记录参数和 Reducer Hook。训练中添加、删除或替换 Parameter 会破坏同步关系。

应在包装 DDP 之前完成模型结构修改。

## 十二、性能模型和调优

一次迭代近似：

```text
T_step =
    T_input
  + T_forward
  + T_backward
  + T_exposed_comm
  + T_optimizer
```

通信总耗时不是关键，未被 Backward 隐藏的：

```text
T_exposed_comm
```

才直接增加 Step Time。

### 12.1 Scaling Efficiency

以单卡吞吐 `P_1`、`N` 卡吞吐 `P_N`：

```text
efficiency = P_N / (N * P_1)
```

例如单卡 100 samples/s，8 卡 680 samples/s：

```text
efficiency = 680 / 800 = 85%
```

### 12.2 计算通信比

参数量大但每卡 Batch 太小时：

- Backward 计算少。
- 梯度通信量基本不变。
- 通信更难隐藏。

增大 Local Batch 或 Gradient Accumulation 通常提高计算通信比，但会增加 Activation 显存和 Global Batch。

### 12.3 Bucket 调优

通过 Profiler/NSYS 观察：

- NCCL Kernel 是否在 Backward 早期启动。
- AllReduce 是否与 Compute Kernel 重叠。
- 最后是否存在长 Communication Tail。
- 是否有大量很小的 NCCL 调用。

根据结果调整 `bucket_cap_mb`，不要只根据总通信字节猜测。

### 12.4 DataLoader

GPU 等数据时，增加 GPU 数只会放大输入瓶颈。

可检查：

```text
num_workers
pin_memory
non_blocking H2D
persistent_workers
数据解码和增强耗时
存储与网络吞吐
CPU NUMA 绑定
```

### 12.5 Straggler

同步训练的每一步都由最慢 Rank 决定：

```text
T_step ≈ max(T_rank_0, ..., T_rank_N-1)
```

慢 Rank 可能来自：

- 数据长度不均。
- GPU 降频或硬件差异。
- 跨 NUMA 访问。
- 网络拥塞。
- DataLoader 抖动。
- 其他进程争用。

只看平均 GPU 利用率无法定位 Straggler，应比较 Rank 级 Timeline。

### 12.6 通信拓扑

单机优先使用 NVLink/NVSwitch；跨机依赖网络和 GPUDirect RDMA。

DDP Process Group 的 Rank 放置、网卡选择和 GPU-NIC 拓扑都会影响实际 NCCL 带宽。

## 十三、DDP、DataParallel、FSDP 与 ZeRO

| 方法 | 进程模型 | 参数 | 梯度 | 优化器状态 | 主要目标 |
| --- | --- | --- | --- | --- | --- |
| DataParallel | 单进程多线程 | 主卡集中/复制 | 主卡聚合 | 主卡 | 简单多卡 |
| DDP | 单卡单进程 | 每 Rank 完整 | 每 Rank 完整 | 每 Rank 完整 | 高效数据并行 |
| ZeRO-1 | 多进程 | 完整 | 完整 | 分片 | 减少优化器冗余 |
| ZeRO-2 | 多进程 | 完整 | 分片 | 分片 | 进一步减少梯度冗余 |
| FSDP/ZeRO-3 | 多进程 | 分片 | 分片 | 分片 | 超大模型状态分片 |

### 13.1 DDP 与 DataParallel

`nn.DataParallel` 使用单进程管理多卡，主设备负责 Scatter/Gather，Python 和主卡容易成为瓶颈。

DDP 使用独立进程和高效 Collective，通常性能和可扩展性都更好。

### 13.2 DDP 与 FSDP/ZeRO

DDP：

```text
实现简单
计算期间参数始终本地可用
通信主要是梯度 AllReduce
显存冗余高
```

FSDP/ZeRO-3：

```text
模型状态显著节省
需要 Parameter AllGather、ReduceScatter
执行和通信更复杂
```

模型能在单卡完整容纳时，DDP 往往是更直接的基线；单卡容纳不了模型状态时，才需要进一步分片。

## 十四、Checkpoint 与故障恢复

### 14.1 保存

所有 Rank 参数一致，通常只让 Rank 0 写文件：

```python
if rank == 0:
    torch.save(
        {
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": global_step,
        },
        "checkpoint.pt",
    )
```

使用 `model.module.state_dict()` 可以避免 Checkpoint Key 带 `module.` 前缀。

### 14.2 加载

常见方式有两种：

1. 每个 Rank 从共享存储读取同一个 Checkpoint。
2. Rank 0 加载后广播模型和必要状态。

大模型 Checkpoint 读取可能造成存储拥塞，应选择适合规模的分片 Checkpoint 方案。

### 14.3 还需要保存什么

为了精确恢复：

- Model。
- Optimizer。
- LR Scheduler。
- GradScaler。
- Global Step/Epoch。
- RNG State。
- Data Sampler 进度。

只恢复模型参数不能保证训练轨迹连续。

### 14.4 Barrier

并非每次 Rank 0 保存后都必须 Barrier。只有后续逻辑依赖文件已经写完时才需要：

```python
dist.barrier()
```

无意义的 Barrier 会破坏计算通信流水并增加同步等待。

## 十五、排查清单

### 正确性

- 每个进程是否只绑定一张 GPU？
- 是否在包装 DDP 前把模型移动到对应设备？
- 是否使用 `DistributedSampler` 或等价分片？
- 每个 Epoch 是否调用 `sampler.set_epoch()`？
- 每个 Rank 的训练 Step 数是否一致？
- 所有 Rank 是否都执行 Backward 和 Optimizer Step？
- Loss 是否按相同方式归一化？
- 变长/过滤数据是否造成 Rank 间有效样本数不同？
- 动态分支是否产生 Unused Parameters？
- Checkpoint 是否保存 `model.module`？

### 性能

- Local Batch 是否过小？
- AllReduce 是否与 Backward 重叠？
- Bucket 是否过小或过大？
- 是否频繁使用不必要的 `dist.barrier()`？
- DataLoader 是否让 GPU 等待？
- Rank 间是否存在 Straggler？
- NCCL 是否走预期网卡和 GPU 拓扑？
- 是否因日志、评估或保存导致所有 Rank 同步等待？
- 梯度累积是否用 `no_sync()` 减少重复通信？

### 调试环境变量

需要诊断时可临时开启：

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
```

详细日志开销较高，不建议长期用于正式训练。

## 十六、总结

DDP 的核心流程是：

```text
单卡单进程
-> 每个 Rank 读取不同数据
-> 本地 Forward/Backward
-> 梯度按 Bucket 异步 AllReduce
-> 通信与 Backward 尽量重叠
-> 每个 Rank 独立执行相同 Optimizer Step
```

需要掌握的关键点：

1. DDP 复制模型，只切分数据，不切分参数和优化器状态。
2. DDP 不会自动切分 DataLoader，必须使用正确的数据分片。
3. 默认同步结果是 Rank 间平均梯度；本地 Batch 大小不同时需要额外加权。
4. Bucket 让梯度通信与 Backward 重叠，暴露通信时间决定扩展效率。
5. 所有 Rank 必须以相同顺序参与 Collective，并保持相同步数。
6. 显存不足以保存完整模型时，应使用 FSDP、ZeRO 或模型并行，而不是继续增加 DDP Rank。
