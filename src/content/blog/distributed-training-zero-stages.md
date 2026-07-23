---
title: 'ZeRO 系列详解：ZeRO-1、ZeRO-2 与 ZeRO-3 如何切分训练状态'
description: '从混合精度 Adam 的显存组成出发，逐步推导 ZeRO-1/2/3 的优化器状态、梯度和参数分片，讲清通信流程、内存公式、Offload、FSDP 对应关系与选型。'
category: '分布式训练'
pubDate: '2026-07-20T12:30:00+08:00'
updatedDate: '2026-07-20T12:30:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [ZeRO 要解决什么问题](#一zero-要解决什么问题)
2. [先算清模型状态占多少显存](#二先算清模型状态占多少显存)
3. [普通数据并行为什么浪费显存](#三普通数据并行为什么浪费显存)
4. [ZeRO-1：切分优化器状态](#四zero-1切分优化器状态)
5. [ZeRO-2：再切分梯度](#五zero-2再切分梯度)
6. [ZeRO-3：再切分参数](#六zero-3再切分参数)
7. [三阶段内存公式与数字示例](#七三阶段内存公式与数字示例)
8. [三阶段通信过程](#八三阶段通信过程)
9. [ZeRO 与计算图执行](#九zero-与计算图执行)
10. [ZeRO-Offload 与 ZeRO-Infinity](#十zero-offload-与-zero-infinity)
11. [ZeRO 与 PyTorch FSDP](#十一zero-与-pytorch-fsdp)
12. [配置和代码示例](#十二配置和代码示例)
13. [常见问题](#十三常见问题)
14. [如何选择阶段](#十四如何选择阶段)
15. [总结](#十五总结)

## 一、ZeRO 要解决什么问题

ZeRO 全称 **Zero Redundancy Optimizer**。它解决的是数据并行中的内存冗余问题。

普通数据并行中，每张 GPU 都保存完整的：

- 模型参数。
- 梯度。
- 优化器状态。

假设有 8 张 GPU，模型状态会复制 8 份。计算确实被分摊了，但模型状态没有被分摊。

ZeRO 的核心思想很直接：

```text
既然数据并行 rank 共同训练同一个模型，
就没有必要让每个 rank 永久保存所有训练状态。
```

ZeRO 分三个阶段逐步消除冗余：

```text
ZeRO-1: partition optimizer states
ZeRO-2: partition optimizer states + gradients
ZeRO-3: partition optimizer states + gradients + parameters
```

这里的关键不是简单“把模型切开”，而是：

```text
在需要某段状态时通过 collective communication 临时聚合；
使用结束后只保留本 rank 负责的 shard。
```

## 二、先算清模型状态占多少显存

为了比较三个阶段，先固定一组假设。

设模型参数量为：

```text
Ψ parameters
```

采用 FP16/BF16 混合精度训练和 Adam：

| 状态 | 每参数字节数 |
| --- | ---: |
| 低精度模型参数 | 2 bytes |
| 低精度梯度 | 2 bytes |
| FP32 master parameter | 4 bytes |
| Adam 一阶矩 `m` | 4 bytes |
| Adam 二阶矩 `v` | 4 bytes |

合计：

```text
2 + 2 + 4 + 4 + 4 = 16 bytes / parameter
```

因此仅模型状态约为：

```text
16Ψ bytes
```

其中：

```text
parameters:       2Ψ
gradients:        2Ψ
optimizer states: 12Ψ
```

本文把 FP32 master parameter 归入 optimizer states。

不同框架可能使用 FP32 gradient、不同 optimizer 或不保存 master weight，实际字节数会变化，但 ZeRO 的分片逻辑不变。

### 还有哪些显存没有计入

上述公式没有计算：

- Activation。
- Attention 中间结果。
- 临时 workspace。
- CUDA context。
- 通信 bucket。
- 显存碎片。
- 编译器和算子缓存。

ZeRO 主要减少模型状态，不会自动消除 activation memory。Activation 仍需 gradient checkpointing、sequence parallel 等技术优化。

## 三、普通数据并行为什么浪费显存

设有 `N` 个 data-parallel ranks。

普通 DDP 中每个 rank 保存：

```text
完整 parameter:       2Ψ
完整 gradient:        2Ψ
完整 optimizer state: 12Ψ
```

每个 rank：

```text
M_DDP = 16Ψ
```

整个集群：

```text
M_total = 16NΨ
```

但逻辑上训练只需要一份完整模型状态。其他 `N-1` 份主要是为数据并行执行方便而复制。

### DDP 的一步

```text
每个 rank:
  持有完整参数
  -> 用本地 batch forward/backward
  -> 得到完整梯度
  -> AllReduce 所有梯度
  -> 每个 rank 得到同样的平均梯度
  -> 每个 rank 独立执行相同 optimizer update
```

因为初始参数、聚合梯度和 optimizer 状态相同，每个 rank 更新后仍得到相同参数。

ZeRO 注意到：这些相同状态没有必要在每个 rank 都保存和更新。

## 四、ZeRO-1：切分优化器状态

ZeRO-1 只切分 optimizer states：

```text
参数：每个 rank 完整保存
梯度：每个 rank 完整保存
优化器状态：按 rank 分片
```

假设 `N=4`，参数按逻辑分成 4 份：

```text
Rank 0 owns optimizer state for shard 0
Rank 1 owns optimizer state for shard 1
Rank 2 owns optimizer state for shard 2
Rank 3 owns optimizer state for shard 3
```

### 反向传播

每个 rank 仍计算完整梯度。

梯度聚合后，每个 rank 只需要负责自己 shard 的 optimizer update：

```text
Rank 0 updates parameter shard 0
Rank 1 updates parameter shard 1
Rank 2 updates parameter shard 2
Rank 3 updates parameter shard 3
```

### 更新后如何恢复完整参数

每个 rank 只更新了自己负责的参数 shard。随后通过 AllGather：

```text
[updated shard 0, shard 1, shard 2, shard 3]
```

让所有 rank 重新拿到完整参数，供下一轮 forward 使用。

### 每 rank 内存

参数和梯度仍完整复制，optimizer states 除以 `N`：

```text
M_Z1 = 2Ψ + 2Ψ + 12Ψ/N
     = 4Ψ + 12Ψ/N
```

当 `N=8`：

```text
M_Z1 = 4Ψ + 1.5Ψ = 5.5Ψ bytes
```

相对 DDP 的 `16Ψ` 已显著下降。

### 适用场景

- 模型参数能放入单卡。
- Activation 和参数不是主要瓶颈。
- Adam optimizer states 占用很大。
- 希望较低改造成本获得明显节省。

## 五、ZeRO-2：再切分梯度

ZeRO-2 在 ZeRO-1 基础上继续切分 gradients：

```text
参数：每个 rank 完整保存
梯度：按 rank 分片
优化器状态：按 rank 分片
```

### ReduceScatter 代替完整 AllReduce

DDP 的 AllReduce 可以理解为：

```text
ReduceScatter + AllGather
```

ZeRO-2 不需要让每个 rank 拿到完整聚合梯度。每个 rank 只负责一个参数 shard 的 optimizer update，所以只需 ReduceScatter：

```text
每个 rank 输入完整本地梯度
-> 按 shard 求和并分发
-> Rank i 只得到 gradient shard i
```

例如四个 rank：

```text
Rank 0 receives reduced grad shard 0
Rank 1 receives reduced grad shard 1
Rank 2 receives reduced grad shard 2
Rank 3 receives reduced grad shard 3
```

然后每个 rank 更新对应 parameter shard，再 AllGather updated parameters。

### 每 rank 内存

```text
完整参数:      2Ψ
梯度 shard:    2Ψ/N
优化器 shard: 12Ψ/N
```

所以：

```text
M_Z2 = 2Ψ + 14Ψ/N
```

当 `N=8`：

```text
M_Z2 = 2Ψ + 1.75Ψ = 3.75Ψ bytes
```

### 为什么参数仍然完整

Forward 和 backward 中每个 rank 都执行完整模型，需要完整参数。因此 ZeRO-2 仍复制低精度参数。

### 适用场景

- 参数本身能放入单卡。
- 梯度和 optimizer states 造成显存压力。
- 希望通信模式接近普通数据并行。
- 不希望引入 ZeRO-3 频繁参数 AllGather。

## 六、ZeRO-3：再切分参数

ZeRO-3 切分所有三类模型状态：

```text
参数：按 rank 分片
梯度：按 rank 分片
优化器状态：按 rank 分片
```

每个 rank 常驻只保存约 `1/N` 模型状态。

### 但每个 rank 如何执行完整模型

计算某一层时，rank 临时 AllGather 该层参数：

```text
参数 shards
-> AllGather
-> 得到当前层完整参数
-> 执行当前层计算
-> 释放非本 rank shard
```

这通常按 module 或 parameter bucket 进行，而不是一次性聚合全模型。

Forward：

```text
for layer in model:
    AllGather(layer parameters)
    run layer forward
    release remote parameter shards
```

Backward：

```text
for layer in reversed(model):
    AllGather(layer parameters)
    run layer backward
    ReduceScatter(layer gradients)
    release remote parameters and non-owned gradients
```

### 每 rank 理论内存

```text
parameter shard:  2Ψ/N
gradient shard:   2Ψ/N
optimizer shard: 12Ψ/N
```

所以：

```text
M_Z3 = 16Ψ/N
```

当 `N=8`：

```text
M_Z3 = 2Ψ bytes
```

这是理论稳定状态。实际执行中还需要：

- 当前 module 的 gathered parameter。
- prefetch module 参数。
- communication buffer。
- temporary full parameters。

所以峰值显存高于 `16Ψ/N`。

### ZeRO-3 的代价

- Forward 也有参数通信。
- Backward 需要再次获取参数。
- 小 module 频繁 collective 会产生高延迟。
- 参数生命周期和计算顺序更复杂。
- 动态控制流可能影响预取。
- 保存和加载 checkpoint 更复杂。

## 七、三阶段内存公式与数字示例

在本文假设下，每 rank 模型状态内存：

| 方案 | 每 rank 内存 |
| --- | ---: |
| DDP | `16Ψ` |
| ZeRO-1 | `4Ψ + 12Ψ/N` |
| ZeRO-2 | `2Ψ + 14Ψ/N` |
| ZeRO-3 | `16Ψ/N` |

### 7B 模型、8 张 GPU

设：

```text
Ψ = 7 billion
N = 8
```

每个 `Ψ byte` 对应：

```text
7 GB
```

只计算模型状态：

#### DDP

```text
16 * 7 GB = 112 GB / GPU
```

#### ZeRO-1

```text
(4 + 12/8) * 7
= 5.5 * 7
= 38.5 GB / GPU
```

#### ZeRO-2

```text
(2 + 14/8) * 7
= 3.75 * 7
= 26.25 GB / GPU
```

#### ZeRO-3

```text
16/8 * 7
= 14 GB / GPU
```

这个例子没有包含 activation、临时 buffer 和 allocator overhead，因此不能直接据此断言某显存容量一定能训练。

### Rank 数增加时

当 `N -> ∞`：

```text
ZeRO-1 -> 4Ψ
ZeRO-2 -> 2Ψ
ZeRO-3 -> 0（理论分片项）
```

ZeRO-1 始终保留完整参数和梯度；ZeRO-2 始终保留完整参数；只有 ZeRO-3 能随 rank 数继续切分参数本身。

## 八、三阶段通信过程

### 通信原语

先理解三个 collective：

#### AllReduce

每个 rank 输入一份 Tensor，做归约后每个 rank 都得到完整结果。

```text
output_i = sum_j(input_j)
```

#### ReduceScatter

先归约，再把结果分片给不同 rank。

```text
Rank i gets reduced shard i
```

#### AllGather

每个 rank 提供一个 shard，所有 rank 得到完整 Tensor。

```text
[shard_0, shard_1, ..., shard_N-1]
```

### DDP

```text
Backward:
  AllReduce gradients

Optimizer:
  every rank updates full parameters
```

### ZeRO-1

概念上：

```text
聚合梯度
-> 每个 rank 更新自己负责的 parameter shard
-> AllGather updated parameter shards
```

实现可以对梯度采用 ReduceScatter，只保留 owner 需要的聚合结果。

### ZeRO-2

```text
Backward:
  ReduceScatter gradients

Optimizer:
  each rank updates owned shard

After update:
  AllGather updated parameters
```

### ZeRO-3

```text
Forward:
  per-module parameter AllGather

Backward:
  per-module parameter AllGather
  gradient ReduceScatter

Optimizer:
  local shard update
```

ZeRO-3 比 ZeRO-2 增加参数通信，尤其对：

- 小层很多的模型。
- 慢网络。
- 小 batch。
- 计算通信比低的模型。

影响更明显。

### 通信量和通信次数要区分

两个方案总字节数接近，不代表性能相同。

性能还取决于：

- collective 次数。
- 每次消息大小。
- latency。
- bandwidth。
- 是否能和计算 overlap。
- 节点内 NVLink 与节点间网络差异。

大量小 AllGather 通常比少量大 AllGather 更容易受 latency 限制。

## 九、ZeRO 与计算图执行

### Bucket

框架通常把多个小 parameter/gradient 合并成 bucket：

```text
减少 collective 次数
提高网络带宽利用率
```

bucket 太小：

- 通信次数多。
- latency 高。

bucket 太大：

- 峰值显存高。
- overlap 启动晚。
- 参数释放延迟。

### Prefetch

ZeRO-3 可以在计算 layer `i` 时预取 layer `i+1` 参数：

```text
compute layer i
|| AllGather layer i+1
```

理想情况下，通信被当前层计算隐藏。

Prefetch 太激进会提高峰值显存；太保守则 GPU 等待参数。

### Parameter Persistence

LayerNorm、bias 等小参数若每次都 AllGather，通信启动成本可能超过数据传输成本。

可把小参数持久保留在每个 rank：

```text
用少量冗余换取更少 collective
```

### Activation Checkpointing

ZeRO 和 activation checkpointing 优化不同对象：

```text
ZeRO:                 model states
Activation checkpoint: activations
```

大模型训练常组合使用。

## 十、ZeRO-Offload 与 ZeRO-Infinity

### ZeRO-Offload

当 GPU 显存仍不够，可以把部分状态放到 CPU：

- optimizer states offload 到 CPU。
- gradients offload 到 CPU。
- parameters offload 到 CPU。

优点：

- 利用更大的主机内存。
- 降低 GPU 显存占用。

代价：

- PCIe/NVLink-C2C 数据搬运。
- CPU optimizer update 性能。
- pinned memory 占用。
- NUMA 位置影响。

### ZeRO-Infinity

进一步可利用 NVMe 承载部分状态，形成：

```text
GPU HBM
-> CPU DRAM
-> NVMe
```

容量逐级增大，带宽逐级降低。

高性能依赖：

- 分块。
- 异步预取。
- pipeline。
- 计算与 I/O overlap。
- 合理的 working set。

如果预取不及时，GPU 会等待 PCIe 或 NVMe，吞吐显著下降。

## 十一、ZeRO 与 PyTorch FSDP

PyTorch FSDP，Fully Sharded Data Parallel，与 ZeRO-3 的核心思想接近：

```text
参数、梯度、优化器状态全部分片；
计算 module 前 AllGather 参数；
反向后 ReduceScatter 梯度。
```

常见 FSDP sharding strategy 可概念化为：

| FSDP 策略 | 近似对应 |
| --- | --- |
| `NO_SHARD` | DDP |
| `SHARD_GRAD_OP` | ZeRO-2 风格 |
| `FULL_SHARD` | ZeRO-3 风格 |
| `HYBRID_SHARD` | 节点内 full shard，节点间复制 |

ZeRO 是算法和 DeepSpeed 实现体系；FSDP 是 PyTorch 原生 fully sharded 方案。两者 API、checkpoint、参数扁平化和调度细节不同，不能简单认为完全相同。

### Hybrid Sharding

大集群中，节点内互联快、节点间网络慢。可以：

```text
节点内 8 GPU 做 full shard
节点间不同节点复制同一模型组
```

这样减少跨节点参数 AllGather，但保留节点间数据并行副本。

## 十二、配置和代码示例

### DeepSpeed ZeRO 配置

ZeRO-2 示例：

```json
{
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "zero_optimization": {
    "stage": 2,
    "reduce_scatter": true,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 200000000,
    "allgather_bucket_size": 200000000
  },
  "bf16": {
    "enabled": true
  }
}
```

ZeRO-3 示例：

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 200000000,
    "stage3_prefetch_bucket_size": 50000000,
    "stage3_param_persistence_threshold": 100000
  }
}
```

Offload 示例：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

### PyTorch FSDP 示例

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)


mixed_precision = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mixed_precision,
    use_orig_params=True,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
```

真实大模型通常按 Transformer block 做 auto wrap，避免把整个模型作为单一巨大 FSDP unit，也避免每个小算子都成为独立 unit。

## 十三、常见问题

### 1. ZeRO-3 一定最快吗

不一定。

ZeRO-3 节省最多显存，但参数 AllGather 更多。模型能用 ZeRO-2 放下时，ZeRO-2 可能吞吐更高。

### 2. ZeRO 会降低 activation memory 吗

不会直接降低。Activation 需单独使用：

- activation checkpointing。
- selective recomputation。
- sequence parallel。
- FlashAttention。

### 3. Rank 越多，单卡显存一定越接近理论值吗

模型状态 shard 会变小，但固定开销、activation、通信 buffer 和临时参数不会同比缩小。

### 4. ZeRO-3 为什么可能 OOM

常见原因：

- Prefetch bucket 太大。
- 多个 module 参数同时 materialize。
- Activation 本身过大。
- 通信和计算 stream 生命周期重叠。
- allocator fragmentation。
- checkpoint 保存时聚合完整 state。

### 5. 保存 checkpoint 为什么复杂

ZeRO-3 下每个 rank 只持有 shard。保存方式有两类：

```text
sharded checkpoint:
  每个 rank 保存自己的 shard

full checkpoint:
  聚合完整参数后保存
```

Full checkpoint 更易使用，但聚合时需要额外内存和通信。

### 6. 梯度累积和 ZeRO 是否冲突

不冲突。梯度累积减少 optimizer step 频率，但要注意：

- 是否每个 micro-batch 都触发梯度通信。
- 框架能否延迟 ReduceScatter。
- accumulation boundary 才执行 optimizer step。
- loss 是否正确除以 accumulation steps。

## 十四、如何选择阶段

### 选择 ZeRO-1

适合：

- 参数和梯度能放下。
- Adam states 是主要内存瓶颈。
- 希望最小化复杂度。

### 选择 ZeRO-2

适合：

- 完整参数能放下。
- 还需要切分梯度。
- 希望保持较好的吞吐。
- 网络不适合频繁 parameter AllGather。

### 选择 ZeRO-3

适合：

- 完整参数本身无法放入单卡。
- 需要最大化模型状态分片。
- 网络带宽较好。
- 能接受更复杂的 module wrapping、prefetch 和 checkpoint。

### 选择 Offload

适合：

- GPU 显存仍不足。
- CPU DRAM 或 NVMe 容量充足。
- 模型计算足以覆盖数据搬运。

如果目标只是提高吞吐而不是突破容量，Offload 往往不是第一选择。

## 十五、总结

ZeRO 的三个阶段是逐步消除数据并行冗余：

```text
ZeRO-1: optimizer state sharding
ZeRO-2: optimizer state + gradient sharding
ZeRO-3: optimizer state + gradient + parameter sharding
```

在本文混合精度 Adam 假设下：

```text
DDP:    16Ψ
ZeRO-1: 4Ψ + 12Ψ/N
ZeRO-2: 2Ψ + 14Ψ/N
ZeRO-3: 16Ψ/N
```

ZeRO-1/2 主要在 optimizer step 前后做 gradient partition 和 updated parameter gathering；ZeRO-3 进一步在 forward/backward 期间按 module 动态 AllGather 参数。

选择 ZeRO 阶段时不能只看显存公式，还要同时考虑：

- Activation 占用。
- 网络拓扑。
- collective 次数。
- bucket 大小。
- 是否能通信计算重叠。
- checkpoint 形式。
- CPU/NVMe Offload 带宽。

显存越省通常意味着状态生命周期和通信调度越复杂。最合适的阶段，是能放下目标模型且端到端吞吐最高的阶段，而不是编号最大的阶段。
