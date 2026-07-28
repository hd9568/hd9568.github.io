---
title: 'Tensor Parallel 推理：切分 Attention、MLP 与通信路径'
description: '从 Column/Row Parallel Linear 推导 Transformer 的张量并行，讲解 QKV Head 切分、GQA 的 KV Head 复制、All-Reduce 成本、权重加载和 vLLM 实现。'
category: '推理优化'
pubDate: '2026-07-28T12:43:00+08:00'
updatedDate: '2026-07-28T12:43:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [为什么推理需要 Tensor Parallel](#一为什么推理需要-tensor-parallel)
2. [Column Parallel Linear](#二column-parallel-linear)
3. [Row Parallel Linear](#三row-parallel-linear)
4. [MLP 如何切分](#四mlp-如何切分)
5. [Attention 如何切分](#五attention-如何切分)
6. [一个两卡数值例子](#六一个两卡数值例子)
7. [最小 PyTorch 实现](#七最小-pytorch-实现)
8. [vLLM 中的实现](#八vllm-中的实现)
9. [通信成本](#九通信成本)
10. [TP 对 KV Cache 和延迟的影响](#十tp-对-kv-cache-和延迟的影响)
11. [选型和调优](#十一选型和调优)
12. [总结](#十二总结)

## 一、为什么推理需要 Tensor Parallel

Tensor Parallel（TP）把同一层的大张量切到多张 GPU，每张卡共同计算同一个请求。

主要目的：

- 单卡放不下模型权重。
- 单卡算力不足以满足目标延迟。
- KV Cache 需要按 Attention Head 分散。
- 与 Pipeline Parallel 组合部署更大模型。

它与 Data Parallel 不同：

```text
Data Parallel：
  每张卡有完整模型
  不同请求分给不同副本

Tensor Parallel：
  每张卡只有部分层内权重
  同一个请求在多张卡协作执行
```

TP 降低单卡权重，但每个 Transformer Layer 都会产生通信，因此通常优先放在 NVLink/NVSwitch 域内。

## 二、Column Parallel Linear

设 Linear：

```text
Y = XW
X: [M, K]
W: [K, N]
Y: [M, N]
```

沿输出维 `N` 切分权重：

```text
W = [W0, W1, ..., Wp-1]
Wi: [K, N/p]
```

每个 Rank 都持有完整 X：

```text
Yi = XWi
```

完整输出：

```text
Y = concat(Y0, Y1, ..., Yp-1)
```

若下一个算子也能消费分片输出，就不需要立即 All-Gather。Transformer 中常利用这一点减少通信。

```text
输入 X：每卡相同
权重 W：按列切
输出 Y：按最后一维切
```

## 三、Row Parallel Linear

沿输入维 `K` 切分：

```text
W =
  [W0
   W1
   ...
   Wp-1]

Wi: [K/p, N]
```

输入也按最后一维切：

```text
X = [X0, X1, ..., Xp-1]
```

每个 Rank 计算部分和：

```text
Zi = Xi Wi
```

完整结果：

```text
Y = sum_i Zi
```

因此需要 All-Reduce：

```text
Y = AllReduce(Zi, SUM)
```

```text
输入 X：按最后一维切
权重 W：按行切
输出 Y：All-Reduce 后每卡相同
```

## 四、MLP 如何切分

以 SwiGLU MLP 为例：

```text
G = X W_gate
U = X W_up
H = SiLU(G) * U
Y = H W_down
```

切分方式：

```text
W_gate：Column Parallel
W_up：  Column Parallel
W_down：Row Parallel
```

每个 Rank：

```text
Gi = X W_gate_i
Ui = X W_up_i
Hi = SiLU(Gi) * Ui
Zi = Hi W_down_i
Y  = AllReduce(Zi)
```

Gate 和 Up 的输出都按 Intermediate Dimension 切分，逐元素激活可以在本地完成。直到 Down Projection 才对部分和执行一次 All-Reduce。

这比每个 Linear 后都 All-Gather 更高效。

## 五、Attention 如何切分

标准 MHA：

```text
Q = X W_Q
K = X W_K
V = X W_V
O = Attention(Q, K, V) W_O
```

典型切分：

```text
W_Q/W_K/W_V：按 Head 维 Column Parallel
每个 Rank：   只计算本地 Head 的 Attention
W_O：         Row Parallel
```

每个 Rank 的局部计算：

```text
Qi, Ki, Vi = local_qkv_projection(X)
Ai = Attention(Qi, Ki, Vi)
Zi = Ai W_O_i
O = AllReduce(Zi)
```

因此一个标准 Transformer Layer 常有两次主要 TP All-Reduce：

```text
Attention Output Projection 后一次
MLP Down Projection 后一次
```

实现可以通过融合、Reduce-Scatter 或并行流水隐藏部分通信，但数学依赖仍存在。

### 5.1 GQA/MQA 的特殊情况

设：

```text
num_query_heads = 32
num_kv_heads = 8
TP = 4
```

每卡可分到：

```text
8 个 Query Head
2 个 KV Head
```

若：

```text
num_query_heads = 32
num_kv_heads = 2
TP = 8
```

KV Head 数小于 TP，无法让每卡获得互不重复的整数 Head。常见做法是复制 KV Head：

```text
每个 KV Head 被 4 个 TP Rank 复制
```

因此增加 TP 不一定继续降低每卡 KV Cache。

## 六、一个两卡数值例子

考虑：

```text
X: [2, 4]
W_up: [4, 8]
W_down: [8, 4]
TP = 2
```

### 6.1 Up Projection

按输出维切：

```text
GPU 0: W_up_0 [4, 4] -> H0 [2, 4]
GPU 1: W_up_1 [4, 4] -> H1 [2, 4]
```

`H0/H1` 不需要拼接，激活函数在本地执行。

### 6.2 Down Projection

`W_down` 按输入维切：

```text
GPU 0: W_down_0 [4, 4]
GPU 1: W_down_1 [4, 4]
```

局部输出：

```text
Z0 = H0 W_down_0  # [2, 4]
Z1 = H1 W_down_1  # [2, 4]
```

最后：

```text
Y = Z0 + Z1
```

通过 All-Reduce 后，GPU 0 和 GPU 1 都获得完整 `Y`，可进入下一层。

## 七、最小 PyTorch 实现

下面省略进程组初始化，只展示计算关系。

```python
import torch
import torch.distributed as dist
import torch.nn.functional as F


class ColumnParallelLinear:
    def __init__(self, full_weight: torch.Tensor, rank: int, world_size: int):
        # full_weight: [K, N]
        self.weight = full_weight.chunk(world_size, dim=1)[rank].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 返回 [M, N / TP]，不做 All-Gather。
        return x @ self.weight


class RowParallelLinear:
    def __init__(self, full_weight: torch.Tensor, rank: int, world_size: int):
        # full_weight: [K, N]
        self.weight = full_weight.chunk(world_size, dim=0)[rank].contiguous()

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        partial = x_shard @ self.weight
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial
```

一个两层 MLP：

```python
def tensor_parallel_mlp(x, up_proj, down_proj):
    # 每个 Rank 得到不同的 Intermediate 分片。
    hidden_shard = F.silu(up_proj.forward(x))

    # Row Parallel 汇总各 Rank 的部分和。
    output = down_proj.forward(hidden_shard)
    return output
```

真实实现还需处理 Bias、量化权重、权重加载、通信 Stream、CUDA Graph 和多种并行组。

## 八、vLLM 中的实现

vLLM 把 TP Linear 抽象为不同 Layer：

```text
ColumnParallelLinear
MergedColumnParallelLinear
QKVParallelLinear
RowParallelLinear
```

### 8.1 Column Parallel

初始化时：

```python
output_size_per_partition = output_size // tp_size
```

权重加载只取当前 Rank 的输出切片：

```python
start = tp_rank * shard_size
local_weight = loaded_weight.narrow(
    output_dim,
    start,
    shard_size,
)
```

Forward 默认保留局部输出；只有 `gather_output=True` 时才执行 All-Gather。

### 8.2 Row Parallel

初始化时：

```python
input_size_per_partition = input_size // tp_size
```

每个 Rank 计算局部部分和。`reduce_results=True` 时执行 Tensor Parallel All-Reduce。

### 8.3 QKVParallelLinear

QKV Layer 不只是把一个大矩阵平均切开，还要理解 Head 结构：

```python
num_heads = total_num_heads // tp_size

if tp_size >= total_num_kv_heads:
    num_kv_heads = 1
    num_kv_head_replicas = tp_size // total_num_kv_heads
else:
    num_kv_heads = total_num_kv_heads // tp_size
    num_kv_head_replicas = 1
```

这段逻辑明确处理了 GQA/MQA 中 KV Head 少于 TP Rank 的复制问题。

### 8.4 量化参数也必须正确切分

AWQ、GPTQ、FP8 等量化格式不仅有 `weight`，还包含：

```text
scales
zero points
block metadata
```

Column/Row Parallel 的加载器必须按对应量化维度切这些辅助参数，并满足 Group 或 Block 对齐。

## 九、通信成本

对 `p` 个 Rank、消息大小 `S`，Ring All-Reduce 的近似通信量为：

```text
每 Rank 字节数 ≈ 2 * (p - 1) / p * S
```

简单延迟模型：

```text
T_allreduce ≈
    2 * (p - 1) * alpha
  + 2 * (p - 1) / p * S / bandwidth
```

其中：

- `alpha` 是每一步通信启动延迟。
- `bandwidth` 是有效链路带宽。

Decode 的张量通常较小，启动延迟占比很高；Prefill 张量较大，带宽占比更高。

### 9.1 TP 扩展不是线性的

每卡计算约下降为 `1/p`，但通信增加：

```text
T_layer(p) ≈ T_compute(1) / p + T_collective(p)
```

当 `T_collective` 接近或超过节省的计算时，继续增加 TP 只会变慢。

### 9.2 跨节点 TP 的风险

NVLink/NVSwitch 的带宽和延迟通常明显优于跨节点网络。每层两次 Collective 会把网络抖动直接放大到每个 Decode token 的 TPOT。

因此常见策略是：

```text
节点内：Tensor Parallel
节点间：Pipeline Parallel 或 Data Parallel
```

具体选择仍取决于拓扑和模型大小。

## 十、TP 对 KV Cache 和延迟的影响

### 10.1 权重显存

理想情况下：

```text
weight_per_gpu ≈ total_weight / TP
```

但 Embedding、Norm、部分 Bias 和某些共享参数可能复制。

### 10.2 KV Cache

MHA 中 Head 可按 TP 切分，单卡 KV 大致下降；GQA/MQA 中受 KV Head 数限制，TP 超过 KV Head 数后需要复制，容量不再按 TP 线性下降。

### 10.3 单请求延迟

TP 同时带来：

- 每卡 GEMM 更小。
- Collective 通信。
- 更小 GEMM 的 Tensor Core 利用率可能下降。
- 多 Rank 同步等待最慢卡。

模型必须多卡才能容纳时，TP 是容量需求；模型单卡可容纳时，是否降低延迟需要实测。

### 10.4 吞吐

用 8 卡 TP=8 运行一个副本，与 8 个 TP=1 副本的吞吐目标不同：

```text
TP=8：单请求协作，适合大模型或低延迟
DP=8：8 个完整副本，通常更适合总吞吐
```

当模型能单卡容纳且请求足够多，Data Parallel 往往有更高集群吞吐。

## 十一、选型和调优

### 11.1 TP 大小的基本约束

- Hidden Size、Intermediate Size 能被 TP 整除。
- Query Head 能合理切分。
- KV Head 数与复制策略明确。
- 量化 Group/Block 与分片边界对齐。
- TP Group 尽量位于高速互联域。

### 11.2 观测指标

```text
每层 GEMM latency
All-Reduce latency 和占比
NCCL bandwidth
GPU SM/Tensor Core 利用率
TPOT P50/P99
每卡权重与 KV Cache 占用
Rank 间执行时间差
```

### 11.3 常见优化

- 融合 Residual/Norm 与 Collective。
- 使用自定义低延迟 All-Reduce。
- 让计算和通信在不同 Stream 重叠。
- 避免不必要的 All-Gather。
- 对 Decode 小消息选择低延迟算法。
- 按 NVLink/NVSwitch 拓扑构造 TP Group。
- TP 只开到模型可容纳或延迟最优的最小值。

## 十二、总结

Transformer 的 Tensor Parallel 依赖一对互补切分：

```text
Column Parallel：
  按输出维切
  产生局部输出
  尽量延迟 All-Gather

Row Parallel：
  按输入维切
  产生部分和
  通过 All-Reduce 恢复完整输出
```

Attention 使用 QKV Head 切分和 Row Parallel Output Projection；MLP 使用 Column Parallel Gate/Up 和 Row Parallel Down。TP 能降低单卡权重和部分 KV Cache，却把每层 Collective 放入 Decode 关键路径。合理的 TP 大小由容量、Head 数、量化对齐和互联拓扑共同决定。
