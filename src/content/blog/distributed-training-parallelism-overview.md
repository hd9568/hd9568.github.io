---
title: '大模型分布式训练并行综述：DP、TP、PP、EP、CP 与 SP'
description: '用统一的张量切分和通信视角讲清 DP、TP、PP、EP、CP、SP 的原理、通信原语、主要问题、经典优化以及多维并行组合方式。'
category: '分布式训练'
pubDate: '2026-08-27T10:10:00+08:00'
updatedDate: '2026-08-27T10:10:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [先建立统一坐标系](#一先建立统一坐标系)
2. [DP 与 TP：切 Batch 还是切层内张量](#二dp-与-tp切-batch-还是切层内张量)
3. [PP 与 EP：切深度还是切专家](#三pp-与-ep切深度还是切专家)
4. [CP 与 SP：都切序列但不是一回事](#四cp-与-sp都切序列但不是一回事)
5. [六种并行的通信原语](#五六种并行的通信原语)
6. [多维并行怎样组合](#六多维并行怎样组合)
7. [问题、经典解法与选型](#七问题经典解法与选型)

## 一、先建立统一坐标系

大模型训练中的并行，本质都是选择一个维度切分计算和状态。

设 Transformer 的主要维度为：

```text
B: batch size
S: sequence length
H: hidden size
L: number of layers
E: number of experts
```

六种并行分别切：

| 并行 | 切分对象 | 每卡主要保留什么 | 典型通信 |
| --- | --- | --- | --- |
| DP | `B`，不同样本 | 完整模型，不同数据 | Gradient AllReduce |
| TP | `H`，层内权重/Activation | 每层部分参数与计算 | AllReduce / ReduceScatter / AllGather |
| PP | `L`，连续层 | 一段模型层 | P2P Send/Recv |
| EP | `E`，MoE Experts | 部分专家 | Token AllToAll |
| CP | `S`，完整 Context | 所有层的本地 Token | KV Ring / AllGather / AllToAll |
| SP | `S`，TP 间部分 Activation | 非 Attention 区域的 Token Shard | TP 的 ReduceScatter + AllGather |

它们解决的瓶颈也不同：

```text
模型能放下，希望扩大训练吞吐       -> DP
单层权重或 Hidden 计算放不下       -> TP
模型层数太多，整模型放不下         -> PP
MoE 专家总参数放不下               -> EP
单个样本的 Context Activation 放不下 -> CP
使用 TP 后仍想进一步省 Activation  -> SP
```

不要把“显存下降”视为同一种收益：

- DP 本身不降低模型状态显存，标准 DDP 还会复制模型。
- TP、PP、EP 直接切参数和计算。
- CP、SP 主要切 Activation。
- ZeRO/FSDP 是 DP 维度上的状态分片，不是新的模型计算维度。

## 二、DP 与 TP：切 Batch 还是切层内张量

### DP：不同数据，相同模型

`d` 路数据并行中，每个 Rank 保存完整参数 `theta`，读取不同 Micro-batch：

```text
Rank r:
  g_r = grad L(batch_r; theta)
```

Backward 后同步平均梯度：

```text
g = (1/d) * sum(g_r), r = 0 ... d-1
```

对应代码：

```python
loss = model(local_batch).loss
loss.backward()
dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, group=dp_group)
flat_grad.div_(dp_size)
optimizer.step()
```

真实 DDP 使用 Gradient Bucket，在某个 Bucket 的梯度就绪后立即启动异步 AllReduce，与剩余 Backward 重叠。

全局 Batch 为：

```text
Global Batch
= Micro Batch per GPU
* Gradient Accumulation Steps
* DP Size
```

DP 的主要问题：

- 参数、梯度和优化器状态在各 Rank 重复。
- 大模型梯度 AllReduce 带宽高。
- 同步训练受最慢 Rank 拖累。
- DP 增大后 Global Batch 改变，可能影响收敛。

经典解法：

- ZeRO-1/2/3 或 FSDP 分片 Optimizer、Gradient、Parameter。
- Gradient Bucket、通信计算重叠和梯度累积。
- 节点内 ReduceScatter、节点间 AllReduce 等分层通信。
- 数据均衡、固定 Shape、Backup Worker 或弹性容错。

### TP：相同数据，切一层中的矩阵

设 Linear：

```text
Y = XW
X: [M, K]
W: [K, N]
```

Column Parallel 沿输出维切 `W`：

```text
W = [W_0, W_1, ..., W_{t-1}]
Y_i = X W_i
Y = concat(Y_i)
```

每卡持有完整 `X` 和 `1/t` 输出。如果下游算子能消费分片输出，可以不执行 AllGather。

Row Parallel 沿输入维切：

```text
X = [X_0, X_1, ..., X_{t-1}]
W = vertical_concat(W_0, W_1, ..., W_{t-1})

Y = sum(X_i W_i)
```

每卡先算局部 Partial Result，再通过 AllReduce 得到完整 `Y`：

```python
y_partial = x_local @ w_local
dist.all_reduce(y_partial, group=tp_group)
```

Megatron 的 MLP 使用经典的 Column-Row 配对：

```text
X
-> ColumnParallel(gate/up)
-> GeLU or SwiGLU，保持 Hidden Shard
-> RowParallel(down)
-> AllReduce
```

Attention 则按 Head 或 QKV 输出维切分，Output Projection 再做 Row Parallel。

TP 的主要问题：

- 几乎每层都有 Collective，对延迟和带宽敏感。
- TP 过大后本地 GEMM 变小，Tensor Core 利用率下降。
- Hidden Size、Head 数和量化 Group 必须可切分。
- 跨节点 TP 往往被低带宽和高延迟限制。

经典解法：

- 把 TP 限制在 NVLink/NVSwitch 域内。
- 使用 SP，把 TP AllReduce 改为 ReduceScatter + AllGather 并节省 Activation。
- 异步 AllGather/ReduceScatter 与 GEMM 重叠。
- 通信融合、Fused Kernel，以及必要时采用 2D/多维 Tensor Sharding。

## 三、PP 与 EP：切深度还是切专家

### PP：把不同层放到不同 Stage

`p` 路 Pipeline Parallel 把 `L` 层划成多个 Stage：

```text
Stage 0: layer [0, k)
Stage 1: layer [k, 2k)
...
Stage p-1: remaining layers
```

Forward 只发送 Stage 边界 Activation：

```text
Stage i --Send activation--> Stage i+1
```

Backward 反向发送 Activation Gradient：

```text
Stage i+1 --Send dActivation--> Stage i
```

为了让多个 Stage 同时工作，一个 Global Batch 被拆成 `m` 个 Micro-batch。以 1F1B 为例，Warmup 后每个 Stage 交替执行一个 Forward 和一个 Backward：

```text
warmup -> 1F1B steady state -> cooldown
```

理想均衡、忽略通信时，流水线 Bubble 比例可近似理解为：

```text
Bubble ≈ (p - 1) / (m + p - 1)
```

`m` 越多，Bubble 越小，但在途 Activation、调度次数和梯度累积窗口会增加。

PP 的主要问题：

- Pipeline Bubble。
- 各 Stage 计算量或显存不均。
- Stage 边界 Activation 通信。
- Micro-batch 多时 Activation 峰值和调度复杂。

经典解法：

- GPipe 的全 Forward/全 Backward改为 1F1B，降低 Activation 峰值。
- Interleaved 1F1B / Virtual Pipeline，让每卡持有多个 Layer Chunk，缩短 Bubble。
- 按实测耗时而不是层数均分 Stage。
- Activation Checkpointing、Zero-Bubble Schedule 和双向 Pipeline。

### EP：参数按专家切，Token 按路由移动

MoE 层可写为：

```text
y_x = sum_{e in TopK(x)} p_e(x) * Expert_e(x)
```

`e` 路 Expert Parallel 把不同 Expert 放到不同 Rank。Router 在本地得到每个 Token 的目标 Expert 后，执行：

```text
local tokens
-> permute by destination expert
-> AllToAll dispatch
-> local expert GEMM
-> AllToAll combine
-> unpermute to original token order
```

简化代码：

```python
send = pack_tokens_by_expert(hidden, expert_ids)
recv = all_to_all_variable(send, split_sizes, group=ep_group)
expert_out = grouped_gemm(recv, local_expert_weights)
output = inverse_all_to_all_and_unpack(expert_out)
```

与 TP 不同，EP 不是让所有卡共同计算同一个 Expert，而是把 Token 发给拥有目标 Expert 的 Rank。

EP 的主要问题：

- Router 可能把大量 Token 发给少数 Hot Expert。
- 每个 Rank 收发 Token 数不同，需要 AllToAllv 或 Padding。
- 小 Expert Batch 导致大量小 GEMM。
- Token Dispatch 跨节点时容易成为瓶颈。
- Capacity 限制和 Token Dropping 会改变训练语义。

经典解法：

- Load-balancing Auxiliary Loss、Router Z-loss 或无辅助损失的 Balanced Routing。
- Capacity Factor、Token Dropping；要求无损语义时采用 Dropless MoE。
- 多个本地 Expert 合并为 Grouped GEMM。
- Token Permutation Fusion、DeepEP/Flex Dispatcher 和通信计算重叠。
- Expert Replication、Node-limited Routing 或分层 AllToAll，减少跨节点流量。

## 四、CP 与 SP：都切序列但不是一回事

### CP：所有层保持 Context Shard

`c` 路 CP 把：

```text
X: [B, S, H]
```

切为：

```text
X_i: [B, S/c, H]
```

Linear、Norm 和 MLP 都能处理本地 Token。Attention 中，本地：

```text
Q_i
```

必须读取全局 K/V：

```text
O_i = softmax(Q_i K^T / sqrt(D) + mask) V
```

因此使用 KV AllGather、P2P Ring 或 Ulysses AllToAll。Ring 方案让 KV Block 逐站流动，并用 Online Softmax 合并局部结果。

CP 的主要问题：

- KV 通信与 Attention 同时增长。
- Causal Mask 导致不同序列块计算不均。
- CP 过大后本地 Attention 太小。
- Packed Sequence、全局位置和 Mask 很容易出错。

经典解法：

- 双缓冲 Ring，把 KV P2P 与当前 Attention Block 重叠。
- Zigzag / Load-balanced Ring，平衡 Causal 三角区域。
- GQA/MQA 减少 KV Head，从源头降低通信量。
- AllGather、Ulysses、Ring 或 `a2a+p2p` 混合，按网络与 Head 数选择。

更完整的推导见 [Context Parallelism 详解](/blog/distributed-training-context-parallelism/)。

### SP：复用 TP 通信边界保存部分 Activation

本文中的 SP 特指 Megatron Sequence Parallelism。它不是一套独立的长上下文 Attention 算法，而是 TP 的 Activation 优化。

普通 TP 在 Row Parallel 输出处执行 AllReduce，使每个 TP Rank 都得到完整序列：

```text
partial [S, H]
-> AllReduce
-> replicated [S, H]
```

SP 利用：

```text
AllReduce = ReduceScatter + AllGather
```

把数据流改为：

```text
RowParallel output
-> ReduceScatter(sequence)
-> [S/t, H]
-> local Dropout / Residual / LayerNorm
-> AllGather(sequence)
-> next ColumnParallel Linear
```

于是 LayerNorm、Dropout 等区域的 Activation 每卡约降为 `1/t`，总通信字节并未因拆分 AllReduce 而增加。

SP 的主要问题：

- 依赖 TP Group，`SP Size` 通常就是 `TP Size`，不是额外的 GPU 维度。
- Attention 区域仍需 TP 所要求的完整布局。
- AllGather 临时 Buffer 仍会形成显存峰值。
- 只能缓解部分 Activation，不能独立支撑极长 Context。

经典解法：

- 使用异步 Sequence AllGather 与 Column Parallel GEMM 重叠。
- 配合 Activation Checkpointing。
- 真正长序列使用 CP；TP + SP + CP 可以同时开启。

## 五、六种并行的通信原语

理解 Collective 比背框架参数更重要。

### 四种 Collective

设每个 Rank 有一个 Tensor Shard。

```text
AllReduce:
  每卡输入完整形状的 Partial Result
  求和后每卡得到相同完整结果

ReduceScatter:
  求和，并让每卡只保留结果的一段

AllGather:
  收集所有 Shard，每卡得到完整 Tensor

AllToAll:
  每卡把不同分片发给不同目标卡
  每卡最终收到来自所有源 Rank 的目标分片
```

它们与六种并行的关系为：

| 并行 | Forward | Backward |
| --- | --- | --- |
| DP | 通常无通信 | Gradient AllReduce / ReduceScatter |
| TP | AllReduce 或 RS/AG | 对偶的 AllReduce、AG/RS |
| PP | Send Activation | Send dActivation |
| EP | AllToAll Dispatch + Combine | 反向 AllToAll |
| CP | KV AllGather/P2P/AllToAll | dKV ReduceScatter 或 Ring 累加 |
| SP | Sequence AllGather + ReduceScatter | 对偶 RS/AG |

### 为什么通信位置不同

通信由“谁拥有参数、谁拥有输入、谁需要完整结果”决定。

```text
DP:
  参数复制，数据不同
  -> 参数梯度需要同步

TP:
  参数和计算都切分
  -> 层内 Partial Result 需要组合

PP:
  不同层在不同卡
  -> 只传 Stage 边界 Tensor

EP:
  参数按 Expert 所有权分布
  -> Token 必须移动到参数所在卡

CP:
  Query 按 Context 分布
  -> KV 必须移动到 Query 所在卡
```

通信优化的共同原则是：

- 大消息优先带宽，小消息优先延迟。
- 尽量让 Collective 与独立计算重叠。
- 高频通信留在 NVLink 域，低频大粒度通信再跨节点。
- 避免因为布局转换增加额外 AllGather/Transpose。

## 六、多维并行怎样组合

Dense Transformer 常用设备网格：

```text
World Size = DP * PP * CP * TP
```

SP 复用 TP Group，不再乘一次：

```text
World Size != DP * PP * CP * TP * SP
```

例如 64 张 GPU：

```text
TP = 4
PP = 4
CP = 2
DP = 2

4 * 4 * 2 * 2 = 64
SP = enabled on the TP group
```

配置示例：

```bash
torchrun ... pretrain_gpt.py \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 4 \
  --context-parallel-size 2 \
  --sequence-parallel \
  --use-distributed-optimizer
```

可以把一个 Rank 看成坐标：

```text
rank = (dp_rank, pp_rank, cp_rank, tp_rank)
```

不同 Collective 只在对应坐标变化的 Process Group 内执行：

```python
# 示意：每个维度有独立 communicator
dp_group = ranks_with_same(pp, cp, tp)
tp_group = ranks_with_same(dp, pp, cp)
pp_group = ranks_with_same(dp, cp, tp)
cp_group = ranks_with_same(dp, pp, tp)
```

### MoE 不能简单把 EP 再乘进去

现代 Megatron 的 Dense Attention/MLP Mesh 与 Expert Mesh 可以在同一个 PP Stage 内复用同一批 GPU。约束更准确地写为：

```text
World Size / PP
= TP * CP * DP
= EP * ETP * EDP
```

其中：

```text
ETP: Expert Tensor Parallel size
EDP: Expert Data Parallel size
```

所以通常不是：

```text
World Size = DP * PP * TP * CP * EP
```

例如同一组 8 张卡可以在 Dense Attention 中组成 `TP=2, CP=4`，进入 MoE 层后重新组成 `EP=8`。这是不同 Layer 使用不同 Mesh Shape，而不是再需要 8 倍 GPU。

### 拓扑映射

一个实用原则：

```text
节点内高带宽域：
  TP、较小 EP、较小 CP

跨节点：
  DP、PP，以及不可避免的大规模 EP/CP
```

原因是 TP 每层通信，延迟最敏感；PP 只在 Stage 边界通信，更容易跨节点。EP 和 CP 是否跨节点取决于 Token/KV 通信量及网络能力。

## 七、问题、经典解法与选型

先用一张表压缩结论：

| 并行 | 最大收益 | 核心问题 | 经典解决办法 |
| --- | --- | --- | --- |
| DP | 扩大吞吐 | 状态复制、Gradient 通信 | ZeRO/FSDP、Bucket Overlap、分层 AllReduce |
| TP | 切大层并分摊计算 | 每层通信、本地 GEMM 变小 | 节点内 TP、SP、通信与 GEMM 重叠 |
| PP | 切模型深度 | Bubble、Stage 不均 | 1F1B、Interleaved/VPP、负载均衡 |
| EP | 切 MoE 参数 | Token 不均、AllToAll | Balanced Routing、Grouped GEMM、DeepEP |
| CP | 切长 Context | KV 通信、Causal 不均 | Ring Overlap、Zigzag、GQA、混合 CP |
| SP | 降 TP 区域 Activation | 依赖 TP、覆盖范围有限 | RS/AG Overlap、Checkpointing、配合 CP |

选择顺序可以遵循：

```text
1. 模型和 Activation 都能放下：
   优先 DP，系统最简单、计算粒度最大。

2. 单层权重放不下：
   增加 TP，但尽量留在节点内。

3. 整个模型仍放不下：
   增加 PP，把深度分到更多 Stage。

4. 已启用 TP：
   通常同时启用 SP，降低非 Attention Activation。

5. Context 导致 Activation OOM：
   增加 CP，而不是继续盲目增大 TP。

6. 模型是 MoE：
   用 EP 按专家切分，并优先解决路由负载和 AllToAll。

7. 剩余 GPU：
   用 DP 扩吞吐，并按需要启用 ZeRO/FSDP。
```

最终配置不能只看“能否跑起来”，还要同时观测：

```text
每卡峰值显存
MFU / 每卡 TFLOPS
各 Collective 时间与重叠率
Pipeline Bubble
MoE 每 Expert Token 数
Straggler 与 P99 Step Time
```

判断并行策略是否合理的核心标准只有两个：

```text
内存：
  被切分的状态是否正好是当前瓶颈

性能：
  切分后每卡计算是否仍足够大，
  新增通信能否被高带宽链路或独立计算隐藏
```

并行度越大并不天然越快。优秀的多维并行配置，通常使用满足显存约束的最小 TP/PP/CP/EP，把尽可能多的 GPU 留给通信频率最低、扩展性最好的 DP。

参考资料：

- [Megatron Core Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596)
- [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
