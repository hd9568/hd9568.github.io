---
title: 'FlashAttention 1-4 全解析：从 IO-Aware 到 Blackwell 异步流水线'
description: '从 Online Softmax 与分块推导出发，系统讲解 FlashAttention-1 到 FlashAttention-4 的 IO 优化、并行划分、Hopper 异步流水、FP8、Blackwell TMEM、2-CTA MMA 与条件重缩放。'
category: 'Research & Work'
pubDate: '2026-08-10T12:00:00+08:00'
updatedDate: '2026-08-10T12:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [先给出四代演进主线](#一先给出四代演进主线)
2. [标准 Attention 到底慢在哪里](#二标准-attention-到底慢在哪里)
3. [Online Softmax 是所有版本的数学基础](#三online-softmax-是所有版本的数学基础)
4. [FlashAttention-1：不再物化完整注意力矩阵](#四flashattention-1不再物化完整注意力矩阵)
5. [FlashAttention-1 的反向重计算](#五flashattention-1-的反向重计算)
6. [FlashAttention-2：减少非矩阵乘并重做并行划分](#六flashattention-2减少非矩阵乘并重做并行划分)
7. [FlashAttention-2 的 Split-Q 与反向并行](#七flashattention-2-的-split-q-与反向并行)
8. [FlashAttention-3：为 Hopper 重写异步流水线](#八flashattention-3为-hopper-重写异步流水线)
9. [FlashAttention-3 如何重叠 GEMM 与 Softmax](#九flashattention-3-如何重叠-gemm-与-softmax)
10. [FlashAttention-3 的 FP8 与数值处理](#十flashattention-3-的-fp8-与数值处理)
11. [FlashAttention-4：瓶颈为什么再次转移](#十一flashattention-4瓶颈为什么再次转移)
12. [FlashAttention-4 前向：指数函数也要做流水线](#十二flashattention-4-前向指数函数也要做流水线)
13. [条件重缩放为什么仍然正确](#十三条件重缩放为什么仍然正确)
14. [FlashAttention-4 反向：TMEM 与 2-CTA MMA](#十四flashattention-4-反向tmem-与-2-cta-mma)
15. [确定性反向与负载均衡](#十五确定性反向与负载均衡)
16. [四个版本的源码与实现语言](#十六四个版本的源码与实现语言)
17. [Prefill、Decode 与 Flash-Decoding 的边界](#十七prefilldecode-与-flash-decoding-的边界)
18. [四代版本横向对比](#十八四代版本横向对比)
19. [常见误解](#十九常见误解)
20. [面试时如何完整回答](#二十面试时如何完整回答)
21. [参考资料](#二十一参考资料)

## 一、先给出四代演进主线

FlashAttention 1 到 4 始终计算同一个核心对象：

```text
S = Q K^T * scale
P = softmax(S + mask)
O = P V
```

它们没有通过稀疏化、低秩分解或核函数近似来减少注意力连接。演进发生在计算顺序、线程分工、内存层级和硬件流水线上。

| 版本 | 主要硬件 | 当时的主要瓶颈 | 核心优化 |
| --- | --- | --- | --- |
| FlashAttention-1 | Ampere 等 | `S/P` 的 HBM 物化与读写 | Tiling、Online Softmax、融合、反向重计算 |
| FlashAttention-2 | A100 | 并行度不足、非 Matmul 开销、Warp 通信 | 延迟归一化、序列并行、Split-Q |
| FlashAttention-3 | H100/Hopper | 没有利用异步 WGMMA/TMA，Softmax 形成气泡 | Warp Specialization、Ping-pong、GEMM/Softmax 重叠、FP8 |
| FlashAttention-4 | B200/Blackwell 为重点 | Tensor Core 增长快于 MUFU 和 SMEM | 全异步流水、软件 `exp2`、条件重缩放、TMEM、2-CTA MMA |

可以把四代压缩成四句话：

```text
FA1：不要把 N x N 中间矩阵写回 HBM。
FA2：不要让 SM、Warp 和 Tensor Core 闲着。
FA3：不要按同步程序使用异步硬件。
FA4：不要假设矩阵乘仍然是最慢的部分。
```

这里的最后一句尤其重要。到 Blackwell，Tensor Core 的增长速度远快于指数运算单元和 Shared Memory 带宽。矩阵乘越快，Softmax、数据搬运和同步在总时间中的占比反而越大。

## 二、标准 Attention 到底慢在哪里

### 2.1 形状与计算量

对单个 Attention Head，设：

```text
Q: [Nq, d]
K: [Nk, d]
V: [Nk, dv]
S: [Nq, Nk]
P: [Nq, Nk]
O: [Nq, dv]
```

自注意力常有 `Nq = Nk = N` 和 `d = dv`。两次矩阵乘的计算量约为：

```text
QK^T: 2 * N^2 * d FLOPs
PV:   2 * N^2 * d FLOPs
合计: 4 * N^2 * d FLOPs
```

FlashAttention 没有改变这个二次计算复杂度。Dense Attention 仍然需要计算几乎所有 `Q-K` 点积。

它改变的是额外内存占用和 HBM 流量。

### 2.2 朴素实现会物化 S 和 P

将公式直接拆成多个 Kernel，典型数据流是：

```text
HBM: Q, K
  -> GEMM
HBM: S
  -> Mask + Softmax
HBM: P
  -> GEMM with V
HBM: O
```

`S` 和 `P` 都是 `N x N`。它们不仅占显存，还会在不同 Kernel 之间被写入和重新读取。

以单个 Head、BF16 为例：

```text
N = 16,384
N^2 = 268,435,456 elements
一个 N x N 矩阵约为 512 MiB
```

如果训练前向同时保留更多中间状态，再乘 Batch、Head 和 Layer 数量，二次中间张量很快成为显存瓶颈。

### 2.3 FLOPs 少不等于运行快

GPU 不是一个只有“峰值 FLOPs”的均匀计算器。Attention 同时使用多种资源：

```text
QK^T / PV       -> Tensor Core
max / sum       -> CUDA Core + Reduction
exp / exp2      -> Special Function Unit
Mask / rescale  -> CUDA Core
数据搬运         -> HBM / L2 / Shared Memory
同步             -> Barrier / Scheduler
```

GEMM 的数据复用高，适合 Tensor Core。Softmax 的 `max`、`exp`、`sum` 和逐元素缩放不是同一种工作。即使这些操作的 FLOPs 很少，它们也可能占据大量时钟周期。

因此性能分析至少要同时看：

```text
1. HBM 读写量
2. Shared Memory 读写量
3. Tensor Core 利用率
4. 非 Matmul 单元吞吐
5. CTA 数量和负载均衡
6. 寄存器与片上存储压力
```

### 2.4 “精确 Attention”是什么意思

FlashAttention 被称为 Exact Attention，是因为它没有改变目标数学函数：

```text
O = softmax(QK^T * scale + mask) V
```

但“数学上精确”不表示与另一种 Kernel 逐 bit 相同。分块改变了浮点加法和归约顺序，FP16/BF16/FP8 Kernel 还会使用不同的中间精度。因此实际输出允许存在符合误差容限的浮点差异。

## 三、Online Softmax 是所有版本的数学基础

分块计算最大的障碍不是 `QK^T`，而是 Softmax。矩阵乘天然可以按块累加，但一行 Softmax 的分母依赖整行：

```text
softmax(s_i) = exp(s_i) / sum_j exp(s_j)
```

数值稳定版本先减去全行最大值：

```text
m = max_j(s_j)
l = sum_j exp(s_j - m)
p_j = exp(s_j - m) / l
```

如果当前只能看到一个 Score Tile，就不知道最终的 `m` 和 `l`。Online Softmax 解决了这个问题。

### 3.1 合并两个 Score Block

假设已经处理集合 `A`，维护：

```text
m_A = max(A)
l_A = sum_{x in A} exp(x - m_A)
o_A = sum_{x in A} exp(x - m_A) * v_x
```

这里 `o_A` 是未除以 `l_A` 的输出分子。

新块 `B` 的局部状态：

```text
m_B = max(B)
l_B = sum_{x in B} exp(x - m_B)
o_B = sum_{x in B} exp(x - m_B) * v_x
```

合并时先选择新的基准最大值：

```text
m = max(m_A, m_B)
```

然后把两边都换算到基准 `m`：

```text
l = exp(m_A - m) * l_A
  + exp(m_B - m) * l_B

o = exp(m_A - m) * o_A
  + exp(m_B - m) * o_B
```

最终：

```text
O = o / l
```

这就是 FlashAttention 的核心不变量。它允许 Score 沿 KV 维分成任意多个 Tile，并且只维护每个 Query Row 的：

```text
running max m
running sum l
unnormalized output o
```

### 3.2 为什么旧输出必须重缩放

假设旧最大值为 `m_old`，新 Tile 出现更大值 `m_new`。旧累加器使用的是基准 `m_old`：

```text
o_old = sum exp(s - m_old) * v
```

若后续统一使用 `m_new`，旧状态必须乘：

```text
alpha = exp(m_old - m_new)
```

得到：

```text
alpha * o_old = sum exp(s - m_new) * v
```

不做这一步，来自不同 Tile 的权重不在同一个指数尺度上，结果必然错误。

### 3.3 状态可以做并行归并

上述状态合并具有结合性。不同 KV 分片可以分别计算 `(m, l, o)`，最后再做归并：

```text
(m_1, l_1, o_1) merge (m_2, l_2, o_2)
```

这也是 Flash-Decoding、Split-K Attention 和多 CTA KV 归并的数学基础。

### 3.4 更接近 Kernel 的伪代码

```python
for q_tile in Q:
    m = -inf
    l = 0
    o = 0

    for k_tile, v_tile in zip(K, V):
        s = q_tile @ k_tile.T * scale
        s = apply_mask(s)

        tile_max = rowmax(s)
        m_new = maximum(m, tile_max)
        alpha = exp(m - m_new)
        p = exp(s - m_new[:, None])

        l = alpha * l + rowsum(p)
        o = alpha[:, None] * o + p @ v_tile
        m = m_new

    output = o / l[:, None]
```

真正的优化差异，主要发生在这些状态放在哪里、哪个 Warp 更新、矩阵乘与 Softmax 是否重叠，以及如何减少 `alpha`、`exp`、同步和片上搬运。

## 四、FlashAttention-1：不再物化完整注意力矩阵

### 4.1 基本思想

FlashAttention-1 将 `Q` 按行切成 `Br x d` Tile，将 `K/V` 按序列维切成 `Bc x d` Tile。

每次只形成：

```text
S_ij = Q_i K_j^T
shape = [Br, Bc]
```

这个小 Score Tile 留在片上。完成 Softmax 更新和 `P_ij V_j` 后即可丢弃，不写入 HBM。

融合后的数据流变成：

```text
HBM Q/K/V
  -> Tile load
  -> QK^T
  -> Mask
  -> Online Softmax
  -> PV
  -> 更新 O/m/l
HBM O/LSE
```

`S` 和 `P` 从“需要跨 Kernel 保存的张量”变成“片上短生命周期中间值”。

### 4.2 原论文的循环顺序

FlashAttention-1 论文算法采用：

```text
outer loop: K_j, V_j
inner loop: Q_i, O_i, m_i, l_i
```

即先把一个 `K/V` Tile 搬到 SRAM，再遍历所有 `Q` Tile。这样能复用当前 `K/V`，但 `Q_i`、`O_i`、`m_i`、`l_i` 会随不同 `K/V` Tile 被反复读写。

简化表示：

```python
for k_tile, v_tile in KV_tiles:
    load_to_sram(k_tile, v_tile)

    for q_tile_id in Q_tiles:
        q, o, m, l = load_state(q_tile_id)
        s = q @ k_tile.T
        o, m, l = online_update(o, m, l, s, v_tile)
        store_state(q_tile_id, o, m, l)
```

后续 FA2 会交换主循环组织，让一个 CTA 固定负责一个 Q Tile，并把它的状态尽量留在片上。

### 4.3 Tiling 不只是“把矩阵切小”

Tile Size 需要同时满足：

```text
Q Tile
K/V Tile
Score Tile
Softmax 统计量
Output Accumulator
Pipeline Buffer
```

能够放入有限的 Shared Memory 和寄存器。

Tile 太小：

- Tensor Core Tile 利用差。
- 循环次数和边界判断增加。
- HBM Tile 加载次数增加。

Tile 太大：

- Shared Memory 或寄存器不足。
- Occupancy 降低。
- 可能发生 Register Spill，反而访问 Local Memory/HBM。

因此 FlashAttention 的性能不是只由算法决定，还依赖 Head Dimension、数据类型、架构和具体 Tile 配置。

### 4.4 IO 复杂度

设：

```text
N: sequence length
d: head dimension
M: on-chip SRAM capacity，按元素计
```

FlashAttention-1 论文给出的 HBM 访问复杂度为：

```text
O(N^2 * d^2 / M)
```

标准 Attention 至少需要：

```text
Omega(N*d + N^2)
```

关键比例是：

```text
d^2 / M
```

在实用区域中，片上存储通常显著大于 `d^2`，所以 FlashAttention 的 HBM 访问量远小于物化 `N x N` 矩阵的实现。

这里必须区分两个结论：

```text
计算复杂度：仍然是 O(N^2 * d)
额外内存与中间矩阵物化：从 O(N^2) 降为 O(N)
```

FlashAttention 解决的是 Dense Attention 的 IO 与存储问题，不是把二次计算变成线性计算。

### 4.5 Causal Mask 可以跳过整个 Tile

对于 Causal Attention：

```text
query position i 只能访问 key position j <= i
```

如果一个 `Q_i x K_j` Tile 完全位于因果对角线右上方，这个 Tile 可以直接跳过，不需要先计算再逐元素写 `-inf`。

跨越对角线的边界 Tile 才需要细粒度 Mask。这样既减少计算，也减少 Mask 分支。

## 五、FlashAttention-1 的反向重计算

### 5.1 标准反向需要什么

前向：

```text
S = scale * QK^T
P = softmax(S)
O = PV
```

给定 `dO`，反向为：

```text
dV = P^T dO
dP = dO V^T
D  = rowsum(dO * O)
dS = P * (dP - D)
dQ = scale * dS K
dK = scale * dS^T Q
```

其中 `D` 每个 Query Row 只有一个标量。利用：

```text
D_i = dot(dO_i, O_i)
    = sum_j P_ij * dP_ij
```

可以避免单独保存完整 `P` 来计算 Softmax 梯度的行归约。

### 5.2 Forward 保存什么

FlashAttention 不保存 `S` 和 `P`。Forward 通常保存：

```text
O:   [N, d]
LSE: [N]
```

其中：

```text
LSE_i = logsumexp(S_i)
      = m_i + log(l_i)
```

Backward 遇到一个 Score Tile 时重新计算：

```text
S_ij = scale * Q_i K_j^T
P_ij = exp(S_ij - LSE_i)
```

然后直接在片上完成该 Tile 对 `dQ/dK/dV` 的贡献。

### 5.3 为什么多算反而可能更快

这是一种典型的计算换 IO：

```text
方案 A：Forward 写出 P，Backward 从 HBM 读回 P
方案 B：不写 P，Backward 用 Q/K 和 LSE 重算 P Tile
```

矩阵乘由 Tensor Core 高吞吐执行，而写入、保存并重新读取 `N x N` 的 `P` 会消耗大量 HBM 带宽和显存。只要重计算时间小于节省的 IO 时间，方案 B 同时更省显存且更快。

### 5.4 FA1 解决后的新瓶颈

FA1 去掉最大块的 HBM 流量后，剩余问题开始显现：

- 只沿 Batch 和 Head 并行时，长序列小 Batch 可能没有足够 CTA。
- Online Softmax 的缩放、除法、Mask 和边界判断仍然昂贵。
- Warp 之间存在不必要的 Shared Memory 通信。
- Tensor Core 的时间占比不够高。

这些正是 FA2 的目标。

## 六、FlashAttention-2：减少非矩阵乘并重做并行划分

FlashAttention-2 没有更换 Online Softmax，也没有改变 Attention 数学定义。它重新设计了算法状态和 GPU Work Partition。

论文归纳了三个主要改进：

```text
1. 减少 non-matmul FLOPs。
2. 沿序列长度增加 CTA 级并行。
3. 在 CTA 内重新划分 Warp 工作，减少 Shared Memory 通信。
```

### 6.1 为什么少量 Non-matmul 也很贵

以 A100 论文数据为例：

```text
FP16/BF16 Tensor Core Matmul 峰值：312 TFLOPs/s
FP32 Non-matmul 峰值：             19.5 TFLOPs/s
```

吞吐差距约为 16 倍。因此一个 Softmax 缩放、边界判断或 FP32 逐元素操作，不能按“只占少量 FLOPs”判断成本。

### 6.2 维护未归一化输出，最后只除一次

假设每次都维护已经归一化的输出：

```text
O_normalized = o / l
```

每加入一个 KV Tile，都需要把旧输出从旧分母和旧最大值换算到新状态，涉及更多除法与逐元素缩放。

FA2 在循环中维护未归一化分子：

```text
o_new = exp(m_old - m_new) * o_old
      + exp(S_tile - m_new) @ V_tile

l_new = exp(m_old - m_new) * l_old
      + rowsum(exp(S_tile - m_new))
```

直到所有 KV Tile 完成才做：

```text
O = o / l
```

这样把归一化除法移到主循环之外，并减少中间重缩放。

### 6.3 只保存 LSE

反向不必分别保存：

```text
m: row max
l: exp sum
```

只需保存：

```text
LSE = m + log(l)
```

Backward 可直接恢复：

```text
P_ij = exp(S_ij - LSE_i)
```

这减少了状态存储和地址计算。

### 6.4 交换循环组织

FA2 Forward 的逻辑更接近：

```text
outer parallel dimension: Q Tile
inner serial scan:         all K/V Tiles
```

每个 CTA 负责一个 Q Tile：

```python
parallel for q_tile in Q_tiles:
    q = load(q_tile)
    o, m, l = init_state()

    for k_tile, v_tile in KV_tiles:
        s = q @ k_tile.T
        o, m, l = online_update(o, m, l, s, v_tile)

    store(o / l)
```

这个组织有两个收益：

1. `Q`、`O` 和 Softmax 状态能在整个 KV 扫描中留在片上。
2. `Q Tile` 成为额外的 Grid 并行维度。

Grid 大致从：

```text
batch * heads
```

扩展为：

```text
batch * heads * ceil(Nq / Br)
```

即使 Batch 和 Head 数较少，长序列也能产生足够 CTA 填满 GPU。

### 6.5 Causal 路径减少无效工作

对于一个 Q Tile，Causal Mask 限制了需要扫描的 KV Tile 范围。FA2 进一步减少：

- 完全 Mask Tile 的循环。
- 边界检查。
- 不必要的 Mask 计算。

这类控制流优化不改变渐进复杂度，但会直接影响内层循环长度。

## 七、FlashAttention-2 的 Split-Q 与反向并行

### 7.1 FA1 的 Sliced-K

一个 CTA 通常包含多个 Warp。FA1 的划分可以简化为：

```text
Q：所有 Warp 共享
K/V：沿 Head Dimension 切给多个 Warp
```

每个 Warp 只计算 `QK^T` 的部分 Reduction。由于 K 维被切开，完整 Score 需要合并多个 Warp 的 Partial：

```text
Warp 0: partial S_0
Warp 1: partial S_1
Warp 2: partial S_2
Warp 3: partial S_3

S = S_0 + S_1 + S_2 + S_3
```

这通常要求：

```text
写 Shared Memory
CTA 同步
读 Shared Memory
做归约
```

而 `PV` 阶段还会继续产生 Warp 间的数据交换。

### 7.2 FA2 的 Sliced-Q

FA2 改为沿 Query Row 切分：

```text
Warp 0: Q rows 0..r
Warp 1: Q rows r..2r
Warp 2: Q rows 2r..3r
Warp 3: Q rows 3r..4r

K/V：所有 Warp 共享
```

每个 Warp 独立负责一组完整 Query Rows：

```text
S_w = Q_w K^T
O_w = softmax(S_w) V
```

不同 Warp 写入不同的输出行，不需要为了合并 `QK^T` Partial 在 Warp 间做 Reduction。

对比：

| 划分 | Warp 拥有什么 | 是否需要合并 Score Partial |
| --- | --- | --- |
| Sliced-K | K/V 的 K 维切片 | 需要 |
| Sliced-Q | Q 的行切片 | 不需要 |

这减少了 Shared Memory 读写和同步，让更多中间状态停留在寄存器。

### 7.3 Backward 为什么更复杂

Forward 中不同 Q Tile 写入不重叠的 `O`，天然可并行。

Backward 常沿 KV Tile 并行，因为一个 CTA 固定 `K_j/V_j` 后，可以遍历 Q Tile，累加该 KV Tile 的：

```text
dK_j
dV_j
```

但同一个 `dQ_i` 会接收多个 KV Tile 的贡献：

```text
dQ_i = sum_j dS_ij K_j
```

多个 CTA 可能同时更新相同 `dQ_i`，需要：

- Atomic Add。
- 或额外 Workspace 后归并。
- 或确定性串行化策略。

这个问题会一直延续到 FA4。FA4 的 2-CTA Backward 还专门减少了 `dQ` 原子归约数量。

### 7.4 FA2 的结果

FA2 论文在 A100 上报告：

```text
相对 FA1：约 2x
理论峰值利用率：50% - 73%
峰值吞吐：约 230 TFLOPs/s FP16/BF16
```

这些数字应理解为论文特定 Shape 和硬件上的 Kernel Benchmark，不应直接与不同 GPU 上的 FA3/FA4 数字做代际加速比。

## 八、FlashAttention-3：为 Hopper 重写异步流水线

FA2 在 A100 上接近高效 GEMM，但在 H100 上只达到约 35% 的理论峰值。原因不是 FA2 算法突然失效，而是 Hopper 的高性能路径换成了新的异步硬件模型。

### 8.1 Hopper 提供了什么

#### WGMMA

Hopper 使用 Warpgroup 级矩阵乘：

```text
1 Warpgroup = 4 Warps = 128 Threads
```

`WGMMA` 是异步 Tensor Core 指令。Warpgroup 发出操作后，不必让所有普通指令同步等待矩阵乘完全结束。

#### TMA

Tensor Memory Accelerator 负责多维 Tile 搬运：

```text
HBM/GMEM -> Shared Memory
```

它能在后台完成：

- 多维地址生成。
- 边界处理。
- Tile 搬运。
- 与 Barrier 的到达通知。

相比每个线程都参与地址计算和 Load，TMA 减少了指令与寄存器压力。

#### 异步引擎并存

Hopper 上可以同时有：

```text
TMA Engine：搬下一块 K/V
Tensor Core：计算当前 QK^T 或 PV
CUDA Core/SFU：处理上一块 Softmax
```

但硬件“可以并行”不代表程序会自动并行。Kernel 必须显式安排依赖、Buffer 和 Barrier。

### 8.2 Warp Specialization

FA3 将一个 CTA 内的 Warp 分成不同角色：

```text
Producer Warpgroup:
    发起 Q/K/V 的 TMA Load
    管理多 Stage Shared Memory Buffer

Consumer Warpgroup:
    发起 WGMMA
    计算 max/exp/sum
    更新 Output Accumulator
```

Producer 不执行完整 Attention 数学，Consumer 不负责所有 Global Memory 地址生成。角色专化让不同 Warp 长期执行适合自己的指令流。

### 8.3 多 Stage Circular Buffer

Shared Memory 被组织为若干 Stage：

```text
Stage 0: Consumer 正在计算
Stage 1: Producer 正在填充
Stage 2: 等待复用
```

Producer 和 Consumer 通过 Transaction Barrier 协作：

```text
producer_wait_empty(stage)
tma_load(K_j, V_j, stage)
producer_commit(stage)

consumer_wait_full(stage)
compute(stage)
consumer_release(stage)
```

理想状态下，HBM 搬运延迟被当前 Tile 的计算覆盖。

## 九、FlashAttention-3 如何重叠 GEMM 与 Softmax

仅把 Load 和 Compute 重叠还不够。Attention 的依赖链是：

```text
QK_j^T -> Softmax_j -> P_j V_j
```

同一个 Tile 内，`PV` 必须等待 `P`，看起来无法把 Softmax 隐藏起来。但下一个 KV Tile 的 `QK_{j+1}^T` 不依赖当前 Tile 的 Softmax：

```text
QK_{j+1}^T
```

可以与：

```text
Softmax(QK_j^T)
```

并行。

### 9.1 Inter-warpgroup Ping-pong

FA3 使用两个 Consumer Warpgroup 交替：

```text
时间片 t:
    WG-A: Tensor Core GEMM
    WG-B: CUDA Core/SFU Softmax

时间片 t+1:
    WG-A: Softmax
    WG-B: GEMM
```

这里不是说两个 Warpgroup 完全独占不同物理单元，而是通过异步 WGMMA，让发起矩阵乘的 Warpgroup 与另一个执行 Softmax 指令的 Warpgroup在时间上交错。

### 9.2 Intra-warpgroup 两阶段流水

单个 Consumer Warpgroup 内也可以重排：

```text
1. 发出 QK_{j+1}^T 的异步 WGMMA
2. 同时处理 S_j 的 max/exp/sum
3. 等 P_j 就绪后发出 P_j V_j
4. 等待下一阶段必要结果
```

简化时间线：

```text
Tile j:       QK_j ---- Softmax_j ---- PV_j
Tile j + 1:           QK_j+1 ---- Softmax_j+1 ---- PV_j+1
```

重叠后，Softmax 不再完整暴露在 Tensor Core 的关键路径上。

### 9.3 为什么不能无限增加 Pipeline Stage

更多 Stage 能隐藏更多延迟，但需要保存更多：

- 下一 Tile 的 Score Accumulator。
- 当前 Tile 的 Softmax 状态。
- K/V Shared Memory Buffer。
- Output Accumulator。

Hopper 的 WGMMA 累加器主要位于寄存器。Stage 越深，寄存器压力越高，可能导致：

```text
Occupancy 下降
Register Spill
Local Memory 访问
```

因此 FA3 论文讨论了两阶段和三阶段流水，但更深不一定更快。Pipeline Depth 是延迟隐藏与片上资源之间的权衡。

### 9.4 FA3 的关键变化不是“用了 TMA”

只把 `cp.async` 换成 TMA 不能解释 FA3 的全部收益。真正的组合是：

```text
异步 Load
+ Warp 角色专化
+ 多 Stage Buffer
+ WGMMA
+ Inter-warpgroup Ping-pong
+ Intra-warpgroup GEMM/Softmax 重叠
```

这是一套围绕依赖图重新组织的流水线。

## 十、FlashAttention-3 的 FP8 与数值处理

H100 的 FP8 Tensor Core 吞吐高于 FP16/BF16，但 Attention 对量化误差敏感，尤其是 Q/K 中存在 Outlier 时。

FA3 使用两个关键方法。

### 10.1 Block Quantization

Per-tensor Scale 会被全局最大值主导。大部分普通值只能使用很小的量化区间。

Block Quantization 为更细粒度 Tile 计算 Scale：

```text
scale_block = max_abs(block) / fp8_max
q_fp8 = round(q / scale_block)
```

这样不同 Tile 能独立适应自己的动态范围。

### 10.2 Incoherent Processing

对 Q 和 K 的 Head Dimension 乘同一个正交矩阵 `M`：

```text
Q' = Q M
K' = K M
```

因为：

```text
M M^T = I
```

所以：

```text
Q' K'^T
= Q M (K M)^T
= Q M M^T K^T
= Q K^T
```

数学结果不变，但原本集中在少数通道的 Outlier 能量会被摊到更多维度，使 FP8 Block Quantization 更稳定。

工程上可以使用带随机符号的 Hadamard Transform，复杂度低于任意稠密正交矩阵乘。

### 10.3 Layout Conformance 与 V 转置

FP8 WGMMA 对输入 Layout 有特定要求。Attention 的 `P V` 数据布局不一定直接满足 Tensor Core 最优格式，因此 FA3 还需要处理：

- Score/Probability 的寄存器布局。
- FP8 Operand 的 Shared Memory Layout。
- V 的内核内转置。

低精度路径的难点不只是 Cast，还包括让前一个阶段的输出布局直接匹配下一条 MMA 指令。

### 10.4 数值结果

FA3 论文报告：

```text
FP16 Forward：最高约 740 TFLOPs/s，约 75% H100 峰值
FP8 Forward：接近 1.2 PFLOPs/s
FP8 数值误差：相对基础 FP8 Attention 低约 2.6x
```

FP16/BF16 路径仍在 FP32 中维护关键 Softmax 统计和累加状态。FP8 路径则额外依赖量化 Scale 和 Incoherent Transform。

## 十一、FlashAttention-4：瓶颈为什么再次转移

FA4 论文重点面向 B200/GB200 Blackwell。官方 CuTe-DSL 实现后来同时提供 Hopper 和 Blackwell 路径，但论文中的新硬件分析和主要优化集中在 Blackwell。

### 11.1 非对称硬件扩展

从 H100 到 B200，论文给出的每 GPU BF16/FP16 Tensor Core 峰值约从：

```text
1 PFLOPS -> 2.25 PFLOPS
```

但 B200 上：

```text
MUFU exp throughput: 16 ops / cycle / SM
SMEM read bandwidth: 128 bytes / cycle / SM
```

与 Hopper 相比没有同步翻倍。

结果是：

```text
Tensor Core 变快了
Softmax exp 没有同速变快
Shared Memory 也没有同速变快
```

FA3 中可被 GEMM 隐藏的 Softmax，在 FA4 中可能比 GEMM 本身还长。

### 11.2 Blackwell 新硬件

#### TMEM

Blackwell 每个 SM 提供 256 KB Tensor Memory：

```text
Tensor Core Accumulator -> TMEM
```

它不是普通 Shared Memory：

- 与 Tensor Core 紧密耦合。
- Warp-synchronous。
- 需要显式分配和释放。
- `tcgen05.mma` 可直接把结果累加到 TMEM。

这能降低巨型 Accumulator 对寄存器的压力。

#### Fully Asynchronous MMA

`tcgen05.mma` 由少量线程发起，结果异步写入 TMEM。普通 CUDA Core 可以在 MMA 执行时继续处理 Softmax、地址和同步逻辑。

#### 更大的 MMA Tile

Blackwell 单 CTA MMA 的 M 维可达到 128，Tile 面积比 Hopper 常用的 `64 x N` 更大。更大 Tile 提高复用，也增加 Softmax 每轮要处理的元素数。

#### 2-CTA MMA

两个 CTA 组成 Cluster Pair，协作执行一个 MMA：

```text
CTA 0: staging 一半 Operand B
CTA 1: staging 另一半 Operand B
Tensor Core: 联合消费两边数据
```

这样能减少每个 CTA 的 Shared Memory 容量与带宽压力。

### 11.3 Feeds and Speeds

FA4 对 `M=N=d=128` 的 Tile 做了资源周期估算：

| Forward 资源 | 估算周期 |
| --- | ---: |
| 2 次 MMA | 1024 |
| `M*N` 次 Exp | 1024 |
| Shared Memory | 768 |

Forward 的 Tensor Core 与 Exp 单元同时构成瓶颈。

Backward 有 5 次 MMA：

| Backward 资源 | 估算周期 |
| --- | ---: |
| 5 次 MMA | 2560 |
| Exp | 1024 |
| Shared Memory | 3328 |

Backward 的主要瓶颈变成 Shared Memory。

因此 FA4 分别处理：

```text
Forward：提高并隐藏 Exp 吞吐。
Backward：减少 Shared Memory 流量。
```

## 十二、FlashAttention-4 前向：指数函数也要做流水线

### 12.1 两个 Q Tile Ping-pong

FA4 一个 CTA 同时处理两个 Query Tile：

```text
Q_H: 128 rows
Q_L: 128 rows
```

它们交替推进：

```text
Q_H: MMA
Q_L: Softmax

Q_H: Softmax
Q_L: MMA
```

Score Accumulator 位于 TMEM。Softmax Warpgroup 从 TMEM 读 Score Row 到寄存器，完成：

```text
rowmax
exp2
rowsum
BF16 conversion
store P to TMEM
```

当 `P` 的足够分片写入 TMEM 后，就可以提前触发 `PV` MMA，不必等待整个 Probability Tile 全部处理结束。

### 12.2 Softmax Warpgroup 为什么要错开 Exp

FA4 使用两个 Softmax Warpgroup 分别处理两个 Q Tile，但会同步它们的 Exp 阶段，避免两组 Warp 同时争用吞吐有限的 MUFU。

目标不是让所有 Warp 每时每刻都执行，而是让：

```text
MUFU 持续工作但不过度争用
Tensor Core 持续工作
FMA 单元参与软件 Exp
```

### 12.3 Dedicated Correction Warpgroup

Online Softmax 在 Running Max 更新后，需要重缩放旧 Output Accumulator。

FA4 将这类 Correction 从主 Softmax 关键路径中拆出，交给专门 Warpgroup：

```text
Softmax Warpgroup:
    max / exp / sum / P

Correction Warpgroup:
    对旧 O 与 l 做必要的基准修正
```

TMEM 让不同角色能够围绕同一批 Tensor Core 中间状态协作，而不必把全部 Accumulator 常驻在发起 MMA 的 Warp 寄存器里。

### 12.4 用 FMA 软件模拟 exp2

Softmax 通常将自然指数转换为二进制指数：

```text
exp(x) = exp2(x * log2(e))
```

硬件 `MUFU.EX2` 吞吐有限，但普通 FMA 单元此时可能没有吃满。FA4 将部分 `exp2` 分配给软件多项式近似。

先做 Cody-Waite Range Reduction：

```text
x = n + f
n = floor(x)
f in [0, 1)

2^x = 2^n * 2^f
```

`2^f` 用低阶 Horner Polynomial 近似：

```text
2^f ~= p0 + f * (p1 + f * (p2 + f * p3))
```

`2^n` 可以通过浮点指数位构造。于是 Exp 负载被分到：

```text
硬件 MUFU.EX2
+ 软件 FMA Polynomial
```

FA4 使用的三阶近似针对 BF16 Attention 的最终消费精度设计。它不是把所有指数都强制改成软件实现，而是根据流水线资源配比混合使用。

## 十三、条件重缩放为什么仍然正确

Online Softmax 通常每遇到更大的 Tile Max 都更新基准：

```text
m_new = max(m_old, tile_max)
alpha = exp(m_old - m_new)
o_new = alpha * o_old + exp(S - m_new) V
```

在 Blackwell 上，频繁重缩放是明显的 Non-matmul 开销。FA4 引入 Conditional Rescaling：只有最大值跳变超过阈值 `tau` 时才更新基准并修正旧累加器。

### 13.1 两条路径

如果：

```text
tile_max - m_old > tau
```

执行标准更新：

```text
m_ref = tile_max
o = exp(m_old - m_ref) * o
  + exp(S - m_ref) V
```

如果增量不大：

```text
tile_max - m_old <= tau
```

保留旧基准：

```text
m_ref = m_old
o = o + exp(S - m_ref) V
```

`l` 使用相同基准做对应更新。

### 13.2 Softmax 基准不必始终等于真实最大值

Softmax 可以减去任意公共常数 `c`：

```text
softmax(s_i)
= exp(s_i - c) / sum_j exp(s_j - c)
```

分子分母同时乘了 `exp(-c)`，最终会抵消。

因此 Running Reference 只要对 `o` 和 `l` 保持一致，就不必每一轮都立即追上真实最大值。条件重缩放在实数数学中仍然计算同一个 Softmax。

### 13.3 阈值是数值安全边界

不更新基准时：

```text
exp(S - m_ref)
```

可能大于 1。若参考值长期落后于真实最大值，指数可能溢出或降低有效精度。

所以 `tau` 不是随意的性能开关，它控制：

```text
少做重缩放
vs.
限制指数动态范围
```

此外，软件 Polynomial Exp 与硬件 Exp 都存在有限精度误差。应准确表述为：

```text
条件基准更新在数学上不改变 Softmax；
具体 Kernel 仍受数据类型、近似指数和浮点舍入影响。
```

## 十四、FlashAttention-4 反向：TMEM 与 2-CTA MMA

### 14.1 Backward 的五次 MMA

Backward Tile 需要：

```text
1. S  = Q K^T          # 重算 Score
2. dP = dO V^T
3. dV = P^T dO
4. dQ = dS K
5. dK = dS^T Q
```

逐元素部分：

```text
P  = exp(S - LSE)
D  = rowsum(dO * O)
dS = P * (dP - D)
```

FA4 的目标是让五次 MMA 与 `P/dS` 的逐元素处理尽可能重叠。

### 14.2 为什么使用转置 Score Tile

Backward 常固定一个 KV Tile，再遍历 Q Tile，因为这便于局部累加 `dK/dV`。

FA4 直接重算转置布局：

```text
S^T
P^T
dS^T
```

这样：

```text
dV = P^T dO
dK = dS^T Q
```

可以直接消费 TMEM 中的布局，减少为了下一次 MMA 而做的 Shared Memory 转置。

### 14.3 TMEM 复用

TMEM 容量仍有限，不能同时保存五个完整 Accumulator 和全部中间 Tile。

FA4 按生命周期复用列区：

```text
区域 A: S -> P
区域 B: dP -> dS -> dQ
```

只要前一状态不再被后续阶段读取，同一片 TMEM 就能被下一个中间量覆盖。

这种优化依赖精确的流水线生命周期分析，不只是“把寄存器换成 TMEM”。

### 14.4 Softmax 与上一 Tile 的梯度 MMA 重叠

简化流水：

```text
当前 Tile j:
    重算 S_j
    计算 P_j / dP_j / dS_j

上一 Tile j-1:
    执行 dK / dQ 等 MMA
```

TMEM 承接异步 MMA Accumulator，CUDA Core 同时执行逐元素公式。

### 14.5 2-CTA MMA 如何减少 SMEM 流量

在 2-CTA 模式下，CTA Pair 协作处理更大 Tile：

```text
M = 256
N = K = 128
```

Operand B 沿 N 方向分给两个 CTA，每个 CTA 只在自己的 Shared Memory 中 Stage 一半。Tensor Core 联合读取两边。

收益：

```text
每个 CTA 的 B Operand SMEM 流量下降
单 CTA Shared Memory 占用下降
更大的协作 Tile
```

### 14.6 dQ 的 Reduction Axis 冲突

`dQ` 需要沿 KV 维归约：

```text
dQ_i = sum_j dS_ij K_j
```

而 2-CTA MMA 的默认数据切分不一定沿 `dQ` 最需要的非归约维划分。若直接照搬，两个 CTA 各自只掌握部分 Reduction 数据，无法独立得到完整 `dQ` 行。

FA4 使用 Cluster 内的 Distributed Shared Memory 交换一半 `dS`，把布局重组为：

```text
每个 CTA:
    持有一半 Query Rows
    持有完整的 2N Reduction 范围
```

于是每个 CTA 可以计算：

```text
[M/2, 2N] @ [2N, d] -> [M/2, d]
```

每个 CTA 负责不同的 `dQ` 行。

### 14.7 为什么原子加次数减半

2-CTA 设计让每个 CTA 最终只写 `dQ` Tile 的一半行。相对两个 CTA 都更新完整 `dQ` Partial 的设计，Global Atomic Reduction 数量可以减少约一半。

这同时降低：

- 原子操作成本。
- 非确定性来源。
- HBM 写流量。

## 十五、确定性反向与负载均衡

### 15.1 dQ Atomic 为什么不确定

浮点加法不满足严格结合律：

```text
(a + b) + c != a + (b + c)
```

多个 CTA 使用 Atomic Add 更新同一 `dQ` 地址时，完成顺序由运行时调度决定。因此不同运行可能有细微差异。

### 15.2 Semaphore 串行化

FA4 的 Deterministic Backward 为共享 `dQ` Tile 的 CTA 规定顺序：

```text
wait(semaphore == expected_id)
accumulate dQ
memory_fence()
increment(semaphore)
```

这会引入锁等待和 Device-wide 可见性开销，但能固定 Reduction Order。

### 15.3 为什么还需要 SPT

Causal Attention 中，不同 Tile 的循环长度不同。若长任务先占据锁，后续很多 CTA 会等待。

Deterministic 模式使用 Shortest Processing Time First 思路降低锁等待，让较短任务优先通过串行归约点。

### 15.4 普通调度使用 LPT 消除尾部

非确定性主 Kernel 更关心全局尾部延迟。Causal 和 Varlen 会导致：

```text
Tile A: 扫描很少 KV Blocks
Tile B: 扫描很多 KV Blocks
```

若最后一批恰好都是长 Tile，其他 SM 已经空闲，只剩少数 CTA 拖长 Kernel。

FA4 使用 Grid Linearization、CTA Swizzle 和 Longest Processing Time First 思路，把长 Tile 更早分发，缩短尾部。

## 十六、四个版本的源码与实现语言

官方仓库：

```text
https://github.com/Dao-AILab/flash-attention
```

### 16.1 FA1

FA1 的历史实现已经被后续版本替代。理解 FA1 更适合从论文 Algorithm、Online Softmax 和旧版 CUDA Kernel 入手。

论文：

```text
arXiv:2205.14135
```

### 16.2 FA2

当前仓库中的通用 CUDA 实现主要位于：

```text
csrc/flash_attn/
csrc/flash_attn/src/flash_fwd_kernel.h
csrc/flash_attn/src/flash_bwd_kernel.h
```

FA2 使用 CUTLASS/CuTe C++ 模板构建 Ampere/Ada 等架构路径。

### 16.3 FA3

Hopper C++ 实现位于：

```text
hopper/
hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp
hopper/flash_fwd_kernel_sm90.h
hopper/flash_bwd_kernel_sm90.h
```

文件名中的：

```text
tma   -> Tensor Memory Accelerator
gmma  -> Warpgroup MMA
ws    -> Warp Specialized
```

已经概括了 FA3 的主要硬件映射。

### 16.4 FA4

CuTe-DSL 实现位于：

```text
flash_attn/cute/
flash_attn/cute/flash_fwd_sm90.py
flash_attn/cute/flash_fwd_sm100.py
flash_attn/cute/flash_bwd_sm90.py
flash_attn/cute/flash_bwd_sm100.py
flash_attn/cute/tile_scheduler.py
```

FA4 使用嵌入 Python 的 CuTe-DSL。论文报告相对传统 C++ Template 构建有 20-30 倍编译速度提升。

这里的 Python 是 Kernel DSL 前端，不表示 Attention 在 Python 解释器中逐元素执行。CuTe-DSL 仍会生成面向 GPU 的低层 Kernel。

安装入口：

```bash
pip install flash-attn-4
```

API 示例：

```python
from flash_attn.cute import flash_attn_func

out = flash_attn_func(q, k, v, causal=True)
```

FA4 包含 SM90 与 SM100 实现。论文的核心新贡献主要针对 Blackwell 的异步 MMA、TMEM 和 2-CTA 能力。

## 十七、Prefill、Decode 与 Flash-Decoding 的边界

### 17.1 Prefill

Prefill 中：

```text
Q length ~= prompt length
KV length ~= prompt length
```

`QK^T` 和 `PV` 是较大的矩阵乘，通常具有较高 Arithmetic Intensity。FlashAttention 1-4 的主算法和训练 Forward 与此最接近。

### 17.2 Decode

普通自回归 Decode 每步：

```text
Q length = 1
KV length = current context length
```

矩阵乘的 M 维很小，主要成本是读取完整 KV Cache：

```text
HBM -> K/V Cache -> single-query attention
```

这通常是 Memory-bound 问题，而不是大 Tile Tensor Core 问题。

### 17.3 Flash-Decoding

为了让单 Query Decode 使用更多 CTA，可以把 KV Sequence 切成多个分片：

```text
CTA 0 -> KV chunk 0 -> (m0, l0, o0)
CTA 1 -> KV chunk 1 -> (m1, l1, o1)
...
```

然后用 Online Softmax 状态合并：

```text
(m, l, o) = reduce((m0,l0,o0), (m1,l1,o1), ...)
```

这叫 Split-KV 或 Flash-Decoding 思路。它与 FA 的 Online Softmax 同源，但调度目标不同。

### 17.4 PagedAttention 解决另一个问题

PagedAttention 主要解决 KV Cache 的物理存储和地址映射：

```text
logical token index
  -> block table
  -> physical KV page
```

它不等同于 FlashAttention：

```text
FlashAttention：如何计算并减少中间 IO。
PagedAttention：KV Cache 如何分页存储与访问。
Flash-Decoding：单 Query 如何沿长 KV 维增加并行。
```

现代推理框架通常把三类思想组合起来，而不是只选择一个名字。

## 十八、四代版本横向对比

### 18.1 优化对象

| 版本 | 优化层次 | 关键状态放置 |
| --- | --- | --- |
| FA1 | HBM IO 与中间张量 | Score/Probability Tile 在 SRAM/Register |
| FA2 | CTA/Warp Work Partition | Q/O/Softmax 状态尽量常驻片上 |
| FA3 | Hopper 异步执行 | K/V 在多 Stage SMEM，Accumulator 在 Register |
| FA4 | Blackwell 非对称资源 | Accumulator/中间量大量进入 TMEM |

### 18.2 硬件原语

| 版本 | 主要原语 |
| --- | --- |
| FA1 | CUDA Tiling、Shared Memory、Tensor Core |
| FA2 | CUTLASS/CuTe、Split-Q、序列 CTA 并行 |
| FA3 | TMA、WGMMA、MBarrier、Warpgroup Specialization |
| FA4 | `tcgen05.mma`、TMEM、2-CTA MMA、DSMEM、CuTe-DSL |

### 18.3 性能数字

| 版本 | 论文报告的代表结果 |
| --- | --- |
| FA1 | 相对优化基线约 2-4 倍 Attention 加速，线性额外内存 |
| FA2 | A100 上约 230 TFLOPs/s，50%-73% 理论峰值 |
| FA3 | H100 FP16 约 740 TFLOPs/s，FP8 接近 1.2 PFLOPs/s |
| FA4 | B200 BF16 最高约 1613 TFLOPs/s，71% 理论峰值 |

这些数字来自不同 GPU、数据类型、Shape 和论文版本，不能直接得出：

```text
FA4 比 FA3 固定快 1613 / 740 倍
```

正确比较必须固定：

```text
GPU
dtype
head_dim
sequence length
causal/non-causal
forward/backward
dropout
GQA ratio
```

### 18.4 每代“没有改变”的部分

```text
Dense Attention 的 O(N^2 d) 主计算量没有消失。
Online Softmax 的数学不变量没有改变。
Forward 不物化完整 S/P 的原则没有改变。
Backward 依赖局部重计算的原则没有改变。
```

变化的是怎样把这些步骤映射到当代 GPU。

## 十九、常见误解

### 19.1 “FlashAttention 把 Attention 计算复杂度降到了 O(N)”

错误。

正确说法：

```text
Dense FlashAttention 的计算仍为 O(N^2 d)。
额外中间存储从 O(N^2) 降为 O(N)。
HBM IO 显著下降。
```

### 19.2 “FlashAttention 就是 Kernel Fusion”

不完整。

仅把 `QK^T + Softmax + PV` 放进一个 Kernel，如果没有 Online Softmax 和合适的 Tiling，仍然无法避免完整行依赖或片上容量限制。

更准确的组合是：

```text
算法重排 + Online Reduction + Tiling + Fusion + Recomputation
```

### 19.3 “FA2 只是 FA1 换了更大的 Tile”

错误。

FA2 的核心是：

```text
未归一化输出状态
序列维 CTA 并行
Sliced-K -> Sliced-Q
更少边界与 Mask 开销
```

### 19.4 “FA3 只是加了 TMA”

错误。

FA3 的关键是利用异步依赖图：

```text
Load / QK GEMM / Softmax / PV GEMM
```

在 Producer/Consumer Warpgroup 与多 Stage Pipeline 中重叠。

### 19.5 “FA4 的 Tensor Core 更快，所以只要换 MMA 指令”

恰好相反。

FA4 的出发点是 Tensor Core 已经快到不再单独主导耗时。Forward 卡 Exp，Backward 卡 Shared Memory，因此需要：

```text
软件 Exp
条件重缩放
TMEM
2-CTA
调度优化
```

### 19.6 “Exact 表示逐 bit 一致”

错误。

Exact 表示数学目标没有做稀疏或低秩近似。浮点归约顺序、低精度量化和 Exp 实现不同，仍会产生数值差异。

### 19.7 “版本越高，在所有 GPU 上都越快”

错误。

- FA3 主要为 Hopper 设计。
- FA4 论文主要为 Blackwell 设计，当前包也有 Hopper 路径。
- Ampere/Ada 仍可能使用 FA2。
- 短序列、小 Head、特殊 Mask 和 Decode Shape 可能选择其他 Kernel。

实际框架会根据 GPU、Shape、数据类型和功能支持动态 Dispatch。

## 二十、面试时如何完整回答

可以按以下顺序回答。

### 20.1 先讲共同基础

```text
FlashAttention 是精确 Dense Attention 的 IO-aware 实现。
它不物化完整 N x N Score 和 Probability 矩阵，
而是按 Q/K/V Tile 在片上计算，
通过 Online Softmax 维护每行 running max、sum 和 output accumulator。
Backward 不保存 P，而是用 Q/K/V、O 和 LSE 重算局部 P Tile。
```

### 20.2 再讲四代差异

```text
FA1 解决 HBM IO：
Tiling + Online Softmax + Fusion + Recomputation。

FA2 解决 GPU 利用率：
减少 non-matmul FLOPs，沿 Q 序列增加 CTA 并行，
Warp 划分从 Sliced-K 改成 Sliced-Q，减少 Shared Memory 通信。

FA3 解决 Hopper 异步硬件利用：
TMA 搬运、WGMMA、Producer/Consumer Warp Specialization、
Ping-pong 调度，并支持带 Block Quantization 和正交旋转的 FP8。

FA4 解决 Blackwell 的非对称硬件扩展：
Tensor Core 翻倍后，Forward 的 Exp 和 Backward 的 SMEM 成为瓶颈；
它使用软件 exp2、条件重缩放、TMEM、2-CTA MMA、DSMEM 和新调度。
```

### 20.3 最后补边界

```text
FlashAttention 不降低 Dense Attention 的 O(N^2 d) 计算量。
它对训练和 Prefill 最直接。
单 Token Decode 还要结合 Split-KV、Paged KV Cache 和专用 Decode Kernel。
```

## 二十一、参考资料

1. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
2. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
3. [FlashAttention-2 官方技术博客](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)
4. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
5. [FlashAttention-3 官方技术博客](https://tridao.me/blog/2024/flash3/)
6. [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451)
7. [FlashAttention-4 官方技术博客](https://tridao.me/blog/2026/flash4/)
8. [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)
9. [FlashAttention-4 CuTe-DSL README](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute)
10. [FlexAttention + FlashAttention-4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
11. [The I/O Complexity of Attention, or How Optimal is Flash Attention?](https://arxiv.org/abs/2402.07443)

FlashAttention 的四代演进说明了同一个事实：高性能算子没有永久不变的“最佳实现”。当 HBM IO 被消除后，瓶颈转向线程划分；当线程划分改善后，瓶颈转向异步流水；当 Tensor Core 再次翻倍后，指数单元和 Shared Memory 又成为关键路径。真正稳定的方法论不是记住某个 Tile Size，而是持续识别当前硬件上最慢的资源，并围绕它重排整个算法。
