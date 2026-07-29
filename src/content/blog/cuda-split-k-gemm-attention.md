---
title: 'Split-K 详解：从 GEMM K 维并行到 Attention Split-KV'
description: '系统讲解 Split-K 如何用归约维并行补足 GPU 并行度，覆盖 Parallel/Serial/Atomic Split-K、Workspace 与 Epilogue、Split 数选择、Stream-K 区别，以及 Flash-Decoding 的 Split-KV Softmax 归并。'
category: 'CUDA'
pubDate: '2026-07-29T19:51:00+08:00'
updatedDate: '2026-07-29T19:51:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [Split-K 解决什么问题](#一split-k-解决什么问题)
2. [普通 GEMM 如何映射到 GPU](#二普通-gemm-如何映射到-gpu)
3. [Tile Quantization 与 Wave Quantization](#三tile-quantization-与-wave-quantization)
4. [Split-K 的数学定义](#四split-k-的数学定义)
5. [并行度为什么增加](#五并行度为什么增加)
6. [Parallel Split-K：两阶段归并](#六parallel-split-k两阶段归并)
7. [Serial Split-K：信号量顺序归并](#七serial-split-k信号量顺序归并)
8. [Atomic Split-K](#八atomic-split-k)
9. [Epilogue 为什么必须最后执行](#九epilogue-为什么必须最后执行)
10. [可运行的简化 CUDA 实现](#十可运行的简化-cuda-实现)
11. [CUTLASS 的官方实现模型](#十一cutlass-的官方实现模型)
12. [TensorRT-LLM 的 Parallel Split-K Grouped GEMM](#十二tensorrt-llm-的-parallel-split-k-grouped-gemm)
13. [LMDeploy 的 Serial Split-K](#十三lmdeploy-的-serial-split-k)
14. [Split 数如何选择](#十四split-数如何选择)
15. [性能模型与 Workspace 代价](#十五性能模型与-workspace-代价)
16. [Split-K 与 Sliced-K 的区别](#十六split-k-与-sliced-k-的区别)
17. [Split-K 与 Stream-K 的区别](#十七split-k-与-stream-k-的区别)
18. [Attention 中的 Split-KV](#十八attention-中的-split-kv)
19. [Split-KV 的 Softmax 正确归并](#十九split-kv-的-softmax-正确归并)
20. [Paged KV、GQA 与 Blackwell](#二十paged-kvgqa-与-blackwell)
21. [数值稳定性与确定性](#二十一数值稳定性与确定性)
22. [何时使用，何时不要使用](#二十二何时使用何时不要使用)
23. [Profile 与调优清单](#二十三profile-与调优清单)
24. [参考资料](#二十四参考资料)
25. [总结](#二十五总结)

## 一、Split-K 解决什么问题

GEMM：

```text
C[M, N] = A[M, K] @ B[K, N]
```

传统 GPU GEMM 主要沿输出矩阵的：

```text
M 维
N 维
```

划分 Thread Block。每个 CTA 独立计算一个 `C` Tile，并在 CTA 内串行遍历整个 K 维。

当 M、N 足够大时，输出 Tile 很多，GPU 并行度充足。

当：

```text
M 很小
N 很小或中等
K 很大
```

时会出现：

```text
输出 Tile 数少
-> CTA 数少
-> 大量 SM 空闲
-> 每个 CTA 却要串行扫描很长的 K
```

Split-K 的核心思路：

> 不只沿 M/N 划分输出，还把 K 归约维拆成多个片段，让多个 CTA 并行计算同一个输出 Tile 的部分和，再归并。

示意：

```text
普通 GEMM:

CTA 0:
  C_tile = A[:, 0:K] @ B[0:K, :]

Split-K, S = 4:

CTA 0: P0 = A[:, K0:K1] @ B[K0:K1, :]
CTA 1: P1 = A[:, K1:K2] @ B[K1:K2, :]
CTA 2: P2 = A[:, K2:K3] @ B[K2:K3, :]
CTA 3: P3 = A[:, K3:K4] @ B[K3:K4, :]

C_tile = P0 + P1 + P2 + P3
```

它用额外归并换取更多 CTA 并行度。

## 二、普通 GEMM 如何映射到 GPU

设 CTA Tile：

```text
BLOCK_M x BLOCK_N x BLOCK_K
```

一个 CTA 负责输出：

```text
C[
  block_m * BLOCK_M : ...,
  block_n * BLOCK_N : ...
]
```

CTA 主循环：

```cpp
acc = 0;

for (int k = 0; k < K; k += BLOCK_K) {
    load A_tile[BLOCK_M, BLOCK_K];
    load B_tile[BLOCK_K, BLOCK_N];
    acc += A_tile @ B_tile;
}

store C_tile;
```

Grid 中 CTA 数：

```text
T_MN =
    ceil(M / BLOCK_M)
    * ceil(N / BLOCK_N)
```

普通 GEMM 的 K 循环发生在 CTA 内部，因此：

```text
K 再大
也不会增加 Grid 中 CTA 数
```

这正是 Split-K 要改变的地方。

## 三、Tile Quantization 与 Wave Quantization

Split-K 的动机经常被笼统描述为“增加并行度”。更准确地说，它主要缓解两类量化问题。

### 3.1 Tile Quantization

若：

```text
M % BLOCK_M != 0
N % BLOCK_N != 0
```

边界 Tile 只有部分元素有效，但 CTA 资源仍按完整 Tile 配置。

例如：

```text
M = 136
BLOCK_M = 128
```

需要两个 M Tile，第二个 Tile 只有 8 行有效。

这叫 Tile Quantization。

Split-K 通常不直接解决它，因为输出 Tile 形状没有变化。

### 3.2 Wave Quantization

GPU 一次可并发运行有限数量 CTA。设：

```text
W =
    SM 数
    * 每个 SM 可驻留 CTA 数
```

若总 CTA 数为 `G`，理论 Wave 利用率：

```text
eta_wave =
    G
    /
    (ceil(G / W) * W)
```

例如：

```text
W = 4
G = 9
```

需要三波：

```text
4 + 4 + 1
```

理论利用率：

```text
9 / (3 * 4) = 75%
```

最后一波只有一个 SM 工作，其他 SM 等待。

Split-K 令：

```text
G_split = T_MN * S
```

其中 `S` 为 K Split 数。

若 `S = 2`：

```text
G_split = 18
eta_wave = 18 / (5 * 4) = 90%
```

但这只是并行度上限，还没有扣除归并与更短 K Slice 的效率损失。

## 四、Split-K 的数学定义

GEMM：

```text
C = alpha * A B + beta * C_old
```

把 K 划分为 S 个不重叠区间：

```text
[k_0, k_1)
[k_1, k_2)
...
[k_{S-1}, k_S)

k_0 = 0
k_S = K
```

第 `s` 个 Partial：

```text
P_s[m, n] =
    sum_{k=k_s}^{k_{s+1}-1}
        A[m, k] * B[k, n]
```

完整结果：

```text
C[m, n] =
    alpha * sum_s P_s[m, n]
    + beta * C_old[m, n]
```

成立的原因是加法结合律在实数数学中成立。

浮点数中运算顺序变化会引入舍入误差，后文单独讨论。

### 4.1 K 不一定能整除 S

可先按 K 元素均分：

```text
K_per_split = ceil(K / S)

k_begin = s * K_per_split
k_end = min(K, k_begin + K_per_split)
```

高性能 Kernel 通常按：

```text
BLOCK_K
MMA K Alignment
Scale Group
```

对齐，而不是随意切在任意元素位置。

## 五、并行度为什么增加

普通 GEMM Grid：

```text
grid = [tiles_m, tiles_n]
```

Split-K Grid：

```text
grid = [tiles_m, tiles_n, split_k]
```

每个输出 Tile 原来只有一个 CTA：

```text
(tile_m, tile_n)
```

现在有 S 个 CTA：

```text
(tile_m, tile_n, split_id)
```

因此 CTA 数扩大 S 倍。

### 5.1 典型有效 Shape

```text
M = 128
N = 128
K = 4096
BLOCK_M = 128
BLOCK_N = 128
```

普通 GEMM：

```text
T_MN = 1
只有 1 个 CTA
```

Split-K `S = 16`：

```text
16 个 CTA
每个 CTA 处理 K = 256
```

CUTLASS 官方文档使用的就是这一类例子。

### 5.2 并行度不是免费增加

原来一个 CTA 在寄存器中完成：

```text
完整 C Tile
```

现在每个 CTA 只有：

```text
Partial C Tile
```

不同 CTA 不能直接读取彼此寄存器，因此 Partial 必须通过：

- Global Workspace。
- Output Buffer + Semaphore。
- Global Atomic。
- CTA Cluster/Remote Shared Memory。

之一完成归并。

## 六、Parallel Split-K：两阶段归并

Parallel Split-K 让所有 K Slice 完全独立运行。

### 6.1 第一阶段

Grid：

```text
[tiles_m, tiles_n, S]
```

第 `s` 个 CTA 写：

```text
workspace[
  s,
  tile_m,
  tile_n,
  BLOCK_M,
  BLOCK_N
]
```

逻辑上：

```text
workspace[s, :, :] = P_s
```

### 6.2 第二阶段

Reduction Kernel：

```text
for each output element:
    acc = 0
    for s in [0, S):
        acc += workspace[s, m, n]

    C[m, n] =
        epilogue(alpha * acc + beta * C_old[m, n])
```

### 6.3 特点

优点：

- 所有 Split 完全并行。
- 第一阶段没有同一输出 Tile 的 CTA 间等待。
- Reduction 顺序清晰。
- 容易支持 FP32 Partial。

代价：

- 需要 `S * M * N` 级别 Workspace。
- Partial 要写 HBM，再被 Reduction 读回。
- 多一次 Kernel Launch。
- Epilogue 必须延迟到第二阶段。

### 6.4 Workspace

若 Partial 使用 FP32：

```text
workspace_bytes =
    S * M * N * 4
```

例子：

```text
M = N = 128
S = 16
```

Workspace：

```text
16 * 128 * 128 * 4
= 1,048,576 bytes
= 1 MiB
```

最终 BF16 输出只有：

```text
128 * 128 * 2 = 32 KiB
```

Partial Workspace 是最终输出的 32 倍。

## 七、Serial Split-K：信号量顺序归并

Serial Split-K 仍让不同 K Slice 的 GEMM 主循环并行执行，但通过每个输出 Tile 的 Semaphore 顺序提交 Partial。

简化流程：

```text
Split 0:
  compute P0
  wait semaphore == 0
  workspace = P0
  semaphore = 1

Split 1:
  compute P1
  wait semaphore == 1
  workspace += P1
  semaphore = 2

...

Last Split:
  compute P_last
  wait semaphore == last_id
  total = workspace + P_last
  apply final epilogue
  write C
  semaphore = 0
```

### 7.1 “Serial” 不表示 GEMM 主循环串行

各 CTA 可以并行计算自己的 K Slice：

```text
QK/MMA mainloop in parallel
```

串行的是：

```text
同一个输出 Tile 的 Partial Commit 顺序
```

如果 Split 0 很慢，后续 Split 即使先完成计算，也会在 Epilogue 前等待。

### 7.2 Workspace

一种实现只需要：

```text
每个输出 Tile 一个 Partial Buffer
每个输出 Tile 一个 Semaphore
```

Workspace 不必乘 S。

代价是：

- 信号量轮转。
- 等待和 Cache Coherence。
- 同一输出 Tile 的归并顺序串行。

## 八、Atomic Split-K

另一种方法是让每个 Split 直接：

```cpp
atomicAdd(&C[m, n], partial);
```

调用前先清零 C：

```text
C = 0
```

### 8.1 优点

- 不需要完整 Partial Workspace。
- 不需要独立 Reduction Kernel。
- 实现简单。

### 8.2 缺点

- 多个 Split 争用同一输出地址。
- Atomic 吞吐依赖 Dtype 和架构。
- 浮点累加顺序由调度决定，通常不确定。
- `beta * C_old` 难以直接融合。
- Bias、Activation 等 Epilogue 仍要单独处理。
- 大输出 Tile 上 Atomic 流量高。

Atomic 更适合：

- Split 数不大。
- 输出规模较小。
- 原子类型高效。
- 不要求严格确定性。

不能默认它比两阶段 Reduction 更快。

## 九、Epilogue 为什么必须最后执行

GEMM Epilogue 常包含：

```text
alpha/beta
bias
activation
quantization
gating
residual
```

正确结果：

```text
y =
  activation(
      alpha * sum_s P_s
      + beta * C_old
      + bias
  )
```

错误做法：

```text
sum_s activation(
    alpha * P_s
    + beta * C_old
    + bias
)
```

因为 Activation 通常不满足线性：

```text
ReLU(x + y) != ReLU(x) + ReLU(y)
SiLU(x + y) != SiLU(x) + SiLU(y)
```

### 9.1 Beta 也只能应用一次

若每个 Split 都计算：

```text
P_s + beta * C_old
```

最终会得到：

```text
sum_s P_s + S * beta * C_old
```

这是错误结果。

因此前 S-1 个 Split 通常只产生 Partial，最后归并完成后再执行完整 Epilogue。

## 十、可运行的简化 CUDA 实现

下面的代码强调 Split-K 语义。它让一个线程计算一个 Partial 输出元素，没有 Shared Memory Tiling 或 Tensor Core，因此不代表高性能实现。

### 10.1 Partial Kernel

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

__global__ void split_k_partial(
    const half* A,        // [M, K], row-major
    const half* B,        // [K, N], row-major
    float* partials,      // [S, M, N]
    int M,
    int N,
    int K,
    int S) {

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int split = blockIdx.z;

    if (m >= M || n >= N) {
        return;
    }

    const int k_per_split = (K + S - 1) / S;
    const int k_begin = split * k_per_split;
    const int k_end = min(K, k_begin + k_per_split);

    float acc = 0.0f;
    for (int k = k_begin; k < k_end; ++k) {
        const float a = __half2float(A[m * K + k]);
        const float b = __half2float(B[k * N + n]);
        acc = fmaf(a, b, acc);
    }

    const size_t offset =
        (static_cast<size_t>(split) * M + m) * N + n;
    partials[offset] = acc;
}
```

### 10.2 Reduction Kernel

```cpp
__global__ void reduce_split_k(
    const float* partials,
    half* C,             // 既是 C_old，也是输出
    const half* bias,
    int M,
    int N,
    int S,
    float alpha,
    float beta) {

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N) {
        return;
    }

    float total = 0.0f;
    for (int split = 0; split < S; ++split) {
        const size_t offset =
            (static_cast<size_t>(split) * M + m) * N + n;
        total += partials[offset];
    }

    const int out_idx = m * N + n;
    float y = alpha * total;
    if (beta != 0.0f) {
        // beta == 0 时不能读取可能尚未初始化的 C。
        y += beta * __half2float(C[out_idx]);
    }

    if (bias != nullptr) {
        y += __half2float(bias[n]);
    }

    // 非线性必须在所有 Partial 归并后执行。
    y = fmaxf(y, 0.0f);
    C[out_idx] = __float2half_rn(y);
}
```

### 10.3 Launch

```cpp
void launch_split_k(
    const half* A,
    const half* B,
    half* C,
    const half* bias,
    float* workspace,
    int M,
    int N,
    int K,
    int split_k,
    cudaStream_t stream) {

    dim3 block(16, 16);
    dim3 partial_grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y,
        split_k);

    split_k_partial<<<partial_grid, block, 0, stream>>>(
        A, B, workspace, M, N, K, split_k);

    dim3 reduce_grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    reduce_split_k<<<reduce_grid, block, 0, stream>>>(
        workspace,
        C,
        bias,
        M,
        N,
        split_k,
        1.0f,
        0.0f);
}
```

高性能版本会把 `split_k_partial` 替换为：

- CTA Tiling。
- Shared Memory/TMA Pipeline。
- Tensor Core MMA。
- Vectorized Load。
- Swizzle。
- Warp Specialization。

但两阶段数据依赖与此代码相同。

## 十一、CUTLASS 的官方实现模型

CUTLASS 官方文档把跨 CTA 的方案称为：

```text
parallel reduction splitK
```

实现包括两个 Kernel：

```text
Partitioned-K GEMM
-> Batched Reduction
```

设备级接口：

```text
cutlass::gemm::device::GemmSplitKParallel
```

官方例子：

```text
M = 128
N = 128
K = 4096
S = 16
```

得到 16 个：

```text
[128, 128, 256]
```

的 Partial GEMM。

### 11.1 不整除

CUTLASS 文档还给出：

```text
K = 4096
S = 20
```

其中前 19 个 Partition 处理 204，最后一个处理 220。

实际 Kernel 要保证切分边界满足 MMA/Load Alignment，并正确处理 Tail。

### 11.2 Workspace 所有权

CUTLASS 要求调用方提供存放 Partial 的 Workspace。

这意味着 Split-K 不能只看 Kernel 本身，还要把：

- Workspace 分配。
- 清零。
- 生命周期。
- CUDA Graph 地址稳定性。

纳入算子设计。

## 十二、TensorRT-LLM 的 Parallel Split-K Grouped GEMM

TensorRT-LLM 的 `splitkGroupGemm` 是一个直接的 Parallel Split-K 例子。

### 12.1 接口

```cpp
void splitkGroupedGemm(
    problem_sizes,
    ptrA,
    ptrB,
    ptrC,
    ptrD,
    params_workspace,
    gemm_workspace,
    split_k_slices,
    stream);
```

它同时处理多个不同 Shape 的 GEMM，并为每个 Problem 做 Split-K。

### 12.2 Workspace 公式

源码按所有 Problem 的输出元素总数计算：

```cpp
workspace =
    sum_i(M_i * N_i)
    * sizeof(float)
    * split_k_slices;
```

Partial 使用 FP32，最终再转换为 FP16/BF16。

### 12.3 第一阶段 Grid

```cpp
dim3 grid(
    threadblock_count,
    1,
    split_k_slices);
```

`grid.z` 就是 K Partition。

### 12.4 第二阶段

Reduction Kernel：

```cpp
for each problem:
    for each output element:
        float sum = 0
        for split in [0, S):
            sum += partial[split, element]
        output[element] = cast(sum)
```

这条路径清晰体现：

```text
S 倍 Partial Workspace
两次 Kernel Launch
完全并行的第一阶段
```

## 十三、LMDeploy 的 Serial Split-K

LMDeploy TurboMind GEMM 展示了另一种设计。

### 13.1 Scheduler 切 K

Grid 第三维：

```cpp
tiles_[2] = splits;
```

Scheduler 把 K Chunk 尽量均匀分配给 Split：

```cpp
chunks = ceil(K / chunk_k);
base = chunks / splits;
remainder = chunks % splits;
```

每个 Split 的 Chunk 数最多相差 1。

### 13.2 信号量

Epilogue 为每个输出 Tile 维护：

```text
barrier[tile_id]
```

第 `s` 个 Split：

```cpp
sem_wait(barrier, s);
```

保证按：

```text
0 -> 1 -> 2 -> ... -> S-1
```

归并。

### 13.3 Partial Buffer

首个 Split：

```text
workspace = P0
```

中间 Split：

```text
workspace += Ps
```

最后 Split：

```text
total = workspace + P_last
-> alpha/beta
-> activation
-> output
```

### 13.4 Workspace 不乘 S

该实现明确使用 Serial 模式：

```cpp
barriers_size =
    sizeof(int) * output_tiles;

partials_size =
    sizeof(float)
    * CTA_M * CTA_N
    * output_tiles;
```

Partial Workspace 只保存每个输出 Tile 的 Running Sum，不为每个 Split 单独分配一份。

这是：

```text
更小 Workspace
vs.
信号量等待与顺序提交
```

之间的取舍。

## 十四、Split 数如何选择

`S` 是 Split-K 最重要的超参数。

### 14.1 并行度下界

基础输出 Tile：

```text
T_MN =
    ceil(M / BLOCK_M)
    * ceil(N / BLOCK_N)
```

设备可同时驻留：

```text
W =
    num_sms
    * active_ctas_per_sm
```

若希望至少一波满载：

```text
S >= ceil(W / T_MN)
```

若希望有两波工作以隐藏 CTA 尾部：

```text
S >= ceil(2W / T_MN)
```

这只是并行度估算，不是最终最优值。

### 14.2 K Slice 不能太短

设每次 Mainloop 消耗：

```text
BLOCK_K
```

Split 后每个 CTA 的 K Iteration：

```text
iters_per_split ≈
    ceil(K / BLOCK_K) / S
```

若只剩 1 到 2 次迭代：

- Pipeline 预热/排空占比高。
- TMA/Async Copy 难以隐藏。
- MMA 指令太少。
- Epilogue 占比高。

所以应限制：

```text
K_per_split >= K_min
```

`K_min` 取决于架构、Dtype 和 Kernel Pipeline。

### 14.3 Workspace 上限

Parallel Split-K：

```text
S <=
  workspace_limit
  /
  (M * N * accumulator_bytes)
```

大 M/N 时，即使 K 很大，Workspace 也可能阻止使用较大 S。

### 14.4 对齐

K Slice 边界通常要满足：

- Vector Load Alignment。
- Tensor Core MMA K。
- Quantization Group Size。
- Block Scale Vector Size。

量化 GEMM 尤其不能随意切断一个 Scale Group。

### 14.5 实用启发式

```python
def choose_split_k(
    m, n, k,
    block_m, block_n, block_k,
    resident_ctas,
    max_splits,
    min_k_iters=4):

    output_tiles = ceil_div(m, block_m) * ceil_div(n, block_n)

    if output_tiles >= 2 * resident_ctas:
        return 1

    splits_for_parallelism = ceil_div(
        2 * resident_ctas,
        output_tiles,
    )

    total_k_iters = ceil_div(k, block_k)
    splits_for_pipeline = max(
        1,
        total_k_iters // min_k_iters,
    )

    return max(
        1,
        min(
            splits_for_parallelism,
            splits_for_pipeline,
            max_splits,
        ),
    )
```

实际库还会比较多个候选的实测时间。

## 十五、性能模型与 Workspace 代价

粗略时间：

```text
T_split =
    T_partial_gemm
    + T_partial_write
    + T_reduction_read
    + T_reduction
    + T_extra_launch
```

收益来自：

```text
T_partial_gemm
```

因并行度增加而下降。

代价来自后四项。

### 15.1 FLOPs 没有减少

总 GEMM FLOPs：

```text
2 * M * N * K
```

Split-K 不减少数学计算量，只改变分工。

### 15.2 输入流量

各 Split 读取不重叠 K 区间，理想情况下 A/B 的算法级总字节量不会因 S 直接乘 S。

但实际可能恶化：

- 每个 Slice 重新执行 Prologue。
- Cache Reuse 变化。
- 边界对齐和 Padding。
- 更短 Pipeline。
- 更大的 Grid 调度开销。

### 15.3 输出流量

Parallel Split-K 额外产生：

```text
Partial Write:
  S * M * N * accumulator_bytes

Partial Read:
  S * M * N * accumulator_bytes
```

约：

```text
2 * S * M * N * accumulator_bytes
```

的 Workspace 流量。

当 K 不够大时，这些流量会抵消并行收益。

## 十六、Split-K 与 Sliced-K 的区别

两者都切 K，但切分层级不同。

### Split-K

```text
跨 CTA 切 K
```

特点：

- 增加 Grid CTA 数。
- 能占用更多 SM。
- 需要跨 CTA 归并。
- 通常经过 Global Memory/Atomic/Cluster。

### Sliced-K

```text
同一 CTA 内，不同 Warp 切 K
```

特点：

- 不增加 CTA 数。
- 增加 CTA 内 Warp 并行。
- Partial 在 CTA 内归并。
- 可通过 Shared Memory/Warp Shuffle。
- 适合 M/N 小、CTA 内 Warp 利用不足。

Sliced-K 不能解决：

```text
整个 Grid 只有几个 CTA
```

导致的跨 SM 并行不足。

## 十七、Split-K 与 Stream-K 的区别

Split-K 使用固定切分数：

```text
每个输出 Tile -> S 个 K Slice
```

问题是：

- 所有 Tile 都被切同样多份。
- 可能产生过多 Partial。
- 最后一波仍可能不满。
- S 需要按 Shape 调优。

### 17.1 Stream-K

Stream-K 不固定“每个 Tile 几个 Split”，而是把整个 GEMM 的：

```text
MAC Mainloop Iterations
```

尽量均匀分配给固定数量的 Persistent CTA。

设所有输出 Tile 一共需要：

```text
total_k_iters
```

Stream-K 让每个 CTA 获得接近：

```text
total_k_iters / num_ctas
```

的工作。

一个 CTA 的工作可能：

- 从某个 Tile 中间开始。
- 完成该 Tile。
- 继续处理下一个完整 Tile。
- 在另一个 Tile 中间结束。

只有被多个 CTA 共同处理的 Tile 需要 Partial 归并。

### 17.2 核心差异

| 维度 | Split-K | Stream-K |
| --- | --- | --- |
| 切分单位 | 每个输出 Tile 的固定 K Slice | 全局 MAC Loop Iteration |
| CTA 数 | `output_tiles * S` | 通常接近指定 Persistent Grid |
| 负载均衡 | Shape 依赖 | 更均匀 |
| Partial 数 | 可能很多 | 只在 CTA 边界共享 Tile |
| 调度复杂度 | 较低 | 较高 |
| 主要参数 | Split 数 S | Persistent CTA 数/策略 |

Stream-K 论文的核心目标是缓解 Wave Quantization，并减少对大量 Tile 配置和 Split 启发式的依赖。

## 十八、Attention 中的 Split-KV

Attention Decode：

```text
O =
  softmax(Q K_cache^T * scale)
  V_cache
```

此时：

```text
Q_len 通常为 1
KV_len 可能很长
```

若每个请求/Head 只启动一个 CTA：

```text
CTA 数 ≈ batch * query_heads
```

Batch 较小时可能远少于 SM 数。

Flash-Decoding 增加：

```text
KV Sequence Split
```

并行度变为近似：

```text
batch * query_heads * num_kv_splits
```

这通常称为：

- Split-KV。
- Split-K Attention。
- Multi-CTA KV。
- Flash-Decoding。

### 18.1 它不是 GEMM Head-Dim Split-K

Attention 的 `K` 同时可能指：

- Key Tensor。
- GEMM Reduction Dimension。

Decode Split-KV 通常切的是：

```text
KV Sequence Length
```

不是 Q/K Head Dimension。

每个 Split 读取一段历史 Token 的 K/V。

## 十九、Split-KV 的 Softmax 正确归并

GEMM Partial 可以直接求和：

```text
C = sum_s P_s
```

Attention 不行：

```text
softmax([scores_0, scores_1])
```

不等于两个局部 Softmax 输出直接相加或平均。

### 19.1 每个 Split 的状态

第 `s` 个 KV Segment 的 Score：

```text
x_{s,j} = q k_{s,j}^T * scale
```

局部最大值：

```text
m_s = max_j x_{s,j}
```

局部分母：

```text
l_s =
  sum_j exp(x_{s,j} - m_s)
```

局部未归一化输出：

```text
o_s =
  sum_j exp(x_{s,j} - m_s) v_{s,j}
```

### 19.2 全局归并

全局最大值：

```text
m = max_s m_s
```

修正系数：

```text
alpha_s = exp(m_s - m)
```

全局分母：

```text
l =
  sum_s alpha_s * l_s
```

全局分子：

```text
o =
  sum_s alpha_s * o_s
```

最终输出：

```text
O = o / l
```

### 19.3 使用 LSE 表示

局部 LogSumExp：

```text
LSE_s = m_s + log(l_s)
```

全局：

```text
LSE =
  logsumexp_s(LSE_s)
```

若局部输出已经归一化：

```text
O_s = o_s / l_s
```

则：

```text
O =
  sum_s exp(LSE_s - LSE) * O_s
```

Flash-Decoding 就是保存：

```text
Partial Output + LSE
```

再执行第二阶段归并。

### 19.4 简化合并代码

```python
def merge_attention_splits(partial_out, partial_lse):
    # partial_out: [S, H, D]
    # partial_lse: [S, H]

    global_lse = torch.logsumexp(partial_lse, dim=0)

    weights = torch.exp(
        partial_lse - global_lse.unsqueeze(0)
    )

    output = torch.sum(
        weights.unsqueeze(-1) * partial_out,
        dim=0,
    )
    return output, global_lse
```

实际 Kernel 使用 FP32 统计量，并避免显式构造大 Weight Tensor。

## 二十、Paged KV、GQA 与 Blackwell

### 20.1 Paged KV

KV Segment 不一定对应连续物理地址。

每个 Split 需要：

```text
logical token range
-> logical page
-> block table
-> physical KV page
```

Split 边界最好与 Page/Tile 对齐，减少：

- 首尾 Mask。
- Page Table 查询。
- 非合并访问。

### 20.2 GQA

GQA：

```text
group_size =
  num_query_heads / num_kv_heads
```

同一 KV Head 服务多个 Query Head。

高效 Kernel 会让一个 CTA：

- 读取一次 K/V Tile。
- 为多个 Query Head 计算 Score/Output。

如果 Split-KV 调度只按 Query Head 独立读取 KV，会丢失 GQA 的复用收益。

因此 Split 数不能只看：

```text
KV_len / chunk_size
```

还要看：

- Query Head Group。
- Batch。
- KV Head。
- 每 CTA 处理的 Head 数。
- SM 数。

### 20.3 Blackwell

Split-K 思想与架构无关，但 Blackwell 改变了取舍。

Blackwell Kernel 可使用：

- TMA。
- TMEM。
- UMMA/tcgen05。
- CTA Cluster。
- Persistent Scheduler。

大 Tile 更高效，却会减少 M/N Output Tile 数，因此在部分 Shape 上更需要额外并行维度。

归并可以选择：

- Global Workspace。
- Separate Reduction Kernel。
- Cluster Remote Shared Memory。

TensorRT-LLM Blackwell FMHA 的 Multi-CTA KV 就包含 Global Reduction 和 CGA Shared Memory Reduction 路径。

但不能仅因 GPU 是 Blackwell 就固定增加 Split 数。Cluster Occupancy、TMEM/SMEM 和 Reduction 成本仍需实际调优。

## 二十一、数值稳定性与确定性

### 21.1 GEMM 浮点顺序

数学上：

```text
(a + b) + c = a + (b + c)
```

浮点中不一定成立。

Split-K 改变：

- K 分组。
- Partial 内累加顺序。
- Partial 间归并顺序。

所以结果可能与非 Split-K 有微小差异。

### 21.2 Parallel Reduction

若 Reduction 顺序固定，通常可以做到同配置重复运行一致，但不保证与普通 GEMM Bitwise 相同。

### 21.3 Serial Reduction

Semaphore 固定 Split 顺序：

```text
0 -> 1 -> ... -> S-1
```

确定性通常更容易保证。

### 21.4 Atomic

Atomic 到达顺序依赖 CTA 调度，通常不保证 Bitwise Deterministic。

### 21.5 Attention

Split-KV 必须：

- Max/Sum 使用 FP32。
- 先做局部 Max Subtraction。
- 用 `exp(m_s - m)` 重标定。
- 正确处理全 Mask Segment。
- 避免 `-inf - -inf` 产生 NaN。

不能直接累加局部归一化输出。

## 二十二、何时使用，何时不要使用

### 适合 Split-K

- M/N 小，K 大。
- Output Tile 数少于 SM 并发容量。
- Wave Quantization 严重。
- LLM Decode 小 Batch Linear。
- LoRA 小 N 或小 M GEMM。
- MoE 中单 Expert Token 很少但 K 很大。
- Long-context Decode Attention。

### 不适合 Split-K

- M/N 已产生多波 CTA。
- K 很小。
- 每个 K Slice 只有少量 Mainloop Iteration。
- Parallel Workspace 过大。
- Reduction 已占主要时间。
- 强确定性要求但只能用 Atomic。
- Epilogue 很复杂且难以延后。
- GEMV 极度 Memory-bound，额外 Partial 流量超过并行收益。

### 22.1 Split-K 不是默认越大越好

典型性能曲线：

```text
S = 1:
  并行不足

S 适中:
  SM 利用率提高
  性能最佳

S 过大:
  K Slice 太短
  Workspace/Reduction/Launch 主导
  性能下降
```

应该把 `S` 视为 Kernel 配置的一部分，而不是模型常量。

## 二十三、Profile 与调优清单

### 23.1 Nsight Systems

检查：

- Partial GEMM 与 Reduction 是否为两次 Kernel。
- Reduction Kernel 占总时间比例。
- 是否存在 Workspace 清零。
- CTA Wave 是否更完整。
- Serial Split 是否出现明显等待。
- CUDA Graph 是否包含 Workspace 地址变化。

### 23.2 Nsight Compute

Partial GEMM：

```text
SM Active
Tensor Core Utilization
Eligible Warps
Achieved Occupancy
DRAM Throughput
L2 Hit Rate
Mainloop Iterations
```

Reduction：

```text
DRAM Read/Write
Atomic Throughput
Memory Coalescing
FP32 Pipe
Kernel Launch Duration
```

### 23.3 必须比较的候选

```text
S = 1, 2, 4, 8, 16, ...
```

同时比较：

- 不同 BLOCK_M/N/K。
- Parallel vs Serial。
- Split-K vs Smaller Output Tile。
- Split-K vs Stream-K/Persistent。
- FP16/BF16/FP8。
- 不同 M/N/K。

### 23.4 记录完整成本

不要只测 Partial GEMM：

```text
total =
  workspace_prepare
  + partial_gemm
  + reduction
  + epilogue
```

Attention 还要包括：

```text
block table
partial stats
LSE merge
output write
```

## 二十四、参考资料

1. [NVIDIA CUTLASS：Efficient GEMM in CUDA - Parallelized Reductions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#parallelized-reductions)
2. [NVIDIA CUTLASS GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)
3. [NVIDIA Matrix Multiplication Background Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
4. [Stream-K: Work-centric Parallel Decomposition for Dense GEMM](https://arxiv.org/abs/2301.03598)
5. [CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
6. [Flash-Decoding for Long-Context Inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
7. [FlashInfer：Attention States and Recursive Attention](https://docs.flashinfer.ai/tutorials/recursive_attention.html)
8. [cuBLASLt Reduction Scheme](https://docs.nvidia.com/cuda/nvmath-python/0.3.0/bindings/generated/nvmath.bindings.cublasLt.ReductionScheme.html)

## 二十五、总结

GEMM Split-K 的主线：

```text
普通 GEMM:
  一个输出 Tile由一个 CTA 扫描完整 K

Split-K:
  多个 CTA 扫描不同 K Slice
  -> 产生 Partial
  -> 归并
  -> 最后执行 Epilogue
```

它的收益来自：

```text
增加 CTA 数
减少 SM 空闲
缓解 Wave Quantization
```

它的代价是：

```text
Workspace
Partial HBM 流量
Reduction
同步
额外 Kernel Launch
浮点顺序变化
```

三种主要实现：

```text
Parallel Split-K:
  S 份 Workspace + 独立 Reduction

Serial Split-K:
  单份 Running Partial + Semaphore 顺序提交

Atomic Split-K:
  直接 Atomic 到输出，限制最多
```

与相关技术的边界：

```text
Sliced-K:
  CTA 内 Warp 切 K

Stream-K:
  全局均分 MAC Iteration，减少固定 Split 的量化问题

Attention Split-KV:
  切 KV Sequence，并用 Max/Sum/LSE 规则归并 Softmax 状态
```

最终判断标准不是“是否使用 Split-K”，而是：

```text
增加的并行收益
是否大于
归并、Workspace 和 Pipeline 变短的代价
```
