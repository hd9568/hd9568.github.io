---
title: 'Grouped GEMM 实战：cuBLAS 使用、CUTLASS 调度与 DeepGEMM MoE Kernel'
description: '先讲清 cublasGemmGroupedBatchedEx 的正确用法，再沿 CUTLASS、Triton 与 DeepGEMM 开源源码分析 Persistent CTA、Problem Visitor、MoE contiguous/masked layout，以及 Hopper WGMMA 和 Blackwell UMMA/TMEM 实现。'
category: 'Research & Work'
pubDate: '2026-07-29T13:24:00+08:00'
updatedDate: '2026-07-30T17:13:23+08:00'
---

## 目录

1. [本文解决什么问题](#一本篇解决什么问题)
2. [Grouped GEMM 在 MoE 中计算什么](#二grouped-gemm-在-moe-中计算什么)
3. [cuBLAS Grouped GEMM：只讲公开接口](#三cublas-grouped-gemm只讲公开接口)
4. [LMDeploy 如何调用 cuBLAS](#四lmdeploy-如何调用-cublas)
5. [公开实现应该看哪些](#五公开实现应该看哪些)
6. [CUTLASS：通用异形 Grouped GEMM](#六cutlass通用异形-grouped-gemm)
7. [Triton 官方教程：最小 Persistent 实现](#七triton-官方教程最小-persistent-实现)
8. [DeepGEMM：面向 MoE 的高性能实现](#八deepgemm面向-moe-的高性能实现)
9. [Hopper Kernel：TMA、Warp Specialization 与 WGMMA](#九hopper-kerneltmawarp-specialization-与-wgmma)
10. [Blackwell Kernel：UMMA、TMEM 与 2-CTA](#十blackwell-kernelummatmem-与-2-cta)
11. [TensorRT-LLM 如何使用 CUTLASS Grouped GEMM](#十一tensorrt-llm-如何使用-cutlass-grouped-gemm)
12. [实现与选型对比](#十二实现与选型对比)
13. [参考资料](#十三参考资料)
14. [总结](#十四总结)

## 一、本篇解决什么问题

`cublasGemmGroupedBatchedEx` 的实现闭源。对它能可靠讲清楚的是：

- Group、Problem 和参数数组的语义。
- Host Metadata 与 Device Pointer Array 的位置。
- 数据类型、矩阵布局和输出约束。
- LMDeploy 如何把 MoE Expert GEMM 映射到该 API。

不能从公开资料确认：

- cuBLAS 当前版本采用什么 Tile Shape。
- Hopper/Blackwell 路径是否使用某条确定的 WGMMA/UMMA Pipeline。
- CTA 如何分配 Problem，是否 Persistent。
- TMA、Shared Memory、TMEM 和寄存器如何分工。

所以本文不再推测 cuBLAS 内部，而是把实现分析放到三个可审计项目：

```text
CUTLASS:
  NVIDIA 官方 C++ 模板库
  支持任意 (M_i, N_i, K_i)
  重点看 ProblemVisitor 和 Persistent CTA

Triton:
  官方 Group GEMM 教程
  代码短，适合看清全局 Tile 调度

DeepGEMM:
  DeepSeek 官方高性能库
  针对 MoE 的固定 N/K、变长 M
  支持 Hopper SM90 与 Blackwell SM100
```

结论先给出：

> Grouped GEMM 的核心不在单个 GEMM Tile，而在如何用固定数量的 CTA 持续处理多个不等长 Problem，并让调度、数据布局与硬件流水的额外成本小于合并执行带来的收益。

## 二、Grouped GEMM 在 MoE 中计算什么

### 2.1 MoE 产生一组变长 GEMM

Router 将 Token 分给 Expert 后，通常先按 Expert 排序：

```text
原始 token:
  [t0, t1, t2, t3, t4, t5]

expert id:
  [ 2,  0,  2,  1,  0,  2]

排序后:
  Expert 0: [t1, t4]      M_0 = 2
  Expert 1: [t3]          M_1 = 1
  Expert 2: [t0, t2, t5]  M_2 = 3
```

每个 Expert 计算：

```text
Y_e[M_e, N] = X_e[M_e, K] W_e[K, N]
```

其中：

- `M_e` 是路由到 Expert `e` 的 Token 数，运行时变化。
- `K` 是输入 Hidden Size。
- `N` 是 Expert 输出维度。
- 同一 MoE Layer 的 Expert 通常共享 `K/N`，只有 `M_e` 不同。

### 2.2 为什么不能直接 Batched GEMM

普通 Strided Batched GEMM 要求所有 Problem Shape 相同：

```text
[M, K] @ [K, N], repeated G times
```

MoE 实际是：

```text
[M_0, K] @ [K, N]
[M_1, K] @ [K, N]
...
[M_{G-1}, K] @ [K, N]
```

把所有 `M_e` Padding 到 `M_max` 会浪费：

```text
wasted rows = G * M_max - sum_e M_e
```

例如：

```text
M = [1, 1, 2, 4, 8, 32, 64, 128]
M_max = 128

padded rows = 8 * 128 = 1024
valid rows  = 240
waste       = 76.6%
```

逐 Expert 调用 GEMM 又会产生：

- 多次 Kernel Launch。
- 小 GEMM 无法填满所有 SM。
- Expert 间无法共同均衡 Tile。

Grouped GEMM 用一次 Launch 处理所有有效 Problem。

## 三、cuBLAS Grouped GEMM：只讲公开接口

### 3.1 Group 与 Problem

`cublasGemmGroupedBatchedEx` 有两层组织：

```text
Group:
  一组共享 M/N/K、Transpose、Leading Dimension、Alpha/Beta 的 GEMM

Problem:
  一组具体 A/B/C 指针
```

对 Group `g`：

```text
C_i =
    alpha[g] * op(A_i) * op(B_i)
    + beta[g] * C_i
```

同一 Group 内有 `group_size[g]` 个 Problem，总 Problem 数为：

```text
problem_count = sum_g group_size[g]
```

### 3.2 哪些数组在 Host，哪些在 Device

Host Metadata，长度为 `group_count`：

```text
transa_array, transb_array
m_array, n_array, k_array
alpha_array, beta_array
lda_array, ldb_array, ldc_array
group_size
```

Device Pointer Array，长度为 `problem_count`：

```text
Aarray
Barray
Carray
```

这里容易犯错：

```text
Aarray 本身必须位于 GPU
Aarray[i] 也必须是 GPU 矩阵地址
```

不能把 `std::vector<void*>.data()` 直接作为 `Aarray` 传入。

### 3.3 调用与关键注意事项

```cpp
cublasGemmGroupedBatchedEx(
    handle,
    trans_a, trans_b, m, n, k, alpha,
    d_A_ptrs, type_a, lda,
    d_B_ptrs, type_b, ldb,
    beta,
    d_C_ptrs, type_c, ldc,
    group_count, group_size, compute_type);
```

| 项目 | 要求 |
|---|---|
| 同一 Group | Shape、Transpose、LD、Alpha/Beta 相同 |
| 不同 Group | 可以使用不同参数 |
| `A/B/Carray` | 指针数组位于 Device |
| Metadata 数组 | 位于 Host |
| 输出 | 不同 Problem 的 C 区域不能重叠 |
| 输入 | A/B 可以只读共享 |
| `alpha/beta` | 类型由输入、输出与 Compute Type 组合决定 |
| Workspace | 可通过 Handle 预绑定，避免反复分配 |
| 性能 | 必须与 Batched、单 GEMM 循环和开源专用 Kernel 实测 |

真实代码还必须绑定正确 Stream、保证 Device Pointer Array 生命周期覆盖调用，并检查返回状态。

### 3.4 API 的边界

该 API 适合：

- Shape 不同、指针离散。
- 希望由 cuBLAS 自动选核。
- 不想维护硬件专用 Kernel。

它不直接提供：

- MoE Contiguous Offset 接口。
- Device-only `M_e` Metadata。
- SwiGLU、Bias、Quantization 等自定义 Epilogue。
- 显式 Tile Scheduler 控制。

这些正是 CUTLASS 和 DeepGEMM 值得研究的部分。

## 四、LMDeploy 如何调用 cuBLAS

### 4.1 Row-major 到 Column-major 的映射

LMDeploy 应用侧计算：

```text
Y_e[M_e, N] = X_e[M_e, K] W_e[K, N]
```

Legacy cuBLAS 按 Column-major 解释 Buffer。利用：

```text
Y_e^T[N, M_e] = W_e^T[N, K] X_e^T[K, M_e]
```

Row-major Buffer 与转置后的 Column-major View 字节一致，所以不需要显式转置：

```text
cuBLAS A -> W_e buffer
cuBLAS B -> X_e buffer
cuBLAS C -> Y_e buffer

m = N
n = M_e
k = K

lda = N
ldb = K
ldc = N
```

### 4.2 每个 Active Expert 是一个 Group

LMDeploy 中不同 Expert 的 `M_e` 通常不同，所以构造：

```text
group_count = active_expert_count
group_size  = [1, 1, ..., 1]
```

虽然每组只有一个 Problem，它们仍由一次 Grouped API 提交。

### 4.3 关键源码

调用可缩写为：

```cpp
const int active_count = n_active.size();
std::vector<int> m_array(active_count, N);
std::vector<int> k_array(active_count, K);
std::vector<int> group_size(active_count, 1);

// workspace.tensormaps:
// [weight ptrs][input ptrs][output ptrs]
char* d_ptrs = static_cast<char*>(workspace.tensormaps);

cudaMemcpyAsync(
    d_ptrs,
    weight_ptrs.data(),
    active_count * sizeof(void*),
    cudaMemcpyHostToDevice,
    stream);

// 另外两段 Input/Output Pointer 同样复制。

cublasGemmGroupedBatchedEx(
    handle,
    transa.data(), transb.data(),
    m_array.data(),
    n_active.data(),
    k_array.data(),
    alpha.data(),
    d_weight_ptrs, cuda_type, lda.data(),
    d_input_ptrs,  cuda_type, ldb.data(),
    beta.data(),
    d_output_ptrs, cuda_type, ldc.data(),
    active_count,
    group_size.data(),
    CUBLAS_COMPUTE_32F);
```

### 4.4 这条路径的实际成本

LMDeploy 要从 Expert Offset 得到 `M_e`：

```text
M_e = offsets[e + 1] - offsets[e]
```

当前 Wrapper 可能需要：

```text
Device offsets
  -> D2H copy
  -> Stream synchronize
  -> Host filter empty experts
  -> build A/B/C pointer vectors
  -> H2D copy pointer arrays
  -> cuBLAS launch
```

这部分比猜测 cuBLAS 内部 Tile 更值得关注。Decode 时 `M_e` 很小，Host Metadata 和同步可能占显著比例。

更理想的 MoE 路径应让：

- Expert Count 保留在 Device。
- Group Layout 由 Router/Permute Kernel 直接生成。
- Grouped GEMM 直接消费 Device Metadata。
- CUDA Graph 不依赖 Host 读取动态 `M_e`。

DeepGEMM 的 Masked Layout 就是为此设计。

## 五、公开实现应该看哪些

| 实现 | 平台 | Problem 形态 | 主要价值 |
|---|---|---|---|
| CUTLASS Grouped GEMM | NVIDIA SM80+ | 任意 `M_i/N_i/K_i` | 通用、源码完整 |
| Triton Group GEMM Tutorial | CUDA，含 TMA 路径 | 任意指针与 Shape | 调度代码最易读 |
| PyTorch Triton Persistent GMM | H100 | MoE 变长 M | Cache-aware 调度 |
| DeepGEMM | SM90/SM100 | 固定 N/K，变长 M | FP8/FP4/BF16、MoE 专用 |
| TensorRT-LLM CUTLASS MoE | NVIDIA | 固定 N/K，变长 M | 生产框架接入示例 |
| Composable Kernel | AMD CDNA/RDNA | Grouped GEMM | AMD XDL/WMMA 路线 |

后文重点分析 CUTLASS 与 DeepGEMM，因为二者分别代表：

```text
通用异形调度
vs.
MoE 专用硬件优化
```

## 六、CUTLASS：通用异形 Grouped GEMM

### 6.1 Device 接口

CUTLASS 示例使用：

```cpp
using GemmKernel =
    typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA, LayoutA, ...,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        ThreadblockSwizzle,
        Stages,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly
    >::GemmKernel;

using Gemm =
    cutlass::gemm::device::GemmGrouped<GemmKernel>;
```

运行参数包括：

```text
problem_sizes_device
problem_count
threadblock_count
epilogue_op
A/B/C/D pointer arrays
lda/ldb/ldc/ldd arrays
host_problem_sizes
```

### 6.2 为什么是 Persistent Kernel

普通 GEMM：

```text
Grid size = 当前 GEMM 的总 Tile 数
每个 CTA 通常处理一个 Tile 后退出
```

CUTLASS Grouped GEMM：

```text
Grid size =
  min(总 Tile 数, SM 数 * 每 SM 最大 Resident CTA)

每个 CTA:
  处理 Tile
  获取下一个全局 Tile
  继续处理
  直到所有 Problem 完成
```

`BaseGrouped::sufficient()` 的核心逻辑：

```cpp
int resident_blocks =
    available_sm_count * maximum_active_blocks();

int total_tiles =
    group_tile_count(problem_sizes, problem_count);

return std::min(total_tiles, resident_blocks);
```

这样只启动一波或有限数量的 Persistent CTA，避免最后一波只有少量 CTA 的 Wave Quantization。

### 6.3 把所有 Problem 展平成全局 Tile 空间

对 Problem `g`：

```text
tiles_m[g] = ceil(M_g / BLOCK_M)
tiles_n[g] = ceil(N_g / BLOCK_N)
tiles[g]   = tiles_m[g] * tiles_n[g]
```

全局 Tile 空间：

```text
Problem 0: [0, tiles[0])
Problem 1: [tiles[0], tiles[0] + tiles[1])
...
```

CTA 初始：

```text
tile_idx = blockIdx.x
```

处理完后：

```text
tile_idx += gridDim.x
```

因此 CTA `b` 处理：

```text
b, b + gridDim.x, b + 2*gridDim.x, ...
```

只要能把 `tile_idx` 映射回 `(problem_idx, tile_m, tile_n)`，就能在一条 Persistent Loop 里处理全部 GEMM。

### 6.4 `ProblemVisitor` 的 Device-only 调度

直接逐 Problem 扫描会让每个 CTA 做很多串行 Metadata 读取。CUTLASS 的 `kDeviceOnly` Visitor 每次让一个 Warp 并行处理 32 个 Problem：

```text
lane 0 读取 problem[p + 0]
lane 1 读取 problem[p + 1]
...
lane 31 读取 problem[p + 31]
```

每个 Lane 计算该 Problem 的 Tile 数，然后做 Warp Inclusive Prefix Sum：

```cpp
for (int offset = 1; offset < 32; offset <<= 1) {
    int value = __shfl_up_sync(
        0xffffffff, problem_ending_tile, offset);

    if (lane_id >= offset) {
        problem_ending_tile += value;
    }
}
```

随后：

```cpp
int problem_in_warp = __popc(__ballot_sync(
    0xffffffff,
    problem_ending_tile <= tile_idx));
```

得到当前全局 Tile 属于 Warp 中哪个 Problem。起始 Tile 为前一个 Lane 的 Prefix Sum。

这段实现解决的是：

```text
global tile id
  -> problem id
  -> tile id inside problem
```

而不是 GEMM 数学本身。

### 6.5 Host-precompute 调度

另一种 `kHostPrecompute` 在 CPU 预先生成：

```text
(problem_idx, problem_start_tile)
```

每个 CTA 都有一条自己的访问序列，初始化时复制到 Workspace。Device 只需顺序读取映射，不再扫描 Problem Size。

适合：

- Problem Metadata 本来就在 Host。
- Shape 在多次调用间稳定。
- Host 预计算可与上一层 GPU 计算重叠。
- GEMM 很小，Device Scheduler 占比高。

不适合：

- `M_e` 由 GPU Router 动态产生。
- 每步都重建并复制 Schedule。
- CUDA Graph 要求 Device-only 动态形状。

### 6.6 Kernel 主循环

`gemm_grouped.h` 中的核心结构为：

```cpp
ProblemVisitor visitor(
    params.problem_visitor,
    shared_storage.problem_visitor,
    blockIdx.x);

while (visitor.next_tile()) {
    auto problem = visitor.problem_size();
    int problem_idx = visitor.problem_index();
    int tile_in_problem = visitor.threadblock_idx();

    auto grid = visitor.grid_shape(problem);

    int tile_m = tile_in_problem / grid.n();
    int tile_n = tile_in_problem % grid.n();

    auto ptr_A = params.ptr_A[problem_idx];
    auto ptr_B = params.ptr_B[problem_idx];
    auto ptr_C = params.ptr_C[problem_idx];
    auto ptr_D = params.ptr_D[problem_idx];

    // 为当前 Problem 创建 Iterator。
    Mma mma(...);
    mma(gemm_k_iterations, accum, iter_A, iter_B, accum);

    Epilogue epilogue(...);
    epilogue(output_op, iter_D, accum, iter_C);

    visitor.advance(gridDim.x);
}
```

同一个 CTA 会在不同 Problem 间切换，因此每轮都要重新读取：

- Problem Size。
- A/B/C/D Pointer。
- Leading Dimension。
- K-loop 次数。

Grouped GEMM 的调度开销由此产生。

### 6.7 为什么按 K 降序排序可能更快

调度器尽量让每个 CTA 得到相同 Tile 数，但不同 Problem 的 `K_i` 不同：

```text
Tile A: 128 x 128 x 128
Tile B: 128 x 128 x 1024
```

二者数量相同，计算时间却相差约 8 倍。若某些 CTA 连续拿到大 K Tile，会形成长尾。

CUTLASS 提供 `sort_problems()`，按 `K` 降序重排 Problem 与全部关联 Metadata，使大 K Tile 更均匀地轮转到各 CTA。

官方示例中的特定 Case 约有 30% 改善，但排序不是普遍最优，必须实测。

## 七、Triton 官方教程：最小 Persistent 实现

### 7.1 核心只有两层循环

Triton 官方 Group GEMM 教程把 CUTLASS 的思想写得更直接：

```python
@triton.jit
def grouped_matmul_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    group_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    problem_start = 0

    for g in range(group_size):
        M = tl.load(group_gemm_sizes + g * 3 + 0)
        N = tl.load(group_gemm_sizes + g * 3 + 1)
        K = tl.load(group_gemm_sizes + g * 3 + 2)

        num_m_tiles = tl.cdiv(M, BLOCK_M)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        while (
            tile_idx >= problem_start
            and tile_idx < problem_start + num_tiles
        ):
            tile_in_problem = tile_idx - problem_start
            tile_m = tile_in_problem // num_n_tiles
            tile_n = tile_in_problem % num_n_tiles

            # 普通 Tiled GEMM。
            ...

            tile_idx += NUM_SM

        problem_start += num_tiles
```

Host 只启动固定数量 Program：

```python
grid = lambda meta: (meta["NUM_SM"],)
```

这就是最小的 Static Persistent Grouped GEMM。

### 7.2 与 CUTLASS 的差异

Triton 教程按 `g=0..G-1` 顺序扫描，容易理解，但每个 Program 都重复读取 Problem Metadata。

CUTLASS 进一步提供：

- Warp 并行扫描 32 个 Problem。
- Host-precomputed Schedule。
- 通用 C++ Epilogue 与多种 MMA Pipeline。
- Problem 排序工具。

Triton 的优势是更容易修改：

- Grouped Launch Ordering。
- L2 Cache-aware Tile 顺序。
- Fused Bias/SwiGLU。
- 特定 MoE Layout。

### 7.3 TMA 版本

官方教程在 Hopper 及更新架构上还提供 Tensor Descriptor/TMA 版本：

```text
Pointer + Stride
  -> Tensor Descriptor
  -> TMA load A/B Tile
  -> tl.dot
  -> Tensor Descriptor store C
```

TMA 降低地址计算与搬运指令开销，但每个 Problem 构造/读取 Descriptor 仍有成本。对大量极小 GEMM，调度和 Descriptor 开销可能超过 Tensor Core 计算。

## 八、DeepGEMM：面向 MoE 的高性能实现

### 8.1 为什么不做完全通用的 Shape

DeepGEMM 的 M-grouped 接口约束：

```text
M_e: 每个 Expert 不同
N:   所有 Expert 相同
K:   所有 Expert 相同
```

这与 MoE 完全匹配，因此可以：

- 为所有 Expert 复用同一 Tile Shape。
- 为 B/Weight 使用一个 3D TMA Descriptor。
- 避免通用 Pointer Array 与任意 LD。
- JIT 专门化固定 `N/K`。

它不是 cuBLAS/CUTLASS 通用接口的完整替代。

### 8.2 两种 M-grouped Layout

#### Contiguous Layout：训练与 Prefill

```text
A: [M_total, K]
B: [G, N, K]
D: [M_total, N]
```

默认 `grouped_layout` 是长度 `M_total` 的 Device `int32`：

```text
grouped_layout[row] = expert_id
```

为保证一个 CTA 的 `BLOCK_M` 不跨 Expert，Expert 段需要对齐。不参与计算的 Padding Row 标成 `-1`：

```text
Expert 0 rows: [0, 128)       layout = 0
Padding:       [128, 256)     layout = -1
Expert 1 rows: [256, 384)     layout = 1
```

Kernel 用 Tile 首行查 Expert：

```cpp
expert_id = grouped_layout[m_block_idx * BLOCK_M];
```

再定位：

```text
A offset -> 连续 Token Row
B offset -> expert_id * N * K
D offset -> 连续输出 Row
```

Padding Tile 通过 `expert_id < 0` 跳过。

当前版本还支持 Prefix-sum Layout，只存每个 Group 的结束 Offset，以减少 Layout Tensor 大小。

#### Masked Layout：CUDA Graph Decode

```text
A: [G, M_max, K]
B: [G, N, K]
D: [G, M_max, N]
masked_m: [G]
```

`masked_m[e]` 是 Device 上的真实 Token 数。Kernel 只计算：

```text
row < masked_m[e]
```

Host 不需要知道本轮 `M_e`，固定地址和最大 Shape 可以被 CUDA Graph 捕获。

`expected_m` 只用于 JIT Heuristic 选择 Tile，不替代 Device 上的真实 `masked_m`。

### 8.3 Persistent Scheduler

DeepGEMM 与 CUTLASS/Triton 使用相同的全局轮转：

```cpp
next_block =
    (++current_iter) * kNumSMs + blockIdx.x;
```

每个 CTA 依次处理：

```text
blockIdx.x
blockIdx.x + num_sms
blockIdx.x + 2 * num_sms
...
```

Masked Layout 中，Scheduler 在 Device 扫描各 Expert 的：

```text
ceil(masked_m[e] / BLOCK_M) * num_n_blocks
```

找到 `next_block` 对应的 Expert 与 `(m_block, n_block)`。

Contiguous Layout 已把所有 Token Row 拼成一个 M 轴，Scheduler 直接在 Flattened M 上取 Tile，再通过 `grouped_layout` 得到 Expert。

### 8.4 JIT 与 Heuristic

DeepGEMM 不是预编译一个万能 Kernel。调用路径：

```text
Python API
  -> C++ shape/type check
  -> build GemmDesc
  -> enumerate candidate Tile/Cluster/Pipeline
  -> heuristic choose config
  -> generate CUDA specialization
  -> JIT compile and cache
  -> launch
```

Heuristic 考虑：

- `BLOCK_M/N/K`。
- Shared Memory Stage 数。
- SM 数与 Wave 数。
- Last-wave Utilization。
- TMA Multicast/Cluster。
- L1/L2 数据移动估算。
- 输出类型与 Scale Layout。

这样避免 CUTLASS 大量模板实例的静态编译成本，同时保留 Shape Specialization。

## 九、Hopper Kernel：TMA、Warp Specialization 与 WGMMA

DeepGEMM SM90 FP8 Kernel 的线程角色明确分开。

### 9.1 Warp-group 分工

```text
TMA Warp-group:
  约 40 registers/thread
  发起 A/B/Scale 的 TMA

Math Warp-group:
  约 232/248 registers/thread
  发起 WGMMA
  FP32 Promotion
  写回 Epilogue
```

代码使用：

```cpp
cutlass::arch::warpgroup_reg_dealloc<40>();
cutlass::arch::warpgroup_reg_alloc<248>();
```

让搬运线程少占寄存器，把更多寄存器给 Accumulator。

### 9.2 多 Stage TMA Pipeline

每个 Stage 在 Shared Memory 中保存：

```text
A Tile
B Tile
A Scale
Transaction Barrier
```

Producer：

```cpp
empty_barrier.wait(old_phase);
tma_load(A, B, scale);
full_barrier.arrive_and_expect_tx(bytes);
```

Consumer：

```cpp
full_barrier.wait(phase);
wgmma(...);
empty_barrier.arrive();
```

多个 Stage 让：

```text
Stage s     执行 WGMMA
Stage s+1   TMA 搬运
```

并行重叠。

### 9.3 Fine-grained FP8 Scaling

DeepSeek FP8 使用细粒度 Scale。SM90 Tensor Core 的低精度累加需要修正，因此 Kernel 对 K Block：

```text
WGMMA partial
  -> 读取 A/B Scale
  -> CUDA Core FP32 promotion
  -> 累加到 final_accum
```

简化公式：

```text
C =
  sum_kblock (
      scale_A[kblock]
      * scale_B[kblock]
      * WGMMA_FP8(A_block, B_block)
  )
```

这也是 DeepGEMM 能保持 DeepSeek-V3 FP8 数值语义的关键，不只是“调用 WGMMA”。

### 9.4 TMA Multicast 与 L2 复用

相邻 CTA 若共享 A 或 B Tile，可组成 2-CTA Cluster：

```text
一个 TMA Load
  -> Multicast 到两个 CTA 的 Shared Memory
```

Scheduler 做 M/N Swizzle，使共享同一 Operand 的 Tile 尽量相邻，从而：

- 减少 HBM/L2 读取。
- 提高 Weight/Activation 复用。
- 避免简单 Row-major Tile 顺序破坏 Cache。

## 十、Blackwell Kernel：UMMA、TMEM 与 2-CTA

### 10.1 Accumulator 进入 TMEM

SM100 的 `tcgen05/UMMA` 将 Accumulator 放在 Tensor Memory：

```text
TMA -> Shared Memory A/B
Scale -> Shared Memory -> TMEM
UMMA -> TMEM Accumulator
TMEM -> Shared Memory
TMA Store -> Global Memory
```

不再要求大量普通线程寄存器长期持有 Accumulator。

### 10.2 线程角色

DeepGEMM SM100 Kernel 进一步拆分：

```text
Warp 0:
  TMA Load

Warp 1:
  tcgen05/UMMA Issue

Epilogue Threads:
  TMEM -> SMEM
  TMA Store C/D
```

用 Barrier 连接三个流水：

```text
TMA full/empty
TMEM full/empty
Epilogue store stages
```

### 10.3 2-CTA MMA

当 `kNumMulticast=2` 时：

- 两个 CTA 组成 Cluster。
- `Allocator2Sm` 分配跨两 SM 的 TMEM。
- `tcgen05` 2-CTA 指令协同完成更大的 MMA。
- TMA Multicast 向两个 CTA 供数。

但 2-CTA 需要 Tile 数、Cluster 和 Shape 对齐。Heuristic 只有在：

```text
N Tile 数可配对
SM 数可被 2 整除
TMEM/Shared Memory 合法
```

时才启用。

### 10.4 M-grouped 的 `swap_ab`

SM100 Heuristic 对 M-grouped 固定：

```text
swap_ab = true
BLOCK_N = 128
BLOCK_M = M alignment
```

这样变长 `M_e` 被放到 UMMA 的动态 N 方向。对尾部 Tile，Kernel 可更新 Instruction Descriptor 中的 `UMMA_N`：

```cpp
uint32_t effective_m =
    scheduler.get_aligned_effective_m_in_block(m_block);

update_instr_desc_with_umma_n(
    instr_desc, effective_m);
```

相比固定完整 Tile 后 Mask，这能减少变长 Expert 尾部的无效计算。

### 10.5 Block-scaled FP8/FP4

SM100 Kernel 同时搬运：

```text
A/B data
SFA/SFB scale
```

Scale 以 UE8M0 Packed Layout 存储，经 UTCCP 从 Shared Memory 搬到 TMEM，再由 Block-scaled UMMA 消费。

因此 DeepGEMM 的 Blackwell 高性能依赖特定 Layout：

- 输入不只是 FP8/FP4 Tensor。
- Scale Tensor 也必须转换成硬件要求的布局。
- Layout 转换最好与上游 Quantize/Permute 融合，否则额外 Kernel 会抵消 GEMM 收益。

## 十一、TensorRT-LLM 如何使用 CUTLASS Grouped GEMM

TensorRT-LLM 提供了 CUTLASS 的生产接入示例。

### 11.1 通用 Grouped GEMM

`groupGemm.cu` 使用：

```cpp
using GemmKernel =
    cutlass::gemm::kernel::DefaultGemmGrouped<
        ...,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        Epilogue,
        Swizzle,
        Stages,
        GroupScheduleMode::kDeviceOnly
    >::GemmKernel;

using Gemm =
    cutlass::gemm::device::GemmGrouped<GemmKernel>;
```

它把 Problem Size、Pointer 与 LD 打包进 Device Workspace，再调用：

```cpp
int threadblock_count =
    Gemm::sufficient(problem_sizes.data(), problem_count);

Gemm::Arguments args(
    problem_sizes_device,
    problem_count,
    threadblock_count,
    epilogue_op,
    ptr_A, ptr_B, ptr_C, ptr_D,
    lda, ldb, ldc, ldd,
    problem_sizes.data());
```

### 11.2 MoE 专用 Visitor

TensorRT-LLM MoE 路径不再传任意 Problem Pointer，而是使用连续 Token Buffer 与 Expert 边界：

```text
A:
  sorted token matrix

B:
  expert weights

total_tokens_including_expert:
  Expert Prefix Sum / Boundary
```

同时固定 Persistent CTA 数：

```cpp
int occupancy =
    std::min(2, GemmGrouped::maximum_active_blocks());

int threadblock_count =
    sm_count * occupancy;
```

这与 DeepGEMM 的方向一致：

> 通用 Pointer-array API 适合异形 GEMM；MoE 最高性能通常需要让 Kernel 直接理解 Expert-contiguous Layout。

## 十二、实现与选型对比

### 12.1 功能对比

| 项目 | cuBLAS | CUTLASS | Triton Tutorial | DeepGEMM |
|---|---|---|---|---|
| 源码 | 闭源 | 开源 | 开源 | 开源 |
| Shape | 任意分组 | 任意 `M/N/K` | 任意 `M/N/K` | 固定 N/K、变长 M |
| Metadata | Host + Device Ptr Array | Device arrays/Host schedule | Device arrays | Device Group Layout |
| Scheduler | 不公开 | Device/Host 两种 | Device Static | MoE 专用 Persistent |
| Epilogue | API 支持范围内 | 模板可定制 | 直接改代码 | 库内专用 |
| Hopper | 支持 | 支持 | TMA 路径 | WGMMA 专用 |
| Blackwell | 支持 | 支持 | 可扩展 | UMMA/TMEM 专用 |
| CUDA Graph 动态 M | Host Metadata 不理想 | Device Visitor 可做 | 可做 | Masked Layout 原生支持 |

### 12.2 如何选择

| 条件 | 优先选择 |
|---|---|
| 稳定 API，不维护 Kernel | cuBLAS |
| 任意异形 Problem、自定义 Epilogue | CUTLASS |
| 快速修改调度或融合 | Triton |
| SM90/SM100、固定 N/K 的 FP8/FP4/BF16 MoE | DeepGEMM |

### 12.3 不存在脱离 Shape 的“最优实现”

Grouped GEMM 性能取决于 `M_e` 分布、`N/K`、Expert 数、精度、Scale Layout、SM 数、Epilogue、Metadata 生成位置以及是否与 Dispatch/Combine 融合。

所以正确说法是：

```text
某个实现在某组 Shape、硬件、Layout 和端到端流水中最优
```

而不是“某个库永远最快”。

性能比较必须同时计入 Layout/Scale 转换、Pointer Metadata、Router、Dispatch 和 Combine，不能只报告 GEMM Kernel 时间。

## 十三、参考资料

1. [cuBLAS `cublasGemmGroupedBatchedEx`](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex)
2. [NVIDIA Grouped GEMM API 介绍](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
3. [cuBLAS Grouped GEMM 官方样例](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLAS/Extensions/GemmGroupedBatchedEx)
4. [CUTLASS Grouped Kernel Scheduler 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)
5. [CUTLASS `24_gemm_grouped`](https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped)
6. [CUTLASS `grouped_problem_visitor.h`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/grouped_problem_visitor.h)
7. [CUTLASS `gemm_grouped.h`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/gemm_grouped.h)
8. [CUTLASS Blackwell Contiguous-offset Grouped GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/005_grouped_gemm_contiguous_offset.html)
9. [Triton Group GEMM 官方教程](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
10. [PyTorch Persistent Cache-aware Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)
11. [DeepSeek DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
12. [TensorRT-LLM Grouped GEMM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/groupGemm.cu)
13. [AMD Composable Kernel](https://github.com/ROCm/composable_kernel)

## 十四、总结

cuBLAS 部分只需记住：

```text
Host:
  每个 Group 的 Shape/LD/Scale

Device:
  每个 Problem 的 A/B/C Pointer

同一 Group:
  参数相同

不同输出:
  不能重叠
```

开源高性能实现的共同主线是：

```text
所有 Problem 展平为 Tile 空间
  -> 固定数量 Persistent CTA
  -> CTA 轮转领取 Tile
  -> 映射回 Problem/Expert
  -> 普通 Tiled GEMM Pipeline
```

CUTLASS 解决通用异形 Problem：

- Device-only Warp Prefix Scan。
- Host-precomputed Schedule。
- Persistent `ProblemVisitor`。
- 按 K 排序改善负载均衡。

DeepGEMM 进一步利用 MoE 约束：

- 固定 N/K，只让 M 变化。
- Contiguous Layout 用于训练与 Prefill。
- Masked Layout 避免 Decode 的 Host 动态 Metadata。
- Hopper 使用 TMA + WGMMA + FP32 Promotion。
- Blackwell 使用 TMA + UMMA + TMEM + 2-CTA。
- JIT 根据 Shape、Wave、Stage 和 Cluster 选择专用配置。

真正的优化方向不是继续猜 cuBLAS 内部，而是减少 Group Metadata、Layout 转换和 Expert 通信，并让 Grouped GEMM 与 Router、Dispatch、Activation、Combine 形成连续流水。
