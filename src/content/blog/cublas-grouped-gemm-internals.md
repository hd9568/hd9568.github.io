---
title: 'cuBLAS Grouped GEMM 深入解析：API、内部调度与 LMDeploy Blackwell 接入'
description: '结合 NVIDIA 官方文档、官方样例和 LMDeploy 源码，讲解 cublasGemmGroupedBatchedEx 的 Group/Problem 语义、行列主序映射、Host/Device 元数据、运行时选核、Warp MMA、SM100 接入与性能边界。'
category: 'Research & Work'
pubDate: '2026-07-29T13:24:00+08:00'
updatedDate: '2026-07-29T13:24:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [先说明公开资料的边界](#一先说明公开资料的边界)
2. [Grouped GEMM 的两层分组语义](#二grouped-gemm-的两层分组语义)
3. [官方 API 计算公式](#三官方-api-计算公式)
4. [Host Metadata 与 Device Pointer Array](#四host-metadata-与-device-pointer-array)
5. [官方已经公开了哪些内部信息](#五官方已经公开了哪些内部信息)
6. [高性能 Grouped Kernel 必须解决什么](#六高性能-grouped-kernel-必须解决什么)
7. [设备端 Tile 调度可以怎样实现](#七设备端-tile-调度可以怎样实现)
8. [LMDeploy 为什么只在 SM100 路径注册](#八lmdeploy-为什么只在-sm100-路径注册)
9. [LMDeploy 调用前如何整理 Expert](#九lmdeploy-调用前如何整理-expert)
10. [空 Expert 为什么被过滤](#十空-expert-为什么被过滤)
11. [Row-major 到 cuBLAS Column-major 的零拷贝映射](#十一row-major-到-cublas-column-major-的零拷贝映射)
12. [逐项解释 cublasGemmGroupedBatchedEx 调用](#十二逐项解释-cublasgemmgroupedbatchedex-调用)
13. [LMDeploy 为什么让每个 Expert 单独成组](#十三lmdeploy-为什么让每个-expert-单独成组)
14. [一个完整数字例子](#十四一个完整数字例子)
15. [BF16/FP16 输入与 FP32 计算](#十五bf16fp16-输入与-fp32-计算)
16. [Workspace 和 Stream 顺序](#十六workspace-和-stream-顺序)
17. [CPU 开销从哪里来](#十七cpu-开销从哪里来)
18. [Blackwell 上能确认什么，不能确认什么](#十八blackwell-上能确认什么不能确认什么)
19. [Grouped GEMM 为什么可能更快](#十九grouped-gemm-为什么可能更快)
20. [什么时候反而可能更慢](#二十什么时候反而可能更慢)
21. [如何正确 Profile](#二十一如何正确-profile)
22. [正确性与接入检查表](#二十二正确性与接入检查表)
23. [参考资料](#二十三参考资料)
24. [总结](#二十四总结)

## 一、先说明公开资料的边界

`cublasGemmGroupedBatchedEx` 是 cuBLAS 闭源库中的 API。

公开资料可以确认：

- API 参数和内存位置。
- Group 与 Problem 的组织方式。
- 支持的数据类型。
- 一次提交可以处理不同 Shape、转置和 Scale。
- NVIDIA 在 CUDA 12.5 发布时描述其为一次 Kernel Launch。
- CUDA 12.5 初版 Grouped Kernel 使用 Warp-level MMA。
- cuBLAS 有运行时 Recommender 选择 Kernel 和 Launch 参数。

公开资料不能确认：

- 当前 cuBLAS 版本中每个 Shape 对应的具体 Kernel 源码。
- Blackwell 上是否使用某个确定的 UMMA/TMA Pipeline。
- CTA 如何精确映射到 Group。
- Shared Memory Tile、Pipeline Stage 和寄存器分配。
- Recommender 对 Grouped GEMM 使用的全部特征。

因此本文将内容分成三类：

```text
官方事实：
  来自 cuBLAS 文档、CUDA 发布说明和 NVIDIA 技术博客。

LMDeploy 事实：
  来自调用侧源码，可以逐行确认。

实现推导：
  根据 Grouped GEMM 必须完成的工作和公开 CUTLASS 设计解释，
  但不冒充 cuBLAS 闭源 Kernel 的逐行源码。
```

这种边界很重要。看到一个库 API 调用，不能把常见 GEMM 模板的实现细节直接当作该闭源 Kernel 的事实。

## 二、Grouped GEMM 的两层分组语义

`cublasGemmGroupedBatchedEx` 中有两层概念：

```text
Group：
  一组使用相同 Shape、转置、Leading Dimension 和 Scale 的 GEMM。

Problem：
  一次具体的 A/B/C 指针组合。
```

对第 `g` 个 Group：

```text
Shape:
  m[g], n[g], k[g]

Transpose:
  transa[g], transb[g]

Leading Dimension:
  lda[g], ldb[g], ldc[g]

Scale:
  alpha[g], beta[g]

Problem 数:
  group_size[g]
```

总 Problem 数：

```text
problem_count =
    sum(group_size[g])
```

### 2.1 官方样例

NVIDIA 官方样例：

```cpp
group_count = 2;
group_size = {2, 1};
```

含义：

```text
Group 0：
  2 个相同 Shape/参数的 GEMM

Group 1：
  1 个另一种 Shape/参数的 GEMM

总 Problem 数：
  2 + 1 = 3
```

因此：

```text
m_array/n_array/... 长度 = group_count = 2
Aarray/Barray/Carray 长度 = problem_count = 3
```

### 2.2 与普通 Batched GEMM 的区别

普通 Batched GEMM：

```text
所有 Problem 共用一组：
M/N/K、Transpose、LD、Alpha/Beta
```

Grouped GEMM：

```text
不同 Group 可以使用不同：
M/N/K、Transpose、LD、Alpha/Beta
```

但同一 Group 内仍然是 Uniform Batch。

## 三、官方 API 计算公式

对属于 Group `g` 的第 `idx` 个 Problem：

```text
C[idx] =
    alpha[g]
    * op(A[idx])
    * op(B[idx])
    + beta[g]
    * C[idx]
```

形状：

```text
op(A[idx]): [m[g], k[g]]
op(B[idx]): [k[g], n[g]]
C[idx]:     [m[g], n[g]]
```

矩阵按 cuBLAS 的 Column-major 语义解释。

### 3.1 输出不能重叠

官方文档要求不同 `C[idx]` 不能互相重叠。

原因是 Grouped GEMM 中的 Problem 应当能独立执行：

```text
若两个 Problem 同时写同一地址
-> 写入顺序未定义
-> 结果未定义
```

输入 A/B 可以只读共享，例如多个 Problem 使用同一个权重；输出不能发生写冲突。

## 四、Host Metadata 与 Device Pointer Array

该 API 最容易混淆的部分是参数位于不同内存空间。

### 4.1 Host 数组

以下数组在 Host：

```text
transa_array
transb_array
m_array
n_array
k_array
alpha_array
beta_array
lda_array
ldb_array
ldc_array
group_size
```

它们的长度都是：

```text
group_count
```

### 4.2 Device Pointer Array

以下是位于 Device 的“指针数组”：

```text
Aarray
Barray
Carray
```

每个数组元素本身又是一个 Device Pointer：

```text
Aarray[idx] -> 第 idx 个矩阵 A 的 GPU 地址
```

数组长度：

```text
problem_count
```

注意区分：

```text
Aarray 本身在 GPU
Aarray[idx] 指向的矩阵也在 GPU
```

### 4.3 为什么这样设计

Host Metadata 便于库在 Launch 前：

- 检查参数。
- 选择 Kernel。
- 计算任务规模。
- 构造内部调度描述。

Device Pointer Array 允许最终 Kernel 直接读取各 Problem 的矩阵地址，而不需要把矩阵指针逐个作为 Kernel 参数传入。

### 4.4 Pointer Mode

官方文档明确说明该 API 不支持：

```text
CUBLAS_POINTER_MODE_DEVICE
```

Alpha/Beta 数组必须按 Host Scale 使用。LMDeploy 没有把 Handle 切换到 Device Pointer Mode，符合要求。

## 五、官方已经公开了哪些内部信息

### 5.1 一次 Kernel Launch

NVIDIA 对 CUDA 12.5 Grouped GEMM 的描述是：

```text
不同 Shape、转置和 Scale 的 GEMM
可以被组织并并行到一次 Kernel Launch。
```

这与“Host 循环调用多个 GEMM”有本质区别：

```text
Host loop:
  E 个 Expert -> E 次 cuBLAS Dispatch/Kernel Launch

Grouped API:
  E 个 Expert -> 1 次 Grouped Dispatch
```

### 5.2 初版使用 Warp-level MMA

NVIDIA 2024 年技术博客说明，当时的 Grouped GEMM Kernel 使用：

```text
Warp-level MMA
```

而对比的 Batched GEMM Kernel 可以使用 Hopper：

```text
Warpgroup-level WGMMA
```

即便指令粒度较小，Grouped Kernel 在部分 MoE Generation Shape 上仍达到约 `1.2x` 的性能提升。

这个结果说明：

> 减少 Launch 和改善不规则 Problem 的整体调度，可能比单个 GEMM 使用更大的 MMA 指令更重要。

### 5.3 运行时 Recommender

cuBLAS 一般使用运行时 Recommender，从多个候选中选择：

```text
Kernel Family
Tile/Launch 参数
```

NVIDIA 公布的实验中，该 Recommender 在大规模 GEMM 数据集上达到约：

```text
93% geomean of best available configuration
```

这不表示每个 Shape 都会选中绝对最快方案，也没有公开 Grouped API 的完整选核特征。

### 5.4 Grouped API 没有 Algo 参数

`cublasGemmGroupedBatchedEx` 调用没有：

```text
cublasGemmAlgo_t
```

调用方不能像某些 cuBLASLt 接口那样显式遍历算法。具体实现选择由库内部完成。

## 六、高性能 Grouped Kernel 必须解决什么

即使看不到闭源代码，也可以确定一个 Grouped Kernel 必须解决以下问题。

### 6.1 不规则 Tile 数

第 `i` 个 GEMM：

```text
C_i[M_i, N_i] =
    A_i[M_i, K_i]
    * B_i[K_i, N_i]
```

若 CTA Tile 为：

```text
T_M x T_N
```

第 `i` 个 Problem 的输出 Tile 数：

```text
tiles_i =
    ceil(M_i / T_M)
    * ceil(N_i / T_N)
```

不同 Expert 的 `M_i` 不同，所以 `tiles_i` 不同。

### 6.2 Tile 必须映射到 Problem

设备端工作单元需要知道：

```text
当前 Tile 属于哪个 Problem？
在该 Problem 中的 tile_m/tile_n 是多少？
该 Problem 的 A/B/C 地址和 Shape 是什么？
```

### 6.3 小 Problem 的尾部浪费

若：

```text
M_i < T_M
```

Tile 中很多 Row 无效。

Expert 越多、Token 分布越不均，Padding/Waste 越明显。

### 6.4 负载均衡

若每个 CTA 固定绑定一个 Expert：

```text
大 Expert 仍在运行
小 Expert 已结束
-> 部分 SM 空闲
```

Grouped Kernel 需要在不同 Expert 的 Tile 间做更细粒度调度。

## 七、设备端 Tile 调度可以怎样实现

以下是公开 Grouped GEMM 设计中常见的方法，不应当被表述为 cuBLAS 当前闭源 Kernel 的唯一实现。

### 7.1 前缀 Tile Offset

先计算：

```text
tile_offsets[0] = 0
tile_offsets[i+1] =
    tile_offsets[i] + tiles_i
```

全局 Tile ID：

```text
global_tile in [0, tile_offsets[num_problems])
```

通过查找 Offset 区间得到：

```text
problem_id
local_tile_id
```

### 7.2 Device-side Scan

CTA 拿到 Global Tile ID 后，在设备端扫描 Problem Size，找到所属 Problem。

优点：

- Host 准备简单。
- Shape 可动态变化。

缺点：

- Problem 多时有扫描开销。

### 7.3 Host Precompute

Host 预先生成：

```text
global_tile -> (problem_id, local_tile_id)
```

再复制到 Device。

优点：

- Kernel 查表快。

缺点：

- Host Metadata 和 H2D 开销。
- 动态 Shape 每次都要重建。

### 7.4 Persistent CTA

启动固定数量 CTA：

```text
CTA 处理一个 Tile
-> 从 Work Queue 领取下一个 Tile
-> 继续执行
```

可以改善 Expert 负载不均，但需要原子计数或预计算任务表。

### 7.5 GEMM Tile 内部

拿到 Problem 和 Tile 后，核心仍是普通 Tiled GEMM：

```text
从 HBM 加载 A/B Tile
-> Shared Memory/片上流水
-> MMA 累加
-> Epilogue
-> 写 C Tile
```

CUDA 12.5 初版 Grouped Kernel 使用 Warp-level MMA 是官方事实；具体 Tile、Stage 和 Scheduler 未公开。

## 八、LMDeploy 为什么只在 SM100 路径注册

LMDeploy 的这个 Backend 不是对所有 GPU 注册。

### 8.1 编译条件

CMake 要求：

```text
构建目标包含 SM100
CUDA Compiler >= 12.5
```

才定义：

```text
ENABLE_CUBLAS_GROUPED
```

### 8.2 运行时条件

Registry 只在：

```text
Sm100::is_compatible(arch)
```

时加入 `CublasGroupedKernel`。

项目中的 SM100 范围：

```text
[1000, 1200)
```

SM120 被单独归类，不使用该 Kernel。

### 8.3 这是 LMDeploy 的选择

cuBLAS Grouped API 本身并不是只允许 Blackwell。CUDA 12.5 官方发布时已经展示 Hopper 上的 Grouped GEMM 性能。

LMDeploy 把该 Backend 限制到 SM100，是其当前：

- Kernel Registry。
- 权重布局。
- MoE 路径。
- 验证范围。

共同做出的工程选择，不能反推为 cuBLAS API 的硬件限制。

## 九、LMDeploy 调用前如何整理 Expert

LMDeploy 接收的是连续 Expert Token Buffer 和 Expert Offset。

设：

```text
offsets = [0, M_0, M_0+M_1, ...]
```

第 `i` 个 Expert 的 Token 数：

```cpp
M_i = offsets[i + 1] - offsets[i];
```

输入起始地址：

```cpp
input_ptr_i =
    input_base
    + offsets[i] * input_ld * element_size;
```

输出起始地址：

```cpp
output_ptr_i =
    output_base
    + offsets[i] * output_ld * element_size;
```

### 9.1 权重地址

有两种表示。

连续权重：

```cpp
weight_ptr_i =
    weight_base
    + i * K * N * element_size;
```

或通过 `Bdesc.offsets` 定位。

离散权重指针：

```text
Bdesc.ld == 0
```

表示 `B` 本身是一组 `StridedPtr`。LMDeploy 先把这些指针复制回 Host，提取每个 Expert 的真实 Weight Pointer。

## 十、空 Expert 为什么被过滤

MoE 路由后可能出现：

```text
M_i = 0
```

LMDeploy 直接跳过：

```cpp
if (M_i <= 0) {
    continue;
}
```

收益：

- 不把零尺寸 GEMM 放入 API。
- 减少 Host Metadata。
- 减少 Device Pointer Array。
- 避免空 Expert 影响内部调度。

过滤后：

```text
active_count <= num_experts
```

后续所有数组只为 Active Expert 构造。

## 十一、Row-major 到 cuBLAS Column-major 的零拷贝映射

LMDeploy 应用侧希望计算：

```text
Y_i[M_i, N] =
    X_i[M_i, K]
    * W_i[K, N]
```

X、W、Y 都按 Row-major 存储。

cuBLAS Legacy GEMM 使用 Column-major 语义。关键恒等式：

```text
Y_i^T =
    W_i^T
    * X_i^T
```

Row-major Buffer 与其转置的 Column-major View 使用相同字节顺序：

```text
RowMajor X[M, K]
<same bytes>
ColumnMajor X^T[K, M]
```

### 11.1 cuBLAS 视角

把 Weight Buffer 作为 cuBLAS 的 A：

```text
A_cublas = W_i^T: [N, K]
```

把 Input Buffer 作为 cuBLAS 的 B：

```text
B_cublas = X_i^T: [K, M_i]
```

输出：

```text
C_cublas = Y_i^T: [N, M_i]
```

因此 cuBLAS 参数：

```text
m = N
n = M_i
k = K

transa = N
transb = N

lda = N
ldb = K
ldc = N
```

不用显式 Transpose Kernel，也不用复制矩阵。

### 11.2 为什么调用中的 A 是 Weight

应用公式：

```text
Y = XW
```

cuBLAS 公式：

```text
Y^T = W^T X^T
```

所以调用数组顺序变成：

```text
cuBLAS Aarray -> Expert Weight
cuBLAS Barray -> Expert Input
cuBLAS Carray -> Expert Output
```

这正是选中代码中看起来“A/B 反了”的原因。

## 十二、逐项解释 cublasGemmGroupedBatchedEx 调用

LMDeploy 的调用可以整理成：

```cpp
cublasStatus_t status = cublasGemmGroupedBatchedEx(
    cublas_,
    transa_array.data(),
    transb_array.data(),
    m_array.data(),
    n_active.data(),
    k_array.data(),
    alpha_array.data(),
    reinterpret_cast<const void* const*>(d_ptrs),
    cuda_type,
    lda_active.data(),
    reinterpret_cast<const void* const*>(d_ptrs + one_array),
    cuda_type,
    ldb_array.data(),
    beta_array.data(),
    reinterpret_cast<void* const*>(d_ptrs + 2 * one_array),
    cuda_type,
    ldc_array.data(),
    active_count,
    group_size.data(),
    CUBLAS_COMPUTE_32F);
```

对应含义：

| 参数 | LMDeploy 值 | 含义 |
| --- | --- | --- |
| `handle` | `cublas_` | cuBLAS Context |
| `transa` | 全部 `N` | Weight Buffer 已被解释为 Column-major `W^T` |
| `transb` | 全部 `N` | Input Buffer 已被解释为 Column-major `X^T` |
| `m` | 全部 `N` | cuBLAS 输出行数 |
| `n` | `M_i` | 每个 Expert 的 Token 数 |
| `k` | 全部 `K` | Hidden/Reduction Dimension |
| `alpha` | 每组相同或调用传入 | GEMM Scale |
| `A_ptrs` | Expert Weight Pointer | cuBLAS 左操作数 |
| `lda` | `N` | Weight Column-major Leading Dimension |
| `B_ptrs` | Expert Input Pointer | cuBLAS 右操作数 |
| `ldb` | Input LD，通常 `K` | Input Column-major Leading Dimension |
| `beta` | 调用传入 | 旧输出 Scale |
| `C_ptrs` | Expert Output Pointer | In-place C/D Buffer |
| `ldc` | Output LD，通常 `N` | 输出 Leading Dimension |
| `group_count` | `active_count` | Active Expert 数 |
| `group_size` | 全部 1 | 每个 Group 只有一个 Problem |
| `compute_type` | `CUBLAS_COMPUTE_32F` | FP32 计算/累加语义 |

## 十三、LMDeploy 为什么让每个 Expert 单独成组

LMDeploy 构造：

```cpp
group_size = [1, 1, ..., 1];
group_count = active_count;
```

也就是：

```text
一个 Active Expert = 一个 Group = 一个 Problem
```

### 13.1 原因

Expert 间变化的是：

```text
M_i
```

同一 Group 要求 Shape 一致。只要 `M_i` 不同，就不能放进同一个 Uniform Group。

### 13.2 相同 M 能否合并

理论上，若多个 Expert 恰好有相同 `M_i`，可以：

```text
一个 Group
group_size > 1
```

但需要：

- 按 Shape 重新排序 Pointer Array。
- 构造 Group 到 Expert 的映射。
- 增加 Host 分组逻辑。

当前实现不做这一步。即使 `group_size=1`，所有 Group 仍在同一次 Grouped API 调用中提交。

## 十四、一个完整数字例子

设三个 Expert：

```text
M = [0, 3, 1]
K = 4
N = 2
```

Offset：

```text
offsets = [0, 0, 3, 4]
```

Expert 0 没有 Token，被过滤。

Active Expert：

```text
Expert 1:
  X_1 [3, 4]
  W_1 [4, 2]
  Y_1 [3, 2]

Expert 2:
  X_2 [1, 4]
  W_2 [4, 2]
  Y_2 [1, 2]
```

调用侧数组：

```text
active_count = 2

transa = [N, N]
transb = [N, N]

m = [2, 2]      # 应用侧 N
n = [3, 1]      # 每个 Expert 的 M_i
k = [4, 4]

lda = [2, 2]
ldb = [4, 4]
ldc = [2, 2]

group_size = [1, 1]
```

Device Pointer Array：

```text
A_ptrs = [&W_1, &W_2]
B_ptrs = [&X_1, &X_2]
C_ptrs = [&Y_1, &Y_2]
```

cuBLAS 执行：

```text
Y_1^T[2,3] = W_1^T[2,4] * X_1^T[4,3]
Y_2^T[2,1] = W_2^T[2,4] * X_2^T[4,1]
```

底层字节正好写回 Row-major：

```text
Y_1[3,2]
Y_2[1,2]
```

## 十五、BF16/FP16 输入与 FP32 计算

当前 LMDeploy Backend 只接受：

```text
FP16
BF16
```

并要求：

```text
A/B/C Dtype 相同
不使用 A/B Quantization
```

调用：

```text
computeType = CUBLAS_COMPUTE_32F
```

### 15.1 计算语义

输入和输出可以是 BF16/FP16：

```text
Input Load: BF16/FP16
Multiply: 由 cuBLAS 选择可用 Kernel，合适 Shape 通常可走低精度 Tensor Core
Accumulate/Compute Semantics: FP32
Output Convert: BF16/FP16
```

### 15.2 Alpha/Beta 类型

官方支持表中，这种组合的 Scale Type 是：

```text
FP32
```

所以 LMDeploy 使用：

```cpp
std::vector<float> alpha_array;
std::vector<float> beta_array;
```

### 15.3 禁止 Reduced-precision Reduction

构造 Handle 后设置：

```cpp
cublasSetMathMode(
    handle,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
```

它限制 cuBLAS 使用降低 Reduction 精度的优化，以对齐参考实现的数值语义。

这不表示所有乘法输入都变成 FP32；输入仍是 BF16/FP16，Compute Type 控制累加和计算语义。

## 十六、Workspace 和 Stream 顺序

LMDeploy 使用两块预分配 Workspace。

### 16.1 cuBLAS Workspace

```text
workspace.partials
```

通过：

```cpp
cublasSetWorkspace(...)
```

绑定给 Handle。

### 16.2 Pointer Array Workspace

```text
workspace.tensormaps
```

存放三段连续 Device Pointer Array：

```text
[A pointers][B pointers][C pointers]
```

总大小：

```text
3 * active_count * sizeof(void*)
```

### 16.3 为什么不用每次 cudaMalloc

每次分配三组 Device Pointer Array 会引入：

- Allocator 开销。
- 潜在同步。
- 显存碎片。

预分配 Workspace 只需执行 H2D Copy。

### 16.4 Stream Ordering

代码先在同一个 Stream 中：

```text
H2D copy A pointers
H2D copy B pointers
H2D copy C pointers
```

随后调用 cuBLAS。

cuBLAS Handle 已绑定同一 Stream，因此：

```text
Pointer Copy 完成
-> Grouped Kernel 才能读取 Pointer Array
```

不需要额外 `cudaDeviceSynchronize()`。

## 十七、CPU 开销从哪里来

CUDA 12.5 Update 1 发布说明明确指出，初版 Grouped GEMM API 有较大的 CPU 开销。

LMDeploy 路径本身也包含多个 Host 操作。

### 17.1 Host Vector 构造

每次调用构造：

```text
Shape arrays
LD arrays
Alpha/Beta arrays
Group Size
A/B/C Host Pointer arrays
```

### 17.2 Device Offset 回读

若 Expert Offset 在 GPU：

```text
cudaMemcpyAsync D2H
-> cudaStreamSynchronize
```

同步是为了 Host 立即读取：

```text
M_i = offsets[i+1] - offsets[i]
```

### 17.3 离散权重指针回读

若 Weight 使用 Device `StridedPtr` 数组，也需要：

```text
D2H copy + Stream Sync
```

### 17.4 再把矩阵指针复制回 GPU

Host 过滤 Active Expert 后，又执行：

```text
3 次 H2D Pointer Array Copy
```

### 17.5 CUDA Graph 影响

当前 Wrapper 含：

- 动态 `std::vector`。
- 运行时 D2H。
- Host 读取 Offset。
- 条件 Stream Synchronize。

这条路径不适合直接假设为稳定的 CUDA Graph Capture 路径。

更彻底的 Graph-safe 设计通常需要在 Device 上构建 Active Problem Metadata，避免每步回到 Host。

## 十八、Blackwell 上能确认什么，不能确认什么

### 18.1 能确认

LMDeploy：

- 只在 SM100 范围注册该 Backend。
- 需要 CUDA 12.5+ 编译接口。
- 使用标准 Row-major FP16/BF16 Weight。
- 不做 SM80/SM90 自定义 Tiled Weight 转换。
- 把 Expert GEMM 一次提交给 cuBLAS。

NVIDIA 公开资料：

- Grouped API 支持不同 Shape、Transpose、Scale。
- CUDA 12.5 描述其为一次 Kernel Launch。
- 初版 Grouped Kernel 使用 Warp-level MMA。
- cuBLAS 有 Runtime Recommender。

### 18.2 不能确认

不能仅凭 API 推断当前 Blackwell 库内部一定使用：

- UMMA。
- TMA。
- TMEM。
- Persistent CTA。
- 某个固定 Tile。
- 某种固定 Cluster Shape。

这些细节可能随：

- cuBLAS 版本。
- GPU。
- Dtype。
- Shape 分布。
- Workspace。

变化，而且 Kernel 源码未公开。

### 18.3 为什么初版 Warp MMA 仍可能快

Grouped Workload 的瓶颈不只在单 Tile MMA 峰值：

```text
Host Launch
Kernel Dispatch
小 M 利用率
Expert 负载不均
尾部 SM 空闲
```

一个调度更好的 Warp MMA Grouped Kernel，可能胜过多次独立 WGMMA GEMM。

## 十九、Grouped GEMM 为什么可能更快

### 19.1 减少 Launch

假设有 `E` 个 Active Expert。

逐 Expert：

```text
E 次 API Dispatch
E 次 Kernel Launch
```

Grouped：

```text
1 次 API Dispatch
通常 1 次 Grouped Kernel Launch
```

小 GEMM 下 Launch 占比很高。

### 19.2 更好的跨 Expert 调度

若设备端能够在 Expert Tile 间取任务：

```text
小 Expert 完成后
-> CTA 继续处理其他 Expert
```

减少固定 Expert 映射造成的尾部空闲。

### 19.3 避免 Padding 到统一 M

用普通 Batched GEMM 可能要 Padding：

```text
M_i -> max(M_i)
```

浪费 FLOPs：

```text
wasted_flops =
  2 * K * N
  * sum(max_M - M_i)
```

Grouped GEMM 为每个 Group 保留真实 M。

### 19.4 总 FLOPs

LMDeploy MoE Grouped GEMM：

```text
FLOPs =
  sum_i 2 * M_i * K * N
```

若每个 `M_i` 很小，Kernel 更容易受调度、带宽和 Launch 限制，而非 Tensor Core 峰值限制。

## 二十、什么时候反而可能更慢

### 20.1 CPU Metadata 占主导

Expert GEMM 极小时：

```text
Host 构造 + D2H Sync + H2D Pointer
```

可能比实际 GEMM 更贵。

### 20.2 Problem 数很少

只有一两个大 Expert 时，普通 GEMM 已能高效运行，Grouped 调度收益有限。

### 20.3 Shape 近似统一

若所有 Expert 的 M 相同，普通 Strided/Pointers Batched GEMM 可能更简单。

### 20.4 大 GEMM

每个 Expert 都足够大时：

- 单 GEMM 已占满 GPU。
- 专用 GEMM Kernel 可能使用更强的架构特性。
- Grouped Kernel 的动态调度可能增加开销。

### 20.5 官方建议

cuBLAS 文档明确提醒：

> 某些 Problem Size 下，在不同 CUDA Stream 中多次调用 `cublasGemmBatchedEx` 可能比 Grouped API 更有利。

因此 Grouped GEMM 不是无条件最优。

## 二十一、如何正确 Profile

### 21.1 分开测 Host 和 Device

记录：

```text
Offset D2H 时间
Stream Synchronize 时间
Host Metadata 构造
Pointer Array H2D
cuBLAS Dispatch
Grouped Kernel
后续 Epilogue
```

只看 Kernel Duration 会漏掉该路径的重要 CPU 成本。

### 21.2 Nsight Systems

观察：

- 是否只有一次 Grouped Kernel Launch。
- Launch 前是否有 D2H + Sync。
- Pointer Array H2D 是否与其他工作重叠。
- CPU Thread 是否在 cuBLAS 调用中停留较久。
- 是否有多个 Stream 可并行的空洞。

### 21.3 Nsight Compute

关注：

```text
Tensor Core Pipe Utilization
SM Active
Eligible Warps
HBM Throughput
Shared Memory
Occupancy
Wave Quantization
Instruction Mix
```

若想判断当前 cuBLAS Kernel 是否使用 Warp MMA、WGMMA 或 Blackwell 新指令，应以实际 SASS/Profile 为准，而不是沿用 CUDA 12.5 发布时的描述。

### 21.4 负载分布

Benchmark 至少覆盖：

```text
均匀 M_i
长尾 M_i
大量空 Expert
少量大 Expert
大量小 Expert
不同 Active Expert 数
```

只测平均分布不能代表真实 MoE Router。

## 二十二、正确性与接入检查表

### Shape

- `offsets[num_experts] == total_rows`。
- `M_i >= 0`。
- Input LD 至少为 K。
- Output LD 至少为 N。
- Weight Storage 与零拷贝转置映射一致。

### Pointer

- A/B/C Pointer Array 在 Device。
- Shape/LD/Transpose/Scale Array 在 Host。
- Output Problem 之间不重叠。
- FP16/BF16 Pointer 满足官方对齐要求。
- Device Pointer Array 在 Kernel 完成前保持有效。

### Dtype

- A/B/C 类型组合受 API 支持。
- Alpha/Beta 类型与 Compute Type 匹配。
- FP32 Compute 的数值误差符合预期。
- Reduced-precision Reduction 设置符合基线。

### Runtime

- Handle 绑定正确 Stream。
- Workspace 生命周期覆盖调用。
- Offset D2H 同步不会破坏目标异步流水。
- 空 Expert 已过滤。
- cuBLAS 失败状态被检查。

### 性能

- Grouped Kernel 是否真的单次 Launch。
- CPU Metadata 是否成为瓶颈。
- Active Expert 数和 M 分布是否适合 Grouped。
- 是否应与 Multi-stream Batched GEMM 对比。
- 是否需要缓存稳定 Shape 的 Host Metadata。

## 二十三、参考资料

1. [NVIDIA cuBLAS API：`cublasGemmGroupedBatchedEx`](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex)
2. [NVIDIA Technical Blog：Introducing Grouped GEMM APIs in cuBLAS](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
3. [NVIDIA CUDALibrarySamples：GemmGroupedBatchedEx](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLAS/Extensions/GemmGroupedBatchedEx)
4. [CUDA 12.5 Update 1 Release Notes](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-toolkit-release-notes/index.html)
5. [NVIDIA CUTLASS Grouped GEMM Example](https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped)

## 二十四、总结

`cublasGemmGroupedBatchedEx` 的核心价值不是改变 GEMM 数学，而是改变不规则 GEMM 的提交和调度方式：

```text
多个不同 Shape/参数的 GEMM
-> Host Group Metadata
-> Device Matrix Pointer Arrays
-> cuBLAS Runtime 选核
-> Grouped Kernel 统一执行
```

LMDeploy 在 SM100 MoE 路径中完成了四个关键桥接：

1. 根据 Expert Offset 得到每个 `M_i`，过滤空 Expert。
2. 把连续 Token/Output Buffer 切成每 Expert 指针。
3. 利用 `Y^T = W^T X^T`，把 Row-major 矩阵零拷贝映射到 cuBLAS Column-major。
4. 把 Host Shape 数组和 Device Pointer 数组按 API 要求组织后一次提交。

对内部实现可以确定：

- CUDA 12.5 官方描述为一次 Kernel Launch。
- 初版 Kernel 使用 Warp-level MMA。
- cuBLAS 通过运行时 Recommender 选择实现和 Launch 参数。

不能确定：

- 当前 Blackwell Kernel 的具体 Tile、TMA/TMEM/UMMA 和 CTA Scheduler。

性能判断必须同时看：

```text
Host Metadata
D2H/H2D
Kernel Launch
Expert M 分布
设备端利用率
```

Grouped GEMM 最适合大量不规则小/中型 Expert GEMM；当 Problem 很少、很大或形状统一时，普通 GEMM、Batched GEMM 或 Multi-stream 调度仍可能更快。
