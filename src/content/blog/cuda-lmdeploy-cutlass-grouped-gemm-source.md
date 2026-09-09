---
title: 'LMDeploy Grouped GEMM 源码详解：从 C++ 调度到 CUTLASS WGMMA'
description: '沿 LMDeploy TurboMind 的 MoE FP8 Grouped GEMM 调用链，从 offsets、C++ 模板和运行时选核讲到 Persistent Scheduler、动态 TMA 描述符、CUTLASS Pipeline、Hopper WGMMA 与 Epilogue。'
category: 'CUDA'
pubDate: '2026-09-09T10:00:00+08:00'
updatedDate: '2026-09-09T10:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---
## 目录
1. [先确定要读的 Kernel](#一先确定要读的-kernel)
2. [从普通 CUDA 理解 Grouped GEMM](#二从普通-cuda-理解-grouped-gemm)
3. [MoE 如何把 offsets 传到 GEMM](#三moe-如何把-offsets-传到-gemm)
4. [C++ 模板、注册表与运行时选核](#四c-模板注册表与运行时选核)
5. [Persistent Scheduler 如何分配 Expert Tile](#五persistent-scheduler-如何分配-expert-tile)
6. [TMA 与 WGMMA 如何组成五级流水](#六tma-与-wgmma-如何组成五级流水)
7. [Epilogue、完整执行链与阅读方法](#七epilogue完整执行链与阅读方法)
> **源码基线**：LMDeploy commit `b5cf542`，其 CMake 固定使用 CUTLASS `v3.9.2`。本文分析 `GemmUniversalSm90_v5` 的 Hopper FP8 Grouped GEMM 路径，不讨论 Blackwell 上调用 cuBLAS 的另一条实现。
## 一、先确定要读的 Kernel
LMDeploy 的 Grouped GEMM 不只有一种实现。本文选择下面这个模板实例：
```cpp
KernelImplSm90<
    GemmUniversalSm90_v5<
        kColMajor,  // Tile 的遍历顺序
        1,          // A multicast factor
        1,          // B multicast factor
        true        // grouped GEMM
    >
>
```
它在 Hopper `SM90` 上计算一个专用的 Block-scaled FP8 GEMM：
```text
A: FP8 E4M3，按 Expert 连续排列的激活
B: FP8 E4M3，每个 Expert 一个权重指针
U: FP32，A 沿 K 维分组的 Scale
V: FP32，B 的二维 Block Scale
D: BF16 输出
Accumulator: FP32
```
单个 Expert 的数学形式可以简化成：
```text
D_e = sum_k (A_e,k * U_e,k) @ (B_e,k * V_e,k)
```
实际 Kernel 不是先生成反量化后的 FP16 矩阵。它让 WGMMA 计算 FP8 乘法，再把对应的 `U/V` Scale 应用到 FP32 Partial Accumulator，避免额外反量化 Kernel 和中间 Buffer。
### 它与 CUTLASS 是什么关系
这不是直接调用：
```cpp
cutlass::gemm::device::Gemm<...>
```
LMDeploy 自己实现了 Scheduler、Mainloop 和 Epilogue，但复用了 CUTLASS/CuTe 的底层原语：
```text
cute::SM90_TMA_STORE
cute::SM90_U32x4_STSM_N
cute::warpgroup_arrive / commit_batch / wait
cutlass::PipelineState
cutlass::arch::ClusterTransactionBarrier
cutlass::arch::ClusterBarrier
cutlass::arch::NamedBarrier
cutlass::arch::warpgroup_reg_alloc
```
所以更准确的描述是：
> LMDeploy 使用 CUTLASS/CuTe 的 Hopper 硬件抽象，拼装了一个面向 MoE FP8 的专用 Grouped GEMM。
如果还不熟悉 CTA Tile、Mainloop、Epilogue 和 C++ 模板，可先阅读 [CUTLASS 入门](/blog/cuda-cutlass-from-template-to-gemm/)。
### 先记住调用链
```text
MoeFfnLayer::Forward
  生成 f2n / offsets
        |
        v
LlamaLinear::Forward
  重排并量化 Activation
  构造 MatrixLayout
        |
        v
gemm::Gemm::Run
  Filter -> Dispatch Cache / Heuristic
        |
        v
KernelImplSm90::Launch
  构造 Scheduler、TMA Descriptor、Persistent Grid
        |
        v
gemm_kernel_sm90<Kernel>
        |
        v
GemmUniversalSm90_v5::operator()
  Scheduler -> TMA Load -> WGMMA -> STSM -> TMA Store
```
这条路径涉及的核心源码如下：

| 文件 | 职责 |
|---|---|
| `models/llama/moe_ffn_layer.cc` | Router、Token 映射和 Expert offsets |
| `models/llama/LlamaLinear.cu` | 构造 Grouped GEMM 的 A/B/D 描述符 |
| `kernels/gemm/gemm.cu` | 过滤、选核、缓存与 Launch |
| `kernels/gemm/kernel_impl_sm90.h` | Host 侧 TMA、Grid、Cluster 配置 |
| `kernels/gemm/scheduler.cuh` | Persistent CTA 到 Expert Tile 的映射 |
| `kernels/gemm/gemm_universal_sm90_v5.h` | Producer/Consumer、WGMMA 和写回 |
## 二、从普通 CUDA 理解 Grouped GEMM
### Demo 1：最朴素的多个 Expert
设 Router 将 Token 分给三个 Expert：
```text
Expert 0: 3 tokens
Expert 1: 1 token
Expert 2: 4 tokens
offsets = [0, 3, 4, 8]
```
`offsets` 是 Prefix Sum。第 `e` 个 Expert 的行范围是：
```text
m0  = offsets[e]
m1  = offsets[e + 1]
M_e = m1 - m0
```
将 Token 按 Expert 排列后：
```text
A_grouped:
row 0 1 2 | row 3 | row 4 5 6 7
 expert 0 | exp 1 | expert 2
```
每个 Expert 计算：
```text
D_e[M_e, N] = A_e[M_e, K] @ B_e[K, N]
```
最容易理解的 C++ 写法是循环启动三个 GEMM：
```cpp
for (int e = 0; e < expert_num; ++e) {
    int m0 = offsets[e];
    int m1 = offsets[e + 1];
    int M  = m1 - m0;
    if (M == 0) continue;
    gemm<<<grid_for(M, N), block>>>(
        A + m0 * K,
        weight_ptrs[e],
        D + m0 * N,
        M, N, K);
}
```
它在数学上正确，但性能有三个问题：
```text
1. 每个 Expert 一次 Kernel Launch。
2. 小 M Expert 只有少量 CTA，无法占满 GPU。
3. 某个 Expert 结束后，它空闲的 SM 不能自动帮助其他 Expert。
```
Grouped GEMM 的目标不是改变矩阵乘法，而是把所有 Expert 的 Tile 放进同一个任务池，让固定数量的 CTA 持续取任务。
### Demo 2：最小 Persistent Work Queue
先不考虑 CUTLASS，可以写出等价的调度骨架：
```cpp
__device__ int next_tile;
__global__ void persistent_grouped_gemm(Problem* problems, int total_tiles) {
    while (true) {
        int tile_id = atomicAdd(&next_tile, 1);
        if (tile_id >= total_tiles) return;
        Tile tile = map_global_tile_to_expert(tile_id, problems);
        compute_one_gemm_tile(tile);
    }
}
```
这里的 **Persistent** 表示 Grid 大小不再等于总 Tile 数。只启动接近 GPU 可同时驻留上限的 CTA，每个 CTA 完成一个 Tile 后继续领取下一个：
```text
普通 Grid:
  一个逻辑 Tile 对应一个 CTA
Persistent Grid:
  一个物理 CTA 在生命周期内处理多个逻辑 Tile
```
LMDeploy 的 `TileScheduler::next()` 正是在做 `atomicAdd + tile mapping`，只是还要处理 Cluster、Swizzle、五级 Pipeline 和不同 Expert 的边界。
## 三、MoE 如何把 offsets 传到 GEMM
### Router 先生成四组元数据
`MoeFfnLayer::Forward` 调用 `invokeMoeGate_V2`，得到：
```text
f2n:
  grouped row -> original token
f2E:
  grouped row -> expert id
en2f:
  (expert, original token) -> grouped row
offsets:
  每个 expert 在 grouped rows 中的起止位置
```
以 Top-2 Router 为例，一个 Token 会在 `f2n` 中出现两次，因为它要送到两个 Expert。有效行数为：
```text
M_total = token_num * experts_per_token
```
随后第一层 Expert Linear 调用：
```cpp
linear_.Forward(
    input,
    expert_weight,
    indices,   // f2n
    offsets,   // [expert_num + 1]
    output);
```
`indices` 负责说明每个 Grouped Row 来自哪个原始 Token，`offsets` 负责说明每个 Expert 拥有哪些 Grouped Row。
### 激活为什么先重排
在 `LlamaLinear::GetOperandA` 中，FP8 路径执行：
```cpp
if (indices && A.dtype() == kFloat8_e4m3) {
    Tensor A_e = allocate({M_total, K});
    invokeMoeDispatch(A_e, A, indices.data(), experts_per_token, stream);
    A = A_e;
    indices = {};
}
```
这是源码语义的精简版。重排前，每个 Expert 的 Token 分散在原始输入中，需要 Indexed Gather；重排后：
```text
A = [A_0 rows][A_1 rows]...[A_E-1 rows]
```
Kernel 只需：
```cpp
A_e = A + offsets[e] * lda;
```
连续访问比每个元素读取索引更适合 TMA 大块搬运。
FP8 路径还会同步重排 Activation Scale `U`。这保证 `A` 的第 `r` 行与 `U` 的第 `r` 行仍对应同一个 Token。
### Expert 权重为什么是 Device Pointer Array
各 Expert 权重是独立分配的 Tensor，不保证物理连续。`MoeWeight::LinkLinearExperts` 收集：
```cpp
weights = {
    {expert0_weight_ptr, ld0},
    {expert1_weight_ptr, ld1},
    ...
};
```
`MakeBlockedPtrs` 再生成 GPU 上的：
```text
B_ptrs = [B_0*, B_1*, ..., B_E-1*]
V_ptrs = [V_0*, V_1*, ..., V_E-1*]
```
这里必须区分两层指针：
```text
B_ptrs 本身在 Device Memory；
B_ptrs[e] 指向第 e 个 Expert 的 Device Weight。
```
### `MatrixLayout` 把运行时信息送入后端
LMDeploy 用一个普通 C++ 结构描述矩阵：
```cpp
struct MatrixLayout {
    DataType type;
    Order order;
    int rows, cols, ld;
    Pack pack;
    int num;
    int* offsets;
    int* idxs;
};
```
Grouped 模式的关键字段是：
```text
rows    = 所有 Expert 的有效行总数 M_total
cols    = K 或 N
num     = Expert 数
offsets = Device 上的 int[E+1]
```
`get_mode()` 看到 `offsets != nullptr` 后，将矩阵标记为 `Striding::kBlocked`。这里的 Blocked 不是 CUTLASS Layout 的 Tensor Core Swizzle，而是“矩阵由多个 Expert Block 组成”。
完成描述符后，调用：
```cpp
gemm_.Run(
    operation,
    1.f,
    A, desc_A,
    U, desc_U,
    B_ptrs, desc_B,
    V_ptrs, desc_V,
    0.f,
    D, desc_D,
    D, desc_D,
    workspace,
    stream);
```
到这里仍在 Host 侧。真正的 CUDA Kernel 尚未启动。
## 四、C++ 模板、注册表与运行时选核
这一层同时使用“编译期多态”和“运行时多态”，是初学者最容易迷失的地方。
### 先补齐四个常见 C++ 语法
源码中的：
```cpp
using GroupedGemm = GemmUniversalSm90_v5<kColMajor, 1, 1, true>;
```
只是给复杂类型取别名，没有创建对象。真正创建对象的是：
```cpp
auto kernel = std::make_unique<KernelImplSm90<GroupedGemm>>();
```
`unique_ptr` 表示注册表独占该对象，注册表析构时自动释放，避免手写 `delete`。
`KernelImplSm90` 继承统一基类：
```cpp
class KernelImplSm90 : public Kernel {
    int Launch(...) override;
};
```
注册表保存 `Kernel*`，运行时通过虚函数调用不同模板实例的 `Launch`。这叫运行时多态：调用方不需要知道完整模板类型。
LlamaLinear 中的：
```cpp
auto&& [A, desc_A, U, desc_U] = GetOperandA(...);
```
叫 Structured Binding，它只是把返回的 `std::tuple` 拆成四个变量。`auto&&` 在这里避免无意义复制，并保持返回对象的值类别。
`Gemm::Run` 中还构造了 Lambda：
```cpp
const auto launch = [=](LaunchSpec spec, cudaStream_t stream) {
    return spec.kernel->Launch(/* captured arguments */, stream);
};
```
`[=]` 表示按值捕获外部变量。选核和实测代码只需反复调用 `launch(spec)`，不必重复写长参数列表。
### 模板先生成具体 Kernel 类型
注册代码包含：
```cpp
using GroupedKernel =
    KernelImplSm90<
        GemmUniversalSm90_v5<kColMajor, 1, 1, true>>;
registry.Add(std::make_unique<GroupedKernel>());
```
四个模板参数分别表示：
```text
kColMajor:
  Scheduler 的 Tile Raster Order，不是 A/B 的矩阵布局。
1, 1:
  A/B 的 Cluster Multicast Factor。
true:
  编译 Grouped GEMM 分支。
```
`true` 是编译期常量，因此：
```cpp
if constexpr (is_grouped_gemm) {
    // Grouped 专用代码
}
```
在 Grouped 实例中会保留；普通 GEMM 实例中整段代码不会进入最终设备程序。
同一个模板还实例化 `(2,1)` 和 `(1,2)` 两种 Cluster 版本。运行时选核器根据 Shared Memory、Shape 和估算成本决定使用哪一个。
### 外层 `KernelImplSm90` 是适配器
内层：
```cpp
GemmUniversalSm90_v5<...>
```
定义设备算法和所有编译期常量。外层：
```cpp
KernelImplSm90<Gemm>
```
继承统一的 `Kernel` 接口，负责：
```text
把模板常量转换成 Kernel Descriptor
检查硬件与 Shared Memory 是否可行
创建 TMA Descriptor
计算 Persistent Grid
调用 cudaLaunchKernelEx
```
这相当于：
```text
编译期：
  为每组模板参数生成一个高性能专用 Kernel。
运行时：
  通过 Kernel 基类在多个已编译实例中选择。
```
它不是在运行时重新生成模板。
### Descriptor 如何筛选 Grouped Kernel
`KernelImplSm90` 的构造函数把 Grouped 属性写入描述符：
```cpp
desc_.striding_a = Striding::kBlocked;
desc_.striding_b = Striding::kBlocked;
desc_.striding_c = Striding::kBlocked;
desc_.group_axis = 0;
desc_.cta_tile   = {128, 96, 128};
desc_.stages     = 5;
desc_.arch       = 900;
```
同时写入：
```text
A/B: FP8 E4M3
D:   BF16
A Quant: K-axis group size 128
B Quant: 2D block group size 128
```
`Gemm::Run` 的过程可缩写为：
```cpp
Context ctx;
ctx.Init(operation, A_desc, U_desc, B_desc, V_desc, C_desc, D_desc);
auto feasible = ctx.Filter(registry.kernels());
auto spec = dispatch_cache.Find(ctx.desc());
if (!spec) {
    spec = choose_by_estimated_io_and_mma_cost(feasible);
}
spec.kernel->Launch(...);
```
`Dispatch Cache` 的 Key 包含架构、dtype、Layout、量化格式、Epilogue 和 Shape。Warmup 可以实测候选 Kernel，之后正式推理直接复用结果。
### Host 侧怎样启动 Persistent Cluster
`KernelImplSm90::Launch` 不是按所有逻辑 Tile 创建 Grid，而是先受 SM 数和可驻留 Cluster 数限制：
```cpp
grid = min(
    sm_count,
    max_active_clusters * cluster_size);
block = CTA_SIZE;
```
然后通过扩展 Launch API 设置 Cluster：
```cpp
cudaLaunchAttribute attr{};
attr.id = cudaLaunchAttributeClusterDimension;
attr.val.clusterDim.x = cluster_size;
cudaLaunchKernelEx(&config, kernel, ...);
```
这里 `CTA_SIZE` 为：
```text
5 warp groups * 128 threads = 640 threads
```
一个 CTA 内有 4 个 Math Warpgroup 和 1 个 Producer Warpgroup。
## 五、Persistent Scheduler 如何分配 Expert Tile
Scheduler 最终要把一个全局任务编号映射为：
```text
(expert_id, offset_m, offset_n, m0, m1)
```
其中：
```text
m0 = offsets[expert_id]
m1 = offsets[expert_id + 1]
M_e = m1 - m0
```
### Demo 3：先手算 Tile 数
假设：
```text
offsets = [0, 130, 150, 420]
N = 192
CTA Tile = [128, 96]
```
三个 Expert 的逻辑 Tile 数：
```text
Expert 0:
  ceil(130 / 128) * ceil(192 / 96) = 2 * 2 = 4
Expert 1:
  ceil( 20 / 128) * ceil(192 / 96) = 1 * 2 = 2
Expert 2:
  ceil(270 / 128) * ceil(192 / 96) = 3 * 2 = 6
total = 12 tiles
```
如果只启动 4 个 Persistent CTA，它们会重复领取：
```text
CTA 0: tile 0 -> tile 4 -> tile 8
CTA 1: tile 1 -> tile 5 -> tile 9
CTA 2: tile 2 -> tile 6 -> tile 10
CTA 3: tile 3 -> tile 7 -> tile 11
```
真实顺序还会受到 Cluster 和 Swizzle 影响。
### 全局原子计数器只分配 Cluster ID
`TileScheduler::next()` 的核心是：
```cpp
if (lane_id == 0) {
    wait_until_scheduler_stage_is_free();
    cluster_idx = atomicAdd(next_cluster_id_, 1);
}
cluster_idx = __shfl_sync(FULL_MASK, cluster_idx, 0);
```
只让一个 Lane 执行 `atomicAdd`，再用 Warp Shuffle 广播，避免 32 个线程重复争抢计数器。
`next_cluster_id_` 来自 Workspace。每次 Launch 前 Host 执行：
```cpp
cudaMemsetAsync(workspace.flags, 0, sizeof(int), stream);
```
所以所有常驻 CTA 从任务 0 开始共同消费。
### 一个 `cluster_idx` 怎样定位 Expert
不同 Expert 的 `M_e` 不同，因此不能简单：
```text
expert_id = cluster_idx / fixed_tiles_per_expert
```
Scheduler 根据 `offsets` 计算每个 Expert 的起始任务位置。源码公式为：
```cpp
return (swizzle_tile_x_.div(offsets_[e]) + e)
       * swizzle_unit_.x
       * padded_cluster_tiles_.y;
```
`FastDivmod::div` 完成预计算除数的整数除法。公式中的额外 `e` 和 Padding 为每个 Expert 保留独立的 Swizzle 区域，使一个 Expert 的边界 Tile 不会与下一个 Expert 混在一起。
`update_sync()` 用一个 Warp 并行检查一段 Expert：
```cpp
pred = cluster_idx < start(candidate_expert);
mask = __ballot_sync(FULL_MASK, pred);
```
`__ballot_sync` 把 32 个布尔值压成 32-bit Mask，再通过 `__clz` 找到边界，从而确定当前任务属于哪个 Expert。它相当于一次 Warp 级分段查找，比每个线程独立做完整二分查找更适合这里。
找到 Expert 后再读取：
```cpp
group_m0 = offsets[e];
group_m1 = offsets[e + 1];
group_m  = group_m1 - group_m0;
```
最后 `unswizzle()` 得到该 Expert 内的 `(tile_m, tile_n)`，写入 `Tile1`：
```cpp
struct Tile1 {
    int is_valid_cta;
    int is_valid_cluster;
    int offset_m;
    int offset_n;
    int alive;
    int group_idx;
    int m0;
    int m1;
};
```
### 为什么还需要 `alive` 和两种 Valid
Cluster 中多个 CTA 必须按 Cluster 语义协同。有时当前 CTA 已越过矩阵边界，但同 Cluster 的另一个 CTA 仍有效：
```text
is_valid_cta = false
is_valid_cluster = true
```
无效 CTA 不能直接退出，否则其他 CTA 可能永远等不到 Barrier 到达。它仍需推进 Pipeline、释放 Barrier，只跳过真实计算。
`alive=false` 才表示全局任务队列已经结束，可以退出 Persistent Loop。
## 六、TMA 与 WGMMA 如何组成五级流水
### Kernel 的线程分工
模板固定：
```cpp
WARPGROUPS = 4;
CTA_SIZE   = 128 * (4 + 1);
Stages     = 5;
```
线程角色为：
```text
Warpgroup 4:
  Producer。
  一个 Warp 负责 TMA Load；
  另一个 Warp 负责 Scheduler。
Warpgroup 0..3:
  Consumer / Math。
  两个 Warpgroup 组成一组，共同消费一个 CTA Tile。
```
Producer 调用：
```cpp
cutlass::arch::warpgroup_reg_dealloc<32>();
```
Math Warpgroup 调用：
```cpp
cutlass::arch::warpgroup_reg_alloc<112>();
```
Hopper 支持 Warpgroup Register Reconfiguration。Producer 只做地址和搬运控制，少分配寄存器；释放出的寄存器预算交给持有大量 FP32 Accumulator 的 Math Warpgroup。
### Shared Memory 中放什么
源码中的 `SharedStorage` 可简化为：
```cpp
struct SharedStorage {
    FP8 A[5][128][128];
    FP8 B[5][ 96][128];
    FP32 U[5][aligned_M_scales];
    FP32 V[5][2];
    BF16 C[128][96];
    uint64_t producer_bar[5];
    uint64_t consumer_bar[5];
    CUtensorMap tma_desc[7];
    Scheduler::Storage scheduler;
};
```
每个 Stage 保存一个 K Tile：
```text
A Tile: [128, 128]
B Tile: [ 96, 128]
```
`Stages=5` 表示最多让五个 K Tile 处于不同的搬运/计算阶段。它不是把 K 维固定成 5 段；K 更大时 Ring Buffer 会循环复用这些 Stage。
### Grouped GEMM 为什么需要动态 TMA Descriptor
普通 GEMM 的 B 基地址固定，Host 创建一次 `CUtensorMap` 即可。Grouped GEMM 的 B 是：
```text
B_ptrs[e]
```
每拿到一个新 Expert，TMA 的 Global Address 和 M 维范围都可能改变。
Producer 从 Tile 中取出：
```cpp
int e  = tile->group_idx;
int m0 = tile->m0;
int m1 = tile->m1;
```
然后准备三个地址：
```cpp
A_addr = A_base + m0 * lda;
B_addr = B_ptrs[e];
U_addr = U_base + aligned(m0);
```
接着用 Hopper Tensormap Replace 指令修改模板描述符：
```text
tensormap.replace.tile.global_address
tensormap.replace.tile.global_dim
tensormap.cp_fenceproxy
```
修改后的 Descriptor 写入 Workspace，再用：
```cpp
cute::tma_descriptor_fence_acquire(...)
```
保证 TMA Engine 看到新地址。这也是该实现要求 CUDA 12.3 以上的原因。
### Producer 的五级循环
源码的核心可以还原成：
```cpp
PipelineState<5> write_state{producer_start};
for (int k_tile = 0; k_tile < ceil_div(K, 128); ++k_tile) {
    int stage = write_state.index();
    wait_until_consumers_release(stage);
    expect_transaction_bytes(stage, A_bytes + B_bytes + scale_bytes);
    tma_load(A[k_tile], smem_A[stage], producer_bar[stage]);
    tma_load(B[k_tile], smem_B[stage], producer_bar[stage]);
    tma_load(U[k_tile], smem_U[stage], producer_bar[stage]);
    cp_async(V[k_tile], smem_V[stage]);
    arrive_for_V_copy(producer_bar[stage]);
    ++write_state;
}
```
TMA 完成目标字节数后，`ProducerBar` 才切换 Phase，Consumer 才能读取该 Stage。
### Consumer 如何执行 WGMMA
Consumer 维护自己的 `pipe_state`：
```cpp
wait(producer_bar[stage]);
load_scale_UV_from_smem();
reset_smem_descriptor_to_stage(stage);
GMMA::apply(A_smem_desc, B_smem_desc, accumulator, U, V);
release(consumer_bar[stage]);
++pipe_state;
```
`GMMA::apply` 最终选择类似：
```cpp
cute::SM90::GMMA::
MMA_64x96x32_F32E4M3E4M3_SS_TN
```
指令 Shape 表示：
```text
M = 64
N = 96
K = 32
输入 A/B = FP8 E4M3
输出累加 = FP32
SS = A/B 都来自 Shared Memory Descriptor
TN = 指令看到的 A/B 转置模式
```
WGMMA 是 Warpgroup 级异步矩阵乘。128 个线程协同发出指令，Accumulator 分散在 Warpgroup 的寄存器中。调用顺序包含：
```text
warpgroup_arrive
wgmma(...)
warpgroup_commit_batch
warpgroup_wait<N>
```
`commit_batch` 提交一批异步 WGMMA，`wait<N>` 限制尚未完成的批次数。这样地址计算、下一批 WGMMA 与之前的 Tensor Core 计算可以重叠。
### Scale 为什么在 WGMMA 之后应用
`ScaledGmmaFP8_TN` 先得到一个 K Block 的 FP32 `frag_C`，然后执行：
```text
accum_C += frag_C * scale_A(m, k_block) * scale_B(n_block, k_block)
```
不能等整个 K 循环结束后只乘一次 Scale，因为每个 K Block 的 Scale 不同：
```text
C = sum_kblock [
      (A_fp8_block @ B_fp8_block)
      * U_scale(kblock)
      * V_scale(kblock)
    ]
```
这解释了为什么源码同时存在临时 `FragC` 和最终 `AccumC`：前者接收本次 WGMMA Batch，后者累加已经应用 Scale 的结果。
## 七、Epilogue、完整执行链与阅读方法
### 从寄存器写回 BF16
K 循环完成后，结果仍分散在每个 Lane 的 FP32 Accumulator 中。源码通过 `GMMA::foreach_C` 遍历线程拥有的 Fragment：
```cpp
GMMA::foreach_C(accum_C, [&](auto const& frag, int m, int n) {
    Array<bf16, 8> out = cast<bf16>(frag);
    cute::SM90_U32x4_STSM_N::copy(
        out[0], out[1], out[2], out[3],
        smem_C[swizzled_offset(m, n)]);
});
```
这里发生三件事：
```text
FP32 -> BF16 类型转换
寄存器 Fragment -> Shared Memory 矩阵重排
Shared Memory Swizzle 以满足写回访问模式
```
`STSM` 是寄存器到 Shared Memory 的协作存储。Math Warpgroup 完成后用 `NamedBarrier` 同步，再由部分线程执行 TMA Store：
```cpp
cute::SM90_TMA_STORE::copy(
    output_tma_desc,
    smem_C,
    global_n,
    global_m);
```
Grouped 模式下，输出 Descriptor 也会按：
```text
D_base + offsets[e] * ldd
```
动态更新，因此结果仍按 Expert 连续排列。后续 `invokeMoeCombine` 再通过 `en2f/f2E` 把 Expert 输出乘 Router Weight，并 Scatter-Reduce 回原 Token 顺序。
### 一次请求的完整执行顺序
把所有细节压缩成一条链：
```text
1. Router:
   为每个 Token 选择 Top-K Expert。
2. Prefix Sum:
   生成 offsets[e]，确定每个 Expert 的 M_e。
3. Dispatch:
   按 Expert 重排 Token，得到连续 A_grouped。
4. Descriptor:
   A/D 用 offsets 切分；
   B/V 用 Device Pointer Array 选择 Expert。
5. Dispatch:
   Registry 过滤 dtype/layout/quant/arch；
   Cache 或估价器选择具体模板实例。
6. Persistent Launch:
   固定数量 CTA/Cluster 共同消费全局 Tile Queue。
7. Dynamic TMA:
   每个 Tile 根据 expert_id 修改 A/B/U/D 描述符。
8. Mainloop:
   五级 TMA Pipeline 搬 A/B/U/V；
   WGMMA 计算 FP8 Tile并应用 Scale。
9. Epilogue:
   FP32 Accumulator 转 BF16，经 STSM 重排后 TMA Store。
10. Combine:
    按 Router Weight 聚合回原 Token。
```
### 性能为什么可能更好
主要收益来自：
```text
一次 Launch 覆盖所有 Expert
Persistent CTA 跨 Expert 动态负载均衡
TMA 隐藏 Global Memory 搬运
WGMMA 提供 FP8 Tensor Core 吞吐
Warp Specialization 让调度/搬运与计算并行
寄存器重分配给 Accumulator
STSM + TMA Store 降低 Epilogue 指令开销
```
代价也很明确：
```text
动态 TMA Descriptor 修改有固定开销
小 Expert 的边界 Tile 利用率低
Global Atomic Scheduler 可能竞争
五级 Shared Memory 占用限制驻留 CTA
Cluster 内无效 CTA 仍需参与同步
```
所以 Grouped Kernel 不一定在所有 Shape 上获胜。LMDeploy 保留 Registry、Heuristic、实测 Tuning 和 Dispatch Cache，就是为了避免把一个模板用于所有场景。
### 阅读这类 Kernel 的顺序
不要从 `GMMA::apply` 内层开始。更有效的顺序是：
```text
先看 Tensor Shape:
  A/B/U/V/D 分别是什么。
再看 Tile:
  CTA_M/N/K、Stage、Warpgroup 数。
再看 Ownership:
  offsets 和 pointer array 如何定位 Expert。
再看 Scheduler:
  一个物理 CTA 怎样领取多个逻辑 Tile。
再看 Pipeline:
  谁生产、谁消费、何时等待和释放。
最后看 Instruction:
  WGMMA Shape、Fragment Layout、STSM 和 TMA Store。
```
如果能回答下面五个问题，就已经读通了这段代码：
```text
1. 当前 Tile 属于哪个 Expert？
2. A/B/U/V 的真实地址怎样得到？
3. 当前 Shared Memory Stage 何时可以覆盖？
4. 哪 128 个线程共同拥有一次 WGMMA 的 Fragment？
5. 寄存器中的结果怎样回到正确 Expert 的 D 区间？
```
这也是从基础 CUDA 过渡到 CUTLASS/CuTe 源码时最重要的方法：先沿数据和所有权走一遍，再理解模板与指令。
参考源码：

- [MoE Router 与 Grouped Linear 调用](https://github.com/InternLM/lmdeploy/blob/b5cf542e2c9f51898e5195ecbbc25f0836de1c4e/src/turbomind/models/llama/moe_ffn_layer.cc)
- [LlamaLinear 描述符构造](https://github.com/InternLM/lmdeploy/blob/b5cf542e2c9f51898e5195ecbbc25f0836de1c4e/src/turbomind/models/llama/LlamaLinear.cu)
- [SM90 Grouped Kernel 注册](https://github.com/InternLM/lmdeploy/blob/b5cf542e2c9f51898e5195ecbbc25f0836de1c4e/src/turbomind/kernels/gemm/kernel/sm90_64n32_8.cu)
- [Persistent Tile Scheduler](https://github.com/InternLM/lmdeploy/blob/b5cf542e2c9f51898e5195ecbbc25f0836de1c4e/src/turbomind/kernels/gemm/scheduler.cuh)
- [GemmUniversalSm90 v5](https://github.com/InternLM/lmdeploy/blob/b5cf542e2c9f51898e5195ecbbc25f0836de1c4e/src/turbomind/kernels/gemm/gemm_universal_sm90_v5.h)
- [CUTLASS Hopper Pipeline](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html)
