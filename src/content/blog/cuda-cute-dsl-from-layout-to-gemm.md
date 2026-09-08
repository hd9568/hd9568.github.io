---
title: 'CuTe DSL 入门：从 Layout、Tensor 到 GPU GEMM'
description: '面向掌握基础 CUDA 的读者，通过 Hello World、Vector Add、Layout 映射、Shared Memory GEMM 和 Tensor Core 骨架，讲清 CuTe DSL 的 JIT 编译、线程数据划分与执行过程。'
category: 'CUDA'
pubDate: '2026-09-08T10:10:00+08:00'
updatedDate: '2026-09-08T10:10:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---
## 目录
1. [CuTe DSL 是什么](#一cute-dsl-是什么)
2. [Demo 1：Hello World 与两阶段执行](#二demo-1hello-world-与两阶段执行)
3. [Demo 2：从 CUDA Vector Add 理解 DSL Kernel](#三demo-2从-cuda-vector-add-理解-dsl-kernel)
4. [Demo 3：Layout 是 CuTe 的核心](#四demo-3layout-是-cute-的核心)
5. [Demo 4：Shared Memory Tiled GEMM](#五demo-4shared-memory-tiled-gemm)
6. [从教学 GEMM 到 Tensor Core GEMM](#六从教学-gemm-到-tensor-core-gemm)
7. [编译缓存、调试与学习路线](#七编译缓存调试与学习路线)

> **版本说明**：本文代码和 API 以 CUTLASS `v4.7.1` 的 CuTe DSL 为基线。CuTe DSL 仍在快速演进，示例应与安装包和仓库 Tag 保持一致，不能混用不同版本的源码。
## 一、CuTe DSL 是什么
CuTe DSL 是 CUTLASS 4.x 中面向 GPU Kernel 的 Python DSL。它允许用 Python 语法描述：

- CUDA Grid、Block 和 Thread。
- Tensor 的 Shape、Stride 和地址空间。
- 数据怎样分给 CTA、Warp 和 Thread。
- Global/Shared/Register Memory 之间怎样 Copy。
- CUDA Core、Tensor Core 和异步 Pipeline 怎样执行。

它不是 NumPy，也不是“Python 在 GPU 上逐行解释”。
```text
Python 源码
-> AST Rewrite + Tracing
-> CuTe/MLIR 中间表示
-> NVIDIA GPU 相关 Lowering
-> PTX / CUBIN
-> CUDA Driver 启动 Kernel
```
最终运行在 GPU 上的是编译后的设备代码，Python 主要负责生成和启动它。
### CUTLASS、CuTe C++、CuTe DSL 的关系
```text
CUTLASS:
  整个项目和高性能算子生态。
CuTe C++:
  CUTLASS 3.x 内部的 Layout/Tensor/Copy/MMA 模板库。
CuTe DSL:
  与 CuTe C++ 概念一致的 Python Kernel DSL。
```
CuTe DSL 比 CUTLASS C++ 少了大量模板语法，但它仍是低层编程模型。你依然需要理解 CUDA 线程、地址空间、Tiling、同步和 Tensor Core。

如果还不熟悉 CUTLASS 的 GEMM 分层，可以先阅读 [CUTLASS 入门：从 CUDA Tiling 到可组合 GEMM Kernel](/blog/cuda-cutlass-from-template-to-gemm/)。
### 环境
官方 4.x 安装方式以 Linux 为主：
```bash
git clone --branch v4.7.1 https://github.com/NVIDIA/cutlass.git
# 根据对应 Tag 的 Quick Start 选择 CUDA 12/13。
./cutlass/python/CuTeDSL/setup.sh --cu12
pip install torch
```
安装后先确认：
```python
import cutlass
import cutlass.cute as cute
print(cutlass.__version__)
```
后文 Demo 需要 NVIDIA GPU 和兼容驱动。不同架构支持的 MMA、TMA 和 Pipeline 能力不同；Hello World 与普通 CUDA Core Demo 的硬件要求最低。
## 二、Demo 1：Hello World 与两阶段执行
这是官方 `v4.7.1` 教学示例的核心代码：
```python
import cutlass
import cutlass.cute as cute
@cute.kernel
def hello_kernel(meta_arg: cutlass.Constexpr, dynamic_arg: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 1 or tidx == 3:
        cute.printf(
            "tidx={} meta_arg={} dynamic_arg={}",
            tidx,
            meta_arg,
            dynamic_arg,
        )
@cute.jit
def hello_host(meta_arg: cutlass.Constexpr, dynamic_arg: cutlass.Int32):
    hello_kernel(meta_arg, dynamic_arg).launch(
        grid=(1, 1, 1),
        block=(4, 1, 1),
    )
cutlass.cuda.initialize_cuda_context()
hello_host(10, 20)
```
预期输出：
```text
tidx=1 meta_arg=10 dynamic_arg=20
tidx=3 meta_arg=10 dynamic_arg=20
```
### `@cute.kernel` 对应 `__global__`
CUDA C++：
```cpp
__global__ void kernel(int x) {
    int tid = threadIdx.x;
}
```
CuTe DSL：
```python
@cute.kernel
def kernel(x: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
```
两者都描述 GPU 设备代码。`cute.arch.thread_idx()` 最终会 Lower 到线程索引相关设备指令。
### `@cute.jit` 是 Host 侧入口
Python 不能直接调用 `@cute.kernel`。调用链是：
```text
普通 Python
-> @cute.jit Host Function
-> kernel(...).launch(grid, block)
-> GPU Kernel
```
`@cute.jit` 负责构建 Launch 参数，也可以做编译期布局推导和 Kernel 组合。
### `Constexpr` 与动态参数
```python
meta_arg: cutlass.Constexpr
dynamic_arg: cutlass.Int32
```
两者的区别：
```text
Constexpr:
  编译时已知。
  可用于 Tile 大小、循环展开和类型选择。
  值变化通常会产生另一个特化版本。
Int32:
  Kernel 运行时才知道。
  作为普通参数传入设备代码。
```
可以把 `Constexpr` 类比为 C++ 模板参数，把 `Int32` 类比为普通函数参数。
### Python `print` 与 `cute.printf`
```python
print(x)
```
发生在 Python Meta-stage，用于查看编译器看到的 Shape、Layout 和代理值。
```python
cute.printf("x={}", x)
```
编译进 GPU 程序，在 Kernel 运行时输出。前者不能读取真实设备 Tensor 值，后者会显著拖慢 Kernel，只适合调试。
## 三、Demo 2：从 CUDA Vector Add 理解 DSL Kernel
先回忆 CUDA C++：
```cpp
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```
对应的 CuTe DSL 教学写法：
```python
import torch
import cutlass
import cutlass.cute as cute
@cute.kernel
def add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    n: cutlass.Int32,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    idx = bx * bdx + tx
    if idx < n:
        gC[idx] = gA[idx] + gB[idx]
@cute.jit
def add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    threads = 256
    n = cute.size(c)
    blocks = cute.ceil_div(n, threads)
    add_kernel(a, b, c, n).launch(
        grid=(blocks, 1, 1),
        block=(threads, 1, 1),
    )
a = torch.randn(4096, device="cuda", dtype=torch.float32)
b = torch.randn_like(a)
c = torch.empty_like(a)
a_cute = cute.runtime.from_dlpack(a)
b_cute = cute.runtime.from_dlpack(b)
c_cute = cute.runtime.from_dlpack(c)
compiled_add = cute.compile(add, a_cute, b_cute, c_cute)
compiled_add(a_cute, b_cute, c_cute)
torch.testing.assert_close(c, a + b)
```
这段代码刻意使用一线程一元素，方便与基础 CUDA 对照。它不是最终高带宽版本。
### Tensor 从哪里来
`from_dlpack` 不复制数据：
```text
torch.Tensor
  owns CUDA memory
cute.Tensor
  holds pointer + Layout
  views the same CUDA memory
```
因此 PyTorch 分配的 `c` 会被 CuTe Kernel 直接写入。

原始 PyTorch Tensor 必须保持存活；如果它被释放，CuTe Tensor 中的指针也会失效。
### `cute.Tensor` 不只是 Pointer
普通 CUDA 参数是：
```cpp
float* ptr
```
CuTe Tensor 更接近：
```text
Tensor = Iterator/Pointer + Layout
```
当代码写：
```python
gA[idx]
```
CuTe 先用 Layout 把逻辑坐标 `idx` 映射到线性 Offset，再访问底层指针。
### 为什么还需要 `cute.compile`
```python
compiled_add = cute.compile(add, a_cute, b_cute, c_cute)
```
第一次编译会根据参数类型、dtype、Layout 和 `Constexpr` 生成设备代码。之后：
```python
compiled_add(...)
```
只做参数封装和 Kernel Launch，不应把 JIT 编译时间算进 Kernel 延迟。
### 怎样改成向量化
一线程一元素通常生成标量 Load/Store。若 FP32 每线程连续处理 4 个元素，并且地址 16-byte 对齐，编译器可以生成 128-bit 向量访存。

逻辑变化：
```text
thread 0 -> element 0..3
thread 1 -> element 4..7
...
```
CuTe 的正式写法不是随意强转 `float4*`，而是构造 Thread-Value Layout，显式描述：
```text
(thread_id, value_id) -> tile coordinate
```
这正是下一节的核心。
## 四、Demo 3：Layout 是 CuTe 的核心
Layout 是一个函数：
```text
logical coordinate -> linear offset
```
它由 Shape 和 Stride 组成。
### 最小二维例子
```python
@cute.jit
def show_layout():
    row_major = cute.make_layout(
        shape=(2, 4),
        stride=(4, 1),
    )
    col_major = cute.make_layout(
        shape=(2, 4),
        stride=(1, 2),
    )
    print("row_major =", row_major)
    print("col_major =", col_major)
    print("row_major(1,2) =", row_major(1, 2))
    print("col_major(1,2) =", col_major(1, 2))
show_layout()
```
Row-major 地址：
```text
offset(m, n) = m * 4 + n
       n=0 1 2 3
m=0 ->  0 1 2 3
m=1 ->  4 5 6 7
```
所以：
```text
row_major(1,2) = 1*4 + 2 = 6
```
Column-major：
```text
offset(m, n) = m + n * 2
       n=0 1 2 3
m=0 ->  0 2 4 6
m=1 ->  1 3 5 7
```
### Shape 与 Stride 为什么都可以分层
CuTe 允许：
```text
Shape = ((2, 2), 4)
```
外层仍是二维，但第一维内部又分成两个 Mode。这能表达：

- 一个 Warp 中的 Lane 分组。
- 一条向量 Load 的多个值。
- MMA Fragment 中每个线程拥有的元素。
- Shared Memory Swizzle。

它不是为了把简单矩阵写复杂，而是为了用同一种代数表达 GPU 的层级结构。
### `local_tile`：一个 CTA 拿哪块数据
假设：
```text
A shape = [M, K]
CTA tile = [BM, BK]
block coordinate = [block_m, block_k]
```
CuTe 可以写：
```python
gA = cute.local_tile(
    A,
    tiler=(BM, BK),
    coord=(block_m, block_k),
)
```
`gA` 仍然是 Tensor View，没有复制数据。它只是把原 Tensor 与新 Layout 组合，使坐标 `(i,j)` 指向当前 CTA 的元素。
### Thread-Value Layout：每个线程搬哪些元素
设一个 Tile 有 8 个元素，4 个线程，每线程搬 2 个：
```text
thread 0 -> value 0, 1 -> tile element 0, 1
thread 1 -> value 0, 1 -> tile element 2, 3
thread 2 -> value 0, 1 -> tile element 4, 5
thread 3 -> value 0, 1 -> tile element 6, 7
```
这可以看成：
```text
TV Layout:
  (thread, value) -> logical element
```
CuTe 的 `TiledCopy` 持有这个映射：
```python
tiled_copy = cute.make_tiled_copy_tv(
    copy_atom,
    thr_layout,
    val_layout,
)
thr_copy = tiled_copy.get_slice(thread_idx)
src_per_thread = thr_copy.partition_S(src_tile)
dst_per_thread = thr_copy.partition_D(dst_tile)
cute.copy(copy_atom, src_per_thread, dst_per_thread)
```
`partition_S` 和 `partition_D` 没有发起 Copy，它们只回答：
```text
当前线程在 Source/Destination Tensor 中负责哪部分。
```
真正的数据移动发生在 `cute.copy`。
## 五、Demo 4：Shared Memory Tiled GEMM
下面使用 CUTLASS 4.7.1 附带的教学 Primitive API，完整表达基础 CUDA Shared-Memory GEMM。它故意不使用 Tensor Core。
```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims
@cute.kernel
def gemm_kernel(
    a: cutlass.Array,
    b: cutlass.Array,
    c: cutlass.Array,
    TS: cutlass.Constexpr[int],
):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    a_smem = cutlass.Array(
        cutlass.Float32,
        (TS, TS),
        space=cutlass.AddressSpace.smem,
    )
    b_smem = cutlass.Array(
        cutlass.Float32,
        (TS, TS),
        space=cutlass.AddressSpace.smem,
    )
    K = a.shape[1]
    acc = 0.0
    for bk in range(0, K, TS):
        a_smem[ty, tx] = a[bx * TS + ty, bk + tx]
        b_smem[ty, tx] = b[bk + ty, by * TS + tx]
        prims.barrier_cta_sync(0)
        for k in range(TS):
            acc += a_smem[ty, k] * b_smem[k, tx]
        prims.barrier_cta_sync(0)
    c[bx * TS + ty, by * TS + tx] = acc
@cute.jit
def gemm(a, b, c, TS: cutlass.Constexpr[int]):
    M = a.shape[0]
    N = b.shape[1]
    gemm_kernel(a, b, c, TS).launch(
        grid=(M // TS, N // TS, 1),
        block=(TS, TS, 1),
    )
M = N = K = 1024
a = torch.randn(M, K, device="cuda")
b = torch.randn(K, N, device="cuda")
c = torch.zeros(M, N, device="cuda")
gemm(
    cute.runtime.from_dlpack(a),
    cute.runtime.from_dlpack(b),
    cute.runtime.from_dlpack(c),
    TS=32,
)
torch.testing.assert_close(c, a @ b, atol=1e-3, rtol=1e-3)
```
该 Demo 为了突出主线，要求 `M/N/K` 都能被 `TS` 整除。
### 一个 Block 怎样计算
`TS=32` 时：
```text
block = (32, 32)
每个 Block 有 1024 个线程
每个线程计算 C Tile 中一个元素
```
对 `C[bx, by]` Tile：
```text
第 0 轮:
  加载 A[bx*32 : bx*32+32, 0:32]
  加载 B[0:32, by*32 : by*32+32]
第 1 轮:
  加载 K 维下一块
...
```
每轮两个 Barrier：
```text
第一个 Barrier:
  等所有线程完成 Shared Memory 写入。
第二个 Barrier:
  等所有线程用完当前 Tile，才能被下一轮覆盖。
```
### 为什么它仍然不够快
这个版本的问题：

- 每个线程只算一个输出，寄存器复用较少。
- Global-to-Shared 是同步标量 Copy。
- 没有双缓冲，Load 与 Compute 串行。
- 使用 CUDA Core 标量 FMA，没有 Tensor Core。
- `32 x 32 = 1024` 线程让调度约束较强。
- 没有处理边界 Shape。

它的价值是把基础 CUDA 的执行模型一比一迁移到 DSL，而不是作为最终 GEMM。
## 六、从教学 GEMM 到 Tensor Core GEMM
高性能 CuTe DSL GEMM 不再让“一个线程对应一个 C 元素”，而是先构造两个核心对象：
```text
TiledCopy:
  Copy Atom 在 Thread/Value Layout 上的重复。
  决定谁搬什么、一次搬多少、搬到哪里。
TiledMma:
  MMA Atom 在线程或 Warpgroup Layout 上的重复。
  决定谁持有 A/B/C Fragment、发出什么 MMA。
```
### Tensor Core GEMM 的骨架
以下代码是教学骨架，不是可独立复制的完整 Kernel：
```python
@cute.kernel
def tensorcore_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    sA_layout: cute.Layout,
    sB_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    # 1. 当前 CTA 的 Global-Memory Tile。
    gA = cute.local_tile(mA, (128, 128, 32), (bidx, None, None))
    gB = cute.local_tile(mB, (128, 128, 32), (None, bidy, None))
    gC = cute.local_tile(mC, (128, 128, 32), (bidx, bidy, None))
    # 2. Shared Memory Tensor。
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
    # 3. 当前线程的 Copy 分片。
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)
    # 4. 当前线程/Warp 的 MMA 分片与寄存器 Accumulator。
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    accum = tiled_mma.make_fragment_C(tCgC)
    accum.fill(0.0)
    # 5. 沿 K Tile 执行 Mainloop。
    for k_tile in range(cute.size(gA, mode=[2])):
        cute.copy(tiled_copy_A, tAgA[None, None, k_tile], tAsA)
        cute.copy(tiled_copy_B, tBgB[None, None, k_tile], tBsB)
        cute.arch.sync_threads()
        cute.gemm(tiled_mma, accum, tCrA, tCrB, accum)
        cute.arch.sync_threads()
    # 6. 完整实现还需把 Accumulator 按输出 Copy Layout
    #    重排、转换类型并写回 tCgC；此处省略 Epilogue。
```
这里最重要的不是记函数名，而是看懂对象的变化：
```text
mA/mB:
  完整 Global-Memory Tensor
gA/gB:
  当前 CTA 的 Global-Memory Tile
sA/sB:
  当前 CTA 的 Shared-Memory Buffer
tAgA/tAsA:
  当前线程负责的 Global/Shared Copy Fragment
tCrA/tCrB:
  当前 MMA 参与者负责的 Operand Fragment
accum:
  Register 或新架构专用存储中的 Accumulator
```
### Pipeline 怎样加入
教学版本是：
```text
Load tile 0 -> Compute tile 0
Load tile 1 -> Compute tile 1
```
多 Stage Pipeline 是：
```text
time --->
Load:    tile 0 | tile 1 | tile 2 | tile 3
Compute:          tile 0 | tile 1 | tile 2
```
Ampere 可用 `cp.async` 把 Global Memory 异步搬到 Shared Memory；Hopper 可用 TMA + WGMMA；Blackwell 可用 TMA + `tcgen05.mma` + TMEM。

CuTe DSL 中 Pipeline 对象管理：
```text
producer acquire:
  等待某 Stage 为空。
producer commit:
  标记 Load 已提交。
consumer wait:
  等待该 Stage 数据就绪。
consumer release:
  标记该 Stage 可再次覆盖。
```
这与手写 CUDA 双缓冲中的 Buffer Index、Barrier 和 Async Copy 是同一件事，只是被显式建模为可组合对象。
### 为什么 Layout 决定性能
同一组数值可以有不同 Layout：
```text
Global Memory Layout:
  影响 Coalescing 和向量化。
Shared Memory Layout:
  影响 Bank Conflict 和 ldmatrix/TMA 约束。
Thread-Value Layout:
  影响每个 Lane 搬哪些元素。
MMA Fragment Layout:
  必须匹配硬件指令要求。
```
高性能 CuTe Kernel 的主要工作，就是让这些 Layout 在各级 Copy 和 MMA 之间“可组合且对齐”，而不是手写很多数组下标。
## 七、编译缓存、调试与学习路线
### Static 与 Dynamic Layout
显式转换：
```python
a_cute = cute.runtime.from_dlpack(a)
```
默认得到 Static Layout。Shape/Stride 会参与 JIT 特化，换 Shape 可能需要重新编译。

直接把 PyTorch Tensor 传给 `@cute.jit`，运行时会构造较动态的 Layout，多个 Shape 可以复用同一个编译结果，但编译器掌握的信息更少。

折中方式：
```python
a_cute = cute.runtime.from_dlpack(a)
a_cute = a_cute.mark_compact_shape_dynamic(
    mode=0,
    divisibility=128,
)
```
意思是第 0 维运行时可变，但保证能被 128 整除。编译器可以利用这个对齐信息。
### 调试顺序
先查 Meta-stage：
```python
print(tensor.layout)
print(tile.shape)
print(thread_fragment.type)
```
再查 GPU Runtime：
```python
if tidx == 0 and bidx == 0:
    cute.printf("value={}", tensor[0])
```
最后使用：
```text
compute-sanitizer:
  越界、Race、未初始化访问。
Nsight Compute:
  Memory Coalescing、Bank Conflict、Tensor Core、
  Occupancy、Register 和 Stall。
生成 IR/PTX/SASS:
  确认是否生成向量 Load、cp.async、MMA/TMA。
```
### 常见误区
```text
误区 1：Python 循环一定在 CPU 执行。
  @kernel 内的动态循环会编译成 GPU 循环；
  Constexpr 循环可能在编译期展开。
误区 2：Layout 就是 Row-major/Column-major。
  Layout 还能表达线程、Value、Tile、Fragment 和 Swizzle。
误区 3：partition_S 会复制数据。
  它只生成当前线程的 Tensor View；cute.copy 才搬数据。
误区 4：能用 cute.gemm 就自然高性能。
  TiledMma、Copy、Shared Layout、Pipeline 和 Shape 都必须匹配。
误区 5：JIT 时间就是 Kernel 时间。
  编译应独立缓存，性能测试只测编译后的调用。
```
### 推荐学习顺序
```text
1. Hello World:
   分清 @jit、@kernel、Constexpr 和动态值。
2. Vector Add:
   对照 CUDA 的 Grid/Block/Thread 和边界判断。
3. Layout:
   手算 Shape/Stride 到 Offset 的映射。
4. TiledCopy:
   理解 (thread, value) 到 Tile Coordinate。
5. Shared-Memory GEMM:
   对照熟悉的 CUDA Tiling。
6. TiledMma:
   理解 MMA Atom、Fragment 和 Accumulator。
7. Pipeline:
   最后再学习 cp.async、TMA、WGMMA、TMEM。
```
CuTe DSL 最难的地方不是 Python 语法，而是它要求你把“线程怎样拥有数据”写成 Layout。掌握下面这条主线，就能开始阅读真实 Kernel：
```text
Tensor
-> local_tile 取 CTA Tile
-> TiledCopy.partition_* 分给线程
-> cute.copy 搬到 Shared/Register
-> TiledMma.partition_* 形成 MMA Fragment
-> cute.gemm 更新 Accumulator
-> Epilogue 写回 Global Memory
```
参考资料：

- [CuTe DSL Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
- [CuTe DSL Programming Model](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)
- [CuTe DSL Code Generation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html)
- [CuTe Layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)
- [CuTe DSL Official Examples](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL)
