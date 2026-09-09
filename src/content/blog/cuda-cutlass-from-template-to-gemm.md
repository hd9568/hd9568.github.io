---
title: 'CUTLASS 入门：从 CUDA Tiling 到可组合 GEMM Kernel'
description: '面向掌握基础 CUDA 的读者，从 C++ 模板、Layout 和 GEMM 分块开始，通过多个递进 Demo 讲清 CUTLASS 2.x Device API、3.x Collective API、Mainloop、Epilogue、流水线及调优方法。'
category: 'CUDA'
pubDate: '2026-09-08T10:00:00+08:00'
updatedDate: '2026-09-08T10:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---
## 目录
1. [CUTLASS 到底是什么](#一cutlass-到底是什么)
2. [读懂 CUTLASS 前需要的 C++ 模板知识](#二读懂-cutlass-前需要的-c-模板知识)
3. [Demo 1：调用最小 CUTLASS GEMM](#三demo-1调用最小-cutlass-gemm)
4. [CUTLASS 内部怎样执行 GEMM](#四cutlass-内部怎样执行-gemm)
5. [Demo 2：指定 Tensor Core、Tile 与 Epilogue](#五demo-2指定-tensor-coretile-与-epilogue)
6. [Demo 3：理解 CUTLASS 3.x Collective API](#六demo-3理解-cutlass-3x-collective-api)
7. [编译、验证、调优与学习路线](#七编译验证调优与学习路线)

> **版本说明**：本文以 CUTLASS `v4.7.1` 中仍受支持的 C++ API 为基线。CUTLASS 4.x 同时包含经典 2.x 风格 `device::Gemm`、3.x Collective API、CuTe C++ 和 Python DSL。它们属于同一仓库，但抽象层级不同。
## 一、CUTLASS 到底是什么
CUTLASS 是 NVIDIA 提供的开源 CUDA C++ 模板库，重点是高性能 GEMM 及其衍生算子。

GEMM 的完整形式为：
```text
D = alpha * A * B + beta * C
A: [M, K]
B: [K, N]
C: [M, N]
D: [M, N]
```
cuBLAS 与 CUTLASS 都能计算 GEMM，但使用方式不同：
```text
cuBLAS:
  调用 NVIDIA 已编译好的库函数。
  使用简单，但很难改写 Kernel 内部结构。
CUTLASS:
  在编译期组合数据类型、Layout、Tile、数据搬运、
  MMA、流水线和 Epilogue，生成应用自己的 CUDA Kernel。
```
因此 CUTLASS 不是一门新语言。它仍是 CUDA C++，只是把高性能 Kernel 中反复出现的组件封装成模板。
### 从手写 CUDA GEMM 到 CUTLASS
基础 CUDA Tiled GEMM 通常是：
```text
Global Memory
-> A/B Tile 搬到 Shared Memory
-> __syncthreads()
-> 每个线程在寄存器中累加
-> 写回 C
```
CUTLASS 继续细化这条路径：
```text
Problem
-> CTA Tile
-> Warp / Warpgroup Tile
-> MMA Instruction Tile
-> Register Accumulator
-> Epilogue
-> Global Memory
```
它替你处理的不是矩阵乘法公式，而是以下工程细节：

- 不同线程怎样协作搬运 Tile。
- Shared Memory 怎样布局以减少 Bank Conflict。
- Warp 怎样拥有 MMA Fragment。
- 如何使用 `mma.sync`、WGMMA 或 `tcgen05.mma`。
- 如何双缓冲并重叠数据搬运与计算。
- 边界 Tile 如何 Predication。
- Accumulator 怎样重排后合并写回。

可以把 CUTLASS 理解为：

> 一套把 GEMM 算法映射到 NVIDIA GPU 层级结构的可组合零开销抽象。

“零开销”不是说 Kernel 没有成本，而是模板抽象在编译期展开，运行时不会因为 C++ 类和模板多出虚函数或解释器开销。
## 二、读懂 CUTLASS 前需要的 C++ 模板知识
CUTLASS 代码长，主要因为大量选择被编码进“类型”。先看一个缩小版例子。
### Demo 0：类型也能作为参数
```cpp
#include <cstdio>
#include <type_traits>
struct Simt {};
struct TensorOp {};
template <typename MathTag, int TileM, int TileN>
struct KernelConfig {
  static void print() {
    if constexpr (std::is_same_v<MathTag, TensorOp>) {
      std::printf("Tensor Core, tile = %d x %d\n", TileM, TileN);
    } else {
      std::printf("CUDA Core, tile = %d x %d\n", TileM, TileN);
    }
  }
};
using MyKernel = KernelConfig<TensorOp, 128, 128>;
int main() {
  MyKernel::print();
}
```
这里有四个关键概念。
### `template`：把配置变成编译期输入
```cpp
template <typename MathTag, int TileM, int TileN>
```
表示模板接收一个类型和两个整数。编译器看到：
```cpp
KernelConfig<TensorOp, 128, 128>
```
会为这组配置实例化具体代码。`TileM/TileN` 已是编译期常量，循环展开、数组大小和指令选择都可以据此优化。
### `using`：给复杂类型取别名
```cpp
using MyKernel = KernelConfig<TensorOp, 128, 128>;
```
CUTLASS 中几十行 `using` 通常不是在执行计算，而是在逐层组装 Kernel 类型。
### Tag：空类型也能表达选择
`TensorOp` 没有字段，却能告诉模板选择 Tensor Core 路径。CUTLASS 常见 Tag 包括：
```cpp
cutlass::arch::OpClassSimt
cutlass::arch::OpClassTensorOp
cutlass::arch::Sm80
cutlass::layout::RowMajor
```
它们把“选择什么实现”放进类型系统。
### `if constexpr`：只保留命中的分支
普通 `if` 两条分支都必须是合法运行时代码；`if constexpr` 在编译期选择分支，未选分支不会进入最终 Kernel。

这也是 CUTLASS 编译慢、报错长的原因：编译器需要实例化深层模板。但收益是，运行时得到的是已经特化好的 CUDA 代码。
## 三、Demo 1：调用最小 CUTLASS GEMM
先不自己配置 Tile，直接使用 `cutlass::gemm::device::Gemm` 的默认值。
```cpp
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
using ColumnMajor = cutlass::layout::ColumnMajor;
using CutlassGemm = cutlass::gemm::device::Gemm<
    float, ColumnMajor,   // A 的元素类型和布局
    float, ColumnMajor,   // B
    float, ColumnMajor>;  // C、D
cutlass::Status run_gemm(
    int M, int N, int K,
    float const* A,
    float const* B,
    float const* C,
    float* D) {
  int lda = M;
  int ldb = K;
  int ldc = M;
  int ldd = M;
  float alpha = 1.0f;
  float beta = 0.0f;
  CutlassGemm gemm;
  CutlassGemm::Arguments args(
      {M, N, K},
      {A, lda},
      {B, ldb},
      {C, ldc},
      {D, ldd},
      {alpha, beta});
  return gemm(args);
}
```
这是一个 Host 函数。代码中没有手写 `<<<grid, block>>>`，但 `gemm(args)` 内部最终仍会启动 CUDA Kernel。
### 每个参数对应什么
```text
{M, N, K}
  GEMM 的逻辑形状。
{A, lda}
  A 的设备指针和 Leading Dimension。
{alpha, beta}
  Epilogue 中 D = alpha * accumulator + beta * C。
```
Column-major 的地址公式为：
```text
offset(row, col) = row + col * leading_dimension
```
所以：
```text
A[M,K] 的 lda = M
B[K,N] 的 ldb = K
C[M,N] 的 ldc = M
```
如果 Layout 和 `ld*` 不匹配，程序通常不会崩溃，而是得到错误矩阵。这是学习 CUTLASS 时最常见的正确性问题之一。
### 这段代码何时决定什么
编译期决定：
```text
float
ColumnMajor
默认 CTA/Warp Tile
默认 Mainloop 和 Epilogue
目标架构对应实现
```
运行时决定：
```text
M、N、K
A/B/C/D 指针
Leading Dimension
alpha、beta
CUDA Stream
```
因此一份已编译 Kernel 可以处理多个 `M/N/K`，但不能突然把 `float` 改成 `half`，因为数据类型属于模板参数。
### 编译
假设 CUTLASS 仓库位于 `$CUTLASS_DIR`：
```bash
nvcc -std=c++17 -O3 \
  -I"$CUTLASS_DIR/include" \
  -arch=sm_80 \
  demo.cu -o demo
```
CUTLASS 大部分是 Header-only。`nvcc` 在编译你的 `.cu` 时实例化模板，并把选中的设备代码编译进可执行文件。
## 四、CUTLASS 内部怎样执行 GEMM
假设：
```text
M = N = K = 256
CTA Tile  = 128 x 128 x 32
Warp Tile =  64 x  64 x 32
MMA Tile  =  16 x   8 x 16
```
### CTA 如何覆盖输出
一个 CTA 计算一个 `128 x 128` 的 D Tile：
```text
grid_m = ceil(256 / 128) = 2
grid_n = ceil(256 / 128) = 2
总 CTA 数 = 2 x 2 = 4
```
每个 CTA 沿 K 维循环：
```text
K iterations = 256 / 32 = 8
```
逻辑过程：
```text
for each CTA output tile (cta_m, cta_n):
    accumulator = 0
    for cta_k in [0, 32, ..., 224]:
        load A[cta_m : cta_m+128, cta_k : cta_k+32]
        load B[cta_k : cta_k+32, cta_n : cta_n+128]
        accumulator += A_tile @ B_tile
    D_tile = alpha * accumulator + beta * C_tile
```
### Warp 和 MMA 如何继续切分
一个 `128 x 128` CTA Tile 可以由 4 个 Warp Tile 覆盖：
```text
Warp 0 -> C[0:64,    0:64]
Warp 1 -> C[0:64,   64:128]
Warp 2 -> C[64:128,  0:64]
Warp 3 -> C[64:128, 64:128]
```
每个 Warp 再发出多条 `16 x 8 x 16` MMA 指令。MMA 指令不是一个线程计算整个小矩阵，而是一个 Warp 的线程共同持有 A/B Fragment 和 Accumulator Fragment。

层级可以记成：
```text
Grid:
  把整个 [M,N] 输出分给 CTA
CTA:
  从 Global Memory 搬 A/B Tile 到 Shared Memory
Warp:
  从 Shared Memory 取自己的 Fragment
MMA:
  用 Tensor Core 更新 Register Accumulator
```
### Mainloop 与 Epilogue
CUTLASS 把 GEMM 分成两大阶段。

Mainloop：
```text
沿 K 维循环
-> 搬运 A/B
-> 执行 MMA
-> 结果累加在寄存器
```
Epilogue：
```text
读取寄存器 Accumulator
-> 与 C、alpha、beta 或 Bias 等组合
-> 重排为合并访存
-> 写回 D
```
Epilogue 独立存在有两个原因：

1. MMA 的寄存器 Fragment 布局服务于计算，不一定适合 Global Memory 合并写。
2. `Bias + ReLU` 等操作可以在写回前融合，避免额外 Kernel 和显存往返。
## 五、Demo 2：指定 Tensor Core、Tile 与 Epilogue
下面展示 CUTLASS 2.x 风格的完整配置骨架。它比 Demo 1 多做一件事：不再依赖默认配置。
```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using Accumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    Accumulator,
    float>;
using TensorCoreGemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    Accumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,  // CTA Tile
    cutlass::gemm::GemmShape<64, 64, 32>,    // Warp Tile
    cutlass::gemm::GemmShape<16, 8, 16>,     // MMA Tile
    Epilogue>;
```
这段 `using` 本身不执行 GEMM，而是生成一个具体的 Kernel 类型。
### 为什么输入 FP16、累加 FP32
Tensor Core 读取 FP16：
```text
A/B storage = FP16
```
但 K 维有很多次乘加。如果也用 FP16 Accumulator，舍入误差会快速累积，因此常用：
```text
FP16 x FP16 -> FP32 accumulate
```
最后 Epilogue 再按 `ElementC` 转换输出。
### Alignment 为什么重要
Tensor Core Kernel 常使用 128-bit 向量化 Load：
```text
128 bits / 16 bits per FP16 = 8 elements
```
因此指针、Leading Dimension 和 K 维通常要满足相应对齐。模板实例可以编译成功，但运行时 Shape 不满足 `can_implement()` 时应拒绝执行：
```cpp
TensorCoreGemm gemm;
TensorCoreGemm::Arguments args = /* ... */;
if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
  // 改用更弱对齐的 Kernel，或对输入 Padding。
}
```
不要把对齐只理解成“指针地址是 16 的倍数”。对二维 Tensor，下一行的起始地址也由 Leading Dimension 决定。
### 为什么 Tile 不能越大越好
增大 CTA Tile 可以提高数据复用，但也会增加：
```text
Shared Memory
Register Accumulator
每 CTA 的 Warp 数
边界浪费
```
寄存器和 Shared Memory 过多时，每个 SM 同时驻留的 CTA 变少。最终性能取决于数据复用、并行 CTA 数、流水线深度和问题形状的平衡。
## 六、Demo 3：理解 CUTLASS 3.x Collective API
Hopper 以后，数据搬运和计算方式明显复杂：TMA、WGMMA、Warp Specialization、Threadblock Cluster 都需要组合。CUTLASS 3.x 因而把 GEMM 拆成两个 Collective。
```text
CollectiveMainloop:
  A/B 如何从 Global Memory 到 Shared Memory
  如何形成流水线
  使用哪种 MMA
CollectiveEpilogue:
  Accumulator 如何转换、融合和写回
```
下面是官方 Hopper Quick Start 的核心骨架：
```cpp
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
using namespace cute;
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using Accumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
using ArchTag       = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape     = Shape<_128, _128, _64>;
using ClusterShape  = Shape<_1, _2, _1>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    Accumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```
`CollectiveBuilder` 根据这些约束选择合法实现。`Auto` 不是运行时自动调优，而是编译期规则选择。

然后组装 Epilogue 和完整 Kernel：
```cpp
using Epilogue = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 1, Accumulator, Accumulator>>;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    Mainloop,
    Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```
最后的职责链为：
```text
GemmUniversalAdapter
  Host 侧参数检查、Workspace、初始化和 Launch
GemmUniversal
  Grid/CTA 级 Kernel 骨架和 Tile Scheduler
Mainloop
  A/B 搬运、Pipeline、MMA
Epilogue
  alpha/beta、类型转换和 D 写回
```
### 为什么 3.x 代码反而更长
2.x 的 `device::Gemm` 隐藏了大部分默认选择，适合直接调用。3.x 暴露了 Mainloop 与 Epilogue 的组合边界，适合：

- 更换 TMA/WGMMA Schedule。
- 使用 Persistent Kernel。
- 配置 Threadblock Cluster。
- 自定义融合 Epilogue。
- 在新架构上复用数据搬运与计算组件。

初学时应先理解 2.x 调用和内部层级，再读 3.x Builder。直接从数十个模板参数开始，容易只记住类型名而没有执行模型。
## 七、编译、验证、调优与学习路线
### 先用 Profiler 找到可用 Kernel
构建：
```bash
git clone --branch v4.7.1 https://github.com/NVIDIA/cutlass.git
cd cutlass
cmake -S . -B build \
  -DCUTLASS_NVCC_ARCHS=80 \
  -DCUTLASS_ENABLE_TESTS=OFF
cmake --build build --target cutlass_profiler -j
```
搜索并测试 GEMM：
```bash
./build/tools/profiler/cutlass_profiler \
  --operation=Gemm \
  --m=4096 --n=4096 --k=4096 \
  --A=f16:row --B=f16:column --C=f32:row
```
Profiler 会验证结果并输出 Kernel 名、Tile、Stage 和耗时。实践中应先让 Profiler 找到高性能配置，再把相近配置写入代码，而不是靠直觉猜 Tile。
### 正确测量
至少检查：
```text
1. 与 cuBLAS 或 CPU/PyTorch Reference 比较数值。
2. Warmup 后用 CUDA Event 计时。
3. 计时区间外完成内存分配和初始化。
4. 使用实际业务 Shape，而不只测 4096^3。
5. 分开检查 Kernel 时间和端到端时间。
```
性能不佳时依次检查：
```text
是否真的选中 Tensor Core Kernel
M/N/K 与指针是否满足 Alignment
CTA 数是否足够占满 GPU
Tile 是否与长宽比匹配
Shared Memory/Register 是否限制 Occupancy
Epilogue 是否成为瓶颈
是否需要 Split-K 或 Persistent Schedule
```
### CUTLASS 应放在什么位置
| 目标 | 优先选择 |
| --- | --- |
| 只需要标准 GEMM | cuBLAS/cuBLASLt |
| 需要已实现的 CUTLASS Kernel | `device::Gemm` 或 Operator API |
| 要组合 Hopper/Blackwell Mainloop | CUTLASS 3.x Collective |
| 要直接控制 Layout、Copy、MMA | [CuTe C++ / CuTe DSL](/blog/cuda-cute-dsl-from-layout-to-gemm/) |
| 要快速写普通逐元素或归约 Kernel | CUDA C++ 或 Triton 通常更直接 |

CUTLASS 最重要的学习成果不是记住模板签名，而是建立以下执行模型：
```text
GEMM Problem
-> 用 CTA Tile 覆盖 [M,N]
-> 沿 K 维分 Stage
-> Global Memory 到 Shared Memory
-> Shared Memory 到 Register Fragment
-> Warp/Warpgroup 发出 MMA
-> Accumulator 进入 Epilogue
-> 合并写回 Global Memory
```
能沿这条路径解释每个模板参数，才算真正读懂了 CUTLASS。
进一步阅读真实推理框架中的专用实现：[LMDeploy Grouped GEMM 源码详解：从 C++ 调度到 CUTLASS WGMMA](/blog/cuda-lmdeploy-cutlass-grouped-gemm-source/)。


- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CUTLASS C++ Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html)
- [CUTLASS GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)
- [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [CUTLASS 3.x Design](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html)
