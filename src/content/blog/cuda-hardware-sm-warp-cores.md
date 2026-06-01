---
title: 'CUDA 硬件架构：SM、Warp、CUDA Core 与 Tensor Core'
description: '从 GPU 执行单元出发，理解 SM、Warp、CUDA Core、Tensor Core 的关系，以及它们为什么决定 CUDA kernel 的性能上限。'
category: 'CUDA'
pubDate: '2026-05-29T11:00:00+08:00'
updatedDate: '2026-05-29T11:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [GPU 为什么适合并行计算](#二gpu-为什么适合并行计算)
3. [SM 是什么](#三sm-是什么)
4. [Warp 是什么](#四warp-是什么)
5. [CUDA Core 与 Tensor Core](#五cuda-core-与-tensor-core)
6. [从 kernel 到硬件执行](#六从-kernel-到硬件执行)
7. [性能理解要点](#七性能理解要点)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

CUDA 程序的性能主要取决于大量线程如何被组织到 GPU 硬件上执行。

- SM（Streaming Multiprocessor）是 GPU 上主要的执行单元。
- 一个 GPU 有多个 SM，每个 SM 可以同时驻留多个 Block 和多个 Warp。
- Warp 是线程调度的基本单位，NVIDIA GPU 上通常是 32 个线程一组。
- CUDA Core 更适合执行标量 FP32/INT 等通用计算。
- Tensor Core 是矩阵乘加专用单元，适合 GEMM、卷积、Transformer 等张量计算。
- CUDA 优化的核心是让 SM 忙起来，同时减少访存、分支和同步浪费。

## 二、GPU 为什么适合并行计算

CPU 更擅长复杂控制流和低延迟任务，GPU 更擅长大规模数据并行。

```text
CPU: 少量强核心，复杂控制逻辑，大 Cache
GPU: 大量计算单元，高吞吐，适合相同操作处理大量数据
```

例如向量加法：

```cpp
// 每个元素的计算彼此独立，适合交给大量 GPU 线程并行处理。
c[i] = a[i] + b[i];
```

如果有一百万个元素，可以让大量线程分别处理不同下标。

## 三、SM 是什么

SM 可以理解为 GPU 内部的“并行计算车间”。一个 GPU 通常包含多个 SM，每个 SM 内部包含：

- Warp Scheduler：选择下一个要执行的 Warp。
- Register File：给线程保存寄存器。
- Shared Memory / L1 Cache：Block 内共享和缓存数据。
- CUDA Core：执行普通算术指令。
- Tensor Core：执行矩阵乘加指令。
- Load/Store Unit：处理访存指令。
- Special Function Unit：处理 sin、cos、rsqrt 等特殊函数。

一个 kernel 启动后，Grid 中的 Block 会被分配到不同 SM 上执行。

```text
Grid
+---------+---------+---------+
| Block 0 | Block 1 | Block 2 |  -> 分配到不同 SM
+---------+---------+---------+

SM 0: Block 0, Block 3, ...
SM 1: Block 1, Block 4, ...
SM 2: Block 2, Block 5, ...
```

Block 一旦被分配到某个 SM，通常会在该 SM 上执行到结束，不会中途迁移到别的 SM。

## 四、Warp 是什么

Warp 是 GPU 调度和执行线程的基本单位。NVIDIA GPU 上一个 Warp 通常包含 32 个线程。

```text
Block with 128 threads
+---------+---------+---------+---------+
| Warp 0  | Warp 1  | Warp 2  | Warp 3  |
| 32 thr  | 32 thr  | 32 thr  | 32 thr  |
+---------+---------+---------+---------+
```

虽然 CUDA 代码里写的是“每个线程执行一份代码”，但硬件调度时通常按 Warp 发射指令。一个 Warp 内的线程执行同一条指令，只是处理不同数据。

这也是 SIMT（Single Instruction, Multiple Threads）模型。

## 五、CUDA Core 与 Tensor Core

### CUDA Core

CUDA Core 执行通用标量计算，例如：

```cpp
float z = x * y + b;
```

这类操作适合普通 CUDA Core。

### Tensor Core

Tensor Core 面向小矩阵乘加，例如：

```text
D = A * B + C
```

它不是处理单个标量乘法，而是一次处理一小块矩阵乘加。深度学习中的 GEMM、卷积、Attention 都可以转化成矩阵乘加，因此 Tensor Core 对 AI Infra 很关键。

Tensor Core 常和这些数据类型相关：

- FP16 / BF16。
- TF32。
- INT8 / FP8（取决于硬件代际）。

### 两者差异

| 对比项 | CUDA Core | Tensor Core |
| --- | --- | --- |
| 计算形态 | 标量/向量指令 | 矩阵乘加指令 |
| 常见数据类型 | FP32、INT32 等 | FP16、BF16、TF32、INT8、FP8 等 |
| 典型场景 | 通用 kernel | GEMM、卷积、Attention |
| 性能特点 | 灵活 | 吞吐极高但要求数据布局和类型匹配 |

## 六、从 kernel 到硬件执行

一个简单 kernel：

```cpp
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    // 每个线程负责一个元素。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查避免越界访问。
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

启动：

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
add_kernel<<<blocks, threads>>>(a, b, c, n);
```

执行关系：

1. `blocks` 个 Block 组成一个 Grid。
2. 每个 Block 被调度到某个 SM。
3. Block 内线程被划分成多个 Warp。
4. SM 的 Warp Scheduler 选择 Warp 发射指令。
5. CUDA Core 或 Tensor Core 执行对应计算。

## 七、性能理解要点

### 1. SM 利用率

如果 Block 太少，很多 SM 没活干，GPU 利用率低。

### 2. Warp 执行效率

Warp 内线程如果分支严重，可能导致同一个 Warp 分多次执行不同路径。

### 3. 访存效率

多数 CUDA kernel 不是算力不够，而是数据从 Global Memory 取不够快。访存合并、共享内存复用、减少重复读写非常重要。

### 4. Tensor Core 不是自动使用

代码需要满足合适的数据类型、矩阵形状和库调用方式。例如 cuBLAS、CUTLASS、WMMA 才能更直接地用上 Tensor Core。

## 八、面试回答模板

如果问题是“SM、Warp、CUDA Core、Tensor Core 的关系”，可以这样回答：

1. GPU 由多个 SM 组成，SM 是主要执行单元。
2. Kernel 的 Block 会被调度到 SM 上执行，Block 内线程按 Warp 组织。
3. Warp 是调度基本单位，NVIDIA GPU 通常 32 个线程一个 Warp。
4. CUDA Core 执行普通标量计算，Tensor Core 执行矩阵乘加，适合深度学习算子。
5. CUDA 优化要关注 SM 是否充分占用、Warp 是否高效、访存是否合并、Tensor Core 是否被有效利用。

## 九、总结

CUDA 硬件架构可以按层次理解：

```text
GPU -> SM -> Warp -> Thread -> Instruction
```

SM 决定并行执行资源，Warp 决定调度粒度，CUDA Core 和 Tensor Core 决定计算方式。性能优化不是只看线程数量，而是让计算、访存和调度都尽量高效。
