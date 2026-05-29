---
title: 'CUDA Registers：寄存器使用、Spilling 与 Local Memory'
description: '理解寄存器为什么影响 Occupancy，讲清 Register Spilling 如何把变量溢出到 Local Memory 并拖慢 kernel。'
category: 'CUDA'
pubDate: '2026-05-29'
updatedDate: '2026-05-29'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [寄存器是什么](#二寄存器是什么)
3. [寄存器与 Occupancy](#三寄存器与-occupancy)
4. [Register Spilling 是什么](#四register-spilling-是什么)
5. [Local Memory 的误区](#五local-memory-的误区)
6. [如何观察和优化](#六如何观察和优化)
7. [面试回答模板](#七面试回答模板)
8. [总结](#八总结)

## 一、核心结论

寄存器是线程私有、速度最快的存储资源，但数量有限。

- 每个线程都有自己的寄存器。
- 单线程使用寄存器越多，一个 SM 能同时驻留的线程或 Warp 可能越少。
- 寄存器不够时，编译器会把部分变量溢出到 Local Memory。
- Local Memory 名字里有 local，但物理上通常在 Global Memory 中，延迟很高。
- 优化目标不是盲目减少寄存器，而是在寄存器使用、Occupancy 和指令数之间平衡。

## 二、寄存器是什么

寄存器用于保存线程的局部变量和中间结果。

```cpp
__global__ void saxpy(const float* x, float* y, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // idx、vx、vy 等通常会放在寄存器中。
        float vx = x[idx];
        float vy = y[idx];
        y[idx] = a * vx + vy;
    }
}
```

寄存器访问很快，不需要显式声明，也不需要 `__syncthreads()`。

## 三、寄存器与 Occupancy

一个 SM 的寄存器总数有限。假设：

```text
SM total registers = fixed budget
registers per thread = R
threads resident on SM <= total_registers / R
```

如果每个线程用很多寄存器，SM 上能同时驻留的 Warp 数会下降，Occupancy 下降。

但 Occupancy 不是越高越好：

- Occupancy 太低，可能无法隐藏访存延迟。
- 为了提高 Occupancy 强行减少寄存器，可能增加指令或 spilling，反而变慢。

## 四、Register Spilling 是什么

如果编译器发现寄存器不够，会把某些变量放到 Local Memory。

例如大量局部数组或变量可能导致 spilling：

```cpp
__global__ void heavy_kernel(float* out, const float* in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 局部数组如果索引不能在编译期确定，可能无法完全放进寄存器。
    float tmp[64];

    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        tmp[i] = in[idx + i];
    }

    float sum = 0.0f;
    for (int i = 0; i < 64; ++i) {
        sum += tmp[i];
    }
    out[idx] = sum;
}
```

如果 `tmp` 溢出到 Local Memory，每次访问都可能变成高延迟内存访问。

## 五、Local Memory 的误区

Local Memory 听起来像“本地高速内存”，但在 CUDA 中它通常表示线程私有的内存空间，物理上常位于 Global Memory。

常见导致 Local Memory 的情况：

- 寄存器溢出。
- 过大的线程局部数组。
- 不能静态解析索引的局部数组。
- 大结构体局部变量。

因此，看到 Local Memory 访问要警惕，它可能意味着性能问题。

## 六、如何观察和优化

### 1. 查看编译信息

```bash
nvcc -Xptxas -v kernel.cu -o kernel
```

可能看到类似信息：

```text
ptxas info : Used 64 registers, 0 bytes spill stores, 0 bytes spill loads
```

重点看：

- Used registers。
- spill stores。
- spill loads。

### 2. 用 Nsight Compute

关注指标：

- Registers Per Thread。
- Achieved Occupancy。
- Local Memory Load/Store。
- Stall 原因。

### 3. 减少不必要局部变量生命周期

```cpp
// 不要让很多临时变量同时活着。
float a = ...;
float b = ...;
float c = a + b;
```

可以把计算拆成更短生命周期，帮助编译器回收寄存器。

### 4. 控制展开程度

循环完全展开可能增加寄存器压力。`#pragma unroll` 不是一定越多越好。

```cpp
#pragma unroll 4
for (int i = 0; i < 32; ++i) {
    sum += x[i];
}
```

### 5. 使用 launch_bounds 或 maxrregcount 要谨慎

```cpp
__launch_bounds__(256, 2)
__global__ void kernel(...) {
}
```

或者：

```bash
nvcc --maxrregcount=64 kernel.cu
```

这些方式可以限制寄存器使用，但如果限制过严，可能导致更多 spilling。

## 七、面试回答模板

如果问题是“寄存器溢出为什么慢”，可以这样回答：

1. 寄存器是线程私有的高速存储，但每个 SM 的寄存器总量有限。
2. 如果单线程使用寄存器太多，会降低 SM 上可驻留的 Warp 数，影响 Occupancy。
3. 如果寄存器不够，编译器会把部分变量 spill 到 Local Memory。
4. Local Memory 虽然名字叫 local，但通常在 Global Memory 中，访问延迟高。
5. 优化时要结合 ptxas、Nsight Compute，看寄存器数、spill loads/stores 和实际性能。

## 八、总结

寄存器优化的关键是平衡：

- 寄存器多：减少重复访存和指令，但可能降低 Occupancy。
- 寄存器少：Occupancy 可能提高，但可能增加 spilling。
- 最终应通过 profiling 决定，而不是只追求某个单一指标。
