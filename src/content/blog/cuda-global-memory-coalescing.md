---
title: 'CUDA Global Memory：合并访问、对齐与带宽优化'
description: '理解 Global Memory 的高延迟特性，讲清 Coalesced Memory Access、对齐要求和常见访存优化写法。'
category: 'CUDA'
pubDate: '2026-05-29'
updatedDate: '2026-05-29'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Global Memory 的特点](#二global-memory-的特点)
3. [合并访问是什么](#三合并访问是什么)
4. [连续访问示例](#四连续访问示例)
5. [非合并访问问题](#五非合并访问问题)
6. [对齐与向量化访问](#六对齐与向量化访问)
7. [优化 checklist](#七优化-checklist)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

Global Memory 容量大，但延迟高，是 CUDA 性能优化中最常见瓶颈。

- Global Memory 是 GPU 显存，所有线程都可以访问。
- 合并访问指一个 Warp 内线程访问连续、对齐的地址，硬件能用更少内存事务完成访问。
- 不连续、跨步很大、乱序访问会降低带宽利用率。
- 访问模式通常比单个线程里的指令数量更重要。
- 性能优化目标是让一个 Warp 的访存尽量连续、对齐、少重复。

## 二、Global Memory 的特点

Global Memory 可以理解为 GPU 的主存。

```text
Thread -> L1/L2 Cache -> Global Memory
```

它的特点：

- 容量大。
- 延迟高。
- 带宽高但需要合适访问模式才能打满。
- 所有 Block 都能访问。

很多 kernel 是 Memory-bound：算术操作不多，主要时间花在读写显存。

## 三、合并访问是什么

NVIDIA GPU 以 Warp 为单位组织访存。一个 Warp 里 32 个线程如果访问连续地址，硬件可以把这些访问合并成少量内存事务。

理想访问：

```text
lane 0 -> x[0]
lane 1 -> x[1]
lane 2 -> x[2]
...
lane31 -> x[31]
```

这种模式对 `float` 来说是 32 个连续 4 字节，共 128 字节，通常很适合硬件合并。

差访问：

```text
lane 0 -> x[0]
lane 1 -> x[1024]
lane 2 -> x[2048]
...
```

每个 lane 访问相隔很远，内存事务变多，带宽利用率下降。

## 四、连续访问示例

```cpp
__global__ void copy_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 相邻线程访问相邻元素，通常能形成合并访问。
        output[idx] = input[idx];
    }
}
```

如果 `blockDim.x` 是 32 的倍数，Warp 内线程的 `idx` 通常连续。这个访问模式对 Global Memory 友好。

## 五、非合并访问问题

### 跨步访问

```cpp
__global__ void strided_copy(const float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 相邻线程访问相隔 stride 的元素。
        output[idx] = input[idx * stride];
    }
}
```

如果 `stride` 很大，一个 Warp 会访问很多分散地址，导致内存事务数量增加。

### 矩阵按列访问

C/C++ 行主序矩阵中，连续的是同一行的相邻列。

```cpp
// 好：相邻线程访问同一行相邻列。
int row = blockIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
y[row * width + col] = x[row * width + col];
```

如果让相邻线程访问不同行同一列，地址会相隔 `width`，容易不合并。

## 六、对齐与向量化访问

对齐指访问地址最好落在硬件喜欢的边界上。例如 4 字节 `float` 按 4 字节对齐，`float4` 按 16 字节对齐。

向量化访问可以一次读写多个元素：

```cpp
__global__ void copy_float4(const float4* input, float4* output, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n4) {
        // 每个线程一次拷贝 4 个 float，要求地址满足 float4 对齐。
        output[idx] = input[idx];
    }
}
```

向量化可能减少指令数，但前提是：

- 数据地址对齐。
- 数据长度能按向量宽度处理。
- 尾部元素需要单独处理。

## 七、优化 checklist

- 相邻线程是否访问相邻地址。
- Block 线程数是否是 32 的倍数。
- 是否存在跨步访问或转置式访问。
- 是否能调整数据布局，让读写更连续。
- 是否存在重复读取，可否用 Shared Memory 复用。
- 是否能使用 `__restrict__` 帮助编译器优化别名分析。
- 是否能用 `float2/float4` 做安全向量化访问。

## 八、面试回答模板

如果问题是“什么是合并访问”，可以这样回答：

1. 合并访问是指一个 Warp 内线程访问连续且对齐的 Global Memory 地址。
2. 硬件可以把多个线程的访问合并成更少内存事务，提高带宽利用率。
3. 相邻线程访问相邻元素通常是合并访问，跨步或随机访问会破坏合并。
4. CUDA 访存优化的核心是让 Warp 级访问连续、对齐、少重复。
5. 对矩阵类 kernel，要特别注意行主序/列主序和线程映射方式。

## 九、总结

Global Memory 优化的第一原则是看 Warp 的访存模式，而不是只看单个线程代码。相邻线程访问连续数据，通常比每个线程写很复杂的局部优化更重要。
