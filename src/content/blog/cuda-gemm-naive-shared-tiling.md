---
title: '手写 CUDA GEMM：朴素版、Shared Memory Tiling 与 Bank Conflict'
description: '从朴素矩阵乘法写起，逐步引入 Shared Memory Tiling，理解 GEMM 为什么需要数据复用以及如何处理 Bank Conflict。'
category: 'CUDA'
pubDate: '2026-06-22'
updatedDate: '2026-06-22'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [GEMM 定义](#二gemm-定义)
3. [朴素版本](#三朴素版本)
4. [为什么朴素版本慢](#四为什么朴素版本慢)
5. [Shared Memory Tiling](#五shared-memory-tiling)
6. [Tiling 版本代码](#六tiling-版本代码)
7. [Bank Conflict 与 Padding](#七bank-conflict-与-padding)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

GEMM 是 CUDA 手写算子核心题。

- GEMM 计算 `C = A * B`。
- 朴素版本每个线程计算一个 `C[row][col]`。
- 朴素版本会重复从 Global Memory 读取 A 和 B，数据复用差。
- Shared Memory Tiling 把 A、B 的小块缓存到 shared memory，多线程复用。
- Padding 可以缓解某些 shared memory Bank Conflict。
- 真正高性能 GEMM 还会涉及寄存器 tiling、向量化、双缓冲、Tensor Core。

## 二、GEMM 定义

矩阵形状：

```text
A: M x K
B: K x N
C: M x N
```

计算：

```text
C[row][col] = sum(A[row][k] * B[k][col]), k = 0..K-1
```

## 三、朴素版本

```cpp
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;

        // 每个线程独立计算一个 C[row][col]。
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}
```

启动：

```cpp
dim3 block(16, 16);
dim3 grid((N + block.x - 1) / block.x,
          (M + block.y - 1) / block.y);
gemm_naive<<<grid, block>>>(A, B, C, M, N, K);
```

## 四、为什么朴素版本慢

朴素版本中，相邻线程会重复读取 A 或 B。

例如同一个 Block 计算 `C` 的一个 tile：

- 同一行的多个线程会读取相同的 `A[row][k]`。
- 同一列的多个线程会读取相同的 `B[k][col]`。

这些数据本可以复用，但朴素版本反复访问 Global Memory，带宽压力大。

## 五、Shared Memory Tiling

Tiling 思路：

1. 把 A 的一个 tile 和 B 的一个 tile 从 Global Memory 读到 Shared Memory。
2. Block 内线程复用这两个 tile 做一部分累加。
3. 沿 K 维循环多个 tile，最终得到 C 的一个 tile。

```text
A tile: TILE x TILE
B tile: TILE x TILE
C tile: TILE x TILE
```

## 六、Tiling 版本代码

```cpp
#define TILE 16

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    // 沿 K 维分块。
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        // 加载 A tile，越界位置补 0。
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // 加载 B tile，越界位置补 0。
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // 等待整个 tile 加载完成。
        __syncthreads();

        // 使用 shared memory 中的 tile 做局部乘加。
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 确保本轮 tile 使用完成，再进入下一轮覆盖 shared memory。
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

## 七、Bank Conflict 与 Padding

如果访问 shared memory 的列方向，可能发生 Bank Conflict。可以加 padding：

```cpp
__shared__ float As[TILE][TILE + 1];
__shared__ float Bs[TILE][TILE + 1];
```

在上面的简单 GEMM 中，访问模式：

```cpp
As[ty][k]  // 同一行不同 k，通常较友好。
Bs[k][tx]  // 不同线程 tx 不同，通常也较友好。
```

但更复杂的 tile 布局、转置加载、Tensor Core layout 中，Bank Conflict 会更明显，Padding 或 Swizzle 会很重要。

## 八、面试回答模板

如果问题是“怎么手写 GEMM”，可以这样回答：

1. 朴素版是一线程计算一个 `C[row][col]`，循环 K 维累加。
2. 朴素版会重复从 Global Memory 读取 A 和 B，数据复用差。
3. 优化版使用 Shared Memory Tiling，把 A 和 B 的小块加载到 shared memory。
4. Block 内线程复用 tile，减少 Global Memory 访问。
5. 每轮 tile 加载后需要 `__syncthreads()`，避免读到未完成数据。
6. 进一步优化包括 padding 避免 Bank Conflict、寄存器 tiling、向量化、双缓冲、Tensor Core。

## 九、总结

GEMM 优化的核心是数据复用：

```text
Global Memory 读一次 -> Shared Memory 复用多次 -> 寄存器累加
```

朴素版用于理解映射，Shared Memory Tiling 用于理解复用，高性能版本则继续向寄存器 tiling 和 Tensor Core 演进。
