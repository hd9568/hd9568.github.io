---
title: '手写 CUDA Vector Add 与 Vector Dot Product'
description: '从最基础的向量加法开始，逐步写出 Dot Product 的并行规约版本，理解线程映射、边界检查和块级归约。'
category: 'CUDA'
pubDate: '2026-05-29'
updatedDate: '2026-05-29'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Vector Add](#二vector-add)
3. [Host 端启动方式](#三host-端启动方式)
4. [Vector Dot Product 的拆分思路](#四vector-dot-product-的拆分思路)
5. [块内规约版本](#五块内规约版本)
6. [优化要点](#六优化要点)
7. [面试回答模板](#七面试回答模板)
8. [总结](#八总结)

## 一、核心结论

Vector Add 和 Dot Product 是 CUDA 手写算子的入门题。

- Vector Add 是一线程处理一个元素，重点是 global id 和边界检查。
- Dot Product 需要先逐元素相乘，再并行求和。
- 求和不能让所有线程直接 `atomicAdd` 到同一个地址，否则竞争很重。
- 常见做法是每个 Block 先规约出一个 partial sum，再对 partial sum 做二次规约。
- 对大数组，访存合并比计算本身更重要。

## 二、Vector Add

Kernel：

```cpp
__global__ void vector_add(const float* a,
                           const float* b,
                           float* c,
                           int n) {
    // 全局线程编号：每个线程负责一个元素。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid 通常向上取整，线程数可能超过 n，因此必须检查边界。
    if (idx < n) {
        // 相邻线程访问相邻元素，Global Memory 访问容易合并。
        c[idx] = a[idx] + b[idx];
    }
}
```

这个 kernel 的特点：

- 每个元素独立，没有线程间同步。
- 算术强度低，通常是 Memory-bound。
- 优化重点是合并访问和减少额外读写。

## 三、Host 端启动方式

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;

vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
cudaDeviceSynchronize();
```

为什么要向上取整：

```cpp
(n + threads - 1) / threads
```

如果 `n = 1000`，`threads = 256`，需要 4 个 Block，总线程数 1024，多出的 24 个线程通过 `if (idx < n)` 被屏蔽。

## 四、Vector Dot Product 的拆分思路

Dot Product：

```text
sum = a[0] * b[0] + a[1] * b[1] + ... + a[n-1] * b[n-1]
```

并行化思路：

1. 每个线程计算若干个 `a[i] * b[i]`。
2. Block 内把这些乘积规约成一个 partial sum。
3. 每个 Block 写出一个 partial sum。
4. 再对 partial sum 求和。

## 五、块内规约版本

```cpp
__global__ void dot_product_partial(const float* a,
                                    const float* b,
                                    float* partial,
                                    int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < n) {
        // 每个线程先计算一个乘积。
        val = a[idx] * b[idx];
    }

    // 把每个线程的局部结果放入 shared memory。
    smem[tid] = val;
    __syncthreads();

    // 块内并行规约。stride 每次减半。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // 每个 block 输出一个 partial sum。
    if (tid == 0) {
        partial[blockIdx.x] = smem[0];
    }
}
```

启动：

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
size_t shared_bytes = threads * sizeof(float);

dot_product_partial<<<blocks, threads, shared_bytes>>>(d_a, d_b, d_partial, n);
```

`d_partial` 长度是 `blocks`。后续可以：

- 再启动一个 reduction kernel 规约 `d_partial`。
- 或者把 `partial` 拷回 Host 求和（只适合 blocks 很少的场景）。

## 六、优化要点

### 1. Grid-stride loop

如果 `n` 很大，或者希望每个线程处理多个元素：

```cpp
for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    val += a[i] * b[i];
}
```

这样可以减少 Block 数限制带来的问题，也提高每线程工作量。

### 2. Warp shuffle 优化尾部规约

Shared Memory 规约到 32 个线程后，可以用 `__shfl_down_sync` 做 Warp 内规约，减少 `__syncthreads()`。

### 3. 避免全局 atomicAdd 热点

```cpp
// 不推荐：所有线程竞争同一个地址。
atomicAdd(result, a[idx] * b[idx]);
```

大量线程对同一地址原子加，会严重串行化。更好的方式是 Block 先局部规约。

## 七、面试回答模板

如果问题是“怎么手写 Vector Add / Dot Product”，可以这样回答：

1. Vector Add 一线程处理一个元素，global id 是 `blockIdx.x * blockDim.x + threadIdx.x`。
2. 由于 Grid 向上取整，kernel 内要做 `idx < n` 边界检查。
3. Dot Product 先每线程计算乘积，再做并行规约。
4. 常见做法是每个 Block 输出 partial sum，再对 partial sum 二次规约。
5. 不建议所有线程直接 `atomicAdd` 到同一个全局结果，因为竞争开销很大。

## 八、总结

Vector Add 检查线程映射和访存合并，Dot Product 检查并行规约能力。这两个题虽然简单，但覆盖了 CUDA 手写 kernel 的基本功：线程编号、边界检查、shared memory、同步和规约。
