---
title: '手写 CUDA Parallel Reduction：从交错寻址到 Warp 级规约'
description: '系统拆解 CUDA 并行规约，从朴素交错寻址、连续寻址、Shared Memory 到 Warp Shuffle 和循环展开。'
category: 'CUDA'
pubDate: '2026-06-21'
updatedDate: '2026-06-21'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Reduction 要解决什么](#二reduction-要解决什么)
3. [交错寻址版本](#三交错寻址版本)
4. [连续寻址版本](#四连续寻址版本)
5. [一次加载两个元素](#五一次加载两个元素)
6. [Warp 级规约](#六warp-级规约)
7. [完全展开思路](#七完全展开思路)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

Parallel Reduction 是 CUDA 高频手撕题，核心是把很多元素并行求和。

- 朴素版本容易出现线程利用率低、分支和 Bank Conflict。
- 连续寻址规约比交错寻址更友好。
- 每个线程加载多个元素可以减少 Block 数和同步次数。
- Warp 内规约可以用 `__shfl_down_sync`，避免 shared memory 和 `__syncthreads()`。
- 高性能版本通常结合连续寻址、展开、Warp shuffle 和多阶段规约。

## 二、Reduction 要解决什么

目标：

```text
out = x[0] + x[1] + ... + x[n-1]
```

并行思路：

```text
全局数组 -> 每个 Block 求 partial sum -> partial sums 再规约
```

## 三、交错寻址版本

这是容易理解但不够高效的版本。

```cpp
__global__ void reduce_interleaved(const float* input, float* partial, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    smem[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // stride 从 1 开始，每轮活跃线程变少。
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = smem[0];
    }
}
```

问题：

- `%` 取模开销较高。
- 活跃线程分布不连续，容易造成分支和访存不友好。
- 后期大量线程闲置。

## 四、连续寻址版本

更常见写法是 stride 从一半开始减半。

```cpp
__global__ void reduce_contiguous(const float* input, float* partial, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    smem[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 活跃线程连续：tid < stride。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = smem[0];
    }
}
```

优点：

- 活跃线程连续。
- 分支更简单。
- Shared Memory 访问更规整。

## 五、一次加载两个元素

每个线程先加两个元素，可以减少一半 Block 数。

```cpp
__global__ void reduce_two_loads(const float* input, float* partial, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;

    float sum = 0.0f;
    if (idx < n) {
        sum += input[idx];
    }
    if (idx + blockDim.x < n) {
        sum += input[idx + blockDim.x];
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // 最后一个 warp 用 shuffle 规约。
    float val = smem[tid];
    if (tid < 32) {
        unsigned mask = 0xffffffff;
        val += __shfl_down_sync(mask, val, 16);
        val += __shfl_down_sync(mask, val, 8);
        val += __shfl_down_sync(mask, val, 4);
        val += __shfl_down_sync(mask, val, 2);
        val += __shfl_down_sync(mask, val, 1);
    }

    if (tid == 0) {
        partial[blockIdx.x] = val;
    }
}
```

## 六、Warp 级规约

独立函数：

```cpp
__inline__ __device__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffff;
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);
    return val;
}
```

它的特点：

- Warp 内线程交换寄存器值。
- 不需要 shared memory。
- 不需要 `__syncthreads()`，因为 Warp 内同步执行。

## 七、完全展开思路

如果 `blockDim.x` 固定为 256，可以手写展开：

```cpp
if (blockDim.x >= 256) {
    if (tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();
}
if (blockDim.x >= 128) {
    if (tid < 64) smem[tid] += smem[tid + 64];
    __syncthreads();
}
```

完全展开减少循环控制开销，但会增加代码体积。实际高性能库通常用模板参数控制 Block 大小，在编译期生成展开版本。

## 八、面试回答模板

如果问题是“怎么优化 Parallel Reduction”，可以这样回答：

1. 先让每个 Block 对一段数据做局部规约，输出 partial sum。
2. 交错寻址版本简单但有取模、分支和线程利用率问题。
3. 连续寻址版本让活跃线程连续，访存和分支更友好。
4. 每个线程加载两个或多个元素，减少 Block 数和同步次数。
5. 最后一个 Warp 用 `__shfl_down_sync` 做规约，避免 shared memory 和同步。
6. 固定 Block 大小时可以循环展开，但要注意寄存器和代码体积。

## 九、总结

Reduction 的优化路线很清晰：

```text
朴素 shared memory -> 连续寻址 -> 多元素加载 -> Warp shuffle -> 展开
```

它考察的不只是代码，而是对线程利用率、同步成本、shared memory 和 Warp 原语的综合理解。
