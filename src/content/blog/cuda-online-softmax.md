---
title: '手写 CUDA Softmax：Online Softmax 与数值稳定性'
description: '从数值稳定 Softmax 写法出发，理解 Online Softmax 如何减少访存，并给出单行 Softmax kernel 的清晰实现思路。'
category: 'CUDA'
pubDate: '2026-05-29T13:20:00+08:00'
updatedDate: '2026-05-29T13:20:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Softmax 的数值稳定写法](#二softmax-的数值稳定写法)
3. [朴素三阶段 Softmax](#三朴素三阶段-softmax)
4. [Online Softmax 的思想](#四online-softmax-的思想)
5. [单 Block 处理一行的示例](#五单-block-处理一行的示例)
6. [性能优化要点](#六性能优化要点)
7. [面试回答模板](#七面试回答模板)
8. [总结](#八总结)

## 一、核心结论

Softmax 是 Attention 中非常核心的算子。

- 直接计算 `exp(x)` 可能溢出，需要先减去最大值。
- 稳定 Softmax 通常需要求 max、求 sum、再归一化。
- 朴素实现会多次读写 Global Memory。
- Online Softmax 可以边扫描边更新 max 和 sum，减少访存或适配分块场景。
- 高性能 Softmax 通常结合 Warp/Block 规约、向量化访问和寄存器缓存。

## 二、Softmax 的数值稳定写法

定义：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

如果 `x_i` 很大，`exp(x_i)` 可能溢出。稳定写法：

```text
m = max(x)
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
```

减去最大值不改变结果，因为分子分母同时除以 `exp(m)`。

## 三、朴素三阶段 Softmax

对一行长度为 `cols` 的数据：

1. 求最大值 `max_val`。
2. 求 `sum(exp(x_i - max_val))`。
3. 写出 `exp(x_i - max_val) / sum`。

朴素伪代码：

```cpp
for each row:
    max_val = max(x[row])
    denom = sum(exp(x[row][i] - max_val))
    y[row][i] = exp(x[row][i] - max_val) / denom
```

问题是同一行数据至少被读两遍，输出再写一遍。对于长序列，访存成本很明显。

## 四、Online Softmax 的思想

Online Softmax 在扫描元素时同时维护当前最大值 `m` 和归一化分母 `d`。

假设已有前缀的：

```text
m_old = max(previous values)
d_old = sum(exp(x_i - m_old))
```

新来一个值 `x`：

```text
m_new = max(m_old, x)
d_new = d_old * exp(m_old - m_new) + exp(x - m_new)
```

为什么要乘 `exp(m_old - m_new)`：

- 旧的 `d_old` 是基于 `m_old` 归一化的。
- 如果最大值变成 `m_new`，旧分母要换到新的基准。

这让 max 和 sum 可以在一次扫描中更新。

## 五、单 Block 处理一行的示例

下面示例适合一行长度不太大、一个 Block 处理一行的教学版本。

```cpp
__inline__ __device__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffff;
    v = fmaxf(v, __shfl_down_sync(mask, v, 16));
    v = fmaxf(v, __shfl_down_sync(mask, v, 8));
    v = fmaxf(v, __shfl_down_sync(mask, v, 4));
    v = fmaxf(v, __shfl_down_sync(mask, v, 2));
    v = fmaxf(v, __shfl_down_sync(mask, v, 1));
    return v;
}

__inline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffff;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__global__ void softmax_one_block_per_row(const float* x, float* y, int rows, int cols) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    // 每个线程处理多个列元素，先求自己的局部最大值。
    float local_max = -INFINITY;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_x[col]);
    }

    // 先做 warp 内 max。
    float warp_max = warp_reduce_max(local_max);

    // 每个 warp 的 lane 0 把结果写到 shared memory。
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) {
        smem[warp_id] = warp_max;
    }
    __syncthreads();

    // 第一个 warp 规约所有 warp 的结果。
    float block_max = -INFINITY;
    if (warp_id == 0) {
        block_max = (tid < (blockDim.x + 31) / 32) ? smem[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
    }

    // 广播 block_max。
    if (tid == 0) smem[0] = block_max;
    __syncthreads();
    block_max = smem[0];

    // 求 exp 和 sum。这里把 exp 结果先写到 y，避免第三次读 x。
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = expf(row_x[col] - block_max);
        row_y[col] = v;
        local_sum += v;
    }

    float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        smem[warp_id] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (tid < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }

    if (tid == 0) smem[0] = block_sum;
    __syncthreads();
    block_sum = smem[0];

    // 归一化输出。
    for (int col = tid; col < cols; col += blockDim.x) {
        row_y[col] = row_y[col] / block_sum;
    }
}
```

这个版本便于理解，但不是最终极性能版本。长序列 Attention 中通常还要结合分块、Online Softmax 和 FlashAttention 思路。

## 六、性能优化要点

- 对每行做 max 和 sum 规约。
- 使用 Warp shuffle 减少 shared memory 和同步开销。
- 对较短行，一个 Block 处理一行较简单。
- 对很长行，需要多 Block 协作或分块 Online Softmax。
- 减少 Global Memory 往返，尽量把中间值留在寄存器或 shared memory。
- 使用近似 `exp` 或向量化时要注意精度要求。

## 七、面试回答模板

如果问题是“Softmax 为什么要减 max”，可以回答：

1. 直接 `exp(x)` 可能溢出，减去最大值可以保证指数输入不大于 0。
2. Softmax 对整体平移不敏感，减 max 不改变最终结果。
3. 朴素稳定 Softmax 需要求 max、求 sum、归一化。
4. Online Softmax 能在扫描时同时更新 max 和 sum，适合分块处理，减少访存。
5. CUDA 实现中通常用 Warp/Block 规约求 max 和 sum。

## 八、总结

Softmax 的关键不是公式，而是数值稳定和访存控制。面试中能讲清“减 max 防溢出”“max/sum 两类规约”“Online 更新公式”“CUDA 中如何做 Warp/Block 规约”，就已经覆盖核心逻辑。
