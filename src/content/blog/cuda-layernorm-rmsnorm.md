---
title: '手写 CUDA LayerNorm / RMSNorm：均值与方差的高效计算'
description: '从归一化公式出发，理解 LayerNorm 和 RMSNorm 的区别，并用单 Block 单行的 CUDA kernel 讲清均值、方差与规约实现。'
category: 'CUDA'
pubDate: '2026-06-24'
updatedDate: '2026-06-24'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [LayerNorm 与 RMSNorm 公式](#二layernorm-与-rmsnorm-公式)
3. [计算流程](#三计算流程)
4. [LayerNorm CUDA 示例](#四layernorm-cuda-示例)
5. [RMSNorm CUDA 示例](#五rmsnorm-cuda-示例)
6. [性能优化要点](#六性能优化要点)
7. [面试回答模板](#七面试回答模板)
8. [总结](#八总结)

## 一、核心结论

LayerNorm 和 RMSNorm 是 LLM 中非常高频的归一化算子。

- LayerNorm 对一行 hidden 维度求均值和方差。
- RMSNorm 不减均值，只用平方均值计算 RMS。
- 二者都需要对一行数据做规约。
- CUDA 实现常用一个 Block 处理一行，线程并行求 sum / sum of squares。
- 优化重点是规约效率、访存次数、向量化加载和融合后处理。

## 二、LayerNorm 与 RMSNorm 公式

LayerNorm：

```text
mean = sum(x_i) / H
var = sum((x_i - mean)^2) / H
y_i = (x_i - mean) / sqrt(var + eps) * gamma_i + beta_i
```

RMSNorm：

```text
rms = sqrt(sum(x_i^2) / H + eps)
y_i = x_i / rms * weight_i
```

区别：

- LayerNorm 需要均值和方差。
- RMSNorm 不计算均值，少一次中心化，通常更简单更快。
- RMSNorm 在很多 LLM 中常见。

## 三、计算流程

以一行 hidden size 为 `H` 的数据为例。

LayerNorm：

1. 多线程读入 `x`，求 `sum(x)`。
2. 得到 `mean`。
3. 多线程求 `sum((x - mean)^2)`。
4. 得到 `inv_std = rsqrt(var + eps)`。
5. 写出归一化结果。

RMSNorm：

1. 多线程求 `sum(x^2)`。
2. 得到 `inv_rms = rsqrt(sum_sq / H + eps)`。
3. 写出 `x * inv_rms * weight`。

## 四、LayerNorm CUDA 示例

下面是单 Block 处理一行的教学版本。

```cpp
__inline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffff;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__device__ float block_reduce_sum(float v, float* smem) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    v = warp_reduce_sum(v);

    if (lane == 0) {
        smem[warp_id] = v;
    }
    __syncthreads();

    float out = 0.0f;
    if (warp_id == 0) {
        out = (tid < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }

    if (tid == 0) {
        smem[0] = out;
    }
    __syncthreads();
    return smem[0];
}

__global__ void layernorm_kernel(const float* x,
                                 const float* gamma,
                                 const float* beta,
                                 float* y,
                                 int rows,
                                 int hidden,
                                 float eps) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float* row_x = x + row * hidden;
    float* row_y = y + row * hidden;

    // 第一次规约：求 sum。
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        local_sum += row_x[i];
    }
    float sum = block_reduce_sum(local_sum, smem);
    float mean = sum / hidden;

    // 第二次规约：求方差。
    float local_var = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float diff = row_x[i] - mean;
        local_var += diff * diff;
    }
    float var_sum = block_reduce_sum(local_var, smem);
    float inv_std = rsqrtf(var_sum / hidden + eps);

    // 写出归一化结果。
    for (int i = tid; i < hidden; i += blockDim.x) {
        float norm = (row_x[i] - mean) * inv_std;
        row_y[i] = norm * gamma[i] + beta[i];
    }
}
```

## 五、RMSNorm CUDA 示例

RMSNorm 少了求均值和 beta。

```cpp
__global__ void rmsnorm_kernel(const float* x,
                               const float* weight,
                               float* y,
                               int rows,
                               int hidden,
                               float eps) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float* row_x = x + row * hidden;
    float* row_y = y + row * hidden;

    // 求平方和。
    float local_sq_sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = row_x[i];
        local_sq_sum += v * v;
    }

    float sq_sum = block_reduce_sum(local_sq_sum, smem);
    float inv_rms = rsqrtf(sq_sum / hidden + eps);

    // 写出结果。
    for (int i = tid; i < hidden; i += blockDim.x) {
        row_y[i] = row_x[i] * inv_rms * weight[i];
    }
}
```

RMSNorm 比 LayerNorm 少一次中心化和一次减法相关计算，因此实现更简单。

## 六、性能优化要点

### 1. 减少访存次数

LayerNorm 需要至少多次读取同一行。若 hidden 较小，可以把数据缓存到 shared memory 或寄存器中，减少重复 Global Memory 访问。

### 2. 向量化加载

对 hidden 维度较大且地址对齐时，可以用 `float4` 加载，提高带宽利用率。

### 3. Warp 级规约

对 hidden 较小的场景，Warp 级规约可能比 Block 级更轻。

### 4. 融合后处理

LayerNorm/RMSNorm 常和 residual、bias、activation 融合，减少中间结果写回。

### 5. 精度处理

输入可能是 FP16/BF16，但规约通常用 FP32 累加，减少数值误差。

## 七、面试回答模板

如果问题是“LayerNorm 和 RMSNorm 怎么实现”，可以这样回答：

1. LayerNorm 对每一行 hidden 维度求均值和方差，再做归一化、乘 gamma、加 beta。
2. RMSNorm 不减均值，只计算平方均值的倒数平方根，再乘 weight。
3. CUDA 实现常用一个 Block 处理一行，线程分片读取 hidden 元素。
4. 用 Warp/Block 规约求 sum、sum of squares 或 variance。
5. 性能优化关注访存次数、向量化、规约效率、FP32 累加和与其他 elementwise 操作融合。

## 八、总结

LayerNorm/RMSNorm 的核心是“按行规约 + 按元素写回”。能写清楚 sum、variance、rsqrt、gamma/beta，并解释为什么 RMSNorm 更省计算，就已经覆盖了大部分面试重点。高性能实现再进一步考虑向量化、融合、半精度输入和 FP32 累加。
