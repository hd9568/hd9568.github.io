---
title: 'CUDA Kernel Fusion：为什么融合能减少显存带宽瓶颈'
description: '从 Memory-bound 和 Compute-bound 角度理解 Kernel Fusion，讲清融合如何减少中间结果读写以及可能带来的副作用。'
category: 'CUDA'
pubDate: '2026-05-29'
updatedDate: '2026-05-29'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Memory-bound 与 Compute-bound](#二memory-bound-与-compute-bound)
3. [为什么多个 kernel 会慢](#三为什么多个-kernel-会慢)
4. [Kernel Fusion 示例](#四kernel-fusion-示例)
5. [融合为什么可能加速](#五融合为什么可能加速)
6. [融合的副作用](#六融合的副作用)
7. [常见融合场景](#七常见融合场景)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

Kernel Fusion 是把多个相邻算子合并成一个 kernel。

- 融合能减少中间结果写回 Global Memory 和再次读出。
- 融合能减少 kernel launch overhead。
- 对 Memory-bound 算子通常收益明显。
- 对 Compute-bound 算子，融合收益不一定大。
- 过度融合可能增加寄存器压力、降低 Occupancy、破坏复用或增加代码复杂度。

## 二、Memory-bound 与 Compute-bound

Memory-bound：瓶颈在显存带宽。

```text
读写数据很多，计算很少
例如：y = x + bias、ReLU、简单逐元素操作
```

Compute-bound：瓶颈在计算吞吐。

```text
计算量很大，数据复用高
例如：大矩阵 GEMM
```

Kernel Fusion 主要优化 Memory-bound 场景，因为它减少显存读写。

## 三、为什么多个 kernel 会慢

假设有两个操作：

```cpp
tmp = x + bias;
y = relu(tmp);
```

如果拆成两个 kernel：

```text
Kernel 1: read x, read bias, write tmp
Kernel 2: read tmp, write y
```

`tmp` 是中间结果，被写入 Global Memory 后又读回来。这个过程消耗显存带宽。

## 四、Kernel Fusion 示例

拆分写法：

```cpp
__global__ void add_bias(const float* x, const float* bias, float* tmp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        tmp[idx] = x[idx] + bias[idx];
    }
}

__global__ void relu(const float* tmp, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = tmp[idx] > 0.0f ? tmp[idx] : 0.0f;
    }
}
```

融合写法：

```cpp
__global__ void add_bias_relu(const float* x, const float* bias, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 中间结果 v 保存在寄存器里，不写回 Global Memory。
        float v = x[idx] + bias[idx];
        y[idx] = v > 0.0f ? v : 0.0f;
    }
}
```

融合后访存从：

```text
read x + read bias + write tmp + read tmp + write y
```

变成：

```text
read x + read bias + write y
```

省掉了 `tmp` 的一次写和一次读。

## 五、融合为什么可能加速

### 1. 减少显存流量

对逐元素算子，显存读写往往比计算更贵。减少中间结果读写直接降低时间。

### 2. 减少 kernel launch overhead

多个小 kernel 合成一个 kernel，可以减少 Host 提交开销。

### 3. 中间值留在寄存器

融合后中间结果可以保存在寄存器里，避免 Global Memory 往返。

### 4. 增加算术强度

算术强度是计算量和内存访问量的比值。

```text
Arithmetic Intensity = FLOPs / Bytes
```

融合减少 Bytes，如果 FLOPs 不变，算术强度提高，可能让 Memory-bound 更接近 Compute-bound。

## 六、融合的副作用

### 1. 寄存器压力增加

融合后一个线程同时保存更多中间变量，可能增加寄存器数量。

### 2. Occupancy 降低

寄存器或 shared memory 使用增加，可能让 SM 上驻留的 Warp 变少。

### 3. 代码复杂度增加

融合太多算子后，kernel 更难维护，也更难适配不同形状。

### 4. 破坏库算子优化

有些单独算子已经由 cuBLAS/cuDNN 高度优化。手动融合可能反而不如库实现。

## 七、常见融合场景

- Bias + Activation。
- Add + LayerNorm 前后处理。
- Scale + Mask + Softmax 某些阶段。
- Elementwise 链式操作。
- Transformer 中残差、bias、activation、dropout 的组合。

不太适合盲目融合的场景：

- 大 GEMM 主体。
- 计算密集且 Tensor Core 利用率已高的 kernel。
- 融合后寄存器压力明显爆炸的复杂逻辑。

## 八、面试回答模板

如果问题是“为什么 Kernel Fusion 可以加速”，可以这样回答：

1. 很多逐元素算子是 Memory-bound，时间主要花在 Global Memory 读写。
2. 多个 kernel 拆开执行时，中间结果需要写回显存，再由下一个 kernel 读出。
3. Fusion 把多个操作合到一个 kernel，中间结果可以留在寄存器中，减少显存流量。
4. 它还能减少 kernel launch overhead，并提高算术强度。
5. 但融合不是越多越好，可能增加寄存器压力、降低 Occupancy、增加代码复杂度。

## 九、总结

Kernel Fusion 的本质是减少不必要的内存往返。对 Memory-bound 的 elementwise 链路很有效；对 Compute-bound 的大算子，收益要具体分析。好的融合策略应该用 Nsight 指标验证显存流量和实际耗时是否下降。
