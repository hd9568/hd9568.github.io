---
title: 'CUDA 线程模型：Grid、Block、Thread 与 Global Thread ID'
description: '用一维、二维、三维映射理解 CUDA Grid、Block、Thread 的组织方式，并讲清 Global Thread ID 的计算方法。'
category: 'CUDA'
pubDate: '2026-06-10'
updatedDate: '2026-06-10'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Grid、Block、Thread 的关系](#二gridblockthread-的关系)
3. [一维 Global Thread ID](#三一维-global-thread-id)
4. [二维映射](#四二维映射)
5. [三维映射](#五三维映射)
6. [线程块大小如何选择](#六线程块大小如何选择)
7. [常见错误](#七常见错误)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

CUDA 用层次化线程模型组织并行计算。

- 一个 kernel 启动一个 Grid。
- 一个 Grid 包含多个 Block。
- 一个 Block 包含多个 Thread。
- `blockIdx` 表示当前 Block 在 Grid 中的位置。
- `threadIdx` 表示当前 Thread 在 Block 中的位置。
- `blockDim` 表示每个 Block 的维度。
- `gridDim` 表示 Grid 的维度。
- Global Thread ID 用于把每个线程映射到全局数据下标。

## 二、Grid、Block、Thread 的关系

CUDA 启动语法：

```cpp
kernel<<<gridDim, blockDim>>>(args);
```

例如：

```cpp
dim3 block(256);
dim3 grid((n + block.x - 1) / block.x);
add_kernel<<<grid, block>>>(a, b, c, n);
```

层次结构：

```text
Grid
+-----------------------------+
| Block 0 | Block 1 | Block 2 |
| 256 thr | 256 thr | 256 thr |
+-----------------------------+
```

Block 是资源分配单位，Thread 是最细粒度执行单元，Warp 是硬件调度单位。

## 三、一维 Global Thread ID

一维数组最常见。

```cpp
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    // blockIdx.x：当前 block 的编号。
    // blockDim.x：每个 block 有多少线程。
    // threadIdx.x：当前线程在 block 内的编号。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // n 不一定刚好整除 blockDim.x，因此必须做边界检查。
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

启动：

```cpp
int threads_per_block = 256;
int blocks = (n + threads_per_block - 1) / threads_per_block;
add_kernel<<<blocks, threads_per_block>>>(a, b, c, n);
```

向上取整公式：

```cpp
blocks = (n + threads - 1) / threads;
```

它保证即使 `n` 不是 `threads` 的整数倍，也能覆盖所有元素。

## 四、二维映射

二维映射适合图像、矩阵、二维张量。

```cpp
__global__ void scale_2d(float* x, int width, int height) {
    // col 对应矩阵列，row 对应矩阵行。
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int offset = row * width + col;  // 行主序展开。
        x[offset] *= 2.0f;
    }
}
```

启动：

```cpp
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);
scale_2d<<<grid, block>>>(x, width, height);
```

二维 Block 常见选择是 `16x16` 或 `32x8`。选择要结合访存模式、寄存器、共享内存和 Occupancy。

## 五、三维映射

三维映射适合三维网格、体数据、`batch x row x col` 这类结构。

```cpp
__global__ void scale_3d(float* x, int width, int height, int depth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z < depth && row < height && col < width) {
        // 三维数组按 z, row, col 展开成一维。
        int offset = (z * height + row) * width + col;
        x[offset] *= 2.0f;
    }
}
```

启动：

```cpp
dim3 block(8, 8, 4);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y,
          (depth + block.z - 1) / block.z);
```

三维映射不是必须的。很多时候也可以手动把一维 `idx` 解码成多维坐标。

## 六、线程块大小如何选择

线程块大小影响：

- 一个 Block 内有多少 Warp。
- 每个 SM 能驻留多少 Block。
- 寄存器和共享内存占用。
- 访存合并和执行效率。

常见经验：

- 一维向量操作：`128`、`256`、`512` 线程常见。
- 二维矩阵操作：`16x16`、`32x8` 常见。
- Block 线程数通常选择 Warp 大小 32 的倍数。

示例：

```cpp
// 256 是 32 的 8 倍，一个 block 包含 8 个 warp。
int threads = 256;
```

最终选择应通过 profiling 验证，而不是只靠经验。

## 七、常见错误

### 1. 忘记边界检查

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
c[idx] = a[idx] + b[idx];  // n 不能整除时可能越界
```

正确写法：

```cpp
if (idx < n) {
    c[idx] = a[idx] + b[idx];
}
```

### 2. 二维下标展开写错

行主序矩阵 `[row][col]` 展开为：

```cpp
offset = row * width + col;
```

常见错误是把 `width` 和 `height` 混用。

### 3. Block 大小不是 Warp 倍数

不是绝对错误，但通常会导致最后一个 Warp 有大量无效线程。除非有特殊原因，线程数一般选 32 的倍数。

## 八、面试回答模板

如果问题是“CUDA 中如何计算 global thread id”，可以这样回答：

1. CUDA kernel 启动一个 Grid，Grid 里有多个 Block，Block 里有多个 Thread。
2. 一维场景下，全局线程编号是 `blockIdx.x * blockDim.x + threadIdx.x`。
3. 二维场景下，分别计算 `row` 和 `col`，再用 `row * width + col` 展开到一维地址。
4. Grid 大小通常用向上取整公式 `(n + blockDim - 1) / blockDim`。
5. 因为总线程数可能大于数据元素数，所以 kernel 内必须做边界检查。

## 九、总结

CUDA 线程模型本质上是把大规模数据映射给大量线程：

```text
数据下标 <-> Global Thread ID
```

一维、二维、三维只是不同数据形状的映射方式。只要下标计算正确、边界检查完整、Block 大小合理，后续优化才有可靠基础。
