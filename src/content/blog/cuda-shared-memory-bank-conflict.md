---
title: 'CUDA Shared Memory：Bank Conflict、Padding 与 Swizzle'
description: '理解 Shared Memory 的 Bank 组织方式，讲清 Bank Conflict 的产生原因，以及 Padding 和 Swizzle 的解决思路。'
category: 'CUDA'
pubDate: '2026-05-29T11:40:00+08:00'
updatedDate: '2026-05-29T11:40:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Shared Memory 是什么](#二shared-memory-是什么)
3. [Bank 是什么](#三bank-是什么)
4. [Bank Conflict 如何产生](#四bank-conflict-如何产生)
5. [Padding 解决冲突](#五padding-解决冲突)
6. [Swizzle 的思路](#六swizzle-的思路)
7. [矩阵转置示例](#七矩阵转置示例)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

Shared Memory 是 Block 内线程共享的高速片上内存，但访问方式不合理会产生 Bank Conflict。

- Shared Memory 延迟远低于 Global Memory。
- 它常用于缓存数据、复用数据、做 Block 内协作。
- Shared Memory 被划分成多个 Bank，线程访问不同 Bank 可以并行。
- 同一个 Warp 中多个线程访问同一个 Bank 的不同地址，会产生 Bank Conflict。
- Padding 通过改变行跨度，打散 Bank 映射。
- Swizzle 通过重排索引，让访问更均匀落到不同 Bank。

## 二、Shared Memory 是什么

Shared Memory 属于一个 Block，Block 内所有线程可见。

```cpp
__global__ void kernel(float* out) {
    __shared__ float tile[256];

    int tid = threadIdx.x;
    tile[tid] = static_cast<float>(tid);

    // 确保所有线程都完成写入，再读取其他线程写的数据。
    __syncthreads();

    out[blockIdx.x * blockDim.x + tid] = tile[255 - tid];
}
```

它常用于：

- GEMM Tiling。
- Reduction 中间结果。
- Stencil 邻域缓存。
- 避免重复 Global Memory 访问。

## 三、Bank 是什么

Shared Memory 内部被划分为多个 Bank。常见理解是 32 个 Bank，对应一个 Warp 的 32 个 lane。

对 4 字节数据，一个简化映射是：

```text
bank_id = address_index % 32
```

如果 Warp 内 32 个线程分别访问：

```text
tile[0], tile[1], tile[2], ..., tile[31]
```

它们通常落到 32 个不同 Bank，可以并行访问。

## 四、Bank Conflict 如何产生

如果多个线程访问同一个 Bank 的不同地址，就会产生冲突，访问被拆成多次。

典型场景：按列访问二维 shared memory。

```cpp
__shared__ float tile[32][32];

int lane = threadIdx.x;
float x = tile[lane][0];
```

行主序下，`tile[lane][0]` 的地址间隔是 32 个 float。简化映射：

```text
bank = (lane * 32 + 0) % 32 = 0
```

所有 lane 都访问 Bank 0 的不同地址，发生 32 路 Bank Conflict。

### 广播例外

如果多个线程访问同一个地址，硬件通常可以广播，不一定造成冲突。

```cpp
float x = tile[0][0];  // 所有线程读同一个地址，通常可广播。
```

## 五、Padding 解决冲突

Padding 是在每行末尾多加一列，改变行跨度。

```cpp
__shared__ float tile[32][33];
```

再按列访问：

```cpp
int lane = threadIdx.x;
float x = tile[lane][0];
```

简化映射：

```text
bank = (lane * 33 + 0) % 32 = lane % 32
```

32 个 lane 落到不同 Bank，冲突消失。

## 六、Swizzle 的思路

Swizzle 是通过改变索引映射，让物理存储位置和逻辑位置不同，从而减少冲突。

简化例子：

```cpp
int swizzled_col = col ^ (row & 31);
tile[row][swizzled_col] = value;
```

读取时使用相同规则还原对应位置。Swizzle 常用于更复杂的矩阵乘和 Tensor Core 数据布局中。

它的本质不是多存一列，而是用重排让不同线程访问分散到不同 Bank。

## 七、矩阵转置示例

朴素转置会有非合并写或非合并读。优化思路是先把 tile 读入 Shared Memory，再转置写出。

```cpp
#define TILE 32

__global__ void transpose(const float* input, float* output, int width, int height) {
    // 多加 1 列，避免按列读取 shared memory 时发生 bank conflict。
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < width && y < height) {
        // Global Memory 连续读。
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    int out_x = blockIdx.y * TILE + threadIdx.x;
    int out_y = blockIdx.x * TILE + threadIdx.y;

    if (out_x < height && out_y < width) {
        // 从 shared memory 转置读取，再连续写到 Global Memory。
        output[out_y * height + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

关键点：

- `tile[TILE][TILE + 1]` 通过 padding 避免列访问冲突。
- `__syncthreads()` 保证 tile 完整写入后再读取。
- 读写 Global Memory 时尽量保持连续访问。

## 八、面试回答模板

如果问题是“什么是 Bank Conflict”，可以这样回答：

1. Shared Memory 被划分成多个 Bank，Warp 内线程访问不同 Bank 可以并行。
2. 如果同一个 Warp 内多个线程访问同一 Bank 的不同地址，就会发生 Bank Conflict。
3. 冲突会让访问串行化，降低 shared memory 带宽。
4. 常见解决方法是 Padding，把二维数组列数从 32 改成 33，打散 Bank 映射。
5. 更复杂场景可以用 Swizzle 重排索引，减少冲突。

## 九、总结

Shared Memory 优化不是“用了就快”。它需要同时满足：

- 减少 Global Memory 重复访问。
- Block 内同步正确。
- 访问模式避免 Bank Conflict。
- Padding 和 Swizzle 是解决冲突的常见工具。
