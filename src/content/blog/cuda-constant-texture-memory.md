---
title: 'CUDA Constant Memory 与 Texture Memory：只读数据的访存选择'
description: '理解 Constant Memory 和 Texture Memory 的适用场景，掌握只读广播数据、空间局部性数据的访存优化思路。'
category: 'CUDA'
pubDate: '2026-06-15'
updatedDate: '2026-06-15'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [为什么需要特殊只读内存](#二为什么需要特殊只读内存)
3. [Constant Memory](#三constant-memory)
4. [Texture Memory](#四texture-memory)
5. [二者如何选择](#五二者如何选择)
6. [常见误区](#六常见误区)
7. [面试回答模板](#七面试回答模板)
8. [总结](#八总结)

## 一、核心结论

Constant Memory 和 Texture Memory 都适合只读数据，但优化目标不同。

- Constant Memory 适合所有线程读取相同地址的广播场景。
- Texture Memory 适合具有空间局部性的只读访问，例如图像、二维数据、插值采样。
- 如果 Warp 内线程读取不同 Constant Memory 地址，访问可能串行化。
- 现代 GPU 还有只读数据缓存路径，普通 `const __restrict__` 指针也可能表现很好。
- 选择特殊内存前，应先确认访问模式。

## 二、为什么需要特殊只读内存

很多 kernel 会访问只读参数：

- 卷积权重。
- 标量配置。
- 查表数据。
- 图像或二维纹理。
- 小型常量数组。

如果访问模式稳定，可以让硬件使用更合适的缓存路径，减少 Global Memory 压力。

## 三、Constant Memory

Constant Memory 用 `__constant__` 声明。

```cpp
// 设备端常量数组，大小通常有限，适合小型只读参数。
__constant__ float kWeights[256];
```

Host 端拷贝数据：

```cpp
float h_weights[256];
// 初始化 h_weights ...

cudaMemcpyToSymbol(kWeights, h_weights, sizeof(h_weights));
```

Kernel 使用：

```cpp
__global__ void apply_weight(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 如果 Warp 内线程读取同一个 kWeights 下标，constant cache 可以广播。
        y[idx] = x[idx] * kWeights[0];
    }
}
```

### Constant Memory 适合广播

好模式：

```cpp
float scale = kWeights[0];  // Warp 内所有线程读同一地址。
```

差模式：

```cpp
float w = kWeights[threadIdx.x];  // Warp 内线程读不同地址，可能串行化。
```

因此，Constant Memory 最适合“很多线程读同一个小常量”。

## 四、Texture Memory

Texture Memory 更偏向图像和二维/三维空间局部性访问。它通过 texture cache 优化只读数据访问，支持一些硬件采样能力。

现代 CUDA 中常使用 texture object。

伪代码结构：

```cpp
// Host 端创建 cudaTextureObject_t，把 CUDA array 或线性内存绑定给 texture。
// Kernel 中通过 tex2D<float>(tex, x, y) 读取。
```

Kernel 侧示意：

```cpp
__global__ void sample_kernel(cudaTextureObject_t tex, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // tex2D 适合二维空间局部性访问。
        float value = tex2D<float>(tex, x + 0.5f, y + 0.5f);
        out[y * width + x] = value;
    }
}
```

Texture Memory 的典型优势：

- 对二维局部性访问友好。
- 支持地址模式和过滤模式。
- 适合图像处理、采样、查表类场景。

## 五、二者如何选择

| 场景 | 更适合 |
| --- | --- |
| 所有线程读同一个小参数 | Constant Memory |
| Warp 内线程读不同常量地址 | 不一定适合 Constant Memory |
| 二维图像采样 | Texture Memory |
| 普通连续数组只读 | Global Memory + 只读缓存路径可能足够 |
| 大型模型权重连续读 | 通常关注 coalescing、L2、Tensor Core 数据流 |

对于 AI 算子，权重通常很大，Constant Memory 容量不够。它更适合小表、小配置、广播标量。

## 六、常见误区

### 1. Constant Memory 不是所有只读访问都快

如果 Warp 内每个线程读不同地址，它可能被串行化，反而不快。

### 2. Texture Memory 不是必须用于所有二维数据

如果二维数据访问本身已经合并、连续，普通 Global Memory 加缓存也可能很好。

### 3. 特殊内存不能替代数据布局优化

访存优化第一步仍然是看访问模式：连续、对齐、复用、少重复。

## 七、面试回答模板

如果问题是“Constant Memory 和 Texture Memory 适用场景”，可以这样回答：

1. Constant Memory 适合小型只读数据，尤其是 Warp 内线程读取同一地址的广播场景。
2. 如果 Warp 内线程读取不同 constant 地址，访问可能串行化。
3. Texture Memory 适合有空间局部性的只读访问，例如图像、二维采样和查表。
4. 现代 GPU 普通只读 Global Memory 也有缓存路径，是否使用特殊内存要看访问模式和 profiling。
5. AI 算子中大权重通常不放 Constant Memory，小参数和配置更适合。

## 八、总结

Constant Memory 和 Texture Memory 的本质是给特定只读访问模式提供更合适的缓存路径。优化前先判断访问模式：广播读选 Constant，二维局部性读考虑 Texture，连续大数组优先保证合并访问和数据复用。
