---
title: 'Attention Mask 优化：Causal、Paged 与 Block Sparse Mask'
description: '梳理推理场景中常见 Attention Mask 的类型，解释它们在 kernel 内的实现成本，以及如何避免 mask 处理成为额外瓶颈。'
category: '推理优化'
pubDate: '2026-05-29'
updatedDate: '2026-05-29'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Mask 的本质](#二mask-的本质)
3. [Causal Mask](#三causal-mask)
4. [Padding / Length Mask](#四padding--length-mask)
5. [Paged Mask](#五paged-mask)
6. [Block Sparse Mask](#六block-sparse-mask)
7. [Kernel 内的成本](#七kernel-内的成本)
8. [面试表达](#八面试表达)
9. [总结](#九总结)

## 一、核心结论

Attention Mask 决定哪些 token 可以被关注。

- Causal Mask 防止看到未来 token。
- Padding/Length Mask 防止访问无效 token。
- Paged Mask 处理 KV Cache 分页后的地址映射和有效长度。
- Block Sparse Mask 让 attention 只计算部分 block，减少无效计算和访存。
- 高性能 kernel 通常不显式生成完整 mask，而是在计算过程中用索引判断。

## 二、Mask 的本质

Attention 先得到分数：

```text
S = QK^T / sqrt(d)
```

Mask 的作用是把不允许关注的位置变成一个极小值：

```text
S_masked[i, j] = -inf  if position j is not allowed
```

Softmax 后：

```text
exp(-inf) = 0
```

因此被 mask 的位置不会贡献到输出。

## 三、Causal Mask

Causal Mask 用于自回归模型：

```text
key_idx <= query_idx 允许
key_idx > query_idx  禁止
```

Kernel 中不一定需要真实 mask 矩阵，只需判断：

```cpp
if (key_idx > query_idx) {
    score = -INFINITY;
}
```

在 Prefill 中，Causal Mask 是下三角结构。Decode 中通常只有一个新 query，未来 token 不存在，更多关注有效长度和 KV 地址。

## 四、Padding / Length Mask

Serving 中 batch 内请求长度不同：

```text
request A length = 128
request B length = 2048
```

如果为了 batch 对齐而 padding，不能让 attention 访问 padding token。

判断逻辑：

```cpp
if (key_idx >= sequence_length[request_id]) {
    score = -INFINITY;
}
```

高性能实现更常用长度数组，而不是构造 `[batch, seq, seq]` 的大 mask。

## 五、Paged Mask

PagedAttention 中，逻辑 token 连续，物理 KV block 不连续。

这里 mask 和地址映射经常绑定在一起：

```cpp
int logical_block = token_id / BLOCK_SIZE;
int offset = token_id % BLOCK_SIZE;
int physical_block = block_table[request_id][logical_block];

bool valid = token_id < context_length[request_id];
```

Paged Mask 的成本包括：

- 读取 Block Table。
- 计算 logical/physical block。
- 判断最后一个 block 的有效 token 数。
- 处理不同请求的不同长度。

优化重点是让地址计算简单、访存合并尽量好、避免大量分支分化。

## 六、Block Sparse Mask

Block Sparse Mask 按块决定是否计算 attention。

例如长上下文中只关注：

- 当前 token 附近的局部窗口。
- 少量全局 token。
- 检索到的重要 block。

Mask 结构：

```text
block allowed matrix:
1 0 0 1
1 1 0 0
0 1 1 0
0 0 1 1
```

如果某个 `(Q block, K block)` 不需要关注，kernel 可以直接跳过整块计算。

这比逐元素 mask 更有工程价值，因为它减少的是整个 block 的 matmul 和 KV 读取。

## 七、Kernel 内的成本

Mask 处理可能带来额外开销：

### 1. 分支成本

```cpp
if (masked) {
    score = -INFINITY;
}
```

如果同一 warp 内不同 lane 的 mask 状态差异很大，可能产生分支分化。

### 2. 额外内存读取

如果 mask 是显式矩阵，读取 mask 本身也消耗带宽。

```cpp
bool masked = mask[row * n + col];
```

长上下文下，显式 mask 可能非常大，通常不推荐。

### 3. 地址计算成本

PagedAttention 需要 block table lookup；Block Sparse 需要 sparse metadata lookup。

### 4. 不规则访存

Sparse 或 paged 结构可能破坏连续读取，降低 memory coalescing。

## 八、面试表达

Attention Mask 优化可以这样回答：

1. Mask 本质上是把不允许关注的位置设为 `-inf`，Softmax 后概率为 0。
2. Causal Mask 用于自回归，防止 query 关注未来 key。
3. Serving 中还要处理不同请求长度，因此常用 length mask 避免访问无效 KV。
4. PagedAttention 中 mask 和 block table 地址映射相关，需要处理逻辑块、物理块和最后一块有效长度。
5. Block Sparse Mask 按块跳过无用 attention，真正减少计算和 KV 读取，但会引入 sparse metadata 和不规则访存成本。
6. 高性能 kernel 通常不显式生成完整 mask，而是在 kernel 内用索引和元数据判断。

## 九、总结

Mask 不是简单的布尔矩阵。推理系统中，mask 直接影响 kernel 分支、地址计算、KV 读取和并行效率。Causal Mask 保证正确性，Paged Mask 支撑动态 KV，Block Sparse Mask 则试图减少长上下文中的无效计算。
