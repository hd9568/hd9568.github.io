---
title: '标准 Attention 的复杂度：为什么长上下文会迅速变慢'
description: '从 Q、K、V 的矩阵形状出发，推导标准 Attention 的时间复杂度和显存复杂度，理解长上下文推理的根本瓶颈。'
category: '推理优化'
pubDate: '2026-06-25'
updatedDate: '2026-06-25'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Attention 解决什么问题](#二attention-解决什么问题)
3. [公式与张量形状](#三公式与张量形状)
4. [时间复杂度](#四时间复杂度)
5. [显存复杂度](#五显存复杂度)
6. [长上下文为什么变慢](#六长上下文为什么变慢)
7. [最小代码示例](#七最小代码示例)
8. [面试表达](#八面试表达)
9. [总结](#九总结)

## 一、核心结论

标准 Attention 的核心瓶颈来自 `QK^T` 这张注意力分数矩阵。

- 序列长度为 `n`，每个 head 的维度为 `d`。
- `QK^T` 的形状是 `n x n`。
- 计算复杂度约为 `O(n^2 d)`。
- 注意力分数和 Softmax 结果如果落到显存，空间复杂度约为 `O(n^2)`。
- 当上下文长度翻倍，注意力矩阵大小变为 4 倍。

这就是长上下文推理和训练变慢的根源之一。

## 二、Attention 解决什么问题

Attention 的作用是让每个 token 从其他 token 中取信息。

例如句子中第 `i` 个 token 需要回答：

```text
我应该关注前面哪些 token？每个 token 权重是多少？
```

它通过 Query、Key、Value 完成：

- `Q`：当前 token 发出的查询。
- `K`：每个 token 提供的匹配特征。
- `V`：每个 token 真正被聚合的信息。

直观理解：

```text
相似度 = Q 和 K 的点积
输出 = 按相似度加权求和 V
```

## 三、公式与张量形状

单个 attention head 的公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

设：

```text
Q: n x d
K: n x d
V: n x d
```

则：

```text
QK^T:       n x n
softmax:    n x n
output:     n x d
```

注意力矩阵中第 `(i, j)` 个元素表示：第 `i` 个 token 对第 `j` 个 token 的关注程度。

## 四、时间复杂度

### 1. 计算 QK^T

`Q` 是 `n x d`，`K^T` 是 `d x n`。

矩阵乘法复杂度：

```text
O(n * d * n) = O(n^2 d)
```

### 2. Softmax

Softmax 需要对每一行的 `n` 个元素做：

- 求最大值。
- 求指数。
- 求和。
- 归一化。

共有 `n` 行，因此复杂度约为：

```text
O(n^2)
```

### 3. 乘以 V

`softmax(QK^T)` 是 `n x n`，`V` 是 `n x d`。

复杂度：

```text
O(n * n * d) = O(n^2 d)
```

因此标准 Attention 总体主要是：

```text
O(n^2 d)
```

## 五、显存复杂度

朴素实现会显式保存 attention scores：

```text
scores = QK^T       # n x n
probs = softmax(scores)  # n x n
```

单 head 下，如果 `n = 8192`，`scores` 元素数量是：

```text
8192 * 8192 = 67,108,864
```

FP16 每个元素 2 字节，则单个 `n x n` 矩阵约：

```text
67,108,864 * 2 bytes ≈ 128 MB
```

多 head、多 batch、多层叠加后，显存压力非常明显。

## 六、长上下文为什么变慢

注意力复杂度对序列长度是二次增长。

| 序列长度 | 注意力矩阵元素数 |
| --- | --- |
| 2K | 4M |
| 4K | 16M |
| 8K | 67M |
| 16K | 268M |
| 32K | 1B |

上下文从 8K 增加到 32K，长度是 4 倍，attention matrix 是 16 倍。

这会带来两个问题：

- 算得更多：`QK^T` 和 `PV` 计算量变大。
- 搬得更多：attention scores/probs 的读写流量变大。

在 GPU 上，很多时候不是算力不够，而是 HBM 读写太多。

## 七、最小代码示例

下面是一个简化版 PyTorch Attention。它很清楚，但不是高性能实现。

```python
import torch
import math


def attention(q, k, v):
    # q, k, v: [batch, heads, seq_len, head_dim]
    d = q.shape[-1]

    # scores: [batch, heads, seq_len, seq_len]
    # 这里会显式生成 n x n 注意力矩阵。
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    # 对每个 query token 的所有 key token 做归一化。
    probs = torch.softmax(scores, dim=-1)

    # output: [batch, heads, seq_len, head_dim]
    out = torch.matmul(probs, v)
    return out
```

这段代码的问题不是数学错误，而是中间的 `scores` 和 `probs` 都可能非常大。FlashAttention 的核心就是避免把完整 `n x n` 矩阵写回 HBM。

## 八、面试表达

标准 Attention 可以这样回答：

1. 对单个 head，`Q/K/V` 形状通常是 `n x d`。
2. Attention 先计算 `QK^T`，得到 `n x n` 分数矩阵。
3. `QK^T` 和后面的 `softmax(QK^T)V` 都是 `O(n^2 d)` 量级。
4. 朴素实现还会保存 `n x n` 的 scores/probs，因此空间复杂度是 `O(n^2)`。
5. 长上下文变慢是因为 `n` 翻倍后，注意力矩阵变成 4 倍，计算和显存读写都会明显上升。

## 九、总结

标准 Attention 的瓶颈不是某个实现细节，而是 `n x n` 注意力矩阵本身。理解 `O(n^2 d)` 和 `O(n^2)`，才能理解 FlashAttention、PagedAttention、Linear Attention、Sparse Attention 这些优化为什么存在。
