---
title: 'FlashAttention-1：Tiling、SRAM/HBM 与 Online Softmax'
description: '从标准 Attention 的 HBM 读写瓶颈出发，解释 FlashAttention-1 如何通过分块、片上 SRAM 和 Online Softmax 避免保存完整注意力矩阵。'
category: '推理优化'
pubDate: '2026-05-29T14:00:00+08:00'
updatedDate: '2026-05-29T14:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [标准 Attention 的 IO 问题](#二标准-attention-的-io-问题)
3. [Tiling 思想](#三tiling-思想)
4. [Online Softmax](#四online-softmax)
5. [Forward 过程](#五forward-过程)
6. [Backward 重计算逻辑](#六backward-重计算逻辑)
7. [伪代码](#七伪代码)
8. [面试表达](#八面试表达)
9. [总结](#九总结)

## 一、核心结论

FlashAttention-1 的核心不是改变 Attention 数学公式，而是改变计算和访存方式。

- 标准 Attention 会显式写出 `S = QK^T` 和 `P = softmax(S)`。
- `S` 和 `P` 都是 `n x n`，长序列下 HBM 读写很重。
- FlashAttention 用 tiling 把 Q、K、V 分块搬到 SRAM/shared memory。
- 通过 Online Softmax 分块计算 softmax，不需要保存完整 attention matrix。
- Forward 保存少量统计量，Backward 通过重计算 attention block 减少显存占用。

## 二、标准 Attention 的 IO 问题

朴素实现：

```text
S = QK^T
P = softmax(S)
O = PV
```

这里 `S` 和 `P` 都可能写入 HBM。

GPU 性能常受 HBM 带宽限制。即使矩阵乘本身很快，把巨大中间矩阵反复写回/读出也会拖慢整体。

FlashAttention 的目标：

```text
不把完整 S/P 写入 HBM，只在片上分块计算最终 O。
```

## 三、Tiling 思想

Tiling 就是把大矩阵切成小块处理。

```text
Q: 按行切成 Q_i
K/V: 按行切成 K_j, V_j
```

每次处理一个 `Q_i` 和一个 `K_j/V_j`：

```text
S_ij = Q_i K_j^T
P_ij = softmax 的一部分
O_i += P_ij V_j
```

关键是 `S_ij` 小，可以放在 SRAM/shared memory/register 中，不必写入 HBM。

## 四、Online Softmax

Softmax 需要全局分母：

```text
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
```

分块计算时，不能一次看到整行所有元素。Online Softmax 维护两个统计量：

```text
m: 当前已经处理元素的最大值
l: 当前基于 m 的 exp 累加和
```

当新块分数 `s` 到来：

```text
m_new = max(m_old, max(s))
l_new = l_old * exp(m_old - m_new) + sum(exp(s - m_new))
```

旧输出也要按新最大值缩放：

```text
O_new = O_old * (l_old * exp(m_old - m_new) / l_new)
      + exp(s - m_new) V / l_new
```

这个公式保证分块计算结果和完整 softmax 一致。

## 五、Forward 过程

FlashAttention forward 的简化流程：

1. 读入一个 Q block。
2. 遍历所有 K/V block。
3. 在片上计算当前 block 的 `QK^T`。
4. 用 Online Softmax 更新当前 Q block 的 `m`、`l` 和输出 `O`。
5. 最后只把 `O`、`m`、`l` 写回 HBM。

保存 `m` 和 `l` 是为了 backward 时恢复 softmax 归一化信息。

## 六、Backward 重计算逻辑

训练时 backward 需要 attention probabilities。

朴素做法会保存 `P`，但 `P` 是 `n x n`，显存巨大。

FlashAttention 的做法是：

- Forward 不保存完整 `P`。
- Backward 再按块重算 `S_ij` 和 `P_ij`。
- 使用 forward 保存的 `m/l` 恢复对应 block 的 softmax。

这是典型的“用计算换显存”。

推理只需要 forward，但理解 backward 有助于解释 FlashAttention 为什么在训练中也省显存。

## 七、伪代码

简化伪代码：

```python
def flash_attention_forward(Q, K, V):
    # O 是最终输出；m/l 是每一行 softmax 的统计量。
    O = zeros_like(Q)
    m = full([num_rows], -inf)
    l = zeros([num_rows])

    for Q_block in blocks(Q):
        O_i = 0
        m_i = -inf
        l_i = 0

        for K_block, V_block in blocks(K, V):
            # 当前小块分数，只存在片上内存中。
            S = Q_block @ K_block.T

            m_new = maximum(m_i, rowmax(S))
            P = exp(S - m_new[:, None])

            # 把旧分母和旧输出换到新的最大值基准。
            alpha = exp(m_i - m_new)
            l_new = l_i * alpha + rowsum(P)

            O_i = O_i * (l_i * alpha / l_new)[:, None] + (P @ V_block) / l_new[:, None]
            m_i = m_new
            l_i = l_new

        write(O_i, m_i, l_i)
```

实际 CUDA kernel 会把这些步骤映射到 block、warp、register、shared memory 和 Tensor Core。

## 八、面试表达

FlashAttention-1 可以这样回答：

1. 它没有改变 Attention 的数学结果，而是优化 IO。
2. 标准 Attention 会把 `QK^T` 和 softmax 后的 `P` 写入 HBM，显存读写量很大。
3. FlashAttention 把 Q/K/V 分块，每次在 SRAM/shared memory 中计算一个小块。
4. 用 Online Softmax 维护每行的最大值和归一化分母，因此不需要保存完整 `n x n` attention matrix。
5. 训练 backward 中不保存完整概率矩阵，而是按块重计算，用计算换显存。

## 九、总结

FlashAttention 的关键是 IO-aware。标准 Attention 慢在巨大中间矩阵的 HBM 往返；FlashAttention 用 tiling 和 Online Softmax 把中间结果留在片上，只写最终输出和少量统计量。
