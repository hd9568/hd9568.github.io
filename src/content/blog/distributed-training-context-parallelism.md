---
title: 'Context Parallelism 详解：长序列如何跨 GPU 计算 Attention'
description: '从 Attention 数学语义出发，讲清 Context Parallelism 的序列切分、KV 通信、Ring Attention、Online Softmax、反向传播、因果负载均衡以及与 SP/Ulysses 的区别。'
category: '分布式训练'
pubDate: '2026-08-27T10:00:00+08:00'
updatedDate: '2026-08-27T10:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [先理解单卡 Attention](#一先理解单卡-attention)
2. [长序列为什么单卡放不下](#二长序列为什么单卡放不下)
3. [CP 怎样切分一次 Forward](#三cp-怎样切分一次-forward)
4. [从 AllGather 到 Ring Attention](#四从-allgather-到-ring-attention)
5. [Online Softmax 与因果负载均衡](#五online-softmax-与因果负载均衡)
6. [Backward、通信成本与工程正确性](#六backward通信成本与工程正确性)
7. [CP、SP 与 Ulysses 如何选择](#七cpsp-与-ulysses-如何选择)

## 一、先理解单卡 Attention

CP 解决的是“一个样本的序列太长，单张 GPU 放不下”的问题。理解它之前，只需要先明确几个词。

```text
Token:
  序列中的一个位置，例如一句话中的一个词元。

Hidden State:
  模型给每个 Token 保存的一行向量。

Parameter:
  训练要学习的权重，例如后文的 Wq、Wk、Wv。

Forward:
  输入经过模型得到预测和 Loss 的过程。

Backward:
  从 Loss 反向计算 Activation Gradient 和 Parameter Gradient 的过程。

Activation:
  Forward 期间产生、Backward 计算梯度时还要使用的中间结果。

Rank:
  一个分布式进程，训练中通常对应一张 GPU。
```

假设一句话被切成 8 个 Token，每个 Token 用长度为 `H` 的向量表示：

```text
X = [x0, x1, x2, x3, x4, x5, x6, x7]
X shape = [S, H]

S = 8，表示序列长度
H = 每个 Token 的隐藏维度
```

Self-Attention 先通过三个 Linear 得到 Q、K、V：

```text
Q = X Wq
K = X Wk
V = X Wv
```

可以先用一个不严格但直观的比喻理解：

```text
Q（Query）: 当前 Token 想找什么
K（Key）:   每个 Token 可以用什么特征被匹配
V（Value）: 匹配后真正取回的信息
```

第 `i` 个 Query 会与所有 Key 计算相关性，再按 Softmax 权重汇总 Value：

```text
score(i, j) = q_i dot k_j / sqrt(D)

weight(i, :) = softmax(score(i, :))

o_i = sum_j weight(i, j) * v_j
```

Softmax 把一行任意实数 Score 转成非负权重，并让这一行权重之和等于 1。

矩阵形式为：

```text
O = softmax(Q K^T / sqrt(D) + Mask) V
```

模型通常把 Hidden State 拆成多个 Attention Head，让不同 Head 学习不同关系；`D` 是每个 Head 的维度。Decoder-only LLM 还使用 Causal Mask：第 `i` 个 Token 只能读取位置 `0...i`，不能偷看未来 Token。

这里最重要的结论是：

> MLP（逐 Token 的前馈网络）、LayerNorm（对单个 Token 的特征做归一化）通常都能逐 Token 计算；Self-Attention 却需要让一个 Token 与同一序列中的其他 Token 交互。

CP 在 Forward 中新增的 KV 通信，正是来自这个差别。

## 二、长序列为什么单卡放不下

训练与推理不同。训练做完 Forward 后不能立即丢掉所有中间值，因为 Backward 还要用它们计算参数梯度。

设输入 Shape 为：

```text
X: [B, S, H]

B: 一张卡一次处理的样本数
S: 每个样本的 Token 数
H: 每个 Token 的隐藏维度
```

Q、K、V、MLP 中间结果等 Activation 的大小大多与：

```text
B * S * H
```

成正比。序列从 8K 增长到 128K，这部分显存约增长 16 倍。

最朴素的 Attention 还会保存每对 Token 的 Score：

```text
Attention Score shape = [B, N_head, S, S]
```

它随 `S^2` 增长。FlashAttention 用分块计算避免把完整 `S x S` Score Matrix 写入显存，但它没有消除 Q/K/V、层输出和 MLP Activation。因此：

```text
FlashAttention:
  解决 Attention 中间矩阵的 O(S^2) 显存问题。

Context Parallelism:
  把仍随 S 增长的 Activation 和计算分到多张 GPU。
```

Activation Checkpointing 是另一种办法：Forward 少保存一些 Activation，Backward 时重新计算。它用额外计算换显存；CP 用多卡计算和通信换显存，两者可以一起使用。

如果使用 `p` 路 CP，理想情况下每卡只保存 `1/p` 的序列 Activation：

```text
M_activation_per_rank ≈ M_activation / p
```

实际值还会包含通信 Buffer、Attention Workspace 等开销，所以不会严格等于 `1/p`。

## 三、CP 怎样切分一次 Forward

先看最简单的例子：8 个 Token 分到 2 张 GPU。

```text
GPU 0: x0, x1, x2, x3
GPU 1: x4, x5, x6, x7
```

一般地，`p` 路 CP 会把：

```text
X: [B, S, H]
```

切成每卡：

```text
X_i: [B, S/p, H]
```

CP Group 是共同处理同一条序列的 `p` 个 Rank。组内通常保存相同的模型参数，但每个 Rank 只处理本地 Token。若同时使用 TP、PP 或 ZeRO，参数还会沿相应维度继续切分。

### 哪些操作不需要通信

Linear 对每个 Token 独立执行：

```text
Y = XW
```

切分后就是：

```text
GPU 0: Y_0 = X_0 W
GPU 1: Y_1 = X_1 W
```

LayerNorm、RMSNorm、MLP、Activation Function 和 Residual Add 也不会读取其他 Token，因此都能直接处理本地分片。

### Attention 为什么不能只算本地

两张卡各自生成：

```text
GPU 0: Q_0, K_0, V_0
GPU 1: Q_1, K_1, V_1
```

如果 GPU 1 只计算：

```text
softmax(Q_1 K_1^T) V_1
```

那么 `x4...x7` 永远看不到 GPU 0 上的 `x0...x3`，结果已经不是原来的全局 Attention。

正确的 GPU 1 输出应为：

```text
O_1 = softmax(Q_1 [K_0; K_1]^T / sqrt(D) + Mask) [V_0; V_1]
```

分号表示沿序列维拼接。由此得到 CP 最核心的数据流：

```text
1. 输入按 Token 切到多张 GPU。
2. 每张卡为本地 Token 生成 Q、K、V。
3. Q 留在本地，K/V 在 CP Rank 之间交换。
4. 本地 Q 消费全局 K/V，得到本地 Attention Output。
5. Output Projection、MLP 等继续处理本地 Token。
6. 下一层仍保持相同的序列切分。
```

因此 CP 不是每层都把完整序列重新拼起来。跨卡交换集中在 Attention 的 KV，其他模块继续使用本地序列分片。

CP 只重排数据和计算，不改变 Attention 公式，因此它是精确并行，不是稀疏 Attention 或近似 Attention。

后文用以下 Shape：

```text
Q: [B, S, Nq,  D]
K: [B, S, Nkv, D]
V: [B, S, Nkv, D]

Nq:  Query Head 数
Nkv: Key/Value Head 数
D:   每个 Head 的维度
```

MHA 中 `Nq = Nkv`；GQA 让多组 Query 共享较少的 KV Head；MQA 只有一组 KV Head。因为 CP 主要传输 K/V，所以 GQA/MQA 能直接降低通信量。

## 四、从 AllGather 到 Ring Attention

实现正确 CP 的第一种办法，是先把所有 Rank 的 K/V 收集到每张卡。

### 最直观的 AllGather

AllGather 的含义是：每张卡贡献自己的分片，操作结束后每张卡都拿到完整 Tensor。

```text
before:
  GPU 0 has K_0
  GPU 1 has K_1

after AllGather:
  GPU 0 has [K_0; K_1]
  GPU 1 has [K_0; K_1]
```

Forward 可以写成：

```python
# 仅表达数据流，不对应某个框架的完整 API。
K_full = all_gather(K_local, group=cp_group)
V_full = all_gather(V_local, group=cp_group)

O_local = flash_attention(
    Q_local,
    K_full,
    V_full,
    causal=True,
    global_query_positions=position_ids,
)
```

它容易理解，Mask 也容易实现，但每张卡在 Attention 期间都要容纳完整 K/V。序列特别长时，这会抬高峰值显存；AllGather 还必须在 Attention 开始前基本完成，通信较难隐藏。

高效实现只把完整 K/V 当作临时计算数据，不把它保存到 Backward；Backward 需要时再次收集，避免长期占据显存。

### Ring：不收集完整 KV，只让分块依次经过

两张卡时，GPU 0 先使用 `KV_0`，再接收并使用 `KV_1`；GPU 1 则相反。扩展到 4 张卡：

```text
step 0: Rank i 使用自己的 KV_i
step 1: Rank i 使用从前一个 Rank 收到的 KV
step 2: 继续接收下一块 KV
step 3: 使用最后一块 KV
```

经过 `p` 轮后，每个本地 Q 都与全部 `p` 个 KV Block 计算过 Attention。

Ring 使用 Point-to-Point，简称 P2P：

```text
Send: 把一个 KV Block 发给下一个 Rank
Recv: 从上一个 Rank 接收一个 KV Block
```

真正的优化点不是“把 AllGather 拆成 Send/Recv”，而是**边计算当前 KV，边传输下一块 KV**：

```python
kv = kv_local
kv_owner = cp_rank
state = online_softmax_init(Q_local)

for step in range(cp_size):
    # 在通信 Stream 上提前旋转当前 KV；Send 和 Attention 都只读 kv。
    if step + 1 < cp_size:
        recv_kv = empty_like(kv)
        reqs = rotate_async(
            send_tensor=kv,
            recv_tensor=recv_kv,
            send_to=next_rank,
            recv_from=prev_rank,
            group=cp_group,
        )

    # 在计算 Stream 上处理当前 KV，与上面的通信并行。
    if block_is_visible(Q_local, kv_owner, causal=True):
        state = flash_attention_update(state, Q_local, kv)

    if step + 1 < cp_size:
        wait_until_recv_ready(reqs)
        kv = recv_kv
        kv_owner = (kv_owner - 1) % cp_size

O_local = online_softmax_finalize(state)
```

真实实现还需要独立 CUDA Stream、Event 和双缓冲。上面的代码只说明正确顺序：

```text
先异步启动下一块通信
-> 同时计算当前块
-> 当前计算结束后等待下一块就绪
```

如果每轮先等接收完成再启动 Attention，就没有通信计算重叠，Ring 只会变成多次串行小通信。

## 五、Online Softmax 与因果负载均衡

Ring 每次只能看到一部分 KV。问题是：对每个 Query，Softmax 的分母依赖所有 Key，怎样分块计算后仍与一次性 Attention 完全等价？

### 局部 Attention 结果不能直接相加

假设 KV 被分成 A、B 两块：

```text
S_a = Q K_a^T / sqrt(D)
S_b = Q K_b^T / sqrt(D)
```

错误做法是：

```text
softmax(S_a) V_a + softmax(S_b) V_b
```

因为两块各自把概率归一化到 1，破坏了全局 Softmax 的分母。

正确做法是让每块返回三个状态：

```text
m_a: 该行 Score 的最大值
l_a: 以 m_a 为基准的指数和
o_a: 尚未除以 l_a 的加权 Value
```

即：

```text
m_a = rowmax(S_a)
l_a = sum(exp(S_a - m_a))
o_a = exp(S_a - m_a) V_a
```

B 块同理。合并时先统一数值基准：

```text
m = max(m_a, m_b)

l = exp(m_a - m) * l_a
  + exp(m_b - m) * l_b

o = exp(m_a - m) * o_a
  + exp(m_b - m) * o_b

O = o / l
```

这就是 Online Softmax。它保证 KV Block 无论以什么顺序到达，最终结果都等价于对完整 Score Row 做一次 Softmax。FlashAttention 也依赖同一思想，只是它在单张 GPU 的 SRAM/HBM 层次间分块；Ring Attention 把分块扩展到了多张 GPU。

### Causal Mask 为什么导致负载不均

继续看 8 个 Token、2 张 GPU：

```text
GPU 0 的 Query: x0, x1, x2, x3
GPU 1 的 Query: x4, x5, x6, x7
```

在 Causal Attention 中：

```text
x0 只能看 x0
x3 能看 x0...x3
x7 能看 x0...x7
```

因此 GPU 0 的 Query 不需要计算后半段 KV，而 GPU 1 的 Query 要计算前后两段 KV。若简单使用连续切分，后部 Rank 的有效 Attention 工作更多，会拖慢整轮同步。

经典做法是把序列切成更多小块，再把“前部小块”和“后部小块”配给同一 Rank：

```text
连续切分:
  GPU 0 = [0,1,2,3]
  GPU 1 = [4,5,6,7]

负载均衡切分示意:
  GPU 0 = [0,1] + [6,7]
  GPU 1 = [2,3] + [4,5]
```

这样每张卡同时拥有较早和较晚的 Query，有效三角区域更接近。该思路常称为 Zigzag 或 Load-balanced Ring Attention。实现还应直接跳过完全被 Causal Mask 遮住的 Q-K Block，避免做完无效计算再把结果置零。

注意：使用这种非连续布局后，不能再用 `cp_rank * chunk` 推断位置。每个本地 Token 必须携带真实的全局 `position_ids`。

## 六、Backward、通信成本与工程正确性

Forward 只解释了结果怎么计算，训练还必须保证 Backward 正确。

### Attention Activation 的梯度

本地 `Q_i` 只属于当前 Rank，因此：

```text
dQ_i 留在本地
```

但同一个 `K_j/V_j` 被所有 Rank 的 Query 使用。每个 Rank 都会产生一部分：

```text
dK_j, dV_j
```

这些贡献必须求和，并让 KV Owner 最终保留自己的梯度分片。从 Collective 语义看：

```text
Forward:  AllGather(K, V)
Backward: ReduceScatter(dK, dV)
```

ReduceScatter 可以理解为“先把各卡的梯度相加，再让每卡只拿回自己负责的一段”。Ring 实现则在多轮 P2P 中重新流动所需状态，并累计属于同一 KV Block 的梯度。

### 参数梯度也必须同步

CP Group 内通常复制相同的模型参数，但每个 Rank 只处理部分 Token，所以每卡得到的：

```text
dW_local
```

也只是全序列参数梯度的一部分。它们必须跨 CP 参数副本求和；若还有 DP，还要同时包含不同数据副本的贡献。

在 Megatron 一类实现中，这通常由 `data-parallel-with-context-parallel` 参数同步组完成。它与 Attention 内部的 `dK/dV ReduceScatter` 是两件事：

```text
dK/dV:
  Activation Gradient，决定梯度回到哪个 Token 分片。

dW:
  Parameter Gradient，决定所有参数副本如何得到相同更新。
```

漏掉后者时程序仍可能运行，但不同 CP Rank 的参数会在一次 Optimizer Step 后分叉。

### 显存和通信量

完整 K/V 的元素数为：

```text
|KV| = 2 * B * S * Nkv * D
```

乘以 dtype 字节数才是实际通信字节。每卡初始已有 `1/p`，Forward 还需接收：

```text
V_fwd_receive ≈ (p - 1) / p * |KV|
```

本地 Query 数只有 `S/p`，所以每卡 Attention 计算量约为：

```text
F_attn_per_rank = O(B * Nq * S^2 * D / p)
```

通信随 `S` 线性增长，Attention 计算随 `S^2` 增长。序列足够长时，每个 KV Block 的计算较重，Ring 更容易隐藏通信；序列短或 CP 过大时，本地矩阵太小，Kernel 利用率下降，P2P 延迟会成为主导。

Ring 的额外显存主要是：

```text
本地 Q/K/V
+ 当前 KV Block
+ 下一块 KV 的双缓冲
+ Online Softmax 状态
```

它不需要同时保存完整 KV，因此峰值通常低于 AllGather。

### 上线前检查四件事

1. **位置和 Mask**：本地 Token 必须保留全局位置；把多篇短文档拼成一个长样本的 Document Packing 场景，还要阻止不同文档互相 Attention。
2. **进程组**：KV 只在对应 CP Group 内交换；参数梯度在正确的参数副本组中同步，不能误用 Global Group。
3. **通信重叠**：用 Nsight Systems 或 PyTorch Profiler 的时间线确认 P2P 与 Attention Kernel 重叠，而不是只看代码里是否使用了 Async API。
4. **CP 大小**：选择刚好解决 OOM 的较小 CP，避免 `S/CP` 太小导致计算碎片化。

语言模型 Loss 也应按 Token 数归约：

```text
global_loss =
  sum(local valid-token loss)
  / sum(local valid-token count)
```

不能直接平均各 Rank 的 Mean Loss，因为 Padding 或 Packed Sequence 可能让有效 Token 数不同。

Megatron Core 的典型配置为：

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
  --context-parallel-size 4 \
  --cp-comm-type p2p \
  --seq-length 32768
```

参数名和支持的通信方式会随 Megatron Core、Transformer Engine 版本变化，实际配置应以对应版本文档为准。

## 七、CP、SP 与 Ulysses 如何选择

“序列并行”在不同论文中含义并不统一。本文按 Megatron 常用术语区分：

| 方法 | 核心动作 | 主要通信 | 解决的问题 |
| --- | --- | --- | --- |
| Megatron SP | 只让部分 TP 区域保持 Sequence Shard | ReduceScatter + AllGather | 降低 LayerNorm/Dropout 等 Activation |
| CP / Ring | Q 留在本地，KV 在 Context Rank 间流动 | P2P Ring 或 KV AllGather | 训练单卡放不下的长 Context |
| Ulysses | Sequence Shard 转为 Head Shard，再转回来 | Attention 前后两次 AllToAll | 按 Head 并行完整序列 Attention |

Megatron SP 依附于 TP。它利用：

```text
AllReduce = ReduceScatter + AllGather
```

让 LayerNorm、Dropout 等区域保存 `[S/TP, H]`，但它本身没有解决“本地 Q 如何读取全局 KV”，所以不能替代 CP。

Ulysses 在 Attention 前执行：

```text
[S/p, N_head] --AllToAll--> [S, N_head/p]
```

每张卡得到完整序列、部分 Attention Head；Attention 后再做逆 AllToAll，恢复 Sequence Shard。它容易复用现有 FlashAttention，但并行度受 Head 可切分性约束。

对 MHA，纯 `a2a` 通常要求 CP 大小不超过并能整除可见的 Attention Head 数。对 GQA/MQA，真正的切分单位是 KV Head；若还启用了 TP，约束通常是：

```text
CP <= global_num_kv_heads / TP
```

并且本地 KV Head 数需要能被 CP 合理切分。部分实现可通过复制 KV Head、不均匀切分或 `a2a+p2p` 放宽限制，但会引入额外通信或实现复杂度。

选择可以简化为：

```text
已经使用 TP，只想进一步省普通 Activation：
  开 Megatron SP。

长 Context 使单个样本的 Activation OOM：
  开 CP。

Head 数充足，AllToAll 网络高效：
  可以采用 Ulysses。

CP 规模很大，单一 Ring 轮次过多：
  考虑 Ulysses x Ring 的二维混合。
```

从零理解 CP，只需抓住一条主线：

```text
序列切分后，大多数算子都能处理本地 Token；
只有 Attention 需要全局上下文；
因此 Q 留在本地，KV 通过 AllGather 或 Ring 到达 Q；
Online Softmax 保证分块结果与完整 Attention 等价；
Backward 再把 dKV 和参数梯度归约到正确的所有者。
```

判断一个 CP 实现是否技术正确，也只需沿这条数据流检查：本地 Token 的全局位置是否正确、每个 Q 是否看到了合法的全局 KV、分块 Softmax 是否正确合并、所有梯度是否回到了正确的 Tensor 或参数副本。

参考资料：

- [Megatron Core Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889)
- [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)
- [Megatron Core Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
