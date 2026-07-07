---
title: 'MHA、MQA、GQA 与 MLA：大模型注意力头和 KV Cache 优化'
description: '从张量形状、KV Cache 成本和手写代码角度讲清 MHA、MQA、GQA、MLA 的区别，理解推理中为什么要减少 KV heads。'
category: '推理优化'
pubDate: '2026-07-07T16:32:00+08:00'
updatedDate: '2026-07-07T16:32:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [先统一符号和形状](#二先统一符号和形状)
3. [MHA：每个 Q head 都有自己的 K/V head](#三mha每个-q-head-都有自己的-kv-head)
4. [MQA：所有 Q heads 共享一组 K/V](#四mqa所有-q-heads-共享一组-kv)
5. [GQA：一组 Q heads 共享一组 K/V](#五gqa一组-q-heads-共享一组-kv)
6. [MLA：压缩 KV Cache 的另一条路线](#六mla压缩-kv-cache-的另一条路线)
7. [KV Cache 成本对比](#七kv-cache-成本对比)
8. [手搓简化版 GQA](#八手搓简化版-gqa)
9. [实现时容易错的点](#九实现时容易错的点)
10. [面试表达](#十面试表达)
11. [总结](#十一总结)

## 一、核心结论

MHA、MQA、GQA、MLA 都在回答同一个问题：

```text
Q 可以有很多 heads，但 K/V 是否也必须有同样多 heads？
```

答案是不一定。

- `MHA`：`num_q_heads == num_kv_heads`。每个 Q head 对应自己的 K/V head。
- `MQA`：`num_kv_heads = 1`。所有 Q heads 共享同一组 K/V。
- `GQA`：`1 < num_kv_heads < num_q_heads`。多个 Q heads 组成一组，共享一组 K/V。
- `MLA`：不只是减少 KV heads，而是把 K/V Cache 压缩到低秩 latent 表示，计算时再恢复或吸收到权重中。

推理优化最关心的是 KV Cache：

```text
KV Cache size ∝ layers * seq_len * num_kv_heads * head_dim * 2 * dtype_size
```

这里的 `2` 是 K 和 V。

所以当 `num_kv_heads` 从 32 降到 8，KV Cache 大约降到原来的 1/4；从 32 降到 1，大约降到原来的 1/32。

## 二、先统一符号和形状

假设：

```text
B = batch size
T = 当前 token 数，prefill 时是 prompt_len，decode 时通常是 1
S = KV Cache 中历史长度
Hq = num_q_heads
Hkv = num_kv_heads
D = head_dim
hidden_size = Hq * D
```

Attention 的核心公式：

```text
score = Q K^T / sqrt(D)
prob = softmax(score)
out = prob V
```

多头注意力中常见张量形状：

```text
Q: [B, T, Hq,  D]
K: [B, S, Hkv, D]
V: [B, S, Hkv, D]
O: [B, T, Hq,  D]
```

注意：`Hq` 和 `Hkv` 不一定相等。MHA 只是其中一种特殊情况。

Decode 阶段通常 `T=1`：

```text
Q:       [B, 1, Hq,  D]
K_cache: [B, S, Hkv, D]
V_cache: [B, S, Hkv, D]
```

Decode 的瓶颈经常不是算 `Q`，而是每生成一个 token 都要读历史 `K_cache` 和 `V_cache`。

## 三、MHA：每个 Q head 都有自己的 K/V head

MHA 是 Multi-Head Attention。

在 MHA 中：

```text
Hq = Hkv
```

例如：

```text
hidden_size = 4096
Hq = 32
D = 128
Hkv = 32
```

那么：

```text
Q: [B, T, 32, 128]
K: [B, S, 32, 128]
V: [B, S, 32, 128]
```

第 `h` 个 Q head 只和第 `h` 个 K/V head 做 attention：

```text
O[:, :, h, :] = Attention(Q[:, :, h, :], K[:, :, h, :], V[:, :, h, :])
```

### MHA 的优点

- 表达能力强。
- 每个 head 都有独立 K/V 子空间。
- 训练和实现最直观。

### MHA 的问题

推理阶段 KV Cache 很大。

以单层、单请求、`S=8192`、`Hkv=32`、`D=128`、FP16 为例：

```text
K cache = 8192 * 32 * 128 * 2 bytes = 67,108,864 bytes ≈ 64 MB
V cache = 64 MB
K + V = 128 MB / layer
```

如果模型有 32 层：

```text
128 MB * 32 = 4096 MB ≈ 4 GB
```

这只是一个请求的 KV Cache。在线服务中 batch 一大，显存压力会非常明显。

## 四、MQA：所有 Q heads 共享一组 K/V

MQA 是 Multi-Query Attention。

在 MQA 中：

```text
Hq > 1
Hkv = 1
```

例如：

```text
Hq = 32
Hkv = 1
D = 128
```

张量形状：

```text
Q: [B, T, 32, 128]
K: [B, S, 1,  128]
V: [B, S, 1,  128]
```

所有 Q head 使用同一组 K/V：

```text
kv_head = 0
O[:, :, h, :] = Attention(Q[:, :, h, :], K[:, :, 0, :], V[:, :, 0, :])
```

### MQA 的 KV Cache 成本

还是 `S=8192`、`D=128`、FP16：

```text
K cache = 8192 * 1 * 128 * 2 bytes = 2 MB
V cache = 2 MB
K + V = 4 MB / layer
```

32 层：

```text
4 MB * 32 = 128 MB
```

相比 MHA 的约 4 GB，MQA 约 128 MB，理论上少 32 倍。

### MQA 的优点

- KV Cache 极小。
- Decode 阶段读 KV 的带宽压力显著降低。
- 在线推理吞吐更友好。

### MQA 的缺点

- 所有 Q heads 共享 K/V，表达能力可能下降。
- 对一些模型和任务，质量可能不如 MHA 或 GQA。
- 对训练和迁移来说，完全共享 K/V 比较激进。

## 五、GQA：一组 Q heads 共享一组 K/V

GQA 是 Grouped-Query Attention。

它位于 MHA 和 MQA 之间：

```text
MHA: Hkv = Hq
MQA: Hkv = 1
GQA: 1 < Hkv < Hq
```

要求：

```text
Hq % Hkv == 0
group_size = Hq / Hkv
```

例如：

```text
Hq = 32
Hkv = 8
group_size = 4
D = 128
```

张量形状：

```text
Q: [B, T, 32, 128]
K: [B, S, 8,  128]
V: [B, S, 8,  128]
```

Q head 到 KV head 的映射：

```text
q_head:  0  1  2  3 | 4  5  6  7 | ... | 28 29 30 31
kv_head: 0  0  0  0 | 1  1  1  1 | ... | 7  7  7  7
```

公式：

```text
kv_head = q_head / group_size
```

如果用整除：

```cpp
int kv_head = q_head / (num_q_heads / num_kv_heads);
```

### GQA 的 KV Cache 成本

`S=8192`、`Hkv=8`、`D=128`、FP16：

```text
K cache = 8192 * 8 * 128 * 2 bytes = 16 MB
V cache = 16 MB
K + V = 32 MB / layer
```

32 层：

```text
32 MB * 32 = 1024 MB = 1 GB
```

对比：

```text
MHA: Hkv=32 -> 4 GB
GQA: Hkv=8  -> 1 GB
MQA: Hkv=1  -> 128 MB
```

GQA 在质量和推理成本之间做折中，因此 LLaMA、Qwen 等很多主流模型使用 GQA。

### GQA 和 MQA 的关系

MQA 是 GQA 的特例：

```text
Hkv = 1
group_size = Hq
```

MHA 也是 GQA 的特例：

```text
Hkv = Hq
group_size = 1
```

所以实现上可以统一成：

```text
kv_head = q_head / group_size
```

当 `group_size=1` 是 MHA；当 `group_size=Hq` 是 MQA。

## 六、MLA：压缩 KV Cache 的另一条路线

MLA 是 Multi-head Latent Attention，DeepSeek 系列模型中常见。

MHA/MQA/GQA 主要调整的是：

```text
KV head 数量
```

MLA 调整的是：

```text
KV Cache 存什么
```

普通 attention 存完整 K/V：

```text
K_cache: [S, Hkv, D]
V_cache: [S, Hkv, D]
```

MLA 的思想是把 K/V 相关信息压缩成 latent：

```text
c_kv = down_proj(x)      # low-rank latent
K/V = up_proj(c_kv)      # 计算时恢复或等价变换
```

可以理解为：

```text
不要把每层每个 token 的完整 K/V 都存下来；
存一个更小的 latent 表示；
需要 attention 时再通过投影得到参与计算的 K/V 信息。
```

### 一个简化数字例子

假设：

```text
Hq = 128
D = 128
完整 KV 维度 = Hq * D = 16384
latent_dim = 512
```

如果存完整 K/V，单 token 单层需要：

```text
K + V: 2 * 16384 = 32768 elements
```

如果存 latent：

```text
c_kv: 512 elements
```

理论存储下降非常明显。当然实际 MLA 还涉及 RoPE 部分、nope/rope 拆分、权重吸收等细节，不能只按这个简单比例估算。

### MLA 的优点

- KV Cache 可以显著压缩。
- 对长上下文推理很友好。
- 配合权重吸收，可以减少显式恢复完整 K/V 的开销。

### MLA 的难点

- 实现比 GQA 更复杂。
- K/V 的 RoPE 部分和非 RoPE 部分处理不同。
- kernel 需要适配 latent cache。
- 权重吸收后的计算图更难直观理解。

面试中不要把 MLA 简化成“KV heads 更少”。更准确的说法是：

```text
GQA 减少 KV heads；
MLA 压缩 KV representation。
```

## 七、KV Cache 成本对比

统一公式：

```text
KV Cache bytes per layer
= S * Hkv * D * 2 * dtype_size
```

其中：

- `S`：上下文长度。
- `Hkv`：KV heads 数量。
- `D`：head_dim。
- `2`：K 和 V。
- `dtype_size`：FP16/BF16 是 2 bytes。

设：

```text
S = 8192
D = 128
dtype_size = 2
layers = 32
```

| 形式 | Hq | Hkv | group_size | 单层 KV Cache | 32 层 KV Cache |
| --- | ---: | ---: | ---: | ---: | ---: |
| MHA | 32 | 32 | 1 | 128 MB | 4 GB |
| GQA | 32 | 8 | 4 | 32 MB | 1 GB |
| MQA | 32 | 1 | 32 | 4 MB | 128 MB |

这也是为什么推理服务非常关心 `num_key_value_heads`。

Prefill 阶段计算量大，通常 GEMM/attention kernel 并行度足；Decode 阶段每次只有一个新 token，要不断读历史 KV Cache，更容易 memory-bound。减少 KV Cache 不只是省显存，也是在减少 decode 读带宽。

## 八、手搓简化版 GQA

下面给一个最小版 GQA 代码。重点不是追求性能，而是把 head 映射讲清楚。

### Python 版

```python
import math
import torch

def gqa_attention(q, k, v, mask=None):
    """
    q: [B, T, Hq, D]
    k: [B, S, Hkv, D]
    v: [B, S, Hkv, D]
    return: [B, T, Hq, D]
    """
    B, T, Hq, D = q.shape
    _, S, Hkv, _ = k.shape

    assert Hq % Hkv == 0
    group_size = Hq // Hkv

    out = torch.empty_like(q)

    for hq in range(Hq):
        hkv = hq // group_size

        # [B, T, D]
        q_h = q[:, :, hq, :]

        # [B, S, D]
        k_h = k[:, :, hkv, :]
        v_h = v[:, :, hkv, :]

        # [B, T, S]
        score = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(D)

        if mask is not None:
            score = score + mask

        prob = torch.softmax(score, dim=-1)

        # [B, T, D]
        out[:, :, hq, :] = torch.matmul(prob, v_h)

    return out
```

这个版本可以直接看出 GQA 的核心：

```python
hkv = hq // group_size
```

如果：

```text
Hq = 8
Hkv = 2
group_size = 4
```

那么：

```text
hq=0,1,2,3 -> hkv=0
hq=4,5,6,7 -> hkv=1
```

### C++ 风格伪代码

下面是更接近手写 kernel 的循环结构：

```cpp
void gqa_decode_one_token(
    const float* q,        // [Hq, D]
    const float* k_cache,  // [S, Hkv, D]
    const float* v_cache,  // [S, Hkv, D]
    float* out,            // [Hq, D]
    int S,
    int Hq,
    int Hkv,
    int D
) {
    int group_size = Hq / Hkv;

    for (int hq = 0; hq < Hq; ++hq) {
        int hkv = hq / group_size;

        // 1. compute score[t] = dot(q[hq], k_cache[t, hkv])
        std::vector<float> score(S);
        for (int t = 0; t < S; ++t) {
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                float qv = q[hq * D + d];
                float kv = k_cache[(t * Hkv + hkv) * D + d];
                sum += qv * kv;
            }
            score[t] = sum / std::sqrt((float)D);
        }

        // 2. softmax
        float max_score = score[0];
        for (int t = 1; t < S; ++t) {
            max_score = std::max(max_score, score[t]);
        }

        float denom = 0.0f;
        for (int t = 0; t < S; ++t) {
            score[t] = std::exp(score[t] - max_score);
            denom += score[t];
        }

        // 3. out[hq] = sum_t prob[t] * v_cache[t, hkv]
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int t = 0; t < S; ++t) {
                float prob = score[t] / denom;
                float vv = v_cache[(t * Hkv + hkv) * D + d];
                acc += prob * vv;
            }
            out[hq * D + d] = acc;
        }
    }
}
```

实际高性能 kernel 不会这样写：

- 不会用 `std::vector`。
- 会做 block/warp 级并行。
- 会用 online softmax。
- 会把 K/V 分块读。
- 会尽量合并访存。
- 可能使用 FlashAttention 或 PagedAttention。

但面试手搓时，这段代码足够说明核心逻辑。

## 九、实现时容易错的点

### 1. 把 K/V 真的 repeat 成 Hq 份

训练框架里有时会写：

```python
k = repeat_kv(k, group_size)
v = repeat_kv(v, group_size)
```

这在语义上简单，但推理服务不能真的把 KV Cache repeat 成 `Hq` 份，否则 GQA 的显存收益就没了。

推理实现更合理的方式是：

```text
KV Cache 只存 Hkv 份；
attention kernel 中用 hq -> hkv 映射读取。
```

### 2. group_size 算反

正确：

```text
group_size = Hq / Hkv
hkv = hq / group_size
```

错误：

```text
group_size = Hkv / Hq
```

### 3. Q projection 和 K/V projection 输出维度不同

GQA 中：

```text
Wq output dim = Hq  * D
Wk output dim = Hkv * D
Wv output dim = Hkv * D
```

不是三个 projection 都输出 `hidden_size`。

例如：

```text
hidden_size = 4096
Hq = 32
D = 128
Hkv = 8

Wq: 4096 -> 4096
Wk: 4096 -> 1024
Wv: 4096 -> 1024
Wo: 4096 -> 4096
```

### 4. KV Cache shape 看 Hkv，不看 Hq

GQA 的 KV Cache 应该是：

```text
K_cache: [S, Hkv, D]
V_cache: [S, Hkv, D]
```

不是：

```text
[S, Hq, D]
```

### 5. MLA 不等同于 GQA

MLA 的核心是 latent compression；GQA 的核心是 KV head sharing。它们都降低 KV Cache，但机制不同。

## 十、面试表达

可以这样回答 MHA、MQA、GQA 的区别：

```text
MHA 中 Q/K/V head 数相同，每个 query head 有独立的 key/value head；
MQA 中所有 query heads 共享一组 key/value head，KV Cache 最小；
GQA 是二者折中，多个 query heads 共享一组 key/value head。
实现时只需要根据 q_head 映射到 kv_head：kv_head = q_head / group_size。
推理中真正重要的是 KV Cache 只按 num_kv_heads 存，而不是把 K/V repeat 到 num_q_heads。
```

可以这样回答为什么 GQA 对推理重要：

```text
Decode 阶段每生成一个 token 都要读取历史 K/V。
KV Cache 大小和 num_kv_heads 成正比。
把 num_kv_heads 从 32 降到 8，可以把 KV Cache 和读带宽大约降到 1/4。
这会降低显存占用，也缓解长上下文 decode 的 memory-bound 问题。
```

可以这样回答 MLA：

```text
MLA 和 GQA 都服务于降低 KV Cache，但路线不同。
GQA 减少 K/V head 数；MLA 把 K/V 信息压缩成低秩 latent cache。
MLA 的实现更复杂，涉及 latent projection、RoPE/nope 拆分和可能的权重吸收。
```

## 十一、总结

MHA、MQA、GQA、MLA 可以放在一条主线上理解：

```text
MHA: 最直观，KV Cache 最大。
MQA: KV Cache 最小，但共享最激进。
GQA: 质量和推理成本之间的主流折中。
MLA: 通过 latent 表示进一步压缩 KV Cache。
```

推理优化里最实用的判断是：

```text
看模型配置中的 num_attention_heads 和 num_key_value_heads。
num_key_value_heads 越小，KV Cache 越小，decode 读带宽压力越低。
```

GQA 的最小实现只有一句核心映射：

```cpp
kv_head = q_head / (num_q_heads / num_kv_heads);
```

但高性能推理框架真正关心的是：如何在不物理复制 K/V 的情况下，让 attention kernel 正确地让多个 Q heads 共享同一组 K/V。
