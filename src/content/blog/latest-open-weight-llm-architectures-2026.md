---
title: '2026 开放权重大模型架构全景：从 Gated DeltaNet、CSA/HCA 到 LatentMoE'
description: '基于官方模型卡、配置文件与技术报告，系统拆解 Qwen3.6、DeepSeek-V4、GLM-5.2、Kimi-K2.7、MiniMax-M2.7、Nemotron 3 Ultra、Gemma 4、Mistral Small 4、Llama 4、gpt-oss、Falcon-H1R 与 RWKV-7 的架构创新及推理代价。'
category: 'Research & Work'
pubDate: '2026-07-30T16:40:14+08:00'
updatedDate: '2026-07-30T16:40:14+08:00'
---

## 目录

1. [范围与阅读方法](#一范围与阅读方法)
2. [当前架构路线图](#二当前架构路线图)
3. [先统一四个基础概念](#三先统一四个基础概念)
4. [Qwen3.6：Gated DeltaNet 与 Full Attention 混合](#四qwen36gated-deltanet-与-full-attention-混合)
5. [DeepSeek-V4：CSA、HCA 与 mHC](#五deepseek-v4csahca-与-mhc)
6. [GLM-5.2：DSA 上的 IndexShare](#六glm-52dsa-上的-indexshare)
7. [Kimi-K2.7 Code：1T MoE、MLA 与 MuonClip](#七kimi-k27-code1t-moemla-与-muonclip)
8. [MiniMax-M2.7：用 10B 激活参数驱动 230B 容量](#八minimax-m27用-10b-激活参数驱动-230b-容量)
9. [Nemotron 3 Ultra：Mamba、LatentMoE 与原生 NVFP4](#九nemotron-3-ultramambalatentmoe-与原生-nvfp4)
10. [Gemma 4：PLE、KV 共享与统一多模态](#十gemma-4plekv-共享与统一多模态)
11. [Mistral Small 4：小激活 MoE、MLA 与统一模式](#十一mistral-small-4小激活-moemla-与统一模式)
12. [Llama 4：iRoPE、Early Fusion 与首代 Llama MoE](#十二llama-4iropeearly-fusion-与首代-llama-moe)
13. [gpt-oss：128-token 局部注意力与 MXFP4](#十三gpt-oss128-token-局部注意力与-mxfp4)
14. [Falcon-H1R：并行 Attention-Mamba 混合块](#十四falcon-h1r并行-attention-mamba-混合块)
15. [RWKV-7：无 Attention、无 KV Cache 的矩阵状态](#十五rwkv-7无-attention无-kv-cache-的矩阵状态)
16. [把关键代价算清楚](#十六把关键代价算清楚)
17. [这些架构如何改变推理系统](#十七这些架构如何改变推理系统)
18. [训练侧真正发生了什么变化](#十八训练侧真正发生了什么变化)
19. [架构选择表](#十九架构选择表)
20. [常见误区](#二十常见误区)
21. [开放权重不等于完全开源](#二十一开放权重不等于完全开源)
22. [参考资料](#二十二参考资料)
23. [总结](#二十三总结)

## 一、范围与阅读方法

本文按 **2026 年 7 月 30 日可验证的公开资料** 整理。纳入范围需要同时满足：

1. 已公开可下载权重，而不是只有网页端或 API。
2. 有官方模型卡、配置文件、代码或技术报告可以交叉核对。
3. 架构上具有代表性，而不只是同一骨干上的一次后训练更新。

“当前最新版本”与“首次引入架构创新的版本”需要区分。例如：

- Kimi-K2.7 Code 是当前较新的 K2 开放检查点，但骨干仍继承 Kimi-K2.5/K2。
- Falcon-H1R 的主要变化是推理后训练，Attention-Mamba 并行骨干来自 Falcon-H1。
- MiniMax-M2.7 的“自我演进”属于训练与 Agent 系统，不是新的 Attention 算子。
- DeepSeek-V4-Pro-DSpark 增加的是推测解码器，不是新的 V4 主模型骨干。

因此，本文对每个模型都分成三层：

```text
主干结构:
  Attention / SSM / Linear Attention / Residual / MoE

训练结构:
  Optimizer / MTP / Low-precision pretraining / RL

系统能力:
  Tool use / Agent swarm / Thinking mode / Speculative decoding
```

这种区分很重要。模型会用发布文案把三者放在一起，但它们对训练框架、推理引擎和 Kernel 的影响完全不同。

## 二、当前架构路线图

当前开放权重大模型已经不再只有“Decoder-only Transformer + GQA + SwiGLU”一条路线。

```text
                         Open-weight LLM
                                |
       +------------------------+------------------------+
       |                        |                        |
  Softmax Attention       Hybrid Sequence Mixer     Recurrent Model
       |                        |                        |
  +----+---------+        +-----+-----------+          RWKV-7
  |              |        |                 |          no KV cache
Dense/SWA    Learned Sparse  Gated DeltaNet   Mamba-2
  |              |        |                 |
Llama 4      DeepSeek-V4  Qwen3.6      Nemotron 3 Ultra
gpt-oss      GLM-5.2                    Falcon-H1R
Gemma 4
       |
       +---------------------- MoE ----------------------+
       |             |              |                   |
 Fine-grained    LatentMoE     Small active ratio   Dense alternative
 DeepSeek/Kimi   Nemotron      MiniMax/gpt-oss      Qwen/Gemma
```

各家的核心问题并不相同：

| 模型 | 主要问题 | 核心结构 |
|---|---|---|
| Qwen3.6 | 长上下文下 Full Attention 太贵 | 3:1 Gated DeltaNet/Full Attention |
| DeepSeek-V4 | 1M 上下文的计算与 KV Cache | CSA + HCA + MQA + mHC |
| GLM-5.2 | DSA Indexer 自身也变贵 | 跨层复用 Top-K 的 IndexShare |
| Kimi-K2.7 | 1T 容量与稳定训练 | MLA + 384 Experts + MuonClip |
| MiniMax-M2.7 | Agent 模型的每 token 成本 | 230B Total / 10B Active |
| Nemotron 3 Ultra | 高吞吐 Agent 推理 | Mamba-2 + LatentMoE + MTP |
| Gemma 4 | 端侧内存与多模态统一 | PLE + KV 共享 + Local/Global |
| Mistral Small 4 | 一个模型统一快答、推理与代码 | 119B/A6.5B MoE + MLA |
| Llama 4 | MoE、多模态和极长上下文 | Early Fusion + iRoPE |
| gpt-oss | 单卡部署与局部长上下文 | Alternating SWA + MXFP4 MoE |
| Falcon-H1R | Attention 质量与 SSM 效率兼得 | 并行 Attention-Mamba Head |
| RWKV-7 | 完全消除 KV Cache 增长 | Generalized Delta Rule RNN |

## 三、先统一四个基础概念

### 3.1 MoE 降低计算量，不自动降低权重内存

标准稀疏 MoE 层可以写成：

```text
score = Router(x)
T = TopK(score, k)

y = SharedExpert(x)
  + sum_{i in T} p_i * Expert_i(x)
```

简化实现：

```python
def sparse_moe(x, router, experts, shared_expert, top_k):
    # x: [tokens, hidden]
    score = router(x).sigmoid()
    weight, expert_id = score.topk(top_k, dim=-1)
    weight = weight / weight.sum(dim=-1, keepdim=True)

    y = shared_expert(x)

    # 真实实现会先按 expert_id 对 token 排序，
    # 再把每个 Expert 的 token 合成 Grouped GEMM。
    for expert in expert_id.unique():
        token_id, slot = (expert_id == expert).nonzero(as_tuple=True)
        y[token_id] += (
            weight[token_id, slot, None]
            * experts[expert](x[token_id])
        )
    return y
```

若模型为 `35B-A3B`，含义是总权重约 35B，每个 token 只经过约 3B 激活参数。它不意味着只需存储 3B：

```text
35B BF16 权重下界:
35e9 * 2 bytes = 70 GB

35B FP8 权重下界:
35e9 * 1 byte = 35 GB

35B 4-bit 权重下界:
35e9 * 0.5 byte = 17.5 GB
```

此外还要加量化 Scale、Embedding、Attention、KV Cache 和运行时 Workspace。

MoE 的系统代价包括：

- Token Dispatch 与 Combine。
- Expert Parallel 下的 All-to-All。
- Expert 负载不均衡。
- 小 Batch 时大量 Expert GEMM 退化成窄矩阵乘。
- 权重仍需常驻 HBM，除非做 Offload。

### 3.2 KV Cache 的统一计算方法

常规 GQA 每层、每个 token 的 KV Cache 字节数为：

```text
KV_bytes_per_token_per_layer
  = 2 * H_kv * D_head * bytes_per_element
    ^   ^
    |   +-- Key 或 Value 的 Head 数
    +------ K 和 V 两份
```

例如 `H_kv=8`、`D_head=128`、BF16：

```text
2 * 8 * 128 * 2 = 4096 bytes
```

如果有 60 层、128K 上下文、Batch=1：

```text
4096 * 60 * 131072
  = 32,212,254,720 bytes
  ≈ 30 GiB
```

这就是 MLA、GQA、KV Sharing、Sliding Window 和线性模型不断被采用的直接原因。

### 3.3 Linear Attention 与 Sparse Attention 不是一回事

```text
Linear Attention:
  不保存所有历史 K/V
  历史被压缩进固定大小状态 S
  Decode 每 token 成本近似 O(1)

Sparse Attention:
  仍保存或压缩历史条目
  每个 Query 只选择 k 个历史位置
  Decode 每 token 主 Attention 成本近似 O(k)

Sliding Window:
  只看最近 W 个位置
  Decode 每 token 成本 O(W)

Full Attention:
  查看全部 L 个历史位置
  Decode 每 token 成本 O(L)
```

线性模型节省最多，但可能损失精确检索；稀疏 Attention 保留基于内容的跳跃检索，但必须支付 Indexer 和 Top-K 代价。

### 3.4 MTP 既是训练目标，也能成为推测解码器

普通语言模型只预测下一个 token：

```text
p(x_{t+1} | x_{\le t})
```

Multi-Token Prediction 同时训练多个未来位置：

```text
L_MTP = sum_{j=1}^{n} lambda_j
        * CE(p_j(x_{t+j} | x_{\le t}), x_{t+j})
```

推理时，MTP Head 可以一次草拟多个 token，再由主模型批量验证：

```text
Context -> MTP draft: [d1, d2, d3, d4]
        -> Target verify in one forward
        -> Accept longest valid prefix
```

MTP 不保证加速。收益取决于：

- 平均接受长度。
- Draft Head 的额外计算。
- Batch 大小时验证阶段的吞吐。
- CUDA Graph 是否能覆盖动态接受长度。
- 调度器能否避免低置信度尾部浪费验证算力。

## 四、Qwen3.6：Gated DeltaNet 与 Full Attention 混合

### 4.1 当前开放版本

Qwen3.6 当前有两条主要开放路径：

| 模型 | 总参数 | 激活参数 | 层数 | 上下文 |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 35B | 3B | 40 | 262,144 |
| Qwen3.6-27B | 27B Dense | 27B | 64 | 262,144 |

35B-A3B 的官方配置为：

```text
hidden_size             = 2048
num_hidden_layers       = 40
num_experts             = 256
num_experts_per_tok     = 8
shared_expert           = 1
moe_intermediate_size   = 512
full_attention_interval = 4
```

其 40 层并不是 40 个标准 Attention，而是：

```text
10 * [
  Gated DeltaNet -> MoE
  Gated DeltaNet -> MoE
  Gated DeltaNet -> MoE
  Full Attention -> MoE
]
```

即 30 层 Gated DeltaNet，10 层 Full Attention。

### 4.2 Gated DeltaNet 在做什么

普通线性 Attention 可以维护一个状态矩阵：

```text
S_t = S_{t-1} + v_t k_t^T
o_t = S_t q_t
```

问题是同一个 Key 再次出现时只能继续累加，不能精确改写旧关联。Delta Rule 把状态看作一个在线学习的映射，先读取预测，再按误差更新：

```text
prediction_t = S_{t-1} k_t
error_t      = v_t - prediction_t

S_t = S_{t-1}
    + beta_t * error_t k_t^T
```

Gated DeltaNet 再加入遗忘门：

```text
S_t = alpha_t * S_{t-1}
    + beta_t * (v_t - S_{t-1} k_t) k_t^T

o_t = S_t q_t
```

其中：

- `alpha_t` 控制旧记忆保留多少。
- `beta_t` 类似当前 token 的在线学习率。
- `v_t - S_{t-1}k_t` 只更新预测错误的部分。

可读性优先的单头伪代码如下：

```python
def gated_delta_step(q, k, v, state, alpha, beta):
    # q, k: [d_k]
    # v:    [d_v]
    # state:[d_v, d_k]
    prediction = state @ k
    error = v - prediction

    state = alpha * state
    state = state + beta * error[:, None] * k[None, :]
    output = state @ q
    return output, state
```

训练时不能真的逐 token 跑 Python 循环。高性能实现把序列切成 Chunk：

```text
Chunk 内:
  计算 Q/K/V、门控和局部依赖
  使用并行 Scan / WY 表示处理递推

Chunk 间:
  只传递固定大小状态 S
```

Prefill 复杂度由 Full Attention 的 `O(L^2)` 降到 GDN 层的近似 `O(L)`；Decode 不再随着上下文长度增加 K/V 读取量。

### 4.3 为什么还保留每四层一次 Full Attention

固定状态必须把历史压缩进去，无法像 Softmax Attention 那样直接定位任意历史 token。Full Attention 层用于恢复：

- 精确 Copy。
- Needle-in-a-haystack 式检索。
- 跨很远位置的强离散关联。
- 视觉 token 与文本 token 的全局对齐。

Qwen3.6 的结构本质是：

```text
GDN:
  低成本连续记忆

Full Attention:
  周期性全局校正与精确读取
```

### 4.4 Full Attention 本身也做了压缩

35B-A3B 的 Full Attention 配置为：

```text
Q heads  = 16
KV heads = 2
head_dim = 256
partial RoPE = 0.25
```

BF16 下，每个 Full Attention 层每 token 的 KV Cache 为：

```text
2 * 2 * 256 * 2 = 2048 bytes
```

只有 10 个 Full Attention 层随上下文增长：

```text
2048 * 10 = 20 KiB/token

256K context:
20 KiB * 262144 ≈ 5 GiB/request
```

这仍然不小，但比 40 层全部 Full Attention 约少 4 倍。GDN 层改为固定状态，代价从“随 L 增长”变成“随 Batch 增长”。

### 4.5 原生多模态

Qwen3.6 不是给纯文本模型外接一个视觉 Adapter。其配置包含 27 层视觉编码器：

```text
Vision input
  -> Conv3D patch embedding
  -> Vision Transformer
  -> Spatial merge
  -> Project to language hidden size
  -> Interleave with text tokens
  -> Shared hybrid language backbone
```

`temporal_patch_size=2` 让视频的相邻帧在 Patch Embedding 阶段直接进入时间维处理。mRoPE 将位置维度拆给时间、高度和宽度。

这里的“原生”主要指视觉-语言 token 在预训练阶段共同进入主干，而不是声称视觉编码器和语言模型完全没有边界。

### 4.6 推理系统含义

Qwen3.6 要求推理引擎同时具备：

- Gated DeltaNet Prefill Chunk Kernel。
- Gated DeltaNet Recurrent Decode Kernel。
- Full Attention Paged KV Cache。
- MoE Grouped GEMM。
- Shared Expert 与 Routed Expert 融合。
- MTP 推测解码。
- 多模态 Conv3D 与视觉 Token Merge。

它不是“把 FlashAttention 换掉”这么简单，而是同一模型内维护两套 Sequence State。

## 五、DeepSeek-V4：CSA、HCA 与 mHC

### 5.1 V4 不再只是 MLA + DSA

DeepSeek-V4 当前包含：

| 模型 | 总参数 | 激活参数 | 层数 | Routed Experts | Top-K |
|---|---:|---:|---:|---:|---:|
| V4-Pro | 1.6T | 49B | 61 | 384 | 6 |
| V4-Flash | 284B | 13B | 43 | 256 | 6 |

两者都支持 1,048,576 token。V4 的主要变化是：

1. 用 CSA/HCA 混合注意力取代 V3 系列 MLA 主路径。
2. 用 Manifold-Constrained Hyper-Connections 改造残差流。
3. 前几层使用静态 Hash MoE。
4. Routed Expert 原生采用 FP4，其他敏感路径保持更高精度。
5. 预训练主要使用 Muon Optimizer。

### 5.2 CSA：先压缩，再按内容选择

Compressed Sparse Attention 的数据流是：

```text
历史 token
  -> overlapping pooling, compression rate m=4
  -> compressed K/V blocks
  -> Lightning Indexer scores all compressed blocks
  -> Top-K compressed blocks
  -> concatenate local sliding-window K/V
  -> core attention
```

记原始序列长度为 `L`，压缩率为 `m`，选中块数为 `k`：

```text
Compressed length ≈ L / m
Indexer cost       ≈ O(L / m)
Core attention     ≈ O(k + W)
```

V4-Pro 配置中：

```text
CSA compression m = 4
index_topk         = 1024
local window W     = 128
```

在 1M 上下文时，主 Attention 每个 Query 实际处理的条目近似：

```text
1024 selected compressed entries + 128 local entries
= 1152 entries
```

相对 1,048,576 个 Dense Key：

```text
1,048,576 / 1,152 ≈ 910
```

即 Core Attention 的打分位置约少 910 倍。注意这 **不等于整层快 910 倍**，因为还要付：

- 压缩 K/V 的成本。
- Indexer 扫描约 `L/4` 个候选的成本。
- Top-K。
- Gather 非连续 Block。
- Query、Output Projection 和 MoE。

### 5.3 HCA：不做 Top-K，极重压缩后全部读取

Heavily Compressed Attention 使用更大的压缩率：

```text
HCA compression m' = 128
Compressed length  ≈ L / 128
```

1M 上下文约得到：

```text
1,048,576 / 128 = 8192 entries
```

HCA 不需要 Indexer，直接对全部压缩条目做 Attention：

```text
HCA:
  较粗粒度的全局概览
  无 Top-K 不规则访存

CSA:
  较细粒度的内容检索
  需要 Indexer 和 Gather
```

V4 将两者交替使用，使模型同时拥有便宜的全局摘要和较精确的稀疏检索。

### 5.4 Shared K=V MQA

V4 配置中：

```text
num_key_value_heads = 1
attention_k_eq_v    = true in the implementation path
head_dim            = 512
```

普通 Attention 保存 K 和 V 两份；V4 的 Long-range 主干让 K 与 V 共享同一表示：

```text
Standard:
  Cache = [K, V]

K=V:
  Cache = [C]
  K <- C
  V <- C
```

这既减少 KV Cache，也减少压缩池写入带宽。代价是 K 与 V 的表达被耦合，所以 V4 又通过更宽 Head、分组低秩输出投影和局部分支补偿表达能力。

### 5.5 Grouped Low-rank Output Projection

V4-Pro 有 128 个 Query Head，若直接把所有输出拼接后做宽矩阵投影，成本很高。配置使用：

```text
o_groups    = 16
o_lora_rank = 1024
```

简化数据流：

```text
128 heads
  -> split into 16 groups
  -> each group projects to low-rank space
  -> concatenate low-rank features
  -> project back to hidden_size=7168
```

它与 MLA 的动机相近，都是用低秩空间削减宽 Attention Projection，但压缩位置不同。

### 5.6 mHC：残差不再只有一条流

标准 Pre-Norm Transformer：

```text
x' = x + Attention(Norm(x))
x'' = x' + FFN(Norm(x'))
```

深层大 MoE 中，所有子层都向同一残差流累加，表示可能在部分方向上不断放大。V4 使用 `hc_mult=4`，维护四条残差流：

```text
X in R^[4, D]

pre-mix:
  x_in = combine four streams

sub-layer:
  y = F(x_in)

post-mix:
  X' = P X + inject(y)
```

关键约束是混合矩阵 `P` 经过 Sinkhorn-Knopp 迭代投影为近似双随机矩阵：

```text
P_ij >= 0
sum_j P_ij = 1
sum_i P_ij = 1
```

双随机约束限制残差混合的无界放大，同时允许模型在四条流之间重排信息。V4 配置使用 20 次 Sinkhorn 迭代。

从系统角度看，mHC 的代价不是参数量，而是 Activation：

```text
普通残差状态: [B, L, D]
mHC=4:       [B, L, 4, D]
```

训练框架必须用融合与重计算控制显存，不能直接把所有中间流完整保存。

### 5.7 前三层 Hash MoE

V4 的前 3 个 MoE 层不是完全由学习 Router 决定 Expert ID，而是使用静态：

```text
token_id -> expert_id
```

Hash 表决定“去哪个 Expert”，学习到的 Gate Score 仍决定权重。这样做的作用包括：

- 训练初期避免 Router 抖动。
- 高频 Token 的底层词法模式得到稳定分工。
- 降低前几层路由不平衡风险。

高层恢复普通 Top-K MoE，因为语义路由不能只依赖 Token ID。

### 5.8 FP4 Expert 与 Muon

V4 将大多数 Routed Expert 权重放到 FP4，而 Attention、Embedding、Shared Expert 等敏感模块保持 FP8 或更高精度。

这是合理的非均匀精度分配：

```text
Routed Experts:
  参数占比最大
  每 token 只激活少数
  适合用 FP4 降权重带宽

Attention / Router / Norm:
  参数较少
  数值误差会影响所有 token
  保持高精度
```

Muon 对二维权重梯度做近似正交化更新，常用 Newton-Schulz 迭代实现。它可能提高 Token Efficiency，但分布式训练需要额外解决：

- Optimizer State Sharding。
- 大矩阵正交化通信。
- Embedding、Norm 等非二维参数仍使用 AdamW。
- 低精度下的更新稳定性。

### 5.9 V4 的实际收益

官方报告给出的关键对比是：在 1M 上下文下，V4-Pro 相对 V3.2：

```text
single-token inference FLOPs: 27%
KV Cache:                     10%
```

这比单纯把 DSA 的 `Top-K` 再减小更系统，因为 V4 同时改了：

- 被检索的表示粒度。
- Indexer 候选长度。
- K/V 表达。
- 局部路径。
- Output Projection。

## 六、GLM-5.2：DSA 上的 IndexShare

### 6.1 与 DeepSeek-V3.2 的关系

GLM-5.2 使用 `glm_moe_dsa`，核心配置为：

```text
total / active params = about 744B / 40B
num_hidden_layers     = 78
hidden_size           = 6144
first dense layers    = 3
routed experts        = 256
selected experts      = 8
shared experts        = 1
kv_lora_rank          = 512
index heads           = 32
index head dim        = 128
index top-k           = 2048
context               = 1,048,576
```

它沿用 MLA + DeepSeek Sparse Attention 的基本结构：

```text
Hidden State
  -> MLA low-rank Q/KV representation
  -> Lightning Indexer scores history
  -> Top-2048 positions
  -> Sparse MLA
```

### 6.2 DSA 的 Indexer 并不免费

对 Query `t` 和历史位置 `s`，Lightning Indexer 使用多个轻量 Head 计算：

```text
I(t,s) =
  sum_j w(t,j) * ReLU(
      dot(q_index(t,j), k_index(s)) / sqrt(d_index)
  )
```

然后：

```text
selected(t) = TopK_s(I(t,s), 2048)
```

主 Attention 只处理 2048 个 token，但 Indexer 仍需扫描 `L` 个历史位置。1M Context 下，Indexer Dot Product 和 Top-K 会成为显著成本。

### 6.3 IndexShare：四层共享一次检索结果

GLM-5.2 的关键变化不是继续缩小 Top-K，而是观察到相邻层需要的相关历史位置往往相近。

```text
Layer 6:
  compute index score
  topk -> indices_6

Layer 7:
  reuse indices_6

Layer 8:
  reuse indices_6

Layer 9:
  reuse indices_6

Layer 10:
  compute new indices_10
```

配置中的：

```text
index_topk_freq        = 4
index_skip_topk_offset = 3
```

对应前三层保留完整路径，之后每四层一次 Full Indexer，其余层标为 `shared`。

复用的是：

```text
Top-K indices
```

而不是：

- 上一层的 Attention Output。
- Index Score。
- K/V Cache。
- Softmax 权重。

每层仍用自己的 Query 和 K/V 在同一组历史位置上重新计算 Attention。

### 6.4 为什么跨层复用可行

相邻 Transformer 层的表示会变化，但长上下文中“哪些历史段与当前 Query 相关”通常不会每层完全重排。例如当前正在修改一个 CUDA Kernel：

```text
相关历史位置:
  函数定义
  编译报错
  调用点
  前一次修改
```

这些位置在连续四层内大概率稳定。IndexShare 用少量选择灵活性换取：

- 省掉 3/4 Indexer GEMM。
- 省掉 3/4 大规模 Top-K。
- 降低临时 Score Buffer。
- 提高 CUDA Graph 稳定性。

官方给出的 1M Context 每 token FLOPs 降低为 2.9 倍，而不是严格 4 倍，因为 MoE、Sparse Attention 和 Projection 本身没有消失。

### 6.5 MTP 也共享信息

GLM-5.2 还改进了 MTP：

```text
Main model:
  DSA IndexShare + KV Cache

MTP layer:
  reuse index selections
  reuse compatible KV representation
  predict draft tokens
```

减少 Draft Layer 重做长上下文检索的成本后，MTP 平均接受长度最高提升约 20%。这是一个重要原则：

> 推测解码器如果复制完整长上下文 Attention，Draft 可能比节省的主模型步骤还贵。

## 七、Kimi-K2.7 Code：1T MoE、MLA 与 MuonClip

### 7.1 版本边界

当前 K2 家族中：

- Kimi-K2.6 是通用版本。
- Kimi-K2.7 Code 是更新的代码与长程 Agent 专用版本。
- K2.7 仍使用 K2.5 公开的多模态骨干配置。

K2.7 Code 的官方结构为：

```text
total parameters       = 1T
active parameters      = 32B
layers                 = 61
dense layers           = 1
hidden size            = 7168
routed experts         = 384
experts per token      = 8
shared experts         = 1
expert hidden size     = 2048
attention              = MLA
context                = 256K
vision encoder         = MoonViT, 400M
```

### 7.2 MLA 如何压缩 KV Cache

普通多头 Attention 为每个 Head 保存 K/V。MLA 先把 Hidden State 压到共享 Latent：

```text
c_KV = W_DKV x

k_nope = W_UK c_KV
v      = W_UV c_KV

k_rope = W_KR x
```

推理时缓存：

```text
[c_KV, k_rope]
```

而不是缓存每个 Head 的完整 `[K_h, V_h]`。

K2 配置：

```text
num_heads       = 64
qk_nope_dim     = 128
qk_rope_dim     = 64
v_dim           = 128
kv_lora_rank    = 512
```

若按 BF16 粗略估算，普通 MHA 缓存为：

```text
64 * (128 + 64 + 128) * 2 bytes
= 40,960 bytes/layer/token
```

MLA 缓存 Latent 与 RoPE Key：

```text
(512 + 64) * 2
= 1,152 bytes/layer/token
```

理论压缩比：

```text
40,960 / 1,152 ≈ 35.6x
```

真实 Kernel 会通过 Weight Absorption 改写矩阵乘顺序，量化和 Cache Layout 也会改变实际字节数，但数量级差异成立。

### 7.3 384 个 Expert 为什么不是简单“384 个领域模型”

每个 token 在每一层都重新路由：

```text
token "CUDA"
  layer 2  -> experts [7, 18, 41, ...]
  layer 3  -> experts [2, 41, 99, ...]
  layer 4  -> experts [5, 66, 71, ...]
```

Expert 的专门化通常是潜在特征组合，不一定能解释成“数学专家”“代码专家”。1T 总参数带来容量，32B 激活量控制单 token FLOPs，但部署仍面临：

- 1T 权重存储与加载。
- 384 Expert 的并行切分。
- 每层 Top-8 的 All-to-All。
- Shared Expert 的 Dense 路径。

### 7.4 MuonClip 解决什么

Kimi K2 训练中，Muon 提高了 Token Efficiency，但部分 Attention Head 出现 Logit 爆炸：

```text
score_h = Q_h K_h^T / sqrt(d)
S_max_h = max(score_h)
```

当 `S_max_h` 远超阈值 `tau` 时，Softmax 极度饱和并可能引发 Loss Spike。MuonClip 计算：

```text
gamma = min(1, tau / S_max)
```

再按平衡系数 `alpha` 缩放 Q/K Projection：

```text
W_Q <- gamma^alpha       * W_Q
W_K <- gamma^(1 - alpha) * W_K
```

常用 `alpha=0.5` 时，Q/K 各承担平方根比例的缩放，二者点积整体约乘 `gamma`。

这和 Gradient Clipping 不同：

```text
Gradient Clipping:
  限制更新量

QK-Clip:
  直接约束造成 Softmax 饱和的前向 Attention Logit
```

官方报告称 K2 在 15.5T Token 预训练中没有 Loss Spike。

### 7.5 MoonViT 与原生多模态

K2.5 以后加入 400M MoonViT：

```text
Image / Video
  -> patch_size 14
  -> 27-layer Vision Transformer
  -> spatial-temporal attention
  -> 2x2 merge + temporal pooling
  -> projector to hidden_size 7168
  -> insert media tokens into language sequence
```

它通过持续预训练约 15T 混合视觉与文本 Token，把视觉能力纳入主干，而不是只训练一个小 Projector。

### 7.6 Agent Swarm 不是网络层

Kimi 的 Agent Swarm 是推理时编排：

```text
Main Agent
  -> decompose task
  -> spawn sub-agents
  -> parallel tool use
  -> aggregate results
```

它不意味着一次 `model.forward()` 内部出现多个 Agent。系统收益来自并行搜索空间，代价是更多 Token、更多上下文合并和更复杂的调度。

## 八、MiniMax-M2.7：用 10B 激活参数驱动 230B 容量

### 8.1 极小激活比

MiniMax-M2.7 延续 M2 骨干：

```text
total parameters       ≈ 230B
active parameters      ≈ 10B
active ratio           ≈ 4.3%
layers                 = 62
hidden size            = 3072
experts                = 256
experts per token      = 8
expert hidden size     = 1536
context                = 204,800
```

它没有 Shared Expert，所有 FFN 都通过 Routed Expert 完成。前向数据流：

```text
x
 -> GQA
 -> sigmoid router + correction bias
 -> Top-8 / 256 experts
 -> weighted combine
 -> residual
```

### 8.2 Sigmoid Routing 与校正 Bias

Softmax Router 在 Expert 维度强制竞争：

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

Sigmoid Router 独立计算每个 Expert 亲和度：

```text
s_i = sigmoid(z_i + b_i)
T   = TopK(s, 8)
```

其中 `b_i` 是根据 Expert 负载更新的校正 Bias。负载过高的 Expert 降低 Bias，负载过低的 Expert 提高 Bias。

好处是：

- 不必让语义 Gate Score 承担全部负载均衡压力。
- 可降低 Auxiliary Loss 对主任务梯度的干扰。
- 每个 Expert 的亲和度不受大量无关 Expert 的 Softmax 分母影响。

### 8.3 GQA 与 Partial RoPE

配置为：

```text
Q heads   = 48
KV heads  = 8
head_dim  = 128
rotary_dim= 64
```

BF16 KV Cache 每层每 token：

```text
2 * 8 * 128 * 2 = 4096 bytes
```

62 层、200K Context：

```text
4096 * 62 * 204800
≈ 52.0 GB/request
```

这说明 200K 是模型能力上限，不代表单请求可无代价跑满。生产部署仍需：

- FP8 KV Cache。
- Context Parallel。
- Prefix Cache。
- Chunked Prefill。
- 限制并发长请求。

### 8.4 多步 MTP

配置中：

```text
num_mtp_modules       = 3
mtp_transformer_layers = 1
```

即最多可形成三步 Draft，而不是只有一个 `t+2` Head。训练时提供额外未来 Token 监督，推理时可以作为多步 Speculative Path。

### 8.5 “自我演进”属于训练闭环

M2.7 的自我演进流程是：

```text
Model analyzes failed trajectories
  -> proposes scaffold / training changes
  -> runs experiments
  -> evaluates artifacts
  -> keeps or reverts changes
```

它改变数据、Agent Scaffold 和 RL 过程，不改变一次前向传播中的 MoE/GQA 数学结构。把它描述成“模型内部自带自我修改模块”并不准确。

## 九、Nemotron 3 Ultra：Mamba、LatentMoE 与原生 NVFP4

### 9.1 总体结构

Nemotron 3 Ultra：

```text
total parameters  = 550B
active parameters = 55B
context           = up to 1M
hidden size       = 8192
architecture      = Mamba-2 + Attention + LatentMoE
routed experts    = 512
experts per token = 22
shared experts    = 1
```

层序列不是每层都同时做所有操作，而是交错：

```text
Mamba -> MoE
Mamba -> MoE
Mamba -> MoE
Mamba -> Attention -> MoE
...
```

多数 Token Mixing 由 Mamba-2 完成，少量 Attention 层负责精确全局检索。

### 9.2 Mamba-2 为什么适合长程 Agent

Mamba-2 使用固定大小状态递推：

```text
h_t = A_t h_{t-1} + B_t x_t
y_t = C_t h_t + D x_t
```

其中 `A_t/B_t/C_t` 由当前输入选择性生成。相比 Full Attention：

```text
Attention Decode:
  read all historical KV
  bytes grow with context L

Mamba Decode:
  read and update fixed-size state
  bytes independent of L
```

Nemotron 配置中 Mamba 使用：

```text
mamba_num_heads = 256
mamba_head_dim  = 64
state_size      = 128
conv_kernel     = 4
chunk_size      = 128
```

Mamba 负责大部分长序列吞吐，Attention 负责修复纯状态压缩模型的精确召回短板。

### 9.3 LatentMoE

普通 Expert 在完整 Hidden 维度 `d` 上运行：

```text
x in R^d -> Expert_i(x) -> y in R^d
```

LatentMoE 先降维：

```text
z = W_down x, z in R^l
y = W_up sum_i p_i Expert_i(z)
```

Ultra 使用：

```text
d = 8192
l = 2048
compression = 4x
```

即：

```text
8192
 -> down-project to 2048
 -> route and run 22 latent experts
 -> up-project to 8192
```

Expert Parallel 的 Token 通信量近似与被路由向量宽度成正比：

```text
standard communication ~ tokens * d
latent communication   ~ tokens * l

d / l = 4
```

因此 All-to-All Payload 约缩小 4 倍。NVIDIA 把节省的计算重新投入到更高 Top-K：

```text
512 experts, Top-22
```

更高 Top-K 提供更多 Expert 组合，同时不让通信按完整 8192 维爆炸。

### 9.4 原生 NVFP4 预训练

Post-training Quantization 的流程是：

```text
BF16 trained model
  -> calibration
  -> quantize to FP4
  -> repair accuracy if needed
```

原生 NVFP4 训练让模型从训练阶段适应 4-bit Microscaling：

```text
forward low precision
backward/update with higher-precision master state
block scaling controls local dynamic range
```

Ultra 同时发布 BF16 与 NVFP4。需要注意：

- NVFP4 权重最适合 Blackwell Tensor Core。
- 在 Hopper 上可能发生解量化或替代 Kernel。
- Router、Norm、部分 Attention 路径不能全部粗暴降到 4 bit。
- “550B NVFP4”仍然是多 GPU 数据中心模型。

### 9.5 MTP Boosting

Ultra 的 MTP 不只在预训练阶段存在，后训练还专门提高 Draft Token 的接受率。MTP Block 包含 Attention + MoE，使 Draft Head 能利用主干表示继续建模未来 token 依赖。

对于 Agent 长输出：

```text
64K output tokens
```

即使每次平均多接受 1 个 token，也可能近似减少一半主模型串行步数。实际收益取决于验证 Batch 和服务负载。

## 十、Gemma 4：PLE、KV 共享与统一多模态

### 10.1 家族不是一种完全相同的模型

Gemma 4 包含：

| 模型 | 总参数 | 有效/激活参数 | 主要定位 |
|---|---:|---:|---|
| E2B | 约 5.1B | 2.3B Effective | 端侧 |
| E4B | 约 8B | 4.5B Effective | 端侧 |
| 12B | 12B Dense | 12B | Encoder-free Any-to-Any |
| 26B-A4B | 25.2B | 3.8B Active | MoE |
| 31B | 30.7B Dense | 30.7B | 高质量 Dense |

“Effective Parameter”与 MoE Active Parameter 不完全相同：

- E2B/E4B 的额外参数主要来自 PLE 表，单 token 只做少量 Lookup。
- 26B-A4B 的 Active Parameter 来自 Top-K Expert Routing。

### 10.2 5:1 Local/Global Attention

26B-A4B 配置为 30 层：

```text
[Local, Local, Local, Local, Local, Global] * 5
```

Local Window 为 1024。其作用是：

```text
Local layers:
  建模语法、局部模式
  KV 只保留固定窗口

Global layers:
  建模跨文档依赖
  KV 随上下文增长
```

这比每层 Full Attention 显著节省 KV Cache 和 Prefill FLOPs，又比纯 Sliding Window 保留更短的跨层信息传播路径。

### 10.3 Local 与 Global 使用不同 RoPE 和 Head

26B-A4B：

```text
Local:
  head_dim   = 256
  rope_theta = 10,000
  window     = 1024

Global:
  head_dim            = 512
  rope_theta           = 1,000,000
  partial rotary ratio = 0.25
```

局部层不需要极低频位置编码；全局层使用更低频、部分旋转的 p-RoPE，降低长上下文高频位置混叠。

### 10.4 K=V 与跨层 KV Sharing

Gemma 4 在 Global Attention 中复用 Key 作为 Value：

```text
attention_k_eq_v = true
```

同时，E2B/E4B 的部分后续层不再生成自己的 K/V，而是读取最近一个同类型 Producer Layer 的 KV Cache：

```text
Producer layer:
  K,V = project(x)
  write cache slot

Consumer layers:
  Q = project(x)
  read producer's K,V slot
  no own K/V projection and cache
```

E2B 35 层中只有 15 个 KV Producer，20 层复用。KV Cache Slot 数量变为：

```text
15 / 35 = 42.9%
```

但不同 Layer 的 Q 仍然不同，所以 Attention Output 不是简单复制。

官方报告给出的组合收益是 Global KV Cache 最多减少 37.5%。实际比例取决于模型尺寸、Local/Global 层数和共享布局。

### 10.5 PLE：参数很多，但每 token 只取很小一片

Per-Layer Embedding 为每个 Layer 提供独立 Token Embedding：

```text
main_embedding[token] -> x

for layer l:
    e_l = per_layer_embedding[l, token]
    x = layer(x, e_l)
```

E2B 的典型 PLE 维度为：

```text
vocab = 262,144
layers = 35
per-layer input dim = 256
```

完整 PLE 表参数很多，但一次请求中每个 token、每层只读取 256 维向量。它用 Weight Capacity 换 Dense GEMM FLOPs：

```text
传统扩容:
  增大 hidden/intermediate
  每 token GEMM 都更贵

PLE 扩容:
  增大 token-indexed table
  每 token 主要增加 lookup bandwidth
```

这也是 E2B 文件参数量约 5.1B，却称 2.3B Effective 的原因。

### 10.6 12B Encoder-free

其他多模态模型通常：

```text
Image -> Vision Encoder -> Projector -> LLM
Audio -> Audio Encoder  -> Projector -> LLM
```

Gemma 4 12B Unified 尝试：

```text
Raw image patches
Raw 40 ms audio chunks
  -> lightweight projection to token embedding
  -> shared language backbone
```

它减少独立 Encoder 带来的：

- 权重切换。
- 显存碎片。
- 不同 Encoder 的调度。
- 多模态特征对齐边界。

但原始 Patch/Audio Token 数量更高，需要主干本身承担更多感知计算。

### 10.7 MoE 版本

26B-A4B 配置：

```text
experts        = 128
top_k_experts  = 8
active params  ≈ 3.8B
```

其激活比例约：

```text
3.8 / 25.2 ≈ 15.1%
```

相比 MiniMax 的 4.3% 更稠密，但模型总权重更小，更适合本地或单机多卡部署。

## 十一、Mistral Small 4：小激活 MoE、MLA 与统一模式

### 11.1 当前结构

Mistral Small 4：

```text
total parameters  = 119B
active parameters = 6.5B
layers            = 36
hidden size       = 4096
experts           = 128
experts per token = 4
shared experts    = 1
context           = officially 256K
modalities        = text + image -> text
```

模型配置中的最大位置上限为 1M，但官方模型卡的稳定支持范围是 256K。配置上限、RoPE 外推上限和官方验证上下文不能混为一谈。

### 11.2 MLA

配置：

```text
q_lora_rank     = 1024
kv_lora_rank    = 256
qk_nope_dim     = 64
qk_rope_dim     = 64
v_dim           = 128
num_heads       = 32
```

其 MLA Cache 主体约为：

```text
(kv_lora_rank + qk_rope_dim) * 2 bytes
= (256 + 64) * 2
= 640 bytes/layer/token
```

普通 32-Head MHA 粗略为：

```text
32 * (128 Key + 128 Value) * 2
= 16,384 bytes/layer/token
```

理论压缩约：

```text
16,384 / 640 = 25.6x
```

### 11.3 Top-4 大 Expert

Mistral 3/4 的路线不是把 Expert 切得极细，而是使用较少、较大的 Expert：

```text
128 experts
Top-4
```

相对 Top-8/Top-22：

- Token Dispatch 份数更少。
- All-to-All Payload 更小。
- 单个 Expert 获得更多 Token，更容易形成适合 Tensor Core 的 GEMM。
- Expert 组合数减少。

这是对部署效率更直接的优化。

### 11.4 一个 Checkpoint 切换三种能力

Mistral Small 4 把此前分开的能力统一到同一骨干：

```text
Instruct:
  快速普通回答

Reasoning / Magistral:
  输出更长推理轨迹

Devstral:
  代码与 Agent 工作流
```

切换主要由 Chat Template、控制 Token 与后训练决定，不是运行时动态增加网络层。

### 11.5 EAGLE Head 是独立 Draft Model

Mistral 提供约数百 MB 的 EAGLE Head：

```text
Main model hidden state
  -> EAGLE draft head
  -> draft several tokens
  -> Mistral Small 4 batch verify
```

它与主模型 119B 权重分开，可按延迟目标选择加载。NVFP4 版本进一步面向 Blackwell 降低权重带宽。

## 十二、Llama 4：iRoPE、Early Fusion 与首代 Llama MoE

### 12.1 官方开放版本

| 模型 | 总参数 | 激活参数 | Experts | Context |
|---|---:|---:|---:|---:|
| Llama 4 Scout | 109B | 17B | 16 | 10M |
| Llama 4 Maverick | 400B | 17B | 128 | 1M |

两者是 Llama 家族第一次从 Dense Transformer 转向 MoE。

### 12.2 Shared Expert + Routed Expert

Llama 4 每个 MoE Layer 包含：

```text
one shared expert:
  every token passes

one routed expert:
  selected from N experts
```

简化为：

```text
y = SharedExpert(x)
  + gate(x) * RoutedExpert_top1(x)
```

Top-1 让 Expert 计算与通信简单，但 Router 选错时没有第二个 Routed Expert 补偿。Shared Expert 用于承载通用能力。

### 12.3 iRoPE

Llama 4 交替使用：

```text
3 RoPE layers
1 NoPE layer
```

RoPE Layer 使用局部 Chunk Attention；NoPE Layer 不施加旋转位置编码并执行 Full Causal Attention：

```text
Layer 0: RoPE + local/chunked attention
Layer 1: RoPE + local/chunked attention
Layer 2: RoPE + local/chunked attention
Layer 3: NoPE + global attention
repeat
```

直觉是：

- Local RoPE Layer 保留精确相对位置。
- NoPE Global Layer 进行不受 RoPE 高频外推影响的内容检索。
- 多层组合仍能从因果顺序和周围 Local Layer 推断位置信息。

NoPE 并不等于模型完全不知道顺序。Causal Mask、Layer 堆叠和 Local RoPE 路径仍提供顺序信号。

Scout 的 10M Context 还依赖长上下文训练与推理时 Attention Temperature Scaling，不能只归因于删除 RoPE。

### 12.4 Early Fusion 多模态

Llama 3.2 Vision 的典型方式是在已经训练好的文本 LLM 上加 Cross-Attention Adapter。Llama 4 从预训练阶段融合：

```text
Image
  -> vision encoder
  -> image tokens

Text
  -> text tokens

[image tokens, text tokens]
  -> shared MoE Transformer
```

Early Fusion 让文本与视觉在所有后续层共同建模。它提升跨模态推理，但也让视觉请求进入昂贵的 MoE 主干。

### 12.5 17B Active 不代表 Scout 与 Maverick 一样快

两者激活参数都约 17B，但：

- Maverick 总权重 400B，权重内存和 Expert 分布更大。
- 128 Experts 的路由和并行布局比 16 Experts 更复杂。
- 相同 Batch 下触发的 Expert 数量可能更多。
- 视觉 Token 数、上下文长度和量化格式会改变瓶颈。

Active Parameter 只近似 FLOPs，不描述 HBM 容量、通信和 Kernel 形状。

## 十三、gpt-oss：128-token 局部注意力与 MXFP4

### 13.1 两种规模

```text
gpt-oss-120b:
  116.8B total
  5.1B active
  36 layers
  128 experts, Top-4

gpt-oss-20b:
  20.9B total
  3.6B active
  24 layers
  32 experts, Top-4
```

两者 Hidden Size 都是 2880，采用：

```text
64 Q heads
8 KV heads
head_dim = 64
```

### 13.2 Full 与 128-token Sliding Window 逐层交替

120B 的 36 层：

```text
Sliding(128)
Full
Sliding(128)
Full
...
```

BF16 GQA 每层每 token：

```text
2 * 8 * 64 * 2 = 2048 bytes
```

18 个 Full Layer 在 128K Context 的 KV：

```text
2048 * 18 * 131072
≈ 4.5 GiB/request
```

18 个 Sliding Layer 只需固定：

```text
2048 * 18 * 128
≈ 4.5 MiB/request
```

因此长上下文 KV 几乎全部来自一半 Full Layer。

### 13.3 Attention Sink

长上下文模型常出现早期 token 获得异常大 Attention 权重的现象。gpt-oss 为每个 Head 学习一个 Sink Bias，相当于给 Softmax 增加一个不承载实际 Value 的吸收槽：

```text
softmax([qk_1, qk_2, ..., qk_L, sink_bias])
```

它让 Attention 可以把“暂时不想分给任何历史 token”的概率放到 Sink，减少对首 token 的非语义依赖。

### 13.4 MXFP4

gpt-oss 只把占参数主体的 MoE Projection 量化为 MXFP4：

```text
MoE weights:
  MXFP4, about 4.25 bits/parameter with scales

Attention / Router / Embedding / LM Head:
  higher precision
```

116.8B 参数若全部按 4.25 bit 粗算：

```text
116.8e9 * 4.25 / 8
≈ 62.1 GB
```

再加未量化模块和 Scale 后仍能放进单张 80GB H100。20B 版本约 13GB，可进入 16GB 设备。

关键不是“4 bit”，而是：

- Microscaling Block 共享 Scale。
- 模型后训练时考虑了该格式。
- Router 与 Attention 没有一起粗暴量化。
- 需要原生 MXFP4 Kernel，不能把它当普通 INT4。

### 13.5 Clamped SwiGLU

配置中的 `swiglu_limit=7.0` 对激活做 Clamp，避免低精度下极端值扩大：

```python
gate = silu(gate_proj(x)).clamp(max=7.0)
up = up_proj(x).clamp(min=-7.0, max=7.0)
y = down_proj(gate * up)
```

低精度模型的稳定性往往来自这些不显眼的数值边界，而不只是 Quantization Format。

## 十四、Falcon-H1R：并行 Attention-Mamba 混合块

### 14.1 与其他 Hybrid Model 的差异

Nemotron 的 Mamba 与 Attention 主要按层交替。Falcon-H1 在同一 Block 内并行：

```text
                         +-> Attention heads --+
x -> Norm -> projection |                      |-> concat -> output proj
                         +-> Mamba-2 heads -----+
                                      |
                                      +-> MLP / residual
```

可写成：

```text
a = Attention(P_A x)
s = Mamba2(P_S x)

y = W_O concat(a, s)
x' = x + y
x'' = x' + MLP(Norm(x'))
```

Attention 与 SSM 的通道数可以独立调整，不需要把整层指定为二选一。

### 14.2 为什么并行而不是串行

串行：

```text
x -> Mamba -> Attention -> y
```

优点是后一模块能直接修正前一模块；缺点是关键路径更长。

并行：

```text
x -> [Mamba || Attention] -> y
```

优点：

- 两个 Mixer 可并行执行。
- 通道预算可直接分配。
- 一层同时获得固定状态与显式 KV 检索。

缺点：

- 两条分支都要读输入。
- Kernel Fusion 更复杂。
- Attention KV Cache 仍然存在。

### 14.3 Falcon-H1R-7B 的具体配置

```text
layers               = 44
hidden size          = 3072
attention Q heads    = 12
attention KV heads   = 2
attention head dim   = 128
mamba heads          = 24
mamba state size     = 256
mamba conv width     = 4
context              = 256K
RoPE theta           = 1e11
```

TII 的实验表明更大的 SSM State Size 比盲目增加 SSM Group 数更有效。`RoPE theta=1e11` 也明显高于常见 10K/1M，是为长上下文位置频率做的激进调整。

H1R 的推理能力主要来自在 H1-7B 骨干上的 SFT + GRPO，以及最长 48K 的推理轨迹，不应误写成新 Attention 结构。

## 十五、RWKV-7：无 Attention、无 KV Cache 的矩阵状态

### 15.1 目标

RWKV-7 同时追求：

```text
训练:
  可用 Chunk/Scan 并行

推理:
  RNN 式逐 token 固定成本

状态:
  不保存历史 token 的 KV Cache
```

其技术报告验证了 0.19B 到 2.9B，社区还公开了更大检查点。这里讨论架构 RWKV-7 Goose，而不是把未完成的 RWKV-8 预览当成稳定版本。

### 15.2 Generalized Delta Rule

RWKV-7 每个 Head 维护矩阵状态：

```text
S_t in R^[d_v, d_k]
```

广义更新可写为：

```text
S_t = S_{t-1} D_t
    + (S_{t-1} a_t) b_t^T
    + v_t k_t^T

o_t = S_t r_t
```

三项分别表示：

1. `S_{t-1}D_t`：逐通道时间衰减。
2. `(S_{t-1}a_t)b_t^T`：基于旧状态的低秩擦除/改写。
3. `v_t k_t^T`：写入新的 Key-Value 关联。

官方 Numpy 参考实现的核心等价形式为：

```python
# S: [heads, value_dim, key_dim]
S = (
    S * decay.transpose(-1, -2)
    - (S @ kk) * (kk * write_gate).transpose(-1, -2)
    + v * k.transpose(-1, -2)
)
y = S @ receptance
```

这不是保存所有 `(k_t, v_t)`，而是让上下文在推理期间持续“训练”一个小型状态映射。

### 15.3 为什么是 Constant Space

Attention：

```text
state size ~ O(L * H_kv * d)
```

RWKV：

```text
state size ~ O(num_layers * num_heads * d_k * d_v)
```

后者与上下文长度 `L` 无关。所以 4K 与 1M token 的 Decode State 大小相同，区别只在已经执行了多少次递推。

### 15.4 固定状态也有代价

- 所有历史必须压缩进有限状态，精确逐 token 回看更难。
- 不同请求无法像 Prefix KV Cache 那样简单共享任意前缀 Page。
- Beam Search 每个 Beam 都需要复制或分叉状态。
- 大 Batch 时矩阵状态本身也会占用显存。
- Recurrent Decode 需要专用 Kernel，不能直接复用 FlashAttention。

RWKV-7 代表的是“完全取消 KV Cache”的极端路线，适合移动端、超长流式输入和大 Batch Decode，但不是所有精确检索任务的无损替代。

## 十六、把关键代价算清楚

### 16.1 总参数、激活参数与激活比例

| 模型 | Total | Active | 激活比例 |
|---|---:|---:|---:|
| DeepSeek-V4-Pro | 1.6T | 49B | 3.1% |
| Kimi-K2.7 | 1T | 32B | 3.2% |
| GLM-5.2 | 744B | 40B | 5.4% |
| Nemotron 3 Ultra | 550B | 55B | 10.0% |
| Llama 4 Maverick | 400B | 17B | 4.3% |
| MiniMax-M2.7 | 230B | 10B | 4.3% |
| Mistral Small 4 | 119B | 6.5B | 5.5% |
| gpt-oss-120b | 116.8B | 5.1B | 4.4% |
| Qwen3.6-35B-A3B | 35B | 3B | 8.6% |
| Gemma 4 26B-A4B | 25.2B | 3.8B | 15.1% |

激活比例越低不一定越快，因为：

```text
Latency ≈
  active GEMM
  + expert weight loading
  + routing
  + all-to-all
  + attention/state mixer
  + synchronization
```

### 16.2 Attention 路线的渐进复杂度

设上下文长度为 `L`：

| 结构 | Prefill 主要复杂度 | Decode 每 token | 随 L 增长的状态 |
|---|---|---|---|
| Full Attention | `O(L^2)` | `O(L)` | `O(L)` KV |
| Sliding Window | `O(LW)` | `O(W)` | `O(W)` KV |
| DSA | Index `O(L^2)` 训练，主干 `O(Lk)` | Index `O(L)` + Attn `O(k)` | 历史表示 |
| CSA | 压缩/Index + `O(Lk)` | Index `O(L/m)` + `O(k+W)` | 压缩历史 |
| HCA | `O(L^2/m')` | `O(L/m')` | 压缩历史 |
| Gated DeltaNet | 近似 `O(L)` | `O(1)` | 固定矩阵状态 |
| Mamba-2 | 近似 `O(L)` | `O(1)` | 固定 SSM State |
| RWKV-7 | 近似 `O(L)` | `O(1)` | 固定矩阵状态 |

“DSA 训练是 `O(Lk)`”只在真正稀疏的 Indexer/Attention Kernel 下成立。若实现先构造 Dense Score 再 Mask，数学稀疏不等于计算稀疏。

### 16.3 128K Context 的 KV Cache 示例

假设 60 层、BF16、Batch=1：

```text
MHA:
  H_kv=64, d=128
  2 * 64 * 128 * 2 * 60 * 131072
  ≈ 240 GiB

GQA:
  H_kv=8, d=128
  ≈ 30 GiB

MQA:
  H_kv=1, d=128
  ≈ 3.75 GiB
```

MLA、KV Sharing 和 Local/Global Hybrid 继续在 MQA/GQA 基础上压缩。

### 16.4 Linear State 不随 L 增长，但并不为零

以 Qwen3.6-35B-A3B 的 GDN 粗略估算。若每个 32 个 Value Head 维护 `128x128` FP32 状态：

```text
32 * 128 * 128 * 4 bytes
= 2 MiB / GDN layer / request
```

30 个 GDN Layer：

```text
2 MiB * 30 = 60 MiB/request
```

它对 256K Context 很划算，但 Batch=256 时主状态本身可达约 15GiB。线性模型把维度从 `Context Length` 转移到了 `Batch * State Size`。

### 16.5 MoE 通信示例

设每张 GPU 向 Expert Parallel Group 发送 4096 个 Token，Hidden=8192，BF16：

```text
standard routed payload:
4096 * 8192 * 2
= 64 MiB per dispatch copy
```

LatentMoE 压到 2048：

```text
4096 * 2048 * 2
= 16 MiB
```

这就是 Nemotron 能把 Top-K 提高到 22 的前提之一。实际通信还乘 Expert 副本、路由份数，并包含 Combine 返回流量。

## 十七、这些架构如何改变推理系统

### 17.1 不能只提供一个 Attention Backend

现代引擎需要按 Layer Type 分派：

```python
def run_layer(layer, state):
    if layer.kind == "full_attention":
        return paged_attention(layer, state.kv_cache)

    if layer.kind == "sliding_attention":
        return sliding_attention(layer, state.ring_kv)

    if layer.kind == "dsa":
        indices = run_indexer_topk(layer, state.index_cache)
        return sparse_attention(layer, indices)

    if layer.kind == "gated_delta":
        return recurrent_gdn(layer, state.gdn)

    if layer.kind == "mamba":
        return selective_state_update(layer, state.ssm)
```

### 17.2 Cache Manager 变成多状态管理器

传统 PagedAttention 只管理 KV Page。现在还需要：

```text
Paged KV:
  Full / MLA / DSA

Ring KV:
  Sliding Window

Recurrent State:
  GDN / Mamba / RWKV

Shared KV Slot:
  Gemma 4 producer-consumer mapping

Compressed Pool:
  DeepSeek-V4 CSA/HCA

Shared Top-K:
  GLM-5.2 IndexShare
```

Prefix Cache 也不再统一：

- Full Attention 可共享 KV Page。
- GDN/Mamba 需要缓存前缀结束时的 State Snapshot。
- DSA 还需要 Indexer Key 或压缩池。
- Shared KV 必须保持 Layer Donor 映射一致。

### 17.3 Continuous Batching 更难

不同请求可能处于：

- 文本 Prefill。
- 视觉 Encoder。
- GDN Chunk Prefill。
- DSA Index。
- MoE Expert Dispatch。
- MTP Verify。

若全部混在一个 Batch，Kernel 类型不断变化；若按类型拆 Batch，又降低 Continuous Batching 的合批率。

高性能系统通常需要：

```text
请求级调度
  -> phase-aware batching
  -> model-layer-aware execution
  -> separate prefill/decode workers when necessary
```

### 17.4 MoE 需要 Wide Expert Parallel

大模型通常：

```text
Attention:
  Tensor Parallel / Context Parallel

MoE:
  Expert Parallel

Layer transition:
  TP layout <-> EP layout
```

DeepSeek/Kimi 的 256/384 Expert 与 Nemotron 的 Top-22 会放大 Token Dispatch。需要：

- Fused Permute/Unpermute。
- DeepEP/NCCL All-to-All。
- Grouped GEMM。
- Local Expert Load Balance。
- Communication-Compute Overlap。

### 17.5 低精度格式必须进入 Kernel 设计

| 格式 | 代表模型 | Kernel 重点 |
|---|---|---|
| FP8 Block Scale | DeepSeek/Kimi/MiniMax | Scale Layout、Accumulation |
| FP4 Routed Expert | DeepSeek-V4 | FP4 Grouped GEMM |
| NVFP4 | Nemotron/Mistral | Blackwell Block Scale |
| MXFP4 | gpt-oss | Microscaling Expert GEMM |
| INT4 Group-wise | Kimi-K2.7 发布权重 | Dequant + GEMM Fusion |

名称都包含“4 bit”，布局、Scale、零点、硬件指令却不同，不能共用一个泛化 INT4 Kernel 后期待同样性能。

## 十八、训练侧真正发生了什么变化

### 18.1 Optimizer 从默认 AdamW 分化

```text
Kimi:
  MuonClip

DeepSeek-V4:
  Muon for matrix weights
  AdamW for sensitive/non-matrix weights

Falcon:
  muP and non-learnable multipliers

Nemotron:
  NVFP4-aware pretraining
```

训练系统需要支持：

- 参数类型分组。
- Newton-Schulz 迭代。
- Optimizer State Sharding。
- QK Logit 监控。
- 不同精度的 Master Weight。

### 18.2 MoE 稳定性仍是核心

主要问题：

```text
Router collapse:
  少数 Expert 吞掉大部分 Token

Expert starvation:
  部分 Expert 几乎没有梯度

Capacity overflow:
  单 Expert token 数超过 buffer

Precision-sensitive routing:
  FP8/FP4 下 Top-K 排名改变
```

解决路线：

- Auxiliary Load Balance Loss。
- Auxiliary-loss-free Correction Bias。
- Group-limited Top-K。
- Router 使用 FP32。
- Shared Expert。
- Hash MoE Warmup。

### 18.3 长上下文训练不是只改 RoPE

从 128K 扩到 1M 需要：

- Context Parallel。
- Ring/Striped Attention 或 Sparse Kernel。
- Sequence Packing。
- Activation Checkpointing。
- 长文档和 Agent Trajectory 数据。
- Position Extrapolation 训练。
- Loss 在长序列中的稳定归一化。

只修改 `max_position_embeddings` 会得到“能加载配置”，不会自动得到可靠 1M 能力。

### 18.4 多模态正在从 Adapter 走向共同预训练

当前主线：

```text
Llama 4 / Qwen3.6 / Kimi K2.x:
  encoder + early-fused multimodal tokens

Gemma 4 12B:
  raw patches/chunks -> shared backbone

Nemotron Nano Omni:
  text/image/video/audio unified sub-agent
```

训练难点从“接上视觉 Encoder”变成：

- 不同模态 Token 比例。
- 时间与空间位置编码。
- 长视频 Token 压缩。
- 文本能力不被视觉数据稀释。
- 多模态 Batch 的计算负载不均。

## 十九、架构选择表

| 需求 | 更匹配的结构 | 原因 |
|---|---|---|
| 1M 精确长上下文 | DeepSeek-V4 / GLM-5.2 | 内容稀疏检索，不只局部窗口 |
| 高吞吐长程 Agent | Nemotron 3 Ultra | Mamba 主干 + LatentMoE + MTP |
| 低成本多模态 Agent | Qwen3.6-35B-A3B | 3B 激活、GDN、视觉原生 |
| 本地 16GB 推理 | gpt-oss-20b / Gemma 4 小模型 | 原生低精度或端侧结构 |
| 单机 Dense 易部署 | Qwen3.6-27B / Gemma 4 31B | 无 Expert Parallel |
| 代码长程 Agent | Kimi-K2.7 Code / GLM-5.2 | 针对长程代码轨迹后训练 |
| 统一快答与推理 | Mistral Small 4 | 同一 Checkpoint 切换 Effort |
| 流式无限输入 | RWKV-7 | 状态不随上下文增长 |
| 研究 Attention/SSM 通道配比 | Falcon-H1 | 同层并行、通道可调 |
| 视觉/音频端侧 | Gemma 4 E2B/E4B/12B | PLE 或 Encoder-free |

选择时至少同时看：

```text
Model quality
Weight memory
Active FLOPs
KV/state memory
Communication
Kernel maturity
License
Target hardware
```

## 二十、常见误区

### 20.1 “A3B 就只占 3B 模型显存”

错误。A3B 描述激活计算，不描述总权重存储。

### 20.2 “Linear Attention 没有 Cache”

不准确。它没有随 `L` 增长的 KV Cache，但仍有固定大小 Recurrent State、Conv State 和请求级状态。

### 20.3 “Sparse Attention 的复杂度天然就是 O(Lk)”

只有真正使用稀疏 Kernel、避免构造 Dense Score Matrix 时才成立。

### 20.4 “1M Context 表示生产环境可随意跑 1M”

模型支持、Kernel 支持、显存容量、TTFT 和并发能力是五件事。1M Prefill 仍然可能需要分钟级延迟或多机 Context Parallel。

### 20.5 “Thinking Mode 是一种新网络层”

通常不是。它主要来自后训练、控制 Token、Chat Template 和输出 Budget。

### 20.6 “Agent Swarm 是 MoE”

不是。

```text
MoE:
  单次前向内部按 token 路由 FFN Expert

Agent Swarm:
  应用层创建多个模型调用并编排工具
```

### 20.7 “MTP 一定能让 Decode 快 n 倍”

MTP 只能草拟。最终仍需主模型验证，接受长度低、Batch 大或 Draft 太重时可能不加速。

### 20.8 “FP4、NVFP4、MXFP4、INT4 是同一种格式”

它们在编码、Scale 粒度、硬件指令和 Kernel Layout 上不同。

### 20.9 “最新模型一定有最新架构”

Kimi-K2.7 Code、MiniMax-M2.7、Falcon-H1R 的主要进步都包含大量后训练与系统工程。版本号更新不等于主干数学结构重写。

## 二十一、开放权重不等于完全开源

本文沿用行业常见的“开源模型”说法，但严格区分：

```text
Open weights:
  可以下载参数

Open code:
  有训练/推理实现

Open data:
  训练数据可获得

Open recipe:
  超参数、并行策略、后训练流程可复现

Permissive license:
  商用、修改、再分发限制较少
```

当前许可证概览：

| 模型 | 许可证类型 |
|---|---|
| Qwen3.6 | Apache 2.0 |
| DeepSeek-V4 | MIT |
| GLM-5.2 | MIT |
| Kimi-K2.7 | Modified MIT |
| MiniMax-M2.7 | Modified MIT |
| Nemotron 3 Ultra | OpenMDW 1.1 |
| Gemma 4 | Apache 2.0 |
| Mistral Small 4 | Apache 2.0 |
| Llama 4 | Llama 4 Community License |
| gpt-oss | Apache 2.0 |
| Falcon-H1/H1R | Apache 2.0-based Falcon License |
| RWKV-7 | Apache 2.0 |

生产使用前应阅读对应 Checkpoint 的原始许可证，而不是根据“开放权重”四个字推断商用条款。

## 二十二、参考资料

### 官方模型与技术报告

1. [Qwen3.6-35B-A3B Model Card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
2. [Qwen3.6-27B Release](https://qwen.ai/blog?id=qwen3.6-27b)
3. [DeepSeek-V4 Technical Report](https://arxiv.org/abs/2606.19348)
4. [DeepSeek-V4-Pro Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
5. [GLM-5.2: Built for Long-Horizon Tasks](https://z.ai/blog/glm-5.2)
6. [GLM-5.2 Model Card](https://huggingface.co/zai-org/GLM-5.2)
7. [Kimi K2.7 Code](https://www.kimi.com/resources/kimi-k2-7-code)
8. [Kimi K2 Technical Report](https://arxiv.org/abs/2507.20534)
9. [Kimi-K2.5 Official Repository](https://github.com/MoonshotAI/Kimi-K2.5)
10. [MiniMax-M2 Series Technical Report](https://arxiv.org/abs/2605.26494)
11. [MiniMax-M2.7 Model Card](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)
12. [Nemotron 3 Ultra Technical Report](https://arxiv.org/abs/2606.15007)
13. [Nemotron 3 Ultra Model Card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)
14. [LatentMoE](https://arxiv.org/abs/2601.18089)
15. [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770)
16. [Gemma 4 26B-A4B Model Card](https://huggingface.co/google/gemma-4-26B-A4B-it)
17. [Mistral Small 4 Model Card](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)
18. [Llama 4 Official Model Card](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md)
19. [gpt-oss Model Card](https://deploymentsafety.openai.com/gpt-oss/model-architecture-data-training-and-evaluations)
20. [gpt-oss-120b Config](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json)
21. [Falcon-H1 Technical Report](https://arxiv.org/abs/2507.22448)
22. [Falcon-H1R Release](https://falcon-lm.github.io/blog/falcon-h1r-7b/)
23. [RWKV-7 Goose Technical Report](https://arxiv.org/abs/2503.14456)
24. [RWKV-7 Numpy Reference](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py)

### 机制与实现

25. [Gated DeltaNet](https://arxiv.org/abs/2412.06464)
26. [DeepSeek-V4 in Hugging Face Transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deepseek_v4.md)
27. [GLM-5.2 SGLang Deployment Guide](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2)
28. [GPT OSS in Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/models/llm/gpt-oss.html)
29. [Falcon-H1 in Megatron Core](https://developer.nvidia.com/blog/implementing-falcon-h1-hybrid-architecture-in-nvidia-megatron-core/)

## 二十三、总结

当前开放权重大模型的架构演进可以压缩成五条主线：

```text
1. Attention 不再统一:
   Full / Sliding / Sparse / Compressed / Linear / SSM 并存

2. KV Cache 成为第一设计约束:
   GQA -> MLA -> K=V -> KV Sharing -> Recurrent State

3. MoE 从“少算参数”走向硬件协同:
   Fine-grained Experts / LatentMoE / FP4 / Wide-EP

4. 训练目标直接服务推理:
   MTP -> Speculative Decoding
   Native low precision -> hardware-native serving

5. 多模态从外挂 Adapter 走向共同主干:
   Early Fusion / Native multimodal / Encoder-free
```

最值得关注的不是哪个模型参数最大，而是不同设计如何移动瓶颈：

```text
Full Attention:
  瓶颈在 KV 读取

Sparse Attention:
  瓶颈转向 Indexer、Top-K 和 Gather

Linear/SSM:
  瓶颈转向固定状态更新与 Batch State

MoE:
  瓶颈转向权重带宽和 All-to-All

Low Precision:
  瓶颈转向 Scale Layout 与专用 Kernel
```

理解这些瓶颈，才能从模型卡上的名词继续走到训练并行、推理调度、Cache Manager 与 GPU Kernel。
