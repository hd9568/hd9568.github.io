---
title: 'Qwen 2.5 到 Qwen3.8 推理优化全解：长上下文、混合注意力、MoE 与 MTP'
description: '以 Qwen 近两年的架构演进为主线，系统讲解 Qwen2.5-1M 的 DCA、稀疏注意力与 Chunked Prefill，Qwen3 的思考预算和 MoE，以及 Qwen3-Next、Qwen3.5、Qwen3.8 的 Gated DeltaNet、超稀疏 MoE、MTP、混合缓存、量化与分布式推理。'
category: '推理优化'
pubDate: '2026-08-14T12:00:00+08:00'
updatedDate: '2026-08-14T12:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [范围、版本与核心结论](#一范围版本与核心结论)
2. [两年演进主线：Qwen 改变了哪些推理成本](#二两年演进主线qwen-改变了哪些推理成本)
3. [先建立 Prefill、Decode 与 Serving 成本模型](#三先建立-prefilldecode-与-serving-成本模型)
4. [Qwen2.5：GQA、RoPE 与 Dense Transformer 基线](#四qwen25gqarope-与-dense-transformer-基线)
5. [Qwen2.5-1M：DCA 如何把 256K 外推到 1M](#五qwen25-1mdca-如何把-256k-外推到-1m)
6. [MInference：用 Vertical-Slash 稀疏注意力降低 Prefill](#六minference用-vertical-slash-稀疏注意力降低-prefill)
7. [Chunked Prefill 与动态流水：让 1M Prompt 真正可部署](#七chunked-prefill-与动态流水让-1m-prompt-真正可部署)
8. [Qwen3：思考模式把推理成本变成可控变量](#八qwen3思考模式把推理成本变成可控变量)
9. [Qwen3 MoE：少算参数不等于少存权重](#九qwen3-moe少算参数不等于少存权重)
10. [Qwen3-Next：推理友好架构的转折点](#十qwen3-next推理友好架构的转折点)
11. [Gated DeltaNet：固定状态如何替代增长的 KV Cache](#十一gated-deltanet固定状态如何替代增长的-kv-cache)
12. [Gated DeltaNet 的 Prefill 与 Decode Kernel](#十二gated-deltanet-的-prefill-与-decode-kernel)
13. [为什么是 3:1 混合注意力，而不是纯线性注意力](#十三为什么是-31-混合注意力而不是纯线性注意力)
14. [Gated Attention、Partial RoPE 与稳定性设计](#十四gated-attentionpartial-rope-与稳定性设计)
15. [超稀疏 MoE：从 128 个专家到 512 个专家](#十五超稀疏-moe从-128-个专家到-512-个专家)
16. [Qwen3.5：原生多模态与推理效率合流](#十六qwen35原生多模态与推理效率合流)
17. [Qwen3.8：2.4T/A95B 的最新开放权重架构](#十七qwen3824ta95b-的最新开放权重架构)
18. [Qwen3.8 显存账本：权重、KV 与 GDN 状态](#十八qwen38-显存账本权重kv-与-gdn-状态)
19. [MTP：用模型原生 Draft 打破单 Token 串行 Decode](#十九mtp用模型原生-draft-打破单-token-串行-decode)
20. [ReplaySSM：线性注意力 Decode 的新优化方向](#二十replayssm线性注意力-decode-的新优化方向)
21. [混合缓存：Paged KV、GDN State 与 Prefix Cache](#二十一混合缓存paged-kvgdn-state-与-prefix-cache)
22. [MoE Kernel：Router、Dispatch、Grouped GEMM 与 Combine](#二十二moe-kernelrouterdispatchgrouped-gemm-与-combine)
23. [并行策略：TP、EP、DP Attention 与 PP 如何组合](#二十三并行策略tpepdp-attention-与-pp-如何组合)
24. [量化：BF16、FP8、NVFP4/MXFP4 与 FP8 KV](#二十四量化bf16fp8nvfp4mxfp4-与-fp8-kv)
25. [Serving：Continuous Batching、Chunked Prefill 与 CUDA Graph](#二十五servingcontinuous-batchingchunked-prefill-与-cuda-graph)
26. [P/D Disaggregation：混合模型需要传输哪些状态](#二十六pd-disaggregation混合模型需要传输哪些状态)
27. [Prefix Cache 与上下文缓存：Agent 场景的关键优化](#二十七prefix-cache-与上下文缓存agent-场景的关键优化)
28. [多模态推理：Vision Encoder、Media Cache 与文本专用路径](#二十八多模态推理vision-encodermedia-cache-与文本专用路径)
29. [vLLM 与 SGLang 部署配置如何读](#二十九vllm-与-sglang-部署配置如何读)
30. [如何在 Qwen 模型之间做工程选型](#三十如何在-qwen-模型之间做工程选型)
31. [性能评估、Profile 与调优清单](#三十一性能评估profile-与调优清单)
32. [常见误解](#三十二常见误解)
33. [参考资料](#三十三参考资料)

## 一、范围、版本与核心结论

本文讨论 2024 年下半年至 2026 年 8 月 Qwen 主干模型中与推理系统最相关的变化：

```text
Qwen2.5 / Qwen2.5-1M
  -> Qwen3
  -> Qwen3-Next
  -> Qwen3.5 / Qwen3.6
  -> Qwen3.8
```

截至本文更新时间，最新旗舰是：

```text
托管服务:
  Qwen3.8-Max
  2.4T total / 95B active
  文本、图像、视频输入
  默认 1M Context
  支持非思考模式和内置工具

开放权重:
  Qwen3.8-2.4T-A95B
  2.4T total / 95B active
  Text-only
  Thinking-only
  262,144 Native Context
  可通过 YaRN 扩展到约 1,010,000
```

这两个名字经常被混用，但能力边界不同。开放权重 checkpoint 是托管 Max 服务的基础之一，不等于完整的托管产品。

Qwen 近两年的推理演进可以压缩为六条主线：

```text
1. Qwen2.5-1M:
   先通过 DCA、MInference 和 Chunked Prefill
   让全注意力 Transformer 可以处理 1M Context。

2. Qwen3:
   用 GQA 和 Sparse MoE 降低 KV 与 FFN 计算，
   用 Thinking Budget 控制输出侧 Test-Time Compute。

3. Qwen3-Next:
   用 3:1 Gated DeltaNet / Full Attention
   从模型结构上减少长上下文 KV 和 Attention 成本。

4. Qwen3-Next / Qwen3.5:
   将专家数扩展到 512，只激活 10 Routed + 1 Shared，
   用更低 Active Parameter 换取更大模型容量。

5. Qwen3-Next / Qwen3.5 / Qwen3.8:
   训练原生 MTP Head，推理时用于 Speculative Decoding，
   减少昂贵主模型的串行 Decode Step。

6. Qwen3.8:
   将混合注意力和超稀疏 MoE 扩展到 2.4T，
   使推理问题从单机 Kernel 优化扩大到机架级通信与调度。
```

最重要的结论不是“新模型 FLOPs 更少”，而是：

> Qwen 的模型结构变化持续把瓶颈从 Attention 计算转移到权重内存、专家通信、固定状态带宽、调度和输出 Token 数。推理框架必须跟着模型架构一起改变。

## 二、两年演进主线：Qwen 改变了哪些推理成本

### 2.1 时间线

| 时间 | 代表模型 | 架构或推理变化 | 主要解决的问题 |
| --- | --- | --- | --- |
| 2024-09 | Qwen2.5 | Dense Transformer、GQA、128K | 通用能力与标准部署 |
| 2024-11 / 2025-01 | Qwen2.5-Turbo / Qwen2.5-1M | DCA、MInference、Chunked Prefill | 1M Context 的能力与 TTFT |
| 2025-04 | Qwen3 | Dense + MoE、Thinking/Non-Thinking | 能力、成本和推理预算统一 |
| 2025-09 | Qwen3-Next | Gated DeltaNet + Gated Attention、512 Experts、MTP | 长上下文与低 Active FLOPs |
| 2026-02 | Qwen3.5 | 原生多模态、397B/A17B、250K Vocabulary | 多模态 Agent 与高效推理 |
| 2026-04 | Qwen3.6 | 35B/A3B MoE 与 27B Dense | 单机部署与 Coding/Agent |
| 2026-08 | Qwen3.8 | 2.4T/A95B、92 层、开放 Max 级权重 | Frontier Agent 与机架级推理 |

Qwen3.6、Qwen3.7 中有很多能力和产品迭代，但从公开架构看，最关键的推理结构转折仍是：

```text
Qwen3 Full Attention
  -> Qwen3-Next Hybrid Attention
  -> Qwen3.5 Native Multimodal Hybrid
  -> Qwen3.8 Rack-Scale Hybrid MoE
```

### 2.2 每代模型主要在优化哪一项

可以把单请求推理成本粗略拆成：

```text
Total Latency =
    Queueing
  + Tokenization / Media Processing
  + Prefill
  + Decode
  + Communication
  + Sampling / Tool Protocol
```

不同技术主要作用于不同部分：

| 技术 | Prefill | Decode | 显存容量 | 多卡通信 | 输出 Token |
| --- | ---: | ---: | ---: | ---: | ---: |
| DCA / YaRN | 能力外推 | 能力外推 | 不直接减少 | 无 | 无 |
| MInference | 大幅减少长 Prompt Attention | 作用有限 | 临时张量降低 | 无 | 无 |
| Chunked Prefill | 改变调度与峰值 | 降低 Decode 干扰 | 降低激活峰值 | 可流水 | 无 |
| GQA | Attention 略降 | KV 读取显著下降 | KV 减少 | TP 通信可能变化 | 无 |
| Sparse MoE | FFN Active FLOPs 下降 | 权重读取下降 | 总权重仍大 | EP All-to-All 增加 | 无 |
| Gated DeltaNet | 长序列近线性 | 不扫描完整历史 | 固定 State | State Sharding | 无 |
| MTP | 无或很少 | 减少串行 Step | Draft State 增加 | Verify 通信增加 | 无损生成 |
| Thinking Budget | 无 | Decode 总量变化 | KV/State 生命周期变化 | 持有资源更久 | 直接控制 |
| Prefix Cache | 重复前缀接近消除 | 无 | Cache 占用增加 | 跨实例路由复杂 | 无 |
| FP8/FP4 | GEMM/权重带宽降低 | 权重带宽降低 | 权重显著减少 | 通信字节可能降低 | 可能影响精度 |

### 2.3 专用分支怎样反向推动推理系统

主干之外，Qwen 近两年的专用模型也揭示了新的推理负载。

| 分支 | 代表成果 | 对推理系统的新要求 |
| --- | --- | --- |
| Reasoning | QwQ-32B | 长 CoT、长输出调度、工具反馈 |
| Coding | Qwen3-Coder-480B-A35B | 256K Repo Context、长 Tool Loop、Prefix Reuse |
| Efficient Coding | Qwen3-Coder-Next-80B-A3B | Hybrid Cache、3B Active、长时间 Agent Serving |
| Vision-Language | Qwen3-VL-235B-A22B | Dynamic Resolution、DeepStack、长视频与 GUI Agent |
| Omni | Qwen3-Omni-30B-A3B | Thinker/Talker 并行、流式多码本语音、首包延迟 |

**QwQ-32B** 将推理时扩展变成主要能力来源。即使模型只有 32B，长 CoT 也会让 Decode 占用远大于普通 Chat。它推动了后续 Qwen3 的 Thinking Budget、Reasoning Parser 和长输出调度。

**Qwen3-Coder-480B-A35B** 使用 480B Total / 35B Active MoE，原生 256K、可外推到 1M。Coding Agent 会反复发送：

```text
Repository Prefix
+ Tool Definition
+ Command Output
+ Test Log
+ Patch History
```

因此它比普通对话更依赖 Prefix Cache、Context Folding、长输出公平调度和低错误率 Tool Parser。

**Qwen3-Coder-Next** 将 Coding 能力迁移到 Qwen3-Next 的 80B/A3B 混合架构。它只支持 Non-Thinking，但 Agent 仍可通过多轮 `plan -> edit -> test -> repair` 完成长任务。这说明“内部生成长 CoT”与“外部 Agent 长时执行”是两种不同的推理时扩展：

```text
Thinking Scaling:
  单次调用输出更多推理 Token

Agent Scaling:
  多轮调用、工具执行和环境反馈
```

**Qwen3-VL** 引入 Interleaved-MRoPE、DeepStack 和文本时间戳。DeepStack 将不同 ViT 层的多级视觉特征注入对应 LLM 层，不需要把所有中间特征追加成更多上下文 Token，但引擎要处理跨层视觉残差。Dynamic Resolution 和长视频则让 Media Token 数、Frame Sampling 与 Vision Encoder 调度成为主要成本。

**Qwen3-Omni** 使用 Thinker-Talker MoE。Thinker 负责理解和文本表示，Talker 自回归生成语音 Codec Frame；每步先生成主码本，MTP 生成剩余 Residual Codebook，Code2Wav 用轻量 Causal ConvNet 流式合成波形：

```text
Audio / Video / Text
  -> Encoder
  -> Thinker
       | \
       |  -> Text Stream
       v
     Talker
       -> Main Codec Token
       -> MTP Residual Codebooks
       -> Causal Code2Wav
       -> Audio Stream
```

这里优化目标从文本的 TTFT/TPOT 扩展为：

```text
First Audio Packet Latency
Real-Time Factor
Audio Chunk Jitter
Thinker-Talker Pipeline Bubble
```

这些专用分支最终汇合到 Qwen3.5/Qwen3.8 的方向：统一多模态主干、长上下文、低 Active MoE 和可扩展 Agent 推理。

## 三、先建立 Prefill、Decode 与 Serving 成本模型

### 3.1 Prefill

Prompt 有 `L` 个 Token。标准全注意力每层需要：

```text
QK^T: O(L^2 * d)
PV:   O(L^2 * d)
MLP:  O(L * d * d_ff)
```

短 Prompt 时 MLP/GEMM 可以占大头。`L` 很大时，Attention 的二次项迅速成为 TTFT 瓶颈。

Prefill 的特点：

- 一次有很多 Token，GEMM 尺寸大。
- Tensor Core 利用率通常较高。
- 更容易 Compute-bound。
- 长 Prompt 会制造巨大的临时激活和调度长尾。

### 3.2 Decode

每轮每请求只生成一个新 Token：

```text
Decode Step:
  读取模型权重
  读取或更新上下文状态
  执行小 Batch GEMM / GEMV
  采样一个 Token
```

Decode 的主要矛盾是：

```text
每一步只产生很少计算工作，
却要读取大量权重和历史状态。
```

它通常更偏 Memory-bandwidth-bound。生成 `N` 个 Token 还必须串行执行约 `N` 次主模型前向。

### 3.3 在线 Serving

单个 Kernel 快不等于服务快。在线系统还受到：

```text
输入长度分布
输出长度分布
请求到达率
Batch 动态变化
KV / GDN State 容量
MoE Expert 热点
跨卡与跨节点网络
SLO 下的排队和抢占
```

因此评估时至少区分：

```text
TTFT:
  请求到达到第一个输出 Token

TPOT / ITL:
  相邻输出 Token 间隔

Request Latency:
  完整请求耗时

Throughput:
  全服务输出 Token/s

Goodput:
  满足 TTFT/TPOT SLO 的有效请求或 Token/s
```

## 四、Qwen2.5：GQA、RoPE 与 Dense Transformer 基线

Qwen2.5 仍是标准 Decoder-only Transformer：

```text
Token
  -> RMSNorm
  -> GQA + RoPE
  -> Residual
  -> RMSNorm
  -> SwiGLU FFN
  -> Residual
```

这一代对推理最重要的是 GQA。

### 4.1 MHA 的 KV Cache

设：

```text
N_layers      = 层数
N_kv_heads    = KV Head 数
D_head        = 每个 Head 维度
L             = 已缓存 Token 数
B             = 并发序列数
bytes         = 每元素字节数
```

KV Cache 大小近似为：

```text
KV_bytes =
    B
  * L
  * N_layers
  * 2                 # K 和 V
  * N_kv_heads
  * D_head
  * bytes
```

MHA 中：

```text
N_kv_heads = N_q_heads
```

GQA 中多个 Q Head 共享一个 KV Head：

```text
N_kv_heads << N_q_heads
```

这同时降低：

- KV Cache 容量。
- Decode 每步从 HBM 读取的 KV 字节数。
- PagedAttention 的 Block Payload。
- KV Cache 跨节点传输成本。

### 4.2 GQA 不减少什么

GQA 不会消除：

- Q Projection。
- 输出 Projection。
- FFN 权重读取。
- Prefill 中完整的 Query-Key 交互。

它主要减少 K/V Projection 与 KV Cache 侧成本。

### 4.3 RoPE 外推不是性能优化

RoPE 让注意力分数包含相对位置信息。调整 `rope_theta`、YaRN 或 DCA 可以提高超出训练长度后的能力，但不自动减少：

```text
Attention FLOPs
KV Cache
HBM 读取
```

“支持 1M Context”和“能低成本处理 1M Context”是两个不同问题。Qwen2.5-1M 同时引入了能力侧和系统侧方案。

## 五、Qwen2.5-1M：DCA 如何把 256K 外推到 1M

Qwen2.5-1M 先通过渐进训练把模型训练到 256K，再用 Dual Chunk Attention，简称 DCA，扩展到 1M。

### 5.1 RoPE 外推的问题

标准 RoPE 下，注意力分数与 Query/Key 的相对距离有关。训练只见过有限距离时，直接推到更长序列会出现未见过的大相对位置：

```text
训练:
  relative_distance <= 256K

直接推理:
  relative_distance 接近 1M
```

DCA 的核心不是减少 Attention Pair，而是重新映射位置，使超长距离落回模型更熟悉的范围。

### 5.2 三种位置关系

DCA 将注意力关系拆成：

```text
Intra-Chunk:
  Query 和 Key 位于同一 Chunk
  保留原始相对位置

Inter-Chunk:
  Query 和 Key 跨越较远 Chunk
  将过大相对距离映射到训练范围

Successive-Chunk:
  Query 和 Key 位于相邻 Chunk
  保留局部连续性，避免 Chunk 边界断裂
```

示意：

```text
Chunk 0        Chunk 1        Chunk 2
[0 ... C)      [C ... 2C)     [2C ... 3C)

同块:
  原始相对距离

相邻块:
  保留跨边界的近距离关系

远距离块:
  压缩或重映射相对距离
```

### 5.3 DCA 的边界

DCA 仍然计算全注意力：

```text
Pair 数量仍约为 L^2 / 2
KV Cache 仍随 L 线性增长
```

它解决“模型看不懂超长位置”，不解决“计算 1M Full Attention 太贵”。后者由 MInference、Chunked Prefill 和推理引擎优化处理。

## 六、MInference：用 Vertical-Slash 稀疏注意力降低 Prefill

Qwen2.5-1M 在 Prefill 中集成了基于 MInference 的稀疏注意力。

### 6.1 Attention 并非所有位置同等重要

长上下文 Attention Map 常出现结构化高权重区域：

```text
Vertical:
  某些全局重要 Token 被很多 Query 关注

Slash / Diagonal:
  与 Query 保持特定相对偏移的 Token

Local:
  Query 附近的局部窗口
```

ASCII 示意：

```text
Key position ->

|  |       /           Query
|  |      /            position
|  |     /             |
|  |    /              v
|  |   /

Vertical: 全局重要列
Slash:    相对位置对角线
```

MInference 为不同 Attention Head 搜索适合的稀疏模式和预算，只计算高价值的 Query-Key Pair。

### 6.2 为什么需要按 Head 配置

不同 Head 可能承担不同功能：

```text
Head A:
  关注局部语法

Head B:
  关注文档开头和分隔符

Head C:
  关注远距离引用
```

统一 Sliding Window 会丢失部分全局关系。统一 Top-K 又会产生昂贵的在线索引。MInference 的方法是在离线搜索和在线稀疏执行之间折中。

### 6.3 Qwen2.5-1M 的 Sparsity Refinement

原始离线搜索如果只在短序列上做，得到的稀疏模式不一定适合 1M。Qwen2.5-1M 使用 Full Attention 的 Softmax LSE 等统计，在超长序列上细化每个 Head 的稀疏配置，降低稀疏化精度损失。

### 6.4 稀疏 Attention 的真实性能

理论 Pair 数下降不等于 Kernel 按比例加速。稀疏路径还需要：

- 计算索引。
- 合并 Vertical 与 Slash 位置。
- 对不连续 KV 做 Gather。
- 避免 Warp 间负载不均。
- 让稀疏 Tile 仍能映射到 Tensor Core。

Qwen2.5-1M 报告在 1M Prefill 上取得约 `3.2x-6.7x` 加速，而不是稀疏比例本身对应的理想倍数。

## 七、Chunked Prefill 与动态流水：让 1M Prompt 真正可部署

### 7.1 为什么一次 Prefill 1M 会 OOM

即使 FlashAttention 不物化完整 `L x L` Score，Transformer 其他层仍会产生与 Token 数线性增长的激活。

Qwen2.5-1M 官方示例中，Qwen2.5-7B 一次处理 1M 序列时，仅 MLP 激活就可达到约 71 GB。

### 7.2 Chunked Prefill

将 Prompt 切成多个连续 Chunk：

```text
1M Prompt
  -> chunk 0: [0, 32768)
  -> chunk 1: [32768, 65536)
  -> ...
```

每个 Chunk：

1. 读取之前 Chunk 的 KV。
2. 计算当前 Token 的 Attention 与 MLP。
3. 写入当前 Chunk 的 KV。
4. 释放当前 Chunk 的临时激活。

官方报告使用 32,768 Chunk 时，激活显存下降约 96.7%。

### 7.3 它不减少总工作量

Chunked Prefill 主要改变：

```text
峰值显存
单次 Kernel Shape
调度粒度
与 Decode 的混排方式
Pipeline Bubble
```

如果没有稀疏注意力，所有 Chunk 累计仍需完成完整因果 Attention。

### 7.4 Dynamic Chunked Pipeline Parallelism

超长 Prefill 还可以沿 Pipeline Stage 流水：

```text
time ->

Stage 0: chunk0 chunk1 chunk2 chunk3
Stage 1:        chunk0 chunk1 chunk2 chunk3
Stage 2:               chunk0 chunk1 chunk2
```

固定 Chunk 太大时激活高，太小时：

- Kernel 效率下降。
- Pipeline 调度次数增加。
- 通信占比上升。

因此 Chunk Size 应根据：

```text
GPU Memory
Attention Kernel
Pipeline Stage
当前 Decode 流量
TTFT / TPOT SLO
```

动态调整。

## 八、Qwen3：思考模式把推理成本变成可控变量

Qwen3 把 Thinking 和 Non-Thinking 融合进同一个模型。

### 8.1 推理成本不再只由输入决定

传统请求成本近似为：

```text
Cost ~= Prefill(input_tokens)
      + Decode(output_tokens)
```

Reasoning Model 中：

```text
output_tokens =
    hidden_reasoning_tokens
  + final_answer_tokens
```

同一个问题在不同 Thinking Budget 下，Decode Step 数可能相差一个数量级。

### 8.2 Thinking Budget 的系统含义

Thinking Budget 不只是产品参数，它直接影响：

- 请求占用 KV/GDN State 的时间。
- Continuous Batch 中的剩余长度预测。
- P99 Request Latency。
- Prefix Cache 生命周期。
- Output Token 计费。
- Speculative Decoding 的收益。
- Decode Worker 数量。

例如：

```text
请求 A:
  2K Prompt + 200 Output

请求 B:
  2K Prompt + 32K Reasoning + 2K Answer
```

两者 Prefill 相同，但 B 对 Decode Capacity 的占用约高两个数量级。

### 8.3 Qwen 各代开关并不完全相同

```text
Qwen3:
  enable_thinking
  /think 和 /no_think 软切换

Qwen3.5:
  默认 Thinking
  API 参数可禁用
  不再官方支持 Qwen3 的软切换语义

Qwen3.8 开放权重:
  Thinking-only
  reasoning_effort = low / medium / xhigh
  preserve_thinking 默认开启

Qwen3.8-Max 托管服务:
  额外支持 Non-Thinking、多模态和内置工具
```

服务端必须按具体 Model Card 和 Chat Template 处理，不能只根据“Qwen3 系列”写一个固定 Parser。

### 8.4 思考历史是否保留

旧版 Qwen3 的常见建议是多轮历史只保留 Final Answer，避免重复塞入 Thinking Content。

Qwen3.8 开放权重引入 `preserve_thinking`，允许保留历史推理上下文。这会改变：

- 输入 Token 数。
- Prefix Cache Key。
- 模型行为。
- 下一轮 Prefill 成本。

因此它不是纯展示参数，而是模型语义的一部分。

## 九、Qwen3 MoE：少算参数不等于少存权重

Qwen3 同时提供 Dense 和 MoE。

代表型号：

```text
Qwen3-30B-A3B:
  30B Total
  3B Active
  128 Experts
  8 Routed Experts / Token

Qwen3-235B-A22B:
  235B Total
  22B Active
  128 Experts
  8 Routed Experts / Token
```

### 9.1 MoE 前向

对每个 Token Hidden State `x`：

```text
router_logits = x @ W_router
expert_ids = topk(router_logits, k)

y = sum(
    routing_weight[e] * Expert_e(x)
    for e in expert_ids
)
```

### 9.2 Active Parameter 决定计算，Total Parameter 决定存储

粗略地：

```text
FLOPs per Token:
  更接近 Active Parameters

Weight Memory:
  更接近 Total Parameters
```

所以 `235B-A22B` 不等于一个容易部署的 Dense 22B：

- 仍需存储约 235B 参数。
- Router 让每个 Batch 的 Expert Shape 动态变化。
- Expert Parallel 引入 All-to-All。
- 小 Batch 下每个 Expert 的 Token 很少。

### 9.3 Qwen3 MoE 的变化

相较 Qwen2.5-MoE，Qwen3 MoE 取消共享专家，并使用全局 Batch 负载均衡目标促进专家专业化。

从推理看，没有 Shared Expert 表示：

- 每个 Token 的专家计算都来自 Routed Expert。
- Router 和负载均衡更重要。
- 不需要额外执行一个所有 Token 必经的 Shared FFN。

Qwen3-Next 又重新引入了一个 Shared Expert，原因是它同时将 Routed Expert 变得更多、更小、更稀疏。

## 十、Qwen3-Next：推理友好架构的转折点

Qwen3-Next-80B-A3B 的主要规格：

```text
48 Layers
80B Total / 3B Active

12 x [
  Gated DeltaNet -> MoE
  Gated DeltaNet -> MoE
  Gated DeltaNet -> MoE
  Gated Attention -> MoE
]

512 Experts
10 Routed + 1 Shared
262K Native Context
MTP
```

它同时从四个方向减少推理成本：

```text
长上下文:
  75% Layer 用固定 Recurrent State

FFN:
  80B 中只激活约 3B

Decode:
  MTP 草拟多个 Token

数值与训练稳定性:
  Output Gate、Zero-Centered RMSNorm、Router Init
```

官方报告在超过 32K Context 时，相对 Qwen3-32B 给出超过 10 倍吞吐提升。该数字依赖模型、框架、硬件和 Batch，不应当作所有场景的固定倍数。

## 十一、Gated DeltaNet：固定状态如何替代增长的 KV Cache

### 11.1 标准 Attention 的记忆方式

Softmax Attention 保留所有历史 K/V：

```text
K_cache = [k_1, k_2, ..., k_t]
V_cache = [v_1, v_2, ..., v_t]

o_t = softmax(q_t K_cache^T) V_cache
```

优点：

- 可以精确访问任意历史 Token。
- 检索能力强。

代价：

```text
State Memory: O(t * d)
Decode Compute: O(t * d)
```

### 11.2 线性注意力的记忆方式

线性注意力把历史压缩进矩阵状态 `S_t`：

```text
S_t = S_{t-1} + v_t k_t^T
o_t = S_t q_t
```

状态大小与序列长度无关：

```text
State Memory: O(d_v * d_k)
Decode Compute: O(d_v * d_k)
```

但简单累加会产生记忆冲突，旧值无法被精确覆盖。

### 11.3 Delta Rule

先用当前 Key 查询状态：

```text
v_old = S_{t-1} k_t
```

计算希望写入的 Value 与旧记忆之间的误差：

```text
delta_t = v_t - v_old
```

再只沿 `k_t` 方向修正：

```text
S_t = S_{t-1}
    + beta_t * delta_t * k_t^T
```

若状态已经能正确返回 `v_t`，误差接近 0，不需要重复写入。

### 11.4 Gated Delta Rule

Gated DeltaNet 再加入可学习的遗忘门。一个便于理解的简化形式是：

```text
S_decay = alpha_t * S_{t-1}
delta_t = v_t - S_decay * k_t
S_t     = S_decay + beta_t * delta_t * k_t^T
o_t     = S_t * q_t
```

其中：

```text
alpha_t:
  控制旧状态整体保留多少

beta_t:
  控制当前 Key-Value 修正写入多强
```

这结合了：

- Mamba2 风格的快速全局遗忘。
- DeltaNet 风格的定向覆盖。

实际 Qwen Kernel 还包含归一化、Gate、Causal Convolution、Grouped Value Head 等细节，上式用于解释核心状态更新，不是完整源码逐行等价式。

### 11.5 固定状态不是无损 KV Cache

`S_t` 是压缩记忆。它不可能无损保存无限多个任意 Key-Value Pair。

因此：

```text
Full Attention:
  高保真随机访问
  成本随 Context 增长

Gated DeltaNet:
  固定状态、流式更新
  历史信息被压缩
```

这就是 Qwen3-Next 不使用纯 GDN，而保留 25% Full Attention 的根本原因。

## 十二、Gated DeltaNet 的 Prefill 与 Decode Kernel

同一个 GDN Layer 在 Prefill 和 Decode 中需要两套明显不同的实现。

### 12.1 Decode：Recurrent Kernel

每次只有一个新 Token：

```text
read S_{t-1}
compute gates / q / k / v
update S_t
compute o_t
write S_t
```

其成本不随 Context Length 增长，但需要每步读写整个状态矩阵。

这会形成新的 Memory Wall：

```text
标准 Attention Decode:
  读取不断增长的 KV

GDN Decode:
  每步读取并写回固定但较大的 State
```

当 Context 很短时，GDN 不一定天然更快，因为固定 State 可能比短 KV 更大，而且 Recurrent Kernel 启动与状态写回成本无法忽略。

### 12.2 Prefill：不能真的逐 Token 串行

直接按照 Recurrent 公式扫描 `L` 个 Token，会失去 GPU 并行性：

```text
S_1 -> S_2 -> S_3 -> ... -> S_L
```

高性能实现使用 Chunkwise Parallel Algorithm：

```text
Sequence
  -> 切成多个 Chunk
  -> Chunk 内用矩阵运算并行计算
  -> Chunk 间只传递压缩 State
```

Delta Rule 的低秩更新可写成 Householder-like 矩阵乘积，并使用 WY Representation 将一组顺序更新压缩为适合 GEMM/Triangular Solve 的形式。

概念流程：

```text
Chunk Input Q/K/V/Gate
  -> Intra-Chunk Pair Interaction
  -> 小型 Lower-Triangular Solve
  -> State Transition
  -> Chunk Output
  -> Next Chunk State
```

### 12.3 Kernel 的主要难点

- Chunk Size 太小：GEMM 利用率低。
- Chunk Size 太大：三角求解、临时张量和寄存器压力高。
- State Accumulator 需要更高精度。
- Gate 的指数累乘容易产生数值范围问题。
- Variable Length Batch 需要 Packed Layout。
- Decode 与 Prefill 的最优 Backend 不同。

因此 SGLang/vLLM 会分别选择：

```text
Linear Attention Prefill Backend
Linear Attention Decode Backend
```

而不是使用一个统一 Attention Kernel。

## 十三、为什么是 3:1 混合注意力，而不是纯线性注意力

Qwen3-Next、Qwen3.5 和 Qwen3.8 都采用：

```text
3 x Gated DeltaNet
1 x Gated Full Attention
```

### 13.1 两类 Layer 分工

```text
GDN Layer:
  流式压缩历史
  维护状态
  提供低成本长程统计

Full Attention Layer:
  访问具体历史 Token
  恢复高保真检索
  处理 Needle / Copy / In-Context Learning
```

### 13.2 复杂度必须严谨理解

若总层数为 `N`，Full Attention 层占 `1/4`：

```text
Prefill Attention:
  GDN 部分约 O(3N/4 * L)
  Full 部分约 O(N/4 * L^2)

KV Cache:
  只在 N/4 个 Full Attention 层随 L 增长

GDN State:
  在 3N/4 个层中与 L 无关
```

所以整个混合模型的渐进复杂度仍然包含：

```text
Prefill: O(L^2)
KV:      O(L)
```

它不是严格的全模型线性复杂度或全模型常数缓存，只是把二次 Attention 和线性 KV 的系数降低到约四分之一。

这是一个重要边界。宣传中的“线性注意力模型”常省略了保留的 Full Attention 层。

### 13.3 为什么这仍然很有价值

实际部署关心常数和可并行性：

- Full Attention 层减少 75%。
- KV Cache Layer 数减少 75%。
- 超长 Decode 中多数层不再扫描历史。
- 关键检索仍可由周期性 Full Attention 完成。

在 32K、256K、1M Context 下，这个常数变化足以改变系统瓶颈。

## 十四、Gated Attention、Partial RoPE 与稳定性设计

Qwen3-Next 的 Full Attention 也不是原样保留 Qwen3 Attention。

### 14.1 Output Gate

Attention 输出加入可学习 Gate：

```text
O = sigmoid(G(X)) * Attention(Q, K, V)
```

它让模型按 Token 和 Channel 控制 Attention 输出强度，并缓解 Attention Sink、低秩输出和 Massive Activation 等问题。

推理代价包括：

- 额外 Projection 或 Gate 计算。
- 逐元素激活与乘法。

但它改善数值稳定性，为更低精度和更深 MoE 扩展提供条件。

### 14.2 Head Dimension 从 128 到 256

Qwen3-Next Full Attention 使用更大的 Head Dimension，并减少 KV Head：

```text
Qwen3-Next-80B-A3B:
  Q Heads = 16
  KV Heads = 2
  Head Dim = 256

Qwen3.5-397B-A17B:
  Q Heads = 32
  KV Heads = 2
  Head Dim = 256

Qwen3.8-2.4T-A95B:
  Q Heads = 64
  KV Heads = 4
  Head Dim = 256
```

这是强 GQA：

```text
Qwen3.8:
  16 个 Q Head 共享一个 KV Head
```

更少 KV Head 控制 Full Attention 层的 KV Cache。

### 14.3 Partial RoPE

Qwen3-Next 只对 Head Dimension 的前 25% 应用 RoPE。

Qwen3.8：

```text
Head Dim = 256
RoPE Dim = 64
```

剩余维度不旋转。这样可以：

- 保留部分与绝对距离无关的内容通道。
- 减少长距离位置旋转对全部特征的扰动。
- 改善长度外推。

它不把 Attention 复杂度变成线性。

### 14.4 Zero-Centered RMSNorm

Qwen3 使用 QK-Norm 时观察到部分 Norm Weight 变大。Qwen3-Next 使用 Zero-Centered RMSNorm 并对 Norm Weight 加 Weight Decay。

概念上把 Scale 写成：

```text
y = RMSNorm(x) * (1 + w)
```

参数 `w` 以 0 为中心初始化，而不是直接学习远离 1 的 Scale。它主要是训练稳定性设计，但稳定的激活范围也有利于 FP8/FP4 推理。

## 十五、超稀疏 MoE：从 128 个专家到 512 个专家

### 15.1 架构变化

```text
Qwen3 MoE:
  128 Experts
  8 Routed
  无 Shared Expert

Qwen3-Next / Qwen3.5 / Qwen3.8:
  512 Experts
  10 Routed
  1 Shared Expert
```

Qwen3-Next-80B-A3B 只激活约 3.7% 参数。

### 15.2 Shared Expert 的含义

每个 Token 都经过 Shared Expert：

```text
y =
    SharedExpert(x)
  + sum_{e in TopK(x)} p_e * RoutedExpert_e(x)
```

Shared Expert 学习通用变换，Routed Expert 更容易专业化。

系统侧：

- Shared Expert 可以视为 Dense FFN 路径。
- Routed Expert 需要 Dispatch/Combine。
- Shared 与 Routed 计算可尝试重叠。

### 15.3 Qwen3.8 Expert 参数量粗算

Qwen3.8：

```text
Hidden = 8192
Expert Intermediate = 2048
Experts = 512
Layers = 92
```

一个 SwiGLU Expert 常包含三组矩阵：

```text
gate_proj: [8192, 2048]
up_proj:   [8192, 2048]
down_proj: [2048, 8192]
```

每 Expert 参数近似：

```text
3 * 8192 * 2048
= 50,331,648
~= 50.3M
```

每层 512 个 Expert：

```text
50.3M * 512 ~= 25.8B
```

92 层：

```text
25.8B * 92 ~= 2.37T
```

这解释了 2.4T 参数主要存在哪里。

每 Token 只激活 `10 Routed + 1 Shared`，但所有 Expert Weight 仍要分布在集群内。

### 15.4 Fine-Grained Expert 的收益与代价

收益：

- 更细的专业化粒度。
- 固定 Active Expert 数时扩大总容量。
- 每个 Expert 更小。

代价：

- 每 Expert 获得的 Token 更少。
- Grouped GEMM 的 M 更小。
- Router Metadata 更多。
- Expert Parallel 通信更离散。
- 热门 Expert 更容易形成尾部。

## 十六、Qwen3.5：原生多模态与推理效率合流

Qwen3.5-397B-A17B 延续 Qwen3-Next 架构，并把它扩展为原生 Vision-Language Foundation。

### 16.1 主要规格

```text
397B Total / 17B Active
60 Layers

15 x [
  3 x (Gated DeltaNet -> MoE)
  1 x (Gated Attention -> MoE)
]

512 Experts
10 Routed + 1 Shared
262K Native / ~1M Extended
MTP
Vision Encoder
```

### 16.2 Early Fusion 的系统影响

图像和视频先由 Vision Encoder 转成 Visual Token，再与 Text Token 进入统一语言主干：

```text
Image / Video
  -> Vision Encoder
  -> Visual Tokens
                     \
Text -> Text Tokens ---+-> Hybrid MoE LLM -> Text Output
```

优势：

- 视觉和文本在主干中共同推理。
- Agent 可以把截图、文档和工具输出放进同一上下文。

成本：

- Visual Token 增加 Prefill 长度。
- Vision Encoder 有独立计算和显存。
- 视频帧采样决定 TTFT。
- Media Preprocess 可能成为 CPU 瓶颈。

### 16.3 250K Vocabulary

Qwen3.5 将 Vocabulary 扩展到约 250K。官方报告在多数语言上获得 10%-60% 的 Token 编解码效率提升：

```text
同一段文本
  -> 更少 Token
  -> 更少 Prefill Position
  -> 更少 KV / State 更新
```

但 Vocabulary 更大也会增加：

- LM Head Matrix。
- Logits Tensor。
- Sampling Top-K/Softmax 成本。
- MTP Draft Head 的输出成本。

高性能框架需要 Vocab Parallel、Fused Sampling 和避免物化无用 Logits。

### 16.4 官方吞吐对比应如何理解

Qwen3.5 官方报告：

```text
相对 Qwen3-Max:
  32K Decode Throughput 约 8.6x
  256K Decode Throughput 约 19.0x

相对 Qwen3-235B-A22B:
  约 3.5x / 7.2x
```

长 Context 下倍数更大，说明收益主要来自：

- 75% Layer 不再读取增长 KV。
- Active Parameter 更少。
- GQA KV Head 更少。

这不是单个 Kernel 的固定加速比。

## 十七、Qwen3.8：2.4T/A95B 的最新开放权重架构

### 17.1 模型结构

Qwen3.8-2.4T-A95B：

```text
Type:
  Text-only Causal Language Model
  Thinking-only

Parameters:
  2.4T Total
  95B Active

Hidden:
  8192

Layers:
  92

Layout:
  23 x [
    Gated DeltaNet -> MoE
    Gated DeltaNet -> MoE
    Gated DeltaNet -> MoE
    Gated Attention -> MoE
  ]

GDN:
  128 V Heads
  16 QK Heads
  Head Dim 128

Full Attention:
  64 Q Heads
  4 KV Heads
  Head Dim 256
  RoPE Dim 64

MoE:
  512 Experts
  10 Routed + 1 Shared
  Expert Intermediate 2048

Context:
  262,144 Native
  ~1,010,000 Extended

MTP:
  Multi-Step Training
```

### 17.2 为什么称为 Max 级开放权重

此前 Qwen Max 主要通过 API 提供。Qwen3.8 首次将 Max 级别架构的权重公开。

这对 Infra 的意义不是“任何人都能本地运行”，而是：

- 推理框架能针对真实 Frontier MoE 做 Kernel 和通信优化。
- 可以研究 2.4T 模型的 EP/TP/FP4。
- 可以做私有部署与后训练。
- 开放评估不再只依赖 API。

### 17.3 它仍是数据中心级模型

理论权重大小：

```text
BF16:
  2.4T * 2 B ~= 4.8 TB

FP8:
  2.4T * 1 B ~= 2.4 TB

4-bit:
  2.4T * 0.5 B ~= 1.2 TB
```

还要加上：

- Scale / Metadata。
- LM Head 与 MTP。
- KV Cache。
- GDN State。
- Workspace。
- CUDA Graph Pool。
- 通信 Buffer。

所以即使 FP4，也通常需要高显存多 GPU 单节点；BF16/FP8 则需要多节点。

### 17.4 托管 Max 与开放权重版

| 项目 | Qwen3.8-2.4T-A95B | Qwen3.8-Max |
| --- | --- | --- |
| 权重 | 开放 | 托管服务 |
| 输入 | Text | Text / Image / Video |
| Thinking | 必须开启 | 支持多种模式 |
| Context | 262K Native，约 1M 外推 | 默认 1M |
| 工具 | 由部署者集成 | 官方内置工具 |
| 推理优化 | 由 vLLM/SGLang 等负责 | 服务端内部实现 |

部署和 Benchmark 时必须写清楚是哪一版。

## 十八、Qwen3.8 显存账本：权重、KV 与 GDN 状态

### 18.1 Full Attention KV Cache

Qwen3.8 只有 23 个 Full Attention Layer。

每 Token、每序列的 KV 元素：

```text
23 Layers
* 2 (K + V)
* 4 KV Heads
* 256 Head Dim
= 47,104 elements
```

BF16：

```text
47,104 * 2 B
= 94,208 B
~= 92 KiB / Token / Sequence
```

262,144 Token：

```text
94,208 * 262,144
~= 24.7 GB
~= 23.0 GiB
```

FP8 KV 近似减半：

```text
~= 12.35 GB / 262K Sequence
```

这还不包含 GDN State。

### 18.2 如果 92 层全是 Full Attention

保持同样 KV Head 配置，KV 会约为当前的 4 倍：

```text
~98.8 GB BF16 / 262K Sequence
```

混合注意力把这部分降到约四分之一。

### 18.3 GDN State 粗算

Qwen3.8 每个 GDN Layer：

```text
V Heads = 128
K Dim = 128
V Dim = 128
```

若每个 V Head 维护一个 `128 x 128` BF16 State：

```text
128 * 128 * 128 * 2 B
= 4 MiB / GDN Layer / Sequence
```

69 个 GDN Layer：

```text
69 * 4 MiB
= 276 MiB / Sequence
```

这是固定成本，不随 Context Length 增长，但随并发序列数线性增长。

实际引擎还要保存：

- Causal Conv State。
- Gate/State Alignment Padding。
- Prefix Branch Snapshot。
- Overlap Scheduler Extra Buffer。
- Speculative Decode State。

所以 SGLang 的 Qwen3.8 Recipe 明确指出：

> 对混合模型，高并发时稀缺资源可能是 GDN State Pool，而不再只是 KV Pool。

### 18.4 一个显存计算函数

```python
def full_attention_kv_bytes(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    sequence_length: int,
    bytes_per_element: int,
    batch_size: int = 1,
) -> int:
    return (
        batch_size
        * sequence_length
        * num_layers
        * 2  # K and V
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )


qwen38_kv = full_attention_kv_bytes(
    num_layers=23,
    num_kv_heads=4,
    head_dim=256,
    sequence_length=262_144,
    bytes_per_element=2,
)

print(f"{qwen38_kv / 1024**3:.2f} GiB")
```

这个值只是 Payload。Paged Allocator 的 Block 对齐和 Fragmentation 还会增加实际占用。

### 18.5 Context 越长，混合模型越占优势

设 Full Attention KV 每 Token 为 `a`，GDN 固定状态为 `s`：

```text
Hybrid State = a * L + s
```

短 Context：

```text
s 可能占主导
```

长 Context：

```text
a * L 占主导
混合模型相对全 Attention 的优势扩大
```

因此不能只用 2K Prompt Benchmark 评价 Qwen3.8 的架构收益。

## 十九、MTP：用模型原生 Draft 打破单 Token 串行 Decode

### 19.1 普通自回归

```text
step 1 -> token t+1
step 2 -> token t+2
step 3 -> token t+3
```

每个 Token 都要重新读取主模型权重。

### 19.2 MTP 训练目标

除了预测下一个 Token，还训练未来多个位置：

```text
Main Head:
  predict x[t+1]

MTP Head 1:
  predict x[t+2]

MTP Head 2:
  predict x[t+3]
```

简化损失：

```text
L =
  L_next_token
  + lambda_1 * L_t+2
  + lambda_2 * L_t+3
  + ...
```

Qwen3-Next 特别进行多步 MTP 训练，让训练时的 Draft 链更接近推理时的多步生成，提升真实接受率。

### 19.3 推理时不是直接相信 MTP

MTP 只负责 Draft，主模型仍要 Verify：

```text
Prefix
  -> MTP Draft: [d1, d2, d3]
  -> Main Model Verify in parallel
  -> Accept longest valid prefix
  -> Reject suffix and continue
```

所以结果仍由主模型分布决定。

### 19.4 为什么 MTP 比独立小模型 Draft 更适合 Qwen

MTP Head：

- 共享主干 Hidden State。
- Vocabulary 完全一致。
- 与主模型联合训练。
- 参数远少于完整 Draft Model。
- 不需要单独 Prefill 一套模型。

对 3B Active 的 MoE Target，另起一个 1B Dense Draft 未必足够便宜；原生 MTP Head 的相对开销更小。

### 19.5 期望接受长度

若每个 Draft Token 的近似条件接受率为 `alpha`，Draft 长度为 `K`：

```text
E[accepted] =
    1 + alpha + alpha^2 + ... + alpha^K
```

但实际加速还要除以：

```text
Draft Cost
+ Verify Cost
+ State Rollback
+ Scheduler / Sampling
+ 额外 Cache Capacity
```

### 19.6 为什么低并发开启、高并发关闭

低并发：

- Decode 强烈受权重带宽和 Kernel Launch 限制。
- Verify 多 Token 能提高算术强度。
- MTP 更容易降低 TPOT。

高并发：

- Continuous Batch 已经让 GEMM 变大。
- Draft Token 占用额外 KV/GDN State。
- Verify 增加 Batch Token 数。
- 接受率不足时浪费计算。

因此 Qwen3.8 的生产 Recipe 常见策略是：

```text
Low Latency:
  NEXTN / MTP On

High Throughput:
  MTP Off
```

不能把 MTP 视为无条件加速开关。

## 二十、ReplaySSM：线性注意力 Decode 的新优化方向

GDN Decode 的一个问题是每步都要：

```text
Read Full State
Update Full State
Write Full State
```

即使 State 与 Context Length 无关，State HBM 流量仍可能主导 Decode。

### 20.1 ReplaySSM 的核心

不在每步写回完整状态，而是：

```text
保存最近 L 步的小型输入记录
定期把它们 Flush 进完整 State
非 Flush Step 直接从 Checkpoint State + Ring Buffer 计算输出
```

示意：

```text
Full State S0  --read-----------------------+
                                             |
Ring Buffer: [update1, update2, update3] ----+-> current output

每 L 步:
  replay updates
  write new Full State
  clear ring
```

完整状态写回从：

```text
每步一次
```

摊薄为：

```text
每 L 步一次
```

### 20.2 为什么它也帮助 Speculative Decoding

普通 Recurrent State 是历史压缩结果。Draft 被拒绝后，没有像 KV Page 一样直接截断的 Token 维度。

ReplaySSM 缓存近期更新输入后：

- 接受 Token：保留对应记录。
- 拒绝 Token：丢弃 Ring Buffer 中未提交记录。
- 无需恢复多份完整 State Snapshot。

这降低了 GDN 模型 MTP Verify 的回滚成本。

### 20.3 收益边界

公开 ReplaySSM 实验在 Qwen3.5 等混合模型上报告：

```text
GDN Kernel:
  约 1.43x-1.64x

End-to-End:
  约 1.20x-1.27x
```

端到端更低，因为：

- MoE GEMM 没变。
- Full Attention 没变。
- Router/All-to-All 没变。
- Sampling 没变。

这是典型的 Amdahl 定律。

## 二十一、混合缓存：Paged KV、GDN State 与 Prefix Cache

纯 Transformer 只需管理按 Token 增长的 KV Page。Qwen3.8 需要同时管理：

```text
Full Attention KV Pages
GDN Recurrent State
Causal Conv State
MTP / Speculative State
Multimodal Prefix State
```

### 21.1 两种 Cache 的粒度不同

```text
Full Attention:
  每 Token 一组 K/V
  可按固定 Token Block 分页

GDN:
  一个 Prefix 结束位置对应一份压缩 State
  不是每个 Token 独立可拼接的 KV Row
```

### 21.2 Hybrid KV Cache Manager

vLLM 为不同 Layer Type 建立 Cache Group：

```text
Group A:
  23 Full Attention Layers
  Paged KV

Group B:
  69 GDN Layers
  Recurrent State
```

不同 Group 每 Token/每请求需要的字节不同。为了统一分页，框架可能调整 Logical Block Size，使不同 Group 的一个物理分配单元占用相近显存。

### 21.3 Prefix Cache 不能只保存 KV

两个请求共享前缀 `P`：

```text
Request A = P + suffix_A
Request B = P + suffix_B
```

复用时需要：

```text
Full Attention:
  P 对应的 KV Pages

GDN:
  扫描完 P 后的 State Snapshot
```

若只有 KV，没有 GDN State，后缀仍需从头扫描前缀才能重建状态。

### 21.4 State Snapshot 的空间代价

纯 KV Prefix Cache 可以让多个请求共享同一组 Token Page。

GDN Prefix Cache 若在很多分叉点保存完整 State：

```text
Radix Tree 每个 Branch
  -> 一份或多份 State Snapshot
```

显存会快速增长。

SGLang 因此提供不同 Mamba/GDN Radix Cache Strategy，在：

- 分叉点缓存。
- Overlap Scheduling。
- State Slot 数量。
- 可服务并发数。

之间取舍。

### 21.5 Prefix Cache 仍然有效，但抽象已经变化

错误说法：

```text
GDN 没有 KV，所以不能 Prefix Cache
```

更准确的是：

```text
可以缓存 Prefix Endpoint 的 Recurrent State，
但不能只复用普通 KV Block；
缓存、分叉、驱逐和回滚都要理解 State Snapshot。
```

## 二十二、MoE Kernel：Router、Dispatch、Grouped GEMM 与 Combine

### 22.1 Qwen3.8 每层执行链

```text
Hidden [T, 8192]
  -> Router [T, 512]
  -> Top-10 Routed Experts
  -> Token Permute / Dispatch
  -> 512 组变长 Expert GEMM
  -> SwiGLU
  -> 第二组 Expert GEMM
  -> Combine
  -> Shared Expert Output
```

### 22.2 Decode 小 Batch 下 Expert GEMM 为什么很差

设 Batch 中有 `T` 个 Token：

```text
Total Routed Assignments = T * top_k
Average Tokens Per Expert = T * top_k / num_experts
```

Qwen3.8：

```text
top_k = 10
num_experts = 512
```

若 Decode Batch 只有 64 Token：

```text
64 * 10 / 512
= 1.25 Token / Expert
```

很多 Expert 只有 0、1、2 个 Token。对应 GEMM 的 M 极小：

```text
[M_e, 8192] x [8192, 2048]
M_e ~= 1
```

此时问题更接近大量 GEMV，而不是高效大 GEMM。

### 22.3 为什么需要 Grouped GEMM

不能为每个 Expert 单独 Launch：

```python
for expert in experts:
    gemm(tokens[expert], weight[expert])
```

应在一个 Persistent/Grouped Kernel 中处理多个变长 Problem：

```text
Problem e:
  M = token_count[e]
  N = 2048
  K = 8192
```

Kernel 内部通过 Problem Visitor 或 Work Queue 动态领取 Tile，减少：

- Launch Overhead。
- 小 GEMM 尾部。
- Expert 间负载不均。

### 22.4 Router 和 Permute 不能忽略

Expert GEMM 变快后，以下操作会暴露：

- Top-K。
- Prefix Sum。
- Token Histogram。
- Permute。
- Scale。
- Unpermute / Combine。

高性能 FusedMoE 会尽量融合：

```text
Router -> Dispatch -> GEMM1 -> Activation -> GEMM2 -> Combine
```

或至少减少中间结果回写 HBM。

### 22.5 Expert 热点

平均 `1.25 Token/Expert` 不表示负载均匀：

```text
expert 17: 35 tokens
expert 93: 0 tokens
expert 201: 12 tokens
...
```

最慢 Rank 或最热 Expert 决定 Step Latency。

需要观测：

```text
Expert Token Histogram
Max / Avg Tokens Per Expert
Rank Send/Recv Imbalance
All-to-All Tail
Grouped GEMM Tile Utilization
```

## 二十三、并行策略：TP、EP、DP Attention 与 PP 如何组合

2.4T MoE 不能只靠一种并行。

### 23.1 Tensor Parallel

TP 切分 Dense Matrix：

```text
QKV Projection
Output Projection
LM Head
Shared Expert
```

优点：

- 单请求延迟低。
- 每张卡权重减少。

代价：

- 每层 AllReduce / ReduceScatter。
- 跨节点 TP 对网络延迟敏感。

### 23.2 Expert Parallel

EP 将不同 Expert 放到不同 Rank：

```text
rank 0: expert 0 ... 31
rank 1: expert 32 ... 63
...
```

Token 按 Router 结果 All-to-All：

```text
dispatch:
  token -> expert owner

combine:
  expert output -> original token owner
```

EP 降低每卡 Expert Weight，但增加不规则通信。

### 23.3 通信量粗算

Qwen3.8 Hidden 为 8192。BF16 一个 Token Hidden：

```text
8192 * 2 B = 16 KiB
```

一个 Token 路由到 10 个 Routed Expert，最朴素的 Dispatch Payload：

```text
16 KiB * 10 = 160 KiB / Token / MoE Layer
```

Combine 还要返回结果。

实际通信会受 TP Slice、本地 Expert 命中、量化和通信融合影响，但这个数量级说明：

> Active FLOPs 很低的超稀疏 MoE，可能被 Expert All-to-All 而不是 Tensor Core 限制。

### 23.4 Data Parallel Attention

MoE 部分适合 EP，Attention 部分不一定需要同样的切分。

DP Attention 的思路：

```text
Attention:
  每个 DP Rank 处理不同请求

MoE:
  所有 Rank 组成更大的 EP Group
```

这样可以：

- Attention 避免过宽 TP。
- MoE 使用 Wide EP 分摊 512 Expert。
- 高并发时提高吞吐。

### 23.5 Pipeline Parallel

PP 按 Layer 切分模型：

```text
Stage 0: layers 0-22
Stage 1: layers 23-45
Stage 2: layers 46-68
Stage 3: layers 69-91
```

它降低每 Rank Weight Memory，但：

- 单请求增加 Stage Hop。
- Decode Microbatch 小时 Bubble 明显。
- Hybrid State 和 MTP 跨 Stage 更复杂。
- 某些框架下 PP 与 Speculative Decoding 不能同时启用。

### 23.6 典型组合

```text
低并发、低延迟:
  较窄 EP
  较宽 TP
  MTP On

高并发、高吞吐:
  DP Attention
  Wide EP
  MTP Off
  Expert Load Balancing On

权重放不下:
  TP + EP + PP
  或 FP4 降到单节点
```

没有一个组合对所有负载最优。

## 二十四、量化：BF16、FP8、NVFP4/MXFP4 与 FP8 KV

### 24.1 为什么 MoE 特别需要权重量化

MoE 的计算按 Active Parameter 缩小，但 Weight Memory 按 Total Parameter 保留。

因此量化首先解决：

```text
模型能否放入更小拓扑
是否能从多节点变成单节点
每步 Expert Weight 读取字节数
```

从跨节点变成单 NVLink Domain，收益可能远大于单个 FP4 GEMM 的理论加速。

### 24.2 精度路径

```text
BF16:
  最高兼容性
  权重约 2 B/param

FP8:
  权重约 1 B/param + scale
  Hopper/Blackwell/MI300 系列有硬件支持

NVFP4:
  NVIDIA Blackwell 的 4-bit 浮点路径

MXFP4:
  支持 Microscaling 的硬件路径
  需要匹配 AMD/NVIDIA 实现和 Checkpoint
```

NVFP4 和 MXFP4 不是同一种可互换格式。

### 24.3 不是所有张量都应同精度

高性能部署常是混合精度：

```text
MoE Weight:
  FP4 / FP8

Dense / Attention Weight:
  FP8 或 BF16

Full Attention KV:
  FP8

GDN Recurrent State:
  常保留 BF16

Softmax / Norm / Accumulator:
  更高精度
```

GDN State 经数十万次递归更新，误差会累积。SGLang Qwen3.8 Recipe 显式设置 BF16 SSM State，这说明“全模型 FP4”不是所有内部状态都用 FP4。

### 24.4 FP8 KV 只缩小 Full Attention 部分

Qwen3.8 中：

```text
FP8 KV:
  23 个 Full Attention Layer 减半

GDN State:
  不自动变化
```

如果并发瓶颈来自 GDN State Slot，继续压 KV 的收益会变小。

### 24.5 量化评估

至少验证：

- Perplexity / 任务准确率。
- Thinking 长度是否异常。
- Tool Call JSON 是否稳定。
- Expert Router 分布是否漂移。
- MTP Acceptance 是否下降。
- 长上下文状态误差。
- Throughput 是否真的提高。

低比特 Kernel 不成熟时，量化模型也可能比 BF16/FP8 更慢。

## 二十五、Serving：Continuous Batching、Chunked Prefill 与 CUDA Graph

### 25.1 Continuous Batching

每个 Step 重新组合 Batch：

```text
running requests
  + newly admitted requests
  - finished requests
  -> next model step
```

对 Qwen 混合模型，调度预算不只包括 Token：

```text
Full Attention KV Blocks
GDN State Slots
MTP Draft Tokens
Media Tokens
Expert Parallel Capacity
```

### 25.2 Token Budget

```text
scheduled_tokens <= max_num_batched_tokens
running_requests <= state_pool_capacity
kv_blocks <= free_kv_blocks
```

Thinking Request 输出长度高度不确定，需要：

- Fair Scheduling。
- Max Output Budget。
- Preemption。
- Long-running Request 隔离。

### 25.3 Chunked Prefill 在混合模型中仍然重要

GDN Prefill 已是 Chunkwise，但服务层还需要 Chunked Prefill 控制：

- Vision/Text 长输入的单 Step 时间。
- Full Attention Layer 的峰值。
- Prefill 对 Decode TPOT 的阻塞。
- Pipeline Stage 工作粒度。

模型内部 Chunk 与 Serving Scheduler Chunk 是不同层次，大小也不必一致。

### 25.4 CUDA Graph

Decode Shape 重复，CUDA Graph 可减少：

- Python/CPU Launch。
- Dynamic Dispatch。
- 多 Kernel 间空隙。

但 Qwen3.8 有更多动态状态：

```text
Batch Size
Expert Token Histogram
MTP Verify Length
GDN State Slot
Full KV Block Table
Multimodal Path
```

框架通常为一组 Batch Size 预先 Capture：

```text
1, 2, 4, 8, 16, 32, ...
```

若 Graph Capture Max Batch 大于 GDN State Pool 可提供的 Cache Line，就会启动失败或浪费大量显存。

### 25.5 CPU 端也可能成为瓶颈

超稀疏 MoE + MTP + Dynamic Batch 需要处理更多 Metadata：

- Router Metadata。
- Block Table。
- Draft Tree。
- Accept/Reject。
- Tool/Reasoning Parser。
- Media Preprocess。

当 GPU Kernel 很短时，CPU Scheduler 和 Triton Launch Overhead 可能暴露。vLLM 对 Qwen3-Next 默认使用完整 CUDA Graph，正是为了降低 Decode-only Batch 的 Launch 开销。

## 二十六、P/D Disaggregation：混合模型需要传输哪些状态

Prefill 和 Decode 资源特征不同：

```text
Prefill:
  长序列
  大 GEMM
  Full Attention Compute-heavy
  GDN Chunkwise Parallel

Decode:
  单 Token
  Weight/State Bandwidth-heavy
  GDN Recurrent Update
  MTP / Sampling
```

因此可以分离：

```text
Prefill Worker
  -> 生成上下文状态
  -> 传输
Decode Worker
  -> 接管生成
```

### 26.1 纯 Transformer 传什么

```text
每层 K/V Cache
Block Metadata
Position / Request State
```

### 26.2 Qwen3.8 还要传什么

```text
23 层 Full Attention KV
69 层 GDN Recurrent State
Causal Conv State
MTP Draft State
Thinking / Parser State
Multimodal Prefix Metadata
```

若 Prefill Worker 和 Decode Worker 的 Cache Layout 不一致，还需重排或转换。

### 26.3 状态传输与计算重叠

可以按 Layer 或 Chunk 流水：

```text
Prefill Layer Group 0 complete
  -> transfer state group 0
  -> continue group 1

Decode Worker:
  wait until all required state ready
```

但 Recurrent State 是后缀计算的必要输入，不能像部分可延迟 KV 一样随意缺失。

### 26.4 什么时候值得 P/D 分离

适合：

- 长 Prompt 多。
- Decode SLO 严格。
- Prefill/Decode 最优硬件不同。
- 有高带宽网络。

不适合：

- Prompt 很短。
- 状态传输接近 Prefill 时间。
- 请求量不足以填满两类 Worker。
- GDN/MTP 状态传输支持不成熟。

## 二十七、Prefix Cache 与上下文缓存：Agent 场景的关键优化

Agent 请求经常重复：

```text
System Prompt
Tool Definitions
Repository Context
Policy Documents
Conversation Prefix
```

Qwen3.8-Max 甚至支持 1M Context。如果每个 Tool Turn 都重新 Prefill，成本不可接受。

### 27.1 自托管 Prefix Cache

vLLM/SGLang 在 Token 前缀相同时复用：

- Full Attention KV。
- Hybrid Model 的 GDN/Mamba State Snapshot。

收益：

```text
TTFT 下降
Prefill GPU 时间下降
重复工具描述和代码库不再重算
```

### 27.2 托管上下文缓存

Qwen API 提供显式和隐式 Context Cache：

```text
Explicit:
  调用方标记可缓存 Prefix
  命中确定性更强

Implicit:
  服务端自动识别公共 Prefix
  使用简单，但命中不保证
```

### 27.3 Cache Key 必须包含什么

```text
Model Revision
Tokenizer / Chat Template
Token IDs
Position Encoding Config
Thinking Mode
LoRA / Adapter
Multimodal Input Hash
Tool Definitions
Quantization / Execution Semantics
```

`preserve_thinking` 改变历史 Token，自然也改变 Cache Key。

### 27.4 Agent 路由要兼顾 Cache Locality

多个 Replica 时：

```text
Round Robin:
  负载均衡好
  Prefix 可能落到不同 Worker

Cache-aware Routing:
  Prefix 命中高
  热点 Worker 可能排队
```

应优化的是：

```text
Queueing Delay
+ Recompute Cost
```

而不是只追求 Cache Hit Rate。

## 二十八、多模态推理：Vision Encoder、Media Cache 与文本专用路径

### 28.1 Qwen3.5 请求路径

```text
Image / Video Decode
  -> Resize / Frame Sample
  -> Vision Encoder
  -> Visual Embedding
  -> Hybrid LLM Prefill
  -> Decode
```

### 28.2 Vision Encoder Data Parallel

Vision Encoder 的输入在请求间独立，适合 Data Parallel：

```text
GPU 0: encode request A media
GPU 1: encode request B media
...
```

语言主干则使用 TP/EP。

这比把 Vision Encoder 也做宽 TP 更适合高吞吐多请求。

### 28.3 Media Processor Cache

相同图片或视频可能在多轮中重复出现。可缓存：

- 解码后的 Frame。
- Resize/Normalize 结果。
- Vision Embedding。

Shared Memory Cache 能避免每个 Worker 重复预处理和跨进程复制。

### 28.4 文本专用部署

如果服务只处理文本，可以：

```text
skip Vision Encoder
skip Multimodal Profiling
free memory for KV / GDN State
increase concurrency
```

vLLM 对 Qwen3.5 提供 `--language-model-only`。

### 28.5 视频采样是质量与成本旋钮

视频 Token 数近似受：

```text
Duration
FPS
Resolution
Patch Size
Temporal Compression
```

共同决定。

提高 FPS 不只增加 Vision Encoder 成本，还增加语言主干 Prefill 和上下文状态。应针对任务调整，而不是永远使用最大帧率。

## 二十九、vLLM 与 SGLang 部署配置如何读

下面不是唯一最优命令，而是说明配置如何映射到前文的模型结构。

### 29.1 Qwen3.5 基础部署

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

含义：

```text
FP8:
  降低 397B Weight Memory 和 GEMM 成本

TP=8:
  将模型切到 8 GPU

262K:
  为 Native Context 预留 Cache

Reasoning Parser:
  将 Thinking 与 Final Answer 正确拆分
```

### 29.2 文本高吞吐

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --language-model-only \
  --enable-prefix-caching \
  --reasoning-parser qwen3
```

目标：

- DP Rank 处理不同请求。
- Expert 在 Rank 间分片。
- 不加载 Vision Encoder。
- 复用 Agent Prefix。

它适合高并发，不等于最低单请求延迟。

### 29.3 低并发 MTP

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --speculative-config \
    '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --reasoning-parser qwen3
```

应同时观测：

```text
Accepted Tokens / Verify
TPOT
Total Throughput
State Pool Occupancy
```

若高并发吞吐下降，应缩短 Draft 或关闭 MTP。

### 29.4 Qwen3.8 配置项映射

Qwen3.8 是多节点模型，正式部署应从目标硬件对应的官方 Recipe 开始。典型参数类别：

```text
--tp-size:
  Dense/Attention Tensor Parallel

--dp-size:
  并行处理不同请求

--ep-size:
  512 Expert 分片宽度

--enable-dp-attention:
  Attention 使用 DP，MoE 使用宽 EP

--moe-a2a-backend:
  Expert Dispatch/Combine 通信后端

--enable-eplb:
  Expert Parallel Load Balancing

--linear-attn-prefill-backend:
  GDN Chunkwise Prefill Kernel

--mamba-ssm-dtype:
  GDN Recurrent State 精度

--mamba-radix-cache-strategy:
  Prefix Branch 与 State Buffer 策略

--kv-cache-dtype:
  23 个 Full Attention Layer 的 KV 精度

--speculative-algorithm NEXTN:
  启用模型原生 MTP

--enable-linear-replayssm-spec:
  降低 Recurrent Spec Verify/回滚成本

--chunked-prefill-size:
  控制长 Prompt 单 Step 工作量
```

不要从别的 GPU 拷贝完整命令。Qwen3.8 的：

- FP8/FP4 Checkpoint。
- TP/EP 对齐。
- Linear Attention Backend。
- All-to-All Backend。
- CUDA Graph Batch。

都依赖硬件代际与框架版本。

## 三十、如何在 Qwen 模型之间做工程选型

### 30.1 先问五个问题

```text
1. 质量目标是什么？
2. 典型和 P99 Context 多长？
3. Thinking 输出多长？
4. 单请求延迟还是总吞吐优先？
5. 有多少 GPU、显存和互联带宽？
```

### 30.2 Dense 与 MoE

| 场景 | 更合适的方向 |
| --- | --- |
| 单机、简单部署、低通信 | Qwen3.6-27B 等 Dense |
| 高并发、较强能力、可做 EP | Qwen3.5-35B-A3B / 122B-A10B |
| 超长上下文 Agent | Qwen3-Next / Qwen3.5 Hybrid |
| Frontier 质量、机架级集群 | Qwen3.8-2.4T-A95B |
| 只需 API 与完整多模态能力 | Qwen3.8-Max |

Dense 27B 虽然 Active Parameter 高于 A3B，但没有 Router、All-to-All 和大量小 Expert GEMM。在小集群上可能更快、更稳定。

### 30.3 Context 不要按最大值配置

若业务 P99 只有 32K：

```text
直接配置 262K / 1M
  -> 预留过多 Cache
  -> 降低并发
  -> 增加 Graph/Allocator 压力
```

应为不同 Context Tier 建不同 Pool：

```text
Short:
  <= 16K

Medium:
  <= 64K

Long:
  <= 262K

Ultra-long:
  > 262K, YaRN / dedicated workers
```

### 30.4 Thinking Budget 分层

```text
Fast:
  Non-Thinking 或 Low Reasoning

Balanced:
  Medium Reasoning

Deep:
  XHigh / Large Budget
```

按请求难度路由，比所有请求都使用最大 Thinking 更有效。

### 30.5 量化选型

```text
Hopper:
  优先成熟 FP8

Blackwell:
  评估 NVFP4

AMD CDNA4:
  评估 MXFP4

消费级或 CPU:
  GPTQ / AWQ / GGUF
  但需确认 GDN/MTP Kernel 支持
```

## 三十一、性能评估、Profile 与调优清单

### 31.1 工作负载必须真实

至少覆盖：

```text
Input:
  2K / 32K / 128K / 262K / 1M

Output:
  128 / 2K / 32K Reasoning

Concurrency:
  1 / 8 / 64 / Saturation

Mode:
  Thinking / Non-Thinking / Tool Use

Prefix:
  Cold / Warm / Shared

Modality:
  Text / Image / Video
```

### 31.2 服务指标

- TTFT P50/P95/P99。
- TPOT/ITL P50/P95/P99。
- Request Latency。
- Output Token/s/GPU。
- Goodput under SLO。
- Queue Time。
- Preemption/Abort Rate。

### 31.3 Hybrid Attention 指标

- Full Attention Kernel Time。
- GDN Prefill Kernel Time。
- GDN Decode State Read/Write Bytes。
- GDN State Pool Capacity。
- KV Pool Capacity。
- Prefix GDN State Hit Rate。
- ReplaySSM Flush Frequency。

### 31.4 MoE 指标

- Router Time。
- Dispatch/Combine Time。
- All-to-All Time。
- Expert Token Max/Avg/P99。
- Empty Expert 比例。
- Grouped GEMM Tile Utilization。
- Rank 间 Send/Recv Imbalance。
- Shared Expert 与 Routed Expert Overlap。

### 31.5 MTP 指标

```text
Acceptance Rate by Position
Average Accepted Length
Draft Time
Verify Time
Rollback / Replay Time
Bonus Token Rate
Wasted Draft Tokens
MTP On/Off Total Throughput
```

不能只报告 Acceptance Rate。最终关心的是端到端 TPOT 和 Goodput。

### 31.6 Cache 指标

- Prefix Cache Token Hit Rate。
- Request Hit Rate。
- KV Block Occupancy。
- GDN Snapshot Occupancy。
- Cache Eviction。
- Cache-aware Router Queue Imbalance。
- Explicit/Implicit Context Cache 命中。

### 31.7 Nsight Systems / Compute

NSYS：

- Prefill、Decode、All-to-All 是否重叠。
- CPU Launch Gap。
- Pipeline Bubble。
- MTP Draft/Verify Timeline。
- P/D State Transfer。

NCU：

- GDN State Kernel 是否 HBM-bound。
- Grouped GEMM Tensor Core 利用率。
- Router/Permute 的 Memory Coalescing。
- All-to-All 前后 GPU Idle。
- Full Attention 的 L2/DRAM 与 Tensor Pipe。

### 31.8 单变量调优

每次只改一个主要变量：

```text
TP / EP
Max Batched Tokens
Max Running Requests
MTP Draft Length
KV Dtype
GDN State Strategy
Chunked Prefill Size
CUDA Graph Batch Set
Cache Routing Policy
```

同时验证准确率和服务指标。

## 三十二、常见误解

### 32.1 “Qwen3.8 是 Qwen3 的 8B 版本”

错误。这里的 `3.8` 是代际版本；`Qwen3-8B` 才表示 Qwen3 的 8B Dense Model。

### 32.2 “Qwen3.8-Max 和开放权重完全相同”

错误。开放权重版是 Text-only、Thinking-only；托管 Max 额外提供多模态、Non-Thinking、1M 默认 Context 和内置工具。

### 32.3 “A95B 表示只需存 95B 参数”

错误。95B 是每 Token 激活参数量。2.4T 权重仍要驻留或分片。

### 32.4 “混合注意力让整个模型变成 O(L)”

错误。25% Full Attention 仍让全模型 Prefill 含 `O(L^2)` 项，KV 仍含 `O(L)` 项，只是系数显著下降。

### 32.5 “GDN 没有 KV，所以 Context 完全免费”

错误。GDN 有固定 State，每步需要读写；Full Attention 层仍有 KV。超长 Context 还会增加 Prefill Token、Full KV 和请求持有时间。

### 32.6 “MoE Active FLOPs 少，所以单 Token 一定快”

错误。小 Batch 会产生大量极小 Expert GEMM，EP 还需要 All-to-All。低并发时 Dense Model 可能更快。

### 32.7 “MTP 一次预测 4 个 Token，就能 4 倍加速”

错误。Draft 需要验证，拒绝后的 Token 浪费，Recurrent State 还要回滚。高并发时 MTP 甚至可能降低总吞吐。

### 32.8 “FP4 会把所有显存都减半”

错误。KV、GDN State、Workspace、Scale 和部分敏感 Layer 不一定使用 FP4。

### 32.9 “Prefix Cache 对 GDN 无效”

错误。可以缓存 Prefix 结束位置的 Recurrent State，但需要不同于纯 KV Page 的 Snapshot、分叉与驱逐机制。

### 32.10 “最大 Context 配得越大越好”

错误。更大上限会预留更多 Cache，降低并发；静态 YaRN 还可能影响短文本能力。应按真实长度分层部署。

### 32.11 “官方吞吐倍数可以直接复现”

错误。吞吐取决于：

```text
GPU / Interconnect
Precision
Batch / Context
Framework Commit
Kernel Backend
MTP
TP/EP/DP
SLO
```

必须在相同条件下比较。

## 三十三、参考资料

### Qwen 官方资料

1. [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
2. [Qwen2.5-1M Technical Report](https://arxiv.org/abs/2501.15383)
3. [Qwen2.5-1M 官方博客](https://qwenlm.github.io/blog/qwen2.5-1m/)
4. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
5. [Qwen3: Think Deeper, Act Faster](https://qwen.ai/blog?id=1e3fa5c2d4662af2855586055ad037ed9e555125)
6. [Qwen3-235B-A22B Model Card](https://huggingface.co/Qwen/Qwen3-235B-A22B)
7. [Qwen3-Next: Towards Ultimate Training & Inference Efficiency](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd)
8. [Qwen3-Next-80B-A3B Model Card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
9. [Qwen3.5: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5)
10. [Qwen3.5-397B-A17B Model Card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
11. [Qwen3.6-27B 官方博客](https://qwen.ai/blog?id=qwen3.6-27b)
12. [QwQ-32B 官方博客](https://qwenlm.github.io/blog/qwq-32b/)
13. [Qwen3-Coder 官方博客](https://qwen.ai/blog?id=qwen3-coder)
14. [Qwen3-Coder-Next Model Card](https://huggingface.co/Qwen/Qwen3-Coder-Next)
15. [Qwen3-VL 官方博客](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef)
16. [Qwen3-Omni 官方博客](https://qwen.ai/blog?id=65f766fc2dcba7905c1cb69cc4cab90e94126bf4)
17. [Qwen3.8-Max 官方博客](https://qwen.ai/blog?id=qwen3.8)
18. [Qwen3.8-2.4T-A95B Model Card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)
19. [Qwen3.8-Max API 文档](https://help.aliyun.com/en/model-studio/qwen3-8-max)
20. [Qwen Context Cache 文档](https://help.aliyun.com/zh/model-studio/context-cache)

### 架构与算法

21. [Gated Delta Networks](https://arxiv.org/abs/2412.06464)
22. [Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484)
23. [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708)
24. [Better & Faster Large Language Models via Multi-Token Prediction](https://arxiv.org/abs/2404.19737)
25. [Dual Chunk Attention](https://arxiv.org/abs/2402.17463)
26. [MInference](https://arxiv.org/abs/2407.02490)
27. [ReplaySSM](https://tridao.me/blog/2026/replayssm/)

### 推理框架与部署

28. [vLLM Qwen3-Next 支持说明](https://blog.vllm.ai/2025/09/11/qwen3-next.html)
29. [vLLM Hybrid KV Cache Manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/)
30. [vLLM Qwen3.5/3.6 Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
31. [vLLM Qwen3.8 Recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-2.4T-A95B)
32. [SGLang Qwen3-Next Recipe](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-Next)
33. [SGLang Qwen3.8 Recipe](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8)
34. [SGLang Speculative Decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
35. [NVIDIA Qwen3.8 GB300 部署说明](https://developer.nvidia.com/blog/serve-qwen3-8-2-4t-a95b-a-2-4t-parameter-model-with-configurable-reasoning-on-nvidia-gb300-nvl72/)

Qwen 的演进展示了一个清晰趋势：

```text
Qwen2.5:
  用系统优化支撑 Full Attention 长上下文

Qwen3:
  用 MoE 和 Thinking Budget 调整每 Token 与每请求成本

Qwen3-Next / Qwen3.5:
  用 Hybrid Attention、Ultra-Sparse MoE 和 MTP
  从模型结构上为推理优化

Qwen3.8:
  把同一套思想扩展到 2.4T，
  推理进入模型、Kernel、通信、缓存和调度的全栈协同阶段
```

真正理解 Qwen 推理，不能只看参数量或 Benchmark。需要同时回答：

```text
多少参数被激活？
多少权重必须驻留？
哪些层保存增长 KV？
哪些层保存固定 Recurrent State？
每步有多少 Expert Token？
通信发生在哪里？
MTP 实际接受多少 Token？
Thinking 让请求持有资源多久？
Prefix 能复用哪些状态？
```

这些问题共同决定一个 Qwen 模型在真实服务中的延迟、吞吐、显存和成本。
