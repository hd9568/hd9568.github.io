---
title: 'vLLM 与 SGLang 源码面试题：17 个 AI Infra 高频问题'
description: '基于 vLLM V1 和 SGLang SRT 当前源码，整理调度器、Paged KV Cache、Prefix Cache、Chunked Prefill、CUDA Graph、投机解码、并行与 P/D 分离等 AI Infra 高频问题。'
category: '推理优化'
pubDate: '2026-08-17T15:00:00+08:00'
updatedDate: '2026-08-17T15:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [vLLM 和 SGLang 的请求执行链路分别是什么](#q1vllm-和-sglang-的请求执行链路分别是什么)
2. [两个框架如何实现 Continuous Batching](#q2两个框架如何实现-continuous-batching)
3. [Paged KV Cache 解决什么问题，两个框架如何管理](#q3paged-kv-cache-解决什么问题两个框架如何管理)
4. [vLLM Prefix Cache 与 SGLang RadixAttention 有什么区别](#q4vllm-prefix-cache-与-sglang-radixattention-有什么区别)
5. [Prefix Cache 如何驱逐，怎样保证多租户正确性](#q5prefix-cache-如何驱逐怎样保证多租户正确性)
6. [两个框架如何实现 Chunked Prefill](#q6两个框架如何实现-chunked-prefill)
7. [KV Cache 不足时如何抢占请求](#q7kv-cache-不足时如何抢占请求)
8. [SGLang Overlap Scheduler 与 vLLM 异步调度是什么](#q8sglang-overlap-scheduler-与-vllm-异步调度是什么)
9. [两个框架如何使用 CUDA Graph](#q9两个框架如何使用-cuda-graph)
10. [Attention Backend 为什么不能随意替换](#q10attention-backend-为什么不能随意替换)
11. [两个框架如何实现投机解码](#q11两个框架如何实现投机解码)
12. [结构化输出如何在 GPU 采样中实现](#q12结构化输出如何在-gpu-采样中实现)
13. [TP、PP、DP、EP 在两个框架里如何组合](#q13tpppdpep-在两个框架里如何组合)
14. [两个框架如何实现 Prefill-Decode 分离](#q14两个框架如何实现-prefill-decode-分离)
15. [权重量化和 KV Cache 量化有什么区别](#q15权重量化和-kv-cache-量化有什么区别)
16. [多模态请求如何缓存和调度](#q16多模态请求如何缓存和调度)
17. [如何正确比较 vLLM 与 SGLang 的性能](#q17如何正确比较-vllm-与-sglang-的性能)

## 阅读说明

本文按 vLLM V1 与 SGLang SRT 当前主干设计整理。两者迭代很快，类名和配置可能变化，但核心抽象相对稳定。

每题建议按以下顺序回答：

```text
先说解决什么问题
-> 再说核心数据结构
-> 再说一次执行流程
-> 最后说代价和边界
```

不要只背参数，也不要简单回答“都用了 PagedAttention”。

## Q1：vLLM 和 SGLang 的请求执行链路分别是什么

**直接回答**

两个框架都把 API、调度和 GPU 执行解耦，但进程组织和核心对象不同。

```text
vLLM V1:
API Server
  -> AsyncLLM / EngineCore Client
  -> EngineCore
  -> Scheduler
  -> Executor / Worker
  -> GPUModelRunner
  -> OutputProcessor

SGLang:
HTTP Frontend
  -> TokenizerManager
  -> Scheduler
  -> ScheduleBatch
  -> ModelRunner
  -> Output Stream / Detokenization
```

vLLM 的 `EngineCore` 负责初始化模型执行器、KV Cache 和 Scheduler，一轮核心流程是：

```text
schedule
-> execute_model
-> update_from_output
```

SGLang 的 `Scheduler` 同时管理 `waiting_queue`、`running_batch`、Prefix Cache 和 ModelRunner，并将 CPU 侧 `ScheduleBatch` 转成 GPU 侧 `ForwardBatch`。

**关键差异**

- vLLM V1 更强调统一、模块化的 EngineCore/Scheduler/Executor 接口。
- SGLang Scheduler 与 Radix Cache、Overlap Schedule、P/D Mixins 结合更紧。
- 两者都可能使用多进程和 ZMQ，不能把 Python API 调用等同于 GPU 同步执行。

**源码入口**

```text
vLLM:
  vllm/v1/engine/core.py
  vllm/v1/core/sched/scheduler.py
  vllm/v1/worker/gpu_model_runner.py

SGLang:
  python/sglang/srt/managers/tokenizer_manager.py
  python/sglang/srt/managers/scheduler.py
  python/sglang/srt/model_executor/model_runner.py
```

## Q2：两个框架如何实现 Continuous Batching

**直接回答**

Continuous Batching 的核心不是等待固定 Batch，而是每个模型 Step 后重新决定运行哪些请求。

vLLM V1 不再硬分 Prefill 和 Decode 队列，而是统一计算：

```text
num_new_tokens =
  当前已知 Token 数
  - 已完成计算的 Token 数
```

`Scheduler.schedule()` 先给 Running 请求分配 Token Budget，再从 Waiting Queue 接纳新请求：

```text
sum(num_scheduled_tokens)
<= max_num_batched_tokens
```

SGLang 保留更显式的：

```text
waiting_queue
running_batch
chunked_req
last_batch
```

`get_next_batch_to_run()` 决定本轮运行 Decode、Prefill、混合 Batch，`PrefillAdder` 在 KV 和 Token 预算内接纳请求。

**关键差异**

- vLLM V1 的统一 Token 抽象让 Prefill、Decode 和 Spec Verify 共用调度逻辑。
- SGLang 通过 `SchedulePolicy` 支持 FCFS、Longest Prefix Match、DFS Weight 等 Cache-aware 排序。
- 两者都同时受 Token Budget、请求数和 KV 容量约束。

**容易答错**

Continuous Batching 不表示“Batch 一直增大”。请求完成后会退出，新请求才能补入。

## Q3：Paged KV Cache 解决什么问题，两个框架如何管理

**直接回答**

连续显存分配会产生内部碎片和外部碎片。Paged KV Cache 将每个请求的逻辑 Token 映射到不连续物理页：

```text
Request Logical Tokens
  -> Block Table / Token Mapping
  -> Physical KV Slots
```

vLLM 以固定 Token Block 为基本单位：

- `KVCacheManager` 负责查找缓存和分配 Block。
- `BlockPool` 管理物理 Block、引用计数和空闲队列。
- ModelRunner 将 Block Table 转成 Attention Slot Mapping。

SGLang 使用两级映射：

```text
ReqToTokenPool:
  request -> token slot indices

TokenToKVPoolAllocator:
  token slot -> physical K/V storage
```

Attention Kernel 根据映射读取非连续 KV。

**核心收益**

- 按需增长，不按最大长度提前预留。
- 请求结束后页可立即复用。
- Prefix Cache 可让多个请求引用相同物理页。
- Speculative Decoding 可提前预留 Lookahead Slot。

**容易答错**

Paged KV Cache 主要减少浪费和提高并发，不会减少每个有效 Token 的理论 KV Payload。

## Q4：vLLM Prefix Cache 与 SGLang RadixAttention 有什么区别

**直接回答**

两者都复用最长公共前缀 KV，主要区别是索引结构。

### vLLM：Hash Chain

每个完整 Block 的 Key 近似为：

```text
H_i = hash(
  H_{i-1},
  tokens_in_block_i,
  LoRA / multimodal / cache_salt
)
```

全局 Hash Table 将 `block_hash` 映射到物理 KV Block。vLLM 只缓存完整 Hash Block；即使命中完整 Prompt，也要重算最后一个 Token 以获得 Logits。

### SGLang：Compressed Radix Tree

```text
root
 └─ system prompt
     ├─ user A suffix
     └─ user B suffix
```

Radix Node 保存一段 Token Key 和对应 KV Index。匹配在 Node 中间结束时会 `_split_node()`，`lock_ref` 保护正在使用的路径不被驱逐。

**关键差异**

| vLLM | SGLang |
| --- | --- |
| Block Hash Table | Token Radix Tree |
| 只复用完整 Hash Block | 按 Page 对齐后匹配树前缀 |
| Block 相对独立 | 显式保存前缀父子关系 |
| 易加入 LoRA、图片 Hash、Salt | 易做 LPM/DFS Cache-aware 调度 |

不能简单说 Radix Tree 一定比 Hash 快。命中率还取决于路由、Block/Page Size、驱逐和真实 Prompt 分布。

## Q5：Prefix Cache 如何驱逐，怎样保证多租户正确性

**直接回答**

正在被请求引用的 KV 不能驱逐，只能回收引用计数为零的缓存。

vLLM 的 `KVCacheBlock` 有：

```text
ref_cnt
block_hash
block_id
```

无空闲 Block 时，从可驱逐 Block 中按缓存顺序回收，并删除旧 Hash 映射。

SGLang 在 Radix Tree Node 上维护：

```text
lock_ref
last_access_time
hit_count
```

优先驱逐无锁 Leaf，支持 LRU/LFU 等策略；驱逐后若父节点也成为无锁 Leaf，可继续向上回收。

**多租户正确性**

Cache Key 不能只有 Token：

```text
Token IDs
+ Model / Revision
+ LoRA ID
+ Multimodal Content Hash
+ Position/Cache Semantics
+ Tenant Cache Salt
```

vLLM 默认支持安全 Hash 算法和 Cache Salt。SGLang 的 `RadixKey.extra_key` 可隔离 LoRA、Salt 或其他命名空间。

**容易答错**

Prefix Cache 命中不会改变模型结果，但错误 Hash、跨 Adapter 复用或碰撞可能直接造成错误结果和数据泄露。

## Q6：两个框架如何实现 Chunked Prefill

**直接回答**

Chunked Prefill 将长 Prompt 分成多轮计算，目的是控制单 Step 工作量和激活峰值：

```text
Prompt 12K
-> 4K + 4K + 4K
```

vLLM V1 中，Waiting 请求的 `num_new_tokens` 会被以下约束截断：

```text
Token Budget
Long Prefill Threshold
Max Model Length
KV Allocation
Speculative Lookahead
```

未完成的请求进入 Running，下一轮从 `num_computed_tokens` 继续。

SGLang 用 `chunked_req` 保存跨轮 Prefill 状态，`PrefillAdder.add_chunked_req()` 分配当前 Chunk；新 Prefill 可与 Running Decode 合并。长上下文 PP 场景还支持 Dynamic Chunking。

**权衡**

- Chunk 小：Decode TPOT 更稳定，但 Kernel/调度开销增加，长请求 TTFT 可能变差。
- Chunk 大：Prefill 吞吐高，但更容易阻塞 Decode。

**容易答错**

Chunked Prefill 主要降低峰值和 Head-of-Line Blocking，不会自动减少 Full Attention 的总计算量。

## Q7：KV Cache 不足时如何抢占请求

**直接回答**

当 Scheduler 选中请求但无法分配新 KV Slot 时，需要释放部分 Running 请求。

vLLM 中：

```text
allocate_slots() returns None
-> 选择 Preemption Victim
-> free KV blocks
-> Request 重新放回 Waiting
-> 之后 Recompute
```

V1 默认重点使用 Recompute，而不是把大量 KV Swap 到 CPU。原因是 PCIe 传输和状态管理可能比重算 Prefix 更贵。

SGLang 称为 Retract：

```text
check_decode_mem()
-> retract_decode()
-> 释放可回收 KV
-> 请求重新入队
```

若 Prefix 已进入 Radix Cache，恢复时可复用仍存在的公共 Prefix；其余部分重算。

**怎样减少抢占**

- 降低 `max_num_seqs` / `max_running_requests`。
- 降低 Token Budget。
- 增加 KV Cache 显存。
- 使用 FP8 KV。
- 减少最大 Context。

**面试追问**

抢占保证服务不 OOM，但会增加重复 Prefill、TTFT 和 P99，频繁发生说明 Admission Control 配置有问题。

## Q8：SGLang Overlap Scheduler 与 vLLM 异步调度是什么

**直接回答**

它们都尝试重叠 CPU 调度、GPU 执行和输出回传，而不是让每轮严格同步。

### SGLang

同步路径：

```text
schedule N
-> GPU N
-> process output N
-> schedule N+1
```

Overlap 路径：

```text
GPU N 执行时
  CPU 处理 N-1 输出并准备 N+1
```

`event_loop_overlap()`、`forward_stream` 和 CUDA Event 用于控制读写依赖。代价是存在 In-flight Batch、输出延迟一拍，KV/状态释放必须推迟到 GPU 真正不再访问。

### vLLM

`EngineCore` 可启用 Batch Queue 和异步调度，ModelRunner 还使用独立 Copy Stream 将采样结果非阻塞拷回 CPU。Pipeline Parallel 时 Batch Queue 用来减少 Bubble。

**关键区别**

Overlap Scheduler 是 Host/GPU 流水化；CUDA Graph 是复用一组已捕获 GPU Launch。两者解决不同问题，可以同时存在。

## Q9：两个框架如何使用 CUDA Graph

**直接回答**

Decode 每步 Shape 相对稳定但 Kernel 很碎，CUDA Graph 将多次 Launch 捕获后一次 Replay，降低 CPU 和 Driver 开销。

### vLLM

V1 支持：

```text
FULL
PIECEWISE
FULL_AND_PIECEWISE
NONE
```

`CudagraphDispatcher` 根据 `BatchDescriptor`、Attention Backend 能力和 Capture Key 选择 Full、Piecewise 或 Eager。没有匹配 Graph 时回退 `NONE`。

### SGLang

- Decode Graph：为一组 Batch Size 预先 Capture，运行时 Pad 到捕获尺寸。
- Piecewise CUDA Graph：借助 `torch.compile` 将动态 Extend/Prefill 图切成多个 Piece，每个 Piece 按 Token Bucket 捕获。

**代价**

- 启动 Capture 慢。
- 静态 Buffer 和 Graph Pool 占显存。
- 动态控制流、MoE Dispatch、Attention Backend 可能造成 Graph Break。
- Padding 到更大 Bucket 会做无效计算。

**源码入口**

```text
vLLM:
  vllm/v1/cudagraph_dispatcher.py
  vllm/v1/worker/gpu_model_runner.py

SGLang:
  python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py
  python/sglang/srt/compilation/cuda_piecewise_backend.py
```

## Q10：Attention Backend 为什么不能随意替换

**直接回答**

Attention Backend 不只是一个 Kernel 函数，还定义了 KV Layout、Page Size、Metadata 和功能兼容性。

Backend 选择至少取决于：

```text
MHA / MLA / GDN / Sparse Attention
Prefill 或 Decode
GPU 架构
Head Dim 和 Dtype
KV Page Size
FP8/FP4 KV
Sliding Window
Speculative Tree
CUDA Graph
```

vLLM 通过 `AttentionBackend`、`AttentionMetadataBuilder` 和 KV Cache Spec 抽象不同实现，`GPUModelRunner` 为每个 Attention Group 构造 Metadata。

SGLang 在 `python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py` 中构造 Backend，可分别指定 MHA、MLA、GDN、Spec Draft/Verify，当前还支持 Prefill/Decode 使用不同 Backend。

**容易答错**

把 FlashInfer 换成 FA3 不只是改一个参数。Page Size、KV Dtype、Spec Top-K 或 Multimodal 不兼容时，可能启动失败、回退慢路径或结果错误。

## Q11：两个框架如何实现投机解码

**直接回答**

投机解码包含三步：

```text
Propose K Draft Tokens
-> Target 一次并行 Verify
-> 接受最长合法前缀并拒绝后缀
```

vLLM 的 ModelRunner 中有 EAGLE、MTP、Draft Model、N-Gram、Suffix 等 Proposer，Scheduler 为 Draft Token 预留 Lookahead KV Slot，`RejectionSampler` 解析 Target 输出。

SGLang 在 `srt/speculative/` 中实现 EAGLE-2/3、MTP、DFlash、Standalone Draft 和 N-Gram，并为 Draft、Extend、Verify 分别维护 Attention Metadata 与 CUDA Graph。Overlap Spec 路径还会并行准备下一轮。

**性能判断**

近似收益取决于：

```text
平均接受长度
Draft 成本
Verify 成本
回滚成本
Batch Size
```

低并发、Target Decode 受权重带宽限制时最容易受益。高并发下 Target GEMM 已经高效，Draft 可能降低总吞吐。

**容易答错**

一次 Draft 4 个 Token 不等于 4 倍加速；被拒绝的 Token 仍消耗计算和临时 Cache。

## Q12：结构化输出如何在 GPU 采样中实现

**直接回答**

不是生成完整文本后再检查 JSON，而是在每个 Decode Step 只允许 Grammar 合法的 Token：

```text
Grammar State
-> Allowed Token Bitmask
-> Mask Invalid Logits
-> Sample
-> Advance Grammar State
```

vLLM 的 `StructuredOutputManager` 编译 Grammar，可异步生成 `StructuredOutputGrammar`；Scheduler 在 Grammar 未就绪时暂缓请求，ModelRunner 在采样前调用 `apply_grammar_bitmask`。

SGLang 的 `GrammarManager` 支持 XGrammar、Outlines 和 llguidance，Grammar Object 随请求状态推进，Sampling Batch 使用 Vocab Mask。

**为什么需要异步编译**

复杂 JSON Schema/Regex 编译可能阻塞调度线程。缓存和异步编译可避免请求首次命中时拖慢整个 Batch。

**容易答错**

Grammar 保证语法约束，不保证字段语义正确，也可能因为合法 Token 集很小降低采样吞吐。

## Q13：TP、PP、DP、EP 在两个框架里如何组合

**直接回答**

四种并行解决不同问题：

| 并行 | 切分对象 | 主要通信 |
| --- | --- | --- |
| TP | 单层矩阵 | AllReduce/AllGather |
| PP | 模型层 | Stage 间激活 |
| DP | 请求 | 路由与负载均衡 |
| EP | MoE Expert | All-to-All |

vLLM 使用 `ParallelConfig` 和 Executor 创建 TP/PP/DP Group；MoE 可启用 EP，并与 DP 组合分片 Expert。Pipeline Parallel 使用 Batch Queue 减少 Bubble。

SGLang 的 `ParallelState` 同时描述 TP、PP、DP、Attention CP 和 MoE EP。DP Attention 可以让 Attention 按请求做 DP，而 MoE 使用更宽 EP；`--moe-a2a-backend` 与 `--moe-runner-backend` 分别选择通信和 Grouped GEMM。

**选择原则**

- 单机低延迟：优先 TP。
- 多机超大模型：PP 避免跨机每层 AllReduce。
- 高吞吐：DP。
- 大 MoE：EP + DP Attention。

并行度更大不一定更快，通信拓扑必须与 NVLink/IB 对齐。

## Q14：两个框架如何实现 Prefill-Decode 分离

**直接回答**

P/D 分离将计算密集的 Prefill 和带宽敏感的 Decode 放在不同实例：

```text
Client
  -> Router
  -> Prefill Worker
  -> KV Transfer
  -> Decode Worker
  -> Stream Output
```

vLLM 启动两个 Engine，通过 `vllm/distributed/kv_transfer/` 下的 KV Connector 传输 KV，支持 NIXL、Mooncake、LMCache 等后端。Scheduler Connector 和 Worker Connector 分别负责控制面与数据面。

SGLang 在 Scheduler 上组合 Prefill/Decode Disaggregation Mixin，支持 Mooncake、NIXL 等传输后端，并由 Router 配对 P/D Worker。

**收益**

- Prefill 与 Decode 独立扩缩容。
- 分别优化 TTFT 和 TPOT。
- 长 Prefill 不再直接打断 Decode。

**代价**

- KV 传输增加延迟和网络带宽。
- Router 要管理 P/D 配对和失败。
- Prefix Cache Locality 更复杂。

P/D 分离不保证提高吞吐，主要价值是资源解耦和尾延迟稳定。

## Q15：权重量化和 KV Cache 量化有什么区别

**直接回答**

```text
Weight Quantization:
  减少模型权重显存与每步权重读取

KV Cache Quantization:
  减少随并发和 Context 增长的状态显存与 Attention 读取
```

vLLM 在模型加载与 Linear/MoE Layer 中选择 AWQ、GPTQ、FP8、NVFP4 等 Quant Method；KV 由 `kv_cache_dtype`、Scale 和 Attention Backend 共同决定。

SGLang 同样将 Weight GEMM Backend 与 KV Cache Dtype 分离，可为 MoE Runner、Attention Backend 选择不同实现。

**为什么不能只看位宽**

- W4A16 可能受 Dequant Kernel 限制。
- FP8 KV 需要 Scale，且 Backend 必须支持。
- FP4 Weight 不表示 Norm、Softmax、Accumulator 也用 FP4。
- 量化后模型可能放进更小拓扑，避免跨机通信，这往往比单 Kernel 加速更重要。

**观测**

同时比较准确率、模型显存、KV 容量、TTFT、TPOT 和吞吐。

## Q16：多模态请求如何缓存和调度

**直接回答**

多模态请求多一条 Encoder 路径：

```text
Image/Video
-> Processor
-> Vision Encoder
-> Embedding
-> LLM Prefill
```

vLLM 有独立 `EncoderCacheManager` 和 Multimodal Budget。Prefix Block Hash 会加入图片等输入的 Content Hash，避免相同 Placeholder Token 对应不同图片却错误复用 KV。

SGLang 的 `MultimodalDataItem` 保存内容 Hash 和特殊 Pad Value，Radix Key 因而能区分媒体；TokenizerManager 可处理原始媒体、Processor Output 或预计算 Embedding。更大部署还可做 Encoder-Prefill-Decode 分离。

**Scheduler 还要约束**

```text
Encoder Compute Budget
Encoder Cache Capacity
Visual Token 数
LLM KV Capacity
```

**容易答错**

只 Hash `<image>` Placeholder 是错误的。Cache Key 必须包含真实媒体内容或稳定内容摘要。

## Q17：如何正确比较 vLLM 与 SGLang 的性能

**直接回答**

没有脱离 Workload 的“谁更快”。至少固定：

```text
Model / Revision
Precision
GPU 与拓扑
Attention/MoE Backend
Input/Output Length 分布
并发度和到达率
Prefix 重复率
Spec Decode
CUDA Graph
SLO
```

同时报告：

| 指标 | 说明 |
| --- | --- |
| TTFT P50/P99 | Prefill、排队与 Cache |
| TPOT/ITL P50/P99 | Decode 交互速度 |
| Output Tokens/s | 总生成吞吐 |
| Request Goodput | 满足 SLO 的有效吞吐 |
| Prefix Hit Rate | 重复前缀收益 |
| KV Usage/Preemption | 容量压力 |
| Spec Accept Length | 投机解码有效性 |
| A2A/Collective Time | 分布式瓶颈 |

**判断框架差异**

- Prefix 高重复：关注 Hash/Radix 命中和 Cache-aware Router。
- 低并发 Decode：关注 Full CUDA Graph、Spec Decode、CPU Overhead。
- 长 Prompt 混部：关注 Chunked Prefill 和 TPOT P99。
- 大 MoE：关注 EP Backend、负载均衡和通信重叠。

正确结论应是“在某个负载和 SLO 下哪个配置更优”，而不是只比较离线峰值 Tokens/s。

## 一页速记

| 问题 | vLLM 关键词 | SGLang 关键词 |
| --- | --- | --- |
| 架构 | EngineCore、Executor | TokenizerManager、Scheduler、ModelRunner |
| 调度 | Unified Token Budget | RunningBatch、PrefillAdder |
| KV | BlockPool、BlockTable | ReqToTokenPool、TokenToKVPool |
| Prefix | Hash Chain | Radix Tree |
| Chunked Prefill | 截断 `num_new_tokens` | `chunked_req` |
| 抢占 | Preempt + Recompute | Retract + Recompute |
| CPU/GPU 重叠 | Async Scheduling、Batch Queue | Overlap Scheduler |
| CUDA Graph | Dispatcher、Full/Piecewise | Decode Graph、PCG |
| Spec Decode | Proposer + RejectionSampler | Spec Worker + Verify |
| P/D | KV Connector | Disaggregation Mixins |

## 参考资料

### vLLM

1. [vLLM V1 Guide](https://docs.vllm.ai/en/stable/usage/v1_guide/)
2. [vLLM Scheduler Source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
3. [vLLM KVCacheManager Source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)
4. [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
5. [vLLM CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
6. [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
7. [vLLM Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
8. [vLLM Structured Outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/)

### SGLang

9. [SGLang Scheduler Source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)
10. [SGLang Radix Cache Source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
11. [SGLang Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
12. [SGLang Attention Backend](https://docs.sglang.io/docs/advanced_features/attention_backend)
13. [SGLang Piecewise CUDA Graph](https://docs.sglang.io/advanced_features/piecewise_cuda_graph.html)
14. [SGLang Speculative Decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
15. [SGLang Structured Outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs)
16. [SGLang Expert Parallelism](https://docs.sglang.io/advanced_features/expert_parallelism.html)
17. [SGLang PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)

这 17 题的共同主线是：

```text
请求怎样进入系统
-> Scheduler 怎样分配 Token 和状态
-> KV 怎样分页、复用和回收
-> ModelRunner 怎样降低 Launch 与 Kernel 成本
-> 多卡和多实例怎样传输状态
-> 最终怎样在 SLO 下提高 Goodput
```

只要围绕这条主线回答，即使源码类名变化，也不会把面试题答成孤立参数列表。
