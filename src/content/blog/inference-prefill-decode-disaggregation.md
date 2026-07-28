---
title: 'Prefill/Decode 分离：异构部署与 KV Cache 传输'
description: '讲解 Prefill 和 Decode 的资源特征、P/D 分离请求链路、KV Cache 传输量、路由和负载均衡、故障处理，并结合 vLLM KVConnector 与 TensorRT-LLM 分离式服务说明实现。'
category: '推理优化'
pubDate: '2026-07-28T12:45:00+08:00'
updatedDate: '2026-07-28T12:45:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [为什么要分离 Prefill 和 Decode](#一为什么要分离-prefill-和-decode)
2. [完整请求链路](#二完整请求链路)
3. [KV Cache 传输量](#三kv-cache-传输量)
4. [两种调度顺序](#四两种调度顺序)
5. [资源池如何配比](#五资源池如何配比)
6. [最小服务流程](#六最小服务流程)
7. [KV Transfer 接口](#七kv-transfer-接口)
8. [开源框架中的实现](#八开源框架中的实现)
9. [并行配置和 Layout 映射](#九并行配置和-layout-映射)
10. [故障、一致性与背压](#十故障一致性与背压)
11. [何时有效及如何评估](#十一何时有效及如何评估)
12. [总结](#十二总结)

## 一、为什么要分离 Prefill 和 Decode

Prefill 和 Decode 使用同一个 Transformer，却有不同的工作负载：

| 特征 | Prefill | Decode |
| --- | --- | --- |
| 单请求 token 数 | Prompt 全部或 Chunk | 每轮 1 个 |
| GEMM Shape | M 较大 | M 很小 |
| 主要瓶颈 | 更容易 Compute-bound | 更容易 Memory-bound |
| 关键指标 | TTFT、Prefill tokens/s | TPOT、Decode tokens/s |
| 执行时间 | 随输入长度显著变化 | 单步短、重复次数多 |

混合部署时，长 Prefill 会拉长同一 GPU 上 Decode step 的时间，导致 TPOT 抖动。为了兼顾两者，单一调度器往往需要牺牲 Prefill 吞吐或 Decode 延迟。

P/D Disaggregation 把两阶段放到不同资源池：

```text
Prefill Pool：负责 Prompt 前向并生成 KV Cache
Decode Pool：接收 KV Cache，逐 token 生成
```

这样可以：

- 分别优化 Prefill 大矩阵和 Decode 小矩阵 Kernel。
- 独立扩缩容两个资源池。
- 避免长 Prompt 直接阻塞 Decode。
- 为两阶段选择不同 GPU、并行度和批处理策略。

代价是需要传输完整 Prompt KV Cache。

## 二、完整请求链路

典型流程：

```text
Client
  |
  v
Global Router
  |
  +--> 选择 Prefill Worker
          |
          +-- Tokenize / Prefix Lookup
          +-- 执行 Prefill
          +-- 生成首 token 或首个 logits
          +-- 保存/发送 KV Cache
  |
  +--> 选择 Decode Worker
          |
          +-- 分配本地 KV Block
          +-- 接收 KV Cache
          +-- 恢复请求状态
          +-- Continuous Batching Decode
          +-- Streaming 返回
```

请求状态至少要携带：

```text
request_id
token_ids
已计算 token 数
KV block 描述
首 token 或首个 logits
采样参数与随机状态
停止条件
Prefix Cache 元数据
```

若 Prefill 端已经采样首 token，Decode 端必须从完全一致的位置继续，避免重复或丢失 token。

## 三、KV Cache 传输量

Prompt 长度为 `S`，传输量：

```text
KV_transfer_bytes =
    S
  * num_layers
  * 2
  * num_kv_heads
  * head_dim
  * dtype_bytes
```

使用一个 80 层、8 KV Head、Head Dim 128 的 GQA 模型：

```text
BF16 每 token KV = 320 KiB
```

8K Prompt：

```text
320 KiB * 8192 = 2.5 GiB
```

即使链路有效带宽为 100 GB/s，纯传输下界也约：

```text
2.5 GiB / 100 GB/s ≈ 25 ms
```

还要加协议、内存注册、Block Mapping 和同步开销。

### 3.1 为什么网络是第一约束

P/D 分离节省的是计算干扰，却新增了一次与 Prompt 长度成正比的数据移动。若：

```text
T_transfer >= 被消除的排队和干扰时间
```

分离后 TTFT 可能更差。

### 3.2 减少传输字节的方法

- GQA/MQA/MLA 减少 KV Head 或缓存维度。
- FP8/INT8 KV Cache。
- Prefix Cache 命中后只传未共享的后缀。
- P/D 节点共享远端 KV Store。
- 使用 GPUDirect RDMA，避免 CPU Bounce Buffer。
- 分 Layer 流水传输，隐藏部分通信。

## 四、两种调度顺序

### 4.1 Context-first

先选择 Prefill Worker，Prefill 完成后再选择 Decode Worker：

```text
route P -> run P -> route D -> transfer -> run D
```

优点：

- Prefill 路由简单。
- Decode 只在真正需要时占位。

风险：

- Prefill 完成时 Decode Pool 可能没有容量。
- KV 已生成却等待传输，增加缓存占用。

### 4.2 Generation-first

先为请求预留 Decode Worker，再选择合适 Prefill Worker：

```text
reserve D -> route P -> run P -> transfer to reserved D -> run D
```

优点：

- KV 目的地提前确定。
- Decode 容量和会话粘性更可控。

风险：

- Decode 槽位在 Prefill 期间被预留但未计算。
- Prefill 很长或失败时造成资源浪费。

### 4.3 条件分离

不是所有请求都必须 P/D 分离。路由器可以根据：

```text
Prompt 长度
预计输出长度
当前 P/D 队列
KV 传输成本
Prefix Cache 命中位置
SLA
```

选择：

```text
短 Prompt：Decode 节点本地 Prefill + Decode
长 Prompt：独立 Prefill 后传输
```

## 五、资源池如何配比

设请求到达率为 `lambda`。

Prefill Pool 每秒工作量近似：

```text
W_p = lambda * E[input_tokens]
```

Decode Pool 每秒工作量：

```text
W_d = lambda * E[output_tokens]
```

粗略卡数：

```text
N_p >= W_p / prefill_capacity_per_gpu
N_d >= W_d / decode_capacity_per_gpu
```

但实际还要考虑：

- 长度分布而非只有平均值。
- P95/P99 SLA。
- Batch 效率随并发变化。
- KV Cache 容量约束。
- 故障和扩缩容冗余。
- 传输链路容量。

### 5.1 为什么输出长度很重要

Prefill 只执行一次，Decode 要执行 `output_length` 轮。输出越长，Decode Worker 占用时间越长，通常更容易形成会话驻留和容量瓶颈。

### 5.2 Little's Law

若 Decode 请求平均驻留时间为 `T_d`：

```text
average_decode_concurrency = lambda * T_d
```

这可用于估算 Decode KV Cache 和活跃请求槽位，而不能只根据 QPS 配置副本。

## 六、最小服务流程

下面展示控制面，不包含真实 GPU KV 传输。

```python
from dataclasses import dataclass


@dataclass
class PrefillResult:
    request_id: str
    first_token: int
    num_computed_tokens: int
    kv_descriptor: dict


async def handle_request(request, prefill_pool, decode_pool, kv_transport):
    # 1. 提前选择 Decode，确保目标端有 KV 容量。
    decode_worker = await decode_pool.reserve(
        estimated_tokens=request.max_context_length,
    )

    try:
        # 2. 选择 Prefill Worker 并完成 Prompt 计算。
        prefill_worker = await prefill_pool.select(request)
        result: PrefillResult = await prefill_worker.prefill(request)

        # 3. 把 KV 传到已经预留的 Decode Worker。
        await kv_transport.transfer(
            src=prefill_worker,
            dst=decode_worker,
            descriptor=result.kv_descriptor,
        )

        # 4. 从 Prefill 已提交的位置继续生成。
        async for token in decode_worker.decode(
            request=request,
            first_token=result.first_token,
            num_computed_tokens=result.num_computed_tokens,
        ):
            yield token
    finally:
        await decode_pool.release(decode_worker, request.request_id)
```

工程实现需要把“预留成功”和“KV 已加载成功”设计为明确的状态，而不是假设异步调用一定完成。

## 七、KV Transfer 接口

一个通用 KV Connector 需要同时服务 Scheduler 和 Worker。

### 7.1 Scheduler 侧

Scheduler 负责：

```text
查询远端命中的 token 数
为目标 KV 分配本地 Block
把请求标记为等待远端 KV
接收传输完成或失败通知
决定失败后重算还是终止
```

状态可表示为：

```text
WAITING
-> WAITING_FOR_REMOTE_KV
-> RUNNING
```

### 7.2 Worker 侧

Worker 需要：

```text
register_kv_caches()
start_load_kv()
wait_for_layer_load(layer)
save_kv_layer(layer, kv)
wait_for_save()
```

这组接口允许 Layer-wise Pipeline：

```text
Prefill 计算 Layer i
-> 发送 Layer i KV
-> 同时计算 Layer i+1
```

Decode 端也可在 Layer i KV 就绪后立即执行该层，而不等待所有 Layer 完整传输。

### 7.3 为什么要在覆盖前等待 Save

Paged KV Block 会被复用。若异步发送尚未完成，Scheduler 就把源 Block 分配给新请求，传输内容会被覆盖。Connector 必须在 Block 可复用前确认 Save 完成，或持有引用保护。

## 八、开源框架中的实现

### 8.1 vLLM KVConnector

vLLM 使用 `KVTransferConfig` 描述：

```text
kv_connector
engine_id
kv_buffer_device
kv_buffer_size
kv_role
kv_load_failure_policy
connector-specific config
```

角色分为：

```text
kv_producer：Prefill 生产 KV
kv_consumer：Decode 消费 KV
kv_both：    同时支持两种方向
```

调度器发现外部 KV 命中后：

1. 计算本地与远端命中 token 数。
2. 为远端 KV 预留本地 Block。
3. 请求进入 `WAITING_FOR_REMOTE_KVS`。
4. Worker 异步加载 KV。
5. 完成后请求重新进入可运行队列。

加载失败可选择：

```text
fail：立即失败，避免隐藏数据问题
recompute：把失败 Block 当作 miss，重新 Prefill
```

### 8.2 TensorRT-LLM 分离式请求

TensorRT-LLM 使用不同 Request Type 表示：

```text
context_only
generation_only
context_and_generation
```

Context 阶段结果会携带：

```text
first_gen_tokens
context request id
opaque state
draft tokens
首 token logits/logprobs
usage metadata
```

服务层支持 Context-first 和 Generation-first 两种路由。Context Response 被转换成 Generation Request 后，Prompt Token、首 token 和上下文状态继续传递到 Decode 服务。

这说明 P/D 分离不仅是 KV Tensor Copy，还涉及请求协议和采样状态交接。

## 九、并行配置和 Layout 映射

### 9.1 P/D 使用相同 TP

最简单情况：

```text
Prefill TP = 4
Decode TP = 4
Rank i -> Rank i
```

每个 Rank 发送自己持有的 KV Head 分片。

### 9.2 P/D 使用不同 TP

若：

```text
Prefill TP = 8
Decode TP = 4
```

需要重新映射或聚合分片。对 GQA 还要处理 KV Head 复制关系。传输层必须知道：

```text
Layer
KV Head 范围
Block ID
Token Offset
Cache Dtype
Physical Layout
```

不能把源 Rank 的原始字节直接按相同 Rank 号复制。

### 9.3 HND 与 NHD

不同 Attention Backend 可能使用：

```text
HND: [head, token, dim]
NHD: [token, head, dim]
```

若 P/D Layout 不同，传输中要 Permute。额外 Kernel 和内存流量可能明显影响收益。

### 9.4 模型配置必须一致

至少保证：

- 模型权重版本。
- Tokenizer 和位置编码。
- KV Cache Dtype/Scale。
- Attention Head 配置。
- Layer 数。
- Adapter/LoRA。
- Cache Block 语义。

## 十、故障、一致性与背压

### 10.1 Prefill 成功、传输失败

可选策略：

- 在另一个 Prefill Worker 重算。
- 保留源 KV 并重试传输。
- 让 Decode Worker 本地重算 Prompt。
- 请求失败。

应设置最大重试次数，避免大 Prompt 无限重算。

### 10.2 Decode Worker 在传输后失效

若 KV 只存在该 Decode Worker，需要重新传输或重算。远端共享 KV Store 能提高恢复能力，但增加写入成本。

### 10.3 幂等 Request ID

重试可能让同一请求到达多个 Worker。全局 `request_id` 和阶段标识应保证：

- 重复 Prefill 不会产生两个同时生效的 Decode。
- 旧传输完成事件不会唤醒新请求实例。
- 取消操作可传播到 P、D 和 KV Store。

### 10.4 背压

当 Decode Pool 满载时，继续高速 Prefill 只会产生大量待传 KV。Router 应根据：

```text
Decode 可用槽位
KV Cache 剩余容量
传输队列长度
网络带宽
预计输出长度
```

限制 Prefill 接纳速率。

## 十一、何时有效及如何评估

### 11.1 更适合

- Prompt 较长、输出也较长。
- 混部时 Decode P99 被 Prefill 明显干扰。
- 高速 GPU/网络互联。
- P/D 比例随时间变化，需要独立扩缩容。
- 两阶段适合不同硬件或不同并行度。

### 11.2 可能不适合

- Prompt 很短，分离固定开销占比高。
- KV 很大而网络较慢。
- 单机已有足够 GPU，Chunked Prefill 能满足 SLA。
- 请求量低，两个资源池都难形成有效 batch。
- Decode 很短，传输完成后很快结束。

### 11.3 必须比较的指标

```text
Prefill queue time
Prefill compute time
KV transfer wait time
Decode queue time
TTFT
TPOT
端到端 latency
P/D GPU utilization
KV transfer bytes/s
传输失败和重算率
每个资源池的 active/waiting requests
```

建议同时测三组：

```text
混合部署
混合部署 + Chunked Prefill
P/D 分离
```

否则无法判断收益来自资源分离，还是只来自更好的调度参数。

## 十二、总结

P/D 分离把两个执行特征不同的阶段拆成独立资源池：

```text
Prefill Worker 计算 Prompt
-> 生成并传输 KV Cache
-> Decode Worker 恢复请求状态
-> Continuous Batching 逐 token 生成
```

它能隔离 Prefill 对 Decode 的干扰，并允许独立扩缩容和异构配置；同时新增与 Prompt 长度线性相关的 KV 传输。高质量实现必须处理 KV Layout、TP 映射、异步 Layer 传输、请求状态交接、失败重算和 Decode 背压。是否采用 P/D 分离，应由端到端 TTFT/TPOT 和链路成本决定，而不是只比较 GPU Kernel 吞吐。
