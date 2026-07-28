---
title: 'Chunked Prefill：拆分长 Prompt，降低在线推理长尾延迟'
description: '讲解长 Prefill 为什么阻塞 Decode、Chunked Prefill 的调度和注意力语义、token budget、块大小选择，以及 vLLM 中的实现方式。'
category: '推理优化'
pubDate: '2026-07-28T12:38:00+08:00'
updatedDate: '2026-07-28T12:38:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [Prefill 为什么会阻塞 Decode](#一prefill-为什么会阻塞-decode)
2. [Chunked Prefill 的核心思想](#二chunked-prefill-的核心思想)
3. [切块后计算是否等价](#三切块后计算是否等价)
4. [调度过程与具体例子](#四调度过程与具体例子)
5. [简化实现](#五简化实现)
6. [开源框架中的实现](#六开源框架中的实现)
7. [Chunk 大小如何影响性能](#七chunk-大小如何影响性能)
8. [与其他技术的关系](#八与其他技术的关系)
9. [常见错误](#九常见错误)
10. [评估与调参](#十评估与调参)
11. [总结](#十一总结)

## 一、Prefill 为什么会阻塞 Decode

大模型推理包含两个计算形态明显不同的阶段：

| 阶段 | 单请求每轮 token 数 | 主要特征 |
| --- | ---: | --- |
| Prefill | 整个 Prompt | 大矩阵计算，容易 Compute-bound |
| Decode | 1 | 读取历史 KV，容易 Memory-bound |

假设线上已有 32 个 Decode 请求，每轮各生成 1 个 token。此时到达一个 32K token 的长 Prompt。

如果整个 Prompt 一次执行：

```text
step 0: 32 个 Decode token
step 1: 32768 个 Prefill token
step 2: 32 个 Decode token
```

`step 1` 可能持续很久。虽然 Prefill 吞吐很高，所有 Decode 请求仍必须等它结束，TPOT 和 P99 延迟会突然升高。这类现象称为 Head-of-Line Blocking。

## 二、Chunked Prefill 的核心思想

Chunked Prefill 不要求一次完成整个 Prompt，而是把它拆成多个 token 区间：

```text
Prompt: token [0, 32768)

chunk 0: [0, 4096)
chunk 1: [4096, 8192)
...
chunk 7: [28672, 32768)
```

调度器可以把 Decode token 和一个 Prefill chunk 放进同一轮：

```text
step 0: Decode 32 + Prefill 4064 = 4096 token
step 1: Decode 32 + Prefill 4064 = 4096 token
...
```

这样长 Prompt 的总 Prefill 工作量没有减少，但它不再独占一个超长 step。

核心目标是：

> 用可控的 Prefill chunk 限制单轮执行时间，在吞吐、TTFT 和 TPOT 之间取得平衡。

## 三、切块后计算是否等价

对因果 Transformer，第 `i` 个 token 只依赖 `[0, i]` 范围。处理后续 chunk 时，只要前面 token 的 K/V 已缓存，就可以得到与一次性 Prefill 相同的注意力结果。

设 Prompt 分成两段：

```text
X = [X_a, X_b]
```

第一段计算：

```text
K_a = X_a W_K
V_a = X_a W_V
O_a = Attention(Q_a, K_a, V_a)
```

第二段计算时，新 Query 需要同时关注历史缓存和当前块：

```text
K = concat(K_a, K_b)
V = concat(V_a, V_b)
O_b = Attention(Q_b, K, V)
```

其中 `Q_b` 的形状只覆盖当前 chunk，而 K/V 覆盖完整前缀：

```text
Q_b: [chunk_size, num_heads, head_dim]
K/V: [prefix_len + chunk_size, num_kv_heads, head_dim]
```

因此 Chunked Prefill 不是把每块当作独立序列。若第二块看不到第一块的 KV，计算结果就会错误。

### 3.1 位置编码必须连续

第二块的位置不能从 0 重新开始：

```python
positions = torch.arange(
    num_computed_tokens,
    num_computed_tokens + chunk_size,
)
```

RoPE、ALiBi 或相对位置编码都依赖正确的逻辑位置。

### 3.2 Causal Mask 仍然存在

当前块内部的 token 仍不能看到未来 token：

```text
历史 KV：全部可见
当前 chunk：只可见当前位置及之前
```

## 四、调度过程与具体例子

设：

```text
max_num_batched_tokens = 2048
当前有 64 个 Decode 请求
新 Prompt 长度 = 6000
```

每轮先给 Decode 分配 64 token，Prefill 获得：

```text
prefill_budget = 2048 - 64 = 1984
```

调度结果：

| Step | Decode token | Prefill 区间 | Prefill 剩余 |
| ---: | ---: | --- | ---: |
| 0 | 64 | `[0, 1984)` | 4016 |
| 1 | 64 | `[1984, 3968)` | 2032 |
| 2 | 64 | `[3968, 5952)` | 48 |
| 3 | 64 | `[5952, 6000)` | 0 |

此请求经过 4 轮才进入 Decode。其 TTFT 可能比一次性 Prefill 更高，但其他在线请求不会被一个 6000-token step 长时间阻塞。

这体现了典型权衡：

```text
较小 chunk：
  更稳定的 Decode TPOT
  更多调度与 kernel 开销
  长 Prompt 的 TTFT 可能上升

较大 chunk：
  Prefill 吞吐更高
  单轮执行时间更长
  Decode 长尾更明显
```

## 五、简化实现

### 5.1 调度端

```python
from dataclasses import dataclass


@dataclass
class RequestState:
    prompt_tokens: list[int]
    num_computed_tokens: int = 0

    @property
    def remaining_prefill(self) -> int:
        return len(self.prompt_tokens) - self.num_computed_tokens


def schedule_prefill_chunk(
    request: RequestState,
    token_budget: int,
    long_prefill_limit: int,
) -> tuple[int, int]:
    start = request.num_computed_tokens

    # 不能超过请求剩余 token、全局预算和单请求长 Prefill 上限。
    num_tokens = min(
        request.remaining_prefill,
        token_budget,
        long_prefill_limit,
    )
    end = start + num_tokens
    return start, end
```

### 5.2 模型执行端

```python
def run_prefill_chunk(model, request, start, end, kv_cache):
    input_ids = request.prompt_tokens[start:end]
    positions = list(range(start, end))

    # attention 会读取 [0, start) 的历史 KV，
    # 并把当前 [start, end) 的 K/V 写入新 slot。
    model(
        input_ids=input_ids,
        positions=positions,
        kv_cache=kv_cache,
        context_length=end,
    )

    request.num_computed_tokens = end
```

真实实现会把多个请求的 chunk 扁平化到同一个张量，并通过 `query_start_loc`、`seq_lens`、`slot_mapping` 等元数据描述边界。

## 六、开源框架中的实现

vLLM 的调度配置包含以下几类约束：

```python
max_num_batched_tokens       # 单轮 token 总预算
max_num_partial_prefills     # 可同时部分 Prefill 的请求数
max_long_partial_prefills    # 可同时执行的长 Prefill 数
long_prefill_token_threshold # 单个长 Prefill 每轮的上限
enable_chunked_prefill       # 是否允许切块
```

调度器计算一个请求本轮的工作量时，逻辑可以概括为：

```python
num_new_tokens = request.num_tokens - request.num_computed_tokens

if 0 < long_prefill_threshold < num_new_tokens:
    num_new_tokens = long_prefill_threshold

if not enable_chunked_prefill and num_new_tokens > token_budget:
    # 禁止切块时，必须等待预算足以容纳整个 Prefill。
    wait()

num_new_tokens = min(num_new_tokens, token_budget)
```

随后调用 KV Cache 管理器：

```python
new_blocks = kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
)
```

这里同时保证：

- 计算预算足够。
- 当前 chunk 的 KV Cache 有可用 slot。
- `num_computed_tokens` 与分配的 KV block 保持一致。

### 6.1 为什么需要 `num_computed_tokens`

请求的完整 token 数和已执行 token 数必须分开：

```text
num_tokens：当前请求已经拥有的输入/输出 token 总数
num_computed_tokens：模型已经为多少 token 完成前向并写入 KV
```

Chunked Prefill 的待执行量是：

```text
num_new_tokens = num_tokens - num_computed_tokens
```

这个统一表达也能覆盖抢占恢复、远端 KV 加载和投机 token 验证。

### 6.2 为什么允许短请求越过长请求

如果所有部分 Prefill 槽位都被 100K Prompt 占据，短 Prompt 会长期排队。限制 `max_long_partial_prefills`，可以给短请求保留部分并发槽位，改善 TTFT 公平性。

## 七、Chunk 大小如何影响性能

### 7.1 Kernel 效率

Prefill 本质上包含较大的 GEMM。chunk 太小时：

- GEMM 的 M 维变小。
- Tensor Core 利用率下降。
- Kernel Launch 和元数据开销占比上升。
- 相同历史 KV 可能被更多次读取。

### 7.2 Decode 干扰

chunk 太大时，混合 batch 的执行时间仍主要由 Prefill 决定，Decode TPOT 会产生明显抖动。

### 7.3 一个简单成本模型

设：

```text
T_step(C, B_d) = T_prefill(C) + T_decode(B_d) + T_overhead
```

其中：

- `C` 是本轮 Prefill token 数。
- `B_d` 是 Decode 请求数。
- `T_overhead` 是调度、输入准备和 kernel launch 开销。

目标不是单纯最小化 `T_prefill(C)`，而是在 SLA 下最大化吞吐：

```text
maximize tokens_per_second
subject to P99_TPOT <= target
           P99_TTFT <= target
```

因此最佳 chunk size 取决于模型、GPU、并发和长度分布，不是固定常数。

## 八、与其他技术的关系

### 8.1 Continuous Batching

Continuous Batching 决定每轮有哪些请求；Chunked Prefill 让一个长 Prefill 可以只占用部分 token budget。二者通常一起使用。

### 8.2 PagedAttention

每个 chunk 会逐步扩展请求的 KV Cache。分页式 block 分配使缓存不需要预留整段连续空间。

### 8.3 Prefix Caching

若前 4096 token 已命中 Prefix Cache，调度器只需从未命中的位置继续切块：

```text
Prompt 长度：10000
前缀命中：4096
实际 Prefill：[4096, 10000)
```

### 8.4 CUDA Graph

动态 chunk 会产生不同 token 数。常见处理方式是：

- 为若干 token bucket 捕获图。
- Padding 到最近 bucket。
- 大 Prefill 使用 Eager，小而稳定的 Decode 使用 CUDA Graph。

### 8.5 Prefill/Decode 分离

若 Prefill 和 Decode 在不同 GPU 上，Chunked Prefill 不再直接阻塞 Decode GPU，但仍可用于：

- 限制 Prefill 节点的长尾。
- 提高多个 Prompt 之间的公平性。
- 分段传输 KV Cache。

## 九、常见错误

### 9.1 每块独立计算

错误做法是每个 chunk 只关注自身 K/V。正确做法必须读取所有历史 KV。

### 9.2 位置从零开始

这会让 RoPE 位置重复，模型输出与完整 Prefill 不一致。

### 9.3 只限制 chunk，不限制总预算

多个请求各自都满足 chunk 上限，但总 token 数仍可能超过单轮预算：

```text
8 个请求 * 2048 token = 16384 token
```

必须同时维护全局 `token_budget`。

### 9.4 只看平均 TPOT

长 Prefill 通常影响 P95/P99，而平均值可能变化不大。需要观察 step latency 分布和最大 batch token 数。

### 9.5 chunk 越小越好

过小 chunk 会损失 Prefill 吞吐，还可能增加长 Prompt 的 TTFT。目标是满足 Decode SLA 后选择尽可能高效的 chunk。

## 十、评估与调参

建议分别构造：

```text
短 Prompt + 短输出
长 Prompt + 短输出
短 Prompt + 长输出
长短 Prompt 混合，并持续有 Decode 请求
```

至少记录：

| 指标 | 作用 |
| --- | --- |
| Prefill tokens/s | 判断 Prefill 效率损失 |
| Decode tokens/s | 判断并发生成吞吐 |
| TTFT P50/P99 | 判断长 Prompt 和排队影响 |
| TPOT P50/P99 | 判断 Prefill 对 Decode 的干扰 |
| Step latency | 直接观察每轮是否出现长尖峰 |
| Queue time | 判断请求是否因调度策略饥饿 |
| KV 使用率 | 判断缓存是否成为约束 |

调参顺序可以是：

1. 确定目标 TPOT 和 TTFT。
2. 设置可容纳典型并发的 KV Cache。
3. 从中等 `max_num_batched_tokens` 开始。
4. 逐步增大预算，直到吞吐不再明显提高或 P99 超限。
5. 调整长 Prefill 上限和并发部分 Prefill 数。

## 十一、总结

Chunked Prefill 不减少 Prompt 的理论计算量，而是改变工作在时间轴上的排布：

```text
一次超长 Prefill
-> 多个有上限的 Prefill chunk
-> 与 Decode token 交错执行
```

正确实现必须保持历史 KV、位置编码和 Causal 语义连续。它牺牲少量调度复杂度和部分 Prefill 峰值效率，换取更稳定的在线 TPOT、更低的 Head-of-Line Blocking 和更可控的长尾延迟。
