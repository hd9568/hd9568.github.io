---
title: 'vLLM 请求调度源码解析：从 Waiting Queue 到模型执行'
description: '沿 vLLM V1 的真实调用链，讲解 Request 状态、统一 Token 调度、Running/Waiting 队列、Prefix Cache、KV Block 分配、抢占、SchedulerOutput 和输出回写。'
category: '推理优化'
pubDate: '2026-07-28T14:02:00+08:00'
updatedDate: '2026-07-28T14:02:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [先看完整调用链](#一先看完整调用链)
2. [调度器的核心统一抽象](#二调度器的核心统一抽象)
3. [Request 保存哪些状态](#三request-保存哪些状态)
4. [三个请求容器](#四三个请求容器)
5. [请求如何进入调度器](#五请求如何进入调度器)
6. [一次 schedule 的完整过程](#六一次-schedule-的完整过程)
7. [Running 请求如何调度](#七running-请求如何调度)
8. [Waiting 请求如何调度](#八waiting-请求如何调度)
9. [KV Cache 分配为什么决定能否运行](#九kv-cache-分配为什么决定能否运行)
10. [显存不足时如何抢占](#十显存不足时如何抢占)
11. [SchedulerOutput 如何交给 Worker](#十一scheduleroutput-如何交给-worker)
12. [模型输出如何更新请求](#十二模型输出如何更新请求)
13. [一个完整数值例子](#十三一个完整数值例子)
14. [简化版调度器](#十四简化版调度器)
15. [关键配置与观测方法](#十五关键配置与观测方法)
16. [总结](#十六总结)

## 一、先看完整调用链

本文分析 vLLM V1 Scheduler。核心源码分布如下：

| 模块 | 作用 |
| --- | --- |
| `vllm/v1/engine/core.py` | 驱动一次调度、执行和状态更新 |
| `vllm/v1/core/sched/scheduler.py` | 请求选择、预算分配、抢占和回收 |
| `vllm/v1/request.py` | 单个请求的状态和 token 信息 |
| `vllm/v1/core/sched/request_queue.py` | FCFS/Priority 等待队列 |
| `vllm/v1/core/kv_cache_manager.py` | Prefix Cache 查询和 KV Block 分配 |
| `vllm/v1/core/sched/output.py` | Scheduler 到 Model Runner 的数据协议 |

同步执行模式下，`EngineCore.step()` 的主线非常清晰：

```python
def step(self):
    if not self.scheduler.has_requests():
        return {}, False

    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(
        scheduler_output,
        non_block=True,
    )

    model_output = future.result()
    engine_outputs = self.scheduler.update_from_output(
        scheduler_output,
        model_output,
    )
    return engine_outputs, True
```

因此可以把调度系统拆成三步：

```text
schedule
  决定本轮执行哪些请求、每个请求执行多少 token

execute_model
  Worker 准备输入、执行模型并采样

update_from_output
  写回 token、修正状态、判断停止、释放资源
```

## 二、调度器的核心统一抽象

理解 vLLM V1 Scheduler 最重要的一点是：

> Scheduler 不再把请求硬分成 Prefill 队列和 Decode 队列，而是统一调度“尚未计算的 token”。

`Request` 中有两个关键量：

```text
num_tokens_with_spec
  = Prompt token
  + 已生成 token
  + Speculative token

num_computed_tokens
  = 已经完成模型前向、拥有有效计算状态的 token 数
```

本轮理论上需要计算：

```text
num_new_tokens =
    num_tokens_with_spec
  + num_output_placeholders
  - num_computed_tokens
```

暂时忽略异步调度的 `num_output_placeholders`，核心就是：

```text
待计算 token 数 = 当前已知 token 数 - 已计算 token 数
```

这个差值统一覆盖多种场景。

### 2.1 Prefill

Prompt 长度为 100，尚未执行：

```text
num_tokens = 100
num_computed_tokens = 0
num_new_tokens = 100
```

若本轮预算只有 32，就执行前 32 个 token，形成 Chunked Prefill。

### 2.2 Decode

Prompt 和已有输出共 101 个 token，其中前 100 个已计算：

```text
num_tokens = 101
num_computed_tokens = 100
num_new_tokens = 1
```

这就是普通单 token Decode。

### 2.3 Speculative Decoding

在 101 个确定 token 后又提出 4 个 Draft token：

```text
num_tokens_with_spec = 105
num_computed_tokens = 100
num_new_tokens = 5
```

同一套 Scheduler 可以一次安排多个候选位置的验证。

因此 Prefill、Decode、Chunked Prefill 和 Speculative Decoding 的区别，不是不同调度状态机，而是 `num_new_tokens` 的大小和来源不同。

## 三、Request 保存哪些状态

`Request` 是调度器的核心状态对象。最重要的字段可以分为五组。

### 3.1 身份和排序

```python
request_id
client_index
priority
arrival_time
```

Priority 模式下，请求按以下顺序比较：

```text
priority 越小越优先
-> arrival_time 越早越优先
-> request_id 用于稳定排序
```

### 3.2 Token 状态

```python
prompt_token_ids
_output_token_ids
_all_token_ids
spec_token_ids
num_computed_tokens
```

关系如下：

```text
all_token_ids = prompt_token_ids + output_token_ids
num_tokens = len(all_token_ids)
num_tokens_with_spec = num_tokens + len(spec_token_ids)
```

### 3.3 停止条件

```python
sampling_params
max_tokens
stop_reason
```

输出写回后会依次检查：

- EOS。
- Stop Token。
- 最大模型长度。
- 最大输出长度。
- 可选的重复模式检测。

### 3.4 缓存信息

```python
block_hashes
cache_salt
num_preemptions
skip_reading_prefix_cache
```

`block_hashes` 随完整 Token Block 增长，用于 Prefix Cache 查询。

### 3.5 状态

主要状态：

```text
WAITING
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
WAITING_FOR_REMOTE_KVS
WAITING_FOR_STREAMING_REQ
RUNNING
PREEMPTED
FINISHED_*
```

基本状态流：

```text
WAITING
  |
  | 预算和 KV 空间充足
  v
RUNNING
  |       \
  | 完成   \ KV 空间不足
  v         v
FINISHED   PREEMPTED
              |
              +----> WAITING ----> RUNNING
```

## 四、三个请求容器

`Scheduler` 同时维护：

```python
self.requests: dict[str, Request]
self.waiting: RequestQueue
self.skipped_waiting: RequestQueue
self.running: list[Request]
```

### 4.1 `requests`

保存所有未完全清理的请求，用于按 ID 查找。它是请求生命周期的主索引，不代表请求一定可调度。

### 4.2 `waiting`

保存当前具备调度条件、但尚未进入 Running 的请求。

支持两种策略：

```text
FCFS：
  使用 deque
  从队尾加入，从队首取出

Priority：
  使用 heap
  按 priority、arrival_time 排序
```

### 4.3 `skipped_waiting`

保存暂时被外部条件阻塞的请求，例如：

- 等待 Structured Output Grammar。
- 等待远端 KV Cache。
- 等待流式会话的新输入。
- 本轮因 LoRA 数量限制被跳过。

单独维护这个队列的意义是：阻塞请求不会永久挡住后续可执行请求。

FCFS 模式会优先重新检查 `skipped_waiting`；Priority 模式则比较两个队首，继续遵守全局优先级。

### 4.4 `running`

保存已经进入执行生命周期、持有 KV Cache 的请求。

注意：

```text
在 running 中
不等于
本轮一定被 scheduled
```

Token Budget、Encoder Budget 或异步流水状态可能让某个 Running 请求在当前 step 暂停一次。

## 五、请求如何进入调度器

前端完成 Tokenize 和参数解析后，`EngineCore.add_request()` 最终调用：

```python
self.scheduler.add_request(request)
```

对普通新请求，`Scheduler.add_request()` 执行：

```python
self._enqueue_waiting_request(request)
self.requests[request.request_id] = request
```

其中 `_enqueue_waiting_request()` 会根据状态选择：

```python
if request is blocked:
    skipped_waiting.add_request(request)
else:
    waiting.add_request(request)
```

请求此时还没有占用模型执行槽位。只有 `schedule()` 成功分配 KV Cache 后，才会进入 `running`。

### 5.1 为什么重复 Request ID 不一定报错

vLLM 还支持可恢复的流式输入会话。相同 Request ID 可能表示向已有会话追加下一段输入。

普通请求不应依赖这个行为；请求 ID 仍应在有效生命周期内保持唯一。

## 六、一次 schedule 的完整过程

`Scheduler.schedule()` 的顺序可以概括为：

```text
1. 初始化 Token Budget
2. 调度已有 Running 请求
3. 若本轮没有发生抢占，再接纳 Waiting 请求
4. 校验预算和请求数约束
5. 计算公共 Prefix Block
6. 构造 SchedulerOutput
7. 更新 num_computed_tokens
8. 返回给 Model Executor
```

本轮有三个主要约束：

```text
Token 约束：
sum(num_scheduled_tokens) <= max_num_scheduled_tokens

请求数约束：
len(running) <= max_num_seqs

KV 约束：
required_blocks <= free_or_evictable_blocks
```

此外还有：

- Encoder Compute Budget。
- Encoder Cache。
- 同时激活的 LoRA 数量。
- 最大模型长度。
- KV Connector 是否完成。

### 6.1 Token Budget

初始值：

```python
token_budget = self.max_num_scheduled_tokens
```

每调度一个请求：

```python
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
```

它限制的是本轮总 token 数，不是请求数。

## 七、Running 请求如何调度

Scheduler 先处理 `running`：

```python
for request in running:
    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )

    num_new_tokens = min(num_new_tokens, token_budget)
```

先处理 Running 有两个原因：

- 已运行请求持有 KV Cache，继续 Decode 可维持稳定 TPOT。
- 先让老请求推进，可以更快完成并释放缓存。

### 7.1 限制长 Prefill

如果配置了长 Prefill 阈值：

```python
num_new_tokens = min(
    num_new_tokens,
    long_prefill_token_threshold,
)
```

长 Prompt 会被拆成多个 step，不会一次占满全部预算。

### 7.2 限制模型长度

投机 token 可能让请求越过最大上下文，因此还要限制：

```python
num_new_tokens = min(
    num_new_tokens,
    max_model_len - 1 - request.num_computed_tokens,
)
```

### 7.3 申请 KV Slot

确定 token 数后：

```python
new_blocks = kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=num_lookahead_tokens,
)
```

返回值有两种：

```text
KVCacheBlocks：分配成功，可以进入本轮
None：        KV 空间不足，需要抢占或停止调度
```

### 7.4 为什么 `num_new_tokens == 0` 时是 `continue`

某个 Running 请求可能因 Encoder Budget 等原因本轮不能执行。Scheduler 会继续检查后面的请求，而不是直接 `break`。

这意味着实际执行不一定严格保持 FCFS，但能避免一个暂时阻塞的请求让 GPU 空转。

## 八、Waiting 请求如何调度

只有本轮没有发生抢占时，Scheduler 才继续接纳 Waiting 请求：

```python
if not preempted_reqs:
    schedule_waiting_requests()
```

这样可以避免刚被抢占的请求在同一个 step 又被接纳，造成反复释放和分配。

### 8.1 先处理阻塞状态

Scheduler 从 `waiting` 或 `skipped_waiting` 选择队首。如果请求仍在等待 Grammar、远端 KV 或流式输入，就暂存到当前 step 的跳过队列，继续检查其他请求。

### 8.2 检查 LoRA 限制

若本轮已经达到 `max_loras`，且新请求使用另一个 LoRA：

```text
不拒绝请求
-> 本轮跳过
-> 放回 skipped_waiting
```

### 8.3 查询本地 Prefix Cache

对第一次调度的请求：

```python
cached_blocks, num_cached_tokens = (
    kv_cache_manager.get_computed_blocks(request)
)
```

Prefix Cache 只能返回完整、连续的前缀 Block。

即使整个 Prompt 命中，也要重新计算最后一个 token，才能获得用于采样下一个 token 的 Logits：

```text
max_cache_hit_length = prompt_length - 1
```

### 8.4 查询外部 KV

如果配置了 KV Connector，Scheduler 会在本地命中之后继续查询远端缓存。

异步加载时：

```text
分配目标 GPU Block
-> 请求变为 WAITING_FOR_REMOTE_KVS
-> 暂不执行模型
-> Worker 上报传输完成
-> 请求恢复为 WAITING/PREEMPTED
```

### 8.5 计算本轮 token 数

普通 Waiting 请求：

```python
num_new_tokens = request.num_tokens - num_computed_tokens
num_new_tokens = min(num_new_tokens, token_budget)
```

Prefix 命中多少，就少计算多少。

若关闭 Chunked Prefill 且剩余预算容纳不下整个 Prefill，Scheduler 会停止接纳：

```python
if not enable_chunked_prefill and num_new_tokens > token_budget:
    break
```

### 8.6 成功进入 Running

KV Slot 分配成功后：

```python
request = request_queue.pop_request()
self.running.append(request)
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_cached_tokens
```

并记录：

```python
num_scheduled_tokens[request_id] = num_new_tokens
```

## 九、KV Cache 分配为什么决定能否运行

Token Budget 只表示本轮愿意计算多少 token，真正执行还必须为这些 token 提供 KV Slot。

`KVCacheManager.allocate_slots()` 需要同时处理：

```text
comp：
  请求过去已经计算的 token

new_comp：
  本轮新命中的本地 Prefix Cache

ext_comp：
  外部 KV Connector 命中的 token

new：
  本轮需要计算的 token

lookahead：
  Speculative Decoding 预留位置
```

简化布局：

```text
| 已计算 | 新 Prefix 命中 | 外部命中 | 本轮计算 | Lookahead |
```

分配过程：

1. 清理 Sliding Window 外不再需要的 Block。
2. 计算需要新增多少物理 Block。
3. 与空闲/可驱逐 Block 数比较。
4. 绑定命中的 Prefix Block。
5. 为外部 KV 和本轮新 token 分配 Block。
6. 只把已确认的 token 放入 Prefix Cache。

若需求超过可用 Block：

```python
return None
```

因此 vLLM 的调度不是纯 CPU 队列算法，而是计算预算和显存资源的联合调度。

## 十、显存不足时如何抢占

Running 请求申请 KV Block 失败时，Scheduler 会选择请求抢占。

### 10.1 FCFS 模式

```python
preempted_req = self.running.pop()
```

即优先抢占 Running 列表尾部的请求，尽量保护更早进入系统的请求。

### 10.2 Priority 模式

```python
preempted_req = max(
    self.running,
    key=lambda req: (req.priority, req.arrival_time),
)
```

`priority` 数值越大优先级越低，因此选择最不重要、到达更晚的请求。

### 10.3 抢占具体做什么

`_preempt_request()` 会：

```python
kv_cache_manager.free(request)
encoder_cache_manager.free(request)

request.status = PREEMPTED
request.num_computed_tokens = 0
request.spec_token_ids = []
request.num_preemptions += 1

waiting.prepend_request(request)
```

这里使用的是 Recompute 策略：

> KV Cache 被释放，恢复时通过 Prefix Cache 命中或重新 Prefill 重建状态。

抢占不是暂停 GPU Kernel 后原地恢复，而是释放缓存换取其他请求继续执行。

### 10.4 抢占为什么代价高

长度为 `S` 的请求被抢占后，最坏要重新计算 `S` 个 token。频繁抢占通常说明：

- KV Cache 过小。
- `max_num_seqs` 过大。
- Admission Control 太激进。
- 长请求和短请求混合不合理。

`num_preemptions` 是重要的运行指标。

## 十一、SchedulerOutput 如何交给 Worker

Scheduler 不直接执行模型，而是构造 `SchedulerOutput`。

关键字段：

```python
scheduled_new_reqs
scheduled_cached_reqs
num_scheduled_tokens
total_num_scheduled_tokens
scheduled_spec_decode_tokens
scheduled_encoder_inputs
num_common_prefix_blocks
finished_req_ids
preempted_req_ids
kv_connector_metadata
```

### 11.1 New Request 与 Cached Request

第一次交给 Worker 的请求需要发送完整元数据：

```text
Prompt token
Sampling 参数
Block ID
LoRA
多模态输入
已计算 token 数
```

Worker 会缓存请求状态。

后续 step 只发送增量：

```text
新增 Block ID
新增 token
新的 num_computed_tokens
是否从抢占中恢复
```

这样避免每个 Decode step 都跨进程传完整 Prompt。

### 11.2 调度后立即推进 `num_computed_tokens`

构造输出后，`_update_after_schedule()` 执行：

```python
request.num_computed_tokens += num_scheduled_tokens[request_id]
```

为什么在模型返回前更新？

- SchedulerOutput 必须保留执行前的位置，供 Worker 准备输入。
- 异步/流水模式可能在前一批返回前继续调度下一批。
- Speculative Token 被拒绝时，再在 `update_from_output()` 中回退。

这是一个“先按计划推进，输出回来后校正”的设计。

## 十二、模型输出如何更新请求

模型执行结束后，`update_from_output()` 遍历本轮的 `num_scheduled_tokens`。

### 12.1 请求可能已被取消

模型执行期间客户端可能断开。更新前先检查：

```python
request = self.requests.get(req_id)
if request is None or request.is_finished():
    continue
```

避免向已结束请求写回 token。

### 12.2 修正投机 token

若 Draft Token 被拒绝：

```python
num_rejected = num_draft_tokens - num_accepted
request.num_computed_tokens -= num_rejected
```

这与前面的“先推进、后校正”对应。

### 12.3 写入生成 token

```python
request.append_output_token_ids(output_token)
```

该操作同时更新：

- Output Token 列表。
- All Token 列表。
- 新增完整 Block 的 Prefix Hash。

### 12.4 检查停止条件

每加入一个 token 都调用 `check_stop()`。如果一次投机解码返回多个 token，可能在中间 token 就满足停止条件，后面的 token 会被截断。

### 12.5 结束并释放

请求真正完成后：

```text
从 running/waiting 移除
-> 通知 KV Connector
-> 释放 Encoder Cache
-> 释放请求持有的 KV Block
-> 从 requests 主索引删除
-> 把 finished_req_ids 传给 Worker
```

Prefix Cache 开启时，`free()` 不一定立即擦除 KV 数据。无引用 Block 可保留为缓存，之后再按驱逐顺序复用。

## 十三、一个完整数值例子

设：

```text
max_num_batched_tokens = 8
max_num_seqs = 3
```

当前：

| 请求 | 状态 | `num_tokens` | `num_tokens_with_spec` | `num_computed_tokens` | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| A | RUNNING | 101 | 101 | 100 | 普通 Decode |
| B | RUNNING | 50 | 52 | 50 | 有 2 个 Draft Token 待验证 |
| C | WAITING | 10 | 10 | 0 | 新 Prompt，前缀命中 4 token |

### 13.1 先处理 Running

A：

```text
num_new_tokens = 101 - 100 = 1
token_budget: 8 -> 7
```

B：

```text
num_new_tokens = 52 - 50 = 2
token_budget: 7 -> 5
```

### 13.2 再处理 Waiting

C 的 Prefix Cache 命中 4 个 token：

```text
num_computed_tokens = 4
remaining = 10 - 4 = 6
```

当前预算只剩 5，因此 Chunked Prefill：

```text
num_new_tokens = 5
token_budget: 5 -> 0
```

本轮结果：

```python
num_scheduled_tokens = {
    "A": 1,
    "B": 2,
    "C": 5,
}
```

调度后：

```text
A.num_computed_tokens = 101
B.num_computed_tokens = 52
C.num_computed_tokens = 9
```

如果 B 的一个 Draft Token 被拒绝，输出回写时：

```text
B.num_computed_tokens: 52 -> 51
```

C 仍有一个 Prompt token 未计算，下一轮继续 Prefill，不会采样输出。

## 十四、简化版调度器

下面省略多模态、KV Connector、LoRA 和异步调度，只保留主干：

```python
def schedule(running, waiting, token_budget, max_running, kv_manager):
    scheduled = {}
    preempted = []

    # 1. 优先推进已有请求。
    index = 0
    while index < len(running) and token_budget > 0:
        request = running[index]
        need = request.num_tokens_with_spec - request.num_computed_tokens
        num_tokens = min(need, token_budget)

        blocks = kv_manager.allocate_slots(request, num_tokens)
        while blocks is None and running:
            victim = running.pop()
            kv_manager.free(victim)
            victim.status = "PREEMPTED"
            victim.num_computed_tokens = 0
            waiting.appendleft(victim)
            preempted.append(victim)

            if victim is request:
                break
            blocks = kv_manager.allocate_slots(request, num_tokens)

        if blocks is None:
            break

        scheduled[request.request_id] = num_tokens
        token_budget -= num_tokens
        index += 1

    # 2. 本轮发生抢占时，不再接纳新请求。
    if preempted:
        return scheduled

    # 3. 利用剩余预算接纳 Waiting 请求。
    while waiting and token_budget > 0 and len(running) < max_running:
        request = waiting[0]

        cached_blocks, cached_tokens = kv_manager.get_computed_blocks(request)
        need = request.num_tokens - cached_tokens
        num_tokens = min(need, token_budget)

        blocks = kv_manager.allocate_slots(
            request,
            num_tokens,
            num_new_computed_tokens=cached_tokens,
            new_computed_blocks=cached_blocks,
        )
        if blocks is None:
            break

        waiting.popleft()
        running.append(request)
        request.status = "RUNNING"
        request.num_computed_tokens = cached_tokens

        scheduled[request.request_id] = num_tokens
        token_budget -= num_tokens

    # 4. 按计划推进，执行结果返回后再修正。
    for request in running:
        request.num_computed_tokens += scheduled.get(request.request_id, 0)

    return scheduled
```

真实 vLLM 的复杂度主要来自：不同优化技术共享这条主干，而不是来自基本队列算法。

## 十五、关键配置与观测方法

### 15.1 `max_num_seqs`

限制 Running 请求数。过大可能提高并发，也可能提前占满 KV Cache。

### 15.2 `max_num_batched_tokens`

限制单轮总 token 数：

- 增大通常提高 Prefill 吞吐。
- 也会拉长单 step 时间，影响 Decode TPOT。

### 15.3 `long_prefill_token_threshold`

限制单个长 Prompt 每轮最多处理多少 token，避免它独占预算。

### 15.4 `policy`

```text
fcfs：     先到先服务，行为更稳定
priority： 按业务优先级和到达时间排序
```

### 15.5 `enable_prefix_caching`

减少重复 Prefill，但也会让空闲 KV Block 同时承担缓存角色。命中率、驱逐和调度是同一个 Block Pool 中的联合问题。

### 15.6 应重点观察

```text
running requests
waiting requests
scheduled tokens per step
batch sequences per step
KV Cache usage
Prefix Cache token hit rate
preemption count
TTFT / TPOT P50、P95、P99
每轮 schedule 和 execute_model 耗时
```

常见判断：

```text
Waiting 很高、GPU 不满：
  可能受 KV、LoRA、Grammar 或 Token Budget 限制

Preemption 持续增长：
  KV 容量或 Admission 配置过激

Prefill 吞吐高、TPOT 抖动：
  单轮 Token Budget 或长 Prefill 阈值过大

KV 使用率高但 Prefix 命中低：
  缓存保留策略可能不适合当前流量
```

## 十六、总结

vLLM V1 请求调度可以压缩成一条主线：

```text
请求进入 Waiting
-> Running 请求优先获得 Token Budget
-> Waiting 请求查询 Prefix Cache
-> 为本轮 token 分配 KV Block
-> 必要时抢占低优先级请求
-> 生成 SchedulerOutput
-> Worker 执行模型
-> 写回 token 并修正状态
-> 完成后释放或缓存 KV Block
```

最关键的三个设计是：

1. 用 `num_tokens - num_computed_tokens` 统一 Prefill、Decode 和投机验证。
2. 用 Token Budget、Running 数量和 KV Block 共同决定本轮可执行工作。
3. 调度后先推进状态，模型输出返回后再处理拒绝、停止和资源回收。

理解这三个点后，Chunked Prefill、Prefix Cache、Speculative Decoding、P/D KV Transfer 等分支都可以放回同一套调度框架中理解。
