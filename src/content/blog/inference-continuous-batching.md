---
title: 'Continuous Batching：让在线大模型推理持续保持高吞吐'
description: '系统讲解静态批处理、动态批处理与 Continuous Batching 的差异，以及 token 预算、请求状态机、KV Cache 分配、抢占和公平性设计。'
category: '推理优化'
pubDate: '2026-07-28T12:37:00+08:00'
updatedDate: '2026-07-28T12:37:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [问题来自哪里](#一问题来自哪里)
2. [三种 Batching 的区别](#二三种-batching-的区别)
3. [Continuous Batching 的执行过程](#三continuous-batching-的执行过程)
4. [为什么按 token 而不是请求计预算](#四为什么按-token-而不是请求计预算)
5. [一个具体调度例子](#五一个具体调度例子)
6. [最小调度器实现](#六最小调度器实现)
7. [开源框架中的关键设计](#七开源框架中的关键设计)
8. [抢占、饥饿与公平性](#八抢占饥饿与公平性)
9. [性能收益与代价](#九性能收益与代价)
10. [调参和观测](#十调参和观测)
11. [总结](#十一总结)

## 一、问题来自哪里

在线生成任务的输入长度和输出长度都不固定：

```text
请求 A：prompt 128 token，输出 20 token
请求 B：prompt 2048 token，输出 200 token
请求 C：prompt 64 token，输出 5 token
```

自回归 Decode 每轮只为每个请求生成一个 token。若把 A、B、C 固定成一个 batch，C 完成后仍要等待 B，C 占据的 batch 槽位也无法立即交给新请求。

Continuous Batching 的核心是：

> 每完成一次模型迭代，就重新决定下一轮由哪些请求、多少 token 参与计算。

因此 batch 不再是一次请求级决策，而是每个推理 step 都可以变化的运行时状态。

## 二、三种 Batching 的区别

### 2.1 静态批处理

先收集固定数量的请求，再一起运行到全部结束：

```text
batch = [A, B, C]
while batch 中仍有请求未完成:
    对整个 batch 执行一步 Decode
```

优点是实现简单、Shape 稳定；缺点是短请求被长请求拖住，空闲槽位不能及时复用。

### 2.2 动态批处理

服务端等待一个很短的聚合窗口，把这段时间内到达的请求组成 batch：

```text
0 ms  到达 A
1 ms  到达 B
3 ms  聚合窗口结束，运行 [A, B]
```

它改善了请求到达时间不同的问题，但 batch 一旦开始运行，通常仍保持到结束。

### 2.3 Continuous Batching

每轮 Decode 后移除完成请求，并把等待队列中的新请求加入下一轮：

```text
step 0: [A, B, C]
step 1: [A, B]       # C 完成
step 2: [A, B, D]    # D 立即加入
step 3: [B, D, E]    # A 完成，E 加入
```

它也称 In-flight Batching。关键不是“batch 更大”，而是 batch 的成员在运行过程中持续变化。

## 三、Continuous Batching 的执行过程

一个推理引擎通常维护以下状态：

```text
waiting：尚未获得计算或 KV Cache 资源的请求
running：当前可参与执行的请求
preempted：因资源不足被抢占、等待恢复的请求
finished：已完成或取消的请求
```

每个调度 step 包含：

1. 回收已完成请求的 KV Cache block。
2. 更新仍在运行的请求状态。
3. 为 running 请求分配本轮 token 预算。
4. 在剩余预算内接纳 waiting 请求。
5. 为新增 token 分配 KV Cache slot。
6. 组装扁平化输入并执行一次模型前向。
7. 采样 token，更新请求并返回流式结果。

简化流程如下：

```text
            新请求
              |
              v
waiting --> schedule --> running --> model step --> sample
               ^            |                       |
               |            | 资源不足              | 完成
               +-- preempted+                       v
                                                finished
```

## 四、为什么按 token 而不是请求计预算

一个请求在同一轮中可能执行：

- 1 个 Decode token。
- 几百个 Chunked Prefill token。
- 多个投机验证 token。

仅限制请求数不能准确限制计算量。例如下面两个 batch 都有 8 个请求：

```text
batch A：8 个 Decode 请求，共 8 token
batch B：8 个 Prefill 请求，共 8192 token
```

二者的算力、临时张量和 KV Cache 需求完全不同。因此现代调度器通常同时约束：

```text
num_sequences <= max_num_seqs
sum(scheduled_tokens) <= max_num_batched_tokens
required_kv_blocks <= free_kv_blocks
```

这里的 token budget 是每个 step 可以交给模型执行的 token 总数：

```text
token_budget = max_num_batched_tokens
```

每接纳一个请求的一部分 token：

```text
scheduled = min(request.remaining_tokens, token_budget)
token_budget -= scheduled
```

## 五、一个具体调度例子

设定：

```text
max_num_seqs = 4
max_num_batched_tokens = 16
```

当前请求：

| 请求 | 阶段 | 本轮最多需要 |
| --- | --- | ---: |
| A | Decode | 1 token |
| B | Decode | 1 token |
| C | Prefill | 30 token |
| D | Decode | 1 token |
| E | Waiting/Prefill | 8 token |

若 Decode 优先，先为 A、B、D 各分配 1 token，剩余预算为 13。C 的 Prefill 可以切出 13 token：

```text
A: 1
B: 1
D: 1
C: 13
总计: 16
```

由于序列数已达到 4，E 暂时等待。下一轮 C 还剩 17 个 Prefill token；若 A 已完成，E 就能占用释放的序列槽位。

这个例子体现了两个独立约束：

- 序列槽位决定能同时维护多少请求状态。
- token budget 决定单轮允许多少计算工作。

## 六、最小调度器实现

下面的代码只保留 Continuous Batching 的核心逻辑：

```python
from collections import deque
from dataclasses import dataclass


@dataclass
class Request:
    request_id: str
    prompt_left: int
    output_left: int

    @property
    def is_prefill(self) -> bool:
        return self.prompt_left > 0

    @property
    def finished(self) -> bool:
        return self.prompt_left == 0 and self.output_left == 0


class ContinuousBatchScheduler:
    def __init__(self, max_seqs: int, token_budget: int):
        self.max_seqs = max_seqs
        self.max_token_budget = token_budget
        self.waiting = deque()
        self.running: list[Request] = []

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    def schedule(self) -> list[tuple[Request, int]]:
        # 每个 step 先清理已完成请求。
        self.running = [req for req in self.running if not req.finished]

        while self.waiting and len(self.running) < self.max_seqs:
            self.running.append(self.waiting.popleft())

        budget = self.max_token_budget
        scheduled: list[tuple[Request, int]] = []

        # 简化策略：先处理 Decode，避免已有请求的 TPOT 被长 Prefill 拉高。
        ordered = sorted(self.running, key=lambda req: req.is_prefill)

        for req in ordered:
            if budget == 0:
                break

            if req.is_prefill:
                num_tokens = min(req.prompt_left, budget)
            else:
                num_tokens = 1

            scheduled.append((req, num_tokens))
            budget -= num_tokens

        return scheduled

    @staticmethod
    def commit(scheduled: list[tuple[Request, int]]) -> None:
        for req, num_tokens in scheduled:
            if req.is_prefill:
                req.prompt_left -= num_tokens
            else:
                req.output_left -= 1
```

真实系统还要处理 KV Cache 分配、前缀命中、优先级、投机 token、LoRA、跨节点 KV 加载和请求取消。

## 七、开源框架中的关键设计

以 vLLM V1 的 `Scheduler` 为例，核心调度入口维护一个整数 `token_budget`，并为每个请求记录 `num_scheduled_tokens`：

```python
token_budget = max_num_scheduled_tokens
num_scheduled_tokens = {}

for request in runnable_requests:
    num_new_tokens = min(request.remaining_tokens, token_budget)
    new_blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens,
    )
    if new_blocks is None:
        # KV 空间不足，不能直接把请求放进本轮。
        continue

    num_scheduled_tokens[request.request_id] = num_new_tokens
    token_budget -= num_new_tokens
```

源码中的重要点有四个：

### 7.1 Running 和 Waiting 分开处理

正在 Decode 的请求通常已经占有 KV Cache。先处理它们可以维持稳定 TPOT，再利用剩余预算接纳新 Prefill。

### 7.2 计算预算和缓存预算同时检查

即使 token budget 还有剩余，KV Cache block 分配失败也不能执行。调度器需要决定等待、驱逐缓存或抢占其他请求。

### 7.3 每个请求可调度多个 token

统一的 `num_scheduled_tokens` 表示允许同一套调度逻辑处理：

- 单 token Decode。
- Chunked Prefill。
- Speculative Decoding 的多 token 验证。

### 7.4 输入在执行前重新打包

运行时通常不会保留形如 `[batch, seq]` 的固定二维输入，而是把本轮 token 压成一维：

```text
input_ids:    [A0, B0, C0, C1, C2, ...]
positions:    [pa, pb, pc, pc+1, pc+2, ...]
slot_mapping: [sa, sb, sc, sc+1, sc+2, ...]
```

元数据负责把每段 token 映射回请求和 KV Cache 位置。

## 八、抢占、饥饿与公平性

### 8.1 KV Cache 不足

常见策略包括：

- 拒绝接纳新请求。
- 抢占低优先级或最近加入的请求。
- 释放被抢占请求的 KV Cache，稍后重算。
- 把 KV Cache 换出到 CPU。

抢占不是免费操作。若需要重新 Prefill，代价约与被丢弃的上下文长度成正比。

### 8.2 Decode 优先的副作用

Decode 优先有利于 TPOT，但持续到达的 Decode 请求可能让长 Prefill 得不到足够预算。可使用：

- 为 Prefill 保留最小预算。
- 限制同时运行的 Decode 数。
- 按等待时间提升优先级。
- 使用 Chunked Prefill 控制单个长请求的占用。

### 8.3 FCFS 与 Priority

FCFS 易理解且较公平，但不能表达业务优先级。Priority 策略需要明确：

```text
主排序键：优先级
次排序键：到达时间
```

还要防止低优先级请求永久饥饿，例如随等待时间增加动态提升其有效优先级。

## 九、性能收益与代价

### 9.1 收益

- 完成请求的资源可立即复用。
- Decode batch 更容易保持较高并发。
- 减少 Padding 计算。
- 支持流式返回和请求取消。
- 与 Paged KV Cache、Prefix Cache、Chunked Prefill 自然配合。

### 9.2 代价

- 每轮都有调度和元数据准备开销。
- batch Shape 动态变化，CUDA Graph 需要 Shape Bucket。
- 请求之间共享 GPU，延迟受其他请求影响。
- 大量小请求会增加 CPU 调度、采样和网络开销。
- 激进批处理提升吞吐时，单请求 TPOT 通常会上升。

Continuous Batching 优化的是系统吞吐和资源利用率，不保证每个请求都获得最低延迟。

## 十、调参和观测

### 10.1 `max_num_seqs`

增大后可容纳更多并发请求，但会增加：

- KV Cache 占用。
- 调度与采样开销。
- Decode kernel 中的元数据规模。

### 10.2 `max_num_batched_tokens`

增大后单轮 GPU 工作量更大，通常提高吞吐；但也会拉长一次 step 的执行时间，使 Decode 请求等待更久。

### 10.3 必须联合观测的指标

```text
TTFT：首 token 延迟
TPOT：相邻输出 token 的平均间隔
request throughput：每秒完成请求数
token throughput：每秒处理 token 数
queue time：请求排队时间
KV cache utilization：KV Cache 使用率
preemption count：抢占次数
batch tokens / batch sequences：每轮实际规模
```

只看 tokens/s 容易掩盖长尾延迟。在线服务至少应同时比较 P50、P95 和 P99 的 TTFT、TPOT。

## 十一、总结

Continuous Batching 把“组 batch”从请求开始前的一次性操作，变成每个模型 step 都执行的资源调度：

```text
请求持续到达
-> 每轮重新选择请求
-> 按 token 分配计算预算
-> 按需分配 KV Cache
-> 完成后立即回收并补入新请求
```

它是现代 LLM Serving 的基础机制。Paged KV Cache 解决动态内存，Continuous Batching 解决动态计算，两者共同使不同长度、不同到达时间的请求能够高效共享 GPU。
