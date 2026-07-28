---
title: 'KV Cache Offload：用 CPU 与分层存储扩展上下文容量'
description: '讲解 GPU/CPU/NVMe 分层 KV Cache、Block Offload、异步 DMA、Pinned Memory、换入换出状态机、带宽成本，以及 vLLM Offloading Manager 的实现。'
category: '推理优化'
pubDate: '2026-07-28T12:44:00+08:00'
updatedDate: '2026-07-28T12:44:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [为什么需要 KV Cache Offload](#一为什么需要-kv-cache-offload)
2. [Offload 与量化、Prefix Cache 的区别](#二offload-与量化prefix-cache-的区别)
3. [分层缓存结构](#三分层缓存结构)
4. [Block 级换入换出](#四block-级换入换出)
5. [带宽和延迟成本](#五带宽和延迟成本)
6. [异步传输与计算重叠](#六异步传输与计算重叠)
7. [最小 Offload Manager](#七最小-offload-manager)
8. [vLLM 中的 CPU Offload](#八vllm-中的-cpu-offload)
9. [多级 Tiering](#九多级-tiering)
10. [驱逐和调度策略](#十驱逐和调度策略)
11. [适用场景与评估](#十一适用场景与评估)
12. [总结](#十二总结)

## 一、为什么需要 KV Cache Offload

GPU HBM 容量有限，而 KV Cache 随活跃 token 数线性增长：

```text
KV_bytes =
    active_tokens
  * num_layers
  * 2
  * num_kv_heads
  * head_dim
  * dtype_bytes
```

即使使用 PagedAttention 消除大部分碎片，物理 KV Payload 仍必须占据存储空间。当以下场景出现时，GPU KV Pool 可能不足：

- 超长上下文。
- 高并发会话。
- 大量可复用前缀。
- 会话暂时空闲，但稍后可能继续。
- Prefill/Decode 分离需要跨节点保存 KV。

KV Cache Offload 把暂时不需要驻留 GPU 的 KV block 移到 CPU 内存、远端内存或 NVMe，需要时再加载回来。

## 二、Offload 与量化、Prefix Cache 的区别

三种技术作用不同：

| 技术 | 核心目标 | 是否减少计算 | 是否减少单份 KV 字节 |
| --- | --- | --- | --- |
| Prefix Cache | 复用已有前缀 | 是 | 否 |
| KV Quantization | 降低存储和读取字节 | 否 | 是 |
| KV Offload | 扩展可用存储层级 | 命中时避免重算 | GPU 中是 |

它们可以组合：

```text
Prefix Cache 发现历史 KV
-> GPU 未驻留
-> 从 CPU/NVMe 加载量化 KV block
-> 继续计算后缀
```

Offload 本质是容量换带宽和延迟。它不会凭空消除 KV 数据。

## 三、分层缓存结构

典型层级：

```text
L0: GPU HBM
    容量小，带宽最高，Attention 可直接访问

L1: CPU Pinned Memory
    容量大，经 PCIe/NVLink-C2C 传输

L2: 本地 NVMe
    容量更大，延迟和带宽明显更差

L3: 远端内存/存储
    可跨实例共享，受网络影响
```

只有 GPU 层能被普通 Attention Kernel 直接消费。低层命中通常需要先 Promotion：

```text
L2 hit
-> L2 到 L1
-> L1 到 L0
-> Attention
```

反方向写出称为 Cascade 或 Demotion：

```text
L0 -> L1 -> L2
```

## 四、Block 级换入换出

Offload 通常与 Paged KV Cache 使用相同 block 粒度：

```text
OffloadKey
  -> CPU/NVMe block id
  -> 对应完整的逐层 K/V Payload
```

### 4.1 为什么不用请求级大块

请求长度动态增长，以请求为单位拷贝会反复搬运已保存的数据。Block 级操作可以只处理新增或被驱逐的 block。

### 4.2 Offload Key

仅使用物理 GPU Block ID 不安全，因为该 ID 会被复用。稳定 Key 通常来自：

```text
前缀 Block Hash + KV Cache Group
```

物理地址是临时位置，内容哈希才表示“这块 KV 属于哪个前缀”。

### 4.3 状态机

一个 CPU Cache Entry 至少有：

```text
EMPTY       没有有效数据
STORING     GPU -> CPU 传输中
READY       可被加载
LOADING     CPU -> GPU 传输中
EVICTABLE   READY 且无引用
```

在传输完成事件到达前，不能把 `STORING` 数据作为命中，也不能覆盖正在 `LOADING` 的 CPU block。

## 五、带宽和延迟成本

传输时间下界：

```text
T_transfer >= bytes / effective_bandwidth
```

使用前文示例，BF16 KV 每 token 为 320 KiB。加载 4096 token：

```text
320 KiB * 4096 = 1.25 GiB
```

若有效主机到设备带宽为 32 GB/s：

```text
T >= 1.25 GiB / 32 GB/s ≈ 39 ms
```

这还没有计算：

- DMA 启动。
- Page Fault 或非 Pinned Memory 开销。
- CPU 索引和拷贝。
- GPU 同步。
- 多 GPU 重复传输。

如果重新 Prefill 4096 token 只需更短时间，加载反而不值得。因此 Offload 策略应比较：

```text
T_load(KV) 与 T_recompute(prefix)
```

而不是默认“缓存命中一定更快”。

### 5.1 量化对 Offload 的价值

KV 从 BF16 降到 FP8，理论上同时把：

```text
CPU 容量
PCIe 传输字节
GPU 恢复后的容量
```

降低约一半。量化与 Offload 的组合价值通常比只看 GPU 容量更大。

## 六、异步传输与计算重叠

同步路径：

```text
CPU -> GPU copy
等待完成
模型计算
```

会把全部传输时间暴露在请求延迟中。更好的方式是使用独立 CUDA Stream：

```text
copy stream:    load layer/block i+1
compute stream: compute layer/block i
```

### 6.1 Pinned Memory

CPU Tensor 使用 Page-locked Memory 后，GPU DMA 可直接访问稳定物理页，异步 H2D/D2H 更可靠：

```python
cpu_buffer = torch.empty(
    shape,
    dtype=dtype,
    pin_memory=True,
)
```

普通 Pageable Memory 往往需要额外 Staging，吞吐和异步性更差。

### 6.2 Event 同步

不能对整个设备执行全局同步。应记录传输 Event：

```python
with torch.cuda.stream(copy_stream):
    gpu_block.copy_(cpu_block, non_blocking=True)
    ready_event.record(copy_stream)

compute_stream.wait_event(ready_event)
```

只有真正依赖该 KV 的计算等待对应 Event。

### 6.3 Layer-wise Pipelining

若缓存按 Layer 组织，可以：

```text
加载 Layer 0 KV
计算 Layer 0，同时加载 Layer 1
计算 Layer 1，同时加载 Layer 2
...
```

能否隐藏传输取决于：

```text
T_copy_per_layer <= T_compute_per_layer
```

Decode 计算很短时，PCIe 传输往往难以完全隐藏。

## 七、最小 Offload Manager

下面展示 Block 映射和引用保护，不包含真实异步 DMA：

```python
from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class CPUEntry:
    tensor: torch.Tensor
    ref_count: int = 0
    ready: bool = False


class CPUKVOffload:
    def __init__(self, capacity_blocks: int):
        self.capacity_blocks = capacity_blocks
        self.entries: OrderedDict[bytes, CPUEntry] = OrderedDict()

    def lookup(self, key: bytes) -> bool:
        entry = self.entries.get(key)
        if entry is None or not entry.ready:
            return False
        self.entries.move_to_end(key)
        return True

    def store(self, key: bytes, gpu_block: torch.Tensor) -> None:
        self._ensure_capacity()
        cpu_block = torch.empty_like(
            gpu_block,
            device="cpu",
            pin_memory=True,
        )
        cpu_block.copy_(gpu_block, non_blocking=False)
        self.entries[key] = CPUEntry(cpu_block, ready=True)

    def load(self, key: bytes, gpu_block: torch.Tensor) -> None:
        entry = self.entries[key]
        entry.ref_count += 1
        try:
            gpu_block.copy_(entry.tensor, non_blocking=False)
            self.entries.move_to_end(key)
        finally:
            entry.ref_count -= 1

    def _ensure_capacity(self) -> None:
        while len(self.entries) >= self.capacity_blocks:
            for key, entry in self.entries.items():
                if entry.ref_count == 0:
                    del self.entries[key]
                    break
            else:
                raise RuntimeError("没有可驱逐的 CPU KV block")
```

真实系统需要批量拷贝多个 Layer/Block，并通过 Event 在 Scheduler 和 Worker 间同步完成状态。

## 八、vLLM 中的 CPU Offload

vLLM 的原生 CPU Offload 分为调度侧和 Worker 侧。

### 8.1 调度侧

Scheduler Manager 维护：

```text
CPU BlockPool
CPU 前缀哈希索引
GPU block -> CPU block 的 Transfer Metadata
Load/Store Request State
传输 Event 到请求或 Block 的映射
```

当 GPU 前缀缓存 miss 时，会继续查询 CPU：

```text
GPU longest prefix hit
-> 从剩余 Block Hash 查询 CPU
-> Pin 命中的 CPU block
-> 分配 GPU slot
-> 创建 Load Metadata
```

Pin 操作保证调度到真正发起传输期间，CPU block 不会被 LRU 驱逐。

### 8.2 Worker 侧

Worker 根据 GPU KV Cache 的真实存储布局创建：

```text
GPU block view: [num_gpu_blocks, bytes_per_block]
CPU block view: [num_cpu_blocks, bytes_per_block]
```

然后：

- 使用 Pinned CPU Tensor。
- 建立独立 Load/Store CUDA Stream。
- 批量发起 GPU/CPU Block Copy。
- 用 Event 上报已完成传输。

开源实现会处理不同 Attention Backend 的物理布局，例如 K/V 是否位于最外层维度、多个 Layer 是否共享底层 Storage。

### 8.3 Eager 与 Lazy Offload

```text
Eager：
  新计算的 Block 尽快写到 CPU
  命中概率高，D2H 流量也更高

Lazy：
  只在 GPU 空闲 Block 低于水位时写出
  减少无效传输，但被突然驱逐时更紧迫
```

Lazy 模式通常维护目标空闲 Block 水位，提前写出潜在驱逐候选。

## 九、多级 Tiering

多级 Offload Manager 把 CPU 作为 GPU 的直接 Gateway：

```text
GPU <-> CPU Primary <-> Storage/Network Secondary
```

设计原则：

1. Secondary Tier 不直接操作 GPU Buffer。
2. Secondary 命中先 Promotion 到 CPU。
3. CPU Entry 就绪后才能继续 Promotion 到 GPU。
4. 异步任务完成前使用引用计数保护 Block。
5. 查询可返回“存在但尚未就绪”，让 Scheduler 稍后重试。

一种三态 Lookup：

```text
True：  数据已在 Primary，立即可读
None：  数据存在，正在 Promotion
False： 所有 Tier 均未命中
```

`None` 与 `False` 必须区分。前者应等待，后者应重算。

### 9.1 向所有 Tier 写出还是按需写出

向所有 Tier Cascade 可提高持久性和跨实例复用，但会增加写流量。可根据：

- 前缀复用频率。
- KV 大小。
- 会话存活时间。
- Secondary 容量。
- 网络成本。

决定 Block-level 或 Request-level Offload。

## 十、驱逐和调度策略

### 10.1 LRU

驱逐最久未访问且引用数为 0 的 block。实现简单，但未考虑重算成本。

### 10.2 Cost-aware

可定义：

```text
value(block) =
    expected_reuse_probability
  * recompute_cost
  - transfer_cost
  - storage_cost
```

优先保留长公共前缀和高复用 block。

### 10.3 请求取消

取消请求时需要区分：

- 请求私有、尚未缓存的 block：可立即释放。
- 已进入 Prefix Cache 的共享 block：减少引用，不一定删除。
- 传输中的 block：等待或取消 DMA 后再回收。

### 10.4 GPU 空间预留

若加载命中前缀占满全部 GPU block，请求没有空间继续 Decode。Scheduler 应同时预留：

```text
加载前缀所需 Block
当前 step 新 token 所需 Block
必要的 Lookahead Block
```

## 十一、适用场景与评估

### 11.1 更适合

- 会话式应用，用户思考期间 KV 可暂时换出。
- 长公共前缀会被重复访问。
- GPU 容量是主要限制。
- CPU 内存充足，主机互联较快。
- 允许较高 TTFT，重点提高并发容量。

### 11.2 不适合

- 每个请求只访问一次。
- TPOT 极端敏感。
- PCIe 带宽紧张或与其他 I/O 争用。
- 重新 Prefill 比加载更快。
- CPU NUMA 放置错误，跨 Socket 搬运。
- KV Cache 很短，管理开销高于收益。

### 11.3 必须观测

```text
GPU/CPU/Secondary cache hit rate
命中 token 数和平均前缀长度
H2D/D2H bytes 与有效带宽
Load/Store latency
传输与计算重叠比例
因 Promotion 等待的 queue time
Offload eviction rate
重算 token 数
TTFT / TPOT P50/P99
```

应分别比较：

```text
无 Offload，容量较低
Offload，但同步传输
Offload + Pinned Memory + Async Stream
Offload + KV Quantization
```

## 十二、总结

KV Cache Offload 把 GPU KV Pool 扩展为分层缓存：

```text
GPU HBM
<-> CPU Pinned Memory
<-> NVMe 或远端存储
```

高质量实现需要 Block Hash、引用保护、异步 Event、Pinned Memory、独立 Stream 和明确的 Promotion/Cascade 状态机。Offload 的核心判断不是“能否保存”，而是加载成本是否小于重算成本，以及传输能否被计算隐藏。它最稳定的收益是扩大可服务 token 容量，延迟收益则高度依赖复用率和互联带宽。
