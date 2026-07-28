---
title: 'Prefix Caching 与 RadixAttention：复用 Prompt 的 KV Cache'
description: '从 KV Cache 可复用条件出发，讲解块哈希前缀缓存、Radix Tree、引用计数、LRU 驱逐、缓存隔离，以及 vLLM 与 SGLang 的实现差异。'
category: '推理优化'
pubDate: '2026-07-28T12:39:00+08:00'
updatedDate: '2026-07-28T12:39:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [Prefix Caching 解决什么问题](#一prefix-caching-解决什么问题)
2. [为什么前缀 KV 可以复用](#二为什么前缀-kv-可以复用)
3. [块哈希方案](#三块哈希方案)
4. [Radix Tree 方案](#四radix-tree-方案)
5. [一个完整例子](#五一个完整例子)
6. [最小实现](#六最小实现)
7. [vLLM 的块哈希缓存](#七vllm-的块哈希缓存)
8. [SGLang 的 RadixAttention](#八sglang-的-radixattention)
9. [引用计数与驱逐](#九引用计数与驱逐)
10. [正确性与隔离](#十正确性与隔离)
11. [收益、限制与评估](#十一收益限制与评估)
12. [总结](#十二总结)

## 一、Prefix Caching 解决什么问题

在线服务中，大量请求会包含相同前缀：

```text
系统提示词 + 工具定义 + 用户问题 A
系统提示词 + 工具定义 + 用户问题 B
系统提示词 + 工具定义 + 用户问题 C
```

如果相同的系统提示词和工具定义每次都重新 Prefill，会重复执行完全相同的 QKV 投影、Attention 和 MLP。

Prefix Caching 保存已经计算过的前缀 KV Cache。新请求命中后，只计算未命中的后缀：

```text
完整 Prompt：8192 token
缓存命中：6144 token
实际 Prefill：2048 token
```

它主要降低：

- 重复 Prompt 的 Prefill 计算量。
- 首 token 延迟 TTFT。
- Prefill 对其他请求的计算干扰。

## 二、为什么前缀 KV 可以复用

在确定性前向中，第 `i` 层第 `t` 个 token 的 K/V 由以下信息决定：

```text
模型权重
token[0:t]
位置编码
模型结构和精度
可能影响前向的适配器或多模态输入
```

因果注意力保证位置 `t` 不依赖未来 token。因此两个请求拥有完全相同的 token 前缀和执行上下文时，前缀位置产生的 K/V 相同。

设：

```text
request A = [P, A_suffix]
request B = [P, B_suffix]
```

共同前缀 `P` 对应的：

```text
K_P^l, V_P^l, l = 0 ... L-1
```

可被 A、B 同时引用。后缀则分别分配新 KV block。

注意，缓存的是每一层的 K/V，不是最终 Hidden State。后续 token 的 Attention 仍要读取这些 K/V。

## 三、块哈希方案

Paged KV Cache 已把 token 划分为固定大小 block。最直接的 Prefix Cache 是对每个完整 block 计算哈希。

假设 block size 为 4：

```text
tokens = [10, 11, 12, 13, 20, 21, 22, 23, 30]

block 0 = [10, 11, 12, 13]
block 1 = [20, 21, 22, 23]
block 2 = [30]                 # 未满，通常暂不加入全局缓存
```

不能只哈希当前 block，因为相同 token block 出现在不同历史前缀后，位置和上下文不同。应构造链式哈希：

```text
h0 = H(None, block0, extra_keys0)
h1 = H(h0,   block1, extra_keys1)
h2 = H(h1,   block2, extra_keys2)
```

这样 `h1` 同时编码了 block 0 和 block 1 的完整前缀。

查找从第一个 block 开始，遇到第一次 miss 就停止：

```text
h0 hit
h1 hit
h2 miss
=> 最长可复用前缀为 2 个 block
```

Prefix Cache 只能复用连续前缀。中间 block miss 后，即使后面的哈希碰巧存在也不能跳过。

### 3.1 为什么通常只缓存完整 block

分页缓存以 block 为分配和引用单位。只缓存完整 block 可以：

- 保持哈希与物理 block 一一对应。
- 避免不同尾部长度造成复杂状态。
- 让引用计数和驱逐以固定粒度进行。

代价是最多浪费 `block_size - 1` 个可复用 token。

## 四、Radix Tree 方案

Radix Tree 是压缩前缀树。边上可以保存一段 token，而不是每条边只保存一个 token。

已有三个序列：

```text
[A, B, C, D, X]
[A, B, C, D, Y]
[A, B, M, N]
```

压缩后的结构：

```text
root
 └─ [A, B]
     ├─ [C, D]
     │   ├─ [X]
     │   └─ [Y]
     └─ [M, N]
```

树节点的 value 指向对应 token 范围的 KV Cache 位置。查找请求 `[A, B, C, D, Z]` 时，最长匹配为 `[A, B, C, D]`。

Radix Tree 的优势是：

- 天然表达多个请求的共享前缀。
- 能在节点分裂后精确表示匹配边界。
- 遍历路径即可得到最长前缀。
- 子树结构可用于 LRU、优先级和引用保护。

块哈希和 Radix Tree 解决的是同一问题，但索引方式不同：

| 方案 | 索引 | 匹配粒度 | 典型特点 |
| --- | --- | --- | --- |
| 块哈希 | 链式 Block Hash | 完整 block | 易与 Paged KV Pool 集成 |
| Radix Tree | Token 序列前缀树 | 可按页对齐 | 显式表达共享前缀结构 |

## 五、一个完整例子

设 block size 为 4，已有缓存：

```text
P0 = [1, 2, 3, 4]
P1 = [5, 6, 7, 8]
```

请求 A：

```text
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

前 8 token 命中，只需计算 `[9, 10]`。

请求 B：

```text
[1, 2, 3, 4, 5, 6, 99, 100]
```

第二个 block 不同，因此只命中前 4 token。即使 `[5, 6]` 相同，也不能复用半个 block，除非缓存索引支持更细粒度。

请求 C 使用另一个 LoRA：

```text
token 与 A 完全相同
adapter_id 不同
```

模型前向不同，必须 cache miss。缓存键应包含 LoRA 或模型版本：

```text
cache_key = H(parent_hash, block_tokens, adapter_id, model_revision)
```

## 六、最小实现

### 6.1 链式块哈希

```python
import hashlib
import json
from dataclasses import dataclass


def hash_block(
    parent_hash: bytes,
    token_ids: tuple[int, ...],
    namespace: str,
) -> bytes:
    # 使用确定性的结构化序列化，避免直接拼接字段造成边界歧义。
    payload = json.dumps(
        {
            "parent": parent_hash.hex(),
            "tokens": token_ids,
            "namespace": namespace,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def build_prefix_hashes(
    token_ids: list[int],
    block_size: int,
    namespace: str,
) -> list[bytes]:
    hashes = []
    parent = b""

    # 只处理完整 block。
    num_full_blocks = len(token_ids) // block_size
    for block_id in range(num_full_blocks):
        start = block_id * block_size
        block = tuple(token_ids[start : start + block_size])
        parent = hash_block(parent, block, namespace)
        hashes.append(parent)

    return hashes
```

### 6.2 缓存索引

```python
@dataclass
class CachedBlock:
    physical_block_id: int
    ref_count: int = 0


class PrefixBlockCache:
    def __init__(self):
        self.blocks: dict[bytes, CachedBlock] = {}

    def match(self, block_hashes: list[bytes]) -> list[CachedBlock]:
        matched = []
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None:
                break
            block.ref_count += 1
            matched.append(block)
        return matched
```

哈希命中后还需要把物理 block 加入新请求的 Block Table。多个请求共享 block 时不能覆盖或释放它。

## 七、vLLM 的块哈希缓存

vLLM 的 `BlockPool` 同时管理：

```text
blocks                         所有 KV block
free_block_queue               空闲和可驱逐 block 的顺序
cached_block_hash_to_block     哈希到缓存 block 的映射
```

完整 block 被缓存时，框架把请求预先计算好的链式哈希写入 block 元数据，并加入哈希表。

核心哈希关系可概括为：

```python
block_hash = H(
    parent_block_hash,
    tuple(current_block_token_ids),
    extra_keys,
)
```

`extra_keys` 不只是附加字段，它决定缓存是否可以安全共享。开源实现会考虑：

- LoRA/Adapter 标识。
- 多模态输入的内容哈希。
- Cache Salt。
- Prompt Embedding。
- KV Cache Group。

### 7.1 为什么要包含父哈希

若只哈希当前 token：

```text
[A, B] + [C, D]
[X, Y] + [C, D]
```

第二块都会得到 `H([C, D])`，但它们在不同上下文和位置下产生的 K/V 不同。加入父哈希后，两条链得到不同 key。

### 7.2 哈希算法的选择

安全哈希碰撞概率极低，但成本较高；非密码哈希更快，却增加理论碰撞风险。在多租户系统中，哈希碰撞可能造成错误结果甚至跨租户信息泄漏，因此不能只比较吞吐。

## 八、SGLang 的 RadixAttention

SGLang 使用 `RadixCache` 显式维护 token 前缀树。其主要接口可以抽象为：

```python
match_prefix(key)  # 查找最长缓存前缀
insert(key, value) # 插入 token 序列和 KV 索引
evict(num_tokens)  # 驱逐可回收叶节点
```

`match_prefix` 会：

1. 按页大小截断到可复用边界。
2. 从根节点沿 token 序列向下匹配。
3. 必要时拆分节点，使匹配点成为明确边界。
4. 返回拼接后的 KV Cache 索引和最后一个节点。
5. 刷新访问时间，供驱逐策略使用。

节点结构可以简化为：

```python
class RadixNode:
    def __init__(self, token_segment, kv_indices):
        self.key = token_segment
        self.value = kv_indices
        self.children = {}
        self.parent = None
        self.lock_ref = 0
        self.last_access_time = 0.0
```

### 8.1 节点分裂

树中已有边 `[1, 2, 3, 4]`，新请求只匹配 `[1, 2]` 时，需要拆分：

```text
拆分前：
root -> [1, 2, 3, 4]

拆分后：
root -> [1, 2] -> [3, 4]
```

KV 数据不需要复制，只需调整节点持有的索引区间。

### 8.2 RadixAttention 的含义

这里的 Attention 数学公式没有改变。“RadixAttention”强调的是用 Radix Tree 自动复用请求之间的 KV 前缀，并让调度器把缓存命中纳入请求执行。

## 九、引用计数与驱逐

共享 block 可能同时被多个运行中请求引用。驱逐必须满足：

```text
ref_count == 0
```

Radix Tree 中，运行请求锁定一个节点时，其祖先路径都不能驱逐，因为这些节点组成完整前缀。

一种简化规则：

```python
def lock_path(node):
    while node.parent is not None:
        node.ref_count += 1
        node = node.parent


def unlock_path(node):
    while node.parent is not None:
        node.ref_count -= 1
        node = node.parent
```

### 9.1 为什么通常从叶子驱逐

若删除中间节点而保留子节点，子节点会失去可用的完整前缀。按叶节点驱逐可维持树结构和前缀连续性：

```text
选择可驱逐叶子
-> 释放叶子 KV
-> 删除叶子
-> 若父节点变成无锁叶子，将其加入候选
```

### 9.2 LRU 不是唯一策略

可根据业务加入：

- 最近访问时间。
- 前缀长度。
- 复用次数。
- 重算成本。
- 请求优先级。

长前缀重算成本高，但占用也大。最优策略取决于工作负载，不应只追求命中率。

## 十、正确性与隔离

以下任一因素不同都可能使 KV 不可复用：

```text
模型权重或版本
Tokenizer 结果
RoPE/位置配置
LoRA 或 Adapter
多模态输入
量化方式与 KV Layout
影响前向的随机或自定义算子
租户隔离策略
```

### 10.1 文本相同不等于 token 相同

缓存索引应建立在最终 token ID 上，而不是原始字符串。Chat Template、特殊 token 和空格处理都可能改变 token 序列。

### 10.2 采样参数通常不影响 Prefill KV

Temperature、Top-k、Top-p 作用于输出采样，通常不改变 Prompt 的前向 K/V。但若系统有自定义模型逻辑，仍需根据实际计算图判断。

### 10.3 多租户安全

即使技术上可共享，也可能因安全策略禁止跨租户命中。常见做法是把租户标识加入 namespace，或为租户维护独立缓存池。

## 十一、收益、限制与评估

### 11.1 适合的工作负载

- 固定系统提示词。
- 大型工具定义或 JSON Schema。
- Few-shot 示例重复。
- RAG 中存在稳定公共前缀。
- 多轮对话不断追加后缀。

### 11.2 收益有限的场景

- 每个 Prompt 从开头就不同。
- 共享部分短于一个缓存 block。
- 模型或 Adapter 频繁切换。
- 缓存容量过小，前缀在复用前已被驱逐。
- Prefill 不是当前瓶颈。

### 11.3 不应只看 Cache Hit Rate

Token 级命中率更有意义：

```text
token_hit_rate =
    reused_prefill_tokens / total_prompt_tokens
```

还应记录：

```text
TTFT before/after
saved prefill tokens
cache occupancy
eviction rate
平均命中前缀长度
哈希/树查找 CPU 开销
因隔离条件导致的 miss
```

一个 100-token 请求命中 90%，与一个 100K-token 请求命中 90%，节省的计算量完全不同。

## 十二、总结

Prefix Caching 的正确抽象是：

```text
相同执行上下文下的相同 token 前缀
-> 相同的逐层 K/V
-> 共享物理 KV block
-> 只计算未命中的后缀
```

块哈希方案用父哈希构造连续前缀 key，容易与分页式 KV Pool 结合；Radix Tree 方案显式表达共享前缀结构，适合最长前缀匹配和树形驱逐。两种实现都必须处理引用保护、缓存隔离、完整前缀语义和容量驱逐，否则命中率提升可能以错误结果或资源失控为代价。
