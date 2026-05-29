---
title: 'PagedAttention：用分页思想管理 KV Cache'
description: '从 KV Cache 的动态增长和内存碎片问题出发，解释 PagedAttention 的逻辑块、物理块、Block Table 与 OS 虚拟内存类比。'
category: '推理优化'
pubDate: '2026-06-30'
updatedDate: '2026-06-30'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [KV Cache 为什么难管理](#二kv-cache-为什么难管理)
3. [连续内存分配的问题](#三连续内存分配的问题)
4. [PagedAttention 的基本思想](#四pagedattention-的基本思想)
5. [Block Table](#五block-table)
6. [Decode 时如何读取](#六decode-时如何读取)
7. [Prefix Sharing](#七prefix-sharing)
8. [面试表达](#八面试表达)
9. [总结](#九总结)

## 一、核心结论

PagedAttention 解决的是 KV Cache 的内存管理问题。

- LLM serving 中，每个请求的输出长度不同，KV Cache 会动态增长。
- 如果给每个请求预留连续大块显存，会造成严重浪费。
- 如果频繁申请释放连续显存，会产生碎片和同步开销。
- PagedAttention 把 KV Cache 切成固定大小 block。
- 每个请求看到的是逻辑连续 token，底层物理 block 可以不连续。
- Block Table 负责从逻辑块映射到物理块。

## 二、KV Cache 为什么难管理

Decode 阶段需要保存每一层的 K/V：

```text
KV Cache size ∝ layers * batch * seq_len * heads * head_dim * dtype_size
```

在线服务中请求有几个特点：

- prompt 长度不同。
- 输出长度未知。
- 请求随时到达，也随时结束。
- 有些请求共享相同系统 prompt。
- 请求取消后需要立即释放资源。

这和传统固定 batch 推理完全不同。

## 三、连续内存分配的问题

如果为每个请求分配一段连续 KV Cache：

```text
request A: [........]
request B: [....]
request C: [............]
```

问题包括：

- 不知道最终输出长度，只能多预留。
- 预留过多造成显存浪费。
- 请求结束后释放中间区域，容易形成碎片。
- 新请求可能找不到足够大的连续空间。

这类似操作系统中连续物理内存分配的问题。

## 四、PagedAttention 的基本思想

PagedAttention 借鉴虚拟内存分页。

```text
逻辑 token block -> 物理 KV block
```

假设 block size 为 16 token。一个请求长度为 40 token，需要 3 个逻辑块：

```text
logical block 0: token 0-15
logical block 1: token 16-31
logical block 2: token 32-39
```

它们可以映射到任意空闲物理块：

```text
logical: [0, 1, 2]
physical: [7, 3, 10]
```

请求视角中 token 是连续的；显存中物理 block 不需要连续。

## 五、Block Table

Block Table 保存逻辑块到物理块的映射。

```text
request_id -> [physical_block_id_0, physical_block_id_1, ...]
```

简化结构：

```python
block_table = {
    0: [7, 3, 10],
    1: [2, 8],
    2: [5, 6, 9, 11],
}
```

Decode kernel 根据 token 位置找到对应 physical block：

```python
logical_block = token_id // block_size
offset = token_id % block_size
physical_block = block_table[request_id][logical_block]
address = physical_block_base[physical_block] + offset
```

## 六、Decode 时如何读取

PagedAttention kernel 读取 KV 时，需要多一步地址翻译。

```cpp
int token_id = ...;
int logical_block = token_id / BLOCK_SIZE;
int offset = token_id % BLOCK_SIZE;

// 根据请求的 block table 找到物理 block。
int physical_block = block_table[request_id][logical_block];

// 再定位到该 token 的 K/V。
float* k_ptr = k_cache + physical_block * block_stride + offset * token_stride;
```

这会增加一点索引开销，但换来显存利用率和动态调度能力。

## 七、Prefix Sharing

多个请求可能共享相同前缀：

```text
system prompt + user question A
system prompt + user question B
```

系统 prompt 对应的 KV block 可以共享。PagedAttention 中可以通过引用计数管理共享 block：

```text
shared prefix block: ref_count = 2
```

当某个请求继续生成新 token 时，只为新增部分分配新 block。共享前缀不重复存储。

如果需要修改共享区域，则使用 copy-on-write 思想，避免影响其他请求。

## 八、面试表达

PagedAttention 可以这样回答：

1. LLM serving 中 KV Cache 随请求动态增长，长度不同且释放时间不同。
2. 连续显存分配会导致预留浪费和内存碎片。
3. PagedAttention 把 KV Cache 切成固定大小物理 block，请求端维护逻辑块到物理块的 Block Table。
4. Decode kernel 通过 Block Table 找到每个 token 对应的 K/V 地址。
5. 这样可以提升显存利用率，支持 continuous batching 和 prefix sharing。

## 九、总结

PagedAttention 的本质不是新的注意力数学公式，而是 KV Cache 的分页式内存管理。它让请求逻辑上连续、物理上离散，从而解决动态 serving 中最棘手的显存碎片和分配问题。
