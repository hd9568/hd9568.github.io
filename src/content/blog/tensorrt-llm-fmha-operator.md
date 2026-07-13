---
title: 'TensorRT-LLM FMHA 算子源码解读：从 Attention 到 SM10x TRTLLM-Gen Backend'
description: '结合 TensorRT-LLM 源码讲解 FMHA 算子、FlashAttention 思想、paged KV cache、GQA、context/generation 阶段、SM100/SM103 trtllm-gen 后端和安全回退机制。'
category: 'Research & Work'
pubDate: '2026-07-13T16:55:00+08:00'
updatedDate: '2026-07-13T16:55:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [FMHA 是什么](#二fmha-是什么)
3. [为什么推理框架需要专门的 FMHA backend](#三为什么推理框架需要专门的-fmha-backend)
4. [Context 和 Generation 两个阶段](#四context-和-generation-两个阶段)
5. [Paged KV Cache 和 GQA](#五paged-kv-cache-和-gqa)
6. [TensorRT-LLM 源码入口](#六tensorrt-llm-源码入口)
7. [Python attention backend：trtllm_gen.py](#七python-attention-backendtrtllm_genpy)
8. [C++ dispatcher：fmhaDispatcher](#八c-dispatcherfmhadispatcher)
9. [参数桥接：从框架 metadata 到 kernel params](#九参数桥接从框架-metadata-到-kernel-params)
10. [为什么 SM10x / B200 需要 TRTLLM-Gen FMHA](#十为什么-sm10x--b200-需要-trtllm-gen-fmha)
11. [简化版 FMHA 伪代码](#十一简化版-fmha-伪代码)
12. [工程落地 checklist](#十二工程落地-checklist)
13. [面试表达](#十三面试表达)
14. [总结](#十四总结)

## 一、核心结论

FMHA 是 Fused Multi-Head Attention。

它的目标是把 attention 中原本分散的步骤融合起来：

```text
QK^T
scale
mask
softmax
PV
```

尽量避免把中间矩阵完整写回 HBM。

对大模型推理来说，FMHA 还必须处理：

- GQA / MQA。
- RoPE。
- paged KV cache。
- context / decode 两种阶段。
- FP16 / BF16 / FP8 / NVFP4。
- sliding window。
- chunked attention。
- spec decoding。
- 不同 GPU 架构的 kernel 支持。

TensorRT-LLM 中有传统 FMHA v2，也有面向 Blackwell SM100/SM103 的 TRTLLM-Gen FMHA。后者更适合 B200/SM10x 上的新 kernel 路径。

## 二、FMHA 是什么

普通 attention 公式：

```text
S = Q K^T / sqrt(D)
P = softmax(S + mask)
O = P V
```

朴素实现会产生中间矩阵：

```text
S: [B, H, Q_len, K_len]
P: [B, H, Q_len, K_len]
```

长上下文下这个矩阵很大。

FMHA 的思想是：

```text
用 tile 分块计算 QK^T、softmax 和 PV；
中间 score/prob 尽量留在寄存器或 shared memory；
不把完整 attention matrix 写回 HBM。
```

这和 FlashAttention 的思想类似。

## 三、为什么推理框架需要专门的 FMHA backend

训练 attention 和推理 attention 不一样。

推理中有两种典型阶段：

```text
Prefill / Context:
  Q_len 较长，K_len 较长。

Decode / Generation:
  Q_len 通常是 1，但 K_len 是历史上下文长度。
```

Decode 阶段关注：

- KV cache layout。
- 每步写入新 K/V。
- 每步读取历史 K/V。
- PagedAttention。
- batch 中每个请求长度不同。
- GQA 中 Q heads 和 KV heads 不同。

所以推理框架需要专门 backend，而不是直接调用训练用 attention。

## 四、Context 和 Generation 两个阶段

### Context 阶段

Context 阶段也叫 prefill。

```text
Q: [prompt_len, num_heads, head_dim]
K: [prompt_len, num_kv_heads, head_dim]
V: [prompt_len, num_kv_heads, head_dim]
```

特点：

- Q_len 大。
- GPU 并行度高。
- 通常接近矩阵计算。
- 需要把 prompt 的 K/V 写入 KV cache。

### Generation 阶段

Generation 阶段也叫 decode。

```text
Q: [1, num_heads, head_dim]
K_cache: [context_len, num_kv_heads, head_dim]
V_cache: [context_len, num_kv_heads, head_dim]
```

特点：

- Q_len 小。
- K_len 长。
- Memory-bound 更明显。
- 对 paged KV cache 和 GQA 支持要求高。
- kernel launch overhead 也更明显。

## 五、Paged KV Cache 和 GQA

Paged KV Cache 把 KV cache 切成固定 token block。

```text
logical block -> physical block
```

Attention kernel 需要根据 block table 找到每个 token 的 K/V 地址。

GQA 中：

```text
num_heads != num_kv_heads
num_heads_q_per_kv = num_heads / num_kv_heads
```

一个 KV head 服务多个 Q heads。

FMHA kernel 必须知道：

```text
num_heads
num_kv_heads
num_heads_q_per_kv
head_dim
tokens_per_block
block_table
sequence lengths
```

否则无法正确读取 KV cache。

## 六、TensorRT-LLM 源码入口

本文参考本地源码：

```text
inference-framework/TensorRT-LLM/
```

关键文件：

```text
tensorrt_llm/_torch/attention_backend/trtllm_gen.py
tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py
cpp/tensorrt_llm/kernels/fmhaDispatcher.h
cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp
cpp/tensorrt_llm/thop/trtllmGenQKVProcessOp.cpp
cpp/tensorrt_llm/thop/attentionOp.cpp
```

高层链路：

```text
Python Attention Backend
  -> C++/THOP preprocessing
  -> FlashInfer / TRTLLM-Gen FMHA kernel
  -> fallback to standard thop attention if unsupported
```

## 七、Python attention backend：trtllm_gen.py

`trtllm_gen.py` 文件开头已经说明：

```text
TrtLLM-Gen Attention Backend
Uses flashinfer's trtllm-gen kernels.
Blackwell architecture: SM100/SM103.
```

它的支持检查在 `TrtllmGenSupportChecker`。

关键限制包括：

```python
sm = get_sm_version()
if not is_sm_100f(sm):
    return False, f"trtllm-gen requires SM100 or SM103. Current: SM{sm}."
```

也就是说：

```text
非 Blackwell SM100/SM103 不走 trtllm-gen。
```

GQA 检查：

```python
assert num_heads > 0
assert num_kv_heads > 0
if num_heads % num_kv_heads != 0:
    return False
```

Generation 阶段还限制：

```python
heads_ratio = num_heads // num_kv_heads
if not is_mla_enable and heads_ratio > MAX_HEADS_RATIO_GENERATION:
    return False
```

Paged KV cache 检查：

```python
if tokens_per_block & (tokens_per_block - 1) != 0:
    return False
if tokens_per_block not in {16, 32, 64}:
    return False
```

这些检查很工程化：不是所有 attention 配置都能走 Gen FMHA。

## 八、C++ dispatcher：fmhaDispatcher

C++ 侧入口在：

```text
cpp/tensorrt_llm/kernels/fmhaDispatcher.h
cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp
```

`FmhaDispatcher` 中有两个 runner：

```cpp
// Runner for fmha v2 kernels (for SM <= 90)
UniqPtrWNullCopy<kernels::FusedMHARunnerV2> mFMHARunner;

// Runner for trtllm-gen fmha kernels (for SM == 100)
UniqPtrWNullCopy<kernels::TllmGenFmhaRunner> mTllmGenFMHARunner;
```

构造函数中决定是否使用 TRTLLM-Gen：

```cpp
mUseTllmGen(tensorrt_llm::common::isSM100Family()
            && fixedParams.headSize != 72)
```

如果是 SM100 family，就创建：

```cpp
mTllmGenFMHARunner.reset(
    new TllmGenFmhaRunner(...)
);
```

否则使用：

```cpp
mFMHARunner.reset(new FusedMHARunnerV2(fixedParams));
```

这就是架构相关 fallback。

## 九、参数桥接：从框架 metadata 到 kernel params

TRTLLM-Gen FMHA runner 需要大量参数。

在 `fmhaDispatcher.cpp` 中：

```cpp
tllmRunnerParams.mNumHeadsQ = mFixedParams.numQHeads;
tllmRunnerParams.mNumHeadsKv = mFixedParams.numKvHeads;
tllmRunnerParams.mNumHeadsQPerKv =
    tllmRunnerParams.mNumHeadsQ / tllmRunnerParams.mNumHeadsKv;
```

Paged KV cache：

```cpp
if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_PAGED_KV) {
    qkvLayout = kernels::QkvLayout::PagedKv;
    auto pagedKvCache = runnerParams.pagedKvCache.copyKVBlockArrayForContextFMHA();
    kvPoolPtr = pagedKvCache.mPrimaryPoolPtr;
    kvPageIdxPtr = reinterpret_cast<int const*>(pagedKvCache.data);
    maxBlocksPerSeq = pagedKvCache.mMaxBlocksPerSeq;
    numTokensPerBlock = pagedKvCache.mTokensPerBlock;
}
```

传给 runner：

```cpp
tllmRunnerParams.kvPtr = kvPoolPtr;
tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<int const*>(kvPageIdxPtr);
tllmRunnerParams.mMaxNumPagesPerSeqKv = maxBlocksPerSeq;
tllmRunnerParams.mNumTokensPerPage = numTokensPerBlock;
```

序列长度：

```cpp
tllmRunnerParams.mMaxSeqLenQ = runnerParams.qSeqLen;
tllmRunnerParams.mMaxSeqLenKv = runnerParams.kvSeqLen;
tllmRunnerParams.mSumOfSeqLensQ = runnerParams.totalQSeqLen;
tllmRunnerParams.mSumOfSeqLensKv = runnerParams.totalKvSeqLen;
tllmRunnerParams.seqLensKvPtr = reinterpret_cast<int const*>(runnerParams.kvSeqLenPtr);
```

这就是框架 metadata 到 kernel 参数的桥接。

## 十、为什么 SM10x / B200 需要 TRTLLM-Gen FMHA

B200 / Blackwell 属于 SM100/SM10x 系列。

旧 FMHA kernel 可能主要面向：

```text
SM70 / SM75 / SM80 / SM90
```

而 SM100 有新的硬件能力和更适合的 kernel 生成路径。TRTLLM-Gen FMHA 主要目标是：

- 使用 Blackwell 上更合适的 kernel。
- 支持新的 dtype 组合。
- 支持 paged KV cache。
- 支持 GQA heads ratio。
- 使用 persistent scheduler / TMA 等新硬件能力。

源码中也体现了：

```cpp
// Runner for trtllm-gen fmha kernels (for SM == 100)
TllmGenFmhaRunner
```

如果配置不支持，就必须 fallback：

```cpp
if (!foundKernels) {
    TLLM_LOG_WARNING("Fall back to unfused MHA ...");
}
```

工程上一定要有安全回退，避免非 SM10x 或缺少 kernel 时直接崩溃。

## 十一、简化版 FMHA 伪代码

下面是普通 attention 的伪代码：

```python
def attention(q, k, v, mask):
    score = q @ k.transpose(-1, -2)
    score = score / math.sqrt(q.shape[-1])
    score = score + mask
    prob = softmax(score)
    out = prob @ v
    return out
```

FMHA 的 tile 思路：

```python
for q_tile in Q_tiles:
    m = -inf
    l = 0
    acc = 0

    for kv_tile in KV_tiles:
        score = q_tile @ k_tile.T
        score = apply_mask(score)

        # online softmax update
        m_new = max(m, max(score))
        p = exp(score - m_new)
        acc = acc * exp(m - m_new) + p @ v_tile
        l = l * exp(m - m_new) + sum(p)
        m = m_new

    out_tile = acc / l
```

Paged KV 版本还要在每个 tile 之前通过 block table 找 K/V：

```python
physical_block = block_table[request_id][logical_block]
k_tile = kv_cache_k[physical_block]
v_tile = kv_cache_v[physical_block]
```

GQA 版本还要做：

```python
kv_head = q_head // (num_heads / num_kv_heads)
```

## 十二、工程落地 checklist

FMHA backend 接入时要检查：

### 1. 支持条件

- GPU 架构。
- dtype。
- head size。
- num_heads / num_kv_heads。
- tokens_per_block。
- mask type。
- beam width。
- paged KV cache。
- MLA / MQA / GQA。

### 2. Metadata 桥接

必须正确传递：

- batch size。
- q seq len。
- kv seq len。
- cuQSeqLen。
- cuKvSeqLen。
- block table。
- KV pool pointer。
- page size。
- num pages。
- scale。

### 3. KV update

如果前处理 kernel 要同时做：

- RoPE。
- LogN scaling。
- GQA KV 写入。
- Q-only buffer 生成。
- paged KV metadata 更新。

就必须保证 context/decode 结果和原 backend 对齐。

### 4. 验证

至少要做：

- 单算子 smoke test。
- context 对比。
- decode 对比。
- 端到端 logits 对比。
- token 输出对比。
- unsupported 配置 fallback。

## 十三、面试表达

可以这样解释 FMHA：

```text
FMHA 是 fused multi-head attention，它把 QK、scale、mask、softmax、PV 融合到一个高效 kernel 里，避免把完整 attention score/prob 写回 HBM。
在大模型推理里，FMHA 还要处理 paged KV cache、GQA、不同 seq len、context/decode 两阶段和低精度。
```

如果结合 TensorRT-LLM：

```text
TensorRT-LLM 中 FmhaDispatcher 会根据 GPU 架构和参数选择 FMHA runner。
SM100 family 上可以走 TRTLLM-Gen FMHA runner；否则走 FMHA v2 或 fallback。
trtllm-gen 参数里会显式传 numHeadsQ、numHeadsKv、numHeadsQPerKv、paged KV cache pool、page index、tokens per page、seq lens 等信息。
```

如果问为什么需要 fallback：

```text
因为 Gen FMHA 对架构、head size、dtype、mask、tokens_per_block、beam width 等都有支持范围。
工程上不能假设所有模型都能走新 kernel；必须在 unsupported 时回退到原 attention 路径，保证正确性和可部署性。
```

## 十四、总结

TensorRT-LLM FMHA 的工程主线是：

```text
高层 attention metadata
-> QKV preprocess / KV cache update
-> paged KV 参数桥接
-> FMHA runner 参数组装
-> SM100 TRTLLM-Gen kernel 或旧 FMHA runner
-> fallback
```

SM10x / B200 上接入 TRTLLM-Gen FMHA 的价值在于：

- 使用新架构专用 kernel。
- 更好支持低精度和 paged KV。
- 减少 attention 中间访存。
- 提升 context/decode 阶段性能。

但正确接入比单纯调一个 kernel 更复杂，关键是：

```text
配置透传、metadata 桥接、KV update、GQA 形状、paged cache、验证和安全回退。
```
