---
title: 'LMDeploy Attention 算子源码详解：FA3、Triton Paged Attention 与 TurboMind'
description: '沿 LMDeploy PyTorch Engine 和 TurboMind 两条实现链，讲解 Paged KV 写入、Prefill FlashAttention、Decode Split-K、在线 Softmax、FA3、量化缓存及 Blackwell 上的实际架构选择。'
category: 'Research & Work'
pubDate: '2026-07-29T12:56:00+08:00'
updatedDate: '2026-07-29T12:56:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [LMDeploy 有两套 Attention 执行引擎](#一lmdeploy-有两套-attention-执行引擎)
2. [算子需要完成哪些工作](#二算子需要完成哪些工作)
3. [PyTorch Engine 的调用链](#三pytorch-engine-的调用链)
4. [Blackwell 上如何选择 FA3 或 Triton](#四blackwell-上如何选择-fa3-或-triton)
5. [Attention Metadata 如何描述变长 Batch](#五attention-metadata-如何描述变长-batch)
6. [公共第一步：把新 K/V 写入 Paged Cache](#六公共第一步把新-kv-写入-paged-cache)
7. [Triton Prefill 为什么先 Flatten KV](#七triton-prefill-为什么先-flatten-kv)
8. [Triton Prefill Kernel 的计算过程](#八triton-prefill-kernel-的计算过程)
9. [Blackwell Prefill Tile 如何选择](#九blackwell-prefill-tile-如何选择)
10. [Decode 如何直接读取 Paged KV](#十decode-如何直接读取-paged-kv)
11. [Decode Split-K 的两阶段实现](#十一decode-split-k-的两阶段实现)
12. [GQA、Sliding Window 与特殊数学功能](#十二gqasliding-window-与特殊数学功能)
13. [KV Cache 量化如何进入 Attention](#十三kv-cache-量化如何进入-attention)
14. [FA3 路径实际优化了什么](#十四fa3-路径实际优化了什么)
15. [TurboMind 的完整 Attention 流水](#十五turbomind-的完整-attention-流水)
16. [TurboMind Prefill 与 Decode Kernel](#十六turbomind-prefill-与-decode-kernel)
17. [TurboMind 如何做 Split-K 和归并](#十七turbomind-如何做-split-k-和归并)
18. [Blackwell 上哪些是原生优化，哪些只是兼容](#十八blackwell-上哪些是原生优化哪些只是兼容)
19. [两套引擎的设计取舍](#十九两套引擎的设计取舍)
20. [伪代码、性能分析与验证](#二十伪代码性能分析与验证)
21. [总结](#二十一总结)

## 一、LMDeploy 有两套 Attention 执行引擎

分析 LMDeploy Attention 时，必须先区分两套实现。

### 1.1 PyTorch Engine

主线：

```text
model Attention module
-> backend builder
-> FA3Impl / TritonAttentionImpl / FlashMLAImpl
-> Triton 或外部 FlashAttention Kernel
```

关键实现：

| 模块 | 作用 |
| --- | --- |
| `pytorch/nn/attention.py` | 模型侧统一 Attention 接口 |
| `backends/cuda/attention/__init__.py` | 选择 FA3、MLA 或 Triton |
| `backends/cuda/attention/default.py` | 默认 Prefill/Decode 调度 |
| `backends/cuda/attention/fa3.py` | FlashAttention-3 包装 |
| `kernels/cuda/fill_kv_cache.py` | 新 K/V 写入 Paged Cache |
| `kernels/cuda/flatten_kv_cache.py` | Prefill 前把 Paged KV 展平 |
| `kernels/cuda/flashattention.py` | Triton Varlen Prefill |
| `kernels/cuda/pagedattention.py` | Triton Decode Split-K |

### 1.2 TurboMind

主线：

```text
UnifiedAttentionLayer
-> QKV Projection / QK Norm
-> KV Process
-> dispatchAttention / dispatchDecoding
-> C++ Template Kernel Registry
```

关键实现：

| 模块 | 作用 |
| --- | --- |
| `unified_attention_layer.cc` | Prefill/Decode 编排和 Stream 并行 |
| `attention.cu` | Prefill Dispatcher |
| `decoding.cu` | Decode Dispatcher |
| `registry.cu` | 按架构、Head Dim、Dtype、量化选择 Kernel |
| `attention_template.h` | Prefill Grid 与 Split-K |
| `decoding_template.h` | Decode Grid 与 Split-K |
| `mainloop_sm80.h` | `cp.async` 多 Stage K/V Pipeline |
| `attention_universal.h` | RoPE、KV 更新、Mask、Softmax 和输出 |

两套引擎都实现 Paged KV Attention，但 Kernel、Metadata 和架构优化程度不同。

## 二、算子需要完成哪些工作

Attention 数学：

```text
S = Q K^T * scale
P = softmax(S + mask)
O = P V
```

推理算子还要负责：

- Prefill 和 Decode 两种 Shape。
- 变长 Batch。
- MHA/MQA/GQA。
- Paged KV Cache 的写入和读取。
- RoPE/ALiBi。
- Sliding Window。
- Logit Softcapping。
- Attention Sink。
- FP8/INT8/INT4/TurboQuant KV。
- Speculative Decode 的多 Token Query。
- Tensor Parallel 后的本地 Head 数。

所以 LMDeploy 的 `Attention.forward()` 接口同时接收：

```text
query, key, value
k_cache, v_cache
block_offsets
q_seqlens, kv_seqlens
quant metadata
```

它不是一个只接收 Q/K/V 的训练算子。

## 三、PyTorch Engine 的调用链

模型层 `Attention` 在初始化时先按 Tensor Parallel 切本地 Head：

```python
num_heads, num_kv_heads = update_num_heads_for_tp(
    num_heads,
    num_kv_heads,
)
```

然后向 CUDA Backend 请求：

```python
builder = backend.get_layer_impl_builder(
    OpType.PagedAttention,
)
impl = builder.build(...)
```

CUDA Backend 返回：

```text
TritonAttentionBuilder
```

Builder 再按能力选择：

```text
use_flash_mla
-> FlashMLAImpl

FA3 可用且功能满足
-> FA3Impl

否则
-> TritonAttentionImpl
```

源码中的分派非常直接：

```python
enable_fa3 = _enable_fa3(
    alibi,
    learnable_sink,
    block_sparse_size,
    head_size,
)

if use_flash_mla:
    return FlashMLAImpl(use_fa3=use_fa3, **common_args)
elif enable_fa3:
    return FA3Impl(**common_args)
else:
    return TritonAttentionImpl(
        block_sparse_size=block_sparse_size,
        **common_args,
    )
```

### 3.1 Forward 主干

`TritonAttentionImpl.forward()`：

```python
max_q_len = self._get_max_q_seqlen(
    query,
    attn_metadata,
)

if key is not None and value is not None:
    self._fill_kv_cache_impl(
        key,
        value,
        k_cache=k_cache,
        v_cache=v_cache,
        attn_metadata=attn_metadata,
        max_q_seqlen=max_q_len,
    )

if attn_metadata.is_decoding:
    return self._forward_decoding(...)
else:
    return self._forward_prefill(...)
```

模型侧 `Attention.forward()` 会在进入 `impl.forward()` 前完成 ALiBi Slope 的懒初始化。

因此无论 Prefill 还是 Decode，都遵循：

```text
先追加本轮新 K/V
再执行 Attention
```

## 四、Blackwell 上如何选择 FA3 或 Triton

### 4.1 FA3 可用条件

LMDeploy 尝试导入 FlashAttention-3，并检查：

```text
Compute Capability Major >= 8
CUDA >= 12.3
flash-attn 接口可用
```

具体 Layer 还要求：

```text
不使用 ALiBi
不使用 Learnable Sink
block_sparse_size == 1
head_size <= 256
```

满足时选择：

```text
FA3Impl
```

### 4.2 Triton 回退

以下情况回到 `TritonAttentionImpl`：

- 没有安装 FA3。
- ALiBi。
- Learnable Sink。
- Block Sparse。
- Head Dim 超出 FA3 Builder 限制。

### 4.3 Blackwell 是否有独立 Backend

没有名为 `BlackwellAttentionImpl` 的独立类。

Blackwell 优化分散在：

- FA3 外部 Kernel 的架构实现。
- Triton Prefill 的 Blackwell Tile 配置。
- Device Properties 中的 SM/Warps 信息。

这与 TensorRT-LLM 的 SM100 TRTLLM-Gen 专用 Backend 不同。

## 五、Attention Metadata 如何描述变长 Batch

`CudaOpsBackend.update_step_context()` 构造：

```text
q_seqlens
kv_seqlens
cu_seqlens_q
cu_seqlens_k
q_start_loc
kv_start_loc
block_offsets
quant_policy
```

### 5.1 Cumulative Sequence Length

若 Query Length：

```text
[3, 1, 5]
```

则：

```text
cu_seqlens_q = [0, 3, 4, 9]
```

第 `i` 个请求的 Flatten 区间：

```text
[cu_seqlens_q[i], cu_seqlens_q[i+1])
```

### 5.2 Block Offsets

```text
block_offsets[batch, logical_block]
    = physical_block_id
```

用于从逻辑 Token Position 找到 Paged KV 的物理 Block。

### 5.3 Prefill 与 Decode

Prefill 额外计算：

```text
kv_start_loc
kv_flatten_size
```

因为默认 Prefill 路径需要把 Paged KV 展开成连续 Varlen Buffer。

Decode 不需要 `kv_start_loc`，Kernel 直接按 Page Table 读取。

## 六、公共第一步：把新 K/V 写入 Paged Cache

`fill_kv_cache()` 的 Grid：

```text
(num_kv_heads, max_num_new_blocks, batch_size)
```

每个 Program 处理：

```text
一个请求
一个 KV Head
一个待写逻辑 Block
```

### 6.1 地址计算

请求历史长度：

```text
history_len = kv_seq_len - q_seq_len
```

当前逻辑 KV Block：

```text
kv_block_id =
    floor(history_len / block_size)
    + local_new_block_id
```

物理 Block：

```text
physical_block =
    block_offsets[batch, kv_block_id]
```

Page 内位置：

```text
page_offset = token_position % block_size
```

### 6.2 Decode 特化

当 `max_q_seq_length == 1`：

```text
max_num_blocks = 1
```

每个请求只写一个 Slot：

```text
page_offset = history_len % block_size
```

避免为整页启动无效线程。

### 6.3 Prefill

Prefill 一次可能跨多个 Block，Kernel 为每个新 Block 计算：

```text
哪些 Token 属于 [history_len, kv_seq_len)
```

只对有效位置加载 K/V 并写缓存。

### 6.4 量化写入

根据 `quant_policy` 分派：

```text
NONE：原 Dtype 写入
FP8：使用标量 Scale 后写 FP8
INT8：每 Token/Head 计算 Scale 和 Zero
INT4：两个 4-bit 值打包到一个 Byte
TurboQuant：Hadamard 旋转 + QJL4 K + INT2 V
```

所以 KV Quantization 是 Cache Write Kernel 的组成部分，不是 Attention 外部独立预处理。

## 七、Triton Prefill 为什么先 Flatten KV

默认 Prefill 路径：

```text
Paged KV Cache
-> flatten_kv_cache
-> 连续 Varlen K/V
-> Triton FlashAttention
```

### 7.1 Flatten 做什么

每个 Program 根据：

```text
batch
logical_page
head
```

查询物理 Block：

```text
physical = block_offsets[batch, logical_page]
```

然后复制到：

```text
flatten_k / flatten_v
```

若 KV Cache 是低精度，还在 Flatten 时反量化到 Query Dtype。

### 7.2 Layout

默认 Triton Prefill 使用：

```text
HSD = [head, sequence, dim]
```

FA3 Prefill 使用：

```text
SHD = [sequence, head, dim]
```

Flatten Kernel 负责产生后端要求的 Layout。

### 7.3 为什么要付出一次复制

连续 Varlen K/V 让 Prefill Kernel：

- 使用规则的 Block Pointer。
- 避免内层循环频繁查 Page Table。
- 更容易获得合并访存。

代价是：

- 额外读一次 Paged KV。
- 额外写一次 Flatten Buffer。
- 占用临时显存。

Context 较短或 Prefix Cache 命中较少时，这个成本可能明显。

## 八、Triton Prefill Kernel 的计算过程

Grid：

```text
(
  ceil(max_q_len / BLOCK_M),
  num_query_heads,
  batch_size
)
```

一个 Program 处理：

```text
一个请求
一个 Query Head
一块 Query Token
```

### 8.1 GQA Head 映射

```text
kv_group_num = num_query_heads / num_kv_heads
kv_head = query_head / kv_group_num
```

多个 Query Head 复用同一个 K/V Head。

### 8.2 加载 Query

根据 `cu_seqlens_q` 得到该请求的起点和长度，再加载：

```text
Q tile: [BLOCK_M, head_dim]
```

### 8.3 K/V Block Pointer

连续 K：

```text
[head_dim, kv_len]
```

连续 V：

```text
[kv_len, v_head_dim]
```

每轮前进 `BLOCK_N` 个 KV Token。

### 8.4 Causal Mask

Prefill 可能带历史 KV：

```text
history_len = kv_len - q_len
```

Query Tile 内第 `i` 个 Token 的全局位置：

```text
history_len + query_offset_i
```

它只能看到：

```text
key_position <= global_query_position
```

### 8.5 在线 Softmax

内层 `_prefill_fwd_inner()` 维护：

```text
m_i：Running Max
l_i：Running Sum
acc：Running PV Numerator
```

对每个 KV Tile：

```text
qk = Q @ K^T
qk = scale + mask + alibi + softcap

m_new = max(m_old, rowmax(qk))
p = exp2(qk - m_new)
alpha = exp2(m_old - m_new)

l_new = alpha * l_old + rowsum(p)
acc_new = alpha * acc_old + p @ V
```

最后：

```text
O = acc / l
```

Score 和 Probability 不写入 HBM。

## 九、Blackwell Prefill Tile 如何选择

Triton Prefill 根据 Compute Capability 选择 Tile。

当 Major >= 10 时进入 `_kernel_meta_sm12x()`。虽然函数名是 `sm12x`，当前分支注释明确也覆盖 B200/B100。

典型选择：

| Head Dim | `BLOCK_M` | `BLOCK_N` | Warps | Stages |
| ---: | ---: | ---: | ---: | ---: |
| `<=128` | 128 | 64 或 128 | 8 | 3 |
| `<=256` | 64 | 64 或 128 | 8 | 3 |
| `<=512` | 32 或 64 | 64 | 4 | 2 |
| `>512` | 32 | 32 或 64 | 4 | 2 |

实际选择代码：

```python
if capability_major < 8:
    meta = kernel_meta_sm7x(...)
elif capability_major < 9:
    meta = kernel_meta_sm8x(...)
elif capability_major < 10:
    meta = kernel_meta_sm9x(...)
else:
    meta = kernel_meta_sm12x(...)
```

`shared_kv` 表示 K/V 共用底层数据时，可使用更大的 KV Tile。

### 9.1 Tile 的含义

```text
BLOCK_M：一次处理多少 Query Token
BLOCK_N：一次扫描多少 KV Token
num_warps：一个 Program 的线程并行度
num_stages：K/V 预取 Pipeline 深度
```

Head Dim 增大后，每个 Tile 的寄存器和 Shared Memory 需求上升，所以 Query Tile 缩小。

### 9.2 这是否等于 Blackwell 专用 FMHA

不等于 TensorRT-LLM TRTLLM-Gen 那种显式 TMEM/TMA Kernel。

这里的源码级 Blackwell 特化主要是：

- Tile Size。
- Warp 数。
- Pipeline Stage。

最终指令由 Triton 编译器生成。该文件没有显式规划 TMEM Column、CTA Cluster 或 2-CTA UMMA。

## 十、Decode 如何直接读取 Paged KV

Decode 不先 Flatten：

```text
Query
-> page_table
-> Paged K/V
-> Split-K Attention
-> Reduction
```

入口：

```python
flash_attn_with_kvcache(...)
```

该函数明确标注：

```text
decoding-only
```

### 10.1 Head 与 Token 组织

若每请求验证 `seq_len` 个 Token：

```text
HEADS_PER_REQ = kv_group_num * seq_len
```

Kernel 把：

```text
Query Head Group
Speculative Query Token
```

合并到一个 Tile 维度。

### 10.2 Page 遍历

对每个 KV Tile：

```python
physical_block = page_table[batch, logical_block]
k_tile = k_cache[physical_block]
v_tile = v_cache[physical_block]
```

然后执行：

```text
QK
Mask/ALiBi/Softcap
Online Softmax
PV
```

### 10.3 Blackwell 的 Decode 配置

当前 Decode Kernel 只有：

```text
default
SM8x
SM9x
```

Major >= 9，包括 Blackwell，复用 `_kernel_meta_sm9x()` 的 Warp/Stage 启发式。

设备属性表单独包含 SM100/SM101/SM120 的：

```text
warps_per_sm
blocks_per_sm
multi_processor_count
```

因此 Split-K 数量会感知 Blackwell 的 SM 数和理论并发，但 Decode Kernel 没有独立 SM100 TMEM/TMA 实现。

## 十一、Decode Split-K 的两阶段实现

单 Token Decode 的 Query 维度太小。LMDeploy 沿 KV Length 拆分。

### 11.1 Split 数选择

`_get_split_k()` 估算：

```text
每个 SM 可驻留 CTA 数
总 CTA 容量
当前 Head Grid
Batch Size
SM 数
```

再选择 2 的幂：

```text
SPLIT_K >= 4
```

并限制不超过与 SM 数接近的上限。

核心启发式：

```python
cta_per_device = num_sm * cta_per_sm

split_k = ceil_div(
    cta_per_device // head_grid,
    next_power_of_2(batch_size),
)
split_k = previous_power_of_2(split_k)
split_k = max(min(split_k, max_split), 4)
```

### 11.2 第一阶段 Grid

```text
(
  query_head_group_tiles,
  SPLIT_K,
  batch_size
)
```

每个 CTA 只处理 KV 序列的一段：

```text
loop_start
loop_end
```

输出 Float32：

```text
partial_acc
partial_max
partial_sum
```

临时 Buffer：

```text
[num_tokens, num_heads, SPLIT_K, v_dim + 2]
```

最后两个元素保存 `m_i` 和 `l_i`。

### 11.3 第二阶段 Grid

```text
(num_heads, num_tokens)
```

一个 Program 合并同一 Query Head 的所有 Split。

全局最大值：

```text
m = max(m_k)
```

修正：

```text
alpha_k = exp2(m_k - m)
```

输出：

```text
O =
  sum_k alpha_k * partial_acc_k
  /
  sum_k alpha_k * partial_sum_k
```

这与完整序列一次 Softmax 数学等价。

## 十二、GQA、Sliding Window 与特殊数学功能

### 12.1 GQA

Prefill 和 Decode 都使用：

```text
kv_head = q_head // kv_group_num
```

Decode 的一个 CTA 可以同时处理多个属于同一 KV Head 的 Query Head，复用加载的 K/V。

### 12.2 Sliding Window

只扫描：

```text
[current_position - window_size, current_position]
```

Kernel 把 Loop Start 对齐到 KV Block 边界，再通过 Mask 去掉窗口外 Token。

### 12.3 ALiBi

每个 Query Head 有独立 Slope：

```text
bias = -abs(relative_position) * slope
```

在 Softmax 前加到 Logit。

### 12.4 Logit Softcapping

Gemma 类模型可使用：

```text
score =
  cap * tanh(score / cap)
```

限制极端 Logit，提高数值稳定性。

### 12.5 Attention Sink

在最终分母中加入：

```text
exp(sink - max)
```

不对应普通 V Token，因此 Reduction Kernel 单独处理。

## 十三、KV Cache 量化如何进入 Attention

### 13.1 FP8

Cache Write：

```text
q = clamp(x / scale, fp8_min, fp8_max)
```

Decode Load：

```text
x = fp8_value * scale
```

Scale 当前作为每层 K/V 标量传入。

### 13.2 INT8

每个 Token/Head 计算：

```text
scale = (max - min) / 255
zero = -min / scale
```

Cache 同时保存 Payload 与 Scale/Zero。

### 13.3 INT4

两个 4-bit 值打包到一个 Byte：

```text
packed = low | (high << 4)
```

Decode Kernel 内解包并反量化。

### 13.4 TurboQuant

当前实现：

```text
K：Hadamard Rotate + QJL4
V：Hadamard Rotate + 2-bit Centroid
```

Attention 前先把 Query 旋转到同一域；最终 Reduction Kernel 融合逆 Hadamard，避免单独生成完整中间输出。

### 13.5 为什么 Prefill 和 Decode 路径不同

Prefill Flatten 时直接反量化成连续高精度 K/V，再运行 Varlen Attention。

Decode 直接从量化 Paged Cache 读取，在内层 Tile 中反量化。这样可保留 KV 带宽收益。

## 十四、FA3 路径实际优化了什么

`FA3Impl` 继承默认实现的 Cache Write，但替换部分 Attention Kernel。

### 14.1 Prefill

```text
Paged KV
-> Flatten 为 SHD
-> flash_attn_varlen_func_v3
```

真正的 FA3 Kernel 位于外部 `flash-attn` 包，LMDeploy 仓库只保留包装和 Dispatch。

### 14.2 Speculative Decode

当：

```text
max_q_seqlen > 1
```

使用：

```text
flash_attn_with_kvcache_v3
```

直接接收：

- Paged KV。
- Page Table。
- Cache Length。
- Scheduler Metadata。

它避免把多 Token Decode 拆成多个单 Token 调用。

### 14.3 标准单 Token Decode

当：

```text
max_q_seqlen == 1
```

当前 `FA3Impl` 仍调用继承的：

```text
Triton paged_attention_fwd
```

所以“Builder 选择 FA3”不代表所有阶段都进入 FA3 Kernel。

### 14.4 限制

FA3 路径当前不覆盖：

- ALiBi。
- Learnable Sink。
- Block Sparse。
- TurboQuant 的多 Token Speculative Decode。

这些情况回退或显式报错。

## 十五、TurboMind 的完整 Attention 流水

`UnifiedAttentionLayer::Forward()`：

```text
Hidden State
-> QKV Linear
-> optional QK RMSNorm
-> core_attention
-> optional Output Gate
-> Output Projection
```

Mixed Batch 的核心编排代码可概括为：

```cpp
if (has_decode && has_prefill) {
    prefill_stream = aux_stream;
    record_event(qkv_ready, main_stream);
    wait_event(prefill_stream, qkv_ready);
}

if (has_prefill) {
    invokeProcessKV(params);
    invokeFlattenKV(params);
    dispatchAttention(params);
}

if (has_decode) {
    dispatchDecoding(params);
}

if (has_decode && has_prefill) {
    record_event(prefill_done, prefill_stream);
    wait_event(main_stream, prefill_done);
}
```

### 15.1 QKV Layout

普通 GQA：

```text
[Q Heads | K Heads | V Heads]
```

每个 Tensor Parallel Rank 只持有本地 Head。

### 15.2 Batch 分类

`Setup()` 统计：

```text
decode.n
prefill.n
q_sum/q_max
k_sum/k_max
```

Batch 被预先按 Decode 和 Prefill 排列。

### 15.3 Mixed Batch 双 Stream

若两类请求同时存在：

```text
Main Stream：Decode
Aux Stream：Prefill
```

用 Event 保证：

```text
QKV Projection 完成
-> Prefill Aux Stream 开始
-> Decode Main Stream 并发执行
-> 输出投影前重新汇合
```

这能减少长 Prefill 对 Decode 的直接串行阻塞。

## 十六、TurboMind Prefill 与 Decode Kernel

### 16.1 Prefill

流程：

```text
invokeProcessKV_v2
-> 写 Paged KV / 应用 RoPE
-> invokeFlattenKV_v2
-> 生成连续 K/V
-> dispatchAttention
```

源码保留了：

```text
TODO: sm80 跳过 Flatten
```

说明当前 Prefill 仍支付 Paged 到 Linear KV 的整理成本。

Prefill SM80 Head Dim 128 实例：

```text
CTA_Q = 64
CTA_S = 64
MMA = 16x8x16
Stages = 2
```

### 16.2 Decode

Decode 直接使用：

```text
BlockIterator
```

读取 Paged KV，不做 Flatten。

`AttentionUniversal` 的 Prologue 同时：

- 加载 Q/K/V。
- 加 Bias。
- 对 Q/K 应用 RoPE。
- 在 Decode 中把新增 K/V 写入 Cache。
- 处理 KV Quantization 参数。

因此 TurboMind Decode 的 Cache Write 与 Attention Kernel 更紧密。

### 16.3 Query Group Kernel

以 Head Dim 128 为例：

```text
GQA Ratio <= 2：
  SIMT Kernel

GQA Ratio > 2：
  MMA 8x16x16 Kernel
  Query Head Tile 取 8 或 16
```

小 GQA Ratio 时，Tensor Core Tile 容易浪费，SIMT 反而更合适。

## 十七、TurboMind 如何做 Split-K 和归并

Prefill 和 Decode Template 都会计算：

```text
tile_count = ceil(valid_kv_len / CTA_S)
max_split_count = min(max_kv_splits, tile_count)
```

再根据：

```text
当前 Grid Size
SM Count
Active CTA
```

选择 Split Count。

### 17.1 主 Kernel

每个 Split 维护：

```text
FragM：局部最大值
FragL：局部分母
FragO：局部输出分子
```

在线 Softmax 使用 `exp2f` 更新。

### 17.2 Partial Store

当 `split_cnt > 1`：

```text
Partial M/L/O
-> Global Workspace
```

### 17.3 ReduceV3

第二阶段按相同 Softmax 修正公式归并：

```text
global_m = max(partial_m)
alpha = exp(partial_m - global_m)

global_l = sum(alpha * partial_l)
global_o = sum(alpha * partial_o) / global_l
```

Context Parallel 时，还会把不同 CP Rank 的 Partial 一起纳入归并。

## 十八、Blackwell 上哪些是原生优化，哪些只是兼容

这是 LMDeploy 源码中最容易误判的部分。

### 18.1 PyTorch Triton Prefill

有 Blackwell 分支：

```text
Major >= 10
-> _kernel_meta_sm12x
```

它调整 Tile、Warp、Stage，但没有显式 Blackwell TMEM/CTA Cluster 实现。

结论：

```text
有架构感知调优
不是专用 SM100 FMHA Pipeline
```

### 18.2 PyTorch Triton Decode

Blackwell 使用：

```text
SM9x Kernel Meta
```

Split-K Heuristic 会读取 SM100/SM120 Device Properties。

结论：

```text
调度参数部分感知 Blackwell
核心 Decode Kernel 仍是通用 Triton 实现
```

### 18.3 FA3

若外部 FA3 包提供当前 GPU 的实现，Prefill 和多 Token Decode 可进入架构优化 Kernel。

LMDeploy 只检查：

```text
SM80+
CUDA 12.3+
接口存在
```

实际 Blackwell 指令级优化由安装的 FA3 版本决定。

### 18.4 TurboMind

Kernel Registry 中显式实例只有：

```text
SM70
SM75
SM80
```

`Sm80` 的兼容范围定义为：

```text
arch >= 800
```

所以 Blackwell 会选择 SM80 Template：

```text
cp.async
MMA 16x8x16 / 8x16x16
Shared Memory Pipeline
```

没有 SM100 专用：

- TMA Pipeline。
- TMEM。
- 2-CTA UMMA。
- CTA Cluster FMHA。

结论：

```text
可以向上运行
但不是 Blackwell 原生 Attention Kernel
```

## 十九、两套引擎的设计取舍

| 维度 | PyTorch Engine | TurboMind |
| --- | --- | --- |
| 实现语言 | Python + Triton + 外部 FA3 | C++ Template + CUDA |
| Prefill | Flatten + Varlen Attention | Process KV + Flatten + 自研 Kernel |
| Decode | Triton Paged Split-K | BlockIterator + 自研 Split-K |
| FA3 | 可选 | 不走该 Python FA3 包装 |
| KV Quant | FP8/INT8/INT4/TurboQuant | INT8/INT4 |
| Mixed Batch | 同一 Forward 分阶段 | Decode/Main 与 Prefill/Aux Stream 并发 |
| Blackwell | Tile 感知/外部 FA3 | SM80 Template 向上兼容 |
| 扩展性 | Triton 修改快 | Template 实例和 Registry 更严格 |

### 19.1 PyTorch Engine

优点：

- Triton 便于快速支持新 Shape 和量化。
- FA3 可直接接入。
- Kernel 逻辑易读。

代价：

- Prefill Flatten 产生额外流量。
- 部分 Blackwell 路径仍沿用通用配置。
- 外部 Kernel 版本影响能力。

### 19.2 TurboMind

优点：

- KV Iterator、RoPE、量化和 Attention 深度融合。
- Decode Query Group 有 SIMT/MMA 专门实例。
- Mixed Batch 可用双 Stream 并发。

代价：

- Kernel 实例组合多。
- 新 Head Dim/架构需要新增模板实例。
- 当前没有 Blackwell 原生 Attention Traits。

## 二十、伪代码、性能分析与验证

### 20.1 PyTorch Engine

```python
def lmdeploy_attention(q, k, v, cache, meta):
    if k is not None:
        write_paged_kv(
            k,
            v,
            cache,
            meta.block_offsets,
            meta.quant_policy,
        )

    if meta.is_decoding:
        if use_fa3 and meta.q_len > 1:
            return fa3_paged_decode(q, cache, meta)
        return triton_split_k_decode(q, cache, meta)

    flat_k, flat_v = flatten_paged_kv(cache, meta)
    if use_fa3:
        return fa3_varlen_prefill(q, flat_k, flat_v, meta)
    return triton_varlen_prefill(q, flat_k, flat_v, meta)
```

### 20.2 TurboMind

```cpp
qkv = qkv_projection(hidden);
apply_optional_qk_norm(qkv);

if (has_prefill) {
    process_and_cache_kv(qkv);
    flat_kv = flatten_kv_cache();
    launch_prefill_attention(qkv, flat_kv);
}

if (has_decode) {
    launch_decode_attention_with_block_iterator(qkv, paged_kv);
}

wait_prefill_decode();
output = output_projection(attention_output);
```

### 20.3 Profile 时看什么

Prefill：

```text
fill_kv_cache 时间
flatten_kv_cache 时间
Attention 主 Kernel 时间
HBM Read/Write
BLOCK_M/BLOCK_N
FA3 是否真正启用
```

Decode：

```text
SPLIT_K
第一阶段 Paged Attention
第二阶段 Reduction
Partial Buffer 大小
KV Cache Dtype
GQA Ratio
Context Length
```

TurboMind：

```text
Prefill/Decode 是否并发
实际 Registry Kernel
SIMT 或 MMA
Split Count
ReduceV3 占比
FlattenKV 占比
```

### 20.4 正确性矩阵

至少测试：

- Prefill only。
- Decode only。
- Mixed Batch。
- MHA/MQA/GQA。
- 不同 Head Dim。
- Prefix Cache。
- Sliding Window。
- 不同 Block Size。
- FP16/BF16。
- FP8/INT8/INT4/TurboQuant。
- 单 Token 与 Multi-token Spec Decode。
- FA3 开启/关闭。
- PyTorch Engine/TurboMind 对齐。

比较：

- Attention Output。
- KV Cache 物理内容。
- Logits。
- 长上下文误差。
- 不同 Batch 排列的一致性。

## 二十一、总结

LMDeploy Attention 的核心不是一个 Kernel，而是：

```text
统一模型接口
-> Backend 选择
-> 变长 Metadata
-> Paged KV 写入
-> Prefill 或 Decode 专用路径
-> 在线 Softmax
-> 必要时 Split-K 归并
```

PyTorch Engine：

1. 新 K/V 先写 Paged Cache。
2. Prefill 把 Paged KV 展平，再进入 Triton 或 FA3 Varlen Attention。
3. 单 Token Decode 直接读取 Page Table，用 Triton Split-K 恢复并行度。
4. Multi-token Spec Decode 在 FA3 可用时走原生 Paged KV 接口。
5. Blackwell Prefill 有 Tile 特化，Decode 仍主要复用 SM9x 配置。

TurboMind：

1. QKV、RoPE、KV Cache 和 Attention 结合更紧。
2. Prefill 仍会 Flatten KV，Decode 使用 Block Iterator 直接读页。
3. GQA Ratio 决定 SIMT 或 Tensor Core Kernel。
4. Prefill 和 Decode 可在两个 Stream 并发。
5. Blackwell 当前使用 SM80 Template 的向上兼容路径，没有 SM100 专用 Attention Kernel。

因此在 Blackwell 上评估 LMDeploy，不能只问“是否支持 Blackwell”，而应继续确认：

```text
实际选择了 FA3 还是 Triton
Prefill 是否被 Flatten 开销限制
Decode 的 SPLIT_K 是否合理
TurboMind 是否受 SM80 Template 限制
KV 量化是否真正减少了主循环带宽
```
