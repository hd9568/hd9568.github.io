---
title: 'TensorRT-LLM FMHA 源码详解：Blackwell TRTLLM-Gen、TMEM 与 Paged KV'
description: '沿 TensorRT-LLM Blackwell Attention 调用链，讲解 TRTLLM-Gen 的 QKV 预处理、Paged KV、在线 Softmax、TMA/TMEM、Persistent Scheduler、Multi-CTA KV、低精度和安全回退。'
category: 'Research & Work'
pubDate: '2026-07-13T16:55:00+08:00'
updatedDate: '2026-07-29T12:55:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [实现边界与源码地图](#一实现边界与源码地图)
2. [FMHA 在计算什么](#二fmha-在计算什么)
3. [Blackwell 上的两条 TRTLLM-Gen 路径](#三blackwell-上的两条-trtllm-gen-路径)
4. [Python 快路径如何选择与回退](#四python-快路径如何选择与回退)
5. [Mixed Batch 如何拆成 Context 和 Generation](#五mixed-batch-如何拆成-context-和-generation)
6. [QKV 预处理做了什么](#六qkv-预处理做了什么)
7. [Paged KV Cache 如何交给 Kernel](#七paged-kv-cache-如何交给-kernel)
8. [FMHA 主循环与在线 Softmax](#八fmha-主循环与在线-softmax)
9. [Blackwell 硬件如何映射到 FMHA](#九blackwell-硬件如何映射到-fmha)
10. [Kernel 类型、Tile 与自动选择](#十kernel-类型tile-与自动选择)
11. [Persistent Scheduler 为什么重要](#十一persistent-scheduler-为什么重要)
12. [Generation 的 GQA 与 Multi-CTA KV](#十二generation-的-gqa-与-multi-cta-kv)
13. [Multi-CTA 结果如何正确归并](#十三multi-cta-结果如何正确归并)
14. [FP8 与 NVFP4 路径](#十四fp8-与-nvfp4-路径)
15. [Context、Decode 与 Speculative Decode](#十五contextdecode-与-speculative-decode)
16. [Cubin、NVRTC 与 Kernel Key](#十六cubinnvrtc-与-kernel-key)
17. [不支持时如何回退](#十七不支持时如何回退)
18. [把执行过程压缩成伪代码](#十八把执行过程压缩成伪代码)
19. [性能分析与正确性检查](#十九性能分析与正确性检查)
20. [总结](#二十总结)

## 一、实现边界与源码地图

TensorRT-LLM 的 Attention 不是单个 CUDA Kernel，而是一条由运行时、缓存管理、预处理和架构专用 Kernel 组成的链路。

Blackwell 相关主线可分为以下层次：

| 层次 | 关键实现 | 作用 |
| --- | --- | --- |
| PyTorch Backend | `trtllm.py` | 接收 Q/K/V 与运行时 Metadata，选择执行路径 |
| Blackwell 快路径 | `trtllm_gen.py` | 拆分 Context/Generation，调用 TRTLLM-Gen 包装 |
| QKV 桥接 | `trtllmGenQKVProcessOp.cpp` | RoPE、变长 Metadata、Paged KV 写入和量化 |
| 通用 C++ Attention | `attentionOp.cpp` | 常规 `thop.attention` 入口 |
| Context Dispatcher | `fmhaDispatcher.cpp` | 在 SM100 Family 选择 TRTLLM-Gen，否则选择 FMHA v2 |
| Kernel Runner | `fmhaRunner.cpp` | 构造架构和数据类型专用 Kernel 集合 |
| Kernel 调度 | `fmhaKernels.h` | 自动选择 Tile、加载 Cubin/NVRTC、设置参数并 Launch |
| Blackwell Traits | `KernelTraits.h` | TMA、TMEM、MMA、Pipeline 和 Shared Memory 布局 |
| Split-K 归并 | `fmhaReduction.cu` | 合并多个 KV 分片的 Softmax 统计量与 Partial Output |

需要先说明一个源码边界：

> TRTLLM-Gen 的大量最终 Kernel 以预编译 Cubin 交付。仓库能完整看到调度、参数、数据布局、Traits、归并和部分代码生成接口，但不能把每个 Cubin 当作普通 `.cu` 文件逐行阅读。

因此本文会严格区分：

- 源码中可以直接确认的行为。
- 根据 Kernel Traits 和接口能够确定的数据流。
- 不应凭空猜测的 Cubin 内部指令排布。

## 二、FMHA 在计算什么

单个 Attention Head：

```text
S = Q K^T * scale
P = softmax(S + mask)
O = P V
```

张量形状：

```text
Q: [Q_len, D_qk]
K: [KV_len, D_qk]
V: [KV_len, D_v]
S: [Q_len, KV_len]
O: [Q_len, D_v]
```

朴素实现会把 `S` 和 `P` 写入 HBM：

```text
QK GEMM
-> 写 Score
-> Mask/Softmax
-> 写 Probability
-> PV GEMM
```

这会产生 `O(Q_len * KV_len)` 的中间存储和额外 HBM 流量。

FMHA 的核心是把：

```text
QK
Scale
Mask
Softmax
PV
```

放入同一个分块流水中，使 Score、Probability 和 Softmax 状态尽量停留在片上存储。

### 2.1 Prefill 与 Decode 的差异

Prefill：

```text
Q_len 较大
KV_len 较大
QK/PV 都有较大的矩阵维度
并行度通常足够
```

Decode：

```text
Q_len 通常为 1
KV_len 是完整历史
读取 KV Cache 的成本占比很高
单请求并行度不足
```

同一个公式需要完全不同的 Tile 和调度策略。

## 三、Blackwell 上的两条 TRTLLM-Gen 路径

当前 PyTorch Runtime 中存在两条容易混淆的 Blackwell 路径。

### 3.1 Python 内部快路径

`TrtllmAttention._run()` 可以在环境开关启用时调用：

```text
FlashInferTrtllmGenAttention
```

高层流程：

```text
TrtllmAttention
-> TrtllmGenSupportChecker
-> C++ QKV preprocess
-> FlashInfer TRTLLM-Gen Context/Decode wrapper
-> C++ postprocess
```

该路径默认关闭，需要显式启用：

```bash
export TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1
```

它支持在同一个 Backend 中处理 Context 和 Generation，并直接面向 Paged KV Cache。

### 3.2 常规 `thop.attention` 路径

若快路径关闭或不满足支持条件：

```text
TrtllmAttention
-> thop.attention
-> C++ AttentionOp
```

Context FMHA 进入 `FmhaDispatcher`。构造时：

```cpp
mUseTllmGen =
    isSM100Family()
    && headSize != 72;
```

在 SM100/SM103 上优先创建：

```text
TllmGenFmhaRunner
```

其他架构使用：

```text
FusedMHARunnerV2
```

所以“Python TRTLLM-Gen 快路径未启用”不等于“Blackwell 完全不使用 TRTLLM-Gen”。常规 C++ Context Dispatcher 仍可能选择它。

Generation 在常规 AttentionOp 中还可能走 XQA、Masked MHA 或其他 Decode 路径；只有 Python 内部快路径把 Context 和 Generation 都显式接入 FlashInfer TRTLLM-Gen 包装。

## 四、Python 快路径如何选择与回退

`TrtllmAttention._run()` 的选择逻辑可以简化为：

```python
use_trtllm_gen = False

if _TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION:
    backend = self._get_trtllm_gen_backend()
    use_trtllm_gen = backend.is_supported(
        q,
        metadata=metadata,
        forward_args=forward_args,
        mask_type=int(forward_args.mask_type),
        active_helix=helix_active,
        use_sage_attn=use_sage_attn,
    )[0]

if use_trtllm_gen:
    backend.attention(
        q,
        metadata=metadata,
        forward_args=forward_args,
        mask_type=int(forward_args.mask_type),
        use_paged_context_fmha=metadata.use_paged_context_fmha,
    )
else:
    thop.attention(...)
```

### 4.1 架构

当前 Python Checker 只接受：

```text
SM100 / SM103
```

这对应数据中心 Blackwell Family。名称中同属 Blackwell 的其他 Compute Capability 不能自动视为该路径支持。

### 4.2 必须使用 Paged KV

快路径要求：

```text
KVCacheManager 存在
kv_cache_block_offsets 存在
```

也就是必须能从：

```text
Block Table + KV Pool
```

构建 Kernel 所需的 Page Table。

### 4.3 Context 限制

当前检查包括：

- 不支持 Head Size 72、80、96。
- 不支持 Custom Mask。
- 不支持 ALiBi。
- 不支持 Padded Input。
- MLA Context 暂时回退。
- Q/KV/O 数据类型组合必须存在对应 Kernel。

### 4.4 Generation 限制

主要包括：

- Beam Width 必须为 1。
- 不支持 Position Shift。
- `num_heads % num_kv_heads == 0`。
- GQA Ratio 不超过 32。
- Paged KV Block Size 必须为 16、32 或 64。
- Block Size 必须是 2 的幂。

### 4.5 为什么必须先检查

TRTLLM-Gen 不是任意 Shape 的即时模板展开。最终执行依赖：

- 对应架构。
- 数据类型组合。
- Head Dimension。
- Tile。
- Page Size。
- Mask。
- Kernel Type。

如果没有预编译 Cubin或可用 NVRTC 生成路径，强行执行会在 Kernel 查找阶段失败。

## 五、Mixed Batch 如何拆成 Context 和 Generation

在线服务的一个 Batch 可以同时包含：

```text
Context 请求：本轮处理多个 Prompt Token
Generation 请求：本轮处理一个或多个生成 Token
```

Metadata 已把请求按阶段排列，并记录：

```text
num_contexts
num_ctx_tokens
host_request_types
prompt_lens
kv_lens
```

`attention()` 先计算：

```python
num_generations = total_sequences - num_contexts
num_gen_tokens = total_tokens - num_ctx_tokens
```

然后按 Token Offset 切输入和输出：

```text
Q[0 : num_ctx_tokens]          -> run_context()
Q[num_ctx_tokens : end]        -> run_generation()
```

每一段都复用同一个 Workspace 和 Layer 静态配置。

### 5.1 为什么不是一个 Kernel 全做

Context 与 Decode 的最优映射差异很大：

```text
Context：沿 Query Tile 和 KV Tile 展开
Decode：沿 Head Group 和长 KV 维度补并行
```

把两者拆开可为每个阶段选择不同 Kernel，而不需要在一个 Kernel 内保留大量运行时分支。

## 六、QKV 预处理做了什么

FMHA 的输入不是拿到 Q/K/V 后直接进入主循环。`trtllmGenQKVProcessOp.cpp` 先完成运行时桥接。

### 6.1 构造变长序列 Metadata

`BuildDecoderInfoParams` 生成：

```text
cuQSeqlens
cuKvSeqlens
tokensInfo
RoPE inverse frequency
FP8/NVFP4 scale
```

变长 Batch 使用：

```text
cuSeqlens[i]
```

定位第 `i` 个请求在 Flatten Token Buffer 中的起点，避免 Padding。

### 6.2 QKV 预处理

核心参数结构是：

```text
QKVPreprocessingParams
```

Context 路径中关键字段的设置如下：

```cpp
qkvParams.qkv_input = qkv_input.data_ptr();
qkvParams.q_output = ptrs.qBufPtr;
qkvParams.kv_cache_buffer = kvArrays.kvCacheBuffer;
qkvParams.kv_cache_block_scales_buffer =
    kvArrays.kvScaleCacheBuffer;

qkvParams.tokens_info = ptrs.tokensInfoPtr;
qkvParams.seq_lens =
    static_cast<int*>(context_lengths.data_ptr());
qkvParams.cache_seq_lens =
    static_cast<int*>(sequence_lengths.data_ptr());

qkvParams.head_num = num_heads;
qkvParams.kv_head_num = num_kv_heads;
qkvParams.qheads_per_kv_head =
    num_heads / num_kv_heads;

qkvParams.separate_q_kv_output = paged_context_fmha;
qkvParams.quantized_fp8_output = fp8_context_fmha;
qkvParams.generation_phase = false;
```

一次预处理可融合：

- QKV Bias。
- RoPE/mRoPE。
- Position Offset。
- Q 输出整理。
- K/V 写入 Paged KV Cache。
- FP8/NVFP4 KV 量化。
- GQA Head 映射。
- Speculative Decode Position Offset。

简化逻辑：

```cpp
for token, kv_head:
    q, k, v = load_qkv(token)
    q, k = apply_rope(q, k, position)

    q_buffer[token] = q

    physical_block = block_table[request][logical_block]
    slot = token_position % tokens_per_block
    kv_cache[physical_block, slot, kv_head] = quantize(k, v)
```

### 6.3 Context 和 Generation 的差别

Context：

```text
一次处理多个 Token
构造完整 cuQ/cuKV 长度
可使用 Packed QKV 或独立 Q + Paged KV
```

Generation：

```text
通常每请求一个 Token
也支持 Speculative Decode 的多个 Query Token
必须把新增 K/V 追加到对应 Page Slot
```

### 6.4 Postprocess 的作用

Context FMHA 后还调用 `trtllm_gen_context_postprocess()`。

它不重新执行 Attention，主要用于处理需要延后完成的缓存语义，例如：

- 特殊 Sparse KV 更新。
- 其他需要在 FMHA 后提交的扩展缓存状态。

普通 Dense Paged KV 的主要 Q/K/V 整理已在 Preprocess 完成。

## 七、Paged KV Cache 如何交给 Kernel

TRTLLM-Gen 默认 KV Layout：

```text
HND:
[num_pages, kv_factor, num_kv_heads, page_size, head_dim]
```

普通 Attention：

```text
kv_factor = 2
```

表示 K Plane 和 V Plane。

MLA：

```text
kv_factor = 1
```

因为缓存的是 Latent State，而不是普通独立 K/V。

### 7.1 Page Table

每个请求维护：

```text
block_tables[request, logical_page] = physical_page
```

Kernel 读取 KV Tile 时：

```text
logical_token
-> logical_page + offset_in_page
-> physical_page
-> KV Pool 地址
```

### 7.2 为什么 Page Size 受限

Page Size 参与：

- TMA Box Shape。
- 每次加载多少 Page Offset。
- KV Tile 是否跨 Page。
- Kernel Key。
- Shared Memory Stage 大小。

因此 Page Size 不是纯 Metadata。当前 Python 快路径只接受 16、32、64。

### 7.3 FlashInfer 接口中的别名

Python Backend 向 FlashInfer 传入：

```python
kv_cache=(kv_pool, kv_pool)
```

看起来 K/V 是同一个 Tensor，实际是同一块扁平 Paged Pool 的两个逻辑 View。`kv_factor`、Stride 和 Page Metadata 决定 Kernel 如何解释 K Plane 与 V Plane。

## 八、FMHA 主循环与在线 Softmax

对一个 Query Tile，FMHA 逐块扫描 KV：

```text
for KV tile:
    S_tile = Q_tile @ K_tile^T
    apply scale/mask
    update online softmax
    O_acc += P_tile @ V_tile
```

### 8.1 在线 Softmax 状态

为每个 Query Row 维护：

```text
m：已处理 Score 的最大值
l：exp(score - m) 的总和
o：未归一化的加权 V 累加
```

新 Tile 的局部最大值：

```text
m_t = max(S_t)
m_new = max(m, m_t)
```

修正旧状态：

```text
alpha = exp(m - m_new)
```

新概率分子：

```text
P_t = exp(S_t - m_new)
```

更新：

```text
l_new = alpha * l + sum(P_t)
o_new = alpha * o + P_t V_t
```

所有 KV Tile 完成后：

```text
O = o / l
```

这样无需保存完整 Score/Probability Matrix。

### 8.2 为什么大量使用 `exp2`

源码把 Softmax Scale 转成 Log2 域：

```text
scale_log2 = scale * log2(e)
```

于是：

```text
exp(x) = exp2(x * log2(e))
```

GPU 上 `exp2` 路径更适合快速近似指令，减少 Softmax 标量开销。

### 8.3 Attention Sink

若模型使用 Attention Sink，分母额外加入：

```text
exp(sink - m)
```

它参与 `l`，但不对应普通 V Token。归并 Kernel 也必须使用同一规则。

## 九、Blackwell 硬件如何映射到 FMHA

TRTLLM-Gen 与传统 SM80/SM90 FMHA 最大区别，不只是换了 Tile Size，而是围绕 Blackwell 的 TMA、TMEM、UMMA 和 CTA Cluster 重新组织数据流。

### 9.1 TMA：搬运而不是计算

TMA 负责把多维 Tile 从 HBM 搬到 Shared Memory。

源码中可以看到：

- 为 Q/K/V 构造 TMA Shape/Stride。
- KV 事务尽量对齐到 128B。
- 使用 Shared Memory Swizzle 减少 Bank Conflict。
- Paged KV 先读取 Page Index，再对 Page 内连续 Tile 发起传输。

简化流水：

```text
HBM Paged KV
-> Page Table
-> TMA load
-> SMEM stage
-> Tensor Core MMA
```

TMA 让加载 Warp 不必为每个元素手工计算地址和执行普通 Load 指令。

### 9.2 TMEM：保存 Tensor Core 中间状态

Blackwell 引入 Tensor Memory。`KernelTraits` 明确规划 TMEM Column 给：

```text
S：QK 的 Score Accumulator
P：Softmax 后的 Probability Tile
Stats：Max/Sum
O：PV Accumulator
Transformed KV：部分低精度转换结果
```

典型布局是：

```text
[S0][S1][Stats0][Stats1][P0][P1][O0][O1][Transformed KV]
```

实际布局会根据：

- 是否让 S/P 共用 Column。
- Stats 放 TMEM 还是 SMEM。
- 是否保存 Transformed KV。
- 是否有两个 Q/KV Pipeline Instance。

动态选择。

价值在于：

- 减少大 Accumulator 对普通 Register File 的压力。
- 避免 Score/Probability 落到 HBM。
- 让两个 MMA 阶段共享片上中间结果。

### 9.3 UMMA 与 2-CTA MMA

源码使用 `UTCMMA`/`2CtaMma` 抽象 Blackwell Tensor Core MMA。

当 Cluster Dimension 为 2：

```text
两个 CTA 协作完成一个更大的 MMA Tile
```

每个 CTA 只加载部分 Operand，并通过 Cluster 机制协作。`KernelTraits` 会相应缩减每个 CTA 需要的 TMEM Column 和 KV Tile。

### 9.4 Warp Specialization

Kernel Traits 为加载任务分配专用 Warp：

```text
Load Task Warp：
  Page Offset
  Q/K/V TMA
  Scale

Compute Task Warp Group：
  QK MMA
  Softmax
  PV MMA

Epilogue：
  TMEM -> SMEM/Register -> Output
```

加载与计算通过多 Stage Pipeline 交错，目标是隐藏 HBM/TMA 延迟。

### 9.5 Programmatic Dependent Launch

Backend 可启用 PDL。Launch 时设置：

```text
PROGRAMMATIC_STREAM_SERIALIZATION
```

后继 Kernel 可以更早被调度，并在真正读取依赖数据前等待前驱 Grid 的信号，减少 Kernel 边界的空洞。

它不是取消依赖，而是把“何时等待”从整个 Kernel Launch 前移到更细的执行位置。

## 十、Kernel 类型、Tile 与自动选择

`FmhaKernelType` 包括：

```cpp
enum class FmhaKernelType
{
    Context = 0,

    // 根据 GQA Ratio 自动选择下面两种 Generation Kernel。
    Generation,

    // 交换 MMA 的 A/B 映射，只支持 numHeadsQPerKv <= 16。
    SwapsMmaAbForGeneration,

    // 保持 MMA 的 A/B 映射。
    KeepsMmaAbForGeneration,
};
```

### 10.1 Context

用于较大 `Q_len`。常见起始配置：

```text
TileQ = 128
TileKV = 128
Persistent Scheduler
```

实际 Tile 会根据 Head Dimension、数据类型和架构调整。

### 10.2 SwapsMmaAbForGeneration

Decode 中 Query/Head Group 维度很小。该 Kernel 交换 MMA 的 A/B 映射，使小 Query/Head Group 维更适合硬件 MMA Shape。

当前启发式：

```text
numHeadsQPerKv <= 16
-> 优先 SwapsMmaAb
```

### 10.3 KeepsMmaAbForGeneration

当 GQA Ratio 较大或 MLA 的 Head Dimension 更复杂时，保持标准 Operand 映射更合适：

```text
numHeadsQPerKv > 16
-> 优先 KeepsMmaAb
```

### 10.4 AutoTuner 选择什么

`FmhaAutoTuner` 根据运行时 Shape 和 GPU SM 数选择：

- Kernel Type。
- `tileSizeQ`。
- `tileSizeKV`。
- Static/Persistent Scheduler。
- Single/2-CTA MMA。
- `headDimPerCtaV`。
- Multi-CTA KV Reduction 模式。
- MMA 顺序。
- Softmax 存储布局。

这里的 AutoTuner 主要是启发式选择，不应理解为每次在线遍历所有 Kernel 做 Benchmark。

## 十一、Persistent Scheduler 为什么重要

Static Scheduler 让一个 CTA 负责一个固定 Tile。

变长 Batch 中：

```text
请求 A: 128 token
请求 B: 8192 token
请求 C: 512 token
```

不同 Tile 工作量差异很大，固定映射容易出现尾部 SM 空闲。

Persistent Scheduler 的做法：

```text
启动接近 SM 数量的常驻 CTA
-> CTA 从全局 Work Queue 取下一个 Tile
-> 完成后继续领取
```

TensorRT-LLM Workspace 中有：

```text
fmhaTileCounter
```

用于分配 Work ID。

收益：

- 变长序列负载更均衡。
- 减少大量小 CTA 的调度尾部。
- 更适合 In-flight Batching。

代价：

- 需要 Work ID Pipeline。
- Tile 顺序和同步更复杂。
- 并非所有小 Shape 都比 Static 更快。

## 十二、Generation 的 GQA 与 Multi-CTA KV

GQA 中：

```text
G = num_heads_q / num_heads_kv
kv_head = q_head / G
```

同一个 KV Head 服务 `G` 个 Query Head。

### 12.1 Decode 为什么缺并行度

单 Token Decode：

```text
Q_len = 1
```

若 Batch 也小，按：

```text
batch * q_heads
```

启动的 CTA 数不足以占满所有 SM。

### 12.2 沿 KV 维拆分

TRTLLM-Gen 可以把长 KV 序列分给多个 CTA：

```text
CTA 0: KV tile range 0
CTA 1: KV tile range 1
CTA 2: KV tile range 2
...
```

每个 CTA 输出：

```text
partial_max
partial_sum
partial_O
```

这就是 Multi-CTA KV，也可理解为 Attention 的 Split-KV。

### 12.3 三种归并模式

```text
GmemReduction：
  Partial 写 Global Memory，再归并

GmemReductionWithSeparateKernel：
  用独立 Kernel 完成较大的归并

CgaSmemReduction：
  Cluster 内通过 Remote Shared Memory 归并
```

若所有 Cluster 能在一波内驻留，CGA Shared Memory 可减少 Global Memory 往返；否则 Global Reduction 更稳妥。

## 十三、Multi-CTA 结果如何正确归并

不能直接平均 `partial_O`，因为每个 KV 分片使用不同的 Softmax 最大值。

第 `j` 个分片输出：

```text
m_j：局部最大 Score
l_j：sum exp(score - m_j)
o_j：sum exp(score - m_j) * V
```

全局最大值：

```text
m = max_j(m_j)
```

修正系数：

```text
alpha_j = exp(m_j - m)
```

全局分母：

```text
l = sum_j alpha_j * l_j
```

全局分子：

```text
o = sum_j alpha_j * o_j
```

最终：

```text
O = o / l
```

`fmhaReduction.cu` 正是按这个关系读取 `partialStats(max, sum)` 和 `partialO`。

### 13.1 为什么需要 Counter

若多个 CTA 写 Partial Buffer 后由其中一个 CTA 负责最终归并，需要 Counter 确认：

```text
所有 KV 分片均完成
```

因此 Gmem Multi-CTA 模式要求：

```text
multiCtasKvScratchPtr != null
multiCtasKvCounterPtr != null
```

源码在 Launch 前显式检查，避免读取未完成结果。

## 十四、FP8 与 NVFP4 路径

Blackwell FMHA 不只支持 FP16/BF16。

### 14.1 FP8

可支持的组合包括：

```text
Q FP8 + KV FP8 + O FP8
Q FP8 + KV FP8 + O FP16/BF16
Q BF16/FP16 + KV FP8 + O 同输入类型
```

Scale 通过 Device Pointer 传入：

```text
scaleSoftmaxLog2Ptr
outputScalePtr
```

Kernel 可直接加载 Scale，避免每次 Host 重新构造。

### 14.2 NVFP4 KV

NVFP4 Payload 使用 E2M1，每组值配套 Scaling Factor。

数据路径：

```text
Paged FP4 KV + Scale
-> TMA Load
-> 片上解码/转换
-> QK/PV MMA
```

`KernelTraits` 在特定条件下把转换后的 KV 保留在 TMEM：

```text
KV = E2M1
Q = E4M3
Head Stage = 128
SwapsMmaAb Generation
```

这样后续 MMA Stage 不必重复转换。

### 14.3 为什么低精度不是只改 Dtype

低精度会同时改变：

- KV Physical Layout。
- Scale Layout。
- TMA Transaction Width。
- Shared Memory Stage 数。
- BMM1/BMM2 Operand Dtype。
- TMEM 分配。
- Output Quantization。
- Kernel Key。

因此不能把 BF16 Kernel 外层加 Cast 当作等价实现。

## 十五、Context、Decode 与 Speculative Decode

### 15.1 Context

Context 调用：

```python
flashinfer.prefill.trtllm_batch_context_with_kv_cache(...)
```

输入包含：

- `cum_seq_lens_q`。
- `cum_seq_lens_kv`。
- `max_q_len`。
- `max_kv_len`。
- Page Table。
- Sliding Window。
- BMM Scale。

输出直接写入最终 Attention Buffer。

### 15.2 单 Token Decode

调用：

```python
flashinfer.decode.trtllm_batch_decode_with_kv_cache(...)
```

主要工作是：

```text
读取一个 Query
-> 扫描历史 Paged KV
-> GQA Head Mapping
-> 在线 Softmax
-> 输出一个 Head 向量
```

### 15.3 多 Token Generation

Speculative Decode 中每个请求可能验证多个 Token：

```text
Q_len > 1
```

Backend 传入：

- 每请求 Generation Length。
- Position Offset。
- 可选 Tree Mask。
- `cu_seqlens_q`。

Kernel 区分：

```text
Causal Multi-token
Custom Tree Speculation
```

并在必要时预处理 Packed Custom Mask。

### 15.4 MLA

MLA Generation 走专门的：

```text
trtllm_batch_decode_with_kv_cache_mla
```

其：

```text
QK Head Dim = kv_lora_rank + qk_rope_head_dim
V Head Dim  = kv_lora_rank
```

与标准 MHA/GQA 的 K/V Head Layout 不同。当前 Python 快路径只对 MLA Generation 开放，MLA Context 回退。

## 十六、Cubin、NVRTC 与 Kernel Key

TRTLLM-Gen 有两种 Kernel 获得方式。

### 16.1 预编译 Cubin

仓库按组合保存 Cubin。文件名已经编码大量 Traits，例如：

```text
Sm100
Q/KV/O dtype
Head Dim
Paged KV
Mask
Page Size
VarSeq
TileQ/TileKV
Persistent/Static
Context/Generation
Multi-CTA
```

运行时把配置编码为 Hash Key。

Key 主要包含：

```text
QKV Layout
Mask Type
Kernel Type
Tile Scheduler
Multi-CTA Mode
Head Dimensions
TileQ/TileKV
Page Size
2-CTA MMA
Sparse Type
Skip Softmax
```

对应查找流程：

```cpp
FmhaAutoTuner autoTuner(
    options,
    optionsFromArgs,
    params.mMultiProcessorCount);

std::tie(options, optionsFromArgs, ctaDim)
    = autoTuner.selectKernel();

computeNumCtas(options, params.mMultiProcessorCount);

auto [hashId, info] = hashFromFmhaOptions(options);
auto findIter = mFunctions.find(hashId);

TLLM_CHECK_WITH_INFO(
    findIter != mFunctions.end(),
    "Trtllm-gen kernels not found: " + info);
```

数据类型和 SM 已由当前 Kernel 集合对象限定。

### 16.2 NVRTC

部分 Generation 或 Sparse 配置走 NVRTC：

```text
根据 FmhaOptions 生成配置
-> 编译 Kernel
-> 缓存 CUmodule/CUfunction
-> 后续复用
```

JIT 的优点是减少预编译组合爆炸；代价是首次编译开销和部署环境复杂度。

### 16.3 为什么 Kernel 缺失不能静默执行

`checkIfKernelExist()` 会先运行同样的 AutoTuner 和 Hash 查找。

找不到时返回：

```text
false + 详细 Traits 信息
```

上层据此回退，而不是等到 Launch 才失败。

## 十七、不支持时如何回退

回退分两层。

### 17.1 Python Backend

以下任一条件失败：

```text
架构
数据类型
Head Dim
Mask
Page Size
Beam
MLA 阶段
Paged KV
```

就回到：

```text
thop.attention
```

### 17.2 C++ Context Dispatcher

在常规路径中：

```text
SM100 Family + TRTLLM-Gen Kernel 存在
-> TllmGenFmhaRunner

其他架构
-> FusedMHARunnerV2

FMHA 不支持
-> Unfused MHA
```

### 17.3 回退为什么不能省

Attention 配置空间包含：

- 多种 Head Dimension。
- MHA/MQA/GQA/MLA。
- Paged/Contiguous KV。
- FP16/BF16/FP8/FP4。
- Causal/Sliding/Custom Mask。
- Context/Decode/Spec Decode。

任何新 Kernel 都只能先覆盖其中一部分。可靠框架必须保证未覆盖组合仍能正确运行。

## 十八、把执行过程压缩成伪代码

### 18.1 Backend

```python
def trtllm_attention(qkv, metadata, config):
    if not trtllm_gen_supported(config, metadata):
        return thop_attention(qkv, metadata, config)

    context_qkv, generation_qkv = split_mixed_batch(qkv, metadata)

    if context_qkv:
        q, kv_pool, page_table, scale = preprocess_context(
            context_qkv,
            metadata,
        )
        out_context = trtllm_gen_context_fmha(
            q,
            kv_pool,
            page_table,
            scale,
        )
        postprocess_context_cache(metadata)

    if generation_qkv:
        q, kv_pool, page_table, scale = preprocess_generation(
            generation_qkv,
            metadata,
        )
        out_generation = trtllm_gen_decode_fmha(
            q,
            kv_pool,
            page_table,
            scale,
        )

    return concat(out_context, out_generation)
```

### 18.2 Kernel

```python
def fmha_tile(q_tile, page_table):
    running_max = -inf
    running_sum = 0.0
    output_acc = 0.0

    for logical_page in assigned_kv_pages:
        physical_page = page_table[logical_page]

        # Blackwell: TMA -> SMEM，MMA accumulator -> TMEM
        k_tile, v_tile = tma_load(physical_page)
        score = tensor_core_qk(q_tile, k_tile)
        score = apply_scale_and_mask(score)

        tile_max = row_max(score)
        new_max = maximum(running_max, tile_max)
        alpha = exp(running_max - new_max)
        prob = exp(score - new_max)

        running_sum = alpha * running_sum + row_sum(prob)
        output_acc = alpha * output_acc + tensor_core_pv(prob, v_tile)
        running_max = new_max

    return output_acc, running_max, running_sum
```

若 KV 被多个 CTA 拆分，再使用第十三节公式合并。

## 十九、性能分析与正确性检查

### 19.1 首先确认实际路径

不能只看到 Blackwell 就假设已走 TRTLLM-Gen。

应确认日志或 Profile 中：

- Python 快路径是否启用。
- `is_supported()` 是否成功。
- 实际 Kernel 名。
- 是否回退到 `thop.attention`。
- C++ Dispatcher 是否选择 `TllmGenFmhaRunner`。

### 19.2 关键性能指标

```text
Kernel Duration
HBM Read Throughput
Tensor Core Utilization
TMA Throughput
SM Occupancy
Active CTA/Cluster
TMEM/SMEM 使用
Multi-CTA Reduction 占比
QKV Preprocess 占比
```

### 19.3 常见瓶颈

Prefill：

- Q/KV Tile 不适合 Head Dim。
- 变长序列尾部不均衡。
- Paged KV Page 太小导致地址和 Page Metadata 开销高。
- QKV Preprocess/Cache 写入占比过大。

Decode：

- Batch 太小导致并行度不足。
- KV 很长但未启用合适 Multi-CTA。
- Multi-CTA Partial Buffer 和 Reduction 成本过高。
- GQA Ratio 与 Kernel Type 不匹配。
- 低精度反量化/Scale 读取未隐藏。

### 19.4 正确性测试矩阵

至少覆盖：

```text
Context only
Generation only
Mixed batch
MHA / MQA / GQA
不同 Head Dim
不同 Page Size
变长序列
Sliding Window
Chunked Context
Speculative Decode
FP16 / BF16 / FP8 / NVFP4
Prefix Cache 命中
Fallback
```

比较内容：

- Attention Output。
- 端到端 Logits。
- KV Cache 内容。
- 长序列数值误差。
- 不同 Batch 排列下的一致性。

## 二十、总结

TensorRT-LLM 在 Blackwell 上的 FMHA 主线不是简单调用一个新 Kernel，而是：

```text
运行时 Metadata
-> Context/Generation 拆分
-> QKV Bias/RoPE/量化/Paged KV 写入
-> Kernel Traits 与 AutoTuner
-> TMA 加载
-> TMEM 中的 QK/Softmax/PV 流水
-> Persistent/Static 调度
-> 必要时 Multi-CTA KV 归并
-> Output/Cache 收尾
```

最关键的优化点：

1. 在线 Softmax 避免 `Q_len * KV_len` 中间矩阵写入 HBM。
2. TMA 负责规则的多维 Tile 搬运，Shared Memory 采用对齐和 Swizzle。
3. TMEM 承载 Score、Probability、统计量和 Output Accumulator，降低寄存器压力。
4. Persistent Scheduler 改善变长 Batch 的 Tile 负载均衡。
5. Generation 沿 KV 维拆成多个 CTA，恢复小 Batch 下的并行度。
6. CGA Shared Memory 或 Global Reduction 用严格的 Softmax 修正公式合并 Partial。
7. FP8/NVFP4 是完整的数据布局和 Pipeline 设计，不是简单 Cast。
8. 支持检查、Cubin/NVRTC 选择和回退共同保证工程可用性。
