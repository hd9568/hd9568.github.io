---
title: 'KV Cache 量化：用 FP8、INT8 与 INT4 降低长上下文成本'
description: '推导 KV Cache 显存和带宽开销，讲解静态与动态 Scale、Per-head/Per-token 量化、融合反量化、误差来源，并结合 vLLM 与 LMDeploy 的缓存实现说明工程细节。'
category: '推理优化'
pubDate: '2026-07-28T12:41:00+08:00'
updatedDate: '2026-07-28T12:41:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [KV Cache 为什么值得量化](#一kv-cache-为什么值得量化)
2. [显存与带宽计算](#二显存与带宽计算)
3. [量化公式](#三量化公式)
4. [Scale 粒度](#四scale-粒度)
5. [FP8、INT8 与 INT4](#五fp8int8-与-int4)
6. [写入和读取流程](#六写入和读取流程)
7. [最小实现](#七最小实现)
8. [开源框架中的实现](#八开源框架中的实现)
9. [误差从哪里来](#九误差从哪里来)
10. [性能为什么不一定线性提升](#十性能为什么不一定线性提升)
11. [评估和选型](#十一评估和选型)
12. [总结](#十二总结)

## 一、KV Cache 为什么值得量化

自回归 Decode 会缓存每层历史 token 的 Key 和 Value，避免每生成一个 token 都重新计算完整前缀。

随着以下变量增长，KV Cache 很快成为主要显存消费者：

```text
并发请求数
上下文长度
Transformer 层数
KV Head 数
Head Dimension
数据类型字节数
```

KV Cache 量化把 BF16/FP16 的 K/V 存成 FP8、INT8 或更低比特：

```text
BF16 -> FP8/INT8：理论负载减半
BF16 -> INT4：理论负载降为四分之一
```

它同时可能带来两类收益：

- 相同显存容纳更多 token，提高并发或最大上下文。
- Decode Attention 读取的字节数减少，提高 Memory-bound 阶段吞吐。

## 二、显存与带宽计算

对标准 KV Cache，每个 token 的存储量为：

```text
bytes_per_token =
    num_layers
  * 2
  * num_kv_heads
  * head_dim
  * bytes_per_element
```

其中 `2` 表示 K 和 V。

总缓存：

```text
KV_bytes =
    batch_tokens
  * num_layers
  * 2
  * num_kv_heads
  * head_dim
  * bytes_per_element
```

### 2.1 具体例子

某 GQA 模型：

```text
num_layers = 80
num_kv_heads = 8
head_dim = 128
```

BF16 每 token：

```text
80 * 2 * 8 * 128 * 2 bytes
= 327680 bytes
= 320 KiB
```

单请求 128K token：

```text
320 KiB * 131072
= 40 GiB
```

改成 FP8 后，Payload 约为 20 GiB；INT4 约为 10 GiB。实际还需加 Scale、Block 元数据、对齐和内存分配开销。

### 2.2 为什么也能减少 Decode 带宽

单 token Decode 的 Attention 需要读取历史 K/V：

```text
Q_t K_0:t^T
Softmax(...)
P V_0:t
```

序列越长，历史 K/V 读取越多。若 Attention Kernel 直接读取低比特缓存并在寄存器中反量化，HBM 流量会显著下降。

## 三、量化公式

### 3.1 对称量化

对一组浮点值 `x`，`b` bit 有符号整数范围近似为：

```text
qmin = -2^(b-1)
qmax =  2^(b-1) - 1
```

Scale：

```text
s = max(abs(x)) / qmax
```

量化和反量化：

```text
q = clamp(round(x / s), qmin, qmax)
x_hat = s * q
```

对称量化只保存 Scale，不需要 Zero Point，Kernel 更简单。

### 3.2 非对称量化

```text
s = (xmax - xmin) / (qmax - qmin)
z = round(qmin - xmin / s)
q = clamp(round(x / s) + z, qmin, qmax)
x_hat = s * (q - z)
```

它能更充分利用偏移分布的整数范围，但多一个 Zero Point，解码路径更复杂。KV Cache 常见实现更偏向对称量化。

### 3.3 FP8 也需要 Scale

FP8 本身有指数和尾数，不等于完全不需要缩放。常见形式是：

```text
q_fp8 = cast_fp8(x / s)
x_hat = cast_high_precision(q_fp8) * s
```

Scale 用于把当前张量动态范围映射到 FP8 可表示区间。

## 四、Scale 粒度

Scale 越细，通常误差越小，但元数据和计算开销越高。

### 4.1 Per-tensor / Per-layer

整层 K 或 V 使用一个 Scale：

```text
k_scale[layer]
v_scale[layer]
```

优点是 Scale 开销极低，访问简单；缺点是局部 Outlier 会压缩大多数值的有效精度。

### 4.2 Per-head

每层每个 KV Head 使用独立 Scale：

```text
k_scale[layer, head]
v_scale[layer, head]
```

不同 Head 的分布可分别适配，是精度和开销之间的常见折中。

### 4.3 Per-token

每个 token 使用独立 Scale。它能适应时间维度上的动态范围变化，但 Scale 数量随序列增长。

### 4.4 Per-token-head

每个 token、每个 KV Head 分别计算：

```text
k_scale[block, slot, head]
v_scale[block, slot, head]
```

若 `head_dim = 128`，缓存数据为 1 byte，Scale 为 FP32，则 Scale 开销比例约为：

```text
2 * 4 bytes / (2 * 128 * 1 byte)
= 3.125%
```

这里分子和分母都包含 K/V。额外 3.125% Payload 换来更细的动态范围控制。

## 五、FP8、INT8 与 INT4

### 5.1 FP8 E4M3

E4M3 有较多尾数位、较小指数范围，通常适合需要较高精度的前向张量。配合合适 Scale，可用于 KV Cache。

特点：

- 1 byte/element。
- GPU 原生低精度支持较好。
- 量化与反量化路径相对自然。
- 仍要处理动态范围和硬件格式差异。

### 5.2 FP8 E5M2

E5M2 指数范围更大、尾数精度更低。它更能覆盖 Outlier，但普通值的相对误差可能更大。不能只根据“范围更大”判断效果，应基于模型校准。

### 5.3 INT8

INT8 规则清晰，通常使用对称 Scale：

```text
q = round(x / s)
```

它的硬件兼容性和 Kernel 路径取决于推理框架。若 Attention Kernel 不能原生消费 INT8 KV，额外转换可能抵消带宽收益。

### 5.4 INT4

INT4 把两个值打包进一个 byte：

```text
byte = low_4bit | (high_4bit << 4)
```

它有更高压缩率，但：

- 动态范围和精度更紧张。
- 需要位操作解包。
- Scale 元数据占比更高。
- 对 K/V 的误差更敏感。
- Kernel 必须融合解包与 Attention 才容易获得真实收益。

## 六、写入和读取流程

### 6.1 写入 KV Cache

每生成一批新 K/V：

```text
浮点 K/V
-> 按量化组统计 absmax
-> 计算 Scale
-> 除以 Scale 并转换到低精度
-> 写入 Paged KV Cache
-> 写入 Scale Cache
```

Paged Layout 下，逻辑 token 先通过 `slot_mapping` 映射到：

```text
physical_block = slot // block_size
offset_in_block = slot % block_size
```

然后把量化结果写到指定物理位置。

### 6.2 Attention 读取

正确的高性能路径不是：

```text
完整低精度 KV
-> 生成完整 BF16 KV 临时张量
-> Attention
```

而是：

```text
按 tile 读取低精度 KV
-> 在寄存器/Shared Memory 中反量化
-> 立即参与 QK 和 PV 计算
```

这样才能保留 HBM 流量降低的收益，避免巨大的中间张量。

## 七、最小实现

### 7.1 PyTorch 版 Per-token-head INT8

```python
import torch


def quantize_kv_per_token_head(x: torch.Tensor):
    """
    x: [num_tokens, num_kv_heads, head_dim]
    """
    qmax = 127.0

    # 每个 token、每个 head 独立统计，保留最后一个维度用于广播。
    absmax = x.float().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(absmax / qmax, min=1e-6)

    q = torch.round(x.float() / scale)
    q = torch.clamp(q, -128, 127).to(torch.int8)
    return q, scale.squeeze(-1)


def dequantize_kv(q: torch.Tensor, scale: torch.Tensor):
    return q.float() * scale.unsqueeze(-1)
```

### 7.2 写入分页缓存

```python
def write_quantized_cache(
    key,
    value,
    slot_mapping,
    key_cache,
    value_cache,
    key_scales,
    value_scales,
    block_size,
):
    qk, sk = quantize_kv_per_token_head(key)
    qv, sv = quantize_kv_per_token_head(value)

    for token_idx, slot in enumerate(slot_mapping):
        block = int(slot) // block_size
        offset = int(slot) % block_size

        key_cache[block, offset] = qk[token_idx]
        value_cache[block, offset] = qv[token_idx]
        key_scales[block, offset] = sk[token_idx]
        value_scales[block, offset] = sv[token_idx]
```

这个实现便于理解，但 Python 循环效率很低。真实系统使用一个 GPU Kernel 完成地址映射、统计、量化和写回。

## 八、开源框架中的实现

### 8.1 vLLM

vLLM 的 Cache 配置把存储类型和计算类型分开管理：

```text
cache_dtype = auto / fp16 / bf16 / fp8 / ...
```

在 Per-token-head 动态量化路径中，一个 Triton Program 负责一个 `(token, head)`：

```python
slot = slot_mapping[token]
block = slot // block_size
offset = slot % block_size

k = load(key[token, head, :])
k_scale = max(abs(k)) / quant_max
k_q = clamp(k / k_scale, quant_min, quant_max)

store(key_cache[block, offset, head, :], k_q)
store(k_scale_cache[block, offset, head], k_scale)
```

V 使用相同流程。这个设计有几个重点：

- Scale 与量化数据使用同一物理 block/slot/head 索引。
- `slot_mapping < 0` 的 Padding token 不写入缓存。
- `head_dim` Padding 到适合 Triton 规约的大小。
- 量化范围由实际 Cache Dtype 决定，可复用同一计算结构。

### 8.2 LMDeploy

LMDeploy 的 Cache Engine 会根据量化策略选择：

```text
原始模型 dtype
FP8 E4M3 / E5M2
INT8
INT4 打包
更低比特的 K/V 组合
```

在 INT4 路径中，逻辑 Head Dimension 与物理存储 Dimension 不同，因为多个低比特元素被打包到一个 byte。Cache Descriptor 必须同时描述 Payload 和 Scale/辅助张量。

这说明 KV 量化不是简单修改 Tensor Dtype，还会改变：

- 物理 Shape。
- 每个 Block 的字节数。
- Cache Capacity 计算。
- Attention Backend 选择。
- Scale Cache 的布局。

## 九、误差从哪里来

### 9.1 Key 误差

Attention Logit：

```text
score_i = q · k_i / sqrt(d)
```

Key 量化误差会改变 Logit，进而通过 Softmax 改变所有位置的注意力概率。若多个候选 Logit 很接近，小误差也可能改变排序。

### 9.2 Value 误差

```text
output = sum_i probability_i * v_i
```

Value 误差直接累积到 Attention Output。长上下文下，大量小误差可能叠加。

### 9.3 Outlier

一组值中若存在很大 Outlier：

```text
scale = absmax / qmax
```

Scale 被 Outlier 拉大，普通值会落到很少的量化格点上。更细粒度 Scale、校准或旋转变换可以缓解。

### 9.4 长上下文累积

短上下文质量正常，不代表 128K 上下文也正常。KV Cache 的量化误差作用于每一层、每个历史位置，必须按目标长度评估。

### 9.5 K 和 V 不一定适合同一策略

K 影响 Softmax 权重，V 影响加权结果，分布也可能不同。部分方法会对 K/V 使用不同位宽、Scale 或校准方式。

## 十、性能为什么不一定线性提升

BF16 到 FP8 把 Payload 减半，不代表 TPOT 一定提升 2 倍：

```text
总时间 =
    KV 读取
  + 反量化
  + QK/PV 计算
  + Softmax
  + 权重读取
  + 调度和通信
```

收益受以下因素限制：

- Context 不长，KV 读取占比不高。
- Kernel 反量化开销大。
- Scale 访问不合并。
- Kernel 先物化高精度临时张量。
- GPU 对目标低精度格式支持不足。
- Batch 较大，计算而不是带宽成为瓶颈。
- 量化后可容纳更多请求，实际 batch 变化导致比较不公平。

KV 量化最稳定的收益通常是容量；速度收益取决于 Attention Kernel。

## 十一、评估和选型

### 11.1 质量

至少覆盖：

- Perplexity。
- 下游任务准确率。
- 长上下文检索或 Needle 测试。
- 不同 Prompt 长度。
- Greedy 和 Sampling。
- 多轮生成稳定性。

### 11.2 性能

记录：

```text
每 token KV 字节数
可容纳最大 token 数
KV Cache 使用率
TTFT / TPOT
Decode tokens/s
Attention kernel latency
HBM read throughput
量化写入开销
```

### 11.3 选择顺序

1. 先以 BF16/FP16 建立正确性基线。
2. 优先测试硬件和框架支持成熟的 FP8/INT8。
3. 确认 Attention Backend 原生消费量化 KV。
4. 使用目标模型、目标上下文长度进行校准。
5. 容量仍不足时再考虑 INT4 或分层 Offload。

## 十二、总结

KV Cache 量化同时作用于容量和 Decode 带宽：

```text
浮点 K/V
-> 按合适粒度计算 Scale
-> 低精度写入 Paged Cache
-> Attention tile 内融合反量化
-> 减少 HBM 字节和缓存容量
```

真正决定效果的不是位宽本身，而是 Scale 粒度、K/V 分布、低精度 Layout、Attention Kernel 融合和目标上下文长度。FP8/INT8 通常是更稳健的起点，INT4 能进一步压缩，但对量化误差和 Kernel 实现提出更高要求。
