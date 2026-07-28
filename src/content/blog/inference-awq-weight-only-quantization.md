---
title: 'AWQ Weight-Only 量化：W4A16 如何降低模型访存'
description: '从 Decode 的权重带宽瓶颈出发，讲解 Group-wise W4A16、Activation-aware Scaling、INT4 打包、融合反量化 GEMM，并结合 LMDeploy 与 vLLM 的实现说明工程约束。'
category: '推理优化'
pubDate: '2026-07-28T12:42:00+08:00'
updatedDate: '2026-07-28T12:42:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [Weight-Only 量化解决什么问题](#一weight-only-量化解决什么问题)
2. [W4A16 的含义](#二w4a16-的含义)
3. [Group-wise INT4 量化](#三group-wise-int4-量化)
4. [AWQ 的 Activation-aware 思想](#四awq-的-activation-aware-思想)
5. [等价缩放为什么有效](#五等价缩放为什么有效)
6. [INT4 权重如何打包](#六int4-权重如何打包)
7. [最小量化实现](#七最小量化实现)
8. [融合反量化 GEMM](#八融合反量化-gemm)
9. [开源框架中的实现](#九开源框架中的实现)
10. [性能收益和边界](#十性能收益和边界)
11. [精度评估与调参](#十一精度评估与调参)
12. [总结](#十二总结)

## 一、Weight-Only 量化解决什么问题

Transformer 的主要参数位于 Linear 层：

```text
Q/K/V/O Projection
MLP Gate/Up/Down Projection
LM Head
```

单 token、小 batch Decode 时，GEMM 的 M 维很小。每轮都要从 HBM 读取大量权重，但每个权重参与的计算次数有限，算术强度低。

以 7B 参数模型为例：

```text
FP16/BF16 权重约 14 GB
INT8 权重约 7 GB
INT4 权重约 3.5 GB
```

若 Decode 主要受权重带宽限制，减少权重字节数可同时：

- 降低模型显存。
- 减少每轮 HBM 读取。
- 为 KV Cache 和更大 batch 留出容量。

## 二、W4A16 的含义

W4A16 表示：

```text
Weight：4 bit 存储
Activation：FP16/BF16 计算输入
Accumulator：通常使用更高精度
```

它不是纯 INT4 GEMM。典型数据流是：

```text
FP16 Activation
  ×
INT4 Weight + FP16 Scale
  ->
Kernel 内解包/反量化
  ->
FP16/BF16 或 FP32 累加
```

Weight-Only 适合 Activation 难以稳定量化、但权重读取是主要瓶颈的 LLM Decode。

## 三、Group-wise INT4 量化

对权重矩阵：

```text
W: [out_features, in_features]
```

沿 `in_features` 每 `G` 个值分成一组。每组独立计算 Scale 和 Zero Point：

```text
num_groups = in_features / G
```

非对称量化：

```text
s = (w_max - w_min) / 15
z = clamp(round(-w_min / s), 0, 15)
q = clamp(round(w / s) + z, 0, 15)
w_hat = (q - z) * s
```

4 bit 无符号整数范围为 `[0, 15]`。

### 3.1 Group Size 的权衡

| Group Size | Scale 数量 | 精度 | 元数据/Kernel 开销 |
| ---: | ---: | --- | --- |
| 32 | 多 | 通常更好 | 更高 |
| 64 | 中 | 中 | 中 |
| 128 | 少 | 常见折中 | 较低 |
| 整行 | 最少 | 更容易受 Outlier 影响 | 最低 |

Group 越小，每组动态范围越集中，但 Scale 和 Zero Point 占比增大。

### 3.2 实际存储量

对 `P` 个权重、Group Size 为 `G`：

```text
INT4 Payload = P / 2 bytes
Scale        ≈ P / G * scale_bytes
Zero Point   ≈ P / G * zero_bytes
```

所以 W4A16 实际模型大小略大于理想的 FP16 四分之一。

## 四、AWQ 的 Activation-aware 思想

普通 Round-to-nearest 只观察权重分布：

```text
minimize ||W - W_hat||
```

但模型输出误差由输入 Activation 加权：

```text
Y = X W^T
Y_hat = X W_hat^T
error = X (W - W_hat)^T
```

某个权重的绝对误差不大，如果对应输入通道 Activation 经常很大，输出误差仍可能显著。

AWQ 的核心是使用少量校准数据统计 Activation 特征，识别更重要的输入通道，再通过通道缩放降低这些通道的权重量化误差。

常见重要性近似：

```text
importance_j ≈ mean(abs(X[:, j]))
```

它不需要反向传播，也不需要完整重训练。

## 五、等价缩放为什么有效

考虑线性层：

```text
Y = X W^T
```

对输入通道引入正数缩放向量 `s`：

```text
X' = X / s
W' = W * s
```

这里 `/` 和 `*` 按输入通道广播。则：

```text
X' W'^T
= (X / s) (W * s)^T
= X W^T
```

浮点模型输出不变，但量化的是缩放后的 `W'`。对重要通道增大 `s_j`，相当于在量化前放大该通道权重，使其拥有更高的有效表示精度。

实际实现会搜索缩放强度，例如：

```text
s_j = activation_scale_j^alpha
      / weight_scale_j^(1-alpha)
```

`alpha` 可通过校准误差搜索，而不是固定为 0.5。

### 5.1 缩放如何进入模型

不能在每次推理时额外执行昂贵的 `X / s`。缩放通常被吸收到相邻算子：

```text
LayerNorm / 前一 Linear 的输出通道除以 s
当前 Linear 的输入通道乘以 s
```

这样浮点计算等价，又不引入独立 Runtime Kernel。

### 5.2 为什么不能无限放大

放大重要通道可能：

- 增大同组其他权重的动态范围。
- 让前一层参数出现极端数值。
- 降低其他通道精度。

因此 AWQ 需要按层搜索缩放比例，以校准集上的输出误差为目标选择参数。

## 六、INT4 权重如何打包

一个 32-bit 整数可保存 8 个 INT4：

```text
packed =
    q0
  | (q1 << 4)
  | (q2 << 8)
  | ...
  | (q7 << 28)
```

解包第 `i` 个值：

```text
qi = (packed >> (4 * i)) & 0xF
```

某些 AWQ Checkpoint 使用特定交错顺序，例如：

```text
[0, 4, 1, 5, 2, 6, 3, 7]
```

Kernel 必须按存储格式重排，不能假设自然顺序。

典型物理形状：

```text
qweight: [K, N / 8], int32
qzeros:  [K / G, N / 8], int32
scales:  [K / G, N], fp16/bf16
```

其中每个 `qweight` 元素打包 8 个 4-bit 权重。

## 七、最小量化实现

### 7.1 Group-wise 伪量化

伪量化把权重先量化再反量化回浮点，用于评估误差：

```python
import torch


def pseudo_quantize_int4(
    weight: torch.Tensor,
    group_size: int = 128,
):
    """
    weight: [out_features, in_features]
    返回反量化权重、整数权重、Scale 和 Zero Point。
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0

    grouped = weight.reshape(-1, group_size).float()
    w_min = grouped.amin(dim=1, keepdim=True)
    w_max = grouped.amax(dim=1, keepdim=True)

    scale = torch.clamp((w_max - w_min) / 15.0, min=1e-5)
    zero = torch.clamp(torch.round(-w_min / scale), 0, 15)

    qweight = torch.clamp(
        torch.round(grouped / scale) + zero,
        0,
        15,
    ).to(torch.uint8)

    dequant = (qweight.float() - zero) * scale

    return (
        dequant.reshape(out_features, in_features),
        qweight.reshape(out_features, in_features),
        scale,
        zero,
    )
```

### 7.2 一个简单的 Activation-aware Scale

```python
def activation_aware_scale(
    weight: torch.Tensor,
    activation_samples: torch.Tensor,
    alpha: float,
):
    # activation_samples: [num_calibration_tokens, in_features]
    act_scale = activation_samples.abs().mean(dim=0).clamp(min=1e-5)
    weight_scale = weight.abs().mean(dim=0).clamp(min=1e-5)

    scale = act_scale.pow(alpha) / weight_scale.pow(1.0 - alpha)

    # 归一化避免 Scale 整体过大或过小。
    scale = scale / torch.sqrt(scale.max() * scale.min())
    return scale
```

真正的 AWQ 会按层、按候选 `alpha` 评估量化输出误差，并处理 QKV、Gate/Up 等打包 Linear 的特殊布局。

## 八、融合反量化 GEMM

### 8.1 低效路径

```text
INT4 Weight
-> 独立 Kernel 反量化出完整 FP16 Weight
-> 写回 HBM
-> GEMM 再从 HBM 读取 FP16 Weight
```

这会重新产生接近原始权重大小的 HBM 流量，破坏 Weight-Only 的意义。

### 8.2 高效路径

融合 Kernel 按 tile 执行：

```text
1. 从 HBM 读取一块打包 INT4
2. 位移和 Mask 解包
3. 读取对应 Scale/Zero
4. 在寄存器或 Shared Memory 中反量化
5. 立即与 Activation 做矩阵乘
6. 累加结果，不写完整反量化权重
```

伪代码：

```cpp
for each output_tile:
    accumulator = 0

    for each k_tile:
        a = load_fp16_activation_tile()
        packed_w = load_int4_weight_tile()
        scale, zero = load_group_params()

        q = unpack_int4(packed_w)
        w = (q - zero) * scale
        accumulator += matmul(a, w)

    store(accumulator)
```

### 8.3 Decode 与 Prefill 的 Kernel 选择

Decode 常见 `M` 很小，适合专门的 GEMV/小 M GEMM Kernel；Prefill 的 `M` 很大，可能选择不同的 Weight-Only GEMM 或直接使用高精度 Tensor Core 路径。

同一种量化格式需要多个 Shape 适配的 Kernel，不能只用一个配置覆盖所有阶段。

## 九、开源框架中的实现

### 9.1 LMDeploy 的 AWQ 流程

LMDeploy 的量化流程包含两个关键阶段：

1. 使用校准 Activation 计算通道 Scale 并平滑相邻层。
2. 对目标 Linear 执行 Group-wise 量化，替换为 `WeightOnlyQLinear`。

权重模块维护：

```text
qweight
scales
qzeros
bias
w_bit
group_size
```

其创建逻辑会检查：

```text
in_features % group_size == 0
out_features % (32 / w_bit) == 0
```

这些约束来自 Group 划分和 INT32 打包。

Tensor Parallel 下还要分别处理：

- Column Parallel：按输出通道切 `qweight/scales/qzeros`。
- Row Parallel：按输入通道切，并保证 Group 边界对齐。

### 9.2 vLLM 的 Triton AWQ Kernel

vLLM 的 AWQ Triton 路径直接接收：

```text
input:   [M, K]
qweight: [K, N/8]
qzeros:  [K/G, N/8]
scales:  [K/G, N]
```

Kernel 在 K 循环内：

```python
packed = load(qweight_tile)
q = (packed >> shifts) & 0xF
zero = unpack(qzeros)
scale = load(scales)
weight = (q - zero) * scale
accumulator += dot(activation, weight)
```

实现还使用 Split-K 增加小 M 场景的并行度，最后对多个 K 分片的部分结果求和。

这说明推理性能不仅取决于量化算法，也取决于：

- 打包布局。
- Tile Size。
- Split-K。
- Group Size。
- 输入 Shape。
- GPU 架构。

## 十、性能收益和边界

### 10.1 容量收益

容量收益最直接。权重从 16 bit 降到 4 bit 后，即使加入 Scale 和 Zero，通常仍接近 3 到 4 倍压缩。

### 10.2 带宽收益

单 token Decode 需要反复读取权重，INT4 能显著降低 HBM 字节。但速度提升小于压缩比，因为还有：

```text
Activation 读取
Scale/Zero 读取
INT4 解包和反量化
累加与 Epilogue
Attention、Norm、通信
调度开销
```

### 10.3 大 Batch 可能收益下降

Batch 增大后，同一权重 tile 被更多 Activation 行复用，GEMM 更接近 Compute-bound。此时 INT4 解包可能成为额外开销，高精度 Tensor Core 路径也可能更高效。

### 10.4 硬件和 Shape 约束

以下情况会影响性能：

- K/N 未满足 Kernel 对齐。
- Group 边界跨 TP Rank。
- 模型包含未支持的 Linear 变体。
- GPU 缺少合适低比特执行路径。
- Kernel 对当前 M/N/K 没有调优配置。

## 十一、精度评估与调参

### 11.1 校准集

校准集应覆盖目标输入分布。只用少量单一文本可能得到偏置的 Activation Scale。

### 11.2 需要比较的配置

```text
Group Size：32 / 64 / 128
对称或非对称量化
AWQ Scale 搜索范围
是否跳过敏感层
LM Head 是否量化
不同 Prefill/Decode batch
```

### 11.3 质量指标

- Perplexity。
- 任务准确率。
- 长文本生成质量。
- 结构化输出合法率。
- 不同语言和领域。

### 11.4 性能指标

- 模型权重显存。
- Prefill tokens/s。
- Decode tokens/s。
- TTFT 和 TPOT。
- 单个 Quantized GEMM 延迟。
- HBM 读取量。
- 反量化指令占比。

只比较模型文件大小不足以证明 Runtime 加速，必须确认实际执行的是融合低比特 Kernel。

## 十二、总结

AWQ 的完整链路是：

```text
校准 Activation
-> 识别重要输入通道
-> 搜索等价通道缩放
-> Group-wise INT4 量化
-> 打包 qweight/qzeros
-> Kernel 内融合解包、反量化和 GEMM
```

W4A16 的主要价值是降低模型权重容量和小 batch Decode 的 HBM 流量。AWQ 通过 Activation-aware Scaling 保护对输出更重要的通道，比只观察权重误差更符合真实推理目标。最终效果仍取决于校准数据、Group Size、打包格式、Tensor Parallel 切分和量化 GEMM Kernel。
