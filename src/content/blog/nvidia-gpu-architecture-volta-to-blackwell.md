---
title: 'NVIDIA GPU 架构中文梳理：From Volta to Blackwell'
description: '面向 AI Infra 和高性能计算场景，系统梳理 NVIDIA 从 Volta、Turing、Ampere、Ada、Hopper 到 Blackwell 的架构演进、代表 GPU、关键差异和代码示例。'
category: 'Research & Work'
pubDate: '2026-07-09T11:12:00+08:00'
updatedDate: '2026-07-09T11:12:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [先理解 GPU 架构和具体显卡的关系](#二先理解-gpu-架构和具体显卡的关系)
3. [从 Volta 到 Blackwell 的主线](#三从-volta-到-blackwell-的主线)
4. [Volta：Tensor Core 时代的起点](#四voltatensor-core-时代的起点)
5. [Turing：RT Core 与 INT8 推理开始进入主流](#五turingrt-core-与-int8-推理开始进入主流)
6. [Ampere：训练和推理的分水岭](#六ampere训练和推理的分水岭)
7. [Ada Lovelace：高效推理、图形和边缘部署](#七ada-lovelace高效推理图形和边缘部署)
8. [Hopper：大模型训练和 FP8 Transformer Engine](#八hopper大模型训练和-fp8-transformer-engine)
9. [Blackwell：面向超大规模生成式 AI](#九blackwell面向超大规模生成式-ai)
10. [各架构代表 GPU 分类](#十各架构代表-gpu-分类)
11. [Tensor Core 精度演进](#十一tensor-core-精度演进)
12. [互联、显存和多卡扩展演进](#十二互联显存和多卡扩展演进)
13. [简单 CUDA 代码：GPU 为什么适合并行](#十三简单-cuda-代码gpu-为什么适合并行)
14. [简单 Tensor Core 代码：为什么架构会影响 GEMM](#十四简单-tensor-core-代码为什么架构会影响-gemm)
15. [AI Infra 选型视角](#十五ai-infra-选型视角)
16. [常见误区](#十六常见误区)
17. [面试表达](#十七面试表达)
18. [总结](#十八总结)

## 一、核心结论

从 Volta 到 Blackwell，可以把 NVIDIA GPU 架构演进理解成一条主线：

```text
Volta      -> Tensor Core 正式进入深度学习训练
Turing     -> 图形 RT Core + INT8 推理能力增强
Ampere     -> TF32、稀疏 Tensor Core、A100 成为训练主力
Ada        -> 高能效推理和图形能力增强，L4/L40S/RTX 40 常见
Hopper     -> FP8 Transformer Engine，H100/H200 面向大模型训练
Blackwell  -> FP4/更强互联/更大系统级扩展，面向超大规模生成式 AI
```

如果只看 AI Infra，最重要的不是显卡名字，而是这些问题：

- 这张卡是哪代架构？
- Tensor Core 支持什么精度？
- 显存容量和带宽是多少量级？
- 是否支持 NVLink / NVSwitch？
- 适合训练、推理、图形还是边缘？
- 软件栈是否成熟？
- 多卡扩展和网络拓扑如何？

一句话判断：

```text
V100 是 Volta 时代的训练卡；
A100 是 Ampere 时代的训练/推理通用主力；
H100/H200 是 Hopper 时代的大模型训练主力；
L4/L40S/RTX 4090 是 Ada 时代常见推理/工作站选择；
B200/GB200 是 Blackwell 时代面向更大规模训练和推理集群的新平台。
```

## 二、先理解 GPU 架构和具体显卡的关系

很多人会混淆三个概念：

```text
GPU architecture
GPU chip / die
GPU product / card
```

### GPU architecture

架构是设计代际，例如：

```text
Volta
Turing
Ampere
Ada Lovelace
Hopper
Blackwell
```

架构决定一代 GPU 的基本能力：

- SM 设计。
- Tensor Core 代际。
- 支持的精度。
- Cache 和 memory hierarchy。
- NVLink / PCIe 代际。
- 对 Transformer、Ray Tracing、视频编解码等功能的支持。

### GPU chip / die

芯片是架构下的具体实现，例如同一代架构可以有多个芯片规格。数据中心大芯片、消费级芯片、移动端芯片通常不同。

### GPU product / card

显卡产品是用户买到或服务器里装的卡，例如：

```text
V100
T4
A100
RTX 3090
L4
H100
B200
```

同一架构下可以有很多产品，它们定位不同：

- 数据中心训练卡。
- 数据中心推理卡。
- 工作站卡。
- 消费级显卡。
- 边缘 / 车载 / 嵌入式模块。

因此不能只说“这张卡是 NVIDIA GPU”，要说明：

```text
它属于哪代架构，面向什么场景。
```

## 三、从 Volta 到 Blackwell 的主线

下面先给一张总览表。

| 架构 | 大致时间 | 代表 GPU | AI 相关关键词 |
| --- | --- | --- | --- |
| Volta | 2017 | V100、Titan V、Quadro GV100 | 第一代 Tensor Core、FP16 训练 |
| Turing | 2018 | T4、RTX 20、Quadro RTX、Titan RTX | RT Core、INT8/INT4 推理、图形与推理 |
| Ampere | 2020 | A100、A30、A10、A40、RTX 30、RTX A 系列 | TF32、第三代 Tensor Core、结构化稀疏 |
| Ada Lovelace | 2022 | L4、L40/L40S、RTX 40、RTX 6000 Ada | 高能效推理、图形、第四代 Tensor Core |
| Hopper | 2022 | H100、H200、GH200 | FP8 Transformer Engine、第四代 Tensor Core、NVLink 4 |
| Blackwell | 2024+ | B100、B200、GB200、RTX 50 / RTX PRO Blackwell | FP4、第五代 Tensor Core、NVLink 5、超大规模 AI |

注意：不同产品线的发布时间和软件成熟度不同。比如 Hopper 和 Ada 都在 2022 附近出现，但 Hopper 面向数据中心训练，Ada 更偏图形、推理和工作站。

## 四、Volta：Tensor Core 时代的起点

Volta 最重要的意义是：

```text
第一次把 Tensor Core 作为深度学习训练加速单元推向主流数据中心 GPU。
```

代表产品：

- Tesla V100 / NVIDIA V100。
- DGX-1 V100、DGX-2。
- Titan V。
- Quadro GV100。
- Jetson AGX Xavier 使用 Volta GPU IP，用于边缘 AI。

### Volta 的关键变化

Volta 之前，GPU 深度学习主要依赖 CUDA Core 做 FP32/FP16 计算。Volta 引入 Tensor Core 后，矩阵乘加变成专门硬件路径。

Tensor Core 的典型操作是：

```text
D = A * B + C
```

Volta 主要面向：

```text
FP16 输入，FP32 accumulate
```

这让 CNN、RNN、Transformer 早期训练都获得显著加速。

### Volta 和后续架构的关系

Volta 是“起点”：

- 后续 Ampere、Hopper、Blackwell 都继续强化 Tensor Core。
- 混合精度训练从 Volta 开始成为主流路线。
- V100 到现在仍能跑很多训练和推理任务，但在大模型场景下显存容量、带宽、精度支持和互联能力已经落后。

### Volta 的局限

- 没有 TF32。
- 没有 FP8 Tensor Core。
- 显存容量和带宽不如 A100/H100。
- 大模型推理中的 KV Cache、长上下文和低精度支持不如后续架构。

## 五、Turing：RT Core 与 INT8 推理开始进入主流

Turing 夹在 Volta 和 Ampere 之间，容易被训练用户忽略，但它很重要。

代表产品：

- Tesla T4。
- GeForce RTX 20 系列：RTX 2060、2070、2080、2080 Ti。
- Quadro RTX 4000 / 5000 / 6000 / 8000。
- Titan RTX。

### Turing 的关键变化

Turing 引入了：

- RT Core：用于光线追踪。
- Tensor Core 继续增强。
- INT8 / INT4 推理能力更实用。

T4 是非常典型的数据中心推理卡：

- 功耗低。
- 单槽小卡。
- 适合视频、推荐、传统 DNN 推理。
- 也常用于云上低成本推理实例。

### Turing 和 Volta 的区别

Volta 更偏 HPC 和训练，V100 是高端训练卡。

Turing 更偏：

- 图形渲染。
- 光线追踪。
- 低功耗推理。
- 工作站可视化。

### Turing 的局限

在大模型时代，Turing 主要问题是：

- 显存容量通常较小。
- 显存带宽有限。
- Tensor Core 代际较旧。
- 不支持 TF32/FP8。

T4 仍可用于小模型或传统模型推理，但不适合高吞吐大模型训练。

## 六、Ampere：训练和推理的分水岭

Ampere 是 AI Infra 中非常重要的一代。

代表产品：

- 数据中心训练/通用：A100。
- 数据中心推理/通用：A30、A10、A2。
- 工作站/数据中心图形：A40、RTX A6000、RTX A5000、RTX A4000、RTX A2000。
- 消费级：RTX 3090、3080、3070、3060 等。
- 边缘：Jetson AGX Orin、Orin NX、Orin Nano。

### A100 为什么重要

A100 是很多训练集群的核心卡。

关键特性：

- 第三代 Tensor Core。
- TF32。
- FP16/BF16。
- 结构化稀疏 Tensor Core。
- HBM2e。
- NVLink / NVSwitch。
- MIG。

### TF32 是什么

TF32 是 Ampere 引入的重要精度格式。

它的意义是：

```text
让很多 FP32 GEMM 在不改太多代码的情况下走 Tensor Core。
```

TF32 的指数位接近 FP32，尾数位更少。它牺牲一部分精度，换来 Tensor Core 加速。

对训练来说，TF32 降低了从纯 FP32 迁移到 Tensor Core 的门槛。

### 结构化稀疏

Ampere 支持 `2:4 sparsity` 的稀疏 Tensor Core。

含义是：

```text
每连续 4 个元素中保留 2 个非零。
```

这种稀疏是硬件友好的，因为稀疏模式受约束，硬件可以高效跳过零。

注意：这和通用非结构化稀疏不同。普通 CSR/COO 的任意稀疏不一定能直接吃到 Ampere 稀疏 Tensor Core。

### MIG

MIG 是 Multi-Instance GPU。A100 可以把一张 GPU 切成多个隔离实例。

适合：

- 多租户推理。
- 小模型服务。
- 云平台资源隔离。

不适合：

- 需要整卡大显存的大模型训练。
- 需要强多实例互联的任务。

## 七、Ada Lovelace：高效推理、图形和边缘部署

Ada Lovelace 常见于消费级 RTX 40、工作站 RTX Ada、数据中心推理卡 L4/L40S。

代表产品：

- 数据中心推理：L4。
- 数据中心图形/推理：L40、L40S。
- 工作站：RTX 6000 Ada、RTX 5000 Ada、RTX 4500 Ada、RTX 4000 Ada、RTX 4000 SFF Ada。
- 消费级：RTX 4090、4080、4070、4060 系列。

### Ada 的定位

Ada 不是 Hopper 的替代品。它更适合：

- 高性价比推理。
- 图形和渲染。
- 视频编解码。
- 工作站开发。
- 中小模型实验。

L4 是典型低功耗推理卡，常用于云上推理、视频处理、embedding / rerank / 小模型服务。

L40S 则常用于：

- 图形 + AI 混合任务。
- 推理服务。
- 轻量训练或微调。

RTX 4090 常见于个人开发和小规模实验，但数据中心部署要考虑：

- ECC。
- 散热。
- 可靠性。
- 多卡互联。
- 驱动和机房规范。

### Ada 与 Ampere 的区别

相对 Ampere，Ada 强化了：

- 图形渲染。
- RT Core。
- 能效。
- 编解码。
- 推理性能。

但对大模型训练来说，A100 仍可能比 RTX 4090 更适合数据中心场景，因为 A100 有 HBM、NVLink、MIG、ECC、数据中心软件和集群成熟度。

## 八、Hopper：大模型训练和 FP8 Transformer Engine

Hopper 是大模型训练和高端推理的重要架构。

代表产品：

- H100 SXM。
- H100 PCIe。
- H100 NVL。
- H200。
- GH200 Grace Hopper。
- 部分面向特定市场的 H20 / H800 等变体。

### Hopper 的关键变化

Hopper 引入：

- 第四代 Tensor Core。
- FP8 Transformer Engine。
- 更强的 NVLink。
- TMA：Tensor Memory Accelerator。
- DPX 指令。
- 更强的异步数据搬运和线程块集群能力。

### FP8 Transformer Engine

FP8 是 Hopper 最重要的 AI 特性之一。

大模型训练中，FP16/BF16 已经很常见，但显存和带宽仍然是瓶颈。FP8 可以进一步降低数据量。

Transformer Engine 会在训练中动态选择 FP8 格式和缩放策略，尽量兼顾速度和数值稳定性。

简单理解：

```text
BF16/FP16: 更稳，成本较高。
FP8: 成本更低，但需要更仔细的缩放和数值管理。
```

### H100 和 H200 的关系

H100 是 Hopper 时代的主力。

H200 可以理解为 Hopper 的增强版本，重点提升显存容量和带宽，对大模型长上下文、推理 KV Cache、训练大 batch 都更友好。

### Hopper 的适用场景

Hopper 适合：

- 大模型预训练。
- 大模型后训练。
- 高吞吐推理。
- 多机多卡训练。
- 需要 FP8 的 Transformer workload。

## 九、Blackwell：面向超大规模生成式 AI

Blackwell 是 Hopper 之后面向生成式 AI 的新一代架构。

代表产品和系统形态包括：

- B100。
- B200。
- GB200 Grace Blackwell。
- GB200 NVL 系统。
- RTX PRO Blackwell 工作站/专业卡。
- GeForce RTX 50 系列等 Blackwell 消费级产品线。

不同产品的定位不同，不能把 B200 和 RTX 50 直接类比。B200/GB200 面向数据中心和大规模 AI；RTX 50 更偏消费级图形、创作和本地 AI。

### Blackwell 的关键方向

Blackwell 继续强化：

- Tensor Core。
- FP4 / FP6 等更低精度。
- 更强 Transformer 推理和训练能力。
- 更高显存容量和带宽。
- 更强 NVLink / NVSwitch 系统级扩展。
- Grace CPU + Blackwell GPU 组成的超级芯片形态。

### FP4 为什么重要

大模型推理中，权重和激活带宽压力很大。FP4 可以进一步减少数据量：

```text
FP16: 2 bytes
FP8 : 1 byte
FP4 : 0.5 byte
```

理论上，FP4 相比 FP16 数据量是 1/4。

但低精度不是免费午餐：

- 需要量化策略。
- 需要 scale。
- 需要处理 outlier。
- 需要验证精度损失。
- kernel 和硬件都要支持。

### Blackwell 和 Hopper 的关系

Hopper 让 FP8 Transformer 训练和推理成为主流高端路线。

Blackwell 继续推进更低精度和更大系统规模：

```text
Hopper: FP8 + HBM + NVLink 4，大模型训练主力。
Blackwell: FP4/FP6 + 更强互联 + 更大系统级扩展，面向更大模型和更高推理吞吐。
```

## 十、各架构代表 GPU 分类

下面按架构和产品定位分类。这里列的是常见代表型号，不是所有 OEM/移动端/区域特供 SKU 的完整列表。

### Volta

| 类别 | 代表 GPU | 说明 |
| --- | --- | --- |
| 数据中心 / HPC | V100 SXM、V100 PCIe | Volta 主力训练卡 |
| 工作站 / 专业 | Quadro GV100 | 专业图形和计算 |
| 消费 / Prosumer | Titan V | 面向研究和高端个人开发 |
| 边缘 / 嵌入式 | Jetson AGX Xavier | 使用 Volta GPU IP |

### Turing

| 类别 | 代表 GPU | 说明 |
| --- | --- | --- |
| 数据中心推理 | Tesla T4 | 低功耗推理、视频、传统 DNN |
| 工作站 / 专业 | Quadro RTX 4000/5000/6000/8000 | RT + 专业图形 |
| 消费级 | RTX 2060/2070/2080/2080 Ti | 第一代 RTX 消费卡 |
| Prosumer | Titan RTX | 高端桌面计算和图形 |

### Ampere

| 类别 | 代表 GPU | 说明 |
| --- | --- | --- |
| 数据中心训练 | A100 SXM、A100 PCIe | Ampere 训练主力 |
| 数据中心推理/通用 | A30、A10、A2 | 推理、视频、轻量训练 |
| 数据中心图形/计算 | A40 | 图形、渲染、推理 |
| 工作站 | RTX A6000、A5000、A4000、A2000 | 专业工作站 |
| 消费级 | RTX 3090/3080/3070/3060 | 个人开发和图形 |
| 边缘 | Jetson AGX Orin、Orin NX、Orin Nano | 边缘 AI |

### Ada Lovelace

| 类别 | 代表 GPU | 说明 |
| --- | --- | --- |
| 数据中心推理 | L4 | 低功耗推理、视频、云实例 |
| 数据中心图形/推理 | L40、L40S | 推理、渲染、工作站服务 |
| 工作站 | RTX 6000 Ada、RTX 5000 Ada、RTX 4500 Ada、RTX 4000 Ada | 专业图形和 AI |
| 消费级 | RTX 4090/4080/4070/4060 系列 | 本地开发、图形、轻量 AI |

### Hopper

| 类别 | 代表 GPU | 说明 |
| --- | --- | --- |
| 数据中心训练 | H100 SXM、H100 PCIe | Hopper 主力 |
| 大显存训练/推理 | H200 | 更高显存容量和带宽 |
| 超级芯片 | GH200 Grace Hopper | Grace CPU + Hopper GPU |
| 双卡 / 推理 | H100 NVL | 面向大模型推理和高显存场景 |
| 区域/特定市场变体 | H20、H800 等 | 规格和定位依市场不同 |

### Blackwell

| 类别 | 代表 GPU / 系统 | 说明 |
| --- | --- | --- |
| 数据中心训练/推理 | B100、B200 | Blackwell 数据中心 GPU |
| 超级芯片 / 机柜级系统 | GB200、GB200 NVL | Grace CPU + Blackwell GPU，面向超大规模 AI |
| 专业工作站 | RTX PRO Blackwell 系列 | 专业图形、仿真、本地 AI |
| 消费级 | GeForce RTX 50 系列 | 图形、创作、本地 AI |

## 十一、Tensor Core 精度演进

Tensor Core 是理解 NVIDIA AI GPU 的关键。

| 架构 | Tensor Core 重点 | 常见 AI 精度 |
| --- | --- | --- |
| Volta | 第一代 Tensor Core | FP16 输入，FP32 accumulate |
| Turing | Tensor Core + INT 推理增强 | FP16、INT8、INT4 |
| Ampere | TF32、BF16、结构化稀疏 | TF32、FP16、BF16、INT8、2:4 sparsity |
| Ada | 第四代 Tensor Core，高效推理 | FP16、BF16、TF32、INT8、稀疏 |
| Hopper | FP8 Transformer Engine | FP8、BF16、FP16、TF32 |
| Blackwell | FP4/FP6/FP8 等更低精度 | FP4、FP6、FP8、BF16、FP16 |

### 为什么精度很重要

矩阵乘法的性能受三件事影响：

```text
计算吞吐
显存带宽
显存容量
```

低精度可以同时减少：

- 读写字节数。
- 显存占用。
- Tensor Core 每次处理的数据量。

但低精度会带来数值风险，所以训练和推理都需要：

- loss scaling。
- quantization scale。
- calibration。
- outlier 处理。
- 精度回退策略。

## 十二、互联、显存和多卡扩展演进

AI Infra 不只看单卡 FLOPS，还要看多卡系统。

### 显存类型

数据中心训练卡通常使用 HBM：

```text
V100: HBM2
A100: HBM2e
H100/H200: HBM3 / HBM3e
B200/GB200: 更高带宽 HBM3e 级别
```

消费级和部分推理卡通常使用 GDDR：

```text
RTX 4090: GDDR6X
L4: GDDR6
L40S: GDDR6
```

HBM 带宽更高，适合训练和高带宽大模型推理；GDDR 成本和部署形态更灵活。

### NVLink / NVSwitch

多卡训练中，GPU 之间通信很重要。

```text
PCIe: 通用，但带宽较低、延迟更高。
NVLink: GPU-GPU 高速互联。
NVSwitch: 多 GPU 全互联交换。
```

H100/B200 这类数据中心 GPU 的多卡系统，价值很大一部分来自互联能力，而不只是单卡算力。

### 为什么 RTX 4090 不等于数据中心训练卡

RTX 4090 单卡算力很强，但和 H100/A100 的定位不同：

- 通常没有数据中心级 NVLink。
- 显存是 GDDR，不是 HBM。
- 数据中心可靠性、ECC、散热、运维支持不同。
- 多机多卡通信能力不同。

所以本地实验可以用 RTX 4090，但大规模训练集群通常更偏 A100/H100/H200/B200。

## 十三、简单 CUDA 代码：GPU 为什么适合并行

下面用向量加法说明 GPU 的基本并行模型。

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

void launch(const float* a, const float* b, float* c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(a, b, c, n);
}
```

这段代码和架构的关系：

- `threadIdx.x` 是 block 内线程。
- `blockIdx.x` 是 block 编号。
- GPU 把大量线程调度到 SM 上执行。
- 不同架构的 SM 数量、寄存器、shared memory、scheduler、cache 都会影响性能。

比如同样的 kernel：

```text
V100、A100、H100、B200 都能跑；
但显存带宽、SM 数、调度能力不同，性能差异会很大。
```

## 十四、简单 Tensor Core 代码：为什么架构会影响 GEMM

Tensor Core 主要加速矩阵乘。下面是 CUDA WMMA 的简化示例。

```cpp
#include <mma.h>
using namespace nvcuda;

__global__ void simple_wmma(const half* A, const half* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}
```

这段代码表达的是：

```text
C = A * B + C
```

不同架构下，Tensor Core 支持的输入精度和 tile 形状不同：

- Volta：FP16 Tensor Core 是重点。
- Ampere：TF32/BF16/结构化稀疏很重要。
- Hopper：FP8 Transformer Engine 很重要。
- Blackwell：FP4/FP6 等更低精度更重要。

所以同样是 GEMM，架构会决定：

- 能不能走 Tensor Core。
- 走什么精度。
- 是否需要改模型 dtype。
- 是否需要量化。
- 是否能使用稀疏 Tensor Core。

## 十五、AI Infra 选型视角

### 大模型预训练

优先考虑：

- H100/H200。
- B200/GB200。
- A100 仍可用，但效率低于 Hopper/Blackwell。

关键指标：

- HBM 容量和带宽。
- FP8/FP4 支持。
- NVLink/NVSwitch。
- 多机网络。
- 软件栈成熟度。

### 大模型推理

取决于模型规模和吞吐目标。

常见选择：

- H100/H200/B200：高吞吐、大模型、长上下文。
- L40S：中高端推理、图形混合。
- L4：低功耗推理、视频、轻量模型。
- RTX 4090/5090 类：本地开发、小规模推理。

关注：

- KV Cache 显存。
- 显存带宽。
- FP8/INT8/FP4 支持。
- batch 和并发。
- 多卡通信。

### CUDA kernel 学习和实验

常见：

- RTX 30 / RTX 40 / RTX 50。
- RTX A 系列。
- L4 / A10。

学习 CUDA 不一定需要 H100。很多线程模型、shared memory、warp、coalescing、reduction、GEMM tiling，都可以在消费级卡上练。

但学习 FP8 Transformer Engine、Hopper TMA、Blackwell FP4，就需要对应架构硬件或至少阅读官方文档和开源 kernel。

### 边缘 AI

常见：

- Jetson Xavier：Volta IP。
- Jetson Orin：Ampere IP。

关注：

- 功耗。
- 显存共享。
- TensorRT 支持。
- 视频编解码。
- 端侧部署工具链。

## 十六、常见误区

### 误区 1：只看 CUDA core 数

AI 计算更应该看：

- Tensor Core。
- 显存容量。
- 显存带宽。
- 精度支持。
- 通信互联。
- 软件栈。

CUDA core 对图形和通用计算有意义，但不能单独代表大模型性能。

### 误区 2：RTX 4090 一定比 A100 好

RTX 4090 在某些单卡 FP16 任务上很强，但 A100 有：

- HBM。
- ECC。
- NVLink。
- MIG。
- 数据中心部署支持。
- 更成熟的多卡训练生态。

所以要看场景。

### 误区 3：H100 只是 A100 的更快版本

H100 不只是更快。它引入了：

- FP8 Transformer Engine。
- 更强互联。
- TMA 等新机制。

这对大模型训练和推理 kernel 设计都有影响。

### 误区 4：低精度一定无损

FP8、FP4、INT8 都需要：

- scale。
- calibration。
- 数值监控。
- fallback。

低精度是性能工具，不是无脑替换。

### 误区 5：Blackwell 只是单卡升级

Blackwell 的重点之一是系统级扩展。GB200/机柜级系统关注的是大规模 GPU 互联、CPU-GPU 协同和更低精度吞吐，不只是单卡 FLOPS 增加。

## 十七、面试表达

可以这样回答“Volta 到 Blackwell 的演进”：

```text
Volta 是 Tensor Core 训练时代的起点，V100 让 FP16 mixed precision 训练成为主流。
Turing 强化图形 RT Core 和 INT8 推理，T4 是典型低功耗推理卡。
Ampere 引入 TF32、BF16 和结构化稀疏，A100 成为训练和推理通用主力。
Ada 更偏高能效推理、图形和工作站，比如 L4、L40S、RTX 4090。
Hopper 面向大模型训练，核心是 FP8 Transformer Engine 和更强互联，代表是 H100/H200。
Blackwell 继续推进 FP4、更大显存和系统级互联，面向更大规模生成式 AI。
```

如果问“训练卡和推理卡怎么区分”：

```text
训练更看重 HBM 容量/带宽、Tensor Core 精度、NVLink/NVSwitch、多机扩展和稳定性；
推理更看重单位成本吞吐、显存能否放下模型和 KV Cache、低精度支持、功耗和部署密度。
所以 H100/B200 更偏高端训练和大模型推理，L4/L40S 更常见于推理和图形混合，RTX 4090 更适合开发和小规模实验。
```

如果问“为什么 GPU 架构会影响 kernel 优化”：

```text
因为不同架构的 SM、Tensor Core、shared memory、L2、显存带宽、异步拷贝和支持精度不同。
同一个 GEMM 或 attention kernel，在 Ampere 上可能用 TF32/BF16，在 Hopper 上会考虑 FP8 和 TMA，在 Blackwell 上还要考虑 FP4 和新一代互联/系统能力。
```

## 十八、总结

从 Volta 到 Blackwell，NVIDIA GPU 的 AI 演进可以概括成：

```text
更多 Tensor Core 能力
更低数值精度
更高显存带宽
更大显存容量
更强多卡互联
更完整软件栈
```

架构之间不是孤立的：

- Volta 开启 Tensor Core 训练。
- Turing 把 RTX、推理和图形能力推向主流。
- Ampere 让 TF32 和 A100 成为训练集群核心。
- Ada 支撑高效推理、图形和本地 AI。
- Hopper 进入 FP8 大模型训练时代。
- Blackwell 继续面向更低精度和更大规模系统。

最重要的学习方式不是背型号，而是看到一张 GPU 时能问出这些问题：

```text
它是哪代架构？
面向训练、推理、图形还是边缘？
Tensor Core 支持什么精度？
显存容量和带宽是否够？
多卡互联能力如何？
软件和部署生态是否成熟？
```

能回答这些问题，就基本能理解不同 GPU 之间的关系和区别。
