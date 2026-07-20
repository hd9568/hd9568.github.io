---
title: 'MoE 推理优化：从 Router、Expert Dispatch 到 TensorRT-LLM FusedMoE'
description: '系统讲解大模型 MoE 推理中的路由、token 重排、专家计算、combine、负载均衡、Expert Parallel 和 fused MoE kernel，并结合 TensorRT-LLM 源码说明工程实现。'
category: 'Research & Work'
pubDate: '2026-07-13T16:53:00+08:00'
updatedDate: '2026-07-13T16:53:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [MoE 是什么](#二moe-是什么)
3. [MoE 前向计算流程](#三moe-前向计算流程)
4. [为什么 MoE 推理难优化](#四为什么-moe-推理难优化)
5. [Router 和 Top-K](#五router-和-top-k)
6. [Token Dispatch 和 Combine](#六token-dispatch-和-combine)
7. [专家 GEMM：MoE 的主要计算](#七专家-gemmmoe-的主要计算)
8. [负载均衡问题](#八负载均衡问题)
9. [并行策略：EP、TP、DP](#九并行策略eptpdp)
10. [TensorRT-LLM 中的 MoE backend](#十tensorrt-llm-中的-moe-backend)
11. [TRTLLMGenFusedMoE 源码要点](#十一trtllmgenfusedmoe-源码要点)
12. [简化代码示例](#十二简化代码示例)
13. [优化 checklist](#十三优化-checklist)
14. [面试表达](#十四面试表达)
15. [总结](#十五总结)

## 一、核心结论

MoE 是 Mixture of Experts。它的核心思想是：

```text
模型里有很多专家 FFN，但每个 token 只激活其中少数几个专家。
```

相比 dense FFN，MoE 可以扩大参数量，但每个 token 的计算量不按专家总数线性增长。

推理优化真正难点不只是专家 GEMM，而是整个链路：

```text
router logits
top-k expert selection
token dispatch
expert-local GEMM
activation
second GEMM
combine
跨卡 all-to-all / all-gather / reduce-scatter
load balance
```

MoE 高性能实现通常需要：

- fused routing。
- token 重排。
- grouped GEMM。
- expert parallel。
- all-to-all 通信优化。
- load balancing。
- FP8 / FP4 / block scaling。
- CUDA Graph 和 autotune。

## 二、MoE 是什么

普通 Transformer FFN：

```text
y = W2 * activation(W1 * x)
```

每个 token 都走同一套 FFN 权重。

MoE FFN：

```text
router_logits = router(x)
topk_experts, topk_weights = topk(router_logits)
y = sum_{e in topk} topk_weight[e] * Expert_e(x)
```

其中每个 expert 通常也是一个 FFN：

```text
Expert_e(x) = W2_e * activation(W1_e * x)
```

如果：

```text
num_experts = 128
top_k = 2
```

每个 token 只经过 2 个专家，而不是 128 个专家。

## 三、MoE 前向计算流程

以 batch 中有 `T` 个 token 为例：

```text
hidden_states: [T, H]
router_weight: [H, E]
router_logits = hidden_states @ router_weight    # [T, E]
topk_ids, topk_weights = topk(router_logits, k)
```

然后对 token 做 dispatch：

```text
token 0 -> expert 3, expert 17
token 1 -> expert 3, expert 5
token 2 -> expert 91, expert 17
...
```

按 expert 分组：

```text
expert 3:  token 0, token 1, ...
expert 5:  token 1, ...
expert 17: token 0, token 2, ...
```

每个 expert 做自己的 FFN：

```text
Y_e = Expert_e(X_e)
```

最后 combine：

```text
output[token] += topk_weight[token, slot] * Y_e[token]
```

这就是 MoE 的基本执行路径。

## 四、为什么 MoE 推理难优化

### 1. 每个专家的 token 数不固定

不同请求、不同 token、不同 batch 下，专家负载会变化。

例如：

```text
expert 0: 512 tokens
expert 1: 7 tokens
expert 2: 0 tokens
expert 3: 290 tokens
```

这会导致 GEMM 的 M 维不规则。

### 2. 小 GEMM 多

每个 expert 都要做 GEMM，但每个 expert 的 token 数可能很小。

如果逐 expert 调 cuBLAS：

```text
for expert in experts:
    gemm(tokens_of_expert, W_expert)
```

会出现大量小 GEMM 和 kernel launch overhead。

### 3. token 重排有成本

为了让 expert GEMM 连续访问，需要把 token 按 expert 分组。这会引入：

- scatter。
- gather。
- prefix sum。
- index mapping。
- combine。

### 4. 跨卡通信复杂

Expert Parallel 下，每张卡只保存部分 experts。token 如果路由到远端 expert，就需要 all-to-all。

```text
rank 0 上的 token -> rank 3 的 expert
rank 1 上的 token -> rank 0 的 expert
```

通信开销可能成为瓶颈。

## 五、Router 和 Top-K

Router 通常是一个线性层：

```python
router_logits = hidden_states @ router_weight
```

然后选 top-k experts：

```python
topk_scores, topk_ids = torch.topk(router_logits, k=top_k, dim=-1)
topk_weights = torch.softmax(topk_scores, dim=-1)
```

简化例子：

```text
num_experts = 4
top_k = 2

token 0 logits: [1.0, 0.2, 3.0, 2.0]
top2 experts: expert 2, expert 3
top2 weights: softmax([3.0, 2.0]) = [0.731, 0.269]
```

Router 的优化点：

- top-k kernel。
- softmax / sigmoid routing。
- group-limited routing。
- bias correction。
- fused routing + dispatch。
- expert load balancing。

## 六、Token Dispatch 和 Combine

Dispatch 是把 token 送到专家。

原始 token 顺序：

```text
token0, token1, token2, token3
```

按专家重排后：

```text
expert0: token2
expert1: token0, token3
expert2: token1
```

实现上通常需要：

```text
expert_counts
expert_offsets
permuted_token_indices
```

伪代码：

```python
expert_buckets = [[] for _ in range(num_experts)]

for t in range(num_tokens):
    for s in range(top_k):
        e = topk_ids[t, s]
        expert_buckets[e].append((t, s))
```

Combine 是反向过程：

```python
out = torch.zeros_like(hidden_states)

for e in range(num_experts):
    for local_idx, (t, s) in enumerate(expert_buckets[e]):
        out[t] += topk_weights[t, s] * expert_output[e][local_idx]
```

高性能实现不会用 Python list，而是用 prefix sum + CUDA kernel 完成。

## 七、专家 GEMM：MoE 的主要计算

每个 expert 通常有两个 GEMM：

```text
gate/up GEMM: [M_e, H] x [H, 2I] -> [M_e, 2I]
down GEMM:    [M_e, I] x [I, H]  -> [M_e, H]
```

其中：

```text
M_e = 分配给 expert e 的 token 数
```

不同 expert 的 `M_e` 不同，所以很适合 grouped GEMM：

```text
problem 0: M0 x H x I
problem 1: M1 x H x I
problem 2: M2 x H x I
...
```

Grouped GEMM 可以在一个 kernel 或一个调度框架中处理多个 GEMM problem，减少 launch overhead，并更好利用 GPU。

## 八、负载均衡问题

MoE 的性能常被最忙 expert 决定。

例如 8 个 experts：

```text
tokens per expert:
[100, 98, 105, 93, 102, 97, 99, 101]
```

负载均衡很好。

但如果：

```text
[500, 20, 10, 15, 8, 4, 3, 2]
```

expert 0 会成为瓶颈。

常见方法：

- 训练时加 load balancing loss。
- 推理时做 expert replica。
- expert parallel 中动态负载均衡。
- capacity / padding。
- routing 限制。
- token dropping，通常推理不希望丢。

## 九、并行策略：EP、TP、DP

### Expert Parallel

Expert Parallel，简称 EP。

```text
不同 GPU 持有不同 experts。
```

优点：

- 每张卡只存部分 expert 权重。
- 支持更多专家。

问题：

- token 需要跨卡 all-to-all。
- 通信和计算需要 overlap。

### Tensor Parallel

Tensor Parallel，简称 TP。

```text
同一个 expert 的权重被切到多张 GPU。
```

适合单个 expert 很大时，但通信也复杂。

### Data Parallel

Data Parallel，简称 DP。

```text
每张卡有完整模型副本，处理不同 batch。
```

MoE 中可能和 EP 组合，例如 Attention DP + MoE EP。

## 十、TensorRT-LLM 中的 MoE backend

TensorRT-LLM 提供了多种 MoE backend：

```text
fused_moe_cutlass.py
fused_moe_triton.py
fused_moe_trtllm_gen.py
fused_moe_densegemm.py
fused_moe_deepgemm.py
fused_moe_cute_dsl.py
fused_moe_wide_ep.py
```

`create_moe.py` 里根据配置选择 backend：

```python
if moe_backend.upper() == "CUTLASS":
    return CutlassFusedMoE
elif moe_backend.upper() == "TRTLLM":
    ...
    return TRTLLMGenFusedMoE
elif moe_backend.upper() == "DENSEGEMM":
    ...
    return DenseGEMMFusedMoE
```

这说明工程上 MoE 没有单一最优实现，而是根据：

- GPU 架构。
- quantization。
- expert 数。
- hidden size。
- parallel strategy。
- 是否有 FlashInfer。
- 是否使用 FP8/NVFP4。

选择不同 backend。

## 十一、TRTLLMGenFusedMoE 源码要点

`fused_moe_trtllm_gen.py` 中 `TRTLLMGenFusedMoE` 描述了 Blackwell 上的 fused MoE 路径。

源码注释很关键：

```text
FusedMoE Op:
routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute
```

也就是把 MoE 的核心过程融合：

```text
routing
scatter / dispatch
GEMM1
activation
GEMM2
finalize / combine
```

支持检查：

```python
if sm_version not in {100, 103}:
    return _warn_and_return(
        f"TRTLLMGenFusedMoE requires SM100 or SM103, got SM{sm_version}"
    )
```

说明 TRTLLMGenFusedMoE 是面向 Blackwell SM100/SM103 的路径。

支持的量化包括：

```python
QuantAlgo.NVFP4
QuantAlgo.FP8_BLOCK_SCALES
QuantAlgo.W4A8_NVFP4_FP8
QuantAlgo.W4A16_MXFP4
QuantAlgo.W4A8_MXFP4_FP8
QuantAlgo.W4A8_MXFP4_MXFP8
```

这和 B200 / Blackwell 上 MoE 优化高度相关。

## 十二、简化代码示例

下面是一个最小 MoE forward，重点理解语义。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.w1 = nn.Linear(hidden, inter)
        self.w2 = nn.Linear(inter, hidden)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

class SimpleMoE(nn.Module):
    def __init__(self, hidden, inter, num_experts, top_k):
        super().__init__()
        self.router = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(hidden, inter) for _ in range(num_experts)
        ])
        self.top_k = top_k

    def forward(self, x):
        # x: [T, H]
        logits = self.router(x)                  # [T, E]
        scores, ids = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(scores, dim=-1)  # [T, top_k]

        out = torch.zeros_like(x)

        for e, expert in enumerate(self.experts):
            # 找出路由到 expert e 的 token
            mask = ids == e
            if not mask.any():
                continue

            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            x_e = x[token_idx]
            y_e = expert(x_e)
            out[token_idx] += weights[token_idx, slot_idx].unsqueeze(-1) * y_e

        return out
```

这段代码语义清楚，但性能很差：

- Python 循环专家。
- 每个 expert 单独 GEMM。
- token gather/scatter 多。
- 没有 grouped GEMM。
- 没有通信 overlap。

高性能 MoE 就是在优化这些问题。

## 十三、优化 checklist

MoE 推理优化可以按这张表排查：

| 环节 | 问题 | 优化方向 |
| --- | --- | --- |
| Router | top-k 慢 | fused top-k / routing kernel |
| Dispatch | token scatter 慢 | prefix sum、连续化、融合 |
| Expert GEMM | 小 GEMM 多 | grouped GEMM、persistent kernel |
| Load balance | expert token 数不均 | load balancing、replica、routing 约束 |
| Combine | scatter add 慢 | fused combine、低精度 combine |
| EP 通信 | all-to-all 慢 | DeepEP、NVLink all-to-all、overlap |
| Quant | 带宽压力大 | FP8、NVFP4、block scale |
| Runtime | launch overhead | CUDA Graph、fused op |

## 十四、面试表达

可以这样解释 MoE 优化：

```text
MoE 的核心不是只做多个 expert FFN，而是一个动态稀疏计算问题。
每个 token 通过 router 选择 top-k experts，随后需要把 token 按 expert 重排，执行每个 expert 的 GEMM，再按权重 combine 回原 token 顺序。
推理优化难点在于每个 expert 的 token 数不固定，导致小 GEMM 多、负载不均和跨卡 all-to-all。
高性能实现通常使用 fused routing、token dispatch、grouped GEMM、expert parallel、load balancing 和低精度量化。
```

如果结合 TensorRT-LLM：

```text
TensorRT-LLM 中 MoE 有多个 backend，例如 Cutlass、Triton、TRTLLMGen、DenseGEMM、DeepGEMM。
TRTLLMGenFusedMoE 面向 SM100/SM103 Blackwell，支持 NVFP4、FP8 block scales 等量化路径，把 routing、scatter、GEMM1、SwiGLU、GEMM2 和 finalize 融合到高性能 MoE op 中。
```

## 十五、总结

MoE 的性能优化主线是：

```text
减少动态路由带来的调度开销；
让专家计算变成高效 grouped GEMM；
减少 dispatch/combine 的访存；
在多卡下优化 all-to-all；
用低精度降低带宽和显存压力；
用 fused kernel 和 CUDA Graph 降低 launch overhead。
```

MoE 不是单个 GEMM 问题，而是：

```text
动态稀疏路由 + 不规则 GEMM + 通信 + 内存重排
```

这也是为什么 MoE 推理优化比 dense FFN 更复杂。
