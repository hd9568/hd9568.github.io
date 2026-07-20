---
title: 'Grouped GEMM 算子介绍：为什么 MoE 和不规则 batch 需要它'
description: '系统讲解 GEMM、Batched GEMM、Grouped GEMM 的区别，结合 MoE expert 计算说明 grouped GEMM 的形状组织、指针数组、调度收益，并参考 TensorRT-LLM 源码介绍工程实现。'
category: 'Research & Work'
pubDate: '2026-07-13T16:54:00+08:00'
updatedDate: '2026-07-13T16:54:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [GEMM、Batched GEMM、Grouped GEMM 的区别](#二gemmbatched-gemmgrouped-gemm-的区别)
3. [为什么 MoE 需要 grouped GEMM](#三为什么-moe-需要-grouped-gemm)
4. [Grouped GEMM 的输入组织](#四grouped-gemm-的输入组织)
5. [一个具体数字例子](#五一个具体数字例子)
6. [朴素实现和问题](#六朴素实现和问题)
7. [C++ 简化接口](#七c-简化接口)
8. [TensorRT-LLM 源码中的 grouped GEMM](#八tensorrt-llm-源码中的-grouped-gemm)
9. [Blackwell block-scaled grouped GEMM](#九blackwell-block-scaled-grouped-gemm)
10. [性能优化关注点](#十性能优化关注点)
11. [面试表达](#十一面试表达)
12. [总结](#十二总结)

## 一、核心结论

Grouped GEMM 解决的是：

```text
一次要执行很多个 GEMM，而且每个 GEMM 的 M/N/K 形状可能不一样。
```

它和普通 GEMM 的区别：

```text
普通 GEMM:        C = A @ B
Batched GEMM:     多个相同形状 GEMM
Grouped GEMM:     多个不同形状 GEMM
```

MoE 推理中，每个 expert 分到的 token 数不同，因此每个 expert 的 GEMM M 维不同：

```text
expert 0: [M0, H] x [H, I]
expert 1: [M1, H] x [H, I]
expert 2: [M2, H] x [H, I]
...
```

如果逐 expert 调 GEMM，会有大量小 kernel launch。Grouped GEMM 可以把这些问题组织到一个更高效的调度中。

## 二、GEMM、Batched GEMM、Grouped GEMM 的区别

### GEMM

GEMM 是矩阵乘：

```text
C[M, N] = A[M, K] @ B[K, N]
```

常见于：

- Linear layer。
- FFN。
- Attention projection。

### Batched GEMM

Batched GEMM 是多个形状相同的 GEMM：

```text
for b in batch:
    C[b] = A[b] @ B[b]
```

形状通常固定：

```text
A: [B, M, K]
B: [B, K, N]
C: [B, M, N]
```

适合 attention 中按 head/batch 做小矩阵乘。

### Grouped GEMM

Grouped GEMM 是多个 GEMM problem 组成一组，每个 problem 可以有不同形状：

```text
C0[M0, N0] = A0[M0, K0] @ B0[K0, N0]
C1[M1, N1] = A1[M1, K1] @ B1[K1, N1]
C2[M2, N2] = A2[M2, K2] @ B2[K2, N2]
```

在 MoE 中通常是：

```text
K 和 N 相同或相近，M 不同。
```

例如每个 expert 的 hidden size 和 intermediate size 相同，但 token 数不同。

## 三、为什么 MoE 需要 grouped GEMM

MoE 中每个 token 只路由到少数 experts。

假设：

```text
num_tokens = 1024
num_experts = 8
top_k = 2
hidden = 4096
intermediate = 14336
```

总 expert-token 数：

```text
1024 * 2 = 2048
```

路由后可能是：

```text
expert 0: 400 tokens
expert 1: 360 tokens
expert 2: 300 tokens
expert 3: 260 tokens
expert 4: 240 tokens
expert 5: 220 tokens
expert 6: 180 tokens
expert 7: 88 tokens
```

每个 expert 的第一个 GEMM：

```text
[M_e, 4096] x [4096, 14336]
```

`M_e` 不同，因此不能简单用 regular batched GEMM。

逐个 expert 调 GEMM：

```python
for e in range(num_experts):
    y_e = x_e @ w1_e
```

问题：

- kernel launch 多。
- 小 M GEMM 利用率低。
- GPU tile 调度不均。
- 很难统一做 persistent scheduling。
- 很难和 activation / down projection 融合。

Grouped GEMM 就是为这种场景设计的。

## 四、Grouped GEMM 的输入组织

Grouped GEMM 通常需要：

```text
problem_count
problem_sizes
ptrA[]
ptrB[]
ptrC[]
ptrD[]
workspace
```

其中每个 problem 有自己的：

```text
M, N, K
A pointer
B pointer
C/D pointer
leading dimension
```

对于 MoE：

```text
ptrA[e] -> expert e 的 token buffer
ptrB[e] -> expert e 的 weight
ptrD[e] -> expert e 的 output buffer
```

problem size：

```text
GemmCoord(M_e, N, K)
```

`M_e` 是路由到 expert e 的 token 数。

## 五、一个具体数字例子

假设 4 个 experts：

```text
hidden = 4096
intermediate = 11008
```

路由结果：

```text
expert 0: M0 = 128
expert 1: M1 = 64
expert 2: M2 = 256
expert 3: M3 = 32
```

第一个 FFN GEMM：

```text
problem 0: [128, 4096] x [4096, 11008]
problem 1: [ 64, 4096] x [4096, 11008]
problem 2: [256, 4096] x [4096, 11008]
problem 3: [ 32, 4096] x [4096, 11008]
```

这就是 grouped GEMM 的典型输入。

如果用普通 GEMM，需要 4 次调用。

如果用 grouped GEMM，可以把 4 个 problem 一起提交，让 kernel / library 内部做统一调度。

## 六、朴素实现和问题

朴素 PyTorch 写法：

```python
outputs = []
for e in range(num_experts):
    x_e = expert_inputs[e]      # [M_e, H]
    w_e = expert_weights[e]     # [H, I]
    y_e = x_e @ w_e             # [M_e, I]
    outputs.append(y_e)
```

问题：

```text
num_experts 多时，GEMM 调用次数多。
M_e 小时，单个 GEMM 不能打满 GPU。
每次调用都有调度开销。
不同 expert 负载不均。
```

如果 `num_experts=128`，每层 MoE 有两个 GEMM，理论上可能有：

```text
128 * 2 = 256
```

个 expert GEMM problem。

这还不包括 dispatch、activation、combine。

## 七、C++ 简化接口

一个 grouped GEMM 接口可以抽象成：

```cpp
struct GemmProblem {
    int M;
    int N;
    int K;
    const void* A;
    const void* B;
    void* C;
};

void grouped_gemm(const std::vector<GemmProblem>& problems) {
    for (auto& p : problems) {
        // 实际实现不会简单循环，而是统一调度这些 problem
        launch_or_schedule_gemm(p.M, p.N, p.K, p.A, p.B, p.C);
    }
}
```

实际高性能实现会做：

- problem 分组。
- tile scheduler。
- persistent CTA。
- workspace 保存参数。
- 根据 shape 选择 kernel。
- 处理不同 dtype 和 scale。
- 合并 epilogue。

## 八、TensorRT-LLM 源码中的 grouped GEMM

TensorRT-LLM 在 `groupGemm.h` 中声明了 C++ grouped GEMM 接口。

核心声明：

```cpp
int64_t getGroupedGemmParamsWorkSpaceSize(int64_t problem_count);

void groupedGemm(
    std::vector<cutlass::gemm::GemmCoord> problem_sizes,
    std::vector<void*> const& ptrA,
    std::vector<void*> const& ptrB,
    std::vector<void*> const& ptrC,
    std::vector<void*> const& ptrD,
    void* gemmParamsWorkspace,
    int64_t gemmParamsWorkSpaceSize,
    void* gemmWorkSpace,
    int64_t gemmWorkspaceSize,
    bool isLoraIn,
    nvinfer1::DataType dataType,
    int minKN,
    cudaStream_t stream);
```

这和前面讲的抽象完全对应：

- `problem_sizes`：每个 GEMM 的 M/N/K。
- `ptrA/ptrB/ptrC/ptrD`：每个 problem 的输入输出指针。
- `gemmParamsWorkspace`：保存 grouped GEMM 参数。
- `gemmWorkSpace`：GEMM 运行时 workspace。
- `dataType`：FP16/BF16 等。
- `stream`：CUDA stream。

`lora.cpp` 中也使用了这个接口。

LoRA 和 MoE 类似，也可能产生多个小 GEMM problem，因此 grouped GEMM 同样有价值。

## 九、Blackwell block-scaled grouped GEMM

TensorRT-LLM 的 `blockscaled_contiguous_grouped_gemm.py` 还实现了 Blackwell Cute DSL grouped GEMM。

类名：

```python
Sm100BlockScaledContiguousGroupedGemmKernel
```

源码注释说明它面向 Blackwell：

```text
batched matrix multiplication
C = A x SFA x B x SFB
support block-scaled data types
persistent tile scheduling
warp specialization
```

支持的类型包括：

```text
MXF8
MXF4
NVF4
```

这些和 Blackwell / B200 上的 FP4、FP8、block scaling 路线相关。

它的配置包含：

- MMA tile shape。
- cluster shape。
- accumulator dtype。
- scale factor vector size。
- shared memory layout。
- TMA / tcgen05。
- persistent scheduler。

这说明 B200 上的 grouped GEMM 不只是“多个 GEMM 放一起”，还会结合：

```text
低精度 block scaling
Blackwell Tensor Core
CTA cluster
TMA 数据搬运
persistent scheduling
```

## 十、性能优化关注点

### 1. M 维分布

MoE 中 `M_e` 分布决定 grouped GEMM 的效率。

如果很多 expert 的 `M_e` 很小，调度难度大。

### 2. expert 排序和对齐

通常会把 token 按 expert 排序或分桶，让每个 expert 的输入连续。

### 3. padding

为了对齐 tile，可能会 padding 到 8、16、32 等倍数。

padding 增加计算，但提升 kernel 规则性。

### 4. dtype 和量化

Grouped GEMM 常结合：

- FP16/BF16。
- FP8。
- NVFP4。
- MXFP4。
- block scale。

低精度能降低带宽和提高 Tensor Core 利用率，但要处理 scale。

### 5. launch overhead

Grouped GEMM 的价值之一就是减少多次小 GEMM launch。

### 6. epilogue fusion

MoE 中 GEMM 后常接：

- bias。
- activation。
- scale。
- combine。

能融合就减少写回和读回。

## 十一、面试表达

可以这样回答 grouped GEMM：

```text
Grouped GEMM 是一组不同形状 GEMM 的批量调度接口。
它和 batched GEMM 不同，batched GEMM 通常要求每个 GEMM 形状相同，而 grouped GEMM 允许每个 problem 有不同 M/N/K。
MoE 中每个 expert 分到的 token 数不同，所以每个 expert 的 GEMM M 维不同，非常适合 grouped GEMM。
```

如果问为什么不用循环 GEMM：

```text
逐 expert 调 GEMM 会产生大量小 kernel launch，而且小 M GEMM 很难打满 GPU。
Grouped GEMM 可以把多个 expert GEMM 合并调度，减少 launch overhead，并让 kernel 内部做更好的 tile scheduling。
```

结合 TensorRT-LLM：

```text
TensorRT-LLM 的 groupGemm 接口接收 problem_sizes 和每个 problem 的 A/B/C/D 指针数组。
在 Blackwell 路径中，还有 block-scaled contiguous grouped GEMM，结合 FP4/FP8 scale factor、tcgen05 Tensor Core、TMA 和 persistent scheduler 来优化 MoE 场景。
```

## 十二、总结

Grouped GEMM 是 MoE 推理优化的基础算子之一。

它解决的问题是：

```text
专家数量多；
每个专家 token 数不同；
每个专家都要做 GEMM；
逐个 GEMM 调用开销大、利用率低。
```

核心输入是：

```text
problem_sizes + ptrA + ptrB + ptrC/ptrD
```

核心收益是：

```text
减少 launch overhead
提升小 GEMM 利用率
统一调度不规则 expert GEMM
为低精度和 epilogue fusion 提供基础
```
