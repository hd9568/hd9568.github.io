---
title: 'CUDA Graph 详解：为什么它能优化大模型 Decode 阶段'
description: '结合大模型推理场景讲解 CUDA Graph 的核心思想、capture/replay 机制、适用条件、PyTorch 示例、形状管理和常见坑。'
category: 'Research & Work'
pubDate: '2026-07-13T16:51:00+08:00'
updatedDate: '2026-07-13T16:51:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [Decode 阶段为什么适合 CUDA Graph](#二decode-阶段为什么适合-cuda-graph)
3. [什么是 CUDA Graph](#三什么是-cuda-graph)
4. [普通执行和 Graph 执行的区别](#四普通执行和-graph-执行的区别)
5. [PyTorch 中的 CUDA Graph 示例](#五pytorch-中的-cuda-graph-示例)
6. [大模型推理中的 capture / replay](#六大模型推理中的-capture--replay)
7. [形状管理和 graph pool](#七形状管理和-graph-pool)
8. [为什么需要静态内存地址](#八为什么需要静态内存地址)
9. [常见问题和安全回退](#九常见问题和安全回退)
10. [面试表达](#十面试表达)
11. [总结](#十一总结)

## 一、核心结论

CUDA Graph 解决的是 GPU kernel launch overhead 和 CPU 调度开销问题。

大模型 Decode 阶段有一个典型特点：

```text
每生成 1 个 token，会发射很多小 kernel。
```

比如：

- RMSNorm。
- QKV projection。
- RoPE。
- KV cache update。
- attention。
- MLP / MoE。
- sampling。

单个 kernel 可能很快，但每个 kernel 都要由 CPU 发射。小 batch、低延迟场景下，CPU launch overhead 会变成明显瓶颈。

CUDA Graph 的做法是：

```text
把一段固定的 GPU 执行序列 capture 成 graph；
后续重复 replay graph；
避免每轮都由 CPU 逐个发射 kernel。
```

对大模型推理来说，它尤其适合：

- Decode 阶段。
- batch size / token shape 稳定。
- kernel 序列固定。
- 需要低延迟和高 QPS 的服务。

## 二、Decode 阶段为什么适合 CUDA Graph

Prefill 阶段：

```text
一次处理 prompt 中多个 token。
矩阵大，GEMM 大，GPU 计算占主导。
```

Decode 阶段：

```text
每次只生成一个或少量 token。
kernel 多，单个 kernel 较小。
CPU 发射开销占比更高。
```

一个简化 decode step：

```text
for each layer:
    rmsnorm
    qkv gemm
    rope + kv cache update
    attention
    output projection
    mlp / moe
final norm
lm head
sampling
```

如果模型有 32 层，每层 5-10 个 kernel，一个 token 可能涉及数百次 kernel launch。

CUDA Graph 可以把这段固定序列变成一次 replay：

```text
CPU launch N kernels -> CPU replay 1 graph
```

注意：Graph 不会让 GPU 单个 kernel 本身变快，它减少的是 CPU 调度和 kernel launch 开销。

## 三、什么是 CUDA Graph

CUDA Graph 是 CUDA 提供的一种机制，用于表示一组 GPU 操作及其依赖关系。

图中节点可以是：

- kernel launch。
- memory copy。
- memset。
- event。
- host callback。

边表示依赖关系。

典型流程：

```text
1. Capture 一段 CUDA stream 上的操作。
2. Instantiate 成可执行 graph。
3. 多次 replay。
```

概念上：

```text
cudaStreamBeginCapture(stream)
kernel_a<<<...>>>(...)
kernel_b<<<...>>>(...)
cudaStreamEndCapture(stream, &graph)
cudaGraphInstantiate(&graph_exec, graph)
cudaGraphLaunch(graph_exec, stream)
```

## 四、普通执行和 Graph 执行的区别

普通执行：

```cpp
for (int step = 0; step < steps; ++step) {
    kernel1<<<grid1, block1>>>(...);
    kernel2<<<grid2, block2>>>(...);
    kernel3<<<grid3, block3>>>(...);
}
```

每一步 CPU 都要发射 3 个 kernel。

CUDA Graph：

```cpp
// capture once
kernel1<<<grid1, block1>>>(static_args...);
kernel2<<<grid2, block2>>>(static_args...);
kernel3<<<grid3, block3>>>(static_args...);

// replay many times
for (int step = 0; step < steps; ++step) {
    update_static_input_buffer(...);
    cudaGraphLaunch(graph_exec, stream);
}
```

CPU 发射成本更低，GPU 端执行序列更稳定。

## 五、PyTorch 中的 CUDA Graph 示例

PyTorch 提供 `torch.cuda.CUDAGraph`。

下面是一个最小例子：

```python
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(1024, 4096),
    torch.nn.GELU(),
    torch.nn.Linear(4096, 1024),
).cuda().eval()

static_x = torch.empty((32, 1024), device="cuda")
static_y = None

# warmup：让 CUDA kernel、allocator 等先稳定下来
for _ in range(3):
    y = model(static_x)

torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()

with torch.cuda.graph(g):
    static_y = model(static_x)

def run(x):
    static_x.copy_(x)
    g.replay()
    return static_y

x = torch.randn((32, 1024), device="cuda")
y = run(x)
```

关键点：

- `static_x` 地址固定。
- capture 时使用 `static_x`。
- replay 前把新输入 copy 到 `static_x`。
- `static_y` 的地址也固定。

这就是大模型推理中常见的 static buffer 思路。

## 六、大模型推理中的 capture / replay

推理服务中通常不会只 capture 一个 graph，而是按 shape bucket capture 多个 graph。

例如：

```text
batch size: 1, 2, 4, 8, 16, 32
num tokens: decode 通常为 1
max seq len bucket: 1024, 2048, 4096, 8192
```

对每个 bucket capture 一份 graph：

```python
graphs = {
    (1, 1): graph_bs1,
    (2, 1): graph_bs2,
    (4, 1): graph_bs4,
    ...
}
```

请求来了之后：

```python
bucket = pick_bucket(batch_size, num_tokens)
copy_inputs_to_static_buffers(bucket)
graphs[bucket].replay()
copy_or_view_outputs(bucket)
```

### 为什么不能随便变 shape

CUDA Graph capture 的是固定 kernel 参数和内存地址。

如果 shape 改了，很多东西会变：

- grid/block 配置。
- Tensor stride。
- workspace 大小。
- 临时 buffer 地址。
- kernel 参数。

所以通常需要 shape bucket。

## 七、形状管理和 graph pool

实际服务中 batch 是动态的，不可能为每个 shape 都 capture graph。常见做法是 bucket。

例如实际 batch size 是 13：

```text
选择 bucket 16
前 13 个位置放真实请求
后 3 个位置 padding
```

这会浪费少量计算，但换来 graph replay 的稳定性。

Graph pool 需要管理：

- static input buffer。
- static output buffer。
- KV cache metadata buffer。
- position ids。
- sequence lengths。
- block tables。
- workspace。
- graph exec。

简化结构：

```python
class DecodeGraph:
    def __init__(self, batch_size):
        self.static_input_ids = torch.empty(...)
        self.static_positions = torch.empty(...)
        self.static_block_table = torch.empty(...)
        self.static_output = None
        self.graph = torch.cuda.CUDAGraph()

    def replay(self, runtime_inputs):
        self.static_input_ids.copy_(runtime_inputs.input_ids)
        self.static_positions.copy_(runtime_inputs.positions)
        self.static_block_table.copy_(runtime_inputs.block_table)
        self.graph.replay()
        return self.static_output
```

## 八、为什么需要静态内存地址

CUDA Graph replay 要求 capture 时用到的内存地址在 replay 时仍然有效。

错误做法：

```python
with torch.cuda.graph(g):
    y = model(torch.randn(..., device="cuda"))
```

这里输入 tensor 是临时的，replay 时不能换成另一个新地址。

正确做法：

```python
static_x = torch.empty(..., device="cuda")

with torch.cuda.graph(g):
    static_y = model(static_x)

static_x.copy_(new_x)
g.replay()
```

对大模型推理来说，KV cache 通常本来就是长期存在的，所以适合 graph；但 metadata 也要稳定，例如：

- block table。
- sequence length。
- slot mapping。
- position id。

这些可以更新内容，但最好保持地址稳定。

## 九、常见问题和安全回退

### 1. 动态 shape 导致 graph 不可复用

解决：

- shape bucket。
- padding。
- 超出 bucket 时 fallback eager。

### 2. capture 中出现 CPU 同步

一些操作可能触发 CPU-GPU 同步，例如读取 Tensor 标量：

```python
x.item()
```

capture 区域应避免这类操作。

### 3. allocator 不稳定

capture 前需要 warmup，让缓存分配器状态稳定。

### 4. 随机性和采样

采样阶段如果涉及随机数，必须保证随机状态处理正确。很多框架会把 sampling 放在 graph 外，或者使用可控的 device-side random state。

### 5. 不支持的 op

不是所有 op 都适合 capture。遇到不支持时要 fallback。

工程上通常需要：

```text
can_capture(config) -> bool
capture_graph(bucket)
replay_graph(bucket)
fallback_eager()
```

## 十、面试表达

可以这样解释 CUDA Graph：

```text
CUDA Graph 是把一段固定的 CUDA 操作序列捕获成图，然后重复 replay 的机制。
它主要减少 CPU 逐个 kernel launch 的开销，不会改变单个 kernel 的计算复杂度。
大模型 decode 阶段每生成一个 token 会发射大量小 kernel，因此非常适合 CUDA Graph。
```

如果问实现难点：

```text
核心难点是 shape 和内存地址稳定。
通常需要为不同 batch size 建 bucket，准备 static input/output/workspace buffer。
replay 前只更新 static buffer 内容，不换地址。
如果 shape 超出 bucket 或遇到不支持 op，就安全回退到 eager 路径。
```

如果问为什么 prefill 不一定收益同样明显：

```text
prefill 的 GEMM 和 attention 通常更大，GPU 计算占主导，CPU launch overhead 占比小；
decode 每步 token 少、kernel 多，小 batch 下 CPU launch overhead 占比高，所以收益更明显。
```

## 十一、总结

CUDA Graph 对大模型推理的意义是：

```text
把重复的小 kernel 发射序列变成稳定的 graph replay。
```

它适合：

- Decode。
- 小 batch。
- 低延迟服务。
- shape 可 bucket。
- 内存地址可静态化。

不适合：

- 高度动态 shape。
- capture 中有 CPU 同步。
- op 或 allocator 不稳定。
- 每次执行路径都不同。

工程实现的核心不是 API，而是：

```text
static buffer + shape bucket + graph pool + safe fallback
```
