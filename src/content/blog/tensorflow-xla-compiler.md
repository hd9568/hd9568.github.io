---
title: 'TensorFlow XLA 详解：从计算图、HLO 到 GPU Kernel 融合'
description: '系统讲解 TensorFlow XLA 的 JIT 编译流程、HLO IR、算子融合、内存规划、tf.function 与 jit_compile、动态 Shape、编译缓存、分布式训练及性能分析。'
category: 'Research & Work'
pubDate: '2026-07-20T12:32:00+08:00'
updatedDate: '2026-07-20T12:32:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [XLA 解决什么问题](#一xla-解决什么问题)
2. [TensorFlow 执行模式与 XLA 的位置](#二tensorflow-执行模式与-xla-的位置)
3. [从 TensorFlow Graph 到可执行程序](#三从-tensorflow-graph-到可执行程序)
4. [HLO 是什么](#四hlo-是什么)
5. [XLA 的核心优化](#五xla-的核心优化)
6. [算子融合为什么有效](#六算子融合为什么有效)
7. [内存规划与 Buffer Assignment](#七内存规划与-buffer-assignment)
8. [tf.function 与 jit_compile](#八tffunction-与-jit_compile)
9. [完整训练步骤示例](#九完整训练步骤示例)
10. [编译边界和 Cluster](#十编译边界和-cluster)
11. [动态 Shape、Retracing 与 Recompilation](#十一动态-shaperetracing-与-recompilation)
12. [XLA 不一定更快的原因](#十二xla-不一定更快的原因)
13. [分布式训练中的 XLA](#十三分布式训练中的-xla)
14. [数值、随机性与副作用](#十四数值随机性与副作用)
15. [如何检查 HLO 和定位性能问题](#十五如何检查-hlo-和定位性能问题)
16. [工程优化方法](#十六工程优化方法)
17. [与其他编译技术的关系](#十七与其他编译技术的关系)
18. [总结](#十八总结)

## 一、XLA 解决什么问题

XLA 全称 **Accelerated Linear Algebra**，是面向机器学习计算的编译器。

普通深度学习框架执行一段 TensorFlow 程序时，常按算子逐个调度：

```text
MatMul
-> BiasAdd
-> GELU
-> Dropout
-> Add
-> LayerNorm
```

每个算子可能对应一个或多个 kernel。算子之间会产生：

- Kernel launch 开销。
- 中间 Tensor 写入显存。
- 下一算子再次读取显存。
- 临时 Tensor 分配。
- 框架调度开销。

XLA 的目标是先观察一段完整计算，再进行跨算子优化：

```text
TensorFlow operations
-> 编译器 IR
-> 图级优化
-> Fusion
-> 设备代码生成
```

它可能带来：

- 减少 kernel 数量。
- 减少中间 Tensor 的 HBM 读写。
- 常量折叠。
- 代数化简。
- 更好的 layout。
- 更紧凑的临时内存规划。
- 针对固定 shape 生成专用代码。
- 将多个 collective 或计算进行更合理调度。

XLA 不是“打开后所有模型都自动变快”。它用编译时间、shape 约束和可编译算子范围换取运行时优化。

## 二、TensorFlow 执行模式与 XLA 的位置

需要区分三个概念。

### Eager Execution

TensorFlow 2 默认 eager：

```python
x = tf.random.normal([1024, 1024])
y = tf.nn.relu(x)
```

每行 Python 调用立即执行，调试方便，但 Python 和框架调度开销较明显。

### tf.function

`tf.function` 把 Python 函数追踪成 TensorFlow Graph：

```python
@tf.function
def f(x):
    return tf.nn.relu(x * 2.0 + 1.0)
```

它减少 Python 调度，并允许 TensorFlow Graph optimizer 做优化。

但：

```text
tf.function != XLA
```

`tf.function` 只表示图执行，不一定交给 XLA 编译。

### tf.function(jit_compile=True)

```python
@tf.function(jit_compile=True)
def f(x):
    return tf.nn.relu(x * 2.0 + 1.0)
```

这要求 TensorFlow 尝试把整个函数编译为 XLA executable。

关系可以表示为：

```text
Python function
  |
  | tf.function tracing
  v
TensorFlow Graph
  |
  | jit_compile=True
  v
XLA compilation
  |
  v
CPU/GPU/TPU executable
```

## 三、从 TensorFlow Graph 到可执行程序

一个简化 XLA 编译链路是：

```text
TensorFlow Graph
-> XLA cluster
-> HLO module
-> HLO optimization
-> target-specific lowering
-> executable
```

### 1. Graph Tracing

`tf.function` 根据输入签名追踪 Python 函数，生成 TensorFlow Graph。

Python 控制流如果依赖 Tensor，通常会通过 AutoGraph 转成图控制流：

```python
if tensor_condition:
    ...
```

变成类似 `If`/`Conditional` 的 graph op。

### 2. XLA Cluster

TensorFlow 找出可以一起交给 XLA 的 op 集合，称为 cluster。

`jit_compile=True` 倾向于要求整个函数可编译；自动聚类模式则可能只选择部分子图。

### 3. Lower to HLO

TensorFlow op 被转换为 XLA HLO operations。

### 4. HLO Optimization

在 HLO 层执行：

- Constant folding。
- Algebraic simplification。
- Common subexpression elimination。
- Fusion。
- Layout assignment。
- Collective optimization。
- Dead code elimination。

### 5. Backend Code Generation

根据设备生成代码：

- CPU backend 生成 CPU machine code。
- GPU backend 生成 GPU kernels 并调用高性能库。
- TPU backend 生成 TPU executable。

### 6. Runtime Execution

编译结果被缓存。同一输入签名再次调用时直接执行 executable，避免重复编译。

## 四、HLO 是什么

HLO 全称 High Level Optimizer，是 XLA 的核心 IR。

一段 TensorFlow：

```python
def f(x, w, b):
    y = tf.matmul(x, w)
    y = y + b
    return tf.nn.relu(y)
```

概念上可能转成：

```text
%dot = dot(%x, %w)
%bias = broadcast(%b)
%add = add(%dot, %bias)
%zero = constant(0)
%relu = maximum(%add, %zero)
ROOT %relu
```

HLO 的特点：

- 强类型。
- Tensor shape 明确。
- 操作语义偏线性代数。
- 与 TensorFlow Python API 解耦。
- 适合跨算子分析。

### HLO 层级

现代 OpenXLA 生态中还会遇到：

- StableHLO：稳定、可序列化的跨框架 IR。
- MHLO：历史上常见的 MLIR HLO dialect。
- Optimized HLO：经过 XLA passes 后的 HLO。
- LLVM IR / GPU-specific IR：更低层表示。

对 TensorFlow XLA 性能分析，最重要的是理解：

```text
TensorFlow Graph 决定语义；
HLO 暴露编译器看到的计算；
后端决定最终 kernel 和设备执行。
```

## 五、XLA 的核心优化

### Constant Folding

编译期已知的计算直接求值：

```python
scale = tf.sqrt(tf.constant(64.0))
```

可在编译时变成常量 `8.0`。

### Algebraic Simplification

例如：

```text
x + 0 -> x
x * 1 -> x
reshape(reshape(x)) -> reshape(x)
transpose(transpose(x)) -> x
```

具体变换必须保证类型、shape 和浮点语义允许。

### Dead Code Elimination

没有影响函数输出和副作用的操作会被删除。

### Common Subexpression Elimination

相同输入上的等价表达式可以复用，避免重复计算。

### Layout Assignment

Tensor 的逻辑 shape 不等于物理 layout。XLA 会根据 backend 和消费者选择内存布局，尽量减少 transpose 和非连续访问。

### Operator Fusion

把多个 HLO op 合并成单个或更少的 kernel，减少中间数据写回。

### Library Call Selection

大型 MatMul、Convolution 通常不适合重新生成普通逐元素 kernel。XLA 会选择：

- cuBLAS/cuBLASLt。
- cuDNN。
- 设备专用实现。

然后尝试把适合的 epilogue 或周边操作融合。

## 六、算子融合为什么有效

考虑：

```python
y = tf.nn.gelu(tf.matmul(x, w) + b)
```

朴素执行可能是：

```text
Kernel 1: MatMul -> write temp_1 to HBM
Kernel 2: BiasAdd -> read temp_1, write temp_2
Kernel 3+: GELU -> read/write intermediates
```

若 BiasAdd 和 GELU 被融合：

```text
MatMul/library call
-> fused epilogue or fused elementwise kernel
-> write final y
```

收益主要来自：

- 减少 kernel launch。
- 中间值保留在 register/cache。
- 减少 HBM traffic。
- 减少临时 buffer。

### 一个具体例子

设 Tensor 有：

```text
16 million FP32 elements = 64 MB
```

如果独立执行 3 个 elementwise op，每个 op 大致读 64 MB、写 64 MB：

```text
3 * (64 + 64) MB = 384 MB traffic
```

若融合为一个 kernel，理想情况只读一次、写一次：

```text
128 MB traffic
```

实际还受 cache、广播和指令影响，但 memory-bound 算子通常能明显获益。

### 哪些操作容易融合

- Elementwise chain。
- Broadcast + elementwise。
- Reduction 前后的 elementwise。
- 简单 reshape/transpose。
- MatMul/Conv 的部分 epilogue。

### 哪些操作不容易融合

- 有副作用的 op。
- 不支持的 custom op。
- 复杂动态 shape。
- 需要独立高性能库调用的巨大计算。
- Host callback。
- 跨设备边界。

## 七、内存规划与 Buffer Assignment

普通框架逐 op 执行时，会为中间 Tensor 申请和释放内存。编译器看到完整生命周期后，可以做静态规划。

假设：

```text
t1 = op1(x)
t2 = op2(t1)
t3 = op3(x)
y  = op4(t2, t3)
```

XLA 能分析：

- `t1` 最后一次使用在哪里。
- `t2` 和其他 Tensor 生命周期是否重叠。
- 哪些 buffer 可以复用。
- 哪些操作允许 input/output alias。

### Buffer Reuse

如果两个 Tensor 生命周期不重叠，可以共享同一块内存：

```text
buffer A:
  first stores t1
  later reused for t3
```

### Donation / Aliasing

若调用者允许输入 buffer 被覆盖，输出可以复用输入内存，减少峰值和 copy。是否可用取决于 runtime、API 语义和数据后续是否继续使用。

### 峰值并不一定下降

Fusion 有时会：

- 增加 register pressure。
- 增加 shared memory。
- 延长某些 buffer 生命周期。
- 为并行执行保留多个临时结果。

因此必须看真实 peak memory，而不能假设 XLA 一定省显存。

## 八、tf.function 与 jit_compile

### 只使用 tf.function

```python
@tf.function
def step(x, w):
    return tf.nn.relu(tf.matmul(x, w))
```

优点：

- 消除大部分 Python per-op 调度。
- TensorFlow Graph optimization。
- 支持 SavedModel 等图工作流。

### 强制 XLA 编译

```python
@tf.function(jit_compile=True)
def step(x, w):
    return tf.nn.relu(tf.matmul(x, w))
```

如果函数中包含 XLA 不支持的 op，调用可能直接报错。

### 指定 input_signature

```python
@tf.function(
    input_signature=[
        tf.TensorSpec([None, 1024], tf.float32),
        tf.TensorSpec([1024, 4096], tf.float32),
    ],
    jit_compile=True,
)
def step(x, w):
    return tf.matmul(x, w)
```

`input_signature` 能减少因 Python 类型和 shape 变化导致的 retracing，但动态维度是否能共用同一 XLA executable，仍取决于 op 和 backend 对动态 shape 的支持。

## 九、完整训练步骤示例

下面把 forward、loss、backward 和 optimizer update 放进编译函数。

```python
import tensorflow as tf


model = tf.keras.Sequential([
    tf.keras.layers.Dense(2048, activation="gelu"),
    tf.keras.layers.Dense(1),
])
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)


@tf.function(jit_compile=True)
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        per_example_loss = tf.square(pred - y)
        loss = tf.reduce_mean(per_example_loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

训练：

```python
for x, y in dataset:
    loss = train_step(x, y)
```

### 为什么编译完整 train step

如果只编译 model forward，以下步骤仍在 cluster 外：

- Loss。
- Backward graph。
- Gradient transform。
- Optimizer update。

编译更大的稳定区域，通常提供更多融合和调度机会。

但 cluster 越大：

- 编译时间越长。
- 不支持 op 的概率越高。
- Shape 变化影响更大。
- 调试更困难。

## 十、编译边界和 Cluster

XLA 只能优化 cluster 内的计算。

假设：

```text
Op A -> Op B -> Unsupported Op -> Op C -> Op D
```

可能形成：

```text
Cluster 1: A + B
TensorFlow runtime: Unsupported Op
Cluster 2: C + D
```

Cluster 边界可能产生：

- 中间 Tensor materialization。
- 设备同步。
- Kernel launch。
- Layout conversion。

### 编译区域不是越大越好

大 cluster：

- 优化空间更大。
- Kernel 更少。
- 编译更慢。
- 峰值内存可能更高。

小 cluster：

- 编译快。
- 容易缓存。
- 边界和中间访存更多。

合理做法是围绕稳定的高计算密度区域划分，例如：

- Transformer block。
- MLP tower。
- 完整 train step 中可支持的核心子图。

数据读取、字符串处理和复杂 host 逻辑通常不适合放入 XLA cluster。

## 十一、动态 Shape、Retracing 与 Recompilation

动态输入是 XLA 工程中最常见的问题之一。

### Retracing

Retracing 发生在 `tf.function` 层：TensorFlow 根据新的 Python 参数、dtype、rank 或 shape 签名重新生成 ConcreteFunction。

例如：

```python
f(tf.ones([8, 128]))
f(tf.ones([16, 128]))
```

如果签名未放宽，可能产生两个 trace。

### Recompilation

Recompilation 发生在 XLA 层：即使复用同一个 TensorFlow function，新的 shape specialization 也可能需要新 executable。

二者不是同一件事：

```text
Retracing: TensorFlow Graph 重新追踪
Recompilation: XLA executable 重新编译
```

### 为什么 Shape 影响大

固定 shape 让编译器可以：

- 固定 tile。
- 展开循环。
- 精确规划内存。
- 选择特定 library algorithm。

但线上变长序列可能出现大量 shape：

```text
[B, 127]
[B, 128]
[B, 129]
...
```

如果每个 shape 都编译，会产生：

- Compile latency。
- Executable cache 增长。
- 首次请求抖动。

### Bucket 与 Padding

常见方案：

```text
真实长度 117 -> bucket 128
真实长度 173 -> bucket 192
真实长度 241 -> bucket 256
```

优点：

- Shape 数有限。
- 编译缓存稳定。
- 更容易批处理。

代价：

- Padding 增加无效计算。

需要在 padding FLOPs 和 recompilation latency 之间平衡。

### reduce_retracing

```python
@tf.function(reduce_retracing=True, jit_compile=True)
def f(x):
    ...
```

它有助于 TensorFlow 生成更通用签名，但不保证所有动态 shape 都只编译一次。

## 十二、XLA 不一定更快的原因

### 1. 编译成本大于运行收益

短任务只运行几步：

```text
compile 10 s
run saves 1 ms/step
only 100 steps
```

总收益无法覆盖编译成本。

### 2. 计算已经由高性能库主导

大 GEMM 已由 cuBLAS 高效执行。若周围没有可融合操作，XLA 改善有限。

### 3. 输入太动态

频繁 recompilation 会抵消运行时收益。

### 4. Cluster 太碎

不支持 op 或副作用切断 cluster，导致跨边界 materialization。

### 5. Fusion 过大

过度融合可能导致：

- Register spilling。
- Occupancy 下降。
- 指令 cache 压力。
- 单 kernel 并行度不足。

### 6. Host 或 Input Pipeline 才是瓶颈

如果 GPU 在等待：

- 数据读取。
- 特征处理。
- Python callback。
- 网络服务。

编译 GPU 计算不会解决主瓶颈。

### 7. 首次调用测量错误

第一次调用包含 tracing 和 compilation。正确 benchmark 要区分：

```text
compile latency
steady-state execution latency
```

### 8. 异步执行未同步

GPU op 通常异步。只测 Python 函数返回时间可能不准确，应在 benchmark 边界获取结果或显式同步。

## 十三、分布式训练中的 XLA

XLA 可以编译单个 replica 的计算，也能表示部分 collective：

- AllReduce。
- AllGather。
- ReduceScatter。
- CollectivePermute。

### Replica 内计算

每个 data-parallel replica 执行相同 HLO program。梯度产生后执行 collective。

### Collective 与计算调度

编译器可看到计算和 collective 的依赖，有机会：

- 合并小 collective。
- 调整 collective 顺序。
- 与独立计算 overlap。
- 避免不必要 layout conversion。

实际能力取决于 TensorFlow distribute strategy、XLA backend 和通信 runtime。

### 固定 Global Shape

分布式训练通常更依赖稳定 shape，因为：

- 每个 rank shape 必须满足 collective 约束。
- 尾 batch 形状变化可能触发新编译。
- 不同 rank 控制流不一致可能造成 collective hang。

常用措施：

- `drop_remainder=True`。
- 固定 per-replica batch。
- Sequence length bucket。
- 确保所有 rank 执行相同 collective 顺序。

### 梯度累积

可以把多个 micro-step 放进一个编译函数：

```python
@tf.function(jit_compile=True)
def accumulated_step(iterator):
    accum = [tf.zeros_like(v) for v in model.trainable_variables]

    for _ in tf.range(accum_steps):
        x, y = next(iterator)
        with tf.GradientTape() as tape:
            loss = compute_loss(x, y) / tf.cast(accum_steps, tf.float32)
        grads = tape.gradient(loss, model.trainable_variables)
        accum = [a + g for a, g in zip(accum, grads)]

    optimizer.apply_gradients(zip(accum, model.trainable_variables))
    return loss
```

这样可以减少函数调用和图边界，但 iterator op、动态循环、内存峰值及分布式 strategy 是否支持，需要实际验证。

## 十四、数值、随机性与副作用

### 浮点重排

浮点加法不满足结合律：

```text
(a + b) + c != a + (b + c)
```

XLA 的 fusion、reduction tree 和代数优化可能改变运算顺序，结果通常在容差内，但不保证逐位一致。

### Fast Math

后端可能使用近似数学函数或更激进的浮点变换。训练应关注：

- Loss 曲线。
- 最终指标。
- NaN/Inf。
- Gradient norm。

而不是只对单步输出要求 bitwise equality。

### 随机数

Dropout、随机采样和初始化需要正确的随机种子语义。编译器不能随意删除或重复有状态随机操作。

需要区分：

- Stateful RNG。
- Stateless RNG。

Stateless RNG 显式接收 seed，通常更容易获得可控、可复现的图语义：

```python
x = tf.random.stateless_normal([1024], seed=[123, step])
```

### Resource Variable

变量读写有副作用和顺序约束。Optimizer update、metric update 等会限制某些重排，但 XLA 可以在保持依赖语义的前提下编译。

### Python Side Effect

`tf.function` 中的普通 Python `print`、list append 等通常只在 tracing 时执行，不能当作每步运行逻辑。

应使用：

```python
tf.print(...)
```

但调试 op 可能阻碍优化或不被目标 backend 支持，生产路径中应谨慎使用。

## 十五、如何检查 HLO 和定位性能问题

### 获取 Compiler IR

TensorFlow 可通过 concrete function 接口查看 compiler IR。示例：

```python
@tf.function(jit_compile=True)
def f(x, w, b):
    return tf.nn.gelu(tf.matmul(x, w) + b)


x = tf.TensorSpec([128, 1024], tf.float32)
w = tf.TensorSpec([1024, 4096], tf.float32)
b = tf.TensorSpec([4096], tf.float32)

ir_getter = f.experimental_get_compiler_ir(x, w, b)
print(ir_getter(stage="hlo"))
print(ir_getter(stage="optimized_hlo"))
```

不同 TensorFlow 版本支持的 stage 名称可能不同。

### HLO 中看什么

1. 是否出现预期的 `dot`/`convolution`。
2. Elementwise chain 是否形成 fusion。
3. 是否有多余 `copy`、`transpose`、`reshape`。
4. 同一个表达式是否重复计算。
5. Reduction 是否拆成多个 kernel。
6. Cluster 输入输出是否过多。
7. Collective 是否出现在关键路径。

### Runtime Profiler

HLO 说明编译器计划，Profiler 说明实际执行。需要同时观察：

- Kernel 数量。
- Kernel duration。
- Launch gap。
- HBM throughput。
- Tensor Core utilization。
- Register usage。
- Compile time。
- Recompilation 次数。
- Host 到 device 的空洞。

### 正确 Benchmark

```python
import time


# 首次调用负责 tracing/compilation，不计入稳态。
result = compiled_fn(x)
_ = result.numpy()

start = time.perf_counter()
for _ in range(100):
    result = compiled_fn(x)
_ = result.numpy()  # 等待设备完成
elapsed = time.perf_counter() - start

print(elapsed / 100)
```

应分别报告：

- First-call latency。
- Steady-state latency。
- Throughput。
- Peak memory。

## 十六、工程优化方法

### 1. 从稳定热点开始

优先编译：

- 重复次数多。
- Shape 稳定。
- Elementwise/Reduction 较多。
- Kernel launch 密集。
- 中间 Tensor 较大。

### 2. 控制 Shape 数量

- 设置 input signature。
- 使用 bucket。
- 对序列做有限 padding。
- 固定训练 batch。
- 监控 compile cache。

### 3. 清理 Cluster Break

找到不支持的 op，判断能否：

- 替换为等价 TensorFlow op。
- 把该 op 移到编译区域外。
- 为 custom op 提供 XLA lowering。
- 缩小编译函数边界。

### 4. 避免 Python 数据依赖

将运行时逻辑表达为 Tensor 和 TensorFlow control flow，避免 Python 根据 Tensor 值做分支。

### 5. 观察 Fusion 而不是只看开关

打开 XLA 后要验证：

- Kernel 是否减少。
- 大中间 Tensor 是否消失。
- 是否出现 register spilling。
- 端到端是否真的提高。

### 6. 编译缓存预热

在线推理可对常见 shape bucket 预热，避免真实请求承担首次编译延迟。

### 7. 分开统计编译和执行

训练任务可能能接受一次较长编译；短任务和在线服务则对首次延迟更敏感。优化目标不同，评估方式也不同。

## 十七、与其他编译技术的关系

### Grappler

Grappler 是 TensorFlow Graph 优化器，可执行：

- Constant folding。
- Arithmetic optimization。
- Layout optimization。
- Remapping。

它工作在 TensorFlow Graph 层；XLA 将 cluster lower 到 HLO 并生成设备 executable。两者可以串联。

### MLIR

MLIR 提供多层 IR 基础设施。TensorFlow/OpenXLA 可通过不同 dialect 逐步降低计算，StableHLO 是其中重要的可移植表示。

### TensorRT

TensorRT 主要面向 NVIDIA GPU inference，提供：

- Layer fusion。
- Precision selection。
- Kernel autotuning。
- Engine optimization。

XLA 更通用，可覆盖训练和多种 backend；TensorRT 在特定 NVIDIA 推理场景通常提供更专门的优化。

### torch.compile

PyTorch 2 的典型编译链路：

```text
TorchDynamo
-> AOTAutograd
-> Inductor
```

它与 TensorFlow XLA 的共同目标是：

```text
保留框架表达能力；
捕获较大计算区域；
通过图优化和代码生成减少运行时开销。
```

具体 IR、动态 shape、算子生态和 runtime 不同。

### JAX

JAX 也广泛使用 XLA/OpenXLA。`jax.jit` 与 TensorFlow `jit_compile=True` 都会把函数交给 XLA，但 tracing 语义、变换系统和 API 不同。

## 十八、总结

TensorFlow XLA 的核心流程是：

```text
TensorFlow Graph
-> XLA cluster
-> HLO
-> HLO optimization
-> target-specific code generation
-> cached executable
```

它的主要收益来自：

1. 跨算子 fusion。
2. 减少中间 Tensor 的 HBM 读写。
3. 减少 kernel launch。
4. 常量折叠和代数化简。
5. Layout 与 buffer 生命周期优化。
6. 针对 shape 和设备生成专用程序。

正确使用时需要区分：

```text
tf.function tracing
XLA compilation
steady-state execution
```

并重点控制：

- 编译边界。
- Shape 数量。
- Unsupported op。
- 首次编译延迟。
- Fusion 后的 register/occupancy。
- 分布式 collective 顺序。
- 浮点数值变化。

XLA 最适合稳定、重复、高计算密度且存在融合空间的子图。是否有效不能只看 `jit_compile=True` 是否成功，而要通过 HLO、Profiler、编译次数和端到端吞吐共同验证。
