---
title: '算子中的吞吐与延迟：从 GPU 延迟隐藏到 Little’s Law'
description: '从指令、Warp、Kernel、Batch 和在线服务五个层次讲清吞吐与延迟的区别，结合 CUDA Benchmark、流水线、内存并发、排队模型和 Little’s Law 推导算子优化中的并发需求与性能权衡。'
category: 'Research & Work'
pubDate: '2026-08-14T16:00:00+08:00'
updatedDate: '2026-08-14T16:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [先给出核心结论](#一先给出核心结论)
2. [吞吐和延迟到底是什么](#二吞吐和延迟到底是什么)
3. [为什么高吞吐不等于低延迟](#三为什么高吞吐不等于低延迟)
4. [先区分五种不同层次的延迟](#四先区分五种不同层次的延迟)
5. [指令延迟与指令吞吐](#五指令延迟与指令吞吐)
6. [依赖链为什么暴露指令延迟](#六依赖链为什么暴露指令延迟)
7. [ILP：用独立指令隐藏同一 Warp 的延迟](#七ilp用独立指令隐藏同一-warp-的延迟)
8. [TLP：用其他 Warp 隐藏当前 Warp 的延迟](#八tlp用其他-warp-隐藏当前-warp-的延迟)
9. [Occupancy 与吞吐的真实关系](#九occupancy-与吞吐的真实关系)
10. [内存延迟与内存带宽为什么可以同时很高](#十内存延迟与内存带宽为什么可以同时很高)
11. [Little’s Law 的定义与推导](#十一littles-law-的定义与推导)
12. [Little’s Law 的单位检查和适用条件](#十二littles-law-的单位检查和适用条件)
13. [用 Little’s Law 推导 HBM 需要多少并发请求](#十三用-littles-law-推导-hbm-需要多少并发请求)
14. [用 Little’s Law 理解软件流水线](#十四用-littles-law-理解软件流水线)
15. [Kernel Latency、Grid Size 与 Wave Quantization](#十五kernel-latencygrid-size-与-wave-quantization)
16. [Batch 为什么提高吞吐却可能增加延迟](#十六batch-为什么提高吞吐却可能增加延迟)
17. [一个可运行的 CUDA Batch Scaling 实验](#十七一个可运行的-cuda-batch-scaling-实验)
18. [如何解读实验结果](#十八如何解读实验结果)
19. [流水线的延迟由求和决定，吞吐由最慢阶段决定](#十九流水线的延迟由求和决定吞吐由最慢阶段决定)
20. [排队延迟：利用率接近 100% 为什么危险](#二十排队延迟利用率接近-100-为什么危险)
21. [Little’s Law 在 LLM Decode 中的例子](#二十一littles-law-在-llm-decode-中的例子)
22. [算子优化怎样分别面向低延迟和高吞吐](#二十二算子优化怎样分别面向低延迟和高吞吐)
23. [Fusion、并行和流水化会怎样移动延迟与吞吐](#二十三fusion并行和流水化会怎样移动延迟与吞吐)
24. [怎样用 NSYS 和 NCU 建立证据链](#二十四怎样用-nsys-和-ncu-建立证据链)
25. [正确的 Benchmark 与报告方式](#二十五正确的-benchmark-与报告方式)
26. [常见误解](#二十六常见误解)
27. [一套可复用的分析框架](#二十七一套可复用的分析框架)
28. [参考资料](#二十八参考资料)

## 一、先给出核心结论

吞吐和延迟描述的是两个不同问题：

```text
Latency:
  一个工作从开始到完成需要多久

Throughput:
  单位时间能完成多少工作
```

一个收银台处理一位顾客需要 1 分钟：

```text
单顾客延迟 = 1 min
单收银台吞吐 = 1 customer/min
```

如果开 10 个完全相同的收银台：

```text
单顾客服务延迟仍约为 1 min
系统吞吐变成 10 customers/min
```

并行没有缩短一次收银操作本身，而是让更多顾客同时处于处理中。

GPU 的基本逻辑也是这样：

```text
Global Memory Load 仍可能要等待数百个 Cycle
Tensor Core MMA 仍有固定 Pipeline Latency

GPU 不一定让单条指令更快，
而是让大量独立指令、Warp、CTA 同时在飞，
用并发把等待时间藏在其他工作后面。
```

Little’s Law 将三者联系起来：

```text
L = lambda * W

L:
  系统内平均正在处理或等待的工作数量

lambda:
  平均吞吐，单位时间完成的工作数量

W:
  一个工作在系统内的平均驻留时间
```

在算子和 GPU 中，它可以写成更直观的形式：

```text
In-Flight Work
= Throughput
* Latency
```

它回答一个非常实用的问题：

> 如果一个硬件操作延迟很高，但希望把吞吐打满，至少需要多少独立工作同时在飞？

本文最重要的六个结论：

1. 延迟是单个工作的时间，吞吐是长期完成速率，二者不能互相替代。
2. Latency Hiding 不会降低物理延迟，只是让等待不再出现在关键路径上。
3. Batch、ILP、TLP、流水线和多 Stream 都是在增加 In-Flight Work。
4. 达到吞吐上限后继续加并发，吞吐不再提升，排队延迟却会继续增长。
5. Little’s Law 是稳定系统的平均量恒等式，不是性能优化的因果定律。
6. 算子优化最终应同时报告 Latency、Throughput、并发度、工作量和 SLO，而不是只报一个 TFLOPS 或 GB/s。

## 二、吞吐和延迟到底是什么

### 2.1 延迟

对一个工作 `i`：

```text
latency_i =
    completion_time_i
  - arrival_or_start_time_i
```

算子中常见延迟单位：

- `cycle`：指令或微架构分析。
- `ns`：Cache、互联、小型 Kernel。
- `us`：CUDA Kernel、Collective。
- `ms`：完整算子、模型 Step。

必须先说明起点和终点。

例如“Kernel Latency”可能指：

```text
GPU Event:
  Kernel 在 Device Timeline 上的执行区间

Host Observed:
  Host 发起 Launch 到同步完成

Service:
  请求入队到 Kernel 结果可用
```

这三个数不会完全相同。

### 2.2 吞吐

吞吐定义为：

```text
Throughput =
    Completed Work
  / Elapsed Time
```

算子中常见单位：

- elements/s。
- requests/s。
- tokens/s。
- bytes/s。
- FLOPs/s。
- instructions/cycle。
- tiles/cycle。

同一个算子可以同时报告多种吞吐：

```text
GEMM:
  matrices/s
  TFLOPS

Vector Add:
  elements/s
  effective GB/s

LLM Decode:
  aggregate tokens/s
  per-user tokens/s
  requests/s
```

### 2.3 有效吞吐与硬件吞吐

Vector Add：

```cpp
c[i] = a[i] + b[i];
```

每个元素的算法数据量通常按：

```text
read a:  4 B
read b:  4 B
write c: 4 B
total:  12 B
```

若每秒处理 `N` 个元素：

```text
Effective Bandwidth =
    12 * N / time
```

但硬件 DRAM 实际传输还受：

- Cache Line。
- Sector。
- Write Allocate。
- ECC。
- 非合并访问。

影响。

因此 `effective GB/s` 是算法视角，NCU 的 `DRAM Bytes/s` 是硬件视角，两者不必相等。

## 三、为什么高吞吐不等于低延迟

### 3.1 增加并行执行单元

假设一个任务耗时 10 ms，一张卡可同时执行 8 个任务：

```text
单任务延迟:
  10 ms

系统吞吐:
  8 / 0.01
  = 800 tasks/s
```

如果增加到 16 个并行执行槽：

```text
单任务延迟仍可能是 10 ms
系统吞吐变成 1600 tasks/s
```

### 3.2 增加 Batch

假设 Batch 变化如下：

| Batch | 一次执行时间 | 请求吞吐 | Batch 内请求完成延迟 |
| ---: | ---: | ---: | ---: |
| 1 | 0.25 ms | 4,000 req/s | 0.25 ms |
| 8 | 0.55 ms | 14,545 req/s | 0.55 ms |
| 32 | 1.40 ms | 22,857 req/s | 1.40 ms |

Batch 从 1 增加到 32：

```text
吞吐:
  提升约 5.7 倍

执行延迟:
  增加约 5.6 倍
```

并且在线服务还要等待 Batch 凑齐：

```text
Request Latency =
    Batch Formation Wait
  + Queue Wait
  + Batch Execution Time
```

### 3.3 流水线

一条 10 Stage 的流水线：

```text
单个工作从 Stage 0 到 Stage 9:
  可能需要 10 cycles

流水线稳定后:
  可能每 cycle 完成 1 个工作
```

```text
Latency = 10 cycles
Throughput = 1 item/cycle
```

二者可以同时成立。

## 四、先区分五种不同层次的延迟

“这个算子延迟高”过于笼统。至少要区分五层。

### 4.1 指令延迟

一条指令从输入就绪到结果可被依赖指令使用的时间：

```text
load latency
FMA latency
shuffle latency
MMA latency
```

单位通常是 Cycle。

### 4.2 指令发射间隔

硬件隔多久能接收一条同类型新指令：

```text
Initiation Interval, II
```

如果：

```text
Instruction Latency = 16 cycles
Initiation Interval = 1 cycle
```

表示第一条结果要等 16 Cycle，但流水线可以每 Cycle 接收一条新的独立指令。

### 4.3 Kernel Latency

一个 Kernel 从 Device 开始执行到结束：

```text
T_kernel
```

它包含：

- Prologue。
- Mainloop。
- Epilogue。
- Barrier。
- Tail Wave。

### 4.4 Host Observed Latency

CPU 看到的：

```text
Launch
-> Driver / Queue
-> GPU Execution
-> Synchronization
```

小 Kernel 中，Launch 和同步可能比 Device Execution 更显著。

### 4.5 Service Latency

在线请求从到达到返回：

```text
T_service =
    T_queue
  + T_batch_wait
  + T_host
  + T_device
  + T_communication
  + T_postprocess
```

优化单个 Kernel 只能减少其中一项。

## 五、指令延迟与指令吞吐

### 5.1 一个乘加流水线

假设 FMA：

```text
latency = 4 cycles
initiation interval = 1 cycle
```

独立 FMA：

```text
cycle 0: issue FMA 0
cycle 1: issue FMA 1
cycle 2: issue FMA 2
cycle 3: issue FMA 3
cycle 4: FMA 0 result ready, issue FMA 4
...
```

稳定后：

```text
每 cycle 完成约 1 条
```

虽然每一条都经历 4 Cycle。

### 5.2 Latency 和 Reciprocal Throughput

CPU 微架构分析中常见：

```text
Latency:
  结果依赖距离

Reciprocal Throughput:
  连续独立指令之间的最小平均间隔
```

GPU 虽然调度模型不同，核心区别仍然适用。

### 5.3 算子为什么要关心

算子 Mainloop 如果是：

```text
Load -> Use -> Load -> Use
```

会暴露 Load Latency。

如果重排为：

```text
Load tile 0
Load tile 1
Compute tile 0
Load tile 2
Compute tile 1
...
```

就能把 Load Latency 与其他工作重叠。

## 六、依赖链为什么暴露指令延迟

### 6.1 串行累加

```cpp
float sum = 0.0f;
for (int i = 0; i < n; ++i) {
    sum = fmaf(a[i], b[i], sum);
}
```

每次 FMA 都依赖上一轮 `sum`：

```text
FMA 0 -> FMA 1 -> FMA 2 -> ...
```

关键路径近似：

```text
T >= n * FMA_latency
```

即使 FMA Pipeline 每 Cycle 能接收很多独立指令，这条依赖链也供不上。

### 6.2 Pointer Chasing

```cpp
index = next[index];
```

下一次 Load 的地址依赖上一次 Load 结果：

```text
load next[0]
  -> get index 17
  -> load next[17]
  -> get index 3
  -> ...
```

无法提前知道后续地址，Memory-Level Parallelism 很低。

典型现象：

```text
DRAM Throughput 不高
Long Scoreboard 很高
Eligible Warps 很少
```

这是 Memory-latency-bound，不是 Bandwidth-bound。

### 6.3 Barrier 依赖

```cpp
load_to_shared();
__syncthreads();
compute();
```

即使多数 Warp 已完成 Load，也必须等待最慢 Warp 到达 Barrier。

此时暴露的是：

- Load Tail。
- Warp 工作不均。
- Barrier 延迟。

## 七、ILP：用独立指令隐藏同一 Warp 的延迟

ILP 是 Instruction-Level Parallelism。

### 7.1 多累加器

把一条累加链拆成四条：

```cpp
float sum0 = 0.0f;
float sum1 = 0.0f;
float sum2 = 0.0f;
float sum3 = 0.0f;

for (int i = 0; i < n; i += 4) {
    sum0 = fmaf(a[i + 0], b[i + 0], sum0);
    sum1 = fmaf(a[i + 1], b[i + 1], sum1);
    sum2 = fmaf(a[i + 2], b[i + 2], sum2);
    sum3 = fmaf(a[i + 3], b[i + 3], sum3);
}

float sum = (sum0 + sum1) + (sum2 + sum3);
```

四条链彼此独立。Scheduler/Compiler 可以交错发射，降低固定 FMA Latency 对关键路径的影响。

### 7.2 提前 Load

```cpp
float next_a = a[0];
float next_b = b[0];

for (int i = 0; i < n - 1; ++i) {
    float cur_a = next_a;
    float cur_b = next_b;

    next_a = a[i + 1];
    next_b = b[i + 1];

    sum = fmaf(cur_a, cur_b, sum);
}
```

编译器若能保留独立性，可以在计算当前元素时等待下一元素 Load。

### 7.3 ILP 的代价

- 更多寄存器。
- 更长 Live Range。
- 更大的代码尺寸。
- 可能降低 Occupancy。
- 过度展开可能 Spill。

所以不是无限增加独立累加器。

## 八、TLP：用其他 Warp 隐藏当前 Warp 的延迟

TLP 是 Thread-Level Parallelism。

### 8.1 Warp Scheduler

假设 Warp A 发出 Global Load 后等待：

```text
cycle 0:
  Warp A issues load

cycle 1:
  Warp A stalled
  Scheduler issues Warp B

cycle 2:
  Warp A stalled
  Scheduler issues Warp C
...
```

只要有其他 Eligible Warp，SM 的 Issue Slot 就不必空闲。

### 8.2 多少 Warp 才够

简化模型：

```text
memory latency = 400 cycles
每 Warp 每 20 cycles 能产生一次独立 memory operation
```

要完全覆盖等待，粗略需要：

```text
required_warps ~= 400 / 20 = 20
```

真实硬件还有：

- 多 Warp Scheduler。
- 多 Load/Store Pipe。
- Cache Hit/Miss。
- 每 Warp 多个 Outstanding Load。
- Instruction Mix。

因此这个除法只用于建立直觉。

### 8.3 TLP 和 ILP 可以互补

```text
TLP:
  更多 Warp

ILP:
  每个 Warp 更多独立工作
```

当寄存器很高、只能驻留少量 Warp 时，可用 ILP 补偿。

当单 Warp 依赖链难以展开时，需要更多 Warp。

## 九、Occupancy 与吞吐的真实关系

Occupancy：

```text
Occupancy =
  Active Warps / Maximum Warps
```

它描述延迟隐藏的潜在容量，不是执行单元利用率。

### 9.1 Occupancy 太低

若同时看到：

```text
Active Warps 少
Eligible Warps 少
No Eligible 周期多
Compute / Memory 都未饱和
```

则可能缺少 TLP。

### 9.2 Occupancy 已经够用

若：

```text
Tensor Pipe 已接近峰值
Eligible Warps 充足
Issue Slot 持续使用
```

继续提高 Occupancy 没有意义。

### 9.3 强行提高 Occupancy 的副作用

限制寄存器：

```text
Occupancy 上升
-> Spill 增加
-> Local Memory Load/Store
-> Long Scoreboard 上升
-> Kernel 更慢
```

缩小 Tile：

```text
CTA 数增加
-> Occupancy 上升
-> 数据复用下降
-> HBM/Shared 流量上升
```

最终目标是吞吐和延迟，不是 Occupancy 百分比。

## 十、内存延迟与内存带宽为什么可以同时很高

### 10.1 水管类比

```text
Latency:
  一滴水从管道入口到出口需要多久

Bandwidth:
  稳定后每秒有多少水流出
```

一条很长但很粗的管道：

- 第一滴水很晚才到。
- 稳定后流量很大。

HBM 就有类似特征：

- 单次访问延迟高。
- 大量并发访问下总带宽极高。

### 10.2 单个请求无法打满 HBM

一个 128-Byte Load，即使耗时 500 ns：

```text
单请求吞吐 =
  128 B / 500 ns
  = 0.256 GB/s
```

目标若是数 TB/s，需要成千上万个请求并发在飞。

### 10.3 合并访问优化的是什么

Coalescing 让一个 Warp 的有效数据落在更少 Sector：

```text
同样一条 Warp Load
-> 更少内存事务
-> 更高有效字节比例
```

它不一定显著改变某个 DRAM Miss 的物理延迟，但减少了完成同样算法工作所需的事务数量。

## 十一、Little’s Law 的定义与推导

### 11.1 公式

```text
L = lambda * W
```

也常写成：

```text
N = X * R
```

在本文中统一使用：

```text
L:
  系统内平均工作数量

lambda:
  平均完成吞吐

W:
  平均驻留时间
```

### 11.2 最简单例子

一个稳定服务：

```text
throughput = 100 requests/s
average latency = 20 ms = 0.02 s
```

平均系统内请求数：

```text
L = 100 * 0.02 = 2 requests
```

长期看，平均有 2 个请求处于排队或执行中。

### 11.3 为什么成立

观察时间窗口 `[0, T]`。

令 `N(t)` 为时刻 `t` 系统内请求数。曲线下面积：

```text
Area = integral_0^T N(t) dt
```

从另一个角度，每个请求 `i` 在系统中停留 `W_i`：

```text
Area ~= sum_i W_i
```

系统平均请求数：

```text
L =
  Area / T
```

完成请求数为 `C(T)`：

```text
lambda =
  C(T) / T

W =
  sum_i W_i / C(T)
```

相乘：

```text
lambda * W
= C(T)/T * sum_i(W_i)/C(T)
= sum_i(W_i)/T
= L
```

长时间稳定运行时，窗口边界上的未完成请求影响可以忽略。

### 11.4 它不依赖 Poisson 到达

Little’s Law 不要求：

- Poisson Arrival。
- 指数服务时间。
- FCFS。
- 单 Server。

只要：

- 系统稳定。
- 统计边界一致。
- 工作不会凭空丢失或重复计数。
- 观察时间足够长。

它就成立。

## 十二、Little’s Law 的单位检查和适用条件

### 12.1 单位检查

请求系统：

```text
requests
= requests/s * s
```

内存系统：

```text
bytes in flight
= bytes/s * s
```

指令流水：

```text
instructions in flight
= instructions/cycle * cycles
```

Token Decode：

```text
active tokens or sequences
= tokens/s * s/token-residence
```

单位不一致通常意味着公式套错了。

### 12.2 统计边界必须相同

错误：

```text
L:
  只统计 GPU 上执行的请求

W:
  统计从网关到返回的端到端延迟
```

正确：

```text
若 W 包含 Queue + GPU，
L 也必须统计 Queue + GPU 中的请求。
```

### 12.3 系统必须稳定

若：

```text
arrival_rate > service_capacity
```

队列会持续增长：

```text
L -> infinity
W -> infinity
```

短窗口仍能算瞬时平均，但不存在稳定稳态。

### 12.4 Little’s Law 不是因果关系

公式：

```text
L = lambda * W
```

不表示“提高 L 必然提高 lambda”。

如果硬件已饱和，继续增加并发：

```text
lambda 基本不变
L 增加
W 被迫增加
```

这正是过载时延迟恶化的原因。

## 十三、用 Little’s Law 推导 HBM 需要多少并发请求

### 13.1 In-Flight Bytes

假设目标 HBM 带宽：

```text
B = 3 TB/s
```

一次访问平均往返延迟：

```text
W = 500 ns
```

要维持该吞吐，链路上平均需要：

```text
L = B * W
  = 3e12 B/s * 500e-9 s
  = 1.5e6 B
  = 1.5 MB
```

也就是说大约需要 1.5 MB 内存请求同时在飞。

### 13.2 换算成事务数

若每个有效事务 128 B：

```text
num_transactions =
  1.5 MB / 128 B
  ~= 11,719
```

这个例子使用简化的平均延迟和有效带宽，具体数字随 GPU、Cache、访问类型和统计口径变化。

但结论稳定：

> 高带宽需要大量 Memory-Level Parallelism。

### 13.3 并发来自哪里

- 更多 Warp。
- 每 Warp 多个 Outstanding Load。
- 更多 CTA。
- 多 Stage `cp.async` / TMA。
- 更大的 Batch。
- 多个 Stream 或请求。

### 13.4 为什么依赖链打不满带宽

Pointer Chasing 每次只有一个请求：

```text
in_flight_bytes ~= 128 B
```

远低于打满 HBM 所需的 1.5 MB。

所以它会：

```text
Latency 高
Bandwidth 低
```

## 十四、用 Little’s Law 理解软件流水线

### 14.1 Pipeline Depth

假设一个硬件流水线：

```text
Latency = 16 cycles
Throughput = 1 instruction/cycle
```

稳态所需 In-Flight 指令：

```text
L = 1 inst/cycle * 16 cycles
  = 16 instructions
```

若只有一条依赖链：

```text
L ~= 1
```

Pipeline 大部分槽位为空。

### 14.2 GEMM Mainloop

典型多 Stage：

```text
Stage 0:
  load tile k+2

Stage 1:
  load tile k+1

Stage 2:
  MMA tile k
```

Stage 数的目标是提供足够 In-Flight Tile，覆盖：

```text
GMEM -> SMEM
SMEM -> Register/TMEM
MMA
```

延迟。

### 14.3 Stage 不是越多越好

增加 Stage：

- 增加 Shared Memory。
- 增加 Barrier State。
- 增加 Register Live Range。
- 可能降低驻留 CTA。

当现有 Stage 已覆盖延迟，继续增加只会浪费资源。

## 十五、Kernel Latency、Grid Size 与 Wave Quantization

### 15.1 一个 Wave

假设：

```text
GPU 有 100 个 SM
每个 SM 同时驻留 2 个目标 CTA
```

一个 Wave 可容纳：

```text
100 * 2 = 200 CTAs
```

Grid 有 450 CTA：

```text
Wave 0: 200
Wave 1: 200
Wave 2: 50
```

最后一个 Wave 只有 25% 槽位被使用。

### 15.2 Latency 和 Throughput

若每 CTA 时间近似相同：

```text
Kernel Latency ~= 3 * T_CTA
```

最后一个 Partial Wave 期间：

- 大多数 SM 空闲。
- 全芯片吞吐下降。

### 15.3 减小 Tile 的权衡

更小 Tile：

```text
CTA 数增加
Wave 更完整
并行度提高
```

但也可能：

```text
数据复用下降
边界和 Epilogue 增加
内存事务增加
```

最终比较总 Kernel Latency，而不是只看 Wave 数。

## 十六、Batch 为什么提高吞吐却可能增加延迟

### 16.1 固定成本被摊薄

一次 Kernel 有固定成本：

```text
Launch
Metadata
Prologue
Epilogue
Synchronization
```

Batch 增大后：

```text
fixed_cost / item
```

下降。

### 16.2 GEMM Shape 变好

LLM Decode 的 GEMM：

```text
[B, K] x [K, N]
```

`B=1` 时接近 GEMV，数据复用低。

`B` 增大：

- Weight 被多个 Token 复用。
- Tensor Core Tile 更完整。
- 算术强度提高。

### 16.3 Step Time 也会增加

设 Batch Service Time 为：

```text
S(B)
```

吞吐：

```text
X(B) = B / S(B)
```

单个 Batch 内请求的 Device Completion Latency 至少为：

```text
S(B)
```

在线请求还要等待 Batch：

```text
W(B) =
  queue_wait
  + batch_wait
  + S(B)
```

### 16.4 最优 Batch 取决于目标

```text
Latency-first:
  小 Batch

Throughput-first:
  增大 Batch 到吞吐接近平台

SLO-first:
  在 P99 约束下选择最大 Batch
```

## 十七、一个可运行的 CUDA Batch Scaling 实验

下面把一个 Batch 看作多个独立 Vector SAXPY 请求。每个请求处理固定数量元素：

```text
y = alpha * x + y
```

Batch 中所有请求放进同一个 Kernel：

```text
total_elements =
  batch_size * elements_per_request
```

代码同时报告：

- 重复 Batched Kernel 的平均 Stream Service Interval。
- Batched Request Throughput。
- Element Throughput。
- Effective Bandwidth。

```cpp
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::fprintf(                                                  \
                stderr,                                                    \
                "%s:%d CUDA error: %s\n",                                  \
                __FILE__,                                                   \
                __LINE__,                                                   \
                cudaGetErrorString(error));                                \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

__global__ void batched_saxpy(
    const float* __restrict__ x,
    float* __restrict__ y,
    float alpha,
    int elements_per_request,
    int batch_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = elements_per_request * batch_size;

    if (index < total_elements) {
        y[index] = fmaf(alpha, x[index], y[index]);
    }
}

float benchmark_batch(
    const float* x,
    float* y,
    int elements_per_request,
    int batch_size,
    int iterations,
    cudaStream_t stream) {
    int total_elements = elements_per_request * batch_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    // 每个 Shape 单独 Warmup，避免首次 JIT、升频和 Cache 状态干扰。
    for (int i = 0; i < 20; ++i) {
        batched_saxpy<<<blocks, threads, 0, stream>>>(
            x,
            y,
            2.0f,
            elements_per_request,
            batch_size);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        batched_saxpy<<<blocks, threads, 0, stream>>>(
            x,
            y,
            2.0f,
            elements_per_request,
            batch_size);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start));

    return total_ms / iterations;
}

int main() {
    // 最大 Batch 的总工作集约为 256 MiB（x + y），通常大于 GPU L2。
    constexpr int kElementsPerRequest = 1 << 18;
    constexpr int kMaxBatchSize = 128;
    constexpr int kIterations = 500;
    constexpr float kBytesPerElement = 12.0f;

    int max_elements = kElementsPerRequest * kMaxBatchSize;
    size_t bytes = static_cast<size_t>(max_elements) * sizeof(float);

    float* x = nullptr;
    float* y = nullptr;
    CHECK_CUDA(cudaMalloc(&x, bytes));
    CHECK_CUDA(cudaMalloc(&y, bytes));
    CHECK_CUDA(cudaMemset(x, 0, bytes));
    CHECK_CUDA(cudaMemset(y, 0, bytes));

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));

    std::vector<int> batch_sizes = {
        1, 2, 4, 8, 16, 32, 64, 128
    };

    std::printf(
        "%8s %14s %16s %18s %14s\n",
        "batch",
        "service_us",
        "requests_per_s",
        "elements_per_s",
        "effective_GB_s");

    for (int batch_size : batch_sizes) {
        float latency_ms = benchmark_batch(
            x,
            y,
            kElementsPerRequest,
            batch_size,
            kIterations,
            stream);

        double latency_s = latency_ms / 1000.0;
        double total_elements =
            static_cast<double>(kElementsPerRequest) * batch_size;

        double request_throughput =
            static_cast<double>(batch_size) / latency_s;
        double element_throughput =
            total_elements / latency_s;
        double effective_gb_s =
            element_throughput * kBytesPerElement / 1e9;

        std::printf(
            "%8d %14.3f %16.1f %18.3e %14.2f\n",
            batch_size,
            latency_ms * 1000.0,
            request_throughput,
            element_throughput,
            effective_gb_s);
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(y));
    CHECK_CUDA(cudaFree(x));
    return 0;
}
```

编译：

```bash
nvcc -O3 -lineinfo -arch=sm_XX \
  operator_throughput_latency.cu \
  -o operator_throughput_latency
```

运行：

```bash
./operator_throughput_latency
```

## 十八、如何解读实验结果

### 18.1 预期曲线

小 Batch：

```text
Kernel Latency 变化不大
GPU 没有填满
Batch 翻倍时请求吞吐显著增加
```

中等 Batch：

```text
更多 SM 和内存请求被使用
Effective Bandwidth 快速上升
```

大 Batch：

```text
HBM 接近平台
Element Throughput 增长放缓
Kernel Latency 随总元素数增长
```

概念图：

```text
Throughput
^
|                    __________
|                ___/
|            ___/
|        ___/
|_______/______________________> Batch

Latency
^
|                         /
|                      __/
|                  ___/
|_________________/___________> Batch
```

### 18.2 为什么 Request Throughput 与 Element Throughput 都要看

每个“请求”固定为 `elements_per_request` 时：

```text
request throughput
```

有意义。

若不同请求 Shape 不同，只报 requests/s 会误导。此时应同时报告：

- elements/s。
- bytes/s。
- FLOPs/s。

### 18.3 Event 时间不等于端到端请求延迟

代码用 CUDA Event 测稳定 Device Timeline 上的平均 Stream Service Interval。大 Kernel 下它接近平均 Kernel Duration；极小 Kernel 若 GPU 消耗工作的速度快于 Host 提交速度，Event 区间也会包含 Device Timeline 上的提交空隙。

它没有完整包含：

- 请求排队。
- Batch Formation。
- CPU Tokenization。
- RPC。
- Host 同步策略。

若要测在线延迟，必须在请求入口和结果出口打点。

### 18.4 Benchmark 的一个陷阱

代码连续提交 1000 个 Kernel 后统一等待。这样适合测稳态吞吐和平均 Device Service Time。

如果每次都：

```cpp
launch();
cudaDeviceSynchronize();
```

测到的是：

```text
Launch + Sync + Device
```

它更接近同步请求延迟，但会破坏并发和流水。

两种测法回答不同问题。

## 十九、流水线的延迟由求和决定，吞吐由最慢阶段决定

假设一个算子 Pipeline：

```text
Stage A: Load       2 us
Stage B: Compute    5 us
Stage C: Store      3 us
```

### 19.1 无重叠

单工作延迟：

```text
W = 2 + 5 + 3
  = 10 us
```

吞吐：

```text
X = 1 / 10 us
  = 100K items/s
```

### 19.2 完全流水

单工作仍要经过所有 Stage：

```text
Latency ~= 10 us
```

稳态完成间隔由最慢 Stage 决定：

```text
Initiation Interval =
  max(2, 5, 3)
  = 5 us

Throughput =
  1 / 5 us
  = 200K items/s
```

### 19.3 Little’s Law 检查

```text
L = X * W
  = 200K items/s * 10 us
  = 2 items
```

稳态平均约有 2 个工作散布在 Pipeline 中。

### 19.4 平衡 Stage

若把 Compute 拆成两个 2.5 us Stage：

```text
2, 2.5, 2.5, 3 us
```

新吞吐上限：

```text
1 / 3 us
~= 333K items/s
```

总单工作计算时间仍约 10 us，但 Pipeline 更平衡。

这就是：

```text
Latency 看路径总和
Throughput 看瓶颈 Stage
```

## 二十、排队延迟：利用率接近 100% 为什么危险

### 20.1 服务能力

服务率：

```text
mu = 1000 requests/s
```

平均纯服务时间：

```text
S = 1 / mu = 1 ms
```

到达率：

```text
lambda
```

利用率：

```text
rho = lambda / mu
```

### 20.2 M/M/1 直觉模型

在 M/M/1 假设下：

```text
W = 1 / (mu - lambda)
```

| 到达率 | 利用率 | 平均系统延迟 |
| ---: | ---: | ---: |
| 500 req/s | 50% | 2 ms |
| 800 req/s | 80% | 5 ms |
| 900 req/s | 90% | 10 ms |
| 990 req/s | 99% | 100 ms |

纯服务时间始终只有 1 ms，但排队让平均延迟在 99% 利用率时变成 100 ms。

真实 GPU 服务不是 M/M/1：

- Batch Service。
- 多 Server。
- 请求长度不均。
- Priority。
- Preemption。

但“饱和附近延迟陡增”的结论非常重要。

### 20.3 Little’s Law 再检查

`lambda=990 req/s`，`W=0.1 s`：

```text
L = 990 * 0.1
  = 99 requests
```

平均 99 个请求在系统中，其中大多数在等待。

### 20.4 为什么线上不能追求永久 100% 利用率

需要为以下波动保留余量：

- Arrival Burst。
- 长请求。
- Expert/Shape 不均。
- Cache Miss。
- 网络抖动。
- GC / JIT / 故障。

线上优化目标通常是：

```text
在 P99 SLO 下最大化 Goodput
```

而不是让 GPU 指标永远 100%。

## 二十一、Little’s Law 在 LLM Decode 中的例子

LLM Decode 每个 Step 为每个活跃序列生成一个 Token。

### 21.1 Batch 128

```text
active sequences = 128
step latency = 20 ms = 0.02 s
```

聚合 Token Throughput：

```text
X =
  128 tokens / 0.02 s
  = 6400 tokens/s
```

Little’s Law：

```text
L = X * W
  = 6400 tokens/s * 0.02 s
  = 128 tokens in flight
```

单序列生成速度：

```text
1 / 0.02
= 50 tokens/s/user
```

### 21.2 Batch 256

假设：

```text
step latency = 30 ms
```

聚合吞吐：

```text
256 / 0.03
~= 8533 tokens/s
```

单序列速度：

```text
1 / 0.03
~= 33.3 tokens/s/user
```

结果：

```text
Aggregate Throughput:
  6400 -> 8533, 提升约 33%

TPOT:
  20 ms -> 30 ms, 恶化 50%
```

这就是大 Batch 提高总吞吐、降低单用户 Token Rate 的典型场景。

### 21.3 并发容量估算

到达率：

```text
lambda = 20 requests/s
```

平均每请求 Decode 驻留：

```text
W_decode = 8 s
```

平均活跃 Decode 请求：

```text
L_decode =
  20 * 8
  = 160 requests
```

KV Cache、Sequence Slot 和 Scheduler 至少要承载这个数量，再加峰值与冗余。

### 21.4 Token Throughput 不能直接换成 QPS

请求输出长度为 `O_i`：

```text
request throughput
```

取决于输出长度分布。

若平均输出 200 Token，Decode Capacity 为 10K tokens/s：

```text
理论平均 QPS上限
~= 10000 / 200
= 50 requests/s
```

但 P99 长输出会占用状态更久，实际容量还要看长度分布。

## 二十二、算子优化怎样分别面向低延迟和高吞吐

### 22.1 延迟优先

目标：

```text
最短单任务关键路径
```

常见手段：

- 减少 Kernel Launch。
- CUDA Graph。
- Kernel Fusion。
- Persistent Kernel。
- 减少 Barrier。
- 缩短依赖链。
- 预取关键数据。
- 使用更宽 TP 降低单请求计算时间。
- 控制 Batch 和排队。

代价可能是：

- 每请求占用更多 GPU。
- 总吞吐下降。
- 通信增多。

### 22.2 吞吐优先

目标：

```text
单位时间完成更多总工作
```

常见手段：

- 增大 Batch。
- Continuous Batching。
- 提高 Occupancy。
- 增加 Pipeline Stage。
- 使用更大 Tile 提高复用。
- 多 Stream。
- Data Parallel Replica。
- 量化减少字节数。

代价可能是：

- 单请求等待更长。
- Step Latency 增加。
- P99 恶化。

### 22.3 Goodput 优先

定义：

```text
Goodput =
  满足 SLO 的有效工作
  / 时间
```

例如：

```text
TTFT <= 100 ms
TPOT <= 30 ms
```

只完成但超时的请求不计入 Goodput。

这比单纯 tokens/s 更符合在线系统。

## 二十三、Fusion、并行和流水化会怎样移动延迟与吞吐

### 23.1 Kernel Fusion

融合前：

```text
Kernel A
-> write HBM
-> Kernel B
```

融合后：

```text
Kernel AB
```

可能同时改善：

- 端到端延迟。
- 吞吐。
- HBM Bytes。
- Launch 数。

但单个融合 Kernel 的 Duration 可能比 A 或 B 单独更长。正确比较对象是：

```text
T_AB
vs
T_A + gap + T_B
```

### 23.2 增加并行度

Split-K：

```text
一个输出 Tile
-> 多 CTA 并行处理 K Slice
-> Reduction
```

可降低低并行 Shape 的 Latency，但增加：

- Workspace。
- Reduction。
- HBM 流量。

大 Shape 已有足够并行时，Split-K 可能降低吞吐。

### 23.3 软件流水

Double Buffer：

```text
Load tile k+1
与
Compute tile k
重叠
```

不减少 Load Latency，但减少其暴露时间。

### 23.4 多 Stream

两个独立小 Kernel：

```text
Stream 0: Kernel A
Stream 1: Kernel B
```

若单个 Kernel 不能占满 GPU，并发可提高系统吞吐。

若 A 已占满 Tensor Core 或 HBM，B 只会竞争资源，甚至增加延迟。

## 二十四、怎样用 NSYS 和 NCU 建立证据链

### 24.1 NSYS 看时间轴

回答：

- Kernel 之间有多少 Gap。
- 多 Stream 是否重叠。
- Batch Formation 是否在 CPU。
- Memcpy 是否阻塞。
- 通信和计算是否重叠。
- 单请求 Latency 由哪些阶段组成。

### 24.2 NCU 看 Kernel 内部

吞吐侧：

- Compute Throughput。
- DRAM/L2/L1 Throughput。
- Tensor Pipe。
- Instructions Per Cycle。
- Eligible/Issued Warps。

延迟侧：

- Long Scoreboard。
- Short Scoreboard。
- Wait / Execution Dependency。
- Barrier。
- No Eligible。
- Achieved Occupancy。

### 24.3 Bandwidth-bound 证据

```text
DRAM Throughput 高
DRAM Bytes 接近算法预期
Eligible Warp 足够
减少 Bytes 后 Duration 下降
```

### 24.4 Latency-bound 证据

```text
DRAM Throughput 不高
Long Scoreboard 高
Eligible Warp 少
依赖链或并发请求不足
增加 ILP/TLP 后 Duration 下降
```

### 24.5 Queue-bound 证据

NCU 看不到服务排队。需要服务指标：

```text
Queue Time 高
GPU Service Time 稳定
Arrival Burst 时 P99 上升
利用率接近饱和
```

此时继续优化 Kernel 可能不是第一优先级，应扩容、限流或调整 Batch。

## 二十五、正确的 Benchmark 与报告方式

### 25.1 明确工作量

至少报告：

```text
Shape
Dtype
Layout
Batch
并发度
输入输出规模
算法 FLOPs / Bytes
```

### 25.2 区分 Cold 与 Warm

Cold 可能包含：

- Context 初始化。
- PTX JIT。
- Kernel Autotune。
- Page Fault。
- Cache Cold Miss。

Warm 才代表稳态。

两者都重要，但不能混在一个数字里。

### 25.3 同时报告延迟与吞吐

推荐：

| 指标 | 目的 |
| --- | --- |
| Device Kernel Latency | 算子本体 |
| Host Observed Latency | Launch/Sync 影响 |
| Throughput | 稳态处理能力 |
| P50/P95/P99 | 长尾 |
| Batch / Concurrency | 解释吞吐来源 |
| FLOPs/Bytes | 解释工作量 |
| Power | 性能功耗 |

### 25.4 不要用 `time / batch` 冒充单请求延迟

Batch 32 用时 1.4 ms：

```text
1.4 / 32
= 0.04375 ms
```

这是平均 Service Cost Per Item，不是 Request Completion Latency。

Batch 内请求通常要等整个 Kernel 完成：

```text
completion latency ~= 1.4 ms
```

### 25.5 统计分位数

平均值会隐藏：

- Tail Wave。
- Cache Miss。
- Expert Imbalance。
- OS 抢占。
- 网络抖动。
- 长 Shape。

在线系统必须看 P95/P99。

## 二十六、常见误解

### 26.1 “吞吐是延迟的倒数”

只在单工作、无并发、无流水的极简系统中近似成立。

稳定平均、统计边界一致时：

```text
Throughput = In-Flight / Latency
```

### 26.2 “吞吐提高，延迟一定降低”

错误。增大 Batch 常使总吞吐提高，但 Batch Latency 和 Queue Wait 增加。

### 26.3 “延迟隐藏就是降低内存延迟”

错误。Latency Hiding 让其他工作覆盖等待，物理访问延迟可能完全没变。

### 26.4 “带宽没满，所以内存不是瓶颈”

错误。可能是 Memory-latency-bound：请求不够、依赖链长、访问不规则。

### 26.5 “Occupancy 越高，吞吐越高”

错误。达到足够隐藏延迟的 Occupancy 后，继续提高可能破坏 ILP 和数据复用。

### 26.6 “Batch 时间除以 Batch Size 就是单请求延迟”

错误。那是平均摊销成本。Batch 内请求通常共同完成。

### 26.7 “利用率 100% 最经济”

错误。在线系统在饱和附近 Queue Latency 和 P99 会快速增长。

### 26.8 “Little’s Law 能预测加并发后的吞吐”

错误。它描述平均量关系，不给出服务能力曲线。需要先知道硬件在该并发下的 `W` 或 `lambda`。

### 26.9 “流水线减少了单工作总计算量”

错误。流水线主要重叠不同工作或不同阶段，提高稳态吞吐，不一定减少单工作 Latency。

### 26.10 “一个高 TFLOPS Kernel 就是低延迟 Kernel”

错误。高 TFLOPS 可能来自大 Batch 和长执行时间。低延迟还要看 Shape、Launch、排队和关键路径。

## 二十七、一套可复用的分析框架

### 第一步：定义工作

```text
一个 Work Item 是什么？
一个 Element、Tile、Request 还是 Token？
```

### 第二步：定义边界

```text
Latency 从哪里开始，到哪里结束？
In-Flight Work 统计哪个范围？
```

### 第三步：计算工作量

```text
FLOPs
HBM Bytes
L2/Shared Bytes
Instruction Count
Communication Bytes
```

### 第四步：测延迟和吞吐曲线

不要只测一个 Batch：

```text
Batch = 1, 2, 4, 8, ...
Concurrency = 1, 2, 4, 8, ...
```

绘制：

```text
Latency vs Batch
Throughput vs Batch
Goodput vs Concurrency
```

### 第五步：用 Little’s Law 做一致性检查

```text
Measured In-Flight
?=
Measured Throughput * Measured Latency
```

若差异很大，检查：

- 单位。
- 统计边界。
- Warmup/Drain。
- 丢弃/重试。
- Throughput 定义。

### 第六步：识别区间

```text
Under-utilized:
  加并发，吞吐快速提高

Saturation:
  吞吐接近平台

Overloaded:
  吞吐基本不变，延迟快速上升
```

### 第七步：按目标优化

```text
低延迟:
  缩短关键路径、减少 Queue、减少 Launch

高吞吐:
  增加独立工作、提高复用、填满 Pipeline

在线服务:
  在 SLO 下最大化 Goodput
```

## 二十八、参考资料

1. [John D. C. Little, A Proof for the Queuing Formula: L = λW](https://doi.org/10.1287/opre.9.3.383)
2. [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
3. [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
4. [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
5. [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
6. [David A. Patterson, John L. Hennessy, Computer Organization and Design](https://www.elsevier.com/books/computer-organization-and-design-risc-v-edition/patterson/978-0-12-820331-6)
7. [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
8. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)

吞吐和延迟的统一视角是：

```text
一个工作需要多久:
  Latency

单位时间完成多少:
  Throughput

为了在高延迟下维持目标吞吐，需要多少工作同时在飞:
  In-Flight = Throughput * Latency
```

GPU 的高性能来自大量并发、流水线和数据复用。它不会自动消灭 HBM、指令和通信延迟，而是用 ILP、TLP、Batch、异步拷贝和多 Stage Pipeline 让这些等待不再暴露在关键路径上。

当并发不足时，吞吐低而延迟暴露；当并发适中时，吞吐提高且等待被隐藏；当系统已经饱和后继续加并发，吞吐不再增加，Queue 和 P99 延迟开始快速恶化。

Little’s Law 不是一个调参公式，而是一条守恒关系。它最有价值的用法是把“带宽、延迟、并发”放进同一个单位系统，检查性能解释是否自洽，并估算为了填满目标硬件到底需要多少独立工作。
