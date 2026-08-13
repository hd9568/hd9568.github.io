---
title: 'Nsight Compute 指标全解：从 NCU 报告定位 CUDA Kernel 优化空间'
description: '面向 CUDA 算子开发，系统讲解 Nsight Compute 的采集方法、SOL、Roofline、Memory、Compute、Scheduler、Warp Stall、Occupancy、Source 与 SASS 指标，并用简单 Kernel 建立从现象到优化动作的证据链。'
category: 'CUDA'
pubDate: '2026-08-13T16:00:00+08:00'
updatedDate: '2026-08-13T16:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [NCU 到底解决什么问题](#一ncu-到底解决什么问题)
2. [先建立正确的性能分析模型](#二先建立正确的性能分析模型)
3. [准备可信的 Profile 环境](#三准备可信的-profile-环境)
4. [NCU 命令行与 Kernel 过滤](#四ncu-命令行与-kernel-过滤)
5. [理解 Set、Section、Metric 与 Replay](#五理解-setsectionmetric-与-replay)
6. [读懂 Metric 名称和归一化方式](#六读懂-metric-名称和归一化方式)
7. [推荐的 NCU 诊断顺序](#七推荐的-ncu-诊断顺序)
8. [Speed Of Light：先看哪个硬件子系统最忙](#八speed-of-light先看哪个硬件子系统最忙)
9. [Roofline：计算上限、带宽上限与算术强度](#九roofline计算上限带宽上限与算术强度)
10. [Launch Statistics：Grid、Block 与 Wave Quantization](#十launch-statisticsgridblock-与-wave-quantization)
11. [Occupancy：理论驻留能力不等于执行效率](#十一occupancy理论驻留能力不等于执行效率)
12. [Scheduler Statistics：每个周期为什么发不出指令](#十二scheduler-statistics每个周期为什么发不出指令)
13. [Warp Stall：每种停顿究竟说明什么](#十三warp-stall每种停顿究竟说明什么)
14. [Memory Workload：从请求追到 DRAM](#十四memory-workload从请求追到-dram)
15. [Global Memory：用 Sector 判断合并访问](#十五global-memory用-sector-判断合并访问)
16. [Cache：Hit Rate 高为什么仍然可能很慢](#十六cachehit-rate-高为什么仍然可能很慢)
17. [Shared Memory：Bank Conflict 与 Wavefront](#十七shared-memorybank-conflict-与-wavefront)
18. [Local Memory：识别 Register Spilling](#十八local-memory识别-register-spilling)
19. [Compute Workload：找到真正饱和的执行管线](#十九compute-workload找到真正饱和的执行管线)
20. [Instruction、Source 与 SASS：把指标定位到代码](#二十instructionsource-与-sass把指标定位到代码)
21. [Branch 与线程利用率：识别 Warp Divergence](#二十一branch-与线程利用率识别-warp-divergence)
22. [四类典型报告如何推导优化动作](#二十二四类典型报告如何推导优化动作)
23. [最容易误读的 NCU 指标](#二十三最容易误读的-ncu-指标)
24. [一套可复用的算子优化流程](#二十四一套可复用的算子优化流程)
25. [参考资料](#二十五参考资料)

## 一、NCU 到底解决什么问题

Nsight Compute，简称 NCU，是 NVIDIA 提供的 Kernel 级性能分析器。它通过 GPU 硬件计数器、软件 Patch 和静态资源信息回答下面这些问题：

```text
这个 Kernel 为什么慢？
它受计算吞吐、内存吞吐、依赖延迟、同步还是并行度限制？
最忙的是 FP32、Tensor Core、L1/TEX、L2、DRAM 还是 Shared Memory？
Warp 为什么没有发出下一条指令？
问题来自哪一行 CUDA、哪条 SASS 指令？
修改之后，减少的是工作量，还是只改变了某个百分比？
```

NCU 不擅长回答整个程序的端到端调度问题。例如：

- CPU 是否及时提交 Kernel。
- 多个 Stream 是否重叠。
- H2D、D2H、NCCL 和 Kernel 如何交错。
- 大量小 Kernel 的 Launch Gap 是否主导延迟。
- 某个 Kernel 在完整请求中的时间占比。

这些问题应该先用 Nsight Systems 分析。通常的工具链是：

```text
Nsight Systems:
  找到端到端热点和时间线问题

Nsight Compute:
  解释一个热点 Kernel 内部为什么慢

CUDA Event / Benchmark:
  在无 Profiler 干扰下确认最终延迟和吞吐
```

只有正在开发一个独立 Kernel，且已经知道它就是分析对象时，才适合直接从 NCU 开始。

## 二、先建立正确的性能分析模型

NCU 指标很多，但 CUDA Kernel 的性能限制可以先压缩成三类。

### 2.1 Throughput-bound：某种硬件吞吐已经接近上限

典型情况：

```text
DRAM 带宽接近可持续上限
FP32 FMA Pipe 接近可持续上限
Tensor Core Pipe 接近可持续上限
Shared Memory Pipe 或某条片上互连接近上限
```

这类 Kernel 不是“没把 GPU 用起来”，而是已经把某个资源用满。优化必须减少经过该资源的工作量，或把工作转移到其他资源。

例如 DRAM 已饱和时，单纯增加 Block 数量通常没有用。有效方向是：

- 减少 HBM 字节数。
- 提高数据复用。
- 融合 Kernel，消除中间结果。
- 使用更窄的数据类型。
- 改变 Layout，减少无效 Sector。

### 2.2 Latency-bound：吞吐没有打满，但 Warp 经常等依赖

典型情况：

```text
Compute Throughput 不高
DRAM Throughput 也不高
Eligible Warps 很少
Long Scoreboard 或 Execution Dependency 很高
```

这不表示硬件没有瓶颈，而是单个 Warp 的依赖链太长，且并发 Warp 或指令级并行不足，无法把延迟隐藏起来。

有效方向可能是：

- 增加独立 Load 或独立累加器。
- 软件流水和 Double Buffer。
- 提前发起 Prefetch。
- 增加可驻留 Warp，但不能引入严重 Spill。
- 调整 Tile，缩短串行依赖链。

### 2.3 Parallelism-bound：总工作量不足以覆盖整张 GPU

典型情况：

```text
Grid 只有少量 CTA
Waves Per SM 小于 1
只有部分 SM 收到工作
Compute 和 Memory 的全芯片吞吐都低
单个 CTA 内部可能并不低效
```

有效方向是增加可调度工作，而不是微调单条指令：

- 减小输出 Tile，增加 CTA 数。
- Split-K、Split-KV 或序列维切分。
- Persistent Kernel 中增加工作队列。
- 合并 Batch。
- 对短任务考虑 Kernel Fusion，但要重新评估 Grid 是否更小。

这三类问题可以同时出现。例如一个小 Grid Kernel 既有 Long Scoreboard，也只有半个 Wave。必须先估计每类问题的性能上限，优先处理收益最大的限制。

## 三、准备可信的 Profile 环境

### 3.1 先保证结果正确

错误的越界访问、Race 或未初始化数据可能在普通运行时没有立即暴露，但在 Replay 时表现完全不同。建议在 Profile 前先运行：

```bash
compute-sanitizer --tool memcheck ./ncu_demo
compute-sanitizer --tool racecheck ./ncu_demo
```

`racecheck` 主要检查 Shared Memory Hazard，不会替代所有并发正确性验证。

### 3.2 使用 Release 优化并保留行号

推荐编译方式：

```bash
nvcc -O3 -lineinfo -Xptxas=-v -arch=sm_90 \
  ncu_demo.cu -o ncu_demo
```

参数含义：

| 参数 | 作用 |
| --- | --- |
| `-O3` | 使用优化后的 Device Code |
| `-lineinfo` | 保留 CUDA 源码与 SASS 的行号映射 |
| `-Xptxas=-v` | 输出寄存器、Shared Memory、Spill 等编译信息 |
| `-arch=sm_XX` | 为实际目标架构生成代码 |

不要用 `-G` 生成 Device Debug Code 后分析性能。`-G` 会改变优化、寄存器分配和指令序列，得到的不是发布版本 Kernel。

### 3.3 Profile 之前先 Warmup

首次运行可能包含：

- CUDA Context 初始化。
- Module Lazy Loading。
- PTX JIT。
- 库内部 Heuristic 或 Autotune。
- 内存页建立。
- GPU 从 Idle P-State 升频。

Benchmark 应 Warmup 后再计时。NCU 则可以通过 Kernel 过滤和 `--launch-skip` 跳过 Warmup。

### 3.4 固定输入、Shape 和运行环境

至少记录：

```text
GPU 型号与 Compute Capability
Driver、CUDA、NCU 版本
Kernel Shape、Dtype、Layout
Grid、Block、Dynamic Shared Memory
编译选项
时钟与功耗状态
MIG 配置
NCU 的 Cache Control、Clock Control 和 Replay Mode
```

动态负载、原子竞争和工作队列会让不同运行的指令路径不同。若两次报告不是同一工作负载，差异视图没有意义。

### 3.5 不要把 NCU 中的耗时当最终性能

NCU 可能：

- 序列化 Kernel Launch。
- 多次 Replay 同一个 Kernel。
- 保存并恢复 Kernel 使用的内存。
- Flush Cache。
- Patch 指令或开启采样。
- 控制 GPU Clock。

因此最终性能必须在无 NCU 的条件下，用 CUDA Event 或框架 Benchmark 重新测量。NCU 的 Duration 适合在相同采集配置下辅助对比，不应直接替代生产延迟。

## 四、NCU 命令行与 Kernel 过滤

### 4.1 先查看当前版本支持什么

不同 NCU 版本和 GPU 架构可用的 Section、Metric 不完全相同。不要从旧文章复制一长串 Metric 名称后直接使用。

```bash
ncu --version
ncu --list-sets
ncu --list-sections
ncu --query-metrics
```

查询某个 Base Metric 支持哪些后缀：

```bash
ncu --query-metrics-mode suffix \
  --metrics sm__throughput
```

### 4.2 从轻量采集开始

不指定 `--set`、`--section` 或 `--metrics` 时，NCU 默认采集 `basic` Set：

```bash
ncu -o report_basic ./ncu_demo
```

更常用的第一轮：

```bash
ncu --set basic \
  --kernel-name regex:copy_ \
  --launch-skip 10 \
  --launch-count 1 \
  -o report_basic \
  ./ncu_demo
```

含义：

- `--kernel-name regex:copy_`：只匹配名字包含 `copy_` 的 Kernel。
- `--launch-skip 10`：在匹配到的 Launch 中跳过前 10 次。
- `--launch-count 1`：只采集随后 1 次。
- `-o report_basic`：保存为 `.ncu-rep` 报告。

Kernel 是 C++ 模板时，完整 Demangled Name 可能很长。先运行轻量采集确认名称，再写更精确的 Regex。

### 4.3 按问题选择 Section

比一上来 `--set full` 更高效的做法是只采集当前需要的 Section：

```bash
ncu \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis \
  --kernel-name regex:copy_ \
  --launch-skip 10 \
  --launch-count 1 \
  -o report_detail \
  ./ncu_demo
```

需要完整探索时再使用：

```bash
ncu --set full \
  --kernel-name regex:copy_ \
  --launch-count 1 \
  -o report_full \
  ./ncu_demo
```

`full` 可能需要很多 Replay Pass。Kernel 越长、写入内存越多，采集成本越高。

### 4.4 只采集少量原始 Metric

自动化回归更适合指定有限 Metric：

```bash
ncu \
  --metrics "gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed" \
  --kernel-name regex:copy_ \
  --launch-count 1 \
  ./ncu_demo
```

这些名称只是常见示例。应使用目标机器上的 `--query-metrics` 验证。

### 4.5 多进程、NVTX 与程序内控制

Python 框架或 Worker 子进程需要：

```bash
ncu --target-processes all [其他参数] python workload.py
```

复杂应用可以用 NVTX 标记目标范围，再启用 NVTX Filter：

```bash
ncu --nvtx --nvtx-include "profile/" [其他参数] ./app
```

也可以在程序中用 `cudaProfilerStart()` 和 `cudaProfilerStop()` 包围目标区间，并使用：

```bash
ncu --profile-from-start off [其他参数] ./app
```

目标是尽量少采集无关 Launch。减少 Profile 范围通常比增加机器资源更有效。

## 五、理解 Set、Section、Metric 与 Replay

### 5.1 Set 是预定义的采集规模

Set 是一组 Section 的集合：

```text
basic:
  采集较少，适合第一轮分类

full:
  覆盖更多分析维度，采集更慢
```

Set 不是“精度等级”。`basic` 中已有的 Metric 不会因为切换到 `full` 就自动更准确，`full` 只是收集更多证据。

### 5.2 Section 是围绕一个问题组织的指标和规则

常用 Section：

| Section | 主要回答的问题 |
| --- | --- |
| `SpeedOfLight` | Compute 和 Memory 哪个大类更接近上限 |
| `SpeedOfLight_RooflineChart` | FLOPs、字节数和算术强度位于哪个 Roof 下 |
| `LaunchStats` | Grid、Block、寄存器和 Shared 配置是什么 |
| `Occupancy` | 哪种资源限制 CTA/Warp 驻留 |
| `SchedulerStats` | Scheduler 每周期有多少 Active、Eligible、Issued Warp |
| `WarpStateStats` | Warp 的周期主要消耗在哪些状态 |
| `MemoryWorkloadAnalysis` | L1/TEX、L2、DRAM、Shared 的请求和吞吐如何 |
| `ComputeWorkloadAnalysis` | 哪条执行管线最忙，指令吞吐如何 |
| `InstructionStats` | SASS 指令混合与执行数量是什么 |
| `SourceCounters` | 哪行源码对应高 Stall、低线程利用率或异常访问 |

Section 内除了 Metric，还可以包含 NVIDIA 提供的 Rule。Rule 会根据指标给出提示，但它是启发式规则，不是性能问题的证明。

### 5.3 为什么需要 Replay

GPU 同时可采集的硬件计数器有限，而一份完整报告需要很多 Metric。NCU 会将 Metric 分成多个 Pass，并多次执行目标 Kernel：

```text
Pass 1: 采集一组 SM Counter
Pass 2: 采集一组 L1/TEX Counter
Pass 3: 采集一组 L2/DRAM Counter
...
```

Kernel Replay 通常会保存 Kernel 写入的内存，并在后续 Pass 前恢复，以尽量让每次执行看到相同状态。

这带来几个重要后果：

1. 原子竞争和动态调度可能无法完全复现。
2. Kernel 写入的内存范围越大，保存与恢复成本越高。
3. Cache Flush 策略会影响 Hit Rate。
4. Profile 期间的 Kernel Duration 不等于无 Profiler Duration。
5. 不同 Pass 的 Counter 来自不同执行，不应分析一次运行内的瞬时相关性。

当 Kernel Replay 不适合时，可以了解 Application Replay 或 Range Replay。它们通过重新执行整个应用或范围收集不同 Pass，但要求应用具有足够的可重现性。

### 5.4 Cache Control 与真实业务缓存

为了让 Replay Pass 可比较，NCU 可以在 Pass 之间控制 Cache。默认策略可能与生产中的 Warm Cache 不同。

例如某个权重常驻 L2 的 Decode Kernel，如果每个 Pass 都 Flush Cache，报告会更接近 Cold Cache；如果关闭 Cache Flush，后续 Pass 又可能比第一次更热。

正确做法不是固定选择某一个模式，而是：

```text
1. 明确生产环境是 Cold、Warm 还是多租户抖动。
2. 所有候选实现使用相同 NCU Cache 配置。
3. 用无 Profiler 的端到端 Benchmark 验证。
4. 必要时分别报告 Cold Cache 和 Warm Cache。
```

## 六、读懂 Metric 名称和归一化方式

PerfWorks Metric 常见形式：

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
```

可以拆成：

```text
sm__throughput
  sm              -> 计数单元或硬件单元
  throughput      -> Base Metric

.avg
  跨实例聚合方式

.pct_of_peak_sustained_elapsed
  相对可持续峰值的百分比，以 Kernel Elapsed Cycles 归一化
```

### 6.1 常见硬件单元前缀

| 前缀 | 大致对应 |
| --- | --- |
| `gpu__` | 整个 GPU 或 Kernel 级信息 |
| `sm__` | Streaming Multiprocessor |
| `smsp__` | SM Subpartition，通常与 Warp Scheduler 侧指标相关 |
| `l1tex__` | L1/TEX 单元，处理 Global、Local、Texture、Shared 等请求 |
| `lts__` | L2 Cache Slice |
| `dram__` | Device Memory / HBM / GDDR |

不能仅凭前缀推导全部含义，最终以 `--query-metrics` 描述和官方文档为准。

### 6.2 常见 Rollup 后缀

| 后缀 | 含义 |
| --- | --- |
| `.sum` | 所有实例或周期累计 |
| `.avg` | 实例平均 |
| `.min` / `.max` | 实例最小值或最大值 |
| `.per_second` | 换算为每秒速率 |
| `.per_cycle_active` | 以单元 Active Cycle 归一化 |
| `.per_cycle_elapsed` | 以整个 Elapsed Cycle 归一化 |
| `.pct_of_peak_sustained_active` | Active 期间相对可持续峰值 |
| `.pct_of_peak_sustained_elapsed` | 整个 Kernel 期间相对可持续峰值 |
| `.per_warp_active` | 以 Active Warp 归一化 |

`active` 与 `elapsed` 不能混着比较。某个单元只在一小段时间内工作时：

```text
pct_of_peak_sustained_active 可能很高
pct_of_peak_sustained_elapsed 仍然很低
```

前者表示它一旦工作就很忙，后者表示它对整个 Kernel 时间的覆盖率有限。

### 6.3 Peak Sustained 不等于规格表峰值

`pct_of_peak_sustained_*` 的分母是 NCU 对该硬件单元定义的可持续峰值，不一定等于产品规格表中通过频率和 Core 数推导的 Marketing Peak。

同样，`Compute Throughput = 80%` 也不表示“80% 的 CUDA Core 都在工作”。它通常是相关计算子单元归一化吞吐中的代表值或较高值。必须展开 Compute Workload 才知道是 FP32、Tensor、ALU、SFU 还是其他 Pipe 在贡献。

## 七、推荐的 NCU 诊断顺序

不要从几百个 Metric 中随机挑高值。推荐按以下顺序收敛。

### 第一步：确认 Kernel 值得优化

用 NSYS 或端到端统计确认：

```text
Kernel 占总延迟多少？
调用频率多高？
优化 100% 后，端到端上限是多少？
```

Amdahl 定律：

```text
端到端加速比 =
1 / ((1 - p) + p / s)

p: 目标 Kernel 的时间占比
s: Kernel 自身加速比
```

### 第二步：看 Duration、Grid 与 Waves

先判断：

- Kernel 是否短到测量噪声很大。
- Grid 是否覆盖全部 SM。
- 是否存在明显 Tail Wave。
- 每个 Block 使用多少寄存器和 Shared Memory。

### 第三步：看 Speed Of Light 和 Roofline

粗分：

```text
Compute Throughput 高 -> 深挖 Compute Pipe
Memory Throughput 高  -> 深挖内存层级
两者都低             -> 深挖并行度、Scheduler、Stall、同步
```

### 第四步：看 Scheduler 和 Warp Stall

确认是：

- 没有足够 Active Warp。
- 有 Active Warp，但没有 Eligible Warp。
- Eligible Warp 足够，某条 Pipe 已满。

### 第五步：进入具体子系统

```text
Memory:
  Request -> Sector -> Wavefront -> L1 -> L2 -> DRAM

Compute:
  SASS Mix -> Pipe Utilization -> Dependency -> Thread Predication

Synchronization:
  Barrier Stall -> CTA 内工作量差异 -> Source Line
```

### 第六步：定位 Source/SASS 并提出单变量修改

每次修改都应该有预测：

```text
改动:
  将跨步 Global Load 改成连续 Load

预期:
  Global Load Sectors/Request 下降
  DRAM Bytes 下降
  Long Scoreboard 下降或更容易隐藏
  Duration 下降
```

如果指标按预期变化但 Duration 不变，说明修复的不是当前最大瓶颈，或者新的瓶颈已经接管。

## 八、Speed Of Light：先看哪个硬件子系统最忙

Speed Of Light，简称 SOL，提供 Kernel 的高层吞吐概览。

常见顶部指标：

```text
Compute Throughput
Memory Throughput
Duration
Elapsed Cycles
SM Frequency
DRAM Frequency
```

### 8.1 四种基本组合

| Compute | Memory | 初步判断 |
| --- | --- | --- |
| 高 | 低 | 某条计算 Pipe 可能是吞吐瓶颈 |
| 低 | 高 | 某级内存或数据通路可能是吞吐瓶颈 |
| 高 | 高 | 计算与搬运较平衡，需判断谁在关键路径 |
| 低 | 低 | 并行度不足、依赖延迟、同步或工作量太小 |

这里的“高”没有跨架构固定阈值。对于一个成熟 GEMM，80% 可能仍有价值；对于包含多种阶段的融合 Kernel，某个单元达到 50% 已可能是关键限制。

### 8.2 Memory Throughput 高不等于 DRAM 高

Memory Throughput 是内存系统高层汇总。它可能由以下任一单元主导：

```text
L1/TEX
L2
DRAM
Shared Memory 路径
片上互连
```

因此：

```text
Memory Throughput = 90%
```

不能直接写成：

```text
HBM 带宽已经达到 90%
```

必须在 Memory Workload Analysis 中展开，看 `dram__throughput`、L2、L1/TEX 和 Shared 具体哪一级高。

### 8.3 Compute Throughput 高不等于 FLOPs 高

地址计算、整数 ALU、类型转换、特殊函数和 Tensor Core 都属于计算资源的一部分。一个只有少量浮点运算的索引 Kernel，也可能因为整数 Pipe 或特殊指令达到较高 Compute Throughput。

下一步应看：

- Compute Workload Analysis 的各 Pipe Utilization。
- Instruction Statistics 的 SASS Mix。
- Source Page 上热点指令。

### 8.4 两者都低时不要立刻判定“还有很大优化空间”

小 Grid Kernel 只占用少量 SM 时，全芯片 SOL 自然很低。单个 SM 上可能已经无法更快。

依赖链很长时，执行单元大部分周期没有指令输入，SOL 也会低。此时增加算术强度未必有效，应该先解决 Eligible Warp 和 ILP。

所以 SOL 是入口，不是最终诊断。

## 九、Roofline：计算上限、带宽上限与算术强度

Roofline 用两个硬件上限约束可达到的计算性能：

```text
P <= P_compute_peak
P <= Bandwidth_peak * Arithmetic_Intensity
```

其中：

```text
Arithmetic Intensity = FLOPs / Bytes
```

于是：

```text
P_attainable = min(
    P_compute_peak,
    Bandwidth_peak * Arithmetic_Intensity
)
```

### 9.1 Vector Add 为什么天然偏 Memory-bound

对 FP32 Vector Add：

```cpp
c[i] = a[i] + b[i];
```

理想情况下每个元素：

```text
读取 a: 4 Bytes
读取 b: 4 Bytes
写入 c: 4 Bytes
计算:   1 FLOP

Arithmetic Intensity = 1 / 12 FLOP/Byte
```

它几乎不可能靠提高 FP32 算力显著加速。正确方向是减少字节数或把 Add 融合进前后 Kernel。

如果表达式是：

```cpp
y[i] = a * x[i] + y[i];
```

FMA 通常按 2 FLOPs 计，字节数仍约为 12：

```text
Arithmetic Intensity = 2 / 12 = 1 / 6 FLOP/Byte
```

仍然是低算术强度。

### 9.2 GEMM 为什么可以变成 Compute-bound

对 `M x K` 与 `K x N` GEMM：

```text
FLOPs ~= 2 * M * N * K
```

通过 Tiling，每个加载到 Shared Memory 或寄存器的数据参与多次 FMA，实际算术强度可以显著提高。当点跨过 Ridge Point 后，性能上限从带宽斜线转为计算横线。

### 9.3 Roofline 中的 Bytes 必须指定层级

同一个 Kernel 在不同内存层级有不同 Arithmetic Intensity：

```text
AI_DRAM = FLOPs / DRAM Bytes
AI_L2   = FLOPs / L2 Bytes
AI_L1   = FLOPs / L1 Bytes
```

一个 GEMM 可能对 DRAM 有很高复用，但在 Shared Memory 上产生巨大流量。只看 DRAM Roofline 会漏掉片上瓶颈。

### 9.4 点落在 Roof 下方不一定是“带宽没优化好”

Roofline 描述吞吐上限，但不自动解释为什么没有接近上限。点远离所有 Roof 可能来自：

- Grid 太小。
- Warp Divergence。
- Barrier。
- Long Dependency Chain。
- 非浮点指令主导。
- Load Imbalance。
- Tail Effect。
- 指令发射或内存请求队列受限。

Roofline 适合做瓶颈分类和性能上限估算，不能替代 Scheduler 与 Source 分析。

## 十、Launch Statistics：Grid、Block 与 Wave Quantization

Launch Statistics 是最容易被忽略、但成本最低的一组信息。

重点查看：

```text
Grid Size
Block Size
Registers Per Thread
Static Shared Memory Per Block
Dynamic Shared Memory Per Block
Threads
Waves Per SM
```

### 10.1 Waves Per SM

假设：

```text
GPU 有 S 个 SM
每个 SM 最多同时驻留 B 个目标 CTA
Grid 一共有 G 个 CTA
```

一轮完整 Wave 最多容纳：

```text
S * B 个 CTA
```

近似有：

```text
Waves Per SM = G / (S * B)
```

解释：

```text
Waves < 1:
  Grid 不足以填满所有并发槽位

Waves ~= 整数:
  CTA 分布通常较完整

Waves = k + 很小的小数:
  最后一轮只有少量 CTA，可能出现明显 Tail
```

例如前 4 个 Wave 都填满 GPU，最后只剩 5 个 CTA。Kernel 末尾大部分 SM 会空闲，但 Duration 要等这 5 个 CTA 完成。

### 10.2 Wave Quantization 的优化方向

- 改变 Tile Size，增加或调整 CTA 数量。
- 对归约维使用 Split-K。
- 对 Attention KV 维使用 Split-KV。
- 让不同 CTA 工作量更均衡。
- 使用 Work Queue 或 Persistent Scheduling。
- 合并多个小 Batch。

不要为了让 Wave 数好看而盲目减小 Tile。Tile 变小会降低数据复用、增加边界处理和 Epilogue 数量。应比较总 Duration。

### 10.3 Block Size 不是越大越好

大 Block 可能：

- 减少 Grid CTA 数。
- 增加每 CTA 寄存器和 Shared Memory。
- 降低每 SM 可驻留 CTA 数。
- 增加 Barrier 等待范围。

小 Block 也可能：

- 每个 CTA 工作太少。
- 增加调度与边界开销。
- 降低跨线程数据复用。

Block Size 必须和 Tile、资源占用、Scheduler 及 Memory Transaction 一起看。

### 10.4 检查 SM 与 L2 Slice 间的工作量不均

全芯片平均值可能掩盖实例间不均衡。例如一部分 SM 工作很久，其他 SM 很早结束：

```text
SM Active Cycles Max 明显高于 Avg
Achieved Occupancy 低于 Theoretical Occupancy
末尾只有少量 SM Active
```

这通常来自：

- CTA 工作量与数据内容相关。
- 不同 Tile 的稀疏度、Mask 或有效长度不同。
- Persistent Work Queue 分配不均。
- 最后一个 Partial Wave。

L2 Slice 也可能不均衡。若地址分布、Partition Camping 或热点数据让少量 Slice 承担大部分 Sector，平均 L2 Throughput 看起来不高，但最忙 Slice 已接近上限。

应比较相关实例的 `min/avg/max`，并在 NCU GUI 中查看 Workload Distribution 或吞吐分解。优化方向包括重新编号 Tile、交错地址、动态任务队列和更细粒度任务切分。

## 十一、Occupancy：理论驻留能力不等于执行效率

Occupancy 定义为：

```text
Occupancy =
Active Warps Per SM / Maximum Warps Per SM
```

NCU 通常同时给出：

```text
Theoretical Occupancy
Achieved Occupancy
```

### 11.1 Theoretical Occupancy 由什么限制

每个 SM 能驻留多少 CTA，受多个上限共同约束：

```text
Block Limit SM
Block Limit Warps
Block Limit Registers
Block Limit Shared Memory
Block Limit Blocks
```

最终：

```text
Resident Blocks Per SM =
min(各资源允许的 Block 数)
```

注意资源按硬件粒度分配。寄存器和 Shared Memory 的实际占用可能向上取整，不是简单连续除法。

### 11.2 Achieved 低于 Theoretical 说明什么

常见原因：

- Grid 太小，无法填满所有 SM。
- 最后一个 Partial Wave 占比大。
- CTA 工作量不均衡。
- 部分 Warp 提前退出。
- Kernel 运行太短，时间平均受到启动和收尾阶段影响。

因此很大的 Theoretical/Achieved Gap 通常应该和 Waves、SM 间工作分布一起看。

### 11.3 Occupancy 高为什么仍然可能慢

高 Occupancy 只表示有很多 Warp 驻留，不表示这些 Warp：

- 正在执行指令。
- 有合并的内存访问。
- 没有 Bank Conflict。
- 没有 Barrier。
- 使用了 Tensor Core。

大量 Warp 可以同时卡在同一条长延迟依赖上。

### 11.4 Occupancy 低什么时候值得优化

同时满足下列证据时，Occupancy 更可能是真问题：

```text
Achieved Occupancy 低
Eligible Warps Per Scheduler 低
No Eligible 周期多
Long Scoreboard 或 Execution Dependency 高
目标 Pipe 和 Memory Throughput 都未饱和
```

此时可以尝试：

- 降低每线程寄存器，但检查 Spill。
- 减少每 CTA Shared Memory。
- 调整 Block Size。
- 缩小 Tile。
- 减少不必要的循环展开。

如果低 Occupancy 下 Tensor Core 已接近峰值，强行提高 Occupancy 通常没有收益，甚至会破坏寄存器复用。

## 十二、Scheduler Statistics：每个周期为什么发不出指令

Scheduler Statistics 是连接“硬件不忙”和“Warp 在等什么”的关键。

对每个 Warp Scheduler，可以把 Warp 分成：

```text
Theoretical Warps:
  Launch 配置理论上允许分配给 Scheduler 的 Warp

Active Warps:
  已驻留、尚未完成的 Warp

Eligible Warps:
  下一条指令已经就绪，可以发射的 Warp

Issued Warps:
  本周期实际被 Scheduler 选中并发射指令的 Warp
```

关系近似为：

```text
Issued <= Eligible <= Active <= Theoretical
```

### 12.1 Active 少

说明可供 Scheduler 隐藏延迟的 Warp 不足。继续检查：

- Occupancy 是否被寄存器或 Shared Memory 限制。
- Grid/Waves 是否过小。
- Block 是否只有很少 Warp。

### 12.2 Active 多，但 Eligible 少

这是典型的延迟隐藏失败：

```text
有很多 Warp 驻留
但大多数 Warp 都在等待依赖、内存或同步
```

下一步看 Warp Stall：

- Long Scoreboard：常见于 Global/Local Load 结果未返回。
- Short Scoreboard：常见于 Shared Memory 或其他较短依赖。
- Execution Dependency：ALU 结果依赖链。
- Barrier：CTA 同步等待。
- Throttle：请求队列或 Pipe 无法继续接收指令。

### 12.3 Eligible 多，但 Issued 接近发射上限

Scheduler 手里有足够工作，也在持续发射。这往往表示：

- 某条执行 Pipe 正在被高效使用。
- `Not Selected` Stall 可能升高，因为多个 Eligible Warp 竞争发射槽。

此时 `Not Selected` 高通常不是坏事。它说明 Warp 已经准备好，只是本周期另一个 Warp 被选中。应检查 Compute 或 Memory Pipe 是否接近吞吐上限，而不是试图“消灭 Not Selected”。

### 12.4 Issued 很低且 No Eligible 很高

说明大量发射槽浪费。优化重点是产生更多 Eligible Warp：

- 提高 TLP：更多可驻留 Warp。
- 提高 ILP：同一 Warp 中更多独立指令。
- 缩短依赖链。
- 更早发起异步拷贝或 Load。
- 减少 Barrier 和负载不均。

## 十三、Warp Stall：每种停顿究竟说明什么

不同版本的指标名称可能带有 `warp_issue_stalled`、`warps_issue_stalled` 或派生 Ratio。应优先在 Warp State Statistics 和 Source Counters 中查看，再查询当前版本的完整 Metric 名。

### 13.1 先理解 Stall 百分比的分母

Warp Stall 指标通常描述 Active Warp 在采样或统计周期中的状态分布。它不等于：

```text
Kernel 有 40% 的墙钟时间完全浪费在 Long Scoreboard
```

多个 Warp 可以在同一周期处于不同状态，而 Scheduler 仍可能从另一个 Eligible Warp 发出指令。

只有 Scheduler 显示 Eligible Warp 不足、Issue Slot 经常空闲时，高 Stall 才直接指向吞吐损失。

### 13.2 Long Scoreboard

含义：

```text
Warp 的下一条指令依赖一个尚未完成的 L1TEX 相关操作
```

常见来源：

- Global Memory Load。
- Local Memory Load，也就是可能的 Spill。
- Texture 或 Surface Load。
- Cache Miss 后等待 L2/DRAM。
- 不规则 Pointer Chasing。

优化方向：

- 合并访问，减少 Sector。
- 提高 L1/L2 数据复用。
- 减少 Spill。
- Prefetch 和软件流水。
- 增加独立 Load 或独立计算。
- 增加足够的并发 Warp。
- 对 Pointer Chasing 改变数据结构。

Long Scoreboard 高不自动等于 DRAM 带宽瓶颈。它更接近“等待 Load 延迟”。DRAM Throughput 可能很低，因为请求太少或访问过于串行。

### 13.3 Short Scoreboard

通常表示等待较短的 Scoreboard 依赖，常见来源是 MIO 路径上的 Shared Memory 操作。

重点检查：

- Shared Load/Store Bank Conflict。
- Shared Memory 数据依赖距离太短。
- `ldmatrix` 或片上数据搬运后的立即消费。
- Shared Memory Pipe 是否饱和。

优化方向：

- Padding 或 Swizzle 消除 Bank Conflict。
- 在 Load 和 Use 之间插入独立计算。
- Double Buffer。
- 调整 Warp 的数据映射。

### 13.4 Barrier

Warp 正在等待 CTA 级 Barrier，例如 `__syncthreads()` 或相关同步原语。

高 Barrier Stall 可能来自：

- Barrier 太频繁。
- Barrier 前不同 Warp 工作量不一致。
- 某些 Warp 的 Global Load 更慢。
- CTA 内 Producer/Consumer 分工不平衡。
- Tile 边界导致部分 Warp 走更长路径。

优化方向：

- 减少不必要的 Block 级同步。
- 能用 `__syncwarp()` 时不要扩大到整个 CTA。
- 平衡 Barrier 前工作量。
- 使用异步 Pipeline 和更细粒度 Barrier。
- 调整 Warp Specialization 的 Producer/Consumer 比例。

不要直接删除 Barrier。必须先证明不存在跨 Warp 数据依赖。

### 13.5 MIO Throttle

MIO 指令队列或相关 Pipe 无法继续接收指令。常见相关操作包括：

- Shared Memory。
- 特殊数学或其他通过 MIO 的操作。
- 部分数据移动指令。

优化方向依赖 Source/SASS：

- 减少 Shared 指令数。
- 提高每次 Shared Load 的有效数据量。
- 改用 Shuffle 传递 Warp 内数据。
- 将某些索引或转换与其他 Pipe 交错。

### 13.6 LG Throttle

Local/Global Memory 指令发射过密，相关队列出现背压。

常见原因：

- 每做很少计算就发很多 Global Load/Store。
- Spill 产生大量 Local Load/Store。
- 非合并访问让请求膨胀。

优化方向：

- 减少内存指令数量。
- 使用向量化 Load/Store，但先保证对齐与连续。
- 提高寄存器或 Shared Memory 复用。
- 消除 Spill。
- 融合相邻访问。

### 13.7 Math Pipe Throttle

目标数学 Pipe 已经接近可接收指令的上限。它通常是吞吐瓶颈证据，而不是调度故障。

下一步：

- 找到具体是 FP32、FP64、Tensor、ALU、SFU 还是其他 Pipe。
- 减少该类指令。
- 使用更合适的数据类型或 Tensor Core。
- 将部分工作转移到其他 Pipe。
- 用近似数学指令时验证数值误差。

### 13.8 Wait / Execution Dependency

不同 NCU 版本可能将这类状态显示为 `Wait` 或 Execution Dependency。Warp 在等待前序固定延迟指令的结果，典型是串行累加：

```cpp
for (int k = 0; k < K; ++k) {
    sum = fmaf(a[k], b[k], sum);
}
```

每次 FMA 都依赖上一次 `sum`。可以使用多个独立累加器：

```cpp
float sum0 = 0.0f;
float sum1 = 0.0f;

for (int k = 0; k < K; k += 2) {
    sum0 = fmaf(a[k],     b[k],     sum0);
    sum1 = fmaf(a[k + 1], b[k + 1], sum1);
}

float sum = sum0 + sum1;
```

代价是更多寄存器。需要同时看 Occupancy 与 Spill。

### 13.9 Not Selected

Warp 是 Eligible 状态，但 Scheduler 选择了另一个 Warp。

高 `Not Selected` 往往表示：

```text
Scheduler 有足够的 Ready Warp
延迟隐藏较好
当前发射槽或执行 Pipe 已被其他 Warp 使用
```

它通常不需要单独优化。只有在某类优先级、调度公平性或特定 Warp Specialization 问题中，才需要进一步分析。

### 13.10 Branch Resolving、No Instruction 与其他 Stall

- `Branch Resolving`：等待分支目标或分支结果解析。
- `No Instruction`：下一条指令尚未可获取，可能涉及指令 Cache 或 Fetch。
- `Membar`：等待内存屏障要求满足。
- `Drain`：Warp 收尾时等待未完成操作排空，常与工作不均和 Store 有关。
- `Sleep`：执行显式 Sleep/Nanosleep。

这些状态通常需要 Source/SASS 和算法上下文。不要只根据名字做修改。

## 十四、Memory Workload：从请求追到 DRAM

内存分析最重要的不是只看一个带宽，而是沿请求路径追踪：

```text
Thread Address
  -> Warp Memory Instruction
  -> Request
  -> 32-Byte Sectors
  -> L1/TEX Wavefront
  -> L2 Sectors
  -> DRAM Bytes
```

### 14.1 Request

一个 Warp 执行一条内存指令时，会形成一个或多个请求。Request 不是每线程一个。

### 14.2 Sector

现代 NVIDIA Cache 统计常以 32-Byte Sector 为粒度。一个 Warp 的地址覆盖多少 Sector，直接反映访问是否连续、对齐和稀疏。

以 32 个线程各读一个 FP32 为例：

```text
有效数据 = 32 * 4 B = 128 B
理想且对齐时 = 4 个 32 B Sector
```

如果地址跨步很大，最坏可能接近：

```text
32 个线程触及 32 个不同 Sector
```

实际传输数据相对有效数据膨胀约 8 倍。

### 14.3 Wavefront

L1TEX 每个周期能处理的 Sector 数有限。一个 Request 若包含过多 Sector，可能拆成多个 Wavefront。

因此：

```text
Request 数不变
Sector 数增加
Wavefront 数增加
L1TEX 占用时间和下游流量增加
```

Shared Memory Bank Conflict 也可能表现为 Excessive Wavefront，因为一次访问需要被序列化成多个阶段。

### 14.4 Bytes 与 Throughput

建议同时看：

```text
DRAM Read/Write Bytes
L2 Read/Write Sectors 或 Bytes
L1/TEX Sectors
实际 GB/s
pct_of_peak_sustained
```

Bytes 回答“做了多少工作”，Throughput 回答“单位时间做得多快”。优化后 Duration 下降时，GB/s 可能上升，即使 Bytes 已下降；这不是矛盾。

### 14.5 判断 Memory-bound 的完整证据

比较可信的证据链：

```text
DRAM Throughput 接近可持续峰值
DRAM Bytes 与算法预期一致或偏高
Roofline 位于 DRAM Bandwidth Roof 附近
Compute Pipe 未成为更强限制
增加计算并行度不能提升性能
减少 DRAM Bytes 后 Duration 近似同比下降
```

只有 `Long Scoreboard` 高，不足以证明 Bandwidth-bound。

## 十五、Global Memory：用 Sector 判断合并访问

下面用两个简单 Kernel 对比连续访问和跨步访问。

### 15.1 可编译示例

```cpp
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(err));      \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

__global__ void copy_contiguous(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int index = row * width + col;
        output[index] = input[index];
    }
}

__global__ void copy_strided_read(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int output_index = row * width + col;
        int input_index = col * height + row;
        output[output_index] = input[input_index];
    }
}

int main(int argc, char** argv) {
    constexpr int width = 4096;
    constexpr int height = 4096;
    constexpr int warmup = 10;
    const size_t bytes = static_cast<size_t>(width) * height * sizeof(float);

    float* input = nullptr;
    float* output = nullptr;
    CHECK_CUDA(cudaMalloc(&input, bytes));
    CHECK_CUDA(cudaMalloc(&output, bytes));
    CHECK_CUDA(cudaMemset(input, 0, bytes));

    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    bool strided = argc > 1 && std::string(argv[1]) == "strided";
    for (int i = 0; i < warmup + 1; ++i) {
        if (strided) {
            copy_strided_read<<<grid, block>>>(
                input, output, width, height);
        } else {
            copy_contiguous<<<grid, block>>>(
                input, output, width, height);
        }
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(output));
    CHECK_CUDA(cudaFree(input));
    return 0;
}
```

编译：

```bash
nvcc -O3 -lineinfo -Xptxas=-v -arch=sm_XX \
  ncu_demo.cu -o ncu_demo
```

采集连续版本：

```bash
ncu \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  --launch-skip 10 \
  --launch-count 1 \
  -o contiguous \
  ./ncu_demo
```

采集跨步版本：

```bash
ncu \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  --launch-skip 10 \
  --launch-count 1 \
  -o strided \
  ./ncu_demo strided
```

### 15.2 为什么连续版本更容易合并

`block.x = 32`，同一 Warp 的 `threadIdx.x` 连续：

```text
input[index + 0]
input[index + 1]
...
input[index + 31]
```

一个 Warp 读取 128 Bytes 连续且对齐的数据，理想情况下覆盖 4 个 32-Byte Sector。

跨步版本中：

```text
input[row + 0  * height]
input[row + 1  * height]
...
input[row + 31 * height]
```

相邻 Lane 相差 `height * sizeof(float)` Bytes。每个 Lane 几乎都会落到不同 Sector。

### 15.3 预期看到的指标方向

| 指标 | 连续访问 | 跨步读取 |
| --- | --- | --- |
| Global Load Sectors/Request | 接近理想值 | 显著增加 |
| L1/TEX Wavefront | 少 | 多 |
| DRAM/L2 Bytes | 接近有效字节 | 可能膨胀 |
| Long Scoreboard | 较低或可隐藏 | 更容易升高 |
| Duration | 较短 | 较长 |

具体 Hit Rate 取决于 Cache 容量、架构和 Replay 策略，不应预设固定值。

### 15.4 常用底层 Metric

不同版本名称可能变化，常见 Base Metric 方向包括：

```text
l1tex__t_requests_*_mem_global_op_ld
l1tex__t_sectors_*_mem_global_op_ld
l1tex__t_requests_*_mem_global_op_st
l1tex__t_sectors_*_mem_global_op_st
dram__bytes_read
dram__bytes_write
```

核心派生量：

```text
Load Sectors Per Request =
Global Load Sectors / Global Load Requests
```

理想值取决于：

- 每 Lane 访问的字节数。
- Active Lane 数量。
- 地址对齐。
- 是否跨越 Cache Line。
- 指令是否为向量 Load。

不能把“4 Sectors/Request”当作所有访问的固定标准。

### 15.5 Vectorized Load 的正确判断

将 4 次标量 Load 改成 `float4` 可能减少指令数，但需要：

- 地址满足 16-Byte 对齐。
- 每线程读取连续 16 Bytes。
- Warp 地址整体仍然连续。
- Tail 正确处理。
- 寄存器压力可接受。

应验证：

```text
SASS Load 指令数是否下降
Sector 数是否没有额外增加
LG Throttle 是否下降
Registers Per Thread 是否恶化
Duration 是否下降
```

## 十六、Cache：Hit Rate 高为什么仍然可能很慢

Hit Rate 是最常被孤立解读的内存指标。

### 16.1 Hit Rate 高不表示流量少

假设一个 Kernel 对同一批数据重复加载 100 次，全部命中 L2：

```text
L2 Hit Rate 很高
但 L2 -> L1 流量仍然巨大
Load 指令和 Scoreboard 依赖仍然存在
L2 或片上互连仍可能饱和
```

更好的优化是把数据提升到 Shared Memory 或寄存器中复用，而不是满足于 L2 Hit。

### 16.2 Hit Rate 低不一定有问题

流式读取只使用一次的数据，本来就没有时间局部性：

```text
L1 Hit Rate 低
但访问完全合并
DRAM 带宽接近峰值
```

这可能已经是合理实现。强行缓存会污染 L1，并挤出真正有复用的数据。

### 16.3 Hit Rate 要和 Bytes 一起看

设：

```text
L2 Requests = 1000
L2 Hits = 900
L2 Misses = 100
Hit Rate = 90%
```

另一个实现：

```text
L2 Requests = 100
L2 Hits = 50
L2 Misses = 50
Hit Rate = 50%
```

第二个实现 Hit Rate 更低，但总请求和 Miss 都更少，可能明显更快。

优先比较：

```text
绝对 Request/Sector/Byte
然后再看 Hit Rate
最后看 Duration
```

### 16.4 L1 与 Shared Memory 的容量权衡

许多架构上 L1 和 Shared Memory 使用统一或可配置的片上容量。增加 Dynamic Shared Memory 可能改变可用 L1 容量和 Occupancy。

引入 Shared Tiling 后应该同时检查：

- DRAM Bytes 是否下降。
- L2/L1 Bytes 是否下降。
- Shared Bytes 和 Bank Conflict 是否上升。
- 每 CTA Shared Memory 是否降低 Occupancy。
- 总 Duration 是否下降。

## 十七、Shared Memory：Bank Conflict 与 Wavefront

Shared Memory 通常由多个 Bank 组成。对常见的 32-Bank 组织，连续 FP32 Lane 访问通常映射到不同 Bank：

```text
bank = (byte_address / 4) % 32
```

具体 Bank 宽度和特殊访问规则应以目标架构文档为准。

### 17.1 构造 32-Way Bank Conflict

```cpp
template <int Padding>
__global__ void shared_access(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n) {
    extern __shared__ float raw_smem[];
    volatile float* smem = raw_smem;

    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    int stride = 32 + Padding;
    int warp_base = warp * warpSize * stride;
    int smem_index = warp_base + lane * stride;

    float value = global_index < n ? input[global_index] : 0.0f;
    smem[smem_index] = value;
    __syncwarp();

    if (global_index < n) {
        output[global_index] = smem[smem_index] + 1.0f;
    }
}
```

启动无 Padding 版本：

```cpp
constexpr int threads = 256;
int warps = threads / 32;
size_t bytes = warps * 32 * 32 * sizeof(float);

shared_access<0><<<blocks, threads, bytes>>>(input, output, n);
```

Lane 地址：

```text
lane 0  -> index 0
lane 1  -> index 32
lane 2  -> index 64
...
```

对 FP32 而言，这些地址都映射到同一个 Bank，但不是同一地址广播，因此会产生严重 Conflict。

加入一列 Padding：

```cpp
size_t bytes = warps * 32 * 33 * sizeof(float);

shared_access<1><<<blocks, threads, bytes>>>(input, output, n);
```

此时：

```text
bank = (lane * 33) % 32 = lane
```

不同 Lane 分散到不同 Bank。

### 17.2 NCU 中看什么

常见指标方向：

```text
Shared Load Bank Conflicts
Shared Store Bank Conflicts
Shared Load/Store Transactions
Excessive Wavefronts
Short Scoreboard
MIO Throttle
L1/TEX Throughput
```

预期：

```text
Padding 后 Bank Conflict 和 Excessive Wavefront 下降
Short Scoreboard 或 MIO 压力下降
Duration 下降
```

### 17.3 不要只看 Conflict 次数

Conflict 次数必须结合 Shared 指令总量。一个 Kernel 有少量 Conflict，但只执行一次；另一个 Kernel 每轮没有 Conflict，却执行了大量 Shared Load。后者仍可能更慢。

还要注意：

- 多个 Lane 读取同一地址可能走广播，不是 32-Way Conflict。
- 64-bit 或 128-bit 访问会跨多个 Bank。
- `ldmatrix`、Tensor Core Layout 和 Swizzle 有自己的访问模式。
- Padding 可能增加 Shared Memory 占用并降低 Occupancy。

## 十八、Local Memory：识别 Register Spilling

CUDA Local Memory 是线程私有地址空间，但物理数据通常位于 Device Memory，并经过 L1/L2 Cache。它不是片上的“本地高速内存”。

### 18.1 什么会进入 Local Memory

- 寄存器压力过大导致 Spill。
- 大型线程局部数组。
- 编译器无法静态展开索引的局部数组。
- 大型局部结构体。
- 函数调用和 Stack Frame。

编译时先看：

```bash
nvcc -O3 -lineinfo -Xptxas=-v kernel.cu -o kernel
```

典型输出字段：

```text
Used N registers
N bytes spill stores
N bytes spill loads
N bytes stack frame
```

### 18.2 NCU 中看什么

常见 Base Metric 方向：

```text
l1tex__t_sectors_*_mem_local_op_ld
l1tex__t_sectors_*_mem_local_op_st
```

并结合：

```text
Local Load/Store Instructions
Long Scoreboard
LG Throttle
Registers Per Thread
Achieved Occupancy
Source/SASS 的 LDL、STL 或相关访问
```

### 18.3 限制寄存器为什么可能更慢

使用：

```bash
nvcc --maxrregcount=64 ...
```

可能提高理论 Occupancy，但也可能把活跃值 Spill 到 Local Memory：

```text
Occupancy 上升
Local Sectors 上升
Long Scoreboard 上升
Duration 变长
```

因此优化目标不是最少寄存器，而是最短 Duration 下的寄存器、ILP 和 Occupancy 平衡。

### 18.4 降低寄存器压力的常见方式

- 缩短变量 Live Range。
- 减少过度展开。
- 缩小每线程 Tile。
- 分阶段计算，避免同时保留所有中间值。
- 重新安排 Producer/Consumer Warp 的职责。
- 对只读值使用合理的共享或重新计算策略。

重新计算有时比 Spill 更便宜，但必须看增加了哪条 Compute Pipe 的压力。

## 十九、Compute Workload：找到真正饱和的执行管线

Compute Workload Analysis 会显示 SM 上不同执行管线的利用率。常见类别包括：

```text
FP32 / FMA
FP64
Tensor
ALU / Integer
SFU / Special Function
Uniform
Load/Store 相关 Pipe
```

具体 Pipe 命名和能力随架构变化。

### 19.1 高 Pipe Utilization 的含义

如果 FP32 Pipe 接近可持续上限，同时：

```text
Eligible Warps 充足
Scheduler 持续 Issued
Memory 没有更强限制
```

那么 FP32 指令吞吐很可能是瓶颈。

优化方向：

- 用 FMA 合并乘加。
- 减少重复算术。
- 使用更低精度。
- 将矩阵运算映射到 Tensor Core。
- 将某些整数或转换工作移出循环。

### 19.2 Tensor Pipe 低不一定表示 Tensor Core 没用好

可能原因：

- Kernel 同时包含 Load、Softmax、Epilogue 等非 MMA 阶段。
- Tile 太小，MMA 在总 Duration 中占比低。
- Grid 太小，只有部分 SM 执行 MMA。
- Tensor Core 在 Active 周期很高，但 Elapsed 归一化较低。
- 数据准备或 Shared Memory 阻塞 Tensor Core。

应同时看：

```text
Tensor Pipe Active 与 Elapsed 归一化
MMA 指令数
Scheduler Eligible Warps
Short Scoreboard / MIO Throttle
Shared Memory Throughput
Source Timeline 或分阶段 Kernel
```

### 19.3 SFU 或特殊数学成为瓶颈

`exp`、`log`、`sin`、倒数和某些转换可能使用特殊函数路径。Softmax、Normalization 或复杂激活中，FLOPs 数量不大，但特殊 Pipe 可能饱和。

优化方向：

- 使用硬件支持的近似指令。
- 用 `exp2` 重写，但正确处理缩放。
- 查表或多项式近似。
- 把特殊函数与独立 MMA/Load 重叠。
- 降低重复计算次数。

所有近似优化必须给出误差上界和端到端数值验证。

### 19.4 Integer/ALU Pipe 高

常见于：

- 复杂 Tensor Index。
- 除法、取模。
- 64-bit 地址运算。
- 动态 Shape 和多级间接寻址。
- 稀疏格式解码。

优化方向：

- 预计算 Stride 和 Offset。
- 将除常数改为乘法、位移或模板常量。
- 使用增量 Pointer，而不是循环内重复构造多维索引。
- 将不变索引移到 CTA/Warp 共享层级。
- 在保证范围安全时减少不必要的 64-bit 算术。

### 19.5 Pipe 低但 Execution Dependency 高

这表示吞吐上限没有打满，问题是指令依赖链，而不是指令数量。

常见改法：

- 多累加器。
- Loop Unroll 增加 ILP。
- 将 Load 提前若干 Iteration。
- 重排互不依赖的计算。

但 Unroll 会增加寄存器和代码尺寸，应重新检查：

```text
Registers
Spill
Occupancy
No Instruction
Duration
```

## 二十、Instruction、Source 与 SASS：把指标定位到代码

高层 Section 告诉你“哪类资源有问题”，Source Page 才能回答“哪一行造成问题”。

### 20.1 为什么必须看 SASS

CUDA 源码不是 GPU 最终执行的指令：

- 编译器可能消除、合并或移动代码。
- 短分支可能被 Predication。
- `float4` 不保证一定生成预期宽度的事务。
- 局部数组可能被放进寄存器，也可能进入 Local Memory。
- FMA 可能由乘加融合生成。
- 模板和常量传播会改变循环结构。

因此需要将 CUDA、PTX 和 SASS 对照。

### 20.2 常见指令类别

不同架构助记符会变化，常见方向：

| 类别 | 常见含义 |
| --- | --- |
| `LDG` / Global Load | 从 Global 地址空间加载 |
| `STG` / Global Store | 写入 Global 地址空间 |
| `LDS` / Shared Load | 从 Shared Memory 加载 |
| `STS` / Shared Store | 写入 Shared Memory |
| `LDL` / Local Load | Local Memory 访问，可能是 Spill |
| `STL` / Local Store | Local Memory 写入，可能是 Spill |
| `FFMA` | FP32 Fused Multiply-Add |
| `HMMA` / `MMA` | Tensor Core 相关矩阵乘加 |
| `BRA` / `BRX` | 分支 |
| `BAR` | Barrier |

不要跨架构死记助记符，应在目标 GPU 的 Source/SASS 页面确认。

### 20.3 Instruction Count 能说明什么

对同一输入和同一算法，Executed Instruction 明显下降通常是好信号。但还要区分：

- Warp-Level Instruction。
- Thread-Level Instruction。
- Predicated-On Thread Instruction。
- Issued 与 Executed。

例如一条 Warp 指令只让 8 个 Lane 有效：

```text
Warp Instruction Count 看起来正常
Predicated-On Thread 数量很低
实际线程利用率差
```

### 20.4 Source Counters 的正确用法

在 Source Page 叠加：

```text
Warp Stall Samples
Global/Local/Shared Memory 指标
Branch Targets
Predicated-On Threads
Instruction Count
```

然后沿最重的源码行检查：

```text
这行生成了哪些 SASS？
它的输入来自哪个 Load？
它是否位于循环内部？
每个线程都会执行吗？
是否被边界 Predicate 屏蔽？
```

Stall 经常显示在“等待结果的消费指令”上，而不是最初发出 Load 的源码行。分析时要向前追数据依赖。

## 二十一、Branch 与线程利用率：识别 Warp Divergence

Warp Divergence 发生在同一个 Warp 的 Lane 走不同控制流路径。硬件需要分别执行各路径，并通过 Active Mask 屏蔽不属于当前路径的 Lane。

### 21.1 一个简单对照

发散版本：

```cpp
__global__ void divergent(
    const float* input,
    float* output,
    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float x = input[i];
    int lane = threadIdx.x & 31;

    if (lane < 16) {
        #pragma unroll 1
        for (int k = 0; k < 32; ++k) {
            x = fmaf(x, 1.0001f, 0.0001f);
        }
    } else {
        #pragma unroll 1
        for (int k = 0; k < 32; ++k) {
            x = __fdividef(x + 0.0001f, 1.0001f);
        }
    }

    output[i] = x;
}
```

同一个 Warp 中一半 Lane 走 FMA，另一半走 Divide，两条路径都需要执行。

如果把条件改为 Warp 粒度：

```cpp
int warp_global = i / 32;

if ((warp_global & 1) == 0) {
    // 整个 Warp 走 FMA
} else {
    // 整个 Warp 走 Divide
}
```

总工作分布近似不变，但 Warp 内不再分裂。

### 21.2 NCU 中看什么

- Branch Efficiency 或相关 Source Counter。
- Avg. Active Threads Per Warp。
- Avg. Not Predicated-Off Threads Per Warp。
- Branch Instructions。
- Source 上每条路径的 Active Mask。
- Warp Stall 中 Branch Resolving。
- Duration。

### 21.3 短分支可能不会生成真正分支

编译器可能使用 Predication：

```cpp
if (condition) {
    y = x + 1.0f;
}
```

可能变成所有 Lane 发出同一条指令，仅通过 Predicate 决定哪些 Lane 写回。此时 Branch Instruction 不一定增加，但 Predicated-On Thread 利用率会下降。

所以不能只看 Branch Count。

### 21.4 边界分支通常不是主要问题

```cpp
if (i < n) {
    ...
}
```

通常只有最后一个 Warp 或少量边界 CTA 发散，占比很小。除非 Tensor 很小、边界 Tile 很多或 Padding 浪费显著，否则不应优先优化它。

更值得关注的是循环内部、每个 Tile 都执行的数据相关分支。

## 二十二、四类典型报告如何推导优化动作

下面不使用固定数值，而是强调指标之间的因果关系。

### 22.1 案例一：连续 Vector Add

报告特征：

```text
Grid 足够大
DRAM Throughput 高
Global Sectors/Request 接近理想
Compute Pipe 低
Eligible Warp 尚可
```

结论：

```text
访问已经合并，Kernel 接近 DRAM Bandwidth-bound
```

低价值改动：

- 继续增加 Block 数。
- 把简单 Add 手工展开很多次。
- 只追求更高 Occupancy。

高价值方向：

- 与前后算子融合。
- 减少输入或输出。
- 使用更窄的数据类型。
- 确认是否可以 In-place。

### 22.2 案例二：跨步 Gather

报告特征：

```text
Global Load Sectors/Request 高
L1/TEX Wavefront 多
Long Scoreboard 高
DRAM Throughput 未必达到峰值
Eligible Warp 少
```

结论：

```text
每条 Load 的空间利用率差，且 Warp 在等待高延迟、不规则访问
这是访问效率和延迟问题，不一定是带宽饱和
```

优化方向：

- 交换线程与数据维度映射。
- 预转置或改变 Layout。
- 让 Warp 访问连续维。
- Shared Memory 中做局部重排。
- 对索引排序或分桶，提高局部性。

### 22.3 案例三：高寄存器 GEMM

报告特征：

```text
Registers Per Thread 高
Theoretical Occupancy 低
但 Tensor Pipe 高
Eligible Warp 足够
Local Load/Store 为 0
```

结论：

```text
低 Occupancy 可能是高寄存器复用的合理代价
当前 Tensor Core 已被充分供给
```

危险改动：

```text
强制 maxrregcount
-> Occupancy 上升
-> Spill 出现
-> Long Scoreboard 上升
-> Duration 变差
```

应先优化：

- MMA 与 Load Pipeline 重叠。
- Shared Memory Layout。
- Epilogue。
- Tile 与 Wave Quantization。

### 22.4 案例四：归约 Kernel Barrier 很高

报告特征：

```text
Compute 与 DRAM 都不高
Active Warp 不少
Eligible Warp 少
Barrier Stall 高
不同 Warp 的工作量不一致
```

结论：

```text
CTA 内同步和负载不均阻止 Scheduler 获得 Ready Warp
```

优化方向：

- Warp Shuffle 替代部分 Shared Reduction。
- 先做 Warp 内归约，只让每 Warp 一个 Lane 写 Shared。
- 减少 `__syncthreads()` 次数。
- 避免交错寻址分支。
- 对尾部使用 Warp 同步。

验证：

```text
Barrier Stall 下降
Shared 指令下降
Eligible Warps 上升
Duration 下降
```

## 二十三、最容易误读的 NCU 指标

### 23.1 “Occupancy 越高越好”

错误。Occupancy 是延迟隐藏的容量，不是实际吞吐。高性能 GEMM 常用大量寄存器保存 Accumulator，Occupancy 不高但 Tensor Core 很忙。

### 23.2 “Long Scoreboard 高就是 DRAM Bandwidth-bound”

错误。它表示等待内存相关依赖。Pointer Chasing 可能 Long Scoreboard 很高，但 DRAM 带宽很低，因为并发请求不足。

### 23.3 “Memory Throughput 高就是 HBM 满了”

错误。高值可能来自 L1/TEX、L2、Shared 或某条片上数据路径。必须展开 Memory Chart。

### 23.4 “L2 Hit Rate 越高越好”

错误。Hit Rate 不反映总请求数。请求减少一半但 Hit Rate 下降，仍可能更快。

### 23.5 “Not Selected 是 Warp 被浪费”

通常错误。它经常说明有多个 Eligible Warp，Scheduler 正常选择其中一个发射。

### 23.6 “某个 Stall 占 40%，消掉后就能快 40%”

错误。Warp Stall 的分母通常不是 Kernel 墙钟时间，且多个 Warp 状态重叠。消除一个 Stall 后，其他瓶颈会接管。

### 23.7 “Compute Throughput 低说明计算优化空间大”

错误。Memory-bound Kernel 本来就不需要高 Compute Throughput。优化计算不会降低 DRAM 时间。

### 23.8 “用了 Shared Memory 就减少了 DRAM”

不一定。若原数据已命中 L1/L2，或 Shared Tile 没有被复用，可能只是增加了一次搬运、Barrier 和 Shared 流量。

### 23.9 “Vectorized Load 一定减少内存流量”

不一定。它主要减少指令数和请求组织开销。若地址不对齐、Tail 处理不当或线程映射不连续，Sector 数可能不降反升。

### 23.10 “`--set full` 的报告最可信”

错误。`full` 只是指标更多，也带来更多 Replay、保存恢复和运行扰动。针对假设采集最少必要 Section，通常更容易得到稳定结论。

## 二十四、一套可复用的算子优化流程

### 24.1 优化前先写理论账本

对给定 Shape 计算：

```text
理论 FLOPs
最少 DRAM Read/Write Bytes
输出 Tile 数
CTA 数与 Waves
每个元素理想访问次数
允许的数值误差
```

没有理论账本，就无法判断 NCU 中的 Bytes 和 Instructions 是必要工作还是浪费。

### 24.2 第一轮：分类

采集：

```text
SpeedOfLight
LaunchStats
Occupancy
```

回答：

```text
Grid 够不够？
哪种资源限制驻留？
Compute 或 Memory 是否有明显高值？
```

### 24.3 第二轮：确认调度状态

采集：

```text
SchedulerStats
WarpStateStats
```

回答：

```text
Active Warp 是否足够？
Eligible Warp 是否足够？
Issue Slot 是否经常空闲？
主要 Stall 是依赖、同步还是 Throttle？
```

### 24.4 第三轮：深挖限制子系统

Memory 路径：

```text
MemoryWorkloadAnalysis
SourceCounters
```

检查：

```text
Request
Sector/Request
Wavefront
Cache Bytes/Hit
DRAM Bytes/Throughput
Shared Conflict
Local Memory
```

Compute 路径：

```text
ComputeWorkloadAnalysis
InstructionStats
SourceCounters
```

检查：

```text
具体 Pipe
SASS Mix
Predicated-On Threads
Dependency
转换与索引指令
```

### 24.5 为每次修改定义可证伪预测

示例：

| 修改 | 预期 NCU 变化 | 最终目标 |
| --- | --- | --- |
| Warp 连续访问 | Sectors/Request 下降 | Duration 下降 |
| Shared Padding | Bank Conflict 下降 | Duration 下降 |
| 多累加器 | Execution Dependency 下降 | Pipe 利用率提高 |
| 减少 Unroll | Registers/Spill 下降 | Duration 下降 |
| Split-K | Waves 增加 | 总 GEMM + Reduction 下降 |
| Kernel Fusion | DRAM Bytes 下降 | 端到端延迟下降 |

如果预测没有发生，说明对编译结果或硬件行为的假设不成立，应回到 SASS 和原始 Metric。

### 24.6 用 Baseline 比绝对阈值更可靠

不同 GPU、频率、Shape 和 NCU 版本之间，绝对百分比不总可比。最可靠的比较是：

```text
同一机器
同一 Shape
同一输入
同一编译选项
同一 NCU Section 和 Replay 配置
Baseline vs Candidate
```

重点比较三层：

```text
工作量:
  Bytes、Sectors、Instructions、Transactions

效率:
  Throughput、Eligible Warp、Stall、Pipe Utilization

结果:
  无 Profiler Duration、吞吐、端到端延迟、数值误差
```

### 24.7 最后的检查清单

```text
[ ] Kernel 确实是端到端热点
[ ] Release 编译并带 lineinfo
[ ] 输入、Shape、时钟和缓存语义可复现
[ ] Grid/Waves 足够，或已识别并行度问题
[ ] SOL 只用于分类，没有直接下结论
[ ] Roofline 的 FLOPs 和 Bytes 层级明确
[ ] Occupancy 与 Eligible Warp 联合分析
[ ] Stall 与 Scheduler Issue Slot 联合分析
[ ] Memory 从 Request/Sector 追到 DRAM
[ ] Cache Hit Rate 与绝对 Bytes 联合分析
[ ] Source 指标已经追到 SASS 和数据依赖
[ ] 每个修改都有预期指标变化
[ ] 最终性能在无 NCU 环境下复测
[ ] 数值正确性和误差没有退化
```

## 二十五、参考资料

1. [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
2. [NVIDIA Nsight Compute CLI User Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
3. [NVIDIA Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/)
4. [NVIDIA Nsight Compute Release Notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/)
5. [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
6. [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
7. [NVIDIA Nsight Compute Roofline Analysis](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)
8. [Nsight Compute Kernel Profiling Guide: Memory Chart and Tables](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#memory-chart)

NCU 的价值不在于把所有百分比都调高，而在于建立一条可验证的因果链：

```text
算法需要多少工作
-> 实际产生了多少指令和内存事务
-> 哪个硬件单元最接近上限
-> Scheduler 为什么无法继续发射
-> 哪一行源码和哪条 SASS 造成等待
-> 修改后工作量、效率和最终延迟是否同时改善
```

成熟的 CUDA 优化不是看到 `Occupancy` 低就提高 Occupancy，也不是看到 `Long Scoreboard` 高就堆更多 Warp。真正有效的方法是先判断 Throughput、Latency 和 Parallelism 中谁在限制性能，再用 NCU 的多组指标互相验证，最后回到无 Profiler 的端到端结果。
