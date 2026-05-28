---
title: 'CUDA Profiling：Nsight Compute 与 Nsight Systems 常用指标'
description: '区分 NCU 和 NSYS 的定位，理解 Occupancy、Compute Workload、Memory Bandwidth 等常用指标如何指导 CUDA 优化。'
category: 'CUDA'
pubDate: '2026-06-19'
updatedDate: '2026-06-19'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [NSYS 和 NCU 的区别](#二nsys-和-ncu-的区别)
3. [先看时间线](#三先看时间线)
4. [Occupancy](#四occupancy)
5. [Compute Workload](#五compute-workload)
6. [Memory Bandwidth](#六memory-bandwidth)
7. [常见瓶颈判断](#七常见瓶颈判断)
8. [优化流程](#八优化流程)
9. [面试回答模板](#九面试回答模板)
10. [总结](#十总结)

## 一、核心结论

Profiling 的目标不是收集指标，而是定位瓶颈。

- Nsight Systems（NSYS）看系统级时间线：CPU、GPU、Memcpy、Kernel、Stream 是否重叠。
- Nsight Compute（NCU）看单个 kernel 的细节：访存、计算、Occupancy、Warp stall 等。
- 先用 NSYS 找慢在哪里，再用 NCU 深挖具体 kernel。
- Occupancy 不等于性能，只表示 SM 上活跃 Warp 的比例。
- Memory Bandwidth 高且算力低，通常说明 Memory-bound。
- Compute Workload 高且访存不是瓶颈，可能是 Compute-bound。

## 二、NSYS 和 NCU 的区别

| 工具 | 主要用途 | 关注对象 |
| --- | --- | --- |
| Nsight Systems | 系统级时间线 | 程序整体、CPU/GPU 并发、Memcpy、Stream |
| Nsight Compute | Kernel 级分析 | 单个 kernel 的访存、计算、occupancy、stall |

常见命令：

```bash
nsys profile -o report ./app
```

```bash
ncu --set full ./app
```

实际工程中，通常先 `nsys`，再对热点 kernel 用 `ncu`。

## 三、先看时间线

NSYS 中重点看：

- GPU 是否有长时间空洞。
- Memcpy 是否和 Kernel 重叠。
- Kernel 是否很多且很短。
- CPU 是否频繁同步 GPU。
- 多 Stream 是否真的并发。

例如，如果时间线是：

```text
H2D -> Kernel -> D2H -> H2D -> Kernel -> D2H
```

说明传输和计算没有重叠。可以考虑 Streams 和 Ping-Pong Buffer。

如果时间线里很多小 kernel：

```text
kernel1 kernel2 kernel3 kernel4 ...
```

可能需要考虑 Kernel Fusion 或减少 launch overhead。

## 四、Occupancy

Occupancy 表示一个 SM 上实际活跃 Warp 数与理论最大活跃 Warp 数的比例。

影响因素：

- 每个线程使用的寄存器数。
- 每个 Block 使用的 shared memory。
- Block 大小。
- 硬件最大线程/Block/Warp 限制。

Occupancy 低可能导致无法隐藏访存延迟，但 Occupancy 高不代表一定快。

例子：

```text
Kernel A: Occupancy 30%, 每个线程做大量计算，数据复用好，可能很快。
Kernel B: Occupancy 90%, 但访存完全随机，仍然可能很慢。
```

因此 Occupancy 只是线索，不是目标本身。

## 五、Compute Workload

Compute Workload 相关指标用来判断计算单元是否繁忙。

关注：

- SM 是否忙。
- FP32/FP16/Tensor Core 利用率。
- 指令吞吐是否接近硬件上限。
- Warp stall 是否来自执行依赖或调度不足。

如果 Tensor Core 利用率低，但理论上应该用 Tensor Core，可能需要检查：

- 数据类型是否是 FP16/BF16/TF32/INT8 等合适类型。
- 矩阵维度是否满足库或硬件要求。
- 是否调用了正确的 cuBLAS/cuDNN/CUTLASS 路径。

## 六、Memory Bandwidth

Memory Bandwidth 指标用于判断显存带宽利用情况。

关注：

- Global Load/Store Throughput。
- L2 Hit Rate。
- DRAM Throughput。
- Memory Workload Analysis。
- Load/Store 是否合并。

如果 DRAM 带宽接近上限，而计算单元利用率不高，通常是 Memory-bound。

优化方向：

- 合并访问。
- 减少重复读写。
- 用 Shared Memory 复用数据。
- Kernel Fusion 减少中间结果写回。
- 使用更紧凑数据类型。

## 七、常见瓶颈判断

### 1. GPU 时间线有空洞

可能原因：

- CPU 端准备数据慢。
- 频繁同步。
- Kernel launch 间隔大。

优化：减少同步、批量提交、使用 CUDA Graph、改进数据加载。

### 2. Memcpy 占比高

可能原因：Host/Device 数据来回拷贝太多。

优化：减少传输、pinned memory、Streams 重叠、数据常驻 GPU。

### 3. Memory-bound

现象：显存带宽高，计算利用率低。

优化：访存合并、Shared Memory、Fusion、减少不必要读写。

### 4. Compute-bound

现象：计算单元利用率高，访存不是主要瓶颈。

优化：Tensor Core、指令级优化、减少分支、循环展开。

### 5. Warp stall 高

需要看具体 stall 类型：

- Memory Dependency：等内存。
- Barrier：等同步。
- Not Selected：可运行 Warp 多但未被调度。
- Execution Dependency：等前序指令结果。

## 八、优化流程

一个比较稳的流程：

1. 用 NSYS 看整体时间线，确认热点在哪里。
2. 找到最耗时 kernel。
3. 用 NCU 分析该 kernel 的访存、计算、Occupancy 和 stall。
4. 判断瓶颈是 Memory-bound、Compute-bound、同步、launch overhead 还是数据传输。
5. 做一个小改动。
6. 重新 profiling，对比指标和端到端时间。

不要只看单个指标，最终目标是端到端耗时下降。

## 九、面试回答模板

如果问题是“NCU 和 NSYS 有什么区别”，可以这样回答：

1. NSYS 是系统级 profiler，用来看 CPU/GPU 时间线、Memcpy、Kernel、Stream 重叠和同步开销。
2. NCU 是 kernel 级 profiler，用来看单个 kernel 的访存、计算、Occupancy、Warp stall 等细节。
3. 优化通常先用 NSYS 找端到端瓶颈，再用 NCU 分析热点 kernel。
4. Occupancy 表示活跃 Warp 比例，不等于性能。
5. Memory Bandwidth 高且计算利用率低，通常是 Memory-bound；计算单元利用率高则可能是 Compute-bound。

## 十、总结

CUDA 优化不能只凭经验。NSYS 回答“时间花在哪里”，NCU 回答“这个 kernel 为什么慢”。好的优化流程是先定位，再假设，再修改，再用指标验证。
