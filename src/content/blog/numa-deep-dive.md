---
title: 'NUMA 详解：CPU、内存、GPU 与网卡拓扑为什么会影响性能'
description: '面向 AI Infra 和高性能计算场景，系统讲解 NUMA 的硬件背景、本地/远端内存访问、CPU/GPU/NIC 亲和性、性能问题与排查思路。'
category: 'Research & Work'
pubDate: '2026-07-09T10:55:00+08:00'
updatedDate: '2026-07-09T10:55:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [为什么会有 NUMA](#二为什么会有-numa)
3. [UMA 与 NUMA 的区别](#三uma-与-numa-的区别)
4. [NUMA node 到底是什么](#四numa-node-到底是什么)
5. [本地内存和远端内存](#五本地内存和远端内存)
6. [一个具体拓扑例子](#六一个具体拓扑例子)
7. [Linux 如何看 NUMA](#七linux-如何看-numa)
8. [NUMA 和 CPU Affinity](#八numa-和-cpu-affinity)
9. [NUMA 和 GPU / PCIe / NIC](#九numa-和-gpu--pcie--nic)
10. [NUMA 对 AI Infra 的影响](#十numa-对-ai-infra-的影响)
11. [常见性能问题](#十一常见性能问题)
12. [如何做 NUMA-aware 优化](#十二如何做-numa-aware-优化)
13. [面试表达](#十三面试表达)
14. [总结](#十四总结)

## 一、核心结论

`NUMA` 是 Non-Uniform Memory Access，意思是“非一致内存访问”。

它描述的是这样一种机器结构：

```text
CPU 访问不同位置的内存，延迟和带宽并不一样。
```

在多 CPU socket 服务器上，每个 CPU socket 通常有自己直接连接的内存。一个 CPU 访问自己 socket 附近的内存更快，访问另一个 socket 管辖的内存更慢。

NUMA 对 AI Infra 和高性能计算很重要，因为现代服务器不只有 CPU 和内存，还会有：

- 多张 GPU。
- 多块 NIC / HCA。
- 多条 PCIe root complex。
- NVLink / NVSwitch。
- InfiniBand / RoCE 网络。
- CPU 侧数据加载、预处理、调度线程。

如果进程、CPU 线程、内存、GPU、网卡不在同一个局部拓扑附近，就可能出现：

```text
CPU 远端访存
GPU 跨 socket 访问 host memory
网卡 DMA 到远端 NUMA 内存
数据加载线程和目标 GPU 不亲和
通信线程被调度到错误 CPU node
```

最终表现可能是：

- 吞吐低。
- P99 延迟高。
- CPU 使用率看起来不高但性能差。
- GPU 等数据。
- 网络带宽打不满。
- 多卡性能不随卡数线性增长。

一句话记忆：

```text
NUMA 不是一个命令，也不是一个软件配置项，而是硬件拓扑对内存访问和设备访问成本的影响。
```

## 二、为什么会有 NUMA

早期单 CPU 机器中，CPU 和内存之间关系比较简单：

```text
CPU -> Memory Controller -> DRAM
```

CPU 访问任何内存位置的成本大致相同，这种结构可以近似理解为 UMA。

随着服务器需要更多核心和更大内存，单个 CPU socket 已经不够。于是出现多 socket 服务器：

```text
Socket 0 + local DRAM
Socket 1 + local DRAM
Socket 2 + local DRAM
...
```

每个 socket 都有自己的内存控制器和本地 DRAM。socket 之间通过互联链路连接，例如 Intel UPI、AMD Infinity Fabric。

这样做的好处是：

- 扩展更多 CPU 核心。
- 扩展更大内存容量。
- 每个 socket 有自己的内存通道，提高总体内存带宽。

问题是：

```text
一个 CPU 访问自己 socket 的内存很近；
访问另一个 socket 的内存要经过 socket 间互联，路径更远。
```

所以访问成本变成非一致的，这就是 NUMA。

## 三、UMA 与 NUMA 的区别

### UMA

`UMA` 是 Uniform Memory Access。

近似特点：

```text
所有 CPU 访问所有内存的延迟和带宽差不多。
```

可以抽象成：

```text
CPU0 \
CPU1  -> shared memory
CPU2 /
```

对程序员来说，内存位置不太重要。

### NUMA

`NUMA` 是 Non-Uniform Memory Access。

近似特点：

```text
CPU 访问本地内存快；
CPU 访问远端内存慢。
```

可以抽象成：

```text
NUMA node 0:
  CPU cores 0-31
  local DRAM 0
  GPU0 / NIC0 nearby

NUMA node 1:
  CPU cores 32-63
  local DRAM 1
  GPU1 / NIC1 nearby
```

CPU0 访问 node 0 的内存是 local access；访问 node 1 的内存是 remote access。

## 四、NUMA node 到底是什么

`NUMA node` 可以理解为一组“距离比较近”的硬件资源。

通常包括：

- 一组 CPU cores。
- 一组本地内存。
- 该 socket 下挂的 PCIe 设备。
- 可能包括 GPU、NIC、NVMe 等。

一个双路服务器可能有两个 NUMA node：

```text
node 0: CPU socket 0 + local memory + some PCIe devices
node 1: CPU socket 1 + local memory + some PCIe devices
```

一个 AMD EPYC 机器可能更复杂。即使是单 socket，也可能因为 chiplet / CCD / memory domain 被划分成多个 NUMA node。

所以不能简单认为：

```text
NUMA node == CPU socket
```

更准确的说法是：

```text
NUMA node 是操作系统看到的一组局部性更强的 CPU 和内存资源。
```

## 五、本地内存和远端内存

假设一台双路机器：

```text
Socket 0: CPU0 cores + Memory0
Socket 1: CPU1 cores + Memory1
```

如果运行在线程 `T0` 上的代码被调度到 Socket 0 的 CPU core：

```text
T0 访问 Memory0 -> local memory access
T0 访问 Memory1 -> remote memory access
```

本地访问路径：

```text
CPU core -> local memory controller -> local DRAM
```

远端访问路径：

```text
CPU core -> socket interconnect -> remote socket memory controller -> remote DRAM
```

远端访问的问题：

- 延迟更高。
- 带宽更低。
- 占用 socket 间互联带宽。
- 多线程下更容易争抢互联链路。

对于 memory-bound 程序，远端访存会非常明显。比如图计算、稀疏算子、数据预处理、KV Cache CPU offload、embedding lookup，都可能受影响。

## 六、一个具体拓扑例子

假设机器拓扑：

```text
NUMA node 0
  CPU cores: 0-31
  Memory: 512 GB
  PCIe devices:
    GPU0
    GPU1
    NIC0

NUMA node 1
  CPU cores: 32-63
  Memory: 512 GB
  PCIe devices:
    GPU2
    GPU3
    NIC1
```

如果一个推理服务使用 GPU0，但它的 CPU 线程被调度到 node 1，内存也主要分配在 node 1：

```text
CPU thread on node 1
memory on node 1
GPU0 under node 0
```

这时 CPU 准备数据给 GPU0，路径可能跨 NUMA：

```text
node 1 CPU / memory -> socket interconnect -> node 0 PCIe root -> GPU0
```

如果服务还用 NIC0 做网络收发，但网络线程也在 node 1：

```text
NIC0 interrupt / DMA / network thread 可能跨 node
```

性能可能比全部放在 node 0 差很多：

```text
CPU cores 0-31
Memory node 0
GPU0
NIC0
```

NUMA-aware 部署的目标就是让相关资源尽量靠近。

## 七、Linux 如何看 NUMA

常用命令如下。

### lscpu

```bash
lscpu
```

重点看：

```text
NUMA node(s)
NUMA node0 CPU(s)
NUMA node1 CPU(s)
```

示例：

```text
NUMA node(s):          2
NUMA node0 CPU(s):     0-31
NUMA node1 CPU(s):     32-63
```

这说明系统有两个 NUMA node，并且每个 node 对应一组 CPU cores。

### numactl

```bash
numactl --hardware
```

常见输出：

```text
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 ... 31
node 0 size: 515000 MB
node 0 free: 320000 MB
node 1 cpus: 32 33 ... 63
node 1 size: 515000 MB
node 1 free: 300000 MB
node distances:
node   0   1
  0:  10  21
  1:  21  10
```

`node distances` 越小表示越近。通常本地 node 是 10，远端 node 是更大的数。

### numastat

```bash
numastat
numastat -p <pid>
```

`numastat -p` 可以看某个进程的内存主要分布在哪些 NUMA node 上。

如果进程主要跑在 node 0 的 CPU，但内存大量分布在 node 1，就可能存在远端访存。

### nvidia-smi topo

GPU 机器上常用：

```bash
nvidia-smi topo -m
```

它可以看到 GPU、NIC、CPU Affinity、NUMA Affinity 之间的拓扑关系。

常见字段：

```text
GPU0 GPU1 NIC0 CPU Affinity NUMA Affinity
```

如果 GPU0 和 NIC0 在同一个 NUMA node，跨机通信或 GPUDirect RDMA 的路径通常更友好。

## 八、NUMA 和 CPU Affinity

`CPU Affinity` 是把进程或线程绑定到指定 CPU cores。

为什么它和 NUMA 有关？

因为 CPU core 属于某个 NUMA node。如果线程频繁访问某个 node 的本地内存，最好把线程调度到同一个 node 的 CPU core。

示例：

```bash
numactl --cpunodebind=0 --membind=0 ./server
```

含义：

```text
让进程在 NUMA node 0 的 CPU 上运行；
让进程优先从 NUMA node 0 分配内存。
```

也可以只绑 CPU：

```bash
taskset -c 0-31 ./server
```

但只绑 CPU 不一定保证内存在同一个 node。内存分配还受 first-touch 策略影响。

### first-touch 策略

Linux 常见内存策略是 first-touch：

```text
哪一个 CPU 第一次写某块内存，这块内存倾向于分配到该 CPU 所属 NUMA node。
```

这意味着初始化线程很重要。

如果主线程在 node 0 初始化大数组，然后 worker 线程在 node 1 长期访问它，就会出现远端访存。

更好的做法是：

```text
让负责访问数据的线程自己初始化对应数据；
或者显式使用 numactl / libnuma 控制内存绑定。
```

## 九、NUMA 和 GPU / PCIe / NIC

AI Infra 中，NUMA 不只影响 CPU 访存，还影响设备访问路径。

### GPU 和 NUMA

GPU 通常挂在某个 PCIe root complex 下，而 PCIe root complex 属于某个 CPU socket / NUMA node。

如果 CPU 线程准备数据给 GPU：

```text
CPU memory -> PCIe -> GPU
```

最好让：

```text
CPU thread
host memory
target GPU
```

尽量处于同一个 NUMA node 附近。

否则 host-to-device copy 可能跨 socket。

### NIC 和 NUMA

NIC / HCA 也挂在某个 PCIe root complex 下。

网络包收发、DMA、IRQ 处理、通信线程，都和 NUMA 相关。

理想情况：

```text
NIC0 under NUMA node 0
communication thread on node 0 CPU
receive/send buffer on node 0 memory
GPU0 also near node 0
```

如果 NIC 在 node 0，通信线程在 node 1，就可能出现：

```text
NIC DMA / CPU processing / memory access 跨 NUMA
```

### GPU 和 NIC 的亲和性

分布式训练或推理中，经常关心：

```text
GPU 到 NIC 是否同 NUMA？
GPU 到 NIC 是否经过同一个 PCIe switch？
GPU 到 GPU 是否走 NVLink/NVSwitch？
```

例如一台 8 卡机器有两块 NIC：

```text
GPU0-3 near NIC0
GPU4-7 near NIC1
```

部署通信进程时，如果 GPU0 的流量走 NIC1，就可能路径更远。

## 十、NUMA 对 AI Infra 的影响

### 1. 数据加载和预处理

训练或推理服务中，CPU 可能负责：

- tokenizer。
- image decode。
- feature preprocessing。
- batch construction。
- pinned memory staging。

如果这些 CPU 线程远离目标 GPU，数据搬运路径会变长。

### 2. Host-to-device copy

CPU 内存到 GPU 显存的拷贝通常经过 PCIe。

如果 host memory 在远端 NUMA node，GPU 在本地 node：

```text
remote memory -> inter-socket link -> PCIe -> GPU
```

相比本地内存：

```text
local memory -> PCIe -> GPU
```

会更慢，也会占用 socket 间互联。

### 3. 分布式通信

NCCL、MPI、RDMA、RoCE 等通信链路都可能受拓扑影响。

常见问题：

- GPU 到 NIC 跨 NUMA。
- NIC IRQ 绑错 CPU。
- 通信线程被调度到远端 node。
- host staging buffer 分配到远端 node。

### 4. CPU offload

大模型推理中可能出现 CPU offload：

- 权重 offload。
- KV Cache offload。
- optimizer state offload。
- embedding table offload。

如果 GPU 频繁访问远端 CPU 内存，性能会明显受 NUMA 影响。

### 5. 多实例部署

一台机器上跑多个推理实例时，如果不做 NUMA-aware 资源划分，可能出现：

```text
实例 A 用 GPU0，但 CPU 在 node 1
实例 B 用 GPU3，但内存在 node 0
两个实例抢同一 NUMA node 的内存带宽
```

更好的做法是按拓扑切分：

```text
实例 A: node 0 CPU + node 0 memory + GPU0/1 + NIC0
实例 B: node 1 CPU + node 1 memory + GPU2/3 + NIC1
```

## 十一、常见性能问题

### 问题 1：CPU 使用率不高，但吞吐低

可能原因：

```text
线程在等远端内存；
跨 socket 访问导致延迟高；
内存带宽被 socket interconnect 限制。
```

排查：

```bash
numastat -p <pid>
numactl --hardware
perf stat -e cycles,instructions,cache-misses <cmd>
```

### 问题 2：GPU 利用率低，数据准备慢

可能原因：

```text
CPU 数据加载线程和 GPU 不在同一个 NUMA node；
pinned memory 在远端 node；
H2D copy 路径跨 socket。
```

排查：

```bash
nvidia-smi topo -m
numastat -p <pid>
```

### 问题 3：网卡带宽打不满

可能原因：

```text
NIC 亲和的 CPU node 不对；
IRQ affinity 不合理；
通信线程跨 NUMA；
buffer 在远端内存。
```

排查：

```bash
nvidia-smi topo -m
lspci -tv
cat /proc/interrupts
```

### 问题 4：多卡扩展效率差

可能原因：

```text
GPU/NIC 拓扑不匹配；
跨 socket 通信比例高；
CPU 线程和 GPU/NIC 绑定混乱；
多个任务抢同一 NUMA node 的内存带宽。
```

排查：

```bash
nvidia-smi topo -m
numactl --hardware
NCCL_DEBUG=INFO
```

## 十二、如何做 NUMA-aware 优化

### 1. 先看拓扑

不要凭感觉绑定 CPU 和 GPU。先看：

```bash
lscpu
numactl --hardware
nvidia-smi topo -m
lspci -tv
```

核心问题：

```text
GPU 属于哪个 NUMA node？
NIC 属于哪个 NUMA node？
哪些 CPU cores 属于同一个 node？
进程内存主要分布在哪个 node？
```

### 2. CPU 线程绑定到靠近设备的 node

如果服务使用 GPU0，而 GPU0 靠近 node 0：

```bash
numactl --cpunodebind=0 --membind=0 ./server --gpu 0
```

如果一台机器跑多个实例：

```bash
numactl --cpunodebind=0 --membind=0 ./server --gpu 0
numactl --cpunodebind=1 --membind=1 ./server --gpu 1
```

实际 node 和 GPU 的对应关系要以拓扑为准。

### 3. 初始化和访问放在同一个 node

避免一个 node 初始化内存，另一个 node 长期访问。

例如多线程程序中：

```text
每个 worker 初始化自己负责的数据分片；
或者在绑定后的线程中分配内存。
```

### 4. 通信线程靠近 NIC

对于网络通信密集任务：

- 通信线程尽量绑定到 NIC 所属 NUMA node。
- buffer 尽量分配在同 node。
- GPU 和 NIC 之间尽量选择更近路径。

### 5. 容器和调度系统也要感知 NUMA

Kubernetes / Slurm / Yarn 等调度系统中，如果只分配 GPU，不分配对应 CPU 和内存 node，仍然可能出现跨 NUMA。

更合理的资源分配应该考虑：

```text
GPU
CPU cores
memory node
NIC
PCIe topology
```

## 十三、面试表达

可以这样解释 NUMA：

```text
NUMA 是多 socket 或多内存域机器上的非一致内存访问模型。
每个 NUMA node 包含一组 CPU cores 和本地内存。
CPU 访问本地内存更快，访问其他 node 的远端内存要经过 socket 间互联，延迟更高、带宽更低。
```

如果面试官问为什么 AI Infra 关心 NUMA：

```text
AI 服务器里 GPU、NIC、NVMe 都挂在 PCIe 拓扑下，而 PCIe root complex 通常归属于某个 NUMA node。
如果 CPU 数据加载线程、host memory、GPU、NIC 不在同一个局部拓扑附近，数据搬运和通信就可能跨 socket，导致 GPU 等数据、网络带宽打不满、P99 延迟变高。
```

如果问怎么排查：

```text
先用 lscpu 或 numactl --hardware 看 NUMA node 和 CPU core 映射；
用 nvidia-smi topo -m 看 GPU、NIC、CPU Affinity、NUMA Affinity；
用 numastat -p 看进程内存分布；
再结合 perf、NCCL_DEBUG、网络指标判断是否存在跨 NUMA 访问。
```

如果问怎么优化：

```text
让相关资源尽量靠近。
例如使用 GPU0 和 NIC0 的进程，应尽量绑定到它们附近的 CPU cores，并把内存分配在同一个 NUMA node。
多实例部署时按 NUMA node 切分 CPU、内存、GPU、NIC，减少跨 socket 数据路径。
```

## 十四、总结

NUMA 的核心不是“有几个 node”，而是：

```text
计算在哪里执行；
内存在哪里分配；
设备挂在哪里；
数据路径是否跨 node。
```

对 AI Infra 来说，NUMA 影响的不只是 CPU 访存，还影响：

- CPU 数据预处理。
- GPU host staging。
- H2D / D2H copy。
- NIC 中断和 DMA。
- 多卡通信。
- CPU offload。
- 多实例部署隔离。

最实用的判断是：

```text
线程、内存、GPU、NIC 尽量同 NUMA；
跨 NUMA 不一定错误，但高带宽、低延迟路径中要尽量避免。
```

如果只记一句话：

```text
NUMA-aware 优化的目标，是让数据尽量在离计算和设备最近的地方被分配、访问和传输。
```
