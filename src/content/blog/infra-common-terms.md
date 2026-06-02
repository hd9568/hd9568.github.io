---
title: 'Infra 领域常见术语速查：硬件、互联、网卡与性能指标'
description: '面向 AI Infra 工作场景，梳理 HBM、NVLink、PCIe、InfiniBand、eth0、mlx5_0、NUMA、TTFT 等常见术语。'
category: 'Research & Work'
pubDate: '2026-06-02T16:09:00+08:00'
updatedDate: '2026-06-02T16:09:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [阅读方式](#一阅读方式)
2. [内存与存储层级](#二内存与存储层级)
3. [GPU 计算相关](#三gpu-计算相关)
4. [互联与总线](#四互联与总线)
5. [网卡、接口与设备名](#五网卡接口与设备名)
6. [拓扑与部署](#六拓扑与部署)
7. [性能指标](#七性能指标)
8. [典型链路理解](#八典型链路理解)
9. [面试与工作表达](#九面试与工作表达)
10. [总结](#十总结)

## 一、阅读方式

Infra 领域的术语很多，但大部分可以放进一条主线：

```text
计算单元 -> 显存/内存 -> 机内互联 -> 跨机网络 -> 服务性能指标
```

如果只记孤立名词，很容易混乱。更好的方式是判断这个词属于哪一层：

- `HBM`、`L2 Cache`：数据在 GPU 内部怎么被访问。
- `PCIe`、`NVLink`、`NVSwitch`：一台机器内设备之间怎么连。
- `InfiniBand`、`Ethernet`：多台机器之间怎么通信。
- `eth0`、`mlx5_0`：Linux 或驱动层看到的设备名字。
- `TTFT`、`TPOT`、`P99`：最终服务表现如何衡量。

本文不展开 RDMA 通信协议细节，也不做命令手册，只解释工作中常被提到的核心概念。

## 二、内存与存储层级

### HBM

`HBM` 是 High Bandwidth Memory，高带宽显存。它通常和 GPU 封装在一起，带宽远高于普通 CPU 内存。

在大模型推理中，`HBM` 很关键，因为模型权重、KV Cache、激活、中间 buffer 都要放在 GPU 显存里。

常见判断：

```text
如果一个算子大量读写 HBM，但计算不多，它很可能是 memory-bound。
```

典型例子：Decode 阶段读取 KV Cache。每生成一个 token，都要读历史 K/V，计算量不算大，但 HBM 读流量很高。

### VRAM

`VRAM` 是显存的泛称。工作中说“这张卡 80G 显存”，通常就是指 GPU 上可用的显存容量。

`HBM` 更强调显存类型和带宽，`VRAM` 更偏泛称。

### DRAM

`DRAM` 通常指 CPU 侧系统内存。模型加载时，权重可能先在 CPU 内存，再搬到 GPU 显存。

如果出现 GPU 显存不足，可能会把部分权重或 KV Cache offload 到 CPU 内存，但这样访问会变慢，因为数据路径更长。

### SRAM

`SRAM` 是片上高速存储。GPU 的 cache、shared memory 这类片上存储常用 SRAM 实现。

FlashAttention 经常强调：

```text
尽量把中间块留在 SRAM/shared memory 中，不要频繁写回 HBM。
```

### L2 Cache

`L2 Cache` 是 GPU 上重要的片上缓存。它位于 SM 和 HBM 之间，可以缓存多次访问的数据。

在推理中，L2 命中率高通常意味着一部分数据被复用得好；如果 L2 命中率低，大量请求会直接打到 HBM，带宽压力更大。

### Register

`Register` 是线程私有的最快存储。CUDA kernel 中的局部变量通常放在寄存器里。

寄存器太少会影响计算效率；寄存器用太多又可能降低 occupancy，甚至产生 register spilling。

### Shared Memory

`Shared Memory` 是一个 CUDA block 内线程共享的片上高速存储。

常见用途：

- GEMM tiling 中缓存 A/B 矩阵块。
- Reduction 中保存中间结果。
- FlashAttention 中保存局部 tile。

注意：Shared Memory 不是越多越好。用得不合理会有 bank conflict，或者降低同一 SM 上可驻留的 block 数。

## 三、GPU 计算相关

### SM

`SM` 是 Streaming Multiprocessor，可以理解为 NVIDIA GPU 的主要计算单元。

一个 GPU 有多个 SM。kernel 的 block 会被调度到 SM 上执行。优化时经常关注：

- SM 是否吃满。
- 是否有足够的活跃 warp 隐藏访存延迟。
- 是否被显存带宽、同步或分支拖住。

### CUDA Core

`CUDA Core` 执行普通标量或向量计算，例如 FP32、INT32 操作。

它适合通用计算，但大模型里的大矩阵乘更希望用 Tensor Core。

### Tensor Core

`Tensor Core` 是专门做矩阵乘加的硬件单元。FP16、BF16、TF32、FP8、INT8 等低精度矩阵计算通常依赖它获得高吞吐。

如果模型推理没有用上 Tensor Core，常见原因包括：

- dtype 不合适。
- 矩阵维度不对齐。
- kernel 路径没有走到 Tensor Core 实现。
- batch 或 shape 太小，硬件利用率不高。

### Warp

`Warp` 是 NVIDIA GPU 的线程调度单位，通常 32 个线程一组。

同一个 warp 内线程执行同一条指令。如果这些线程走不同分支，会出现 warp divergence，导致部分路径串行执行。

### Occupancy

`Occupancy` 表示一个 SM 上实际活跃的 warp 数占理论最大值的比例。

它不是越高越好。更准确的理解是：

```text
Occupancy 太低可能隐藏不了访存延迟；但 Occupancy 高也不代表算子一定快。
```

最终还是要看瓶颈是计算、访存、同步，还是 launch overhead。

## 四、互联与总线

### PCIe

`PCIe` 是 CPU、GPU、网卡、SSD 等设备之间常见的高速总线。

工作中常见表达：

```text
PCIe Gen4 x16
PCIe Gen5 x16
```

`Gen` 表示代际，`x16` 表示通道数。代际越新、通道越多，理论带宽越高。

PCIe 常出现在这些场景：

- CPU 内存到 GPU 显存的数据传输。
- GPU 与网卡之间的数据路径。
- 多张 GPU 没有 NVLink 时的 P2P 通信。

### NVLink

`NVLink` 是 NVIDIA GPU 间的高速互联。相比 PCIe，它通常有更高带宽、更适合 GPU-GPU 通信。

常见场景：

- 张量并行中 GPU 之间 AllReduce/AllGather。
- 多卡推理时跨 GPU 传输中间激活。
- GPU P2P 直接访问。

如果两张 GPU 之间有 NVLink，通信通常比纯 PCIe 更友好。

### NVSwitch

`NVSwitch` 是 NVIDIA 的 GPU 互联交换芯片。它让多张 GPU 之间形成更高带宽、更规则的互联拓扑。

可以简单理解：

```text
NVLink 是线，NVSwitch 是交换结构。
```

在 8 卡或更多 GPU 机器中，NVSwitch 能让多 GPU 通信更接近“全互联”的体验。

### InfiniBand

`InfiniBand` 是高性能网络互联，常用于多机 GPU 集群。它的特点是低延迟、高带宽，适合训练和大规模推理中的跨机通信。

在 AI Infra 中，InfiniBand 常和这些场景相关：

- 多机多卡训练。
- 跨节点 tensor parallel / pipeline parallel。
- 大规模推理服务中的 KV 或激活传输。
- NCCL 跨机集合通信。

这里不展开 RDMA 协议细节，只需要知道：InfiniBand 是高性能集群网络，不是普通业务网络的简单替代品。

### Ethernet

`Ethernet` 是以太网，通用性强，生态成熟。常见速率包括 10G、25G、100G、200G、400G。

相比 InfiniBand，以太网更通用；但在极致低延迟、高性能集合通信场景，InfiniBand 仍然很常见。

### RoCE

`RoCE` 可以理解为在以太网上承载低延迟数据传输能力的一种方案。工作中经常和高性能以太网、无损网络、拥塞控制一起出现。

这里只需要建立一个粗略印象：

```text
InfiniBand 是专用高性能网络路线；RoCE 是在 Ethernet 体系上做高性能通信。
```

## 五、网卡、接口与设备名

### NIC

`NIC` 是 Network Interface Card，网卡。

在 GPU 服务器中，网卡很重要，因为多机训练/推理时，跨机通信往往经过网卡。

### HCA

`HCA` 是 Host Channel Adapter。在 InfiniBand 语境下，它通常指支持高性能网络能力的网卡。

可以近似理解为：

```text
HCA 是 IB/RDMA 语境下的网卡叫法。
```

### eth0

`eth0` 是 Linux 中常见的以太网接口名。

它通常代表一个网络接口，可以承载普通 IP 网络流量。很多容器、虚拟机或服务器里第一张以太网接口就叫 `eth0`。

需要注意：

```text
eth0 是操作系统看到的网络接口名，不等于物理网卡型号。
```

### ensX / enpXsY

这是 Linux 的 predictable network interface naming。名字会根据硬件位置、PCIe 拓扑等生成。

例如：

```text
ens5
ens10f0
enp129s0f0
```

这些名字看起来复杂，但本质上仍然是 Linux 网络接口名。

### ib0

`ib0` 常见于 InfiniBand 网络接口。

如果机器有 IB 网卡，系统中可能出现 `ib0`、`ib1` 这类接口。它们通常用于高性能集群网络。

### mlx5_0

`mlx5_0` 是 Mellanox/NVIDIA ConnectX 系列网卡在底层设备层的名字，常出现在 verbs、NCCL、驱动或硬件拓扑信息里。

它和 `eth0` 的区别：

| 名字 | 所在层次 | 含义 |
| --- | --- | --- |
| `eth0` | Linux 网络接口层 | 一个可配置 IP 的网络接口 |
| `ib0` | Linux 网络接口层 | 一个 IB/IPoIB 相关网络接口 |
| `mlx5_0` | 设备/驱动/verbs 层 | 一块 Mellanox/NVIDIA 网卡设备 |

一块物理网卡可能在不同层次暴露出不同名字。

### mlx5_core

`mlx5_core` 是 Mellanox/NVIDIA ConnectX 网卡的 Linux 内核驱动。

如果工作中看到 `mlx5_core` 报错，通常说明问题更靠近驱动、固件、PCIe 设备或内核网络栈，而不是普通应用层代码。

### ConnectX

`ConnectX` 是 NVIDIA/Mellanox 高性能网卡系列，例如 ConnectX-5、ConnectX-6、ConnectX-7。

训练集群和推理集群里经常使用这类网卡。

## 六、拓扑与部署

### Topology

`Topology` 指硬件连接关系。它回答的问题是：

```text
GPU 到 GPU 走 NVLink 还是 PCIe？
GPU 到网卡是否跨 NUMA？
网卡离哪张 GPU 更近？
```

这对性能很重要。同样是 8 张 GPU，如果拓扑不同，通信性能可能差很多。

### NUMA

`NUMA` 是 Non-Uniform Memory Access，多 CPU socket 机器上常见。

核心观点：

```text
CPU 访问本地内存更快，访问另一个 socket 的内存更慢。
```

GPU、网卡通常也挂在某个 CPU socket 下。如果进程绑定在远端 CPU 上，可能导致数据路径绕远。

### CPU Affinity

`CPU Affinity` 是把进程或线程绑定到指定 CPU 核心。

在高性能推理服务中，它常用于：

- 减少线程迁移。
- 让通信线程靠近对应网卡。
- 让数据准备线程靠近对应 GPU/NUMA node。

### IRQ Affinity

`IRQ Affinity` 是中断亲和性。它决定设备中断由哪些 CPU 核心处理。

如果网卡中断处理落在不合适的 CPU 上，可能增加延迟或造成单核瓶颈。

### P2P

`P2P` 是 Peer-to-Peer，表示设备之间直接通信。

GPU P2P 常见含义：

```text
GPU 之间不经过 CPU 内存中转，直接互相访问或传输数据。
```

如果 P2P 不可用，跨 GPU 数据传输可能退化为经过 CPU 内存，性能明显下降。

### BAR1

`BAR1` 是 GPU 暴露给 CPU 或其他设备访问的一段显存映射窗口。

它常出现在 GPU 设备管理、P2P、GPU Direct 相关场景。一般不需要记公式，只要知道它和“设备如何映射访问 GPU 显存”相关。

### IOMMU

`IOMMU` 管理设备 DMA 地址映射。它影响设备访问内存时的地址转换和隔离。

在高性能场景中，IOMMU 配置可能影响设备直通、DMA 性能或虚拟化环境里的设备访问。

### MIG

`MIG` 是 Multi-Instance GPU，可以把一张支持 MIG 的 NVIDIA GPU 切成多个隔离实例。

常见用途：

- 多用户共享一张 GPU。
- 小模型推理隔离部署。
- 提高资源利用率。

代价是单个实例只能使用部分 SM、显存和带宽，不适合所有 workload。

### MPS

`MPS` 是 Multi-Process Service，让多个 CUDA 进程更高效地共享 GPU。

它适合一些小 kernel、多进程并发的场景，但隔离性和资源控制方式与 MIG 不同。

### Container Network

容器里的网络接口名可能仍然叫 `eth0`，但它背后可能是虚拟网卡、宿主机桥接、SR-IOV VF 或直接挂载的高性能设备。

看容器内 `eth0` 时，需要区分：

```text
这是容器虚拟接口，还是直通/挂载的真实高性能网卡能力？
```

### Kubernetes Device Plugin

K8s 中 GPU、RDMA 网卡、MIG 实例等设备通常通过 device plugin 暴露给容器。

它解决的是资源发现和分配问题：

```text
调度器如何知道某个节点有几张 GPU、几个 MIG 实例或哪些设备可以分给容器。
```

## 七、性能指标

### Bandwidth

`Bandwidth` 是带宽，表示单位时间能传多少数据。

常见单位：

- `GB/s`：常用于 HBM、PCIe、NVLink。
- `Gbps`：常用于网络链路，例如 100Gbps、400Gbps。

注意大小写：

```text
B = Byte 字节
b = bit 比特
1 Byte = 8 bit
```

### Latency

`Latency` 是延迟，表示一次操作从发起到完成需要多久。

高带宽不一定低延迟。大规模数据传输看带宽；小消息、高频请求更怕延迟。

### Throughput

`Throughput` 是吞吐，表示单位时间处理多少工作。

在推理服务中可能是：

- requests/s。
- tokens/s。
- batch/s。

### QPS

`QPS` 是 Queries Per Second，每秒请求数。

它适合衡量服务请求吞吐，但对 LLM 不够完整，因为不同请求的输入/输出 token 数可能差异很大。

### Tokens/s

`Tokens/s` 是大模型推理更常用的吞吐指标。

需要区分：

- Prefill tokens/s：处理输入 token 的速度。
- Decode tokens/s：生成输出 token 的速度。
- 单请求 tokens/s 与整体服务 tokens/s。

### TTFT

`TTFT` 是 Time To First Token，首 token 延迟。

它主要受这些因素影响：

- 排队时间。
- tokenizer 和请求处理。
- prefill 计算。
- 调度策略。

长 prompt 的 TTFT 通常更高。

### TPOT

`TPOT` 是 Time Per Output Token，平均每个输出 token 的耗时。

它更反映 Decode 阶段速度。Decode 阶段常受 KV Cache 读取、batching 和调度影响。

### P50 / P95 / P99

这些是延迟分位数。

- `P50`：一半请求比它快。
- `P95`：95% 请求比它快。
- `P99`：99% 请求比它快。

线上服务特别关注 P99，因为用户体感和 SLA 往往被长尾请求影响。

### Utilization

`Utilization` 是利用率。常见说法有：

- GPU 利用率。
- SM 利用率。
- HBM 带宽利用率。
- 网络带宽利用率。

单看 GPU 利用率不够。比如 GPU 利用率高，可能是在等内存；SM 利用率不高，也可能是网络或 CPU 调度拖慢。

## 八、典型链路理解

### 单机单卡推理

```text
CPU 接收请求 -> tokenizer -> PCIe 传输入 -> GPU 读取 HBM -> Tensor Core 计算 -> 输出回 CPU
```

关键术语：`PCIe`、`HBM`、`Tensor Core`、`TTFT`、`TPOT`。

### 单机多卡推理

```text
GPU0/GPU1/... 之间通过 NVLink/NVSwitch 或 PCIe 通信
```

关键术语：`NVLink`、`NVSwitch`、`P2P`、`Topology`、`NCCL`。

如果模型做 tensor parallel，跨 GPU 通信会直接影响吞吐和延迟。

### 多机多卡推理/训练

```text
节点内：GPU 之间走 NVLink/NVSwitch/PCIe
节点间：机器之间走 InfiniBand 或 Ethernet
```

关键术语：`InfiniBand`、`Ethernet`、`NIC`、`HCA`、`mlx5_0`、`NUMA`。

跨机性能通常不只看网卡带宽，还要看 GPU、网卡、CPU socket 之间的拓扑是否合理。

### Decode 阶段长上下文

```text
每生成一个 token -> 读取历史 KV Cache -> attention -> 采样下一个 token
```

关键术语：`HBM Bandwidth`、`Memory-bound`、`KV Cache`、`TPOT`。

如果上下文很长，HBM 读取会成为主要瓶颈。

## 九、面试与工作表达

### 1. HBM 和 PCIe 的区别

可以这样说：

```text
HBM 是 GPU 本地显存，带宽高，用来存模型权重、KV Cache 和中间结果；PCIe 是设备互联总线，负责 CPU、GPU、网卡等设备之间的数据传输。两者不在一个层次，一个是存储，一个是互联。
```

### 2. NVLink 和 InfiniBand 的区别

```text
NVLink 主要用于单机内 GPU-GPU 高速互联；InfiniBand 主要用于跨机器的高性能网络通信。一个偏节点内互联，一个偏节点间互联。
```

### 3. eth0 和 mlx5_0 的区别

```text
eth0 是 Linux 网络接口名，通常面向 IP 网络配置；mlx5_0 是 Mellanox/NVIDIA 网卡在设备/驱动层的名字。它们可能指向同一块物理网卡的不同抽象层次。
```

### 4. Memory-bound 怎么判断

```text
如果算子计算量不大，但大量读写 HBM，性能主要随显存带宽变化，而不是随 Tensor Core 峰值算力变化，就可以认为它偏 memory-bound。Decode 阶段读取 KV Cache 是典型例子。
```

### 5. 为什么拓扑重要

```text
因为 GPU、网卡、CPU socket 的物理连接会影响通信路径。GPU 到网卡如果跨 NUMA 或只能走较慢 PCIe 路径，跨机通信和数据搬运可能变慢。
```

## 十、总结

Infra 术语可以按层次记忆：

| 层次 | 关键词 |
| --- | --- |
| GPU 内部 | `SM`、`Tensor Core`、`HBM`、`L2 Cache`、`Shared Memory` |
| 机内互联 | `PCIe`、`NVLink`、`NVSwitch`、`P2P` |
| 跨机网络 | `InfiniBand`、`Ethernet`、`NIC`、`HCA` |
| Linux/驱动视角 | `eth0`、`ib0`、`mlx5_0`、`mlx5_core` |
| 部署拓扑 | `NUMA`、`CPU Affinity`、`MIG`、`MPS` |
| 服务指标 | `Bandwidth`、`Latency`、`TTFT`、`TPOT`、`P99` |

