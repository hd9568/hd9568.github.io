---
title: 'Ampere、Hopper 与 Blackwell 算子开发差异：从 mma.sync 到 WGMMA 与 tcgen05'
description: '从 CUDA 算子和大模型训推优化视角，对比 Ampere、Hopper、Blackwell SM100 的 MMA、数据搬运、片上存储、Warp Specialization、CTA Cluster、低精度训练以及 GEMM、Attention、MoE 的实现差异。'
category: 'Research & Work'
pubDate: '2026-07-30T10:14:00+08:00'
updatedDate: '2026-07-30T10:14:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [范围与核心结论](#一范围与核心结论)
2. [算子开发视角的总览](#二算子开发视角的总览)
3. [三代架构真正改变的是执行主体](#三三代架构真正改变的是执行主体)
4. [Ampere：Warp MMA 与 cp.async 流水](#四amperewarp-mma-与-cpasync-流水)
5. [Hopper：WGMMA、TMA 与 Warp Specialization](#五hopperwgmma-tma-与-warp-specialization)
6. [Blackwell SM100：tcgen05、TMEM 与 2-CTA MMA](#六blackwell-sm100tcgen05tmem-与-2-cta-mma)
7. [数据搬运路径如何演进](#七数据搬运路径如何演进)
8. [Accumulator 放在哪里决定 Kernel 结构](#八accumulator-放在哪里决定-kernel-结构)
9. [同步模型与异步 Pipeline](#九同步模型与异步-pipeline)
10. [CTA Cluster 与跨 SM 协作](#十cta-cluster-与跨-sm-协作)
11. [Tile Scheduler：Static、Persistent、Stream-K 与 CLC](#十一tile-schedulerstaticpersistentstream-k-与-clc)
12. [GEMM 在三代架构上的实现差异](#十二gemm-在三代架构上的实现差异)
13. [Attention 与 FMHA 的实现差异](#十三attention-与-fmha-的实现差异)
14. [MoE 与 Grouped GEMM 的实现差异](#十四moe-与-grouped-gemm-的实现差异)
15. [LayerNorm、Softmax 等非 GEMM 算子](#十五layernormsoftmax-等非-gemm-算子)
16. [训练精度：BF16/TF32、FP8、MXFP8 与 NVFP4](#十六训练精度bf16tf32fp8mxfp8-与-nvfp4)
17. [推理优化：Prefill 与 Decode 得到的收益不同](#十七推理优化prefill-与-decode-得到的收益不同)
18. [通信与计算融合](#十八通信与计算融合)
19. [从 Ampere Kernel 迁移到 Hopper](#十九从-ampere-kernel-迁移到-hopper)
20. [从 Hopper Kernel 迁移到 Blackwell](#二十从-hopper-kernel-迁移到-blackwell)
21. [编译目标与二进制兼容](#二十一编译目标与二进制兼容)
22. [多架构代码应如何组织](#二十二多架构代码应如何组织)
23. [Profile 时每代架构看什么](#二十三profile-时每代架构看什么)
24. [常见误区](#二十四常见误区)
25. [算子选型清单](#二十五算子选型清单)
26. [参考资料](#二十六参考资料)
27. [总结](#二十七总结)

## 一、范围与核心结论

本文讨论三条数据中心算子开发路径：

```text
Ampere:
  A100，Compute Capability 8.0，SM80

Hopper:
  H100/H200，Compute Capability 9.0，SM90a

Blackwell:
  B100/B200/GB200，Compute Capability 10.0，SM100a
```

这里的 Blackwell 特指数据中心 SM100。

GeForce RTX 50 系列属于 SM120。SM120 与 SM100 在 Tensor Memory、MMA 指令和资源上有明显区别，不能把 B200 的 Kernel 直接套到 RTX 50。

从算子开发角度，三代架构的主线不是“Tensor Core 越来越快”，而是：

```text
Ampere:
  Warp 发起同步 MMA
  所有 Warp 协作搬运和计算
  Accumulator 在寄存器

Hopper:
  Warpgroup 发起异步 WGMMA
  单线程发起 TMA
  Producer/Consumer Warp 专职化
  Accumulator 仍在寄存器

Blackwell SM100:
  单线程发起 tcgen05 MMA
  Accumulator 移入 TMEM
  一个或两个 CTA 共同执行 MMA
  Scheduler、Load、MMA、Epilogue 可进一步解耦
```

所以高性能 Kernel 的迁移通常意味着重写：

- MMA Atom。
- Shared Memory Layout。
- Copy Atom。
- Pipeline。
- Warp/CTA 角色。
- Epilogue。
- Tile Scheduler。
- Dtype 与 Scale Layout。

而不是只修改 `BLOCK_M/BLOCK_N`。

## 二、算子开发视角的总览

| 维度 | Ampere SM80 | Hopper SM90a | Blackwell SM100a |
| --- | --- | --- | --- |
| 原生 MMA | `mma.sync` | `wgmma.mma_async` | `tcgen05.mma`，CUTLASS 常称 UMMA |
| 发起范围 | 1 Warp，32 Threads | 1 Warpgroup，4 Warps/128 Threads | 1 Thread 发起，1 或 2 CTA 协作 |
| MMA 同步性 | Warp 同步 | 异步，Commit/Wait Group | 异步，Barrier/Commit，结果在 TMEM |
| A/B Operand | 通常从寄存器 Fragment | A 可 RMEM/SMEM，B 在 SMEM | A 可 TMEM/SMEM，B 在 SMEM |
| Accumulator | RMEM | RMEM，分布在 Warpgroup | TMEM |
| GMEM→SMEM | `cp.async` | TMA | TMA，也可组合其他 Copy Path |
| Copy 发起 | 多线程协作 | 单线程描述符式发起 | 单线程描述符式发起 |
| 跨 CTA 协作 | 无原生 Cluster | Thread Block Cluster + DSMEM | Cluster + 2-CTA MMA + CLC |
| 典型 Kernel | Multistage CTA GEMM | Warp-specialized Persistent GEMM | TMEM/UMMA Warp-specialized 1SM/2SM GEMM |
| 训练低精度 | TF32、BF16、FP16 | FP8 E4M3/E5M2 | MXFP8、FP6/FP4、NVFP4 Block Scale |
| 主要开发栈 | CUTLASS 2.x、WMMA/PTX | CUTLASS 3.x、CuTe | CUTLASS 3.x/4.x、CuTe DSL |

最重要的三个变化：

1. 计算协作范围从 Warp 扩展到 Warpgroup，再扩展到 CTA/CTA Pair。
2. 数据搬运从线程指令变成描述符驱动的独立硬件工作。
3. Accumulator 从通用寄存器移入 Tensor Core 专用 TMEM。

## 三、三代架构真正改变的是执行主体

### 3.1 Ampere：Warp 是核心计算主体

`mma.sync` 由一个 Warp 的 32 个线程共同执行。

每个 Lane 持有：

- A Fragment 的一部分。
- B Fragment 的一部分。
- C/D Accumulator 的一部分。

所有 Lane 必须以一致的控制流执行 MMA。

### 3.2 Hopper：Warpgroup 是核心计算主体

WGMMA 将协作范围扩大为：

```text
4 Warps = 128 Threads
```

一个典型 FP16/BF16 WGMMA Atom：

```text
m64nNk16

8 <= N <= 256
N 是 8 的倍数
```

更大的硬件 Atom 降低指令调度开销，但也要求更严格的：

- Warpgroup 布局。
- SMEM Descriptor。
- Register Accumulator 分布。
- Commit/Wait。

### 3.3 Blackwell：CTA 或 CTA Pair 是核心计算主体

`tcgen05.mma` 可以使用：

```text
cta_group::1
cta_group::2
```

`cta_group::2` 中，两个相邻 CTA 位于两个 SM，并共同执行一个更大的 MMA。

与 WGMMA 不同：

```text
一个 Elected Thread
即可发起 tcgen05 MMA
```

Accumulator 不再绑定发起 MMA 的 128 个线程，而是保存在 TMEM。MMA 发起线程与后续 Epilogue 线程可以更彻底地分离。

## 四、Ampere：Warp MMA 与 cp.async 流水

Ampere 高性能 GEMM 的基本数据流：

```text
GMEM
  -> cp.async
SMEM
  -> ldmatrix / shared load
RMEM Fragment
  -> mma.sync
RMEM Accumulator
  -> Epilogue
GMEM
```

### 4.1 `mma.sync`

典型 FP16 指令：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {d0, d1, d2, d3},
    {a0, a1, a2, a3},
    {b0, b1},
    {c0, c1, c2, c3};
```

它表示一个 Warp 共同计算：

```text
D[16, 8] =
    A[16, 16]
    * B[16, 8]
    + C[16, 8]
```

但矩阵 Fragment 被分散到 32 个 Lane 的寄存器中。开发者必须遵守 PTX 规定的 Lane-to-Element Layout。

### 4.2 `cp.async`

Ampere 引入：

```text
Global Memory -> Shared Memory
```

的异步直接复制。

与旧路径：

```text
LDG -> Register -> STS
```

相比，`cp.async`：

- 不需要用普通寄存器中转。
- 可绕过 L1。
- 可与当前 Tile 的 MMA 重叠。
- 使用 Commit/Wait Group 管理多级 Pipeline。

### 4.3 Ampere Multistage Pipeline

结构伪代码：

```cpp
// 所有 Warp 通常都参与 Copy 和 Compute。
prefetch(stage=0);

for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
    wait_for(stage);
    __syncthreads();

    prefetch(next_stage);

    load_fragments_from_smem();
    mma_sync(accumulator, a_frag, b_frag);

    commit_copy_group();
    stage = next_stage;
}
```

这里 Copy 是异步的，但 MMA 本身仍是 Warp 同步操作。

### 4.4 Ampere Kernel 的主要约束

Accumulator 占用大量寄存器：

```text
更大的 Warp Tile
-> 更多数据复用
-> 更高寄存器压力
-> 更低 Occupancy
```

Shared Memory Stage 增加：

```text
更多 Pipeline Stage
-> 更好隐藏 HBM 延迟
-> 更高 SMEM 占用
-> 可能减少 Active CTA
```

优化重点：

- Register Tile。
- Shared Memory Bank Conflict。
- `ldmatrix` Layout。
- `cp.async` Stage。
- Warp 数与 Occupancy。
- Output Tile Rasterization。

## 五、Hopper：WGMMA、TMA 与 Warp Specialization

Hopper 数据流：

```text
GMEM
  -> TMA
SMEM
  -> WGMMA Descriptor
WGMMA
  -> RMEM Accumulator
  -> R2S Epilogue
SMEM
  -> TMA Store
GMEM
```

### 5.1 WGMMA

典型 FP16/BF16：

```text
wgmma.mma_async.sync.aligned.m64nNk16
```

特征：

- 4 个 Warp 协作。
- 异步执行。
- B 必须来自 SMEM Descriptor。
- A 可来自 SMEM Descriptor 或 RMEM。
- Accumulator 位于 Warpgroup 的寄存器。

同步序列：

```text
warpgroup.fence
WGMMA
warpgroup.commit_group
warpgroup.wait_group
```

等待完成前：

- 不能读取未完成的 Accumulator。
- 不能覆盖 WGMMA 仍在读取的 SMEM Buffer。

### 5.2 TMA

TMA 是描述符驱动的异步 Tensor Copy Engine。

它可搬运：

```text
1D 到 5D Tensor
GMEM -> SMEM
SMEM -> GMEM
SMEM -> Cluster DSMEM
```

硬件负责：

- 多维地址计算。
- Shape/Stride。
- 边界处理。
- Bulk Transfer。
- 部分 Reduce Store。

一个线程即可发起大块传输，其他 Warp 不必执行大量地址计算和 Load/Store。

### 5.3 TMA 不等于 cp.async 的简单升级

`cp.async` 更接近：

```text
每个线程描述自己负责的 4/8/16B Copy
```

TMA 更接近：

```text
一个线程提交 Tensor Descriptor + Coordinate
硬件搬完整 Tile
```

因此 TMA 特别适合：

- 规则 GEMM Tile。
- 多维 Tensor。
- Cluster Multicast。
- Warp Specialized Producer。

而对于不规则、很小、动态地址过多的 Copy，普通 Load 或 `cp.async` 仍可能更合适。

### 5.4 Warp Specialization

Hopper 常见角色：

```text
Producer Warp:
  发起 TMA

Consumer Warpgroup:
  发起 WGMMA

Epilogue Warp:
  转换并写回

Scheduler Warp:
  获取下一个 Tile
```

通过 `mbarrier` 构造 Producer/Consumer Pipeline：

```text
Producer:
  等 Empty
  发起 TMA
  TMA 到达后标记 Full

Consumer:
  等 Full
  发起 WGMMA
  等 MMA 完成
  释放 Stage
```

与 Ampere “所有 Warp 都做所有事”相比，Hopper 把线程角色固定下来，让 TMA 与 Tensor Core 更稳定地并行。

### 5.5 Persistent Ping-Pong

CUTLASS Hopper Kernel 常使用：

- Persistent Cooperative。
- Persistent Ping-Pong。

Ping-Pong 中两个 Consumer Warpgroup 处理不同输出 Tile：

```text
Consumer 0:
  Mainloop Tile A

Consumer 1:
  Epilogue Tile B
```

然后交换角色，使：

```text
一个 Tile 的 Epilogue
与另一个 Tile 的 MMA
```

重叠。

## 六、Blackwell SM100：tcgen05、TMEM 与 2-CTA MMA

Blackwell SM100 数据流：

```text
GMEM
  -> TMA
SMEM Operand A/B
  -> tcgen05.mma
TMEM Accumulator
  -> tcgen05.ld
RMEM Epilogue Fragment
  -> Store / TMA
GMEM
```

### 6.1 `tcgen05.mma`

CUTLASS 将第五代 Tensor Core MMA 常称为：

```text
UMMA
```

它支持：

- TF32。
- FP16/BF16。
- INT8。
- FP8/FP6/FP4 Mixed Input。
- MXFP8/MXFP6/MXFP4。
- NVFP4。
- Dense/Sparse。
- 1-CTA/2-CTA。

### 6.2 TMEM

TMEM 是 Tensor Core 专用片上存储。

对 `tcgen05.mma`：

```text
A:
  SMEM 或 TMEM

B:
  SMEM

Accumulator C/D:
  必须在 TMEM
```

这带来两个重要变化。

第一，Accumulator 不再占据普通寄存器：

```text
更大 MMA Tile
不再等比例增加发起 Warp 的 RMEM Accumulator
```

第二，MMA 发起和输出处理解耦：

```text
一个线程发起 MMA
-> MMA 写 TMEM
-> Epilogue Warp 从 TMEM 读取
```

### 6.3 TMEM 不是自动可用的缓存

Kernel 需要管理：

- TMEM Allocation。
- TMEM Address/Layout。
- MMA 写入。
- `tcgen05.ld` 读回。
- Barrier。
- 生命周期和释放。

Epilogue 也发生变化：

```text
Hopper:
  RMEM Acc -> SMEM/GMEM

Blackwell:
  TMEM Acc -> RMEM -> GMEM
```

如果 Epilogue 很复杂，TMEM→RMEM Copy、Layout Reorder 和类型转换可能成为新的瓶颈。

### 6.4 2-CTA MMA

Blackwell 支持两个 CTA 跨两个 SM 共同执行 MMA。

典型 Cluster：

```text
cluster_shape = (2, 1, 1)
```

两个 CTA 分担：

- A Tile。
- B Tile。
- Accumulator。
- Epilogue 输出。

由 Leader CTA 的一个线程发起 MMA。

价值：

- 更大的硬件 Tile。
- 更高数据复用。
- 两个 SM 协作。

代价：

- SM 必须成对调度。
- Cluster Occupancy 更复杂。
- Grid 尾部必须以 Cluster 粒度考虑。
- Barrier/TMA Multicast/Peer SMEM 配置更复杂。
- 小 Shape 可能浪费一半 Tile。

### 6.5 1SM 和 2SM 不是固定优先级

SM100 CUTLASS 同时提供：

```text
KernelTmaWarpSpecialized1SmSm100
KernelTmaWarpSpecialized2SmSm100
```

应根据：

- M/N/K。
- Dtype。
- Layout。
- Cluster Occupancy。
- Wave Quantization。
- Epilogue。

选择，而不是看到 B200 就默认 2SM。

## 七、数据搬运路径如何演进

### 7.1 Ampere

```text
多个线程计算地址
-> 每线程 cp.async 一小段
-> SMEM
```

优点：

- 灵活。
- 适合不规则 Predication。
- Copy 粒度由线程精确控制。

代价：

- 多线程参与 Address Generation。
- Copy 指令占 Warp Issue。
- 跨 CTA 无 Multicast。

### 7.2 Hopper

```text
Host/Device 创建 Tensor Map
-> 一个线程发起 TMA
-> Hardware 处理多维 Copy
-> mbarrier 通知到达
```

优点：

- 大幅减少地址指令和寄存器。
- 易做多 Stage。
- 支持 Cluster Multicast。
- 支持 SMEM→GMEM Store。

代价：

- Tensor Descriptor Setup。
- Layout/Alignment 限制。
- 很小 Tile 的启动成本。
- 动态 Shape 可能要更新 Descriptor。

### 7.3 Blackwell

Blackwell 继续使用 TMA，但数据路径要考虑：

- 1CTA/2CTA Partition。
- Preferred Cluster。
- Scale Factor Tensor。
- TMEM Operand/Accumulator。
- CLC Scheduler。

对于 Block-scaled GEMM，除了 A/B，还要供应：

```text
SFA
SFB
```

Scale 数据流成为 Mainloop 的一部分，而不是 Epilogue 外面的普通乘法。

## 八、Accumulator 放在哪里决定 Kernel 结构

### Ampere

```text
Accumulator:
  每个 Warp Lane 的寄存器
```

后果：

- Warp Tile 越大，Register 越多。
- Fused Epilogue 可以直接使用寄存器结果。
- Register Spilling 会直接毁掉性能。

### Hopper

```text
Accumulator:
  Warpgroup 的寄存器
```

后果：

- WGMMA Tile 更大。
- Consumer Warpgroup 需要大量寄存器。
- Producer Warp 可主动减少 Register 配额。
- Epilogue 前必须等待 WGMMA Group。

### Blackwell

```text
Accumulator:
  TMEM
```

后果：

- MMA 发起线程不需要持有完整 Accumulator。
- 更容易让 Scheduler/Load/MMA/Epilogue Warp 分工。
- Epilogue 要显式从 TMEM Load。
- TMEM Layout 成为算子正确性的一部分。
- TMEM Capacity 与 Column 分配限制 Kernel 组合。

## 九、同步模型与异步 Pipeline

### Ampere

主要对象：

```text
cp.async.commit_group
cp.async.wait_group
__syncthreads
Arrive/Wait Barrier
```

Copy 异步，MMA 同步。

### Hopper

主要对象：

```text
mbarrier
TMA Transaction Arrival
wgmma.fence
wgmma.commit_group
wgmma.wait_group
Cluster Sync
```

Copy 和 MMA 都异步。

### Blackwell

增加：

```text
tcgen05 MMA/Commit
TMEM Allocation Barrier
TMEM Load/Store Synchronization
2-CTA Cluster Barrier
CLC Async Query Pipeline
```

算子正确性不再只是：

```text
“数据是否写进 SMEM”
```

还包括：

- Async Proxy 顺序。
- TMEM 可见性。
- CTA Pair 同步。
- Scheduler Work ID 一致性。
- Epilogue 何时能读取 Accumulator。

## 十、CTA Cluster 与跨 SM 协作

Ampere 的 CUDA Block 原则上独立，跨 Block 协作通常通过：

- Global Memory。
- Atomic。
- 多 Kernel。
- Cooperative Launch。

Hopper 加入 Thread Block Cluster：

```text
Grid
  -> Cluster
      -> CTA
          -> Warp
```

Cluster 内 CTA：

- 保证协同驻留。
- 可 `cluster.sync()`。
- 可访问 Peer CTA 的 Distributed Shared Memory。
- 可执行 DSMEM Atomic。
- 可使用 TMA Multicast。

Blackwell 在此基础上让：

```text
CTA Pair
```

成为 Tensor Core MMA 的硬件协作单位。

### 10.1 Cluster 对算子有什么用

GEMM：

- A/B Tile Multicast。
- 2SM MMA。
- 更大 Tile。

Attention：

- Multi-CTA KV Partial 归并。
- Cluster Shared Reduction。

MoE：

- Expert Tile 协作。
- 动态负载分配。

Reduction：

- 把一部分 Global Workspace 归并搬到 DSMEM。

### 10.2 Cluster 并非免费

Cluster Size 增大：

- 可用 Cluster 数减少。
- 尾部量化更严重。
- 资源必须同时满足多个 SM。
- 一个 CTA 慢会拖住 Cluster。

应使用：

```text
cudaOccupancyMaxActiveClusters
```

而不是只用传统 Block Occupancy 推断。

## 十一、Tile Scheduler：Static、Persistent、Stream-K 与 CLC

### 11.1 Ampere

常见：

- 普通 Grid Tile。
- Swizzle。
- Grouped GEMM Problem Visitor。
- Split-K。

### 11.2 Hopper

Warp-specialized GEMM 常使用 Persistent CTA：

```text
启动接近 SM 数的 Worker
每个 Worker 处理多个 Output Tile
```

优势：

- 摊薄 Prologue。
- 可重用 Pipeline。
- 可重叠连续 Tile 的 Mainloop/Epilogue。

风险：

- Static 分工遇到不规则 Grouped GEMM 会负载不均。
- 当前可用 SM 数可能受并发 Kernel 影响。

### 11.3 Stream-K

Stream-K 按全局 MAC Iteration 分配工作，而不是每个 Output Tile 固定一个 CTA。

它缓解：

- 小 M/N。
- Wave Quantization。
- 不规则 Tile 尾部。

### 11.4 Blackwell CLC

Cluster Launch Control 允许已运行 Worker：

```text
取消一个尚未启动的 Cluster ID
并接管其工作坐标
```

简化理解：

```text
Grid 中声明所有 Tile
-> 硬件启动首批 Worker
-> Worker 完成后从未启动 Grid 中“偷”下一个 Tile
```

相对 Software Atomic Queue：

- 不需要 Global Counter。
- 调度发生在 Cluster Launch Controller。
- 可以适应部分 SM 被其他 Kernel 占用。
- 以 Cluster 为粒度。

CUTLASS Blackwell Kernel 甚至把 Warp 明确分为：

```text
MMA
Scheduler
Mainloop Load
Epilogue Load
Epilogue
```

Scheduler 已经成为 Kernel 内部独立硬件流水。

## 十二、GEMM 在三代架构上的实现差异

### 12.1 Ampere GEMM

典型配置：

```text
CTA Tile:
  128x128x32/64

Mainloop:
  cp.async Multistage

MMA:
  mma.sync m16n8k16

Accumulator:
  RMEM
```

高性能关键：

- `ldmatrix` 与 SMEM Swizzle。
- Register Tile。
- Stage 数。
- Warp Tile。
- Split-K。

### 12.2 Hopper GEMM

典型配置：

```text
CTA Tile:
  128x128/256xK

Mainloop:
  TMA Producer

MMA:
  WGMMA m64nNk16/32

Accumulator:
  Warpgroup RMEM

Kernel:
  Persistent Ping-Pong/Cooperative
```

高性能关键：

- Tensor Map。
- TMA Stage。
- Warpgroup Register Budget。
- WGMMA Commit/Wait 深度。
- Cluster Shape。
- Mainloop/Epilogue Overlap。

### 12.3 Blackwell GEMM

典型配置：

```text
MMA Tile:
  1SM 或 2SM
  可覆盖 128/256 级 M/N Tile

Mainloop:
  TMA + Scale Factor Pipeline

MMA:
  tcgen05

Accumulator:
  TMEM

Scheduler:
  Static Persistent / Stream-K / CLC
```

高性能关键：

- 1SM vs 2SM。
- TMEM Column/Layout。
- TMEM Double Buffer。
- Scale Factor Layout。
- TMA Partition/Multicast。
- CLC。
- TMEM→RMEM Epilogue。

### 12.4 同一 Tile 配置不能照搬

Hopper 的：

```text
128x256 WGMMA Tile
```

迁移到 Blackwell 后需要重新决定：

- tcgen05 Instruction Shape。
- CTA Group。
- TMEM Layout。
- Epilogue Warp 数。
- Cluster Shape。

Tile 数值相同也不代表数据流相同。

## 十三、Attention 与 FMHA 的实现差异

Attention：

```text
S = QK^T * scale
P = softmax(S + mask)
O = PV
```

它不是单纯两个 GEMM，还包含：

- Online Softmax。
- Mask。
- Rescale。
- Reduction。
- Paged KV。
- RoPE。

### 13.1 Ampere：FlashAttention-1/2

常见结构：

```text
cp.async Q/K/V
-> SMEM
-> mma.sync QK
-> FP32 Softmax in Registers
-> mma.sync PV
-> Output
```

主要难点：

- Score/Output Register Pressure。
- SMEM Bank Conflict。
- Warp 间 Softmax Reduction。
- QK 与 PV Layout 转换。

### 13.2 Hopper：FlashAttention-3

FA3 使用：

- TMA。
- WGMMA。
- Warp Specialization。
- Ping-Pong。
- Softmax 与 WGMMA Overlap。
- FP8 Forward。

关键原因是 Hopper Tensor Core 相对普通 FP32 CUDA Core 更快，Softmax、Mask 和 Rescale 更容易暴露为瓶颈。

所以仅把 FA2 的 `mma.sync` 编译到 H100，并不能获得 WGMMA 路径的完整吞吐。

### 13.3 Blackwell：TMEM FMHA

Blackwell FMHA 可把：

- QK Accumulator。
- Score。
- P。
- PV Accumulator。
- Softmax Stats。

按 Kernel Traits 分配到 TMEM/SMEM。

可能的数据流：

```text
TMA K/V
-> tcgen05 QK
-> TMEM Score
-> TMEM/RMEM Softmax
-> tcgen05 PV
-> TMEM Output
-> Epilogue
```

还可结合：

- FP8/NVFP4 KV。
- Multi-CTA KV。
- 2-CTA MMA。
- Cluster Reduction。
- Persistent/CLC Scheduler。

### 13.4 Decode 不一定吃满新 Tensor Core

Decode：

```text
Q_len = 1
```

通常受：

- KV Cache 带宽。
- Page Table。
- Batch/Head 并行度。
- Kernel Launch。

限制。

B200 的 tcgen05 峰值再高，如果只有少量 Query，也可能无法使用大 MMA Tile。

因此 Decode 仍需要：

- Split-KV。
- GQA-aware KV Reuse。
- Small-M GEMM/GEMV。
- CUDA Graph。
- KV Quantization。
- Paged Attention。

## 十四、MoE 与 Grouped GEMM 的实现差异

MoE：

```text
Router
-> TopK
-> Token Permute
-> Grouped GEMM 1
-> Activation
-> Grouped GEMM 2
-> Unpermute/Combine
```

### Ampere

- FP16/BF16 Grouped GEMM。
- Device Problem Visitor。
- `mma.sync`。
- `cp.async`。
- Expert Token 数不均时用 Persistent/Split-K。

### Hopper

- FP8 Expert GEMM。
- TMA/WGMMA。
- Warp-specialized Grouped GEMM。
- Persistent Problem Scheduler。
- Routing/Permute 与 GEMM Fusion。

### Blackwell

- MXFP8/MXFP4/NVFP4 Block-scaled Grouped GEMM。
- Scale Matrix 作为显式 Operand。
- TMEM Accumulator。
- 1SM/2SM。
- CLC 动态取 Expert Tile。
- Activation/Quant/Finalize Fusion。

### 14.1 Blackwell 的新瓶颈

低精度 GEMM 更快后，以下步骤占比上升：

- TopK Router。
- Prefix Sum。
- Token Scatter/Gather。
- TMA Descriptor Setup。
- Scale Factor 生成。
- Expert Load Imbalance。
- Finalize。

因此“把 Expert Weight 换成 FP4”不是完整 MoE 优化。需要把非 GEMM 阶段一起融合或并行。

## 十五、LayerNorm、Softmax 等非 GEMM 算子

这些算子通常不直接使用 Tensor Core。

### LayerNorm/RMSNorm

主要操作：

- Reduction。
- `rsqrt`。
- Vector Load/Store。
- Elementwise Scale。

### Softmax

主要操作：

- Max Reduction。
- Exp。
- Sum Reduction。
- Normalize。

### 架构升级的真实影响

Ampere→Hopper→Blackwell：

```text
Tensor Core 提升很快
普通 FP32/SFU/Reduction 提升没有同倍增长
```

于是非 GEMM 算子更容易成为 Amdahl 瓶颈。

优化方向：

- Vectorization。
- Warp/Block Reduction。
- Online Algorithm。
- Persistent Kernel。
- 与 GEMM Epilogue 融合。
- RMSNorm + Quantization。
- Bias + Activation + Scale。

Blackwell 上尤其要重视：

```text
Norm/Activation 输出
-> 直接生成 FP4/FP8 Payload + Scale
-> 供下一个 Block-scaled GEMM 使用
```

否则单独 Quant Kernel 会抵消低精度 Tensor Core 收益。

## 十六、训练精度：BF16/TF32、FP8、MXFP8 与 NVFP4

### 16.1 Ampere

训练主线：

- BF16。
- FP16 + Loss Scaling。
- TF32。
- FP32 Master State。

TF32 让 FP32 GEMM 更容易进入 Tensor Core，但并不等价于完整 FP32 Mantissa。

### 16.2 Hopper FP8

Hopper 支持：

```text
E4M3:
  更高精度，较小动态范围

E5M2:
  更大动态范围，较低精度
```

典型训练：

```text
Forward Weight/Activation:
  E4M3

Backward Gradient:
  E5M2
```

FP8 需要 Scale：

```text
x_fp8 = quantize(x / scale)
```

策略包括：

- Delayed Scaling。
- Current Scaling。
- Amax History。

因此 Hopper FP8 Linear 不只是调用 FP8 GEMM，还包括：

- Amax。
- Scale 更新。
- Cast/Transpose。
- FP8 GEMM。
- 高精度输出/累加。

### 16.3 Blackwell MXFP8

普通 FP8 常用每 Tensor Scale。

MXFP8：

```text
每 32 个连续值一个 E8M0 Scale
Payload 可统一使用 E4M3
```

对 GEMM：

```text
D_ij =
  sum_k
  (A_ik * SFA_{i,k/32})
  (B_jk * SFB_{j,k/32})
```

Scale 沿 Reduction K 维组织。

### 16.4 MXFP8 Transpose 不是 View

训练 Linear 有三类 GEMM：

```text
Forward:
  Y = X W^T

Input Gradient:
  dX = dY W

Weight Gradient:
  dW = dY^T X
```

不同 GEMM 的 Reduction 轴不同。

MXFP8 要求 Scale Block 沿当前 Reduction 轴连续，所以简单改变 Stride 不能得到正确 Transposed Quantized Tensor。

Transformer Engine 会从高精度输入分别生成：

```text
Non-transposed Quantized Copy
Transposed Quantized Copy
```

而不是把已量化 MXFP8 再转置两次。

### 16.5 Blackwell NVFP4

NVFP4：

- Payload 为 E2M1。
- 每 16 个值一个 FP8 E4M3 Block Scale。
- 另有 Per-tensor FP32 Scale。

训练 Recipe 还可能需要：

- Gradient Stochastic Rounding。
- Weight 16x16 2D Scaling。
- Random Hadamard Transform。
- 敏感 Layer 保持高精度。

硬件支持 FP4 MMA，不代表所有训练 Tensor 都能直接降为 FP4。

### 16.6 Block Scale 是算子接口的一部分

Blackwell GEMM 输入不再只是：

```text
A, B
```

而是：

```text
A_payload
B_payload
SFA
SFB
Tensor Scale
Layout Metadata
```

Kernel 的 Alignment、TMA Layout、Scale Vector Size 和 K Partition 必须一致。

## 十七、推理优化：Prefill 与 Decode 得到的收益不同

### 17.1 Prefill

Prefill 有较大 Query Length：

- GEMM 更大。
- Attention QK/PV 更接近矩阵计算。
- 容易使用 Tensor Core。

代际收益更直接：

```text
Ampere BF16/FP16
-> Hopper FP8 WGMMA/TMA
-> Blackwell FP8/FP4 tcgen05/TMEM
```

### 17.2 Decode

Decode Linear：

```text
M = batch * tokens_per_request
```

可能很小。

当 M 太小：

- 大 MMA Tile 浪费。
- 2SM Tile 更难填满。
- GEMV/CUDA Core Kernel 可能优于 GEMM。
- Weight Load 主导。

Decode Attention：

- KV Load 主导。
- Tensor Core 只占一部分。
- FP4 KV 降低带宽可能比 FP4 QK 吞吐更关键。

### 17.3 选择策略

同一模型应按 Runtime Shape Dispatch：

```text
Large Prefill:
  Tensor Core FMHA/GEMM

Small Decode Batch:
  GEMV/Small-M GEMM

Long Context:
  Split-KV

Large GQA Ratio:
  Grouped Query Tensor Core Decode
```

不能只按 GPU 架构选择一个永久 Kernel。

## 十八、通信与计算融合

训练中的 Tensor Parallel 常出现：

```text
GEMM
-> AllReduce/ReduceScatter/AllGather
```

### Ampere

常见方式：

- 独立 NCCL Kernel。
- Multi-stream Overlap。
- GEMM Chunking。

### Hopper

Hopper NVSwitch 支持 SHARP In-network Reduction，软件栈可使用 NVLS/Multimem 类路径。

这使算子可以探索：

- GEMM Epilogue 触发 Reduce。
- Symmetric Memory。
- Fused GEMM + AllReduce。
- Warp-specialized Communication Agent。

### Blackwell

Blackwell 延续并扩展：

- 更大的 Scale-up Domain。
- SM100 GEMM/AllReduce 专用 Kernel。
- TMA Warp Specialization。
- Programmatic Dependent Launch。

但融合是否有效仍取决于：

- GEMM Tile 产生数据的顺序。
- Collective Chunk。
- Remote Memory Alignment。
- Rank 数。
- 计算/通信比例。

通信能力增强不代表应把整个 Collective 粗暴塞入一个 Kernel。

## 十九、从 Ampere Kernel 迁移到 Hopper

### 19.1 最低成本路径

保留：

- `mma.sync`。
- `cp.async`。
- 原有 Tile。

重新编译 SM90。

优点：

- 快速正确运行。

缺点：

- 通常无法达到 Hopper WGMMA/TMA 路径性能。

### 19.2 原生 Hopper 路径

需要重构：

```text
mma.sync
-> WGMMA

cp.async
-> TMA

All-warps-same-role
-> Producer/Consumer Warpgroup

Block Grid
-> Persistent/Cluster Scheduler
```

### 19.3 Epilogue

WGMMA Accumulator 在 Warpgroup Register 中，需设计：

- R2S Layout。
- TMA Store。
- Bias/Activation。
- Quantization。
- Ping-Pong Overlap。

### 19.4 Attention 的额外工作

还要重排：

- WGMMA Score Layout。
- Softmax Lane Mapping。
- P→V Operand Layout。
- FP8 In-kernel Transpose。

这就是 FA3 不是 FA2 简单替换指令的原因。

## 二十、从 Hopper Kernel 迁移到 Blackwell

### 20.1 WGMMA 不能原样作为 SM100 原生路径

Hopper `sm_90a/compute_90a` 使用架构条件特性，不向 Blackwell 前向兼容。

Blackwell 原生路径需要：

```text
WGMMA
-> tcgen05
```

### 20.2 Accumulator 迁移

```text
Hopper:
  RMEM Accumulator

Blackwell:
  TMEM Accumulator
```

要重写：

- Fragment Creation。
- TMEM Allocation。
- MMA Partition。
- TMEM Load。
- Epilogue Layout。

### 20.3 Warp 角色迁移

Hopper Consumer Warpgroup：

```text
128 Threads 共同发起 WGMMA
```

Blackwell：

```text
一个 Thread 发起 tcgen05
其他 Warp 可专注 Scheduler/Load/Epilogue
```

这不是简单减少线程数，而是整个角色表需要重新设计。

### 20.4 1SM/2SM

新增决策：

```text
CtaGroup.ONE
CtaGroup.TWO
```

2SM 还需处理：

- CTA Pair Mapping。
- Cluster Barrier。
- TMA Multicast。
- 每 CTA 的 Tile Half。
- TMEM Partition。

### 20.5 Block Scale

若启用 FP4/MXFP：

- 新增 Scale Tensor。
- 重排 K Layout。
- 对齐 Scale Vector。
- 修改 Quantization 前处理。
- 修改 MMA Atom。
- 检查 Transpose Quantization。

## 二十一、编译目标与二进制兼容

### 21.1 Cubin

Cubin 通常只在同一 Compute Capability Major 内兼容。

```text
sm_80 Cubin
不能直接当 sm_90 Cubin 使用
```

### 21.2 PTX

普通 PTX 可向更高 Compute Capability JIT。

例如：

```text
compute_80 PTX
可在 Hopper/Blackwell JIT
```

但它不能自动获得源码中不存在的：

- WGMMA。
- TMA Pipeline。
- tcgen05。
- TMEM。
- CLC。

“能运行”不等于“使用新架构”。

### 21.3 Architecture-conditional Target

Hopper：

```text
sm_90a / compute_90a
```

Blackwell：

```text
sm_100a / compute_100a
```

这些 Target 使用 Architecture-specific Feature，不前向或后向兼容。

### 21.4 CUTLASS 构建

```bash
cmake .. -DCUTLASS_NVCC_ARCHS="80;90a;100a"
```

生产库通常为三代架构分别生成 Kernel，并由 Runtime Dispatch。

### 21.5 SM100 与 SM120

```text
SM100:
  数据中心 Blackwell，TMEM/tcgen05 路径

SM120:
  GeForce Blackwell，独立 Compute Capability
```

`sm_100a` Kernel 不兼容 SM120。

## 二十二、多架构代码应如何组织

不要在一个巨型 Kernel 内堆满：

```cpp
#if __CUDA_ARCH__ >= ...
```

更清晰的方式：

```text
Common:
  Shape/Dtype/Problem Description
  Reference
  High-level Epilogue Semantics

SM80:
  MMA Atom
  cp.async Mainloop
  Ampere Epilogue

SM90:
  WGMMA Atom
  TMA Collective
  Hopper Scheduler/Epilogue

SM100:
  tcgen05 Atom
  TMEM/TMA Collective
  1SM/2SM Scheduler/Epilogue
```

Runtime：

```cpp
const int cc =
    device_prop.major * 10 + device_prop.minor;

switch (cc) {
case 80:  // A100
    return run_sm80(args);
case 90:  // H100/H200
    return run_sm90a(args);
case 100: // B100/B200
    return run_sm100a(args);
case 120: // GeForce Blackwell，需要独立实现
    return run_sm120(args);
default:
    return run_fallback(args);
}
```

实际还要区分：

- SM86/SM89。
- SM103。
- SM120。
- Dtype。
- Shape。
- Feature Support。

### 22.1 使用抽象，不隐藏硬件

CUTLASS/CuTe 的价值是把：

- Layout。
- MMA Atom。
- Copy Atom。
- Collective。
- Scheduler。

组合起来。

但高性能开发仍需知道：

- Accumulator 在 RMEM 还是 TMEM。
- Copy 是 cp.async 还是 TMA。
- MMA 是 Warp、Warpgroup 还是 CTA Pair。
- Scale Factor 如何布局。

抽象应消除重复代码，不应掩盖硬件约束。

## 二十三、Profile 时每代架构看什么

### Ampere

检查：

- 是否生成 `mma.sync`。
- 是否生成 `LDGSTS/cp.async`。
- Shared Memory Bank Conflict。
- Register Spilling。
- Async Copy Wait。
- Active CTA/Occupancy。
- Tensor Core Pipe 利用率。

### Hopper

检查：

- 是否真正使用 WGMMA，而非旧 `mma.sync`。
- TMA Transaction。
- WGMMA Commit/Wait Stall。
- Producer 是否饿死 Consumer。
- Consumer Register Pressure。
- `mbarrier` 等待。
- Cluster Occupancy。
- Mainloop/Epilogue 是否重叠。

### Blackwell

检查：

- `tcgen05` 利用率。
- TMEM Allocation/Load。
- TMA 与 Scale Factor 流量。
- 1SM/2SM 实际选择。
- CTA Pair 是否负载均衡。
- Cluster Occupancy。
- CLC Query 是否有效隐藏调度。
- TMEM→RMEM Epilogue 是否成为瓶颈。
- FP4 Quant/Dequant 是否占比过高。

### 工具

```text
Nsight Systems:
  Kernel Timeline、Overlap、Launch、Communication

Nsight Compute:
  单 Kernel Pipeline、Memory、Tensor Core、Stall

cuobjdump/nvdisasm:
  确认实际指令

CUTLASS Profiler:
  比较 Tile/Scheduler/Dtype
```

## 二十四、常见误区

### 误区 1：H100 跑 Ampere Kernel 就能获得 Hopper 峰值

错误。

`mma.sync + cp.async` 可以运行，但 WGMMA/TMA 才是 Hopper 原生高吞吐路径。

### 误区 2：B200 只需把 WGMMA 换成 tcgen05

错误。

Accumulator 从 RMEM 移到 TMEM，Epilogue、同步和 Warp 角色都要改变。

### 误区 3：Blackwell 的 FP4 只是把指针类型改成 4-bit

错误。

需要 Payload、Scale Tensor、Block Layout、Alignment 和 Quantization Recipe。

### 误区 4：Tensor Core 更快，所有算子都同比加速

错误。

Softmax、Norm、TopK、Sampling、Routing、Page Table 可能成为更大瓶颈。

### 误区 5：2SM MMA 一定比 1SM 快

错误。

小 Shape、尾部 Tile、Cluster Occupancy 可能让 2SM 更慢。

### 误区 6：Blackwell 都是 SM100

错误。

SM100 数据中心和 SM120 GeForce 的 Native Kernel 路径不同。

### 误区 7：PTX JIT 能自动把 Ampere 算法升级成 Blackwell 算法

错误。

JIT 可以重新生成目标机器码，但不会发明新的 TMEM/2CTA Pipeline。

## 二十五、算子选型清单

### GEMM

- Shape 是 Large-M 还是 Small-M？
- Dtype 是 BF16/FP8/FP4？
- 是否需要 Block Scale？
- 1SM 还是 2SM？
- 是否需要 Split-K/Stream-K？
- Epilogue 能否融合？

### Attention

- Prefill 还是 Decode？
- MHA/GQA/MLA？
- KV Dtype？
- Paged KV？
- 是否需要 Split-KV？
- Softmax 与 MMA 能否重叠？
- TMEM/SMEM 是否够放 Score、P、O？

### MoE

- Active Expert 数。
- 每 Expert Token 分布。
- Grouped GEMM Scheduler。
- Scale Layout。
- Routing/Permute 是否主导。
- CLC/Persistent 是否有价值。

### Training

- Forward/dX/dW 三个 GEMM 的 Quantized Layout。
- Transposed Copy 是否需要重新量化。
- Amax/Scale 的更新方式。
- Gradient 是否用 Stochastic Rounding。
- 敏感 Layer 是否保留高精度。
- Collective 能否与 GEMM 重叠。

### Multi-architecture

- 是否分别构建 SM80/SM90a/SM100a？
- 是否保留 Generic PTX Fallback？
- 是否明确排除 SM120？
- 每个 Backend 是否独立验证数值与性能？

## 二十六、参考资料

1. [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
2. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
3. [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
4. [Hopper Compatibility Guide](https://docs.nvidia.com/cuda/hopper-compatibility-guide/index.html)
5. [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)
6. [CUTLASS Warpgroup MMA Programming Guide](https://docs.nvidia.com/cutlass/4.5.1/media/docs/pythonDSL/mma_docs/wgmma_programming.html)
7. [CUTLASS tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/4.5.3/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
8. [CUTLASS Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
9. [CUTLASS Blackwell Cluster Launch Control](https://docs.nvidia.com/cutlass/4.5.0/media/docs/cpp/blackwell_cluster_launch_control.html)
10. [NVIDIA Transformer Engine FP8/FP4 Primer](https://github.com/NVIDIA/TransformerEngine/blob/stable/docs/examples/fp8_primer.ipynb)
11. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low Precision](https://www.nvidia.com/en-us/on-demand/session/gtc25-S71368/)
12. [Programming Blackwell Tensor Cores with CUTLASS](https://www.nvidia.com/en-us/on-demand/session/gtc25-S72720/)
13. [CUTLASS 3.x Kernel Design](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/)

## 二十七、总结

从算子开发角度看，三代架构的演进可以概括为：

```text
Ampere:
  Warp-centric
  mma.sync
  cp.async
  RMEM Accumulator

Hopper:
  Warpgroup-centric
  WGMMA
  TMA
  Warp Specialization
  Cluster/DSMEM
  RMEM Accumulator

Blackwell SM100:
  CTA/CTA-pair-centric
  tcgen05/UMMA
  TMA + TMEM
  Single-thread MMA Launch
  1SM/2SM
  Block-scaled FP8/FP6/FP4
  CLC Scheduler
```

真正影响训推优化工作的区别：

1. 谁发起 MMA。
2. Operand 从哪里来。
3. Accumulator 存在哪里。
4. Copy 与 Compute 如何异步。
5. CTA 是否能跨 SM 协作。
6. Scheduler 如何分配动态 Tile。
7. 低精度 Scale 是否成为硬件 Operand。
8. Epilogue、Quant、Softmax 和通信是否成为新瓶颈。

因此：

```text
Ampere Kernel
-> Hopper
不是简单重编译；

Hopper Kernel
-> Blackwell
也不是简单替换 MMA 指令。
```

兼容路径可以让旧 Kernel 运行，原生路径则需要按新执行模型重新设计数据流、同步和调度。
