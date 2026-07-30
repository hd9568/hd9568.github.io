---
title: 'CUDA 高频面试题 340 问：从硬件数值到 Blackwell 算子优化'
description: '按问题与答案整理 340 道 CUDA 高频面试题，覆盖线程模型、内存体系、同步并发、性能分析、硬件资源计算、GEMM、Tensor Core、Ampere/Hopper/Blackwell 与大模型算子。'
category: 'CUDA'
pubDate: '2026-07-31T01:40:23+08:00'
updatedDate: '2026-07-31T01:51:19+08:00'
---

## 目录

1. [基础语法与编译模型：Q1-Q25](#一基础语法与编译模型q1-q25)
2. [线程、Warp、CTA 与 SM：Q26-Q55](#二线程warpcta-与-smq26-q55)
3. [CUDA 内存体系：Q56-Q85](#三cuda-内存体系q56-q85)
4. [合并访问、Shared Memory 与 Cache：Q86-Q115](#四合并访问shared-memory-与-cacheq86-q115)
5. [同步、原子操作与内存模型：Q116-Q145](#五同步原子操作与内存模型q116-q145)
6. [Stream、Event、异步拷贝与 CUDA Graph：Q146-Q175](#六streamevent异步拷贝与-cuda-graphq146-q175)
7. [性能模型、Occupancy 与 Profiling：Q176-Q210](#七性能模型occupancy-与-profilingq176-q210)
8. [Reduction、Scan、Softmax 等手写题：Q211-Q235](#八reductionscansoftmax-等手写题q211-q235)
9. [GEMM 与 Tensor Core：Q236-Q260](#九gemm-与-tensor-coreq236-q260)
10. [Ampere、Hopper 与 Blackwell：Q261-Q280](#十amperehopper-与-blackwellq261-q280)
11. [大模型算子与灵活推理题：Q281-Q300](#十一大模型算子与灵活推理题q281-q300)
12. [硬件数值与资源计算：Q301-Q340](#十二硬件数值与资源计算q301-q340)
13. [参考资料](#十三参考资料)

## 使用方式

本文不是按篇幅展开概念，而是模拟面试中的连续追问。答案先给结论，再用一两句话解释原因；涉及寄存器数、Shared Memory 容量和最大驻留数量时，以目标 GPU 的 Compute Capability 与 `cudaDeviceProp` 为准，不把某一代硬件参数当作 CUDA 永久规则。

## 一、基础语法与编译模型：Q1-Q25

### Q1：CUDA 是什么？

**答：** CUDA 是 NVIDIA 提供的并行计算平台和编程模型，使程序可以在 GPU 上执行通用计算。它不仅是一门语言，还包含编译器、Runtime/Driver API、数学库、调试与性能分析工具。

### Q2：CUDA 和 GPU 是一回事吗？

**答：** 不是。GPU 是硬件，CUDA 是控制 NVIDIA GPU 执行并行程序的软件平台和编程模型；同一块 GPU 还可以通过图形 API 或其他计算接口使用。

### Q3：Host 和 Device 分别指什么？

**答：** Host 通常指 CPU 及其内存，Device 指 GPU 及其显存。CUDA 程序由 Host 发起内存管理、数据传输和 Kernel Launch，Device 执行并行 Kernel。

### Q4：Kernel 是什么？

**答：** Kernel 是在 GPU 上由大量线程并行执行的函数，使用 `__global__` 修饰。一次 Kernel Launch 会创建一个 Grid，Grid 中每个线程执行同一份函数代码，但拥有不同索引和私有状态。

### Q5：`__global__`、`__device__` 和 `__host__` 有什么区别？

**答：** `__global__` 函数由 Host 或支持动态并行的 Device 发起、在 Device 执行；`__device__` 只能在 Device 调用和执行；`__host__` 在 CPU 调用和执行。`__host__ __device__` 会为两个目标各编译一个版本。

### Q6：Kernel Launch 中的四个尖括号参数是什么？

**答：** `kernel<<<grid, block, shared_bytes, stream>>>()` 依次指定 Grid 维度、每个 Block 的线程维度、动态 Shared Memory 字节数和执行 Stream。后两项可以省略。

### Q7：`dim3` 是什么？

**答：** `dim3` 是保存 `x/y/z` 三个无符号维度的辅助类型，用于描述 Grid 和 Block。未指定的维度默认为 1，因此 `dim3 block(256)` 等价于 `(256,1,1)`。

### Q8：如何计算一维 Global Thread ID？

**答：** 常用公式是 `idx = blockIdx.x * blockDim.x + threadIdx.x`。Grid 往往向上取整，因此访问数组前还要判断 `idx < n`。

### Q9：如何把二维 Block 内线程线性化？

**答：** `linear_tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)`。CUDA 按 x 最快、然后 y、最后 z 的顺序线性化，这也决定线程如何被划入 Warp。

### Q10：为什么 Grid 和 Block 支持三维？

**答：** 主要为了自然表达图像、矩阵和三维网格的索引，维度本身不会自动提升性能。最终线程仍被线性化并按 Warp 调度。

### Q11：Kernel Launch 对 Host 是同步还是异步？

**答：** 一般对 Host 异步，Launch 将工作加入 Stream 后即可返回。Launch 配置错误可以立即发现，越界等执行错误通常要在后续同步点才能报告。

### Q12：如何检查 Kernel Launch 错误？

**答：** Launch 后用 `cudaGetLastError()` 或 `cudaPeekAtLastError()` 检查配置错误，再在调试阶段对 `cudaDeviceSynchronize()` 的返回值进行检查以发现异步执行错误。

### Q13：`cudaGetLastError` 和 `cudaPeekAtLastError` 有什么区别？

**答：** 两者都读取当前线程的 Runtime Error State，但 `cudaGetLastError` 会清除该错误，`cudaPeekAtLastError` 不清除。错误检查宏中必须明确是否希望消费错误状态。

### Q14：CUDA Runtime API 和 Driver API 有什么区别？

**答：** Runtime API 更高层，自动管理 Primary Context，常见前缀是 `cuda`；Driver API 更底层，显式管理 Context、Module 和 Function，前缀是 `cu`。两者最终使用同一驱动，可在满足 Context 规则时互操作。

### Q15：NVCC 做了什么？

**答：** NVCC 将 Host 代码交给 Host Compiler，将 Device 代码编译为 PTX 和/或目标架构的机器码 SASS，再把它们封装进 Fat Binary。程序运行时驱动选择可直接执行的 Cubin，或对 PTX 做 JIT。

### Q16：PTX 和 SASS 有什么区别？

**答：** PTX 是与具体芯片解耦的虚拟 ISA，便于驱动 JIT 到未来架构；SASS 是某个真实 GPU 架构执行的机器指令。分析最终性能必须看 SASS，而不能只看 PTX。

### Q17：`-arch` 和 `-code` 分别控制什么？

**答：** `-arch=compute_xy` 指定生成 Device Code 时使用的虚拟架构能力，`-code=sm_xy` 或 `compute_xy` 指定 Fatbin 中保存目标 Cubin 还是 PTX。工程中通常同时嵌入常用 SM 的 Cubin 和一个 PTX 回退。

### Q18：什么是 Compute Capability？

**答：** Compute Capability 用主次版本表示 GPU 支持的硬件与指令特性，例如原子类型、Shared Memory 上限、Tensor Core 和 Cluster 能力。代码应按能力查询或编译分派，不能只按产品名称判断。

### Q19：Cubin 能跨 GPU 架构执行吗？

**答：** Cubin 通常只在二进制兼容范围内执行，不能假设跨主 Compute Capability 兼容。PTX 具有向后 JIT 能力，但新 PTX 也要求驱动足够新。

### Q20：为什么第一次 Kernel Launch 可能更慢？

**答：** 可能包含 CUDA Context 初始化、Module Lazy Loading、PTX JIT、内存页建立和库内部 Autotune。基准测试需要先 Warmup，不能把首次初始化时间算入稳态 Kernel 延迟。

### Q21：`__forceinline__` 一定更快吗？

**答：** 不一定。内联可消除调用开销并暴露优化机会，但也可能扩大代码、增加寄存器生命周期和 Instruction Cache 压力，应查看编译结果与实测。

### Q22：`__restrict__` 的作用是什么？

**答：** 它承诺该指针访问的对象不会与其他受限指针别名，使编译器能更自由地缓存、重排 Load/Store。若实际发生别名，程序行为可能错误，因此不能只把它当性能开关。

### Q23：`constexpr` 和模板参数为什么常用于 CUDA Kernel？

**答：** 它们让 Tile Size、Stage 数等在编译期已知，便于循环展开、静态 Shared Memory 分配和删除无效分支。代价是产生更多 Kernel 实例并增加编译时间和二进制体积。

### Q24：JIT Kernel 的优缺点是什么？

**答：** JIT 可以针对运行时 Shape、数据类型和 GPU 架构生成专用代码，减少通用分支；缺点是首次编译延迟、缓存管理复杂和线上编译环境要求更高。

### Q25：为什么不能只记住某张 GPU 的硬件上限？

**答：** 最大线程数、寄存器、Shared Memory、Cluster 约束会随架构和具体 SKU 变化。面试中应先讲资源约束关系，再说明实际值通过 Programming Guide、`cudaDeviceProp` 或 Occupancy API 查询。

## 二、线程、Warp、CTA 与 SM：Q26-Q55

### Q26：Thread、Block 和 Grid 的关系是什么？

**答：** Thread 是逻辑线程，多个 Thread 构成 Block，多个 Block 构成一次 Kernel 的 Grid。Block 内可以用 Shared Memory 和块级同步协作，普通 Block 之间不能假设执行顺序。

### Q27：CTA 和 Thread Block 是什么关系？

**答：** CTA 是 Cooperative Thread Array，与 CUDA Thread Block 基本同义。CTA 强调这组线程共享资源并协作执行，也是硬件调度到 SM 的资源分配单位。

### Q28：SM 是什么？

**答：** SM 是 GPU 上执行 CTA 和 Warp 的硬件处理器，内部包含 Warp Scheduler、寄存器文件、Shared Memory/L1、Load/Store 单元、CUDA Core 和 Tensor Core 等资源。

### Q29：CTA 和 SM 的关系是什么？

**答：** 一个普通 CTA 一生只驻留在一个 SM 上，不会执行到一半迁移；一个 SM 可以在资源允许时同时驻留多个 CTA。CTA 数量多于可同时驻留数量时会分成多波执行。

### Q30：一个 CTA 能跨多个 SM 吗？

**答：** 普通 CTA 不能。Hopper 以后 Thread Block Cluster 可让多个 CTA 分别驻留在多个 SM 上协作，但 Cluster 中每个 CTA 仍各自属于一个 SM。

### Q31：Warp 是什么？

**答：** Warp 是 NVIDIA GPU 的线程调度基本单位，当前 CUDA 架构通常每个 Warp 有 32 个 Lane。Warp 中线程共享指令发射，但每个 Lane 有自己的寄存器、地址和 Active Mask。

### Q32：SIMT 和 SIMD 有什么区别？

**答：** SIMD 由一条显式向量指令操作多个元素，SIMT 向程序员暴露多个独立线程，再由硬件将它们组成 Warp 执行。两者都利用并行数据通路，但 SIMT 保留线程级控制流和状态。

### Q33：Warp 中线程必须一直执行同一条路径吗？

**答：** 不必须，但分歧路径通常要在不同 Active Mask 下分别执行，导致部分 Lane 暂时空闲。Volta 之后有 Independent Thread Scheduling，但这没有消除 Divergence 成本。

### Q34：什么是 Warp Divergence？

**答：** 同一 Warp 的线程因数据相关分支走不同路径，硬件需要分时执行这些路径并屏蔽不参与的 Lane。不同 Warp 走不同分支不叫 Warp Divergence。

### Q35：所有 `if` 都会导致严重 Divergence 吗？

**答：** 不会。若整个 Warp 条件一致，或者编译器将短分支转成 Predication，代价可能很小；只有同一 Warp 内路径不一致且分支工作较重时影响明显。

### Q36：什么是 Predication？

**答：** 编译器可能让两条短路径都按谓词指令执行，仅对条件满足的 Lane 提交结果，从而避免真正跳转。它减少控制流开销，但被屏蔽指令仍可能消耗发射资源。

### Q37：什么是 Active Mask？

**答：** Active Mask 表示当前 Warp 中哪些 Lane 正参与指令。Warp Vote、Shuffle 和 `__syncwarp` 应使用与实际参与线程一致的 Mask，错误使用完整 Mask 可能造成未定义行为或死锁。

### Q38：`lane_id` 和 `warp_id` 如何计算？

**答：** 对线性线程号 `tid`，`lane_id = tid % warpSize`，`warp_id = tid / warpSize`。代码可用 `warpSize` 而不是硬编码 32，尽管当前 NVIDIA CUDA Warp 通常为 32。

### Q39：Warp Scheduler 如何隐藏延迟？

**答：** 当一个 Warp 等待 Global Memory 或依赖结果时，Scheduler 可发射另一个 Ready Warp 的指令。隐藏延迟依赖足够的 Ready Warp 或单 Warp 内足够的 ILP，而不只是线程总数。

### Q40：什么是 Occupancy？

**答：** Occupancy 是每个 SM 实际可驻留 Warp 数与架构最大 Warp 数之比。它描述可用于隐藏延迟的并发 Warp 容量，不等于 CUDA Core 或 Tensor Core 的实际利用率。

### Q41：Occupancy 越高越好吗？

**答：** 不是。达到足以隐藏延迟的 Occupancy 后，继续压低寄存器或缩小 Tile 可能减少数据复用和 ILP，反而变慢；最终以吞吐和延迟实测为准。

### Q42：哪些资源限制每个 SM 的 Resident CTA 数？

**答：** 主要包括每 CTA 线程/Warps、每线程寄存器、每 CTA Shared Memory，以及架构允许的最大 CTA、Warp 和线程数。最终 Resident CTA 数由最先耗尽的资源决定。

### Q43：寄存器如何限制 Occupancy？

**答：** 每个 SM 的寄存器文件有限，`每线程寄存器数 × 每 CTA 线程数 × 驻留 CTA 数` 不能超过上限，而且分配还受粒度影响。寄存器增加可能减少可同时驻留的 CTA。

### Q44：Shared Memory 如何限制 Occupancy？

**答：** 每个 CTA 的静态和动态 Shared Memory 都从 SM 容量中分配。若单 CTA 使用量过大，SM 即使还有线程和寄存器配额，也可能只能驻留一个 CTA。

### Q45：什么是 Wave Quantization？

**答：** Grid 中 CTA 按 SM 可驻留容量分波执行，最后一波若 CTA 很少会使部分 SM 空闲。例如一波可运行 132 个 CTA，而最后只剩 20 个，就会产生明显 Tail。

### Q46：什么是 Persistent Kernel？

**答：** Persistent Kernel 只启动接近 SM 驻留容量的 CTA，每个 CTA 在循环中持续领取多个任务。它便于统一调度不规则工作、减少重复 Launch，并可通过动态任务队列缓解负载不均。

### Q47：Persistent CTA 能完全消除 Tail 吗？

**答：** 不能。它只有在多个小任务形成统一任务池或支持动态领取时才明显减少尾部；如果总任务本来少于 SM 数，Persistent 也无法创造额外并行度。

### Q48：Block 数小于 SM 数会怎样？

**答：** 最多只有同样数量的 SM 获得工作，其余 SM 空闲。可尝试减小每 Block 工作量、拆分 Reduction 维或提高 Batch 并行度，但拆得过细会增加 Launch、同步和中间结果成本。

### Q49：Block 数越多越好吗？

**答：** 不是。足够覆盖所有 SM 并隐藏 Tail 后，继续增加 Block 可能只增加重复加载、原子竞争和调度开销；任务粒度要结合数据复用与负载均衡选择。

### Q50：如何选择每个 Block 的线程数？

**答：** 通常从 128、256 等 Warp 整数倍开始，结合寄存器、Shared Memory、每线程工作量和数据布局用 Occupancy API 与基准测试调优。1024 线程虽然可能合法，却经常限制 Resident CTA 数。

### Q51：Block 中线程数必须是 32 的倍数吗？

**答：** 不是语义要求，但通常建议如此。否则最后一个 Warp 有无效 Lane，除非边界问题或算法结构使这种浪费可以接受。

### Q52：不同 Block 的执行顺序有保证吗？

**答：** 普通 Kernel 没有保证，不能用 Block ID 推断先后。跨 Block 依赖应拆成多个 Kernel、使用 Cooperative Launch/Grid Sync，或设计正确的全局原子协议。

### Q53：什么是 Thread Block Cluster？

**答：** Cluster 是 Hopper 引入的 CTA 分组，保证一组 CTA 在受支持的硬件范围内协同调度。Cluster 可使用 Cluster Barrier、Distributed Shared Memory 和 TMA Multicast。

### Q54：什么是 Cooperative Launch？

**答：** Cooperative Launch 保证整个 Grid 的 Block 能按要求同时驻留，从而允许 `grid.sync()`。由于必须满足全 Grid Residency，Grid 大小和 Occupancy 受到更严格限制。

### Q55：Dynamic Parallelism 是什么？

**答：** 它允许 Device Kernel 启动新的 Kernel，适合递归或动态工作生成。其 Launch 和同步成本不低，规则也更复杂，许多场景用 Host 调度或 Persistent Work Queue 更高效。

## 三、CUDA 内存体系：Q56-Q85

### Q56：CUDA 主要有哪些内存空间？

**答：** 常见有 Register、Local、Shared、Global、Constant、Texture，以及硬件管理的 L1/L2 Cache。它们的物理位置、作用域、生命周期、容量和访问模式不同。

### Q57：Register 的作用域和生命周期是什么？

**答：** Register 通常是线程私有的片上状态，生命周期随线程执行结束。它延迟低但总量有限，使用过多会降低 Occupancy 或发生 Spilling。

### Q58：Local Memory 在哪里？

**答：** Local Memory 在逻辑上线程私有，物理上位于 Device Memory，并经过 L1/L2 Cache。大数组、动态索引对象或寄存器不足都可能使变量进入 Local Memory。

### Q59：什么是 Register Spilling？

**答：** 编译器无法把所有线程私有值放进寄存器时，会生成 Local Load/Store。Spill 增加显存流量和延迟，但少量缓存命中的 Spill 不一定立即成为主要瓶颈。

### Q60：如何判断发生了 Spilling？

**答：** 编译时查看 `ptxas -v` 的 spill load/store 与 stack frame，运行时用 Nsight Compute 检查 Local Memory 事务和相关 Stall。源码变量数量本身不能准确判断。

### Q61：Shared Memory 是什么？

**答：** Shared Memory 是 CTA 内线程共享的可编程片上 Scratchpad，生命周期与 CTA 相同。它常用于数据复用、线程间交换和重排访问模式。

### Q62：Global Memory 是什么？

**答：** Global Memory 是所有 GPU 线程可访问的大容量 Device Memory，通常对应显存 HBM/GDDR，并经过 Cache。其带宽高但延迟远高于片上存储，需要合并访问和足够并行度。

### Q63：L1 和 Shared Memory 是什么关系？

**答：** 在多代架构上它们共享或紧密耦合同一片片上存储资源，但具体容量划分和配置随架构变化。Shared Memory 由程序显式管理，L1 由硬件 Cache 策略管理。

### Q64：L2 Cache 的作用域是什么？

**答：** L2 是 Device 级共享 Cache，服务所有 SM，并处于 Global Memory 之前。跨 SM 的只读复用、原子操作和部分写回都会经过 L2，但命中并不由程序完全保证。

### Q65：Constant Memory 适合什么访问？

**答：** 适合同一 Warp 的线程读取同一个地址，此时 Constant Cache 可广播一次数据。若每个 Lane 读取不同 Constant 地址，请求可能被串行化。

### Q66：Texture Memory 适合什么访问？

**答：** Texture 路径适合只读且具有二维空间局部性、地址模式不规则或需要硬件采样功能的数据。对普通连续数组，现代只读 Cache 或正常 Global Load 可能已经足够。

### Q67：Static Shared Memory 和 Dynamic Shared Memory 有何区别？

**答：** Static Shared Memory 大小在编译期确定，如 `__shared__ float s[256]`；Dynamic Shared Memory 通过 `extern __shared__` 声明，大小由 Launch 的第三个参数传入。

### Q68：多个 Dynamic Shared Memory 数组如何组织？

**答：** `extern __shared__` 实际提供一段字节区，需要手工切片并满足每种类型的对齐。错误的 Offset 或对齐会造成未对齐访问甚至结果错误。

### Q69：`cudaMalloc` 分配的内存是否自动初始化？

**答：** 不会，内容未定义。需要零初始化时使用 `cudaMemset`，但注意它按字节填充值，不能用来设置任意浮点数。

### Q70：`cudaMemset(ptr, 1, bytes)` 能把 `float` 数组设成 1 吗？

**答：** 不能，它把每个字节写成 `0x01`，得到的浮点位模式不是 `1.0f`。零值位模式全零，因此设 0 通常可用。

### Q71：`cudaMallocPitch` 解决什么问题？

**答：** 它为二维数据分配满足硬件对齐的 Row Pitch，可能在每行尾部加入 Padding。Kernel 必须使用返回的 Pitch 计算行地址，不能假设等于 `width * sizeof(T)`。

### Q72：Pinned Host Memory 是什么？

**答：** Pinned Memory 是不会被 OS 换出的页锁定 Host Memory，DMA 引擎可直接传输。它通常提高 H2D/D2H 带宽并支持真正的异步重叠，但大量锁页会损害系统内存管理。

### Q73：Pageable Host Memory 为什么通常更慢？

**答：** DMA 需要稳定物理页，Runtime 往往先把 Pageable Buffer 复制到内部 Pinned Staging Buffer，再发起传输。额外 CPU Copy 和同步使延迟增大。

### Q74：Unified Memory 是什么？

**答：** `cudaMallocManaged` 提供 CPU/GPU 可访问的统一指针，Runtime 根据访问迁移或映射页面。它简化编程，但缺页迁移和不良访问模式可能造成严重抖动。

### Q75：如何优化 Unified Memory？

**答：** 使用 `cudaMemPrefetchAsync` 提前迁移，配合 `cudaMemAdvise` 声明首选位置和访问者，并避免 CPU/GPU 高频交替写同一批页面。仍应通过 NSYS 查看 Page Fault 与迁移行为。

### Q76：UVA 和 Unified Memory 是一回事吗？

**答：** 不是。Unified Virtual Addressing 让 Host 和多个 Device 的地址位于统一虚拟地址空间，便于识别指针归属；Unified Memory 还负责数据可访问性与页面迁移。

### Q77：Zero-copy Memory 是什么？

**答：** 将 Pinned Host Memory 映射到 Device 地址后，GPU 可直接访问 Host Memory，无需显式复制。它适合一次性、低复用和小规模访问，频繁随机访问会受 PCIe/NVLink 延迟限制。

### Q78：Device Memory Pool 有什么作用？

**答：** Memory Pool 复用已分配的 Device Memory，减少高频 `cudaMalloc/cudaFree` 的分配开销和潜在同步。`cudaMallocAsync/cudaFreeAsync` 还能按 Stream 顺序管理生命周期。

### Q79：`cudaMallocAsync` 为什么更适合并发流水？

**答：** 它把分配操作纳入 Stream 依赖，而不是要求全设备安全点，并从 Pool 中复用内存。正确性仍要求使用该内存的 Stream 依赖在 Free 之前完成。

### Q80：`cudaFree` 可能带来什么问题？

**答：** 传统 `cudaFree` 可能触发隐式同步或昂贵的分配器路径，破坏流水。高频临时 Buffer 更适合预分配、Memory Pool 或 `cudaFreeAsync`。

### Q81：Device Pointer 能在 Host 上直接解引用吗？

**答：** 普通 `cudaMalloc` 返回的 Device Pointer 不能由 CPU 直接读写。UVA 统一了地址形式，不代表每个地址对 CPU 都可访问。

### Q82：Host Pointer 能直接传给 Kernel 吗？

**答：** 普通 `malloc` 指针通常不能。Managed Memory、Mapped Pinned Memory 或支持系统分配访问的特定平台可以，但要满足设备能力和映射规则。

### Q83：为什么大模型推理经常预分配 Workspace？

**答：** 预分配避免请求期间反复分配、同步和碎片，也让地址稳定以支持 CUDA Graph。代价是需要估算上限并设计复用与并发隔离。

### Q84：显存碎片和“总空闲够但分配失败”是什么关系？

**答：** 分配器可能找不到足够大的连续可用块，或内存被不同生命周期的 Buffer 切碎。Pool、分层 Arena、固定大小 Page 和 Paged KV Cache 都是在缓解碎片。

### Q85：为什么不能按固定延迟给内存类型排序？

**答：** 实际延迟受 Cache 命中、冲突、合并访问、并行请求和架构影响。正确表述是理解作用域与数据路径，再通过 Profiling 判断当前 Kernel 的实际瓶颈。

## 四、合并访问、Shared Memory 与 Cache：Q86-Q115

### Q86：什么是 Global Memory Coalescing？

**答：** 一个 Warp 的内存请求若能被硬件合并成尽量少的对齐 Memory Transaction，就称为合并访问。重点不是地址在源码里“看起来连续”，而是最终触及多少内存 Sector。

### Q87：连续线程访问连续 `float` 为什么通常高效？

**答：** 32 个 Lane 共访问连续 128 字节，通常只覆盖少量连续且对齐的 Sector，取回数据利用率高。具体 Transaction 组织随 Cache 配置和架构变化。

### Q88：Stride 访问为什么可能浪费带宽？

**答：** 若 Lane `i` 访问 `base + i * stride`，大 Stride 会让一个 Warp 触及许多 Sector，每个 Sector 只使用少量字节。有效带宽因过量传输下降。

### Q89：未对齐访问一定很慢吗？

**答：** 不一定，但跨越额外 Sector 时会增加 Transaction；相邻 Warp 可能从 Cache 复用多取的数据。应结合 Requested Bytes、Transferred Bytes 和 Sector 数判断。

### Q90：`float4` 一定比 `float` 快吗？

**答：** 不一定。向量化能减少指令数并改善宽 Load/Store，但要求地址和数据长度满足对齐，且可能增加寄存器压力；访存已合并时收益可能有限。

### Q91：AoS 和 SoA 哪个更适合 GPU？

**答：** 若 Warp 同时读取同一字段，Structure of Arrays 往往使该字段连续并更易合并；若每个线程总是读取完整小对象，AoS 也可能合理。选择取决于 Warp 的字段访问模式。

### Q92：矩阵转置为什么容易出现非合并写？

**答：** 直接让线程读取一行再写一列时，读或写的一侧会形成大 Stride。常见方法是先以合并方式读入 Shared Memory，再交换索引以合并方式写出。

### Q93：Shared Memory 有多少个 Bank？

**答：** 当前主流 NVIDIA 架构通常有 32 个 Bank，连续 32-bit Word 映射到连续 Bank。精确 Bank 宽度和特殊模式应按目标架构文档确认。

### Q94：什么是 Bank Conflict？

**答：** 同一 Warp 的一个 Shared Memory 请求访问同一 Bank 中的不同地址时，硬件需要拆成多次服务，形成 Conflict。Conflict 度越高，完成请求所需 Wavefront 越多。

### Q95：同一 Warp 的线程读取同一个 Shared 地址会冲突吗？

**答：** 通常不会按普通冲突串行，而会使用 Broadcast/Multicast 机制返回同一个值。冲突针对的是同一 Bank 的不同地址。

### Q96：为什么转置 Tile 常写成 `[32][33]`？

**答：** `[32][32]` 按列访问时相邻 Lane 的地址可能都映射到同一 Bank；多一列改变行 Stride 的模 32 关系，使相邻 Lane 分散到不同 Bank。

### Q97：Padding 总能消除 Bank Conflict 吗？

**答：** 不能。它只对特定 Element Size、二维 Stride 和访问方向有效，向量 Load、64/128-bit 访问会产生更复杂的 Bank 组合，需要根据实际地址和 NCU 指标验证。

### Q98：Shared Memory Swizzle 是什么？

**答：** Swizzle 用位运算重排 Shared Memory 坐标，使 Tensor Core 或转置访问分散到不同 Bank，同时保持可逆的逻辑布局。它比固定 `+1` Padding 更适合复杂 Tile。

### Q99：Shared Memory 为什么能提高 GEMM 性能？

**答：** CTA 将 A/B Tile 从 Global Memory 加载一次后，可被多个线程和多次 FMA 复用。它用片上容量换取更高 Arithmetic Intensity。

### Q100：使用 Shared Memory 一定更快吗？

**答：** 不一定。若数据只用一次、L1 已充分命中，额外写 Shared、同步和地址计算可能更贵；Shared 还会限制 Occupancy。

### Q101：什么是 Tiling？

**答：** Tiling 把大问题切成适合片上存储与线程层级的小块，在 Tile 内进行数据复用。高性能 Kernel 往往同时有 CTA Tile、Warp Tile、MMA Tile 和 Thread Tile。

### Q102：为什么 Tile 越大不一定越好？

**答：** 大 Tile 增加数据复用和每 CTA 工作量，但消耗更多 Shared Memory、寄存器并减少并行 CTA 数，还可能增加边界浪费。应在复用、Occupancy 和并行度之间平衡。

### Q103：什么是 Double Buffering？

**答：** 使用两组片上 Buffer，让一组参与计算时另一组加载下一 Tile。其目标是重叠数据搬运和计算，而不是减少总传输字节。

### Q104：多 Stage Pipeline 和 Double Buffer 有什么关系？

**答：** Double Buffer 是两 Stage 的特例。增加 Stage 可以覆盖更长的访存延迟，但会消耗更多 Shared Memory 和 Barrier 状态，Stage 过多可能降低 Occupancy。

### Q105：`cp.async` 解决什么问题？

**答：** Ampere 的 `cp.async` 可把数据从 Global Memory 异步复制到 Shared Memory，减少通过普通寄存器中转的 Load/Store 指令，并允许 Copy 与计算形成 Pipeline。

### Q106：TMA 和 `cp.async` 的主要区别是什么？

**答：** TMA 从 Hopper 起可由少量线程通过 Tensor Descriptor 发起 1D 到多维大块搬运，并由硬件完成地址生成；`cp.async` 通常由 Warp 多线程协作搬运较小片段。

### Q107：TMA 为什么适合 Warp Specialization？

**答：** 单个选举线程即可发起大块异步传输，Producer Warp 不必用全部 Lane 做地址计算和 Load/Store，其他 Warp 可专职执行 WGMMA 或 Epilogue。

### Q108：什么是 TMA Multicast？

**答：** Cluster 中多个 CTA 需要同一 Tile 时，一次 TMA 操作可把数据发送到多个 CTA 的 Shared Memory，减少 HBM/L2 重复读取。它要求 Cluster 和 Barrier 正确配置。

### Q109：L1 Cache Miss 一定会访问 HBM 吗？

**答：** 不一定，先查询 Device 级 L2，命中后无需访问 HBM。只有下层 Cache 也未命中或写回需要时才产生显存流量。

### Q110：Cache 命中率越高性能越好吗？

**答：** 不一定。高命中率可能对应很少的有效工作，或 Kernel 瓶颈在计算和同步；应同时看吞吐、Sector 数、请求字节和执行时间。

### Q111：什么时候使用 L2 Persisting Cache？

**答：** 当一小段数据会跨 Kernel 或多 CTA 高频复用时，可通过 Access Policy Window 提示 L2 保留。设置范围过大或多个 Stream 竞争会降低效果。

### Q112：`__ldg` 还有必要吗？

**答：** 在旧架构上它显式走只读数据路径；现代编译器通常能根据 `const __restrict__` 自动选择合适 Load。需要检查目标架构和生成代码，不应机械添加。

### Q113：怎样计算 Arithmetic Intensity？

**答：** `AI = 有效 FLOPs / 从目标内存层传输的字节数`。必须说明分母针对 HBM、L2 还是 Shared Memory，因为不同层对应不同 Roofline。

### Q114：如何判断一个优化是否真正减少 HBM 流量？

**答：** 从算法上计算理论字节数，再用 Nsight Compute 检查 DRAM Bytes/Sectors。源码中加入 Shared Memory 不代表 HBM 一定减少，Cache 与重复加载方式会影响实际流量。

### Q115：为什么边界 Mask 会影响访存效率？

**答：** 尾部 Warp 只有少数 Lane 有效，却可能仍触发完整 Sector 和指令，降低有效字节比例。Padding、专门 Tail Kernel 或 Predication 可改善，但要与额外复杂度权衡。

## 五、同步、原子操作与内存模型：Q116-Q145

### Q116：`__syncthreads()` 做什么？

**答：** 它是 CTA 范围的 Barrier，所有参与线程到达后才能继续，并使 Barrier 前的相关 Shared/Global Memory 访问对 CTA 内线程可见。它不是全 Grid 同步。

### Q117：把 `__syncthreads()` 放在分支里有什么风险？

**答：** 如果同一 CTA 中不是所有线程都执行到同一个 Barrier，部分线程会永久等待，造成死锁。只有条件对整个 CTA 一致时才安全。

### Q118：`__syncwarp()` 做什么？

**答：** 它同步指定 Mask 中的 Warp Lane，并为这些线程提供内存有序性。Mask 必须只包含会实际到达该调用的线程。

### Q119：Volta 以后还可以假设 Warp 隐式同步吗？

**答：** 不应该。Independent Thread Scheduling 允许 Lane 进度更独立，Warp 内通过 Shared/Global Memory 通信时应使用 `__syncwarp` 或正式的 Cooperative Groups 原语。

### Q120：Warp Shuffle 有什么作用？

**答：** `__shfl*_sync` 让 Lane 直接交换寄存器值，适合 Warp Reduction、Broadcast 和 Scan，避免 Shared Memory 与块级 Barrier。它只能在指定 Active Mask 内交换。

### Q121：`__ballot_sync` 返回什么？

**答：** 它收集指定 Warp Lane 的布尔谓词，返回一个 Bit Mask。可用于计算满足条件的线程数、Warp 内压缩和分支状态分析。

### Q122：`__activemask()` 可以直接作为之前分支的参与 Mask 吗？

**答：** 不总是。它只反映调用这一刻仍活跃的 Lane，可能已因调度和控制流变化而丢失原参与集合；更稳妥的是在分支前用 `__ballot_sync` 明确生成 Mask。

### Q123：Atomic Operation 保证什么？

**答：** 它保证指定作用域内对某个地址的读改写不可被其他同类访问撕裂。原子性不等于全局 Barrier，也不自动提供所需的内存顺序。

### Q124：为什么大量 `atomicAdd` 会慢？

**答：** 多线程竞争同一地址时操作必须序列化，并产生 Cache Line/Atomic Unit 争用。常用优化是 Warp 或 Block 内先归约，再减少全局原子次数。

### Q125：Atomic 一定比锁快吗？

**答：** 不一定，但 GPU 上简单原子通常比自旋锁更安全。锁持有线程若未被调度，而同一 Warp 或占满资源的其他线程持续自旋，可能造成严重停顿甚至死锁。

### Q126：如何用 `atomicCAS` 实现自定义原子操作？

**答：** 循环读取旧值、计算新值，再用 CAS 尝试替换；失败说明被其他线程修改，需要重试。要处理浮点 NaN、ABA 和内存顺序等边界。

### Q127：`atomicAdd` 能代替 Reduction 吗？

**答：** 正确性上常可行，性能取决于竞争度和硬件原子吞吐。大规模 Reduction 通常先做 Warp/Block 局部归约，最后每 Block 只原子一次。

### Q128：什么是 Memory Fence？

**答：** Fence 约束调用线程的内存操作在指定作用域内的可见顺序，但不等待其他线程到达。它需要与标志位、原子或 Barrier 配合构造同步协议。

### Q129：`__threadfence_block`、`__threadfence`、`__threadfence_system` 有何区别？

**答：** 它们分别面向 Block、Device 和 System 作用域建立调用线程的内存顺序。作用域越大通常成本越高，应选择满足正确性的最小范围。

### Q130：`volatile` 能代替 Atomic 或 Fence 吗？

**答：** 不能。`volatile` 主要限制编译器对访问的缓存和消除，不保证多线程原子性，也不完整定义跨线程同步顺序。

### Q131：什么是 Data Race？

**答：** 多线程并发访问同一位置，至少一个是写，并且缺少满足内存模型的同步时产生 Data Race。结果可能不确定，不能依赖“多数运行正确”。

### Q132：如何检测 CUDA Race？

**答：** 使用 Compute Sanitizer 的 `racecheck` 检查 Shared Memory Hazard，结合 `memcheck`、`synccheck` 和最小化输入定位。工具不能证明所有并发程序无 Race。

### Q133：为什么 Kernel 之间天然可以作为全局同步点？

**答：** 同一 Stream 中后一个 Kernel 必须等前一个完成，因此前一 Kernel 的全局写入对后一 Kernel 可见。两次 Launch 的成本换来了简单可靠的 Grid 级阶段边界。

### Q134：如何在一个 Kernel 内做 Grid Sync？

**答：** 使用 Cooperative Groups 的 `this_grid().sync()`，并通过 Cooperative Launch 启动。必须保证整个 Grid 可同时驻留，否则无法安全完成全 Grid Barrier。

### Q135：Cooperative Groups 的价值是什么？

**答：** 它把协作线程组显式表示为对象，支持 Block、Warp Tile、Cluster 和 Grid 等不同粒度的同步、Reduction 与 Scan，避免依赖脆弱的隐式 Warp 假设。

### Q136：`cuda::barrier` 和 `__syncthreads` 有何不同？

**答：** `cuda::barrier` 可分离 Arrive 与 Wait，并与异步 Copy 的完成计数结合，更适合 Producer-Consumer Pipeline；`__syncthreads` 是一次固定的全 CTA 到达等待。

### Q137：什么是 Producer-Consumer Pipeline？

**答：** Producer 异步填充某个 Stage，Consumer 等待该 Stage Ready 后计算，再通知 Buffer 可复用。多 Stage 通过 Phase/Barrier 避免覆盖尚未消费的数据。

### Q138：什么是 Memory Order Relaxed？

**答：** Relaxed Atomic 只保证该原子对象的原子修改顺序，不为其他普通 Load/Store 建立同步关系。计数器常可用 Relaxed，发布数据通常需要 Release/Acquire。

### Q139：Release/Acquire 解决什么问题？

**答：** Producer 先写数据再 Release 发布标志，Consumer Acquire 读到该标志后，可以观察 Producer 在 Release 前的写入。前提是作用域和内存位置匹配。

### Q140：CUDA 为什么需要 Thread Scope？

**答：** 同步线程距离不同，成本和一致性点也不同。Block、Device、System 等 Scope 让程序只为实际通信范围支付成本。

### Q141：Cluster Scope 是什么？

**答：** 它位于 Block 与 Device 之间，覆盖同一 Thread Block Cluster 内的 CTA。Cluster Barrier、Distributed Shared Memory 和 Cluster Atomic 可在该范围协作。

### Q142：Distributed Shared Memory 是什么？

**答：** Hopper Cluster 中，一个 CTA 可以访问同 Cluster 其他 CTA 的 Shared Memory 地址。它仍是片上受限资源，远端访问和同步成本高于本 CTA Shared Memory。

### Q143：两个 CTA 能直接用 `__syncthreads()` 同步吗？

**答：** 不能，`__syncthreads` 只覆盖当前 CTA。Cluster CTA 使用 Cluster Barrier，任意 Grid CTA 则需 Cooperative Grid Sync、Atomic 协议或拆 Kernel。

### Q144：为什么自旋等待可能导致 GPU 死锁？

**答：** 等待方 CTA 可能占满全部 Resident 资源，使负责设置条件的 CTA 无法被调度；Warp 内锁持有者和等待者的发散执行也可能互相阻塞。应避免依赖不保证调度公平的协议。

### Q145：如何选择同步粒度？

**答：** 使用满足正确性的最小线程范围：能用 Warp 就不做 CTA Barrier，能用 CTA 就不做 Grid/Device 同步。范围越大，等待线程越多且内存一致性成本越高。

## 六、Stream、Event、异步拷贝与 CUDA Graph：Q146-Q175

### Q146：CUDA Stream 是什么？

**答：** Stream 是按提交顺序执行的 Device 工作队列，可包含 Kernel、Memcpy、Memset、Event 等操作。同一 Stream 内有顺序保证，不同 Stream 的工作在硬件资源允许时可以重叠。

### Q147：不同 Stream 的 Kernel 一定并发吗？

**答：** 不一定。它们只是具备并发资格，还受 SM 资源、Kernel Occupancy、Hyper-Q、依赖关系和默认 Stream 语义限制；一个 Kernel 占满全部 SM 资源时另一个无法驻留。

### Q148：Legacy Default Stream 有什么特殊语义？

**答：** Legacy Default Stream 会与同一 Context 中的普通 Blocking Stream 形成隐式同步，容易破坏并发。Non-blocking Stream 和 Per-thread Default Stream 的语义不同，不能笼统说“Stream 0 总会同步一切”。

### Q149：Per-thread Default Stream 是什么？

**答：** 每个 Host Thread 拥有自己的默认 Stream，并像普通 Stream 一样与其他 Stream 并发，避免 Legacy 全局默认流的隐式同步。可通过编译选项或宏启用。

### Q150：Kernel 和 Host 代码如何异步？

**答：** Kernel Launch 通常只把命令写入 Stream 后返回，CPU 可继续工作。若随后调用同步 API、同步 Memcpy 或访问依赖结果，Host 才会等待。

### Q151：`cudaMemcpyAsync` 为什么可能没有重叠？

**答：** Host Buffer 不是 Pinned、Copy 和 Kernel 在同一 Stream、设备缺少相应 Copy Engine，或存在隐式同步时都可能无法重叠。必须用 Nsight Systems 验证时间线。

### Q152：实现 H2D、Kernel、D2H Pipeline 需要什么？

**答：** 把数据分 Chunk，使用 Pinned Host Buffer 和多个非默认 Stream，让不同 Chunk 分别处于 H2D、计算和 D2H 阶段。设备还需支持 Copy/Compute Concurrent Capability。

### Q153：为什么数据分 Chunk 才容易重叠？

**答：** 单个大 Buffer 必须完整 H2D 后 Kernel 才能开始，也要完整计算后才能 D2H。Chunking 建立生产流水，使第 `i+1` 块传输时第 `i` 块可计算。

### Q154：Stream Priority 能保证高优先级 Kernel 立即抢占吗？

**答：** 不能保证细粒度抢占正在运行的 CTA。Priority 主要影响待调度工作选择，实际抢占能力和粒度随硬件与工作类型变化。

### Q155：CUDA Event 有什么用途？

**答：** Event 可在 Stream 中记录进度，用于跨 Stream 依赖、异步完成查询和 GPU 时间测量。它比全设备同步粒度更小。

### Q156：`cudaStreamWaitEvent` 为什么有用？

**答：** 它让一个 Stream 等待另一个 Stream 中的 Event，而 Host 无需阻塞。这样能在 Device 侧构建 DAG 依赖并保留其他独立工作的并发。

### Q157：为什么用 CUDA Event 测 Kernel 时间？

**答：** Event 时间戳记录在 Device 执行时间线上，可排除 Host 异步 Launch 返回造成的误差。开始和结束 Event 应记录在同一目标 Stream 并等待结束 Event 完成。

### Q158：CUDA Event 计时包含排队时间吗？

**答：** 包含两个 Event 在对应 Stream 时间线之间的等待和执行，若期间有依赖或其他同 Stream 工作也会计入。它不是纯指令执行周期分析工具。

### Q159：`cudaDeviceSynchronize` 和 `cudaStreamSynchronize` 有何区别？

**答：** 前者等待当前 Device 的相关工作，后者只等待指定 Stream。性能敏感代码应优先使用满足依赖的最小同步范围。

### Q160：`cudaStreamQuery` 有什么作用？

**答：** 它非阻塞检查 Stream 是否完成，返回 `cudaErrorNotReady` 表示仍在执行。适合 Host 轮询或与其他 CPU 工作协同。

### Q161：什么是 CUDA Graph？

**答：** CUDA Graph 把一组 Kernel、Memcpy 和依赖预先实例化为可重复提交的执行图，减少每轮 Host Launch 与依赖解析开销。它特别适合 Shape 和地址稳定的迭代工作负载。

### Q162：CUDA Graph 为什么适合 LLM Decode？

**答：** Decode 每步有大量短 Kernel，GPU 计算时间可能接近 CPU Launch 时间，而且执行拓扑重复。Graph 将一整步批量提交，减少 CPU Gap。

### Q163：CUDA Graph 会让单个 Kernel 算得更快吗？

**答：** 通常不会改变 Kernel 内部算法，主要减少 Host 调度和 Launch 间隙。若原瓶颈是大 GEMM 计算，Graph 收益可能很小。

### Q164：Stream Capture 是什么？

**答：** Runtime 记录 Capture 期间提交到 Stream 及其依赖 Stream 的操作，生成 Graph。Capture 期间某些同步、分配和不安全 API 受限制。

### Q165：Graph 中输入数据能变化吗？

**答：** Buffer 内容可以变化，但默认情况下节点参数和地址按实例化时记录。可保持固定地址写入新内容，或使用 Graph Update/Node Parameter Update 修改允许的参数。

### Q166：为什么动态 Shape 给 CUDA Graph 带来困难？

**答：** Shape 变化可能改变 Grid、Workspace、Kernel 选择和内存地址。常见方案是按 Shape Bucket 建多个 Graph，或固定最大 Shape 并传入 Device Mask/有效长度。

### Q167：Graph Capture 中为什么避免普通 `cudaMalloc`？

**答：** 传统分配可能涉及全局状态和同步，不一定可安全捕获。通常预分配稳定 Workspace，或使用支持 Graph/Stream 顺序的 Memory Pool API。

### Q168：什么是 Programmatic Dependent Launch？

**答：** PDL 允许后继 Kernel 在前驱 Kernel 完全结束前被调度，并在真正依赖数据前等待，以重叠前驱尾部和后继准备阶段。它需要硬件、Launch 属性和 Kernel 协议支持。

### Q169：Host Callback 能做什么？

**答：** Callback/Host Function 在 Stream 到达指定位置后由 Host 执行，适合通知和轻量后处理。回调中不应调用可能导致 CUDA 死锁的 API，也不适合重 CPU 任务。

### Q170：多个 Host Thread 使用 CUDA 有什么注意事项？

**答：** 要理解 Primary Context、当前 Device、Per-thread Default Stream 和 Runtime Error State 的线程语义。共享 Stream、Event 和 Buffer 时仍需 Host 侧同步。

### Q171：一个进程可以创建多少 Stream？

**答：** API 可创建很多 Stream，但硬件并发队列、SM 和 Copy Engine 数量有限。Stream 过多会增加管理成本，不会线性增加并发。

### Q172：Stream 顺序能保证内存可见性吗？

**答：** 同一 Stream 中后续操作观察前序操作结果；跨 Stream 必须通过 Event、显式同步或正确内存模型建立依赖。仅按 Host 提交先后不能保证不同 Stream 的 Device 执行顺序。

### Q173：库调用如何绑定 Stream？

**答：** cuBLAS、cuDNN 等 Handle 通常需要显式设置 Stream。若库仍使用默认 Stream，可能破坏框架已有并发和依赖。

### Q174：为什么频繁创建销毁 Stream/Event 不好？

**答：** 它增加 Host Runtime 管理和资源分配开销。长期服务通常建立 Stream/Event Pool 并重复使用。

### Q175：怎样确认多 Stream 真的有效？

**答：** 用 Nsight Systems 查看不同 Stream 的 Kernel/Copy 是否在时间线上重叠，并比较端到端吞吐。只看 API 已使用 `Async` 或创建了多个 Stream 不能证明并发。

## 七、性能模型、Occupancy 与 Profiling：Q176-Q210

### Q176：CUDA Kernel 常见瓶颈类型有哪些？

**答：** 常见有 Compute-bound、Memory-bandwidth-bound、Memory-latency-bound、Launch-bound、Synchronization-bound 和 Instruction-bound。优化前必须先识别主瓶颈。

### Q177：Compute-bound 是什么？

**答：** 算术或 Tensor Core 吞吐接近上限，而内存带宽仍有余量，增加计算单元效率比减少少量访存更重要。典型例子是足够大的高 Arithmetic Intensity GEMM。

### Q178：Memory-bandwidth-bound 是什么？

**答：** Kernel 主要时间花在搬运数据，DRAM/L2 吞吐接近可达上限，计算单元利用率较低。融合、量化、复用和减少中间写回通常有效。

### Q179：Memory-latency-bound 和 Bandwidth-bound 有何区别？

**答：** Latency-bound 可能带宽不高，但请求少、依赖链长或并行 Warp 不足，无法隐藏单次访问延迟；Bandwidth-bound 则已有大量请求并接近持续吞吐上限。

### Q180：Launch-bound 是什么？

**答：** Kernel 很短且数量很多，Host Launch、Driver 调度和 Kernel 间空隙占据显著时间。Kernel Fusion、CUDA Graph 和减少同步比优化单个 Kernel 算术更有效。

### Q181：Roofline Model 怎么用？

**答：** 根据 Arithmetic Intensity 与硬件计算峰值、内存带宽的交点估计性能上界。实测远低于两条 Roof 时，可能还有 Occupancy、依赖、同步或指令效率问题。

### Q182：理论 FLOPs 怎么计算？

**答：** 例如 GEMM `C[M,N]=A[M,K]B[K,N]` 通常按 `2MNK` FLOPs 计数，因为一次乘加算两次浮点操作。必须说明是否包括 Bias、Activation 和稀疏零跳过。

### Q183：理论带宽和有效带宽怎么区分？

**答：** 理论带宽来自硬件频率与总线，实测 DRAM 带宽来自硬件计数器；应用有效带宽按算法必要字节除时间计算。非合并访问可能使 DRAM 很忙而有效带宽很低。

### Q184：什么是 Achieved Occupancy？

**答：** 它根据运行时活跃 Warp 采样得到，可能低于静态理论 Occupancy。短 Kernel、Tail、Barrier 和负载不均都会影响实际值。

### Q185：什么是 Theoretical Occupancy？

**答：** 根据 Block Size、寄存器、Shared Memory 和硬件上限计算每 SM 可驻留 Warp 比例。它是资源可行性估算，不描述 Warp 是否 Ready 或执行单元是否忙。

### Q186：如何计算理论 Occupancy？

**答：** 先分别计算线程/Warps、寄存器、Shared Memory、最大 CTA 对 Resident CTA 的限制，取最小值，再换算 Resident Warp/最大 Warp。实际还要考虑资源分配粒度。

### Q187：Occupancy 低时第一步做什么？

**答：** 先确认它是否真的限制性能，再查限制来源是寄存器、Shared Memory 还是 Block Size。盲目追求更高 Occupancy 可能牺牲 ILP 和数据复用。

### Q188：什么是 Register Pressure？

**答：** 活跃值太多使每线程寄存器需求升高，可能降低 Resident Warp 或 Spill。循环展开、大 Thread Tile 和复杂 Epilogue 都会延长寄存器生命周期。

### Q189：`-maxrregcount` 是万能优化吗？

**答：** 不是。它可迫使编译器减少寄存器以提高 Occupancy，但可能产生大量 Spill；应比较限制前后的 Local Memory 事务和执行时间。

### Q190：`__launch_bounds__` 有什么作用？

**答：** 它向编译器声明最大 Block Threads 和期望每 SM 最小 CTA 数，帮助寄存器分配权衡。声明与真实使用不匹配可能降低性能或限制合法 Launch。

### Q191：Loop Unrolling 为什么可能加速？

**答：** 展开减少循环控制并暴露 ILP、常量索引和向量化机会。过度展开会增大代码、寄存器压力和 Instruction Cache 压力。

### Q192：Kernel Fusion 的收益是什么？

**答：** 减少 Kernel Launch，并让中间结果留在寄存器或 Shared Memory，避免 HBM 写回再读。对于 Memory-bound 的逐元素算子收益通常明显。

### Q193：Kernel Fusion 为什么可能变慢？

**答：** 融合后寄存器和 Shared Memory 增加、Occupancy 降低，不同阶段的最佳并行粒度也可能冲突。过大 Kernel 还会增加编译和调度复杂度。

### Q194：Vectorization 的核心收益是什么？

**答：** 主要是减少内存指令和地址计算数量，并匹配硬件宽事务；它不会减少必要数据字节。必须满足对齐、边界和寄存器容量。

### Q195：ILP 是什么？

**答：** Instruction-level Parallelism 指单线程/Warp 中存在多个彼此独立的指令，使硬件可覆盖依赖延迟。较低 Occupancy 也可能通过高 ILP 达到好性能。

### Q196：TLP 和 ILP 的区别是什么？

**答：** TLP 通过更多 Warp/Thread 隐藏延迟，ILP 通过同一执行上下文中的独立操作隐藏延迟。两者都会消耗寄存器和调度资源，需要平衡。

### Q197：Nsight Systems 和 Nsight Compute 的区别是什么？

**答：** NSYS 看全程序时间线、CPU/GPU 交互、Stream、Memcpy 和通信；NCU 深入单个 Kernel 的指令、吞吐、Cache、Occupancy 与 Stall。通常先 NSYS 找热点，再 NCU 分析原因。

### Q198：NCU 的 Warp Stall 指标怎么理解？

**答：** Stall Reason 表示 Scheduler 没有从该 Warp 发射下一指令的原因，如等待数据、Barrier 或执行管线。它是诊断线索，不能看到某项最高就直接套优化。

### Q199：`long scoreboard` 常意味着什么？

**答：** 通常表示等待 Global/Local 等长延迟内存依赖，但还需结合 Memory Throughput、Cache 命中和访问模式确认。提高并行度、预取或减少依赖链可能有效。

### Q200：`short scoreboard` 常意味着什么？

**答：** 常与 Shared Memory、MIO 或较短延迟依赖相关，Bank Conflict 和异步流水同步可能是原因。应查看 Shared Wavefront 与相关 Pipeline 指标。

### Q201：Barrier Stall 高说明什么？

**答：** Warp 经常等待 CTA/Cluster 同步，可能来自工作不均、Barrier 过多或 Producer-Consumer 失衡。删除 Barrier 前必须先保证内存正确性。

### Q202：DRAM Throughput 低说明不是 Memory-bound 吗？

**答：** 不一定。Kernel 可能是 Memory-latency-bound、受 L2/L1/Shared 带宽限制，或非合并请求导致指令等待但总吞吐不高。

### Q203：Tensor Core Utilization 低可能有哪些原因？

**答：** Shape 太小、Tile 不匹配、数据搬运跟不上、K-loop 太短、Pipeline 有空洞或未真正生成 Tensor Core 指令。需要同时检查 SASS 与 Feed Pipeline。

### Q204：如何正确做 CUDA Microbenchmark？

**答：** 固定输入与环境、Warmup、重复多次，用 CUDA Event 测目标 Stream，避免把初始化/JIT 混入，并验证结果正确。必要时锁定时钟并报告 GPU、CUDA 和数据 Shape。

### Q205：为什么只测一次不可靠？

**答：** 首次运行可能含 JIT、Cache Cold Start 和频率爬升，系统还存在温度与并发噪声。应报告中位数或稳定分位数，而不是挑最快一次。

### Q206：为什么优化后要重新验证数值？

**答：** 并行 Reduction 顺序、Fast Math、低精度和融合会改变舍入误差，竞态也可能只在特定输入出现。性能结果必须建立在满足误差标准的正确实现上。

### Q207：`--use_fast_math` 做了什么？

**答：** 它启用更快但精度和 IEEE 语义可能较弱的运算，如近似特殊函数、FTZ 和较激进的除法。是否可用取决于误差容忍，不是无条件优化。

### Q208：如何查看寄存器和 Shared Memory 使用量？

**答：** 编译时用 `-Xptxas -v` 查看寄存器、Shared、Stack/Spill；运行时用 NCU 或 Runtime Function Attributes。模板不同实例可能使用量完全不同。

### Q209：如何查看最终指令？

**答：** 使用 `cuobjdump` 或 `nvdisasm` 查看 Cubin SASS，确认是否生成向量 Load、Tensor Core、异步 Copy 等目标指令。源码或 PTX 只能说明意图。

### Q210：优化 CUDA Kernel 的推荐顺序是什么？

**答：** 先验证正确性，再用 NSYS 找端到端热点，用 NCU 判断瓶颈，然后优先修复数量级问题，如非合并访问、重复 HBM、低并行度和同步空洞，最后才做微小指令优化。

## 八、Reduction、Scan、Softmax 等手写题：Q211-Q235

### Q211：并行 Reduction 的基本思路是什么？

**答：** 每个线程先处理一个或多个元素，再按树形结构逐步合并局部结果，最后每个 Block 输出一个 Partial。大数组通常再用第二个 Kernel 合并 Partial。

### Q212：为什么最朴素的交错 Reduction 效率低？

**答：** `tid % (2*s) == 0` 会造成 Warp Divergence，活跃线程分散，并可能引入不友好的 Shared Memory 访问。顺序寻址的折半归约更规整。

### Q213：为什么 Reduction 每轮都可能需要 `__syncthreads()`？

**答：** 下一轮读取的是上一轮由其他线程写入的 Shared Memory，必须保证写入完成且可见。进入单个 Warp 后可改用带 Mask 的 Shuffle，减少块级 Barrier。

### Q214：Warp Reduction 如何用 Shuffle 实现？

**答：** 每轮用 `__shfl_down_sync(mask, value, offset)` 读取高 Lane 的值并相加，Offset 依次为 16、8、4、2、1。只有参与 Lane 有效，Mask 必须正确。

### Q215：为什么 Grid 级 Reduction 常用两阶段 Kernel？

**答：** 普通 Grid 没有廉价全局 Barrier，第一阶段每 Block 输出 Partial，Kernel 边界提供全局同步，第二阶段再归约。它简单、可移植且避免高竞争原子。

### Q216：什么时候全局 `atomicAdd` Reduction 反而合适？

**答：** Block 数较少、硬件原子吞吐高或结果维度较多使竞争分散时，一次 Block-level Atomic 可能比第二次 Launch 更便宜。应与两阶段方案实测。

### Q217：并行浮点 Reduction 为什么与 CPU 结果不完全一致？

**答：** 浮点加法不满足结合律，并行树改变了求和顺序，舍入误差会不同。需要用误差容限、FP32 累加、Pairwise/Kahan 等方法控制误差。

### Q218：Prefix Scan 和 Reduction 的区别是什么？

**答：** Reduction 只输出总聚合值，Scan 输出每个位置之前的前缀结果。Scan 需要 Up-sweep/Down-sweep 或 Warp Scan 等更复杂的数据传播。

### Q219：Hillis-Steele 和 Blelloch Scan 有何区别？

**答：** Hillis-Steele 容易实现但工作量约 `O(n log n)`；Blelloch 是 Work-efficient 的 `O(n)`，通过 Up-sweep 建树、Down-sweep 传播前缀，但同步和索引更复杂。

### Q220：如何做跨 Block Scan？

**答：** 第一阶段对每 Block 扫描并输出 Block Sum，第二阶段扫描这些 Sum，第三阶段把对应 Block Prefix 加回。也可用 Decoupled Look-back 等单 Pass 算法，但同步协议更复杂。

### Q221：Histogram 的主要性能问题是什么？

**答：** 大量线程对少量 Bin 做 Atomic，竞争会高度集中。常见优化是每 Warp/Block 在 Shared Memory 建私有 Histogram，再合并到 Global。

### Q222：Shared Histogram 一定更快吗？

**答：** 不一定。Bin 很多时 Shared Memory 占用大，数据分布均匀且全局原子吞吐足够时私有化收益可能有限；还要考虑最后合并成本。

### Q223：数值稳定 Softmax 为什么先减最大值？

**答：** `exp(x)` 对大正数可能溢出，减去同一行最大值不改变 Softmax 比例，却把最大指数变成 1，使其他指数落在可表示范围。

### Q224：Softmax 通常需要哪些 Reduction？

**答：** 先求行最大值，再求 `sum(exp(x-max))`，最后归一化。高性能实现会尽量把行数据和中间值留在寄存器/Shared Memory，减少 HBM 往返。

### Q225：Online Softmax 是什么？

**答：** 它分块读取数据并维护运行最大值 `m` 与归一化和 `l`；遇到更大最大值时，用 `exp(m_old-m_new)` 修正旧和。这样无需先保存整行 Logit。

### Q226：两个 Online Softmax 分块如何合并？

**答：** `m=max(m1,m2)`，`l=exp(m1-m)l1+exp(m2-m)l2`。Attention 还要用相同比例合并加权输出 `o`，最后计算 `o/l`。

### Q227：LayerNorm 的均值方差怎样并行计算？

**答：** 可分别归约 `sum(x)` 和 `sum(x²)`，但数值稳定性一般；Welford 在线算法维护 Count、Mean 和 M2，更适合大范围或低精度输入。

### Q228：RMSNorm 为什么通常比 LayerNorm 简单？

**答：** RMSNorm 只需要均方根，不需要减均值，因此少一次统计量和部分算术。两者仍需读取整行并做 Reduction。

### Q229：矩阵转置手写题的核心优化点是什么？

**答：** Global Read/Write 都要合并，使用 Shared Tile 改变访问方向，并通过 Padding/Swizzle 避免 Shared Bank Conflict。边界 Tile 需要 Mask。

### Q230：Stencil Kernel 为什么需要 Halo？

**答：** 输出 Tile 的边缘依赖相邻输入，加载 Shared Tile 时要额外加载周围 Halo。Halo 会造成 CTA 间重复读取，但换来 Tile 内邻域复用。

### Q231：卷积为什么可转成 GEMM？

**答：** im2col 将每个滑动窗口展平为矩阵行，与卷积核矩阵相乘。它复用成熟 GEMM，但显式 im2col 会膨胀内存，因此高性能库通常隐式生成 Tile。

### Q232：如何手写 Vector Add？

**答：** 每线程处理一个或 Grid-stride Loop 中多个元素，保证相邻线程访问相邻地址并做边界检查。这个题主要考索引、Launch 配置和错误检查。

### Q233：什么是 Grid-stride Loop？

**答：** 线程从全局 ID 开始，每次增加 `blockDim.x * gridDim.x` 处理后续元素。它允许用固定 Grid 覆盖任意长度，也适合 Persistent 风格与调试。

### Q234：Grid-stride Loop 会破坏合并访问吗？

**答：** 通常不会，因为同一轮相邻 Lane 仍访问相邻元素；下一轮整体平移一个 Grid Stride。关键是 Stride 公式和元素布局正确。

### Q235：手写 Kernel 面试时先保证什么？

**答：** 先说明输入 Shape、Layout、数据类型、边界和正确性，再给并行映射，最后讨论合并访问、复用、同步与资源。直接写高度优化代码却不解释约束通常风险更大。

## 九、GEMM 与 Tensor Core：Q236-Q260

### Q236：朴素 GEMM 为什么慢？

**答：** 每个输出元素独立从 Global Memory 重复读取整行 A 和整列 B，数据复用差，B 的列访问还可能不合并。大部分时间耗在冗余访存。

### Q237：Tiled GEMM 如何减少 Global Memory 访问？

**答：** CTA 把 A/B 子块协作加载到 Shared Memory，每个元素在 Tile 内被多个线程重复用于 FMA。Global 读取次数相对朴素实现显著减少。

### Q238：GEMM 的 CTA Tile、Warp Tile 和 Thread Tile 分别是什么？

**答：** CTA Tile 是一个 Block 负责的输出区域，继续划分给多个 Warp，Warp Tile 再划分到 Lane/Thread 持有的寄存器 Accumulator。分层 Tiling 对应 GPU 调度和存储层级。

### Q239：Register Blocking 有什么作用？

**答：** 每个线程在寄存器中累计多个输出元素，使加载的一小段 A/B 被多次复用，提高 ILP 和 Arithmetic Intensity。Thread Tile 太大则增加寄存器压力。

### Q240：GEMM 的 K-loop 为什么适合 Pipeline？

**答：** 每个 K Tile 都重复“加载下一块、计算当前块”，依赖结构规则，可通过 Double/Multi-stage Buffer 重叠 TMA/Copy 与 MMA。

### Q241：Tensor Core 计算什么？

**答：** Tensor Core 执行小矩阵乘加 `D=A×B+C`，吞吐远高于标量 Core，但对数据类型、布局、对齐和 Tile Shape 有要求。

### Q242：WMMA API 是什么？

**答：** `nvcuda::wmma` 是较高层的 Warp Matrix API，用 Fragment 表示输入和 Accumulator，便于使用 Tensor Core。它支持范围有限，高性能库更多使用 CUTLASS/CuTe 或底层 MMA 指令。

### Q243：`mma.sync` 是什么？

**答：** 它是 Ampere 等架构常用的 Warp-level Tensor Core PTX 指令，一整个 Warp 协同提供 Fragment 并同步得到 Accumulator。

### Q244：WGMMA 是什么？

**答：** Hopper 的 Warpgroup MMA 由 4 个 Warp，也就是 128 线程协作发起异步矩阵乘，输入主要来自 Shared Memory，适合与 TMA 和 Warp Specialization 组成流水。

### Q245：UMMA/`tcgen05.mma` 是什么？

**答：** Blackwell 第五代 Tensor Core 指令可由单线程发起，Accumulator 位于 TMEM，并支持 1-CTA 或 2-CTA 协作及 Block-scaled 低精度类型。

### Q246：为什么 Tensor Core 常使用低精度输入和 FP32 累加？

**答：** 低精度减少存储与带宽并提高乘法吞吐，较高精度 Accumulator 降低长 K Reduction 的误差。输出最终可再转换为 BF16/FP16/FP8。

### Q247：TF32 是什么？

**答：** TF32 使用 FP32 的指数范围和较短尾数，在 Ampere Tensor Core 上加速部分 FP32 GEMM。它不是独立的 C++ 存储类型，是否使用由库和 Math Mode 控制。

### Q248：FP8 的 E4M3 和 E5M2 有何差异？

**答：** E4M3 尾数更多、精度较好但范围较小；E5M2 指数更多、范围更大但精度更低。训练中常按前向、反向张量动态范围选择并配合 Scale。

### Q249：Block Scaling 是什么？

**答：** 将张量分块，每块共享一个 Scale，使低位数据只表示局部动态范围。Scale 粒度越细通常精度越好，但 Metadata、Layout 和 Kernel 成本越高。

### Q250：为什么 Tensor Core Kernel 对 Layout 很敏感？

**答：** MMA 指令要求 Lane 以特定方式提供 Fragment，Shared Memory 还要满足对齐、Swizzle 和 Bank 访问。逻辑 Row-major 相同，不代表物理布局都能高效喂给 Tensor Core。

### Q251：什么是 GEMM Epilogue？

**答：** MMA 完成后将 Accumulator 变换并写回的阶段，可融合 Alpha/Beta、Bias、Activation、Scale、Residual 和类型转换，避免额外读写输出矩阵。

### Q252：Split-K GEMM 是什么？

**答：** 多个 CTA 分别计算 K 维子区间的 Partial `C`，再做归并。它为小 M/N、大 K 增加并行度，但需要 Workspace、原子或串行归并。

### Q253：Serial Split-K 和 Parallel Split-K 有何区别？

**答：** Serial Split-K 通过 Semaphore 让多个 Slice 依次累加同一输出，省 Workspace 但有顺序等待；Parallel Split-K 把 Partial 写到 Workspace 后单独归约，并行度更高但流量更大。

### Q254：Stream-K 是什么？

**答：** Stream-K 把 GEMM 的 K 迭代工作更均匀地分给固定 CTA，而不是让每 CTA 只负责完整输出 Tile，用于减少最后一波负载不均。部分 Tile 需要跨 CTA 归并。

### Q255：Grouped GEMM 是什么？

**答：** 一次 Kernel 处理多个独立、Shape 可不同的 GEMM，常用于 MoE Expert 和小矩阵集合。核心难点是 Problem Metadata、Tile 调度和不规则负载均衡。

### Q256：Grouped GEMM 和 Batched GEMM 有何区别？

**答：** Batched GEMM 通常要求批内 Shape 相同，可通过固定 Stride 定位；Grouped GEMM 允许每组 Shape、Pointer 和 Leading Dimension 不同，调度成本更高但无需 Padding。

### Q257：为什么 MoE 适合 Grouped GEMM？

**答：** 每个 Expert 的 `N/K` 通常相同，但收到的 Token 数 `M_e` 不同。Grouped GEMM 可把这些变长小 GEMM 放进一个任务池，减少 Launch 和 Padding。

### Q258：cuBLAS 和自定义 GEMM Kernel 如何选择？

**答：** 首先把 cuBLAS/cuBLASLt 作为正确性和性能基线；只有特殊 Shape、融合、Layout 或动态调度使通用库不理想时才维护自定义 Kernel。

### Q259：为什么手写 GEMM 很难超过 cuBLAS？

**答：** cuBLAS 有大量架构专用 Kernel、运行时选核、特殊 Shape 策略和长期调优。手写实现只有在范围更窄、融合更多或掌握额外业务约束时更可能胜出。

### Q260：比较 GEMM 性能时必须报告什么？

**答：** 至少报告 GPU、CUDA/库版本、M/N/K、Batch、Layout、Dtype、累加精度、Epilogue、Warmup 和计时方法，并验证数值误差；只给 TFLOPS 没有可比性。

## 十、Ampere、Hopper 与 Blackwell：Q261-Q280

### Q261：Ampere 对 Kernel 开发最重要的变化是什么？

**答：** 代表性变化是 `cp.async` 支持 Global-to-Shared 异步流水，以及更成熟的 Warp-level `mma.sync` Tensor Core 路径。高性能 Kernel 开始更系统地重叠搬运和计算。

### Q262：Ampere 的 `cp.async` 为什么能减少寄存器压力？

**答：** 传统路径常先 Global Load 到寄存器再 Store 到 Shared，`cp.async` 可直接完成 Global-to-Shared 复制，减少中转寄存器和相关指令。

### Q263：Ampere GEMM 的主要执行主体是什么？

**答：** 通常以 Warp 为 MMA 协作主体，Warp 内执行 `mma.sync`，多个 Warp 在 CTA 内协作搬运和计算。Accumulator 主要保存在普通寄存器中。

### Q264：Hopper 对 Kernel 开发最重要的变化是什么？

**答：** Hopper 引入 TMA、WGMMA、Thread Block Cluster 和 Distributed Shared Memory，使执行模型从 Warp-centric 向 Warpgroup 与 Producer-Consumer Warp Specialization 演进。

### Q265：Hopper TMA 能搬运什么？

**答：** TMA 可依据 Tensor Map 在 Global 与 Shared 之间搬运 1D 到多维 Tensor，并支持边界处理、部分变换、Cluster Multicast 和异步完成通知。

### Q266：WGMMA 为什么以 Warpgroup 为单位？

**答：** 更大的 128 线程协作范围能驱动更大 Tensor Core Tile，并与 TMA 大块供数匹配。代价是寄存器、Shared Memory 和 Barrier 设计更加整体化。

### Q267：Hopper Thread Block Cluster 解决什么问题？

**答：** 它让一组 CTA 保证协同调度，可进行 Cluster Barrier、访问 Distributed Shared Memory，并通过 TMA Multicast 共享数据，扩展了单 CTA 的协作范围。

### Q268：Distributed Shared Memory 比本 CTA Shared Memory 更快吗？

**答：** 通常本地 Shared Memory 更低延迟；DSMEM 的价值是让 Cluster 内 CTA 直接共享片上数据，避免绕行 Global Memory。应只在跨 CTA 复用能覆盖远端成本时使用。

### Q269：Hopper Warp Specialization 是什么？

**答：** 不同 Warp 长期承担 TMA Load、WGMMA Compute 或 Epilogue 等角色，通过多 Stage Barrier 流水协作。它让搬运与计算重叠，并可把更多寄存器分给 Consumer Warpgroup。

### Q270：Hopper 为什么广泛使用 FP8？

**答：** Hopper Tensor Core 原生支持 FP8，可降低权重/激活带宽并提高矩阵乘吞吐。E4M3/E5M2 仍需配合 Scale、Amax 和高精度累加维持数值稳定。

### Q271：Blackwell 对 Kernel 开发最重要的变化是什么？

**答：** SM100 引入 `tcgen05/UMMA`、TMEM、1SM/2SM MMA、硬件 Block Scale 和更强的 Persistent 调度能力，使 MMA Issue、Accumulator 和 Epilogue 可以进一步解耦。

### Q272：TMEM 是什么？

**答：** Tensor Memory 是 Blackwell SM 内专门服务 Tensor Core 的片上二维存储，Accumulator 必须位于 TMEM，部分场景也可放 Operand A。它减少普通 Register File 的 Accumulator 压力。

### Q273：`tcgen05.mma` 为什么可以单线程发起？

**答：** 指令通过 Descriptor 描述 Shared/TMEM Operand，由一个选举线程提交异步 Tensor Core 工作；数据协作不再要求所有计算线程共同发出同一 MMA 指令。

### Q274：Blackwell 1-CTA 和 2-CTA MMA 有何区别？

**答：** 1-CTA 由一个 CTA/SM 执行，2-CTA 由 Cluster 中相邻 CTA/SM 共同执行更大 MMA Tile。2-CTA 提高复用和吞吐机会，但要求 Cluster、TMEM 和 Tile 对齐。

### Q275：Cluster Launch Control 是什么？

**答：** CLC 允许正在运行的 CTA/Cluster 取消尚未开始的工作并接管其任务，用于 Persistent Kernel 的硬件辅助 Work Stealing，减少静态 Tile 分配造成的尾部不均。

### Q276：NVFP4 的优化重点是什么？

**答：** NVFP4 使用 4-bit 数据和细粒度 Scale，显著降低权重带宽并提高 Tensor Core 吞吐；Kernel 必须处理 Scale Layout、Block-scaled MMA 和高精度累加，不能当普通 INT4。

### Q277：数据中心 Blackwell SM100 和消费级 SM120 能共用 Kernel 吗？

**答：** 不能默认共用。两者 Compute Capability、Tensor Core 指令、TMEM/TMA 能力和资源不同，应分别编译和分派，SM100 的 `tcgen05` Kernel 不能直接假设可在 SM120 运行。

### Q278：为什么有 `sm_90a`、`sm_100a` 这类架构专用目标？

**答：** 后缀 `a` 表示包含不能向同主版本其他实现保证兼容的架构特性，例如 WGMMA 或特定 `tcgen05` 能力。使用这些指令的 Cubin 必须针对对应 Architecture-specific Target 编译。

### Q279：PTX 能让旧 Kernel 自动利用所有新硬件吗？

**答：** 不能。PTX JIT 可提供功能兼容和部分重新优化，但旧 PTX 没有表达 TMA、WGMMA、TMEM 等新算法结构，想充分利用新架构通常需要重写数据流。

### Q280：从 Ampere Kernel 迁移到 Hopper/Blackwell 应先改什么？

**答：** 先重新设计搬运、MMA 和 Accumulator 数据流，而不是只替换指令名；再确定 Warp/CTA 角色、Pipeline Stage、Cluster 和 Epilogue，最后按目标硬件重新调 Tile。

## 十一、大模型算子与灵活推理题：Q281-Q300

### Q281：LLM Prefill 和 Decode 的算子特征有何不同？

**答：** Prefill 一次处理很多 Query Token，GEMM 和 Attention 计算量大；Decode 每步通常只有一个新 Token，读取大量历史 KV，常受显存带宽和 Launch 延迟限制。

### Q282：标准 Attention 为什么会产生 `O(L²)` 中间量？

**答：** `Q[L,D]K^T[D,L]` 形成 `L×L` Score Matrix，再做 Softmax 和乘 V。直接物化该矩阵会产生巨大 HBM 流量和容量需求。

### Q283：FlashAttention 的核心是什么？

**答：** 将 Q/K/V 分块留在片上，使用 Online Softmax 流式更新最大值、归一化和与输出，避免把完整 Attention Matrix 写入 HBM。它减少 IO，不是近似 Attention。

### Q284：FlashAttention 为什么仍要多次读取 K/V Tile？

**答：** 片上空间放不下全部序列，不同 Q Tile 会依次遍历所需 K/V Tile。优化目标是在容量约束下最大化每次读取的复用，而不是让每个 K/V 只读一次。

### Q285：KV Cache 保存什么？

**答：** 自回归 Decode 保存历史 Token 在每层 Attention 中的 K 和 V，使下一步无需重新计算整个前缀。其容量与层数、上下文、KV Head 数、Head Dim 和数据类型成正比。

### Q286：Paged KV Cache 解决什么问题？

**答：** 将逻辑连续 KV 切成固定大小 Page，通过 Block Table 映射到非连续物理块，减少动态请求长度导致的外部碎片，并支持共享前缀和按需扩容。

### Q287：MQA/GQA 为什么能减少 KV Cache？

**答：** MQA 让所有 Query Head 共享一组 KV Head，GQA 让每组 Query Head 共享 KV；KV Head 数减少后 Cache 和 Decode 读取量按比例下降。

### Q288：Split-KV 是什么？

**答：** 将长 KV 序列切成多个区间，由多个 CTA 并行计算局部 Online Softmax 状态 `(m_s,l_s,o_s)`，第二阶段数值稳定地归并。它为小 Batch、单 Token Decode 增加并行度。

### Q289：为什么 Decode Attention 常是 Memory-bound？

**答：** 一个 Query 只对每个历史 K/V 做少量点积，却必须读取整个 Cache，数据复用低，Arithmetic Intensity 小。上下文越长，KV 读取越容易主导时间。

### Q290：CUDA Graph 为什么能降低 Decode 延迟？

**答：** Decode 每步拓扑重复且包含许多短 Kernel，Graph 一次提交整步依赖，减少 CPU Launch Gap。它不能消除 KV Cache 的 HBM 带宽成本。

### Q291：为什么融合 QKV Preprocess、RoPE 和 Cache Write？

**答：** 这些操作依次处理同一批 Q/K/V 元素，分开会重复读写 HBM并增加 Launch。融合后可在寄存器中完成位置旋转和 Layout 转换，再直接写 Cache。

### Q292：KV Cache 量化的收益和代价是什么？

**答：** FP8/INT8/INT4 Cache 减少容量与 Decode 读取字节，但需要 Scale Metadata、反量化和专用 Attention Kernel，并可能损失精度。收益通常在长上下文更明显。

### Q293：MoE Router 后为什么要 Permute Token？

**答：** 同一 Expert 的 Token 原本分散，把它们按 Expert 排成连续段后，才能形成规则 GEMM 并进行合并访存。输出还需按原 Token 与 Top-K Slot 做 Unpermute/Combine。

### Q294：MoE 的 Grouped GEMM 为什么仍可能很慢？

**答：** 每 Expert Token 数可能很小且极不均衡，导致窄 GEMM、尾部浪费、权重读取和 Expert Parallel All-to-All 成为瓶颈。单次 Launch 只解决一部分问题。

### Q295：Persistent CTA 为什么适合 Attention 和 MoE？

**答：** 它把不同请求、Head、KV Split 或 Expert Tile 放入统一任务池，完成短任务的 CTA 可继续领取工作，减少多次小 Kernel 的 SM 空闲与不规则长尾。

### Q296：拿到一个很慢的未知 Kernel，如何分析？

**答：** 先确认结果与 Shape，NSYS 判断是 Launch、Copy 还是 Kernel，NCU 再判断 Compute、Bandwidth、Latency、同步和资源限制；最后针对最大瓶颈提出可验证的单变量改动。

### Q297：Kernel 结果偶尔错误，优先怀疑什么？

**答：** 优先检查越界、未初始化、缺 Barrier、错误 Stream 依赖、Data Race 和异步生命周期，再检查低精度误差。使用 Compute Sanitizer，并把错误检查放到正确同步点。

### Q298：CUDA P2P 和 NCCL 有何区别？

**答：** CUDA P2P 提供 GPU 间直接内存访问与 Copy 的底层能力；NCCL 在其上结合 NVLink、PCIe 和网络实现 AllReduce、AllGather、All-to-All 等集体通信并进行拓扑优化。

### Q299：GPUDirect RDMA 解决什么问题？

**答：** 它允许支持的网卡直接 DMA 访问 GPU Memory，避免网络数据经 CPU Buffer 额外复制。端到端性能仍依赖 PCIe/NVLink 拓扑、注册内存、消息大小和通信协议。

### Q300：面试中回答 CUDA 优化题的通用框架是什么？

**答：** 先定义 Shape、Layout、Dtype 和正确性，再说明线程/Tile 映射，计算 FLOPs 与必要字节判断瓶颈，分析合并访问、复用、同步和 Occupancy，最后给出基线、Profile 指标及优化取舍。

## 十二、硬件数值与资源计算：Q301-Q340

这一节的 `KB/K` 按 NVIDIA 官方表格表示 1024。除非问题明确指定 A100、H100 或 B200，否则应先说明数值依赖 Compute Capability。

### Q301：当前 NVIDIA CUDA Warp 有多少线程？

**答：** 32 个。`warpSize` 在当前主流 Compute Capability 中为 32，写代码时仍建议使用内建变量而不是把 32 散落在索引公式中。

### Q302：一个 Thread Block 最多有多少线程？

**答：** 当前表中上限为 1024。三维 Block 的 `x*y*z` 乘积不能超过 1024，不是每个维度都取到最大后还能同时成立。

### Q303：Thread Block 三个维度的上限是多少？

**答：** 当前表中 `x<=1024`、`y<=1024`、`z<=64`，同时总线程数不超过 1024。例如 `(32,32,1)` 合法，`(1024,1024,1)` 不合法。

### Q304：Grid 三个维度的上限是多少？

**答：** 当前表中 `grid.x<=2^31-1`，`grid.y/grid.z<=65535`。实际程序仍受任务规模、Launch 参数类型和总执行时间约束。

### Q305：一个 Device 最多可有多少个 Resident Grid？

**答：** 当前 Compute Capability 表给出的 Concurrent Resident Grid 上限为 128。它只是 Grid 数量上限，实际并发还受每个 Kernel 的 SM 资源占用限制。

### Q306：A100、H100 和 B200 每个 SM 最多驻留多少个 CTA？

**答：** 对应 SM80、SM90、SM100 的架构上限都是 32 个 Resident CTA/SM。只有 CTA 极小且其他资源足够时才能接近该数值。

### Q307：A100、H100 和 B200 每个 SM 最多驻留多少个 Warp？

**答：** 都是 64 个 Warp/SM。由于一个 Warp 32 线程，这与 2048 个 Resident Thread/SM 的线程上限一致。

### Q308：A100、H100 和 B200 每个 SM 最多驻留多少个线程？

**答：** 都是 2048 个 Resident Thread/SM。它是架构上限，寄存器、Shared Memory 和 CTA 数限制可能使实际线程数更少。

### Q309：A100、H100 和 B200 每个 SM 有多少寄存器？

**答：** 都是 64K 个 32-bit Register，也就是 65,536 个 32-bit 槽位。这里的 K 表示 1024，不是 64,000。

### Q310：一个 CTA 最多能分配多少个 32-bit Register？

**答：** 当前表中是 64K 个，与每 SM Register File 总槽位相同。通常还会先受到每线程 255 个寄存器或分配粒度限制。

### Q311：一个线程最多能使用多少个 Register？

**答：** 当前表中最多 255 个 32-bit Register。接近该上限通常会严重压缩 Occupancy，并可能因编译器决策产生 Spill。

### Q312：一个 64-bit 变量占几个 Register？

**答：** 寄存器统计以 32-bit 槽位计，一个 64-bit 标量通常占两个槽位；更宽向量按组成元素继续展开。最终数量以 `ptxas -v` 为准。

### Q313：256 线程、每线程 64 个寄存器，寄存器最多允许几个 CTA 驻留？

**答：** 每 CTA 理论使用 `256*64=16,384` 个槽位，`65,536/16,384=4`，所以寄存器维度最多允许 4 个 CTA/SM。实际还要取其他资源限制的最小值。

### Q314：256 线程、每线程 96 个寄存器，寄存器最多允许几个 CTA 驻留？

**答：** 每 CTA 需要 `256*96=24,576` 个槽位，`floor(65,536/24,576)=2`，第三个 CTA 放不下；实际分配粒度还可能进一步收紧。

### Q315：A100 每个 SM 有多少 Shared Memory？

**答：** SM80 A100 最多 164 KiB Shared Memory/SM，可通过 Carveout 在多个容量档位间配置。它与统一 L1/Shared 资源相关。

### Q316：A100 单个 CTA 最多可寻址多少 Shared Memory？

**答：** 163 KiB。SM 有 164 KiB，但 CUDA 为每个 CTA 保留 1 KiB，因此单 CTA 上限少 1 KiB。

### Q317：H100 与 B200 每个 SM 有多少 Shared Memory？

**答：** SM90 H100 和 SM100 B200 都最多为 228 KiB Shared Memory/SM。两者还支持 0、8、16、32、64、100、132、164、196、228 KiB 等 Carveout 档位。

### Q318：H100 与 B200 单个 CTA 最多可寻址多少 Shared Memory？

**答：** 227 KiB。原因同样是每 CTA 保留 1 KiB；超过默认阈值还需要显式 Opt-in。

### Q319：为什么静态 Shared Memory 常看到 48 KiB 限制？

**答：** 为保持架构兼容，静态 Shared Allocation 仍限制在 48 KiB；超过 48 KiB 通常要使用 Dynamic Shared Memory，并通过 `cudaFuncSetAttribute` 显式允许更大的每 CTA Dynamic Shared 上限。

### Q320：SM86 与 A100 的资源数字相同吗？

**答：** 不同。SM86 最大 48 Warp、1536 Thread、16 CTA/SM，Shared Memory 为 100 KiB/SM、99 KiB/CTA；虽然 Register File 同样是 64K，Occupancy 上限不同。

### Q321：Constant Memory 总容量多大？

**答：** CUDA 对静态 Constant Memory 给出的容量是 64 KiB，每 SM 的 Constant Cache Working Set 为 8 KiB。容量 64 KiB 不代表所有数据都能同时命中 Cache。

### Q322：每线程 Local Memory 的架构上限是多少？

**答：** 当前表中是 512 KiB/Thread。它是地址空间/资源上限，不是性能目标；Local Memory 位于 Device Memory，使用过多通常非常慢。

### Q323：Shared Memory 有多少个 Bank？

**答：** 当前表中为 32 个 Bank。对常见 32-bit 访问，连续 32-bit Word 依次映射到 32 个 Bank。

### Q324：最坏可能出现多少路 Shared Memory Bank Conflict？

**答：** 一个 32-Lane Warp 在访问同一 Bank 的 32 个不同地址时，可形成 32-way Conflict。若读取同一个地址则通常走 Broadcast，不按 32-way Conflict 处理。

### Q325：一个 Warp 连续读取 32 个 `float`，请求多少有效字节？

**答：** `32*4=128` 字节。按现代 GPU 常用的 32-byte Sector 观察，若起始地址对齐且不跨额外边界，最少覆盖 4 个 Sector。

### Q326：一个 Warp 连续读取 32 个 `half`，请求多少有效字节？

**答：** `32*2=64` 字节，对齐时最少覆盖 2 个 32-byte Sector。若地址错位跨边界，Sector 数会增加。

### Q327：一个 Warp 每个 Lane 读取一个 `float4`，请求多少有效字节？

**答：** 每 Lane 16 字节，总计 `32*16=512` 字节，对齐时对应 16 个 32-byte Sector。向量化减少指令数，不会减少这 512 字节数据本身。

### Q328：256 线程的 CTA 包含多少 Warp？

**答：** `256/32=8` 个 Warp。若目标 SM 最多 64 Warp，仅从 Warp 数看最多可驻留 8 个这种 CTA。

### Q329：1024 线程的 CTA 包含多少 Warp？线程上限允许每 SM 放几个？

**答：** 包含 `1024/32=32` 个 Warp；在 2048 Thread/SM 的 SM80/90/100 上，仅从线程数看最多 2 个 CTA，寄存器和 Shared Memory 还可能把它降到 1。

### Q330：H100 上每 CTA 使用 64 KiB Shared Memory，Shared 维度最多放几个 CTA？

**答：** `floor(228/64)=3`，因为 3 个使用 192 KiB，4 个需要 256 KiB。实际还要检查寄存器、线程和分配粒度。

### Q331：H100 上每 CTA 使用 120 KiB Shared Memory，Shared 维度最多放几个 CTA？

**答：** `floor(228/120)=1`。即使线程和寄存器很少，也无法同时驻留第二个 120 KiB CTA。

### Q332：256 线程、每线程 64 寄存器时，仅按寄存器计算的理论 Occupancy 是多少？

**答：** 最多 4 个 CTA，每 CTA 8 Warp，共 32 Warp；相对 64 Warp 上限是 `32/64=50%`。这是寄存器单项结果，不是最终 Occupancy。

### Q333：H100 上 256 线程、每 CTA 96 KiB Shared Memory时，仅按 Shared 计算的 Occupancy 是多少？

**答：** 最多 2 个 CTA，共 `2*8=16` 个 Warp，因此相对 64 Warp 上限为 25%。若寄存器只允许 1 个 CTA，最终还会更低。

### Q334：为什么 32 线程小 CTA 在 A100 上最多只有 50% Warp Occupancy？

**答：** 每 CTA 只有 1 个 Warp，而 A100 最多 32 CTA/SM，因此最多 32 Warp；相对 64 Warp 上限正好 50%，此时限制项是 CTA 数而不是线程数。

### Q335：64 线程 CTA 在 A100 上理论上能达到多少 Warp Occupancy？

**答：** 每 CTA 2 Warp，最多 32 CTA 可形成 64 Warp，也就是 100%；前提是寄存器和 Shared Memory 足够。

### Q336：100 万个元素、Block Size 256，需要启动多少个 CTA？

**答：** `ceil(1,000,000/256)=3907` 个 CTA，因为 `(1,000,000+255)/256=3907`。最后一个 CTA 只有部分线程有效，需要边界判断。

### Q337：一个 `4096×4096×4096` GEMM 有多少 FLOPs？

**答：** 按一次乘加 2 FLOPs，计算量是 `2*4096^3=137,438,953,472` FLOPs，约 137.4 GFLOPs。用它除以秒数即可得到有效 FLOP/s。

### Q338：32 层、8 个 KV Head、Head Dim 128、BF16、8K Context 的 KV Cache 多大？

**答：** 每层每 Token 为 `2(K/V)*8*128*2=4096` 字节；32 层是 128 KiB/Token；乘 8192 Token 得 1 GiB，尚未包含对齐、Page Metadata 和 Batch。

### Q339：Blackwell SM100 的 TMEM 逻辑容量是多少？

**答：** 每个 CTA 可见的 TMEM 地址空间为 128 行、512 列、每格 32 bit，即 `128*512*4=262,144` 字节，也就是 256 KiB。实际物理分配按列进行，并受布局、分配粒度和 CTA Group 规则约束。

### Q340：Thread Block Cluster 最大包含多少 CTA？

**答：** 可移植 Cluster Size 最大为 8；B200 可通过 `cudaFuncAttributeNonPortableClusterSizeAllowed` Opt-in 到 16。更大 Cluster 会减少可同时激活的 Cluster 数。

## 十三、参考资料

1. [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
2. [CUDA SIMT Kernel 与线程、内存基础](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
3. [CUDA Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
4. [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
5. [CUDA C++ Memory Model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)
6. [CUDA Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
7. [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
8. [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
9. [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
10. [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)
11. [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
12. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
13. [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
14. [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
15. [CUTLASS Blackwell SM100 GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
16. [CUTLASS `tcgen05` Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
17. [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)
18. [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
19. [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
20. [Compute Capability 资源上限表](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
21. [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
