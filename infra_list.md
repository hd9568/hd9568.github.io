🟢 第一阶段：系统编程与底层基石 (C++ & OS)
AI Infra 的底层语言几乎都是 C++。面试官考察 C++ 不仅是语法，更是你对内存和性能的嗅觉。
1. 现代 C++ 特性与内存管理
[ ] 指针与引用：指针与引用的底层区别、悬垂指针与野指针。
[ ] 智能指针：std::unique_ptr, std::shared_ptr, std::weak_ptr 的底层实现（引用计数）、循环引用问题及解决。
[ ] 内存管理：new/delete 与 malloc/free 的区别、内存泄漏排查（如 Valgrind, ASAN）、内存池（Memory Pool）设计思想。
[ ] 现代 C++ 核心机制：右值引用（Rvalue reference）、移动语义（std::move）、完美转发（std::forward）。
[ ] 虚函数与多态：虚函数表（vptr 和 vtbl）的工作机制、纯虚函数、动态绑定与静态绑定。
[ ] 模板编程：模板特化与偏特化、SFINAE（替换失败并非错误）原则、简单的元编程（面试偶有考察）。
[ ] STL 容器底层：std::vector 的扩容机制、std::unordered_map 的哈希冲突解决与负载因子、std::map（红黑树）。
2. 操作系统与多线程并发
[ ] 进程与线程：进程间通信（IPC）方式、线程上下文切换的开销来源。
[ ] 并发控制：互斥锁（Mutex）、读写锁、自旋锁（Spinlock）的使用场景与性能差异。
[ ] 并发原语：条件变量（Condition Variable）、信号量（Semaphore）、死锁的产生条件及预防。
[ ] 内存层级（Memory Hierarchy）：CPU Cache 行大小（Cache Line）、伪共享（False Sharing）问题及避免方法。
[ ] C++11 并发：std::thread, std::atomic（内存序 Memory Order 基础：Relaxed, Acquire-Release, Sequential Consistency）。
🟡 第二阶段：异构计算与 CUDA 核心优化
CUDA 是 AI Infra 工程师的看家本领。面试官会直接让你手撕常见的算子（如 Reduce, Softmax, GEMM 的简易版）。
1. CUDA 基础架构与执行模型
[ ] 硬件架构：SM（Streaming Multiprocessor）、Warp（线程束）、CUDA Core 与 Tensor Core 的区别。
[ ] 线程模型：Grid, Block, Thread 的三维映射，如何计算 Global Thread ID。
[ ] Warp 机制：Warp Divergence（分支分化）的原因与避免、Active Mask、Warp 级原语（__shfl_sync 等）。
2. CUDA 内存模型与访存优化 (极其高频)
[ ] Global Memory（全局内存）：合并访问（Coalesced Memory Access）的原理与对齐要求。
[ ] Shared Memory（共享内存）：Bank Conflict（银行冲突）的产生原理与解决（如 Padding 填充策略）。
[ ] Registers（寄存器）：寄存器溢出（Register Spilling）对性能的影响（Local Memory 访问）。
[ ] Constant Memory 与 Texture Memory：适用场景（如只读广播数据）。
3. CUDA 进阶性能优化策略
[ ] 流与并发（CUDA Streams）：异步执行、Stream 之间的同步（Event）、Overlapping 计算与数据传输（Ping-Pong Buffer）。
[ ] 循环展开（Loop Unrolling）：#pragma unroll 的原理。
[ ] 算子融合（Kernel Fusion）：为什么融合可以加速（减少显存带宽瓶颈，即 Memory-bound 变 Compute-bound）。
[ ] Profiling 工具：Nsight Compute (NCU) 和 Nsight Systems (NSYS) 的常用指标（Occupancy, Compute Workload, Memory Bandwidth）。
4. 高频手撕算子 (必须熟练默写)
[ ] 手写 Vector Add / Vector Dot Product。
[ ] 手写 Parallel Reduction（交错寻址、连续寻址、Warp 级别规约、完全展开）。
[ ] 手写 矩阵乘法 (GEMM)（朴素版 -> Shared Memory Tiling 版 -> 解决 Bank Conflict 版）。
[ ] 手写 Softmax（Online Softmax 算法设计以防止数值溢出及减少访存）。
[ ] 手写 LayerNorm / RMSNorm（均值与方差的高效计算）。
🟠 第三阶段：大模型推理优化 (LLM Inference)
随着大模型爆发，推理基建是目前最缺人的方向，vLLM 和 TensorRT-LLM 是必考背景。
1. 注意力机制优化 (Attention)
[ ] 标准 Attention 的复杂度：时间与空间复杂度分析。
[ ] FlashAttention-1：Tiling 思想、SRAM 与 HBM 的内存层级利用、Forward/Backward 重计算逻辑。
[ ] FlashAttention-2 / 3：对非矩阵乘法（Non-matmul）操作的优化、Warp 切分策略改进。
[ ] PagedAttention：解决 KV Cache 内存碎片化、逻辑块与物理块映射（OS 虚拟内存分页思想的迁移）。
2. 推理调度与系统优化
[ ] KV Cache 机制：为什么需要 KV Cache？它如何将 Generate 阶段变成 Memory-bound 问题。
[ ] Batching 策略：Static Batching vs Dynamic/Continuous Batching（vLLM 核心，In-flight batching）。
[ ] Decode 阶段优化：Prefill（Context phase）与 Decode（Generation phase）的计算特性差异。
[ ] 投机解码（Speculative Decoding）：Draft Model 与 Target Model 的协作、Acceptance rate 分析、加速原理。
3. 推理框架与编译图优化
[ ] TensorRT 基础：Builder, Network Definition, Engine, Execution Context 的工作流。
[ ] TRT 图优化：算子融合（Conv+BN+Relu）、精度转换、内核自动调优（Auto-tuning）。
[ ] 自定义算子接入：编写 TensorRT Plugin。
[ ] ONNX：ONNX 静态图结构、使用 ONNX GraphSurgeon 进行图编辑。
🔴 第四阶段：分布式训练优化 (Distributed Training)
对于千亿参数模型，单卡无法容纳，分布式切分和通信是关键。
1. 通信原语 (NCCL / MPI)
[ ] 基础通信逻辑：Broadcast, Scatter, Gather。
[ ] 核心聚合逻辑：Reduce, All-Reduce, Reduce-Scatter, All-Gather 的数据流向与应用场景。
[ ] 环形拓扑通信（Ring-AllReduce）：通信量和时间的理论计算，为什么它是带宽最优的。
2. 并行切分策略 (Parallelism)
[ ] 数据并行 (DP, Data Parallel) 与 DDP (Distributed DP) 的通信重叠机制。
[ ] 张量并行 (TP, Tensor Parallel)：Megatron-LM 中的 1D TP（MLP 和 Attention 如何切分、在哪里插入 All-Reduce 通信）。
[ ] 流水线并行 (PP, Pipeline Parallel)：GPipe, PipeDream, Bubble（气泡）产生的原理及计算，Micro-batch 的调度。
[ ] 序列并行 (SP, Sequence Parallel)：解决长上下文 (Long Context) 训练的内存瓶颈。
3. 显存优化技术
[ ] 混合精度训练 (AMP)：FP16/BF16 与 FP32 的配合、Loss Scaling 的作用、BF16 相比 FP16 的优势。
[ ] 激活重计算 (Activation Recomputation/Checkpointing)：时间换空间，前向丢弃、反向重算的机制。
[ ] 零冗余优化器 (DeepSpeed ZeRO)：ZeRO-1（优化器状态切分）、ZeRO-2（梯度切分）、ZeRO-3（参数切分）的区别与通信开销。
🟣 第五阶段：量化与低精度基建 (Quantization)
大模型部署离不开量化，这一块既考察算法理解，也考察底层硬件指令（如 INT8/INT4 Tensor Cores）。
1. 量化基础理论
[ ] 映射机制：非对称量化 (Asymmetric) 与 对称量化 (Symmetric) 的公式推导（Scale 和 Zero-point）。
[ ] 校准准则：Min-Max, KL 散度 (Kullback-Leibler Divergence), Percentile 校准。
[ ] 训练与量化结合：PTQ (Post-Training Quantization) 与 QAT (Quantization-Aware Training) 的区别。
2. 大模型前沿量化算法
[ ] Weight-Only 量化原理（如 W4A16, W8A16）：为什么 LLM 主要是 Memory-bound？Weight-Only 如何提升吞吐。
[ ] SmoothQuant：解决 Activation 异常值（Outliers）的策略（权重与激活间的难度转移）。
[ ] AWQ (Activation-aware Weight Quantization)：基于保留显著权重（Salient weights）的量化。
[ ] GPTQ：基于近似二阶信息的权重逐层量化误差补偿机制。
3. 底层量化执行
[ ] 反量化操作 (Dequantization) 发生的位置：为什么 W4A16 在进行 GEMM 前通常要在 Shared Memory/Register 层面反量化为 FP16。
[ ] CUTLASS 库基础：如何利用 CUTLASS 进行高效的低比特矩阵乘。
