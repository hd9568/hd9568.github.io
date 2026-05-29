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
[ ] Shared Memory（共享内存）：Bank Conflict（Bank 冲突）的产生原理与解决（如 Padding 填充策略、 Swizzle 技术）。
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
随着大模型爆发，推理基建是目前最缺人的方向，vLLM、TensorRT-LLM、SGLang、Triton、FlashAttention、PagedAttention、MoE Serving 都是高频背景。
1. Attention 基础与经典优化
[ ] 标准 Attention 的复杂度：时间复杂度 O(n^2 d)、显存复杂度 O(n^2)，为什么长上下文会迅速变慢。
[ ] Causal Attention：自回归推理中的 Mask 机制，为什么 Decode 阶段只能看历史 token。
[ ] FlashAttention-1：Tiling 思想、SRAM 与 HBM 的内存层级利用、Online Softmax、Forward/Backward 重计算逻辑。
[ ] FlashAttention-2 / 3：减少 Non-matmul 开销、改进 Warp/Block 切分、提升 Tensor Core 利用率。
[ ] FlashDecoding：Decode 阶段 batch 小但上下文长，如何把单请求 KV 读取拆到多个 block 并行。
[ ] PagedAttention：解决 KV Cache 内存碎片化，逻辑块、物理块、Block Table 与 OS 虚拟内存分页思想。
[ ] Attention Mask 优化：Causal Mask、Paged Mask、Block Sparse Mask 的实现成本和 kernel 内处理方式。
2. 新型 Attention 与长上下文模型结构
[ ] MHA / MQA / GQA：多头、多查询、分组查询注意力的 KV Cache 占用差异与吞吐影响。
[ ] MLA（Multi-head Latent Attention）：DeepSeek 类模型中通过低秩 latent 压缩 KV Cache 的思想、收益与实现难点。
[ ] Linear Attention：把 Softmax Attention 近似为线性复杂度的核函数形式，理解特征映射、递推状态和精度风险。
[ ] Sparse Attention：局部窗口、块稀疏、全局 token、滑动窗口注意力在长上下文中的收益和限制。
[ ] DSA / Dynamic Sparse Attention：动态选择重要 token 或重要 block，减少无效 KV 读取；需要关注选择策略、mask 构建和 kernel 适配。
[ ] Sliding Window / Sink Token：长上下文推理中保留局部窗口和少量全局 sink token 的方法，StreamingLLM/Mistral 类思想。
[ ] RoPE Scaling：Position Interpolation、NTK-aware scaling、YaRN 等长上下文扩展方法的动机和副作用。
[ ] SSM / Mamba 类架构推理：状态空间模型如何避免二次 Attention，prefill/decode 时状态缓存与并行扫描问题。
[ ] 混合架构推理：Attention + SSM、Dense + MoE、局部注意力 + 全局注意力混合时的调度和 kernel 选择。
3. 推理阶段拆解与性能指标
[ ] Prefill 与 Decode：Prefill（Context phase）是大矩阵计算、吞吐高；Decode（Generation phase）逐 token 生成、批量小、强依赖 KV Cache。
[ ] Compute-bound vs Memory-bound：Prefill 更容易 Compute-bound，Decode 常常 Memory-bound；如何用 Roofline 思想判断瓶颈。
[ ] TTFT / TPOT / Latency / Throughput：首 token 延迟、单 token 延迟、端到端延迟、tokens/s、QPS 的定义与权衡。
[ ] Tail Latency：P50/P95/P99 的含义，长 prompt、长输出、排队和 KV 不足如何造成长尾。
[ ] 请求生命周期：Tokenizer、排队、Prefill、Decode、Detokenizer、Streaming 返回、请求取消和资源回收。
[ ] Workload 建模：输入长度分布、输出长度分布、并发数、到达率、SLA 对系统容量的影响。
4. KV Cache 与显存管理优化
[ ] KV Cache 机制：为什么缓存历史 K/V？它如何把 Attention 从重复计算变成读取历史 KV 的访存问题。
[ ] KV Cache 显存估算：层数、Batch、SeqLen、Head 数、Head Dim、dtype 对显存占用的影响。
[ ] KV Cache Layout：按 layer、head、token、block 组织的不同布局，对 coalescing、分页和并行读取的影响。
[ ] Paged KV Cache：Block Pool、Block Table、逻辑块到物理块映射、引用计数、碎片回收。
[ ] Prefix Caching：系统提示词、Few-shot Prompt、RAG 前缀复用，如何降低重复 Prefill 成本。
[ ] KV Cache Quantization：FP16/BF16 KV 到 INT8/FP8/INT4 KV 的收益、误差来源和长上下文影响。
[ ] KV Cache Offload：CPU/NVMe Offload、分层缓存、换入换出策略，为什么会影响 TPOT 和尾延迟。
[ ] KV Cache Eviction：请求取消、超时、低优先级抢占、长上下文截断时的缓存释放策略。
[ ] Prefix Sharing 与 Copy-on-write：多请求共享相同前缀时如何避免重复存储和错误写入。
5. Batching、调度与 Serving 系统
[ ] Static Batching：固定 batch 的吞吐收益、padding 浪费和延迟问题。
[ ] Dynamic Batching：按时间窗口聚合请求，如何权衡吞吐和等待时间。
[ ] Continuous / In-flight Batching：Decode 每一步动态加入新请求，vLLM/TensorRT-LLM 的核心调度思想。
[ ] Chunked Prefill：把长 Prefill 切块，避免长 prompt 阻塞 Decode，改善在线服务 tail latency。
[ ] Prefill-Decode Disaggregation：Prefill 节点与 Decode 节点分离部署，适用于长上下文和高并发服务。
[ ] Disaggregated KV Transfer：Prefill 节点生成 KV 后如何传给 Decode 节点，RDMA/NVLink/PCIe 带宽瓶颈。
[ ] 请求调度策略：FCFS、优先级队列、短请求优先、最长等待优先、Deadline-aware scheduling。
[ ] Admission Control：根据剩余 KV Cache、预计输出长度、SLA、优先级决定是否接收新请求。
[ ] 多租户与公平性：隔离、限流、配额、抢占、业务优先级和成本控制。
[ ] Streaming 输出：边生成边返回、客户端取消、反压（Backpressure）和部分结果一致性。
6. 解码算法与减少串行生成
[ ] Greedy / Sampling / Top-k / Top-p：采样策略对吞吐、延迟、重复率和输出质量的影响。
[ ] Beam Search：多候选维护、KV Cache 成倍增长、适用场景和服务成本。
[ ] Speculative Decoding：Draft Model 与 Target Model 协作、Acceptance Rate、验证回滚和加速上限。
[ ] Medusa：在主模型上增加多 token head，减少额外 draft model 成本。
[ ] EAGLE / Lookahead Decoding：利用特征预测未来 token，减少串行 decode 步数。
[ ] Parallel Decoding / Jacobi Decoding：并行猜测多个位置再修正，理解适用条件和收敛问题。
[ ] Early Exit / Layer Skip：部分 token 不跑完整层数的思路、置信度判断和精度风险。
[ ] Token Pruning：生成过程中裁剪低价值上下文 token，降低 KV 读取量和注意力开销。
7. MoE 模型推理优化
[ ] MoE 基础：Router、Top-k Expert、Shared Expert、Sparse FFN，为什么参数量大但每 token 激活参数少。
[ ] Expert 负载均衡：热门 expert、长尾 expert、token 分布不均如何导致 GPU 闲置和尾延迟。
[ ] Expert Parallel：Expert 跨 GPU 放置，All-to-All 通信、token dispatch/combine 的开销。
[ ] MoE Batching：按 expert 聚合 token 做 grouped GEMM，减少小矩阵乘和 kernel launch 开销。
[ ] Router 优化：Top-k 选择、capacity factor、drop token、aux loss 与推理时负载控制。
[ ] Expert Placement：根据访问频率、GPU 拓扑、NVLink/PCIe 带宽做 expert 放置和复制。
[ ] Expert Cache：热门 expert 常驻 GPU，冷门 expert CPU/NVMe offload 的收益和风险。
[ ] DeepSeek-MoE / Mixtral 类模型：Shared Expert + Routed Expert、细粒度 expert、推理 serving 的特殊挑战。
[ ] MoE 与量化：不同 expert 的量化误差、per-expert scale、激活 outlier 和吞吐收益。
8. 算子、Kernel 与 Runtime 优化
[ ] Kernel Fusion：RMSNorm + Quant、Bias + Activation、Residual Add、QKV Projection 后处理等融合如何减少显存读写。
[ ] Decode 小 batch 优化：为什么 Decode 阶段 GEMM 形状细长，如何优化 GEMV/GEMM 和小 batch Tensor Core 利用。
[ ] Grouped GEMM：MoE 和多请求 decode 中多个小 GEMM 合并调度的思想。
[ ] Tensor Core 利用：FP16/BF16/FP8/INT8 下的维度对齐、layout、tile size 对吞吐的影响。
[ ] CUDA Graph：减少 CPU launch overhead，适合固定 shape、shape bucket 或 decode loop capture。
[ ] Memory Pool：GPU 显存池、Pinned Memory Pool、减少 cudaMalloc/cudaFree 同步开销。
[ ] Async Copy / TMA：Ampere/Hopper 中 cp.async、TMA 对 tiled kernel 和 GEMM pipeline 的作用。
[ ] Triton Kernel 优化：block size、num_warps、num_stages、mask、program id、autotune 的选择。
[ ] CUTLASS / cuBLASLt：GEMM epilogue fusion、layout 选择、heuristic search、workspace 和 autotune。
[ ] 算子自动调优：不同 batch、seq、hidden shape 下选择不同 kernel 配置和 fallback 策略。
9. 量化、低精度与压缩推理
[ ] Weight-only Quantization：INT8/INT4 权重量化，为什么能减少显存带宽和模型加载成本。
[ ] SmoothQuant：把 activation outlier 平滑到 weight，降低激活量化难度。
[ ] GPTQ / AWQ：离线权重量化方法的基本思想、校准数据、Hessian/Activation-aware 与精度保持。
[ ] AQLM / SpQR / HQQ 等权重量化：更细粒度压缩方法的思想和工程复杂度。
[ ] Activation Quantization：W8A8、W4A8 的收益、激活动态范围和 outlier 处理。
[ ] KV Cache 量化：INT8/FP8/INT4 KV 对长上下文显存和 Decode 带宽的影响。
[ ] FP8 推理：E4M3/E5M2 的区别、动态 scale、per-tensor/per-channel scale 和 Tensor Core 支持。
[ ] 量化粒度：Per-tensor、Per-channel、Group-wise quantization 的精度与性能权衡。
[ ] 反量化融合：Dequant + GEMM 融合，为什么只减少存储还不够，必须关注反量化开销。
[ ] Sparsity：2:4 结构化稀疏、非结构化稀疏、稀疏 kernel 和真实加速条件。
[ ] LoRA / Adapter 推理：多 LoRA 热切换、LoRA 合并、Punica/S-LoRA 类多租户 LoRA serving 思路。
10. 并行推理与多 GPU 部署
[ ] Tensor Parallel 推理：QKV、MLP、Output Projection 的切分方式，以及 All-Reduce / All-Gather 开销。
[ ] Pipeline Parallel 推理：多层模型跨 GPU 切分、bubble、micro-batch 和在线请求调度问题。
[ ] Expert Parallel / MoE 推理：Router、Top-k expert、All-to-All 通信和负载均衡。
[ ] Context Parallel / Sequence Parallel：长上下文场景下按序列维度切分 KV 或 Attention 计算。
[ ] Data Parallel / Replica：多副本服务的负载均衡、会话粘性、KV locality 和故障恢复。
[ ] Hybrid Parallelism：TP + PP + EP + DP 组合时的拓扑映射和通信瓶颈。
[ ] 通信优化：NVLink、PCIe、NCCL、GPUDirect RDMA、IB 网络对推理吞吐的影响。
[ ] 多机推理：跨节点模型切分、跨机 KV 传输、网络抖动和容灾问题。
11. 推理框架与编译图优化
[ ] vLLM：PagedAttention、Continuous Batching、Block Manager、Prefix Cache、Chunked Prefill 的整体架构。
[ ] SGLang：RadixAttention、前缀树缓存、结构化生成、约束解码和多模态 serving。
[ ] TensorRT-LLM：Builder、Network Definition、Engine、Execution Context、Plugin、KV Cache 管理的工作流。
[ ] TensorRT 图优化：算子融合、精度转换、常量折叠、内核自动调优（Auto-tuning）。
[ ] 自定义算子接入：TensorRT Plugin 的接口、序列化、format 支持和动态 shape 处理。
[ ] Triton Inference Server：模型仓库、ensemble、dynamic batching、backend、metrics。
[ ] Triton Kernel：用 Triton 写自定义推理算子，program id、block pointer、mask、num_warps 等核心概念。
[ ] ONNX：静态图结构、Opset、动态维度、导出坑点、ONNX GraphSurgeon 图编辑。
[ ] torch.compile / Inductor：动态图捕获、图断裂、算子融合、shape guard 与推理部署限制。
[ ] 模型格式与权重加载：HuggingFace safetensors、权重分片、懒加载、CPU offload、GPU warmup。
[ ] LLM Runtime 对比：vLLM、TensorRT-LLM、SGLang、llama.cpp、MLC-LLM、TGI 的设计取舍。
12. RAG、工具调用与结构化输出推理
[ ] RAG 推理链路：Embedding、向量检索、重排、Prompt 拼接、LLM 生成的端到端延迟组成。
[ ] Prompt 长度控制：RAG chunk 数量、上下文压缩、摘要和 rerank 对 Prefill 成本的影响。
[ ] Structured Output：JSON Schema、Regex/Grammar constrained decoding、token mask 生成开销。
[ ] Function Calling / Tool Use：多轮工具调用带来的调度、缓存复用和超时控制问题。
[ ] 多模态推理：Vision Encoder + LLM、图像 token 数量、prefill 成本和 KV Cache 增长。
13. 推理服务化、可观测性与容量规划
[ ] OpenAI-compatible Server：请求协议、流式输出、并发连接、取消请求和超时处理。
[ ] Scheduler 指标：队列长度、等待时间、running requests、waiting requests、KV Cache 使用率。
[ ] GPU 指标：SM 利用率、显存带宽、HBM 使用量、Tensor Core 利用率、Kernel timeline。
[ ] 端到端指标：TTFT、TPOT、P50/P95/P99 latency、tokens/s、错误率、超时率。
[ ] Capacity Planning：给定模型大小、GPU 显存、上下文长度、并发数，估算最大吞吐和服务容量。
[ ] Benchmark 方法：ShareGPT / LongBench / Synthetic workload，区分离线吞吐测试和在线延迟测试。
[ ] Profiling 方法：NSYS 看时间线，NCU 看 kernel，Prometheus/Grafana 看线上服务指标。
[ ] 稳定性问题：OOM、显存碎片、请求取消后的 KV 回收、长尾请求、模型热更新与灰度发布。
[ ] 成本优化：GPU 利用率、量化收益、模型副本数、Autoscaling、Spot GPU、混部策略。
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
