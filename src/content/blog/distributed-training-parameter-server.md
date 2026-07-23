---
title: '参数服务器详解：从数据并行、稀疏参数到同步一致性'
description: '系统讲解 Parameter Server 的 Worker/Server 架构、参数分片、Push/Pull、同步与异步训练、稀疏 Embedding 更新、容错、热点和工程优化。'
category: '分布式训练'
pubDate: '2026-07-20T12:29:00+08:00'
updatedDate: '2026-07-20T12:29:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [参数服务器解决什么问题](#一参数服务器解决什么问题)
2. [基本架构：Worker、Server 与 Coordinator](#二基本架构workerserver-与-coordinator)
3. [一次训练迭代如何执行](#三一次训练迭代如何执行)
4. [参数如何分片](#四参数如何分片)
5. [同步、异步与有界延迟训练](#五同步异步与有界延迟训练)
6. [为什么推荐系统常用参数服务器](#六为什么推荐系统常用参数服务器)
7. [稠密参数与稀疏参数的不同处理](#七稠密参数与稀疏参数的不同处理)
8. [一致性、陈旧梯度与收敛](#八一致性陈旧梯度与收敛)
9. [通信与性能瓶颈](#九通信与性能瓶颈)
10. [容错与弹性调度](#十容错与弹性调度)
11. [一个简化实现](#十一个简化实现)
12. [TensorFlow ParameterServerStrategy 示例](#十二tensorflow-parameterserverstrategy-示例)
13. [与 AllReduce 数据并行的区别](#十三与-allreduce-数据并行的区别)
14. [工程设计清单](#十四工程设计清单)
15. [总结](#十五总结)

## 一、参数服务器解决什么问题

单机训练时，模型参数、梯度和优化器状态都在一台机器上：

```text
读取 batch
-> 前向计算
-> 反向计算梯度
-> optimizer 更新参数
```

当数据量、模型参数量或训练吞吐需求超过单机能力后，需要把计算和存储分散到多台机器。参数服务器，Parameter Server，简称 PS，是一种经典的分布式训练架构：

```text
Worker 负责计算；
Parameter Server 负责保存和更新参数。
```

它尤其适合以下场景：

- 数据并行训练。
- 参数量远大于单机内存。
- 大规模稀疏 Embedding。
- 每个 batch 只访问少量参数。
- Worker 数量会动态变化。
- 需要异步训练提高吞吐。

参数服务器不是某个固定算法，而是一组系统设计：

```text
参数怎样分片；
Worker 怎样获取参数；
梯度怎样发送；
Server 何时更新；
如何保证一致性；
节点故障后如何恢复。
```

## 二、基本架构：Worker、Server 与 Coordinator

典型 PS 系统包含三类角色。

### Worker

Worker 持有计算图或模型计算逻辑，负责：

- 读取训练数据。
- 拉取当前所需参数。
- 执行 forward。
- 执行 backward。
- 将梯度或参数增量推送给 PS。

Worker 不一定保存完整模型。对大规模稀疏模型，它可以按需拉取当前 batch 用到的 Embedding 行。

### Parameter Server

PS 负责：

- 存储参数分片。
- 存储优化器状态。
- 接收 Worker 的梯度。
- 执行参数更新。
- 向 Worker 返回参数。
- 保存 checkpoint。

如果模型有 `P` 个参数服务器，则参数通常被切成：

```text
θ = θ_0 ∪ θ_1 ∪ ... ∪ θ_{P-1}
```

每个 Server 只管理一个 shard。

### Coordinator / Scheduler

Coordinator 负责控制面工作：

- 注册 Worker 和 PS。
- 分配训练任务。
- 维护集群成员。
- 检测节点故障。
- 调度输入数据。
- 控制 checkpoint、恢复和训练终止。

Coordinator 通常不参与大规模参数数据传输，避免成为数据面瓶颈。

### 拓扑

```text
                  +----------------+
                  |  Coordinator   |
                  +----------------+
                    /      |      \
                   /       |       \
          +---------+  +---------+  +---------+
          | Worker0 |  | Worker1 |  | Worker2 |
          +---------+  +---------+  +---------+
              |\          /|\          /|
              | \        / | \        / |
              v  v      v  v  v      v  v
          +--------+ +--------+ +--------+
          |  PS 0  | |  PS 1  | |  PS 2  |
          +--------+ +--------+ +--------+
```

每个 Worker 可能与多个 PS 通信，因为一次 forward 可能访问多个参数 shard。

## 三、一次训练迭代如何执行

假设模型参数为 `θ`，Worker `i` 处理 mini-batch `B_i`。

### Pull

Worker 向 PS 请求当前计算需要的参数：

```text
θ_i = Pull(parameter_keys)
```

稠密网络可能拉取完整参数；稀疏模型通常只拉取 batch 中出现的特征 ID 对应的 Embedding。

### Forward 与 Backward

Worker 在本地计算：

```text
loss_i = L(B_i; θ_i)
g_i = ∂loss_i / ∂θ_i
```

### Push

Worker 把梯度发送给负责对应参数 shard 的 PS：

```text
Push(parameter_keys, gradients)
```

### Update

PS 根据优化器更新参数。以 SGD 为例：

```text
θ <- θ - ηg
```

以带动量 SGD 为例：

```text
v <- μv + g
θ <- θ - ηv
```

Adam 还要在 PS 上保存一阶矩和二阶矩：

```text
m <- β1 m + (1 - β1) g
v <- β2 v + (1 - β2) g^2
θ <- θ - η * m_hat / (sqrt(v_hat) + ε)
```

因此 PS 保存的不只是模型权重，还有 optimizer states。

## 四、参数如何分片

参数分片的目标是同时平衡：

- 内存占用。
- 网络流量。
- 请求数量。
- 更新计算量。

### Range Sharding

按连续 ID 范围切分：

```text
PS0: [0, 1M)
PS1: [1M, 2M)
PS2: [2M, 3M)
```

优点：

- 地址计算简单。
- 范围扫描方便。

缺点：

- 如果热门 ID 集中在某段范围，容易产生热点。

### Hash Sharding

按 key 哈希：

```text
server_id = hash(key) % num_servers
```

优点：

- 通常能较均匀地分散 key。
- 适合大规模 Embedding table。

缺点：

- 扩缩容时简单取模会让大量 key 迁移。
- 范围查询不方便。

工程中可用一致性哈希或虚拟节点降低扩缩容迁移量。

### Tensor Sharding

对稠密大 Tensor 按维度切分：

```text
W: [M, N]

PS0: W[:, 0:N/2]
PS1: W[:, N/2:N]
```

要注意：

- 分片维度是否与计算访问模式匹配。
- 是否造成大量跨 PS 小请求。
- optimizer state 必须采用相同分片规则。

### 负载不是只看参数字节数

两个 shard 即使大小相同，负载也可能不同：

```text
Shard A: 100 GB，但每秒访问 1 万次
Shard B: 100 GB，但每秒访问 100 万次
```

合理的分片需要同时考虑：

- 参数大小。
- key 访问频率。
- 每次请求字节数。
- 更新频率。
- 网络拓扑。

## 五、同步、异步与有界延迟训练

参数服务器的关键设计之一是 Worker 之间怎样协调。

### BSP：Bulk Synchronous Parallel

同步训练中，所有 Worker 在每一步形成 barrier：

```text
所有 Worker 读取同一版本参数
-> 分别计算梯度
-> PS 等待所有梯度
-> 聚合并更新一次
-> 进入下一步
```

若有 `N` 个 Worker，常用平均梯度：

```text
g = (1/N) * sum_i(g_i)
θ_{t+1} = θ_t - ηg
```

优点：

- 梯度对应同一个参数版本。
- 语义接近单机大 batch。
- 收敛行为容易分析和复现。

缺点：

- 整步速度由最慢 Worker 决定。
- Straggler 会让所有 Worker 等待。
- Worker 故障可能阻塞训练。

### ASP：Asynchronous Parallel

异步训练中，Worker 独立工作：

```text
Worker 拉取参数
-> 计算梯度
-> 立即 Push
-> PS 立即更新
-> Worker 继续下一步
```

不需要全局 barrier。

优点：

- 吞吐高。
- 对慢 Worker 更不敏感。
- 容易支持弹性 Worker。

缺点：

- 梯度可能基于旧参数计算。
- 不同 Worker 看到不同参数版本。
- 训练结果更难复现。
- Worker 数很多时，陈旧梯度可能影响收敛。

### SSP：Stale Synchronous Parallel

SSP 在同步和异步之间折中，允许最快 Worker 最多领先最慢 Worker `s` 步：

```text
max_clock - min_clock <= s
```

当领先超过阈值时，快 Worker 等待。

特点：

- `s = 0` 时接近 BSP。
- `s -> ∞` 时接近 ASP。
- 在吞吐和参数一致性之间提供可调节折中。

### Backup Workers

同步训练还可以启动额外 Worker，只等待最快的 `K` 个梯度：

```text
启动 K + r 个 Worker
每步收到任意 K 个梯度后更新
忽略本步剩余 r 个迟到梯度
```

它能缓解长尾 Straggler，但会增加计算资源消耗。

## 六、为什么推荐系统常用参数服务器

大型推荐模型常同时包含：

```text
稀疏部分：TB 级甚至更大的 Embedding tables
稠密部分：MLP、Attention、交叉网络
```

一次 batch 只会访问少量 Embedding 行。假设：

```text
Embedding table: 10^10 rows
embedding dim: 64
dtype: FP32
```

仅参数大小约为：

```text
10^10 * 64 * 4 bytes = 2.56 TB
```

如果使用 Adam，还需要一阶矩和二阶矩，存储量进一步增加。

但一个 batch 可能只访问几十万个 ID，因此没必要把完整 table 复制到每个 Worker。PS 可以：

```text
按 key 分片存储 Embedding
-> Worker 按需 Pull 当前 batch 的行
-> 只 Push 被访问行的稀疏梯度
```

这使 PS 很适合超大规模稀疏参数。

## 七、稠密参数与稀疏参数的不同处理

### 稀疏参数

Embedding lookup 的梯度通常是 IndexedSlices：

```text
indices = [12, 98, 12, 301]
values  = [g0, g1, g2, g3]
```

只更新出现过的行：

```text
E[12]  <- update(E[12], g0 + g2)
E[98]  <- update(E[98], g1)
E[301] <- update(E[301], g3)
```

需要处理：

- batch 内重复 ID 聚合。
- 热门 key。
- Embedding cache。
- key 的创建和淘汰。
- 稀疏 optimizer state。

### 稠密参数

稠密网络参数每一步通常都会被访问和更新。若全部通过 PS 传输，网络开销可能很大：

```text
每个 Worker 每步 Pull 全量参数
每个 Worker每步 Push 全量梯度
```

因此混合架构常见做法是：

```text
稀疏参数走 Parameter Server；
稠密参数在 GPU Worker 间走 AllReduce。
```

这样既利用 PS 管理超大 Embedding，也利用 NCCL AllReduce 高效同步稠密梯度。

### 优化器也可以分开

稀疏和稠密参数的统计性质不同：

- 稀疏特征更新频率差异很大。
- 稠密参数每步更新。
- 稀疏参数常用 FTRL、AdaGrad 等。
- 稠密参数常用 SGD、Adam、AdamW、RMSProp 等。

因此一个训练系统可以为不同参数组配置不同优化器、学习率和梯度裁剪策略。

## 八、一致性、陈旧梯度与收敛

异步 PS 中，Worker 计算出的梯度可能不是针对当前参数版本。

Worker 在版本 `t` 拉取参数：

```text
θ_t
```

计算期间，其他 Worker 已让 PS 更新到：

```text
θ_{t+k}
```

当前 Worker 提交的是：

```text
g(θ_t)
```

但梯度被应用在：

```text
θ_{t+k}
```

`k` 就是 staleness。

### 陈旧梯度的影响

如果参数变化平缓、学习率较小，旧梯度仍可能有用。

如果：

- 学习率过大。
- Worker 数量很多。
- 单步计算时间差异大。
- 模型对参数变化敏感。

陈旧梯度可能导致：

- 震荡。
- 收敛变慢。
- 最终精度下降。
- 热门参数被过度更新。

### 常见缓解方法

- 限制最大 staleness。
- 根据梯度延迟衰减学习率。
- 对高频参数提高同步强度。
- Worker warmup，逐步增加并发。
- 梯度裁剪。
- 周期性 barrier。
- 使用局部缓存但设置版本或 TTL。

## 九、通信与性能瓶颈

### 小请求过多

如果每个 key 单独发 RPC，协议和调度开销会压过有效数据：

```text
bad:  每个 Embedding ID 一次 RPC
good: 按 PS shard 聚合成批量请求
```

Worker 应先按目标 Server 对 key 分桶：

```python
buckets = [[] for _ in range(num_servers)]
for key in keys:
    buckets[hash(key) % num_servers].append(key)
```

然后每个 shard 发一次批量 Pull/Push。

### 热点参数

高频 key 可能把某个 PS 打满。常见方法：

- 热点 key 复制。
- Worker-side cache。
- 动态迁移 shard。
- 更细粒度虚拟分片。
- 对只读或低频更新参数做缓存。

缓存必须处理一致性。异步训练通常允许弱一致缓存，同步训练则需要更严格版本控制。

### 带宽压缩

可使用：

- FP32 梯度转 BF16/FP16。
- 量化到 INT8。
- Top-K sparsification。
- error feedback。
- 合并多个小 Tensor。

压缩会引入精度误差和额外计算，必须用端到端吞吐与收敛共同评估。

### 计算与通信重叠

反向传播按层产生梯度，可以边算边 Push：

```text
计算 layer L 梯度
-> 异步 Push layer L
-> 同时计算 layer L-1 梯度
```

Pull 也可以预取下一 batch 的 Embedding。

## 十、容错与弹性调度

### Worker 故障

Worker 通常是无状态或轻状态计算节点，故障后可以：

- 重新调度数据 shard。
- 从当前全局参数继续训练。
- 重试未完成 step。

同步训练要防止其他 Worker 永久等待，可设置超时、backup Worker 或动态成员变更。

### PS 故障

PS 保存参数和优化器状态，故障影响更大。常见方案：

- 周期性 checkpoint。
- WAL，Write-Ahead Log。
- 主从复制。
- 多副本一致性协议。
- shard 迁移和恢复。

### Exactly-once 问题

Worker Push 超时后无法确定 Server 是否已经应用梯度：

```text
请求可能没到；
也可能已更新但响应丢失。
```

简单重试可能让同一梯度更新两次。

可为更新请求附加：

```text
worker_id + step_id + request_id
```

Server 记录已处理请求，实现幂等或去重。

### 弹性扩缩容

异步 PS 更容易动态增加 Worker，但仍要考虑：

- 全局有效 batch 和学习率变化。
- 数据是否重复消费。
- staleness 是否增加。
- PS 带宽是否足够。
- hash shard 是否需要迁移。

## 十一、一个简化实现

下面是用于理解语义的 Python 伪代码，不包含网络和容错。

```python
import threading
import numpy as np


class ParameterServer:
    def __init__(self, shape, learning_rate):
        self.weight = np.zeros(shape, dtype=np.float32)
        self.lr = learning_rate
        self.lock = threading.Lock()
        self.version = 0

    def pull(self):
        with self.lock:
            return self.weight.copy(), self.version

    def push(self, grad, base_version):
        with self.lock:
            staleness = self.version - base_version

            # 示例：对陈旧梯度做简单衰减。
            scale = 1.0 / (1.0 + staleness)
            self.weight -= self.lr * scale * grad
            self.version += 1
            return self.version


def worker_loop(ps, data_loader):
    for x, y in data_loader:
        weight, version = ps.pull()
        pred = x @ weight
        grad = x.T @ (pred - y) / x.shape[0]
        ps.push(grad, version)
```

这个例子展示了：

- Worker Pull 参数和版本。
- Worker 本地计算梯度。
- Worker Push 梯度。
- PS 串行更新参数。
- PS 可以感知梯度 staleness。

真实系统还需要参数分片、批量 RPC、异步执行、优化器状态、checkpoint 和容错。

## 十二、TensorFlow ParameterServerStrategy 示例

TensorFlow 提供 `ParameterServerStrategy`。下面是简化结构：

```python
import tensorflow as tf


cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    optimizer = tf.keras.optimizers.Adam(1e-3)


@tf.function
def worker_step(iterator):
    def replica_fn(batch):
        x, y = batch
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(pred - y))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    return strategy.run(replica_fn, args=(next(iterator),))


coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
per_worker_dataset = coordinator.create_per_worker_dataset(dataset_fn)
iterator = iter(per_worker_dataset)

for _ in range(num_steps):
    coordinator.schedule(worker_step, args=(iterator,))

coordinator.join()
```

这里：

- `strategy.scope()` 决定变量放置。
- Worker 执行计算。
- PS 保存变量。
- `ClusterCoordinator` 调度远程函数。
- `schedule` 可以异步发出任务。

实际使用时还要配置 `TF_CONFIG`、数据切分、容错和 checkpoint。

## 十三、与 AllReduce 数据并行的区别

| 对比项 | Parameter Server | AllReduce |
| --- | --- | --- |
| 参数存储 | 分片存储在 PS | 每个 rank 通常有完整参数副本 |
| 梯度聚合 | Worker Push 到 PS | Rank 之间 collective reduce |
| 稀疏参数 | 天然支持按 key Pull/Push | 对超大稀疏表不友好 |
| 同步模式 | 同步、异步、SSP | 通常同步 |
| Straggler | 异步模式较不敏感 | 同步 collective 受最慢 rank 影响 |
| 中心瓶颈 | PS 可能成为热点 | 通信负载分散 |
| GPU 稠密训练 | 通常不如 NCCL AllReduce 高效 | 非常适合 |
| 弹性 Worker | 较容易支持 | 成员变化会重建通信组 |

纯稠密 GPU 模型通常优先使用 AllReduce，因为 ring/tree collective 能充分利用高速 GPU 网络。

超大稀疏模型通常更适合 PS，因为无需在每个 Worker 复制完整 Embedding。

混合模型可以同时使用：

```text
Sparse Embedding -> Parameter Server
Dense network    -> AllReduce
```

## 十四、工程设计清单

设计 PS 训练系统时，应逐项确认：

### 参数和优化器

- 参数是稀疏还是稠密？
- 参数与 optimizer state 总大小是多少？
- 如何分 shard？
- 是否存在热点 key？
- 不同参数组是否需要不同 optimizer？

### 一致性

- 使用 BSP、ASP 还是 SSP？
- 最大 staleness 是多少？
- Worker 数变化后学习率如何调整？
- 更新请求是否幂等？

### 通信

- 是否按 Server 批量合并 key？
- Pull/Push 能否与计算重叠？
- 是否需要梯度压缩？
- 网络、PS CPU 和内存带宽谁是瓶颈？

### 容错

- checkpoint 周期是多少？
- PS 故障如何恢复？
- Worker 重试会不会重复更新？
- 数据消费是否 exactly-once？

### 可观测性

- 每个 PS 的 QPS、带宽和队列长度。
- Pull/Push latency 的 P50/P99。
- 每个 shard 的参数量和访问频率。
- Worker step time 和 staleness。
- cache hit rate。
- checkpoint 时延。

## 十五、总结

参数服务器的核心思想是把：

```text
训练计算
```

和：

```text
参数存储与更新
```

分离。

它通过 Worker 执行 forward/backward，通过 PS 管理参数 shard 和 optimizer state，并用 Push/Pull 交换参数与梯度。

理解参数服务器需要抓住五条主线：

1. 参数如何分片与放置。
2. Worker 和 PS 如何通信。
3. 同步、异步和 SSP 如何影响吞吐与收敛。
4. 稀疏 Embedding 与稠密参数为何需要不同路径。
5. 热点、陈旧梯度、容错和弹性如何处理。

在现代大规模训练中，PS 和 AllReduce 并不是互斥关系。稀疏参数走 PS、稠密参数走 AllReduce，是兼顾超大参数容量与 GPU 通信效率的常见架构。
