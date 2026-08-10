---
title: '现代搜索模型架构：从 BM25、双塔召回到多任务精排与 LLM 重排'
description: '系统讲解现代互联网搜索的查询理解、多路召回、粗排、精排与重排架构，覆盖 BM25、SPLADE、双塔、ColBERT、Cross-Encoder、行为序列、多任务学习、去偏、蒸馏和生成式搜索。'
category: 'Research & Work'
pubDate: '2026-08-10T18:00:00+08:00'
updatedDate: '2026-08-10T18:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [现代搜索不是一个模型](#一现代搜索不是一个模型)
2. [搜索、推荐与 RAG 的边界](#二搜索推荐与-rag-的边界)
3. [搜索系统的离线链路与在线链路](#三搜索系统的离线链路与在线链路)
4. [查询理解决定召回的入口](#四查询理解决定召回的入口)
5. [词法召回：倒排索引与 BM25](#五词法召回倒排索引与-bm25)
6. [神经稀疏召回：让模型学习词项权重与扩展](#六神经稀疏召回让模型学习词项权重与扩展)
7. [稠密召回：双塔与对比学习](#七稠密召回双塔与对比学习)
8. [负样本决定双塔能学到什么](#八负样本决定双塔能学到什么)
9. [ANN 索引如何服务十亿级向量](#九ann-索引如何服务十亿级向量)
10. [Late Interaction：双塔与 Cross-Encoder 之间的折中](#十late-interaction双塔与-cross-encoder-之间的折中)
11. [混合召回为什么通常比单路更可靠](#十一混合召回为什么通常比单路更可靠)
12. [粗排：在吞吐与表达力之间取平衡](#十二粗排在吞吐与表达力之间取平衡)
13. [精排输入到底包含什么](#十三精排输入到底包含什么)
14. [从 LR、GBDT 到 Embedding 与特征交互](#十四从-lrgbdt-到-embedding-与特征交互)
15. [语义相关性模型：Cross-Encoder](#十五语义相关性模型cross-encoder)
16. [行为序列模型：从目标注意力到 Transformer](#十六行为序列模型从目标注意力到-transformer)
17. [多任务精排：一个列表为什么需要多个目标](#十七多任务精排一个列表为什么需要多个目标)
18. [Pointwise、Pairwise 与 Listwise 损失](#十八pointwisepairwise-与-listwise-损失)
19. [点击日志不是无偏标签](#十九点击日志不是无偏标签)
20. [最终打分与概率校准](#二十最终打分与概率校准)
21. [重排：从独立打分转向列表优化](#二十一重排从独立打分转向列表优化)
22. [蒸馏：让大模型进入低延迟搜索链路](#二十二蒸馏让大模型进入低延迟搜索链路)
23. [LLM 如何改变搜索而不是取代搜索](#二十三llm-如何改变搜索而不是取代搜索)
24. [生成式检索与语义 ID](#二十四生成式检索与语义-id)
25. [一套可落地的现代搜索模型蓝图](#二十五一套可落地的现代搜索模型蓝图)
26. [训练、部署与实时特征的一致性](#二十六训练部署与实时特征的一致性)
27. [搜索质量应该如何评估](#二十七搜索质量应该如何评估)
28. [如何根据业务选择模型](#二十八如何根据业务选择模型)
29. [常见误解](#二十九常见误解)
30. [总结](#三十总结)
31. [参考资料](#三十一参考资料)

## 一、现代搜索不是一个模型

现代互联网搜索不是：

```text
query -> 一个神经网络 -> 排好序的结果
```

更接近：

```text
内容采集与索引
    |
用户 Query
    -> 查询理解
    -> 多路召回
    -> 候选融合
    -> 粗排
    -> 精排
    -> 列表重排
    -> 结果展示或答案生成
```

工业系统使用级联架构，根本原因是候选规模和模型复杂度不能同时取最大值。

假设语料库有 `10^9` 个文档：

```text
召回：   10^9 -> 10^4
粗排：   10^4 -> 10^3
精排：   10^3 -> 10^2
重排：   10^2 -> 10
```

这些数量只是用于说明漏斗关系。真实系统会根据文档类型、延迟预算、机器成本和查询流量设置不同规模。

每一层优化的目标也不同：

| 阶段 | 主要目标 | 可接受的模型成本 |
| --- | --- | --- |
| 召回 | 高 Recall，不能漏掉真正相关结果 | 极低，必须支持全库检索 |
| 粗排 | 快速去除明显低质候选 | 低到中等 |
| 精排 | 精确判断相关性、质量和用户效用 | 中等到较高 |
| 重排 | 优化整个结果列表 | 只处理很小的 Top-K |

因此，评价一个搜索模型时不能只问“离线 NDCG 有多高”，还需要问：

```text
它运行在哪一层？
一次请求要打分多少候选？
文档表示能否离线预计算？
索引需要多少内存？
P99 延迟是多少？
它优化的是相关性、点击还是长期满意度？
```

## 二、搜索、推荐与 RAG 的边界

搜索、推荐和 RAG 共享许多模型，但它们不是同一个问题。

### 2.1 搜索

搜索有显式 Query：

```text
用户明确表达当前信息需求
```

核心约束是 Query-Document Relevance。个性化、热度和质量可以参与排序，但不能让结果偏离当前 Query。

### 2.2 推荐

推荐通常没有显式文本 Query。系统根据：

- 用户画像。
- 行为序列。
- 当前页面和时间上下文。
- 内容与社交关系。

构造隐式 Query。

因此，推荐模型更强调长期兴趣和行为概率，搜索模型更强调当前意图和文本、实体、属性的精确匹配。

### 2.3 RAG

RAG 在检索后增加生成器：

```text
Query
  -> Retrieve
  -> Rerank
  -> Select Evidence
  -> LLM Generate
```

检索仍然决定生成器能看到什么。若相关证据没有进入 Top-K，再强的生成模型也无法可靠恢复缺失事实。

RAG 与传统搜索的主要区别在输出层：

```text
传统搜索：返回排序后的文档或内容卡片。
RAG：将检索结果作为上下文生成答案。
```

但查询理解、混合召回、重排、去重、时效性和权限过滤仍然适用。

## 三、搜索系统的离线链路与在线链路

### 3.1 离线内容链路

离线或准实时链路负责把原始内容变成可检索数据：

```text
文档采集
-> 清洗与去重
-> 分词、实体识别、语言识别
-> 质量与安全分析
-> 生成稀疏词项
-> 生成稠密向量
-> 构建倒排索引和 ANN 索引
-> 发布索引分片
```

文档侧计算应尽量前移到离线：

- 文本 Encoder 的 Document Embedding。
- Token 级多向量表示。
- 文档质量分。
- 主题、实体、类别和安全标签。
- 结构化属性。

这样在线请求只需计算 Query 侧表示。

### 3.2 在线查询链路

在线链路的典型数据流：

```text
Query
-> Normalize / Spell / Intent / Entity
-> 构造多个 Retrieval Query
-> 并行访问倒排索引、向量索引、结构化索引
-> Merge + Deduplicate
-> Pre-rank
-> Fetch expensive features
-> Rank
-> Slate rerank
-> Render
```

这里有一个重要的工程原则：

> 特征越昂贵，越应该晚取。

召回阶段不应为数万候选访问昂贵的实时特征。通常先用可从索引直接读取的轻量特征过滤，再为少量候选拉取：

- 用户实时状态。
- 文档最新统计。
- Query-Document 交叉特征。
- 行为序列与文档的目标相关表示。

### 3.3 为什么要并行多路召回

不同召回器解决不同失败模式：

```text
词法召回：型号、专有名词、罕见词、数字、精确短语。
稠密召回：同义表达、改写、语义相关。
神经稀疏召回：可学习扩展，同时保留倒排索引。
结构化召回：地点、时间、价格、权限、类型等硬约束。
个性化召回：用户历史与当前 Query 的联合意图。
```

单路模型很难同时做到：

```text
精确匹配
+ 语义泛化
+ 可解释
+ 低延迟
+ 低索引成本
+ 强跨域泛化
```

## 四、查询理解决定召回的入口

查询理解不是简单的分词。它要把一个短而不完整的字符串转成检索系统可执行的意图。

### 4.1 规范化

基础处理包括：

- Unicode 和大小写规范化。
- 拼写纠错。
- 数字、单位和日期识别。
- 语言识别和跨语言映射。
- 分词、词干化或 Subword Tokenization。

但规范化必须保留关键精确 Token。型号、代码、错误码和人名被错误纠正，会直接破坏词法召回。

### 4.2 意图分类

Query Intent 常包含：

```text
导航型：寻找特定站点、账号或对象。
信息型：寻找事实、解释或教程。
交易型：购买、下载、预约。
本地型：依赖地理位置和营业状态。
时效型：新闻、天气、比分、价格。
多媒体型：寻找图片、视频、音乐或声音。
```

意图影响：

- 选择哪些索引。
- 召回配额如何分配。
- 是否强调时效性。
- 是否调用结构化数据。
- 排序目标权重。

### 4.3 实体与属性理解

例如：

```text
"2025 款 14 英寸某型号笔记本"
```

应拆成：

```text
实体类型：产品
时间属性：2025
尺寸属性：14 英寸
品牌/系列/型号：若干实体槽位
```

实体链接可以把不同表述归一到同一 Knowledge Entity，避免只依赖 Token 重叠。

### 4.4 Query Rewrite 与 Expansion

改写的目标是补足用户没有显式写出的检索表达：

```text
"预算本" -> "高性价比 笔记本"
"内存不够" -> "memory pressure out of memory"
```

常见方法：

- 同义词词典。
- Session Query Reformulation。
- Seq2Seq Query Rewrite。
- Pseudo Relevance Feedback。
- LLM 生成多个子查询。

扩展不是越多越好。错误扩展会造成 Query Drift，使系统召回主题相近但不回答原问题的内容。

### 4.5 Query Fan-out

复杂问题可以拆成多个子问题：

```text
原问题：
比较两类 GPU 在训练和推理中的差异

子查询：
训练吞吐差异
推理延迟差异
显存容量与带宽
软件生态兼容性
价格与能耗
```

每个子查询独立召回，再做证据融合。这是生成式搜索和 Deep Research 常用的上层编排方法。

Query Fan-out 增加了覆盖率，但也增加：

- 总检索请求数。
- 去重和证据冲突。
- 尾延迟。
- 低质量子查询带来的噪声。

因此需要对子查询做预算控制和动态停止。

## 五、词法召回：倒排索引与 BM25

### 5.1 倒排索引

倒排索引建立：

```text
term -> posting list
```

Posting 通常包含：

```text
document id
term frequency
term positions
field information
```

查询只访问包含 Query Term 的 Posting List，而不是扫描所有文档。

这使精确词项检索具有：

- 高吞吐。
- 可解释。
- 易于增量更新。
- 对罕见词和数字敏感。

### 5.2 BM25 公式

对 Query `Q` 和文档 `D`：

```text
BM25(Q, D)
= sum_{t in Q}
    IDF(t)
    * tf(t,D) * (k1 + 1)
      / [tf(t,D) + k1 * (1 - b + b * |D| / avgdl)]
```

其中：

```text
tf(t,D)：词 t 在文档中的频次
|D|：文档长度
avgdl：语料平均文档长度
k1：控制词频饱和
b：控制长度归一化
```

IDF 的常见形式：

```text
IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
```

### 5.3 词频为什么要饱和

相关词出现一次到两次，通常比出现零次到一次更有价值；但出现一百次不应比出现十次高十倍。

BM25 的分式让 `tf` 增长逐渐饱和，抑制关键词堆叠。

### 5.4 长度归一化

长文档自然包含更多词，若不归一化会得到不合理优势。

参数 `b` 控制长度惩罚：

```text
b = 0：不考虑文档长度。
b = 1：完整使用长度归一化。
```

### 5.5 BM25 仍然重要的原因

稠密模型擅长语义，但容易弱化：

- 稀有实体。
- 产品型号。
- 代码符号。
- 数字与单位。
- 强否定和精确短语。

BEIR 等跨域评估表明，BM25 仍是强 Zero-shot Baseline。现代系统更常见的做法是混合，而不是完全替换。

## 六、神经稀疏召回：让模型学习词项权重与扩展

BM25 的词项权重来自固定统计公式。神经稀疏模型让 Transformer 学习：

```text
文本 -> 词表维度上的稀疏权重向量
```

### 6.1 SPLADE 的基本思路

设词表大小为 `|V|`，对 Query 和 Document 分别生成：

```text
q_sparse in R^|V|
d_sparse in R^|V|
```

相关性：

```text
score(q,d) = q_sparse dot d_sparse
```

模型利用 MLM Head 给词表中的词赋权，因此不仅可以激活原文中的词，也可以激活语义扩展词。

例如文档写：

```text
"显存容量不足"
```

模型可能给：

```text
"OOM"
"memory"
"out-of-memory"
```

非零权重，从而缩小词汇不匹配。

### 6.2 为什么需要稀疏正则

若向量大部分维度非零，就不能高效使用倒排索引。

训练目标通常包含：

```text
L = L_rank + lambda_q * R(q_sparse) + lambda_d * R(d_sparse)
```

`R` 可以是 L1 或 FLOPS-inspired Regularization。它在效果与检索成本之间建立可调节权衡：

```text
更稠密 -> 召回可能更高，但 Posting 更长、延迟更高。
更稀疏 -> 索引更快，但可能损失扩展能力。
```

### 6.3 神经稀疏与稠密召回的区别

| 维度 | 神经稀疏 | 稠密召回 |
| --- | --- | --- |
| 表示基底 | 词表维度 | 学习得到的连续维度 |
| 索引 | 倒排索引 | ANN |
| 精确词匹配 | 强 | 相对弱 |
| 语义扩展 | 可以学习 | 天然具备 |
| 可解释性 | 可查看词项贡献 | 较弱 |
| 跨语言 | 依赖词表和训练 | 多语言 Encoder 可统一空间 |

## 七、稠密召回：双塔与对比学习

### 7.1 双塔结构

Query 和 Document 独立编码：

```text
q = E_q(query)
d = E_d(document)
```

相关性使用：

```text
score(q,d) = q^T d
```

或：

```text
score(q,d) = cosine(q,d)
```

Document Embedding 可以离线计算并写入 ANN Index。在线只需要：

```text
Query Encoder
+ ANN Search
```

这就是双塔能用于大规模召回的原因。

### 7.2 Query Tower 与 Document Tower 可以不共享参数

搜索中的 Query 和 Document 分布通常不对称：

```text
Query：短、口语化、信息不完整。
Document：长、结构化、包含标题与正文。
```

因此可以：

- 使用同构但不共享的 Encoder。
- 使用共享底座加不同 Projection Head。
- 对 Query 和 Document 使用不同输入模板。

参数共享减少模型大小并促进空间对齐；不共享增加表达能力。选择取决于数据量和部署成本。

### 7.3 对比学习目标

一个 Query 有正文档 `d+` 和多个负文档 `d_j-`：

```text
L_q =
  -log [
    exp(score(q,d+) / tau)
    /
    sum_j exp(score(q,d_j) / tau)
  ]
```

其中 `tau` 是温度。

该损失让正样本相似度上升，负样本下降。

### 7.4 In-batch Negative

一个 Batch 有 `B` 个正对：

```text
(q_1,d_1+), ..., (q_B,d_B+)
```

对 `q_i`，其他 `d_j+` 可作为负样本。一次矩阵乘得到：

```text
S = Q D^T
shape = [B, B]
```

优点：

- 不需额外编码负文档。
- 负样本数随 Batch Size 增长。
- 适合多卡 AllGather 扩大负样本池。

风险：

- Batch 中可能存在 False Negative。
- 随机负样本通常太简单。
- 不同 Query 的正样本分布可能引入偏差。

### 7.5 为什么双塔有表达力上限

所有 Query-Document 关系必须压缩为：

```text
一个 Query Vector
一个 Document Vector
一次内积
```

细粒度关系很难完整保留：

- 哪个 Query Token 对应哪个 Document Token。
- 否定词修饰关系。
- 多约束是否同时满足。
- 数字、时间和实体组合。

这就是双塔效率高，但通常需要精排或 Late Interaction 补充的原因。

## 八、负样本决定双塔能学到什么

### 8.1 随机负样本

从全库随机取文档，大多数与 Query 完全无关：

```text
score(q,d-) 很低
```

损失很快接近零，模型学不到细粒度决策边界。

### 8.2 BM25 Hard Negative

使用 BM25 召回但没有标注为相关的文档：

```text
词面相似
语义可能不相关
```

它迫使模型学习超越关键词重叠的语义差异。

### 8.3 ANN Hard Negative

使用当前双塔从全库检索近邻，将高分误召回作为负样本。

ANCE 的核心思想是：

```text
定期用当前模型编码全库
-> 更新 ANN Index
-> 挖掘模型最容易混淆的全局负样本
-> 继续训练
```

这让训练负样本更接近推理时真正遇到的错误。

### 8.4 Teacher Negative

先用强 Cross-Encoder 给候选打分：

```text
双塔高分但 Teacher 低分 -> Hard Negative
```

也可以用 Teacher Score 作为 Soft Label 蒸馏。

### 8.5 False Negative

未点击或未标注不代表不相关。

Hard Negative 越难，越容易包含真实相关文档。常见处理：

- 使用多级相关性标签。
- Teacher 过滤。
- 对近似相关负样本降低权重。
- 使用 Debiased Contrastive Loss。
- 排除同实体、同答案或重复文档。

负样本工程通常比简单增加 Encoder 层数更重要。

## 九、ANN 索引如何服务十亿级向量

精确搜索需要计算：

```text
q dot d_i, for every document i
```

复杂度：

```text
O(N * dim)
```

在十亿级文档上不可接受，因此使用 Approximate Nearest Neighbor。

### 9.1 HNSW

HNSW 构建多层近邻图：

```text
高层：节点少，负责长距离导航。
低层：节点多，负责局部精细搜索。
```

查询从顶层入口贪心移动，再逐层下降。

关键参数：

```text
M：每个节点连接数。
efConstruction：建图搜索宽度。
efSearch：查询搜索宽度。
```

一般规律：

```text
更大 M / efSearch
-> 更高 Recall
-> 更高内存与延迟
```

HNSW 查询质量高，但图边有显著内存开销，动态删除和分布式分片也需要额外工程。

### 9.2 IVF

Inverted File 先用 Coarse Centroid 划分空间：

```text
document vector -> nearest coarse cluster
```

查询只访问最近的 `nprobe` 个 Cluster。

```text
nlist：总 Cluster 数
nprobe：查询访问 Cluster 数
```

更大 `nprobe` 提升 Recall，但增加距离计算。

### 9.3 Product Quantization

将 `d` 维向量切成 `m` 个子空间，每个子空间只存一个 Codeword ID：

```text
x = [x_1, x_2, ..., x_m]
-> [code_1, code_2, ..., code_m]
```

距离通过 Lookup Table 累加，显著减少：

- 向量存储。
- HBM/DRAM 带宽。
- 距离计算成本。

代价是量化误差。

### 9.4 ScaNN

ScaNN 将：

- Search Space Pruning。
- MIPS-aware Quantization。
- SIMD 优化。

结合起来。

它强调量化目标不应只最小化向量重建误差，而应优先保持最大内积候选的排序。

### 9.5 ANN 的召回率也是模型指标

即使双塔理论 Top-K 很好，ANN 近似搜索仍可能漏掉它。

最终召回可以分解为：

```text
业务 Recall
= 模型表示能力
* ANN 近似 Recall
* 过滤与索引覆盖率
```

调模型时必须固定 ANN 配置，调索引时必须固定 Embedding 版本，否则难以定位收益来源。

## 十、Late Interaction：双塔与 Cross-Encoder 之间的折中

### 10.1 单向量双塔

```text
q -> one vector
d -> one vector
score = q^T d
```

文档索引小、速度快，但 Token 交互弱。

### 10.2 Cross-Encoder

```text
[CLS] query [SEP] document [SEP]
-> Transformer
-> relevance score
```

Query 和 Document Token 从第一层开始交互，效果强，但每个候选都要完整推理。

### 10.3 ColBERT Late Interaction

Query 和 Document 分别编码成 Token Vector：

```text
Q = [q_1, ..., q_m]
D = [d_1, ..., d_n]
```

MaxSim：

```text
score(Q,D)
= sum_i max_j(q_i^T d_j)
```

每个 Query Token 找最匹配的 Document Token，再求和。

它保留：

- Token 级精确匹配。
- 上下文化表示。
- Document Token Vector 离线预计算。

代价：

- 每个文档要存多个向量。
- MaxSim 比单内积更贵。
- Index 与压缩更复杂。

ColBERTv2 使用 Residual Compression 和 Denoised Supervision 降低多向量存储开销。

### 10.4 三类模型的表达力与成本

| 模型 | Document 可预计算 | 在线交互 | 典型用途 |
| --- | --- | --- | --- |
| 双塔 | 可以 | 单向量内积 | 全库召回 |
| Late Interaction | 可以 | Token MaxSim | 高质量召回或粗排 |
| Cross-Encoder | 不完整，需与 Query 联合算 | 全层 Attention | 精排与重排 |

## 十一、混合召回为什么通常比单路更可靠

### 11.1 词法与稠密的错误不完全重合

词法容易漏掉：

- 同义词。
- 改写。
- 跨语言表达。

稠密容易漏掉：

- 罕见型号。
- 精确数字。
- 代码和符号。
- 强约束词。

两路结果并集可提高 Coverage。

### 11.2 Raw Score 不能直接相加

BM25 Score：

- 无固定上界。
- 依赖 Query 长度和语料统计。

Cosine 或 Inner Product：

- 分布由 Embedding 训练决定。

直接计算：

```text
score = bm25 + cosine
```

通常会被尺度更大的信号支配。

### 11.3 Reciprocal Rank Fusion

RRF 不使用 Raw Score，只使用排名：

```text
RRF(d) = sum_r 1 / (k + rank_r(d))
```

如果文档在多个召回器中排名靠前，会获得更高分。

优点：

- 不需要 Score Calibration。
- 对异构召回器稳定。
- 参数少。

缺点：

- 丢弃 Score Margin。
- 不同召回器默认权重相同。
- 不能自动学习 Query-dependent Fusion。

### 11.4 Learned Fusion

可以训练融合模型输入：

- 各路 Rank。
- 各路标准化 Score。
- 是否多路命中。
- Query Intent。
- 文档类型。
- 各召回器历史置信度。

学习动态配额与融合权重。

### 11.5 召回配额

假设总候选预算为 5000：

```text
BM25：2000
Dense：1500
Learned Sparse：1000
结构化与实时：500
```

固定配额容易浪费。更合理的方法是根据 Query 类型调整：

```text
精确型号 Query -> 提高词法与结构化配额。
自然语言问题 -> 提高稠密与语义扩展配额。
时效 Query -> 提高实时索引配额。
```

## 十二、粗排：在吞吐与表达力之间取平衡

召回后可能还有几千到几万候选，无法全部进入昂贵 Cross-Encoder 或多任务精排。

### 12.1 粗排的职责

粗排要：

- 保留精排 Top-K 潜在候选。
- 去除明显不相关或低质结果。
- 严格控制 CPU/GPU 成本。

它不要求完美复现最终排序，而要求较高的：

```text
Recall of downstream top results
```

### 12.2 常用粗排模型

- 线性模型或 GBDT。
- 小型 MLP。
- 双塔点积加少量轻特征。
- Low-rank Cross Network。
- Late Interaction 的压缩版本。
- 精排模型的蒸馏学生。

### 12.3 为什么粗排不能只学点击

粗排样本来自上游召回分布。若只优化点击：

- 热门文档可能挤掉长尾相关结果。
- 未展示候选没有标签。
- 精排想要的高相关候选可能提前丢失。

粗排常同时学习：

```text
相关性
精排分数蒸馏
用户行为
召回保真目标
```

### 12.4 Cascade Distillation

让精排 Teacher 给大量候选打 Soft Label：

```text
L = alpha * L_ground_truth
  + beta * L_score_distill
  + gamma * L_pair_order
```

其中：

- `L_score_distill` 对齐分数。
- `L_pair_order` 对齐候选相对顺序。
- Ground Truth 防止学生继承 Teacher 全部偏差。

## 十三、精排输入到底包含什么

精排不是只看 Query 和文档文本。典型特征可以分为五组。

### 13.1 Query 特征

- Query Token 与语义向量。
- 语言、地区和时间。
- 意图、实体和类目。
- Query 频率与历史质量。
- Query Rewrite 和召回来源。

### 13.2 Document 特征

- 标题、正文和多模态内容。
- 类型、作者、类目和实体。
- 新鲜度。
- 内容质量与安全分。
- 长期和近期统计。

### 13.3 Query-Document 交叉特征

- BM25 与 Field BM25。
- Phrase、Proximity、Exact Match。
- Query Term 在标题/正文/实体字段的覆盖。
- Dense Similarity。
- Token MaxSim。
- Cross-Encoder Relevance。
- 语言、地区、实体和属性一致性。

这类特征通常最能直接表达相关性，但无法完全离线预计算。

### 13.4 User 与 Session 特征

- 长期偏好。
- 当前 Session 中的查询改写和点击。
- 近期行为序列。
- 历史结果的跳过、点击和停留。

搜索个性化必须被当前 Query 约束。用户过去喜欢某类内容，不意味着任何 Query 下都应返回该类内容。

### 13.5 Context 特征

- 设备和网络。
- 当前时间。
- 地理位置。
- 请求入口。
- 可用展示样式。

Context 会改变用户行为概率，但不一定改变语义相关性。建模时应区分这两类目标。

## 十四、从 LR、GBDT 到 Embedding 与特征交互

### 14.1 Logistic Regression

```text
p(click=1|x) = sigmoid(w^T x + b)
```

优点：

- 低延迟。
- 易解释。
- 适合高维稀疏输入。
- 在线增量训练容易。

缺点是只能表达线性关系，交叉特征需要人工构造。

### 14.2 GBDT 与 LambdaMART

树模型擅长：

- 数值特征非线性。
- 缺失值。
- 特征阈值。
- 少量高质量人工特征。

LambdaMART 用 Lambda Gradient 将 NDCG 等离散排序指标的变化注入 Boosted Tree Training。

对一对文档 `i,j`，交换它们造成的指标变化：

```text
|Delta NDCG_ij|
```

可以作为 Pair Gradient 的权重，使模型优先修复 Top Position 的关键错误。

在结构化特征占主导、模型延迟严格的场景，LambdaMART 仍然很强。

### 14.3 Embedding & MLP

高基数离散特征通过 Embedding Table：

```text
feature id -> dense vector
```

多个 Field Embedding 拼接后进入 MLP：

```text
x = concat(e_1, e_2, ..., dense_features)
h = MLP(x)
score = Linear(h)
```

它减少人工交叉，但普通 MLP 学习低阶显式交叉并不高效。

### 14.4 Wide & Deep

Wide 部分记忆高置信交叉：

```text
wide = w^T phi(x)
```

Deep 部分泛化未见组合：

```text
deep = MLP(Embeddings(x))
```

联合输出：

```text
logit = wide + deep
```

核心不是简单 Ensemble，而是同时保留：

```text
Memorization + Generalization
```

### 14.5 Factorization Machine 与 DeepFM

FM 二阶交互：

```text
y_FM =
  w0 + sum_i w_i x_i
  + sum_{i<j} <v_i, v_j> x_i x_j
```

将每对特征的参数分解为低维向量内积，避免为每个组合独立学习参数。

DeepFM 共享 Embedding：

```text
FM Component -> 低阶显式交互
DNN Component -> 高阶隐式交互
```

### 14.6 DCN-V2

Cross Layer：

```text
x_{l+1}
= x_0 elementwise_mul (W_l x_l + b_l)
  + x_l
```

它显式构造有界阶数的特征交叉。

为降低 `W_l in R^{d x d}` 的成本，可以做低秩分解：

```text
W_l = U_l V_l^T
rank r << d
```

DCN-V2 还可使用多个 Low-rank Expert 和 Gate，提升不同样本上的交互表达。

### 14.7 为什么搜索精排仍需要“传统”特征

纯文本 Transformer 不一定知道：

- 文档刚刚发布。
- 结果是否可用。
- 当前地区是否适配。
- 历史质量是否稳定。
- 索引字段是否精确命中。

现代精排通常是：

```text
Language Model Signal
+ Sparse Structured Features
+ Behavioral Features
+ Quality/Freshness Signals
```

而不是只选其中一个。

## 十五、语义相关性模型：Cross-Encoder

### 15.1 输入结构

```text
[CLS] query [SEP] document_title [SEP] document_body [SEP]
```

经过 Transformer 后：

```text
score_rel = MLP(h_[CLS])
```

由于 Query Token 和 Document Token 在每一层相互 Attention，模型可以判断：

- 词义依赖上下文。
- 多个约束是否同时满足。
- 否定和修饰关系。
- 实体与属性是否对应。

### 15.2 为什么 Cross-Encoder 不能做全库召回

若有 `N` 个文档，每个 Query 要执行 `N` 次联合编码：

```text
cost ~= O(N * Transformer(query, document))
```

Document 表示不能完全离线复用，因为它依赖当前 Query。

因此 Cross-Encoder 一般只处理召回后的 Top-K。

### 15.3 BERT Reranker

BERT Passage Reranker 将相关性转成二分类：

```text
p(relevant | query, document)
```

用 Pointwise Cross Entropy 训练，按相关概率排序。

简单 Pointwise BERT 已经证明 Pretrained Language Model 能显著提升 Passage Reranking。

### 15.4 MonoT5 与 RankT5

MonoT5 将输入写成文本：

```text
Query: ...
Document: ...
Relevant:
```

模型生成：

```text
true / false
```

用 `true` Token 的概率作为分数。

RankT5 进一步让 T5 直接输出 Ranking Score，并使用 Pairwise/Listwise Loss，而不是把排序完全转成独立分类。

### 15.5 长文档问题

Transformer 输入长度有限。常见处理：

- 文档切 Passage。
- Title 与最佳 Passage 联合。
- MaxP：取最高 Passage Score。
- SumP：聚合多个 Passage。
- Hierarchical Encoder。
- 先做 Passage Retrieval，再做 Document Aggregation。

## 十六、行为序列模型：从目标注意力到 Transformer

语义相关性回答：

```text
这个文档是否回答当前 Query？
```

行为模型回答：

```text
在相关结果中，这个用户此刻更可能满意哪一个？
```

两者不能混为一谈。

### 16.1 Average Pooling 的问题

用户历史有：

```text
[h_1, h_2, ..., h_T]
```

简单平均：

```text
u = (1/T) * sum_t h_t
```

假设每段历史对当前候选同等重要，会把多种兴趣混在一起。

### 16.2 Target-aware Attention

给定当前候选 `c`：

```text
alpha_t = Attention(c, h_t, side_t)
u(c) = sum_t alpha_t h_t
```

不同候选得到不同用户表示：

```text
u(c_1) != u(c_2)
```

这类结构适合从历史行为中提取与当前 Query 或候选最相关的部分。

### 16.3 时间与行为类型

序列元素不应只有 Item Embedding，还可包含：

- 时间间隔。
- 行为类型。
- Query 或 Session Context。
- 内容类别。

时间衰减不应完全替代注意力。旧行为可能与当前 Query 高度相关，近期行为也可能只是噪声。

### 16.4 Transformer Sequence Encoder

将行为序列表示为：

```text
z_t = item_emb_t
    + action_emb_t
    + time_emb_t
    + position_emb_t
```

Self-Attention 捕获行为之间的依赖：

```text
H = Transformer(z_1, ..., z_T)
```

再用当前 Query/Candidate 做 Cross Attention 或 Target Attention。

相比 RNN：

- 并行训练更好。
- 更容易建模长距离依赖。
- 能直接表达行为之间的成对关系。

代价：

- `O(T^2)` Attention 成本。
- 长序列需截断、分层或兴趣压缩。
- 在线特征准备复杂。

### 16.5 搜索行为与推荐行为应分开理解

搜索行为有明确 Query Context。相同点击对象在不同 Query 下可能表示不同意图。

因此序列建模应考虑：

```text
(historical query, clicked document, action, time)
```

而不只是 Item ID Sequence。

## 十七、多任务精排：一个列表为什么需要多个目标

搜索体验不能由点击单独定义。

点击可能来自：

- 真正相关。
- 标题吸引但内容不满足。
- 位置靠前。
- 误触。

更完整的目标可能包括：

```text
Relevance
Click
Long Dwell
Conversion
Save / Share / Follow
Query Reformulation
Abandonment
Return Rate
```

### 17.1 Shared-bottom

```text
h = SharedNetwork(x)
y_t = TaskTower_t(h)
```

优点：

- 参数共享。
- 推理高效。
- 稀疏任务能利用密集任务信息。

缺点：

- 不相关任务产生梯度冲突。
- 一个任务提升可能损害另一个任务。

### 17.2 MMoE

多个共享 Expert：

```text
e_k = Expert_k(x)
```

每个任务有独立 Gate：

```text
g_t = softmax(W_t x)
h_t = sum_k g_{t,k} e_k
y_t = Tower_t(h_t)
```

不同任务可以选择不同专家组合。

### 17.3 PLE

PLE 显式区分：

```text
Shared Experts
Task-specific Experts
```

并在多层结构中逐步路由与分离信息，目标是减轻 Negative Transfer。

### 17.4 多任务不是 Head 越多越好

增加任务会带来：

- 标签密度差异。
- Loss Scale 差异。
- 收敛速度差异。
- 梯度冲突。
- Gate Collapse。

需要监控：

- 每任务梯度范数。
- Expert 使用率。
- Gate Entropy。
- 每任务离线和在线指标。
- 对主任务的增益或损害。

### 17.5 连续目标

Dwell Time 等连续标签常有：

- 长尾分布。
- 截断。
- 零膨胀。
- 不同内容长度造成不可比。

可以使用：

- Log Transform。
- Huber/Quantile Loss。
- Ordinal Classification。
- Survival Modeling。
- 分类加回归的多任务结构。

## 十八、Pointwise、Pairwise 与 Listwise 损失

### 18.1 Pointwise

将每个 Query-Document 独立分类：

```text
L_point =
  -y log sigmoid(s)
  -(1-y) log(1-sigmoid(s))
```

优点：

- 简单。
- 输出可解释为概率。
- 易与 CTR/CVR 等目标结合。

缺点：

- 不直接优化同一 Query 内相对顺序。
- 负样本比例影响概率。
- 对 Top-K 位置不敏感。

### 18.2 Pairwise

对同一 Query 的正负文档：

```text
L_pair = log(1 + exp(-(s_pos - s_neg)))
```

它只关心：

```text
s_pos > s_neg
```

RankNet 就是典型 Pairwise Ranking。

Pairwise 的难点：

- Pair 数量二次增长。
- Pair 采样影响训练。
- 所有 Pair 等权不符合 Top-K 目标。

### 18.3 LambdaRank

给 Pair Loss 乘交换该 Pair 对 NDCG 的影响：

```text
weight_ij = |Delta NDCG_ij|
```

于是靠近顶部、相关等级差异大的错误权重更高。

### 18.4 Listwise

直接对一个 Query 的候选列表建模：

```text
P_model(i | q)
= exp(s_i)
  / sum_j exp(s_j)
```

可以对齐 Label Distribution 或优化 ListMLE、ListNet 等目标。

优点：

- 训练视角更接近最终排序。
- 能处理候选间关系。

缺点：

- 一个 Batch 需要 Query Group。
- 列表长度变化。
- Sampling Distribution 与在线候选分布高度相关。

### 18.5 Loss 组合

工业精排常见：

```text
L_total =
  lambda_point * L_point
  + lambda_pair * L_pair
  + lambda_list * L_list
  + lambda_aux * L_aux
```

Pointwise 保持概率学习，Pairwise 强化相对顺序，Listwise 对齐 Top-K 指标。

## 十九、点击日志不是无偏标签

### 19.1 Position Bias

用户更容易观察前排结果：

```text
P(click)
= P(examine | position)
  * P(click | examined, relevance)
```

前排点击更多，不代表它更相关。

### 19.2 Selection Bias

只有被召回、排序并展示的文档才有机会得到反馈：

```text
未展示 != 不相关
```

训练数据由旧系统策略选择，因此新模型容易继承旧系统盲区。

### 19.3 Trust Bias

用户可能因为相信高位结果而点击，即使结果并不真正相关。

### 19.4 Negative Feedback 的歧义

未点击可能因为：

- 没有被看到。
- 前面的结果已经满足需求。
- 标题不吸引。
- 结果不相关。

将所有未点击直接标为负例，会混合多种原因。

### 19.5 Inverse Propensity Scoring

若位置 `k` 被观察的倾向为 `p_k`，可以用：

```text
weight = 1 / p_k
```

修正损失：

```text
L_IPS =
  sum_clicked
    L_rank / propensity(position)
```

低曝光位置得到更高权重。

问题：

- 小 Propensity 导致高方差。
- Propensity 估计错误会引入新偏差。
- 只能纠正被建模的偏差。

通常需要 Clipping、Self-normalization 或 Doubly Robust Estimator。

### 19.6 随机化实验

小流量 Position Swap 或 Exploration 可以估计观察倾向，但会暂时损害部分用户体验。

工程上需要在：

```text
探索质量
数据可识别性
线上风险
```

之间平衡。

### 19.7 人工相关性标签仍然重要

人工标签可以提供：

- 不受旧排序位置影响的相关性。
- 对未展示候选的判断。
- 多级 Relevance。
- 新 Query 和长尾 Query 的评估。

但人工标签成本高，也不完全代表真实用户效用。可靠系统会结合：

```text
人工相关性
+ 去偏行为数据
+ 在线实验
```

## 二十、最终打分与概率校准

### 20.1 相关性与行为概率应分开

设：

```text
r = semantic relevance
p_click = click probability
p_long = long-dwell probability
p_conv = conversion probability
q = content quality
f = freshness
```

最终分数可以是：

```text
score =
  w_r * phi_r(r)
  + w_c * phi_c(p_click)
  + w_l * phi_l(p_long)
  + w_v * phi_v(p_conv)
  + w_q * q
  + w_f * f
```

`phi` 是单调变换，用于控制不同信号的尺度和边际收益。

### 20.2 乘法与 Log-space

若需要多个条件同时成立，可以使用乘法：

```text
utility = p_click^a * p_long^b * quality^c
```

在 Log-space：

```text
log utility
= a log p_click
  + b log p_long
  + c log quality
```

乘法会强烈惩罚任一低分项，适合硬门槛逻辑；加法更容易做补偿。

### 20.3 为什么需要校准

如果模型输出 `0.8`，理想含义是：

```text
大量预测为 0.8 的样本中约 80% 发生目标行为
```

Pairwise/Listwise Ranker 的 Raw Score 通常没有概率含义。

校准方法包括：

- Platt Scaling。
- Isotonic Regression。
- Temperature Scaling。
- 分 Field 或分人群校准。

### 20.4 校准对多目标融合很重要

若一个 Head 系统性过置信，它会在加权组合中支配其他目标，即使单任务 AUC 很高。

因此多任务系统不仅要看排序能力，还要看：

```text
LogLoss
Brier Score
ECE
分人群 Calibration
```

### 20.5 保序校准

单调校准函数：

```text
p = g(score), g' >= 0
```

可以保持文档相对顺序，同时将 Raw Score 转成可组合概率。

## 二十一、重排：从独立打分转向列表优化

精排通常独立打分：

```text
score_i = f(query, document_i)
```

但最终体验是整个列表的函数。

### 21.1 重复与多样性

若 Top-10 都来自同一来源或表达同一观点，单文档都相关，列表仍然低效。

重排考虑：

- 去重。
- 类目和来源分散。
- 观点覆盖。
- 内容形态。
- 新鲜度。

### 21.2 Maximal Marginal Relevance

从未选候选中逐个选择：

```text
argmax_d [
  lambda * relevance(q,d)
  - (1-lambda) * max_{s in selected} similarity(d,s)
]
```

第一项保持 Query Relevance，第二项惩罚与已选结果重复。

### 21.3 Determinantal Point Process

DPP 为集合 `S` 定义：

```text
P(S) proportional det(L_S)
```

`L` 同时编码：

- Item Quality。
- Pairwise Similarity。

相似结果会降低行列式，天然鼓励多样性。

### 21.4 Set/Slate Model

Transformer 或 Pointer Network 可以读取候选列表：

```text
[candidate_1, ..., candidate_K]
```

联合输出新顺序。它能学习：

- 位置间依赖。
- 类别重复。
- 上下文互补。
- 列表级效用。

代价是：

- 推理更贵。
- 训练标签更难构造。
- 对输入初始顺序可能敏感。

### 21.5 规则仍不可避免

最终列表通常还有：

- 安全与合规过滤。
- 权限。
- 去重。
- 来源约束。
- 强时效规则。
- 结果类型混排。

这些约束适合显式系统实现，不应全部寄希望于模型隐式学习。

## 二十二、蒸馏：让大模型进入低延迟搜索链路

强模型常无法直接服务全流量：

```text
大 Cross-Encoder / LLM
-> 高延迟、高成本
```

蒸馏将其知识转给小模型。

### 22.1 Score Distillation

```text
L_score = MSE(s_student, s_teacher)
```

简单，但 Teacher 与 Student Score Scale 可能不同。

### 22.2 Probability Distillation

使用温度：

```text
p_t = softmax(s_t / T)
p_s = softmax(s_s / T)
L_KD = KL(p_t || p_s)
```

温度提高后，Teacher 的非 Top-1 相对偏好更明显。

### 22.3 Pairwise Distillation

对齐 Margin：

```text
L_margin =
  [(s_i^S - s_j^S)
   - (s_i^T - s_j^T)]^2
```

比单点分数更适合排序。

### 22.4 Listwise Distillation

让 Student 对齐 Teacher Top-K Distribution 或 Permutation，重点保留顶部顺序。

### 22.5 数据扩展

Teacher 可以给大量无人工标签 Query-Document Pair 生成：

- Soft Relevance。
- Pair Preference。
- Hard Negative。
- 解释或 Query Rewrite。

学生有时能通过更大 Teacher-labeled Dataset 获得比只在小规模人工标签上训练更好的泛化。

### 22.6 蒸馏不能消除 Teacher 偏差

Teacher 可能：

- 偏向长文档。
- 忽略新实体。
- 对特定语言较弱。
- 继承训练数据偏差。

因此学生应同时使用：

```text
Teacher Signal
+ Human Label
+ Debiased Interaction
```

## 二十三、LLM 如何改变搜索而不是取代搜索

### 23.1 Query Rewrite

LLM 能把短 Query 改写为：

- 更完整问题。
- 多个同义 Query。
- 结构化 Filter。
- 多语言 Query。

但生成结果必须经过：

- 实体保真检查。
- 数字和否定约束检查。
- Query Drift 检测。

### 23.2 Query Decomposition

复杂问题分成子问题，分别检索，再合并证据。适合：

- 比较。
- 多跳问题。
- 条件组合。
- 调研型 Query。

### 23.3 LLM Reranker

LLM 排序有三种范式。

Pointwise：

```text
逐文档判断相关性
```

Pairwise：

```text
给两个文档，判断谁更相关
```

Listwise：

```text
一次输入多个文档，直接输出顺序
```

Pairwise 容易得到稳定比较，但朴素复杂度接近 `O(K^2)`。Listwise 直接建模全局顺序，但受上下文长度、输入顺序和格式遵循影响。

### 23.4 为什么大 LLM 不直接打分所有候选

若每个 Query 有 1000 个候选，逐候选运行大模型成本过高。

更实际的路径：

```text
Retriever
-> Small Ranker
-> Top-20 LLM Rerank
```

或：

```text
LLM Teacher Offline Label
-> Distill Small Online Ranker
```

### 23.5 答案生成

生成式搜索不是只输出“最相关文档”，而是：

```text
检索证据
-> 判断证据质量
-> 处理冲突
-> 生成带引用答案
```

生成器必须受到 Evidence Grounding 约束。否则系统会从“搜索错误”升级为“流畅地给出错误答案”。

### 23.6 传统索引仍然是事实入口

LLM 参数中的知识：

- 有截止时间。
- 难以删除。
- 难以保证来源。
- 无法覆盖实时权限。

所以当前可靠架构仍然需要外部索引作为：

```text
实时性
可更新性
权限
引用
可审计性
```

的基础。

## 二十四、生成式检索与语义 ID

生成式检索尝试把：

```text
Query -> Search Index -> Document
```

改成：

```text
Query -> Autoregressive Model -> Document ID
```

### 24.1 DSI

Differentiable Search Index 训练两个映射：

```text
Document Text -> DocID
Query Text -> DocID
```

模型参数同时承担索引和检索。

### 24.2 为什么随机 DocID 难学

若 DocID 是随机整数：

```text
语义相似文档的 ID 没有共享前缀
```

模型只能记忆 Query 到随机符号的映射。

### 24.3 Semantic ID

先将 Document Embedding 离散化：

```text
document vector
-> code_1, code_2, ..., code_m
```

语义相近文档共享部分 Code Prefix，使 Beam Search 可以沿语义树搜索。

Residual Quantization 的典型过程：

```text
r_0 = z
c_l = argmin_k ||r_l - e_{l,k}||^2
r_{l+1} = r_l - e_{l,c_l}
```

最终：

```text
semantic_id = [c_0, c_1, ..., c_{m-1}]
```

### 24.4 优点

- 检索成为端到端生成。
- ID Prefix 可表达层次语义。
- 能将序列建模与召回统一。
- 对冷启动内容有潜在泛化能力。

### 24.5 当前难点

- 新文档加入和旧文档删除。
- ID Collision。
- Corpus 更新需要重训或增量学习。
- Beam Search 成本。
- 长尾文档记忆。
- DocID 设计决定效果。
- 难以保证全库 Coverage 和可解释性。

因此生成式检索是重要方向，但尚不能简单替代成熟的倒排和 ANN 系统。

## 二十五、一套可落地的现代搜索模型蓝图

下面给出一套不绑定具体业务的参考架构。

### 25.1 召回层

并行运行：

```text
Retriever A: BM25 / Fielded BM25
Retriever B: Learned Sparse
Retriever C: Dense Dual Encoder
Retriever D: Structured / Fresh / Entity
```

输出统一 Candidate：

```text
candidate_id
source_mask
rank_per_source
score_per_source
```

### 25.2 融合与粗排

特征：

- 各路 Rank 与 Score。
- Query Intent。
- 基础 Quality/Freshness。
- 轻量 Query-Document Cross。

模型：

```text
Small MLP / GBDT / Low-rank Cross Network
```

目标：

```text
保留精排 Top-K，而不是追求最终 CTR 最优。
```

### 25.3 精排特征编码

```text
Query Encoder
Document Encoder
Cross Feature Encoder
Behavior Sequence Encoder
Context Encoder
```

共享表示：

```text
h_base = FeatureInteraction(
    query_features,
    document_features,
    cross_features,
    context_features
)
```

个性化表示：

```text
h_seq = TargetAwareSequence(
    query,
    candidate,
    behavior_history
)
```

语义表示：

```text
h_sem = CrossEncoder(query, document)
```

融合：

```text
h = Fusion(h_base, h_seq, h_sem)
```

### 25.4 多任务 Head

```text
p_click = Head_click(h)
p_satisfy = Head_satisfy(h)
p_long = Head_long(h)
p_convert = Head_convert(h)
r_semantic = Head_relevance(h)
```

共享部分可以用：

- Shared-bottom。
- MMoE。
- PLE。

### 25.5 最终排序

```text
utility =
    semantic_gate(r_semantic)
    * behavior_utility(
        p_click,
        p_satisfy,
        p_long,
        p_convert
      )
    + quality_bonus
    + freshness_bonus
```

这里 `semantic_gate` 可防止行为模型把不相关但热门的内容推到顶部。

### 25.6 列表重排

对 Top-N：

- 去重。
- 来源和主题多样性。
- 安全与权限。
- 列表级模型。
- 结果样式与结构化卡片。

### 25.7 伪代码

```python
def search(request):
    query = understand_query(request.text, request.context)

    retrieval_results = parallel(
        lexical_retrieve(query),
        learned_sparse_retrieve(query),
        dense_retrieve(query),
        structured_retrieve(query),
    )

    candidates = fuse_and_deduplicate(retrieval_results)
    candidates = pre_rank(query, candidates, limit=1000)

    expensive_features = fetch_online_features(
        request.user,
        query,
        candidates,
    )

    task_scores = fine_rank(
        query=query,
        candidates=candidates,
        features=expensive_features,
    )

    utility = combine_calibrated_scores(task_scores)
    top_results = slate_rerank(candidates, utility, limit=20)
    return render(top_results)
```

## 二十六、训练、部署与实时特征的一致性

### 26.1 Training-Serving Skew

训练和在线必须使用一致的：

- 分词。
- 字典和 Hash。
- 特征默认值。
- 时间窗口。
- Embedding Version。
- 归一化参数。

否则离线提升无法复现到线上。

### 26.2 Point-in-time Correctness

训练样本只能使用请求发生时可见的信息。

若用未来统计特征：

```text
label leakage
```

离线指标会异常好，线上无收益。

### 26.3 Feature Freshness

特征更新频率不同：

```text
静态文本语义：小时或天级。
长期质量统计：小时级。
近期热度：分钟级。
用户 Session：秒级。
```

Feature Store 需要记录：

- 事件时间。
- 处理时间。
- 版本。
- TTL。

### 26.4 Embedding Version

更新 Query Encoder 或 Document Encoder 后，旧向量索引可能不兼容。

常见方案：

- 双写新旧索引。
- 版本化 Embedding。
- Backfill 后切流。
- Teacher-Student 对齐旧空间。

### 26.5 模型服务优化

- Document 表示预计算。
- Embedding Cache。
- Candidate Batch Inference。
- INT8/FP8 Quantization。
- TensorRT/ONNX Runtime。
- Dynamic Batching。
- Early Exit。
- Query Result Cache。

### 26.6 尾延迟

平均延迟正常不代表系统可用。搜索更关心：

```text
P95 / P99
```

多路召回并行后，总延迟通常由最慢分支决定。可以使用：

- Soft Timeout。
- 分路 Deadline。
- 迟到结果丢弃。
- 降级到轻量 Ranker。
- 热 Query Cache。

### 26.7 Cost-aware Routing

并非每个 Query 都需要最贵模型：

```text
简单导航 Query -> 词法 + 轻量排序。
复杂自然语言 Query -> 混合召回 + Cross-Encoder。
高价值少量 Query -> LLM Rerank。
```

Routing Model 可以根据复杂度和置信度动态分配计算预算。

## 二十七、搜索质量应该如何评估

### 27.1 Recall@K

```text
Recall@K
= Top-K 中相关文档数
  / 全部相关文档数
```

适合评估召回层。召回层重点是不要漏掉下游可能需要的结果。

### 27.2 MRR

设第一个相关结果排名为 `rank_q`：

```text
MRR = mean_q(1 / rank_q)
```

适合：

- Known-item Search。
- 问答。
- 用户通常只需要第一个正确结果。

### 27.3 NDCG

```text
DCG@K =
  sum_{i=1}^K
    (2^rel_i - 1) / log2(i + 1)

NDCG@K = DCG@K / IDCG@K
```

它同时考虑：

- 多级相关性。
- 排名位置。

适合精排评估。

### 27.4 AUC 与 GAUC

AUC 衡量随机正例分数高于随机负例的概率。

全局 AUC 可能被大 Query Group 支配。搜索中常按 Query 或用户分组计算，再加权得到 GAUC。

但 AUC 对 Top-K 位置不敏感，不能替代 NDCG。

### 27.5 Calibration

对于概率 Head，评估：

- LogLoss。
- Brier Score。
- Expected Calibration Error。
- 分国家、语言、设备或内容类型的 Field ECE。

### 27.6 系统指标

- P50/P95/P99 Latency。
- QPS。
- Timeout Rate。
- Index Freshness。
- Cache Hit Rate。
- Feature Missing Rate。
- ANN Recall。
- GPU/CPU Cost per Query。

### 27.7 在线指标

- Search Success Rate。
- Reformulation Rate。
- Long Click / Dwell。
- Conversion。
- Abandonment。
- Return Rate。
- 用户投诉和安全指标。

只优化 CTR 可能诱导标题党。在线实验必须同时设置：

```text
主指标
+ 质量护栏
+ 延迟护栏
+ 长期护栏
```

### 27.8 Slice Evaluation

总体指标会掩盖长尾问题。应按以下切片：

- Head/Tail Query。
- Query 长度。
- 语言。
- 新文档。
- 新用户。
- 意图。
- 文档类型。
- 有无拼写错误。
- 有无实体和数字。

## 二十八、如何根据业务选择模型

### 28.1 小规模企业文档搜索

推荐起点：

```text
BM25
+ Dense Retriever
+ RRF
+ Top-50 Cross-Encoder
```

理由：

- 工程简单。
- 领域数据通常不够训练复杂 CTR 模型。
- Cross-Encoder 可以直接提升语义相关性。

### 28.2 大规模 Web 或内容搜索

需要：

- 多索引、多路召回。
- Query Understanding。
- 粗排与精排。
- Quality/Freshness。
- 行为去偏。
- 列表重排。

单一向量数据库远远不够。

### 28.3 电商搜索

特别重视：

- 属性与结构化 Filter。
- 型号、品牌和数字精确匹配。
- Query-Product Relevance。
- 点击与转化多任务。
- 库存、价格和时效。
- 多样性与同款去重。

词法和结构化召回通常不能被纯 Dense 完全替代。

### 28.4 多媒体搜索

需要共同空间：

```text
Text Query Encoder
Image/Video/Audio Encoder
```

并结合：

- OCR。
- ASR。
- Caption。
- 多模态 Cross-Encoder。

多模态 Embedding 适合召回，精排还需要页面上下文和文本证据。

### 28.5 RAG 与 Agent Search

重点不只是相关文档，还包括：

- Passage 粒度。
- Evidence Coverage。
- Source Diversity。
- 权限。
- 时效性。
- 可引用性。

可使用：

```text
Hybrid Retrieval
-> Cross-Encoder
-> MMR
-> Evidence Packing
-> LLM
```

### 28.6 模型选择表

| 需求 | 优先模型 |
| --- | --- |
| 罕见词与精确匹配 | BM25 / Neural Sparse |
| 全库语义召回 | Dual Encoder + ANN |
| Token 级细粒度召回 | ColBERT 类 Late Interaction |
| Top-K 高精度相关性 | Cross-Encoder / T5 Ranker |
| 用户历史与当前候选 | Target-aware Sequence Model |
| 多个行为目标 | MMoE / PLE |
| 列表多样性 | MMR / DPP / Slate Model |
| 低延迟部署大模型知识 | Ranking Distillation |
| 复杂问答搜索 | Query Fan-out + Hybrid Retrieval + Rerank |

## 二十九、常见误解

### 29.1 “向量检索已经淘汰 BM25”

错误。

Dense Retrieval 擅长语义，BM25 擅长精确词项。现代系统通常混合两者。

### 29.2 “CTR 越高，搜索相关性越好”

错误。

点击受位置、标题、展示样式和旧系统策略影响。搜索还要评估语义相关性、满意度和长期行为。

### 29.3 “Cross-Encoder 效果最好，所以应该直接做召回”

错误。

Cross-Encoder 无法经济地对全库逐文档联合编码。它适合候选规模已经很小的阶段。

### 29.4 “双塔只要把 Batch 做大就能训练好”

错误。

大 Batch 提供更多负样本，但不保证负样本有信息。Hard Negative、False Negative 处理和索引挖掘更关键。

### 29.5 “多任务一定优于单任务”

错误。

任务冲突会导致 Negative Transfer。必须设计共享和任务特定路径，并监控梯度和 Gate。

### 29.6 “离线 AUC 提升就可以上线”

错误。

AUC 不直接对应 Top-K 排序、Calibration、P99 延迟和长期用户价值。必须进行分层离线评估与在线实验。

### 29.7 “LLM 可以把检索系统都替换掉”

错误。

LLM 不擅长承担实时索引、权限、删除、更新和全库覆盖。可靠系统把 LLM 用于理解、改写、重排与生成，并继续依赖外部索引。

### 29.8 “重排只是加入业务规则”

不完整。

重排优化的是集合效用，包括相关性、多样性、去重、公平性和展示约束，可以由规则、组合优化和神经 Slate Model 共同完成。

## 三十、总结

现代搜索模型的主线不是从 BM25 单向升级到一个越来越大的 Transformer，而是形成分层协作：

```text
BM25 保留精确词项能力
Neural Sparse 学习词项权重和扩展
Dual Encoder 提供可索引语义表示
Late Interaction 保留细粒度 Token 匹配
Cross-Encoder 完成高精度相关性判断
Behavior Model 建模个性化与满意度
Multi-task Model 协调多个用户目标
Slate Reranker 优化整个结果列表
LLM 负责复杂理解、重排、编排和答案生成
```

真正决定搜索质量的不是单个模型是否先进，而是每一层是否解决了正确的问题：

```text
召回层不能漏
粗排层不能误杀
精排层要分清相关性与行为概率
重排层要优化列表而不是单点
训练数据必须去偏
离线指标必须与在线目标对齐
模型复杂度必须服从延迟和成本
```

搜索系统的本质，是在有限计算预算内，从海量内容中逐步提高信息密度。模型架构只是实现这个目标的手段，级联、索引、样本、目标、校准和在线实验同样属于模型能力的一部分。

## 三十一、参考资料

1. [How Google Search Organises Information](https://www.google.com/search/howsearchworks/how-search-works/organizing-information/)
2. [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
3. [RankNet: Learning to Rank using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
4. [From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
5. [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/)
6. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
7. [Approximate Nearest Neighbor Negative Contrastive Learning](https://arxiv.org/abs/2007.00808)
8. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)
9. [SPLADE v2](https://arxiv.org/abs/2109.10086)
10. [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832)
11. [ColBERTv2](https://arxiv.org/abs/2112.01488)
12. [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)
13. [HNSW](https://arxiv.org/abs/1603.09320)
14. [ScaNN: Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396)
15. [Elasticsearch Reciprocal Rank Fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
16. [Vespa Phased Ranking](https://docs.vespa.ai/en/ranking/phased-ranking.html)
17. [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)
18. [Document Ranking with a Pretrained Sequence-to-Sequence Model](https://arxiv.org/abs/2003.06713)
19. [RankT5](https://arxiv.org/abs/2210.10634)
20. [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
21. [DeepFM](https://arxiv.org/abs/1703.04247)
22. [DCN-V2](https://arxiv.org/abs/2008.13535)
23. [Deep Interest Network](https://arxiv.org/abs/1706.06978)
24. [Behavior Sequence Transformer](https://arxiv.org/abs/1905.06874)
25. [Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007)
26. [Progressive Layered Extraction](https://dl.acm.org/doi/10.1145/3383313.3412236)
27. [Unbiased Learning-to-Rank with Biased Feedback](https://www.cs.cornell.edu/~adith/docs/Click-LTR.pdf)
28. [Maximal Marginal Relevance](https://dl.acm.org/doi/10.1145/290941.291025)
29. [RankDistil: Knowledge Distillation for Ranking](https://proceedings.mlr.press/v130/reddi21a.html)
30. [Knowledge Distillation for E-commerce Search Relevance Using LLMs](https://arxiv.org/abs/2505.07105)
31. [Transformer Memory as a Differentiable Search Index](https://arxiv.org/abs/2202.06991)
32. [A Neural Corpus Indexer for Document Retrieval](https://arxiv.org/abs/2206.02743)
33. [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
34. [Google AI Visual Search and Query Fan-out](https://blog.google/company-news/inside-google/googlers/how-google-ai-visual-search-works/)
35. [Azure AI Search Semantic Ranking](https://learn.microsoft.com/en-us/azure/search/semantic-search-overview)
