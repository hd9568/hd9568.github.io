---
title: '从 llama.cpp 源码看 GQA：num_kv_heads、KV Cache 与注意力计算'
description: '结合 llama.cpp 源码讲解 GQA 的工程实现，从模型元数据、n_gqa、Q/K/V reshape、KV Cache 分配到 attention 计算逐层拆解。'
category: '推理优化'
pubDate: '2026-07-07T16:42:00+08:00'
updatedDate: '2026-07-07T16:42:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [为什么选择 llama.cpp](#二为什么选择-llamacpp)
3. [源码入口总览](#三源码入口总览)
4. [模型配置：n_head 与 n_head_kv](#四模型配置n_head-与-n_head_kv)
5. [n_gqa 怎么算](#五n_gqa-怎么算)
6. [Q/K/V 投影后的形状](#六qkv-投影后的形状)
7. [KV Cache 只按 KV heads 分配](#七kv-cache-只按-kv-heads-分配)
8. [写入 KV Cache](#八写入-kv-cache)
9. [读取 KV Cache 并参与 attention](#九读取-kv-cache-并参与-attention)
10. [手搓一个 llama.cpp 风格的 GQA](#十手搓一个-llamacpp-风格的-gqa)
11. [具体数字例子](#十一具体数字例子)
12. [工程实现要点](#十二工程实现要点)
13. [面试表达](#十三面试表达)
14. [总结](#十四总结)

## 一、核心结论

llama.cpp 中 GQA 的实现可以概括为一句话：

```text
Q 使用 n_head 个 heads，K/V 只使用 n_head_kv 个 heads；
KV Cache 的宽度是 n_head_kv * head_dim；
attention 计算时让多个 Q heads 共享同一个 KV head。
```

源码中关键变量：

```text
n_head       : Query heads 数量
n_head_kv    : Key/Value heads 数量
n_gqa        : 每个 KV head 对应多少个 Q heads
n_embd_k_gqa : K cache 每个 token 的总宽度
n_embd_v_gqa : V cache 每个 token 的总宽度
```

核心关系：

```text
n_gqa = n_head / n_head_kv
n_embd_k_gqa = n_embd_head_k * n_head_kv
n_embd_v_gqa = n_embd_head_v * n_head_kv
```

重点：GQA 的推理收益来自 KV Cache 只存 `n_head_kv` 份，而不是把 K/V repeat 成 `n_head` 份。

## 二、为什么选择 llama.cpp

本地 `inference-framework/` 下有多个主流推理框架：

```text
vllm
TensorRT-LLM
SGLang
llama.cpp
lmdeploy
text-generation-inference
mlc-llm
```

本文选择 `llama.cpp`，原因是它对 GQA 的变量命名非常直接：

- `n_head`
- `n_head_kv`
- `n_gqa`
- `n_embd_k_gqa`
- `n_embd_v_gqa`

并且它的 Q/K/V reshape 和 KV Cache 分配都在 C++ 代码中，适合结合面试手搓实现讲解。

本文参考的本地源码路径：

```text
inference-framework/llama.cpp/src/llama-hparams.cpp
inference-framework/llama.cpp/src/llama-hparams.h
inference-framework/llama.cpp/src/llama-model.cpp
inference-framework/llama.cpp/src/llama-graph.cpp
inference-framework/llama.cpp/src/llama-kv-cache.cpp
```

## 三、源码入口总览

llama.cpp 的 GQA 链路可以按下面顺序读：

```text
1. llama-model.cpp
   读取模型元数据中的 n_head 和 n_head_kv。

2. llama-hparams.cpp
   定义 n_gqa、n_embd_k_gqa、n_embd_v_gqa。

3. llama-graph.cpp
   根据 n_head 和 n_head_kv reshape Q/K/V。

4. llama-kv-cache.cpp
   按 n_embd_k_gqa / n_embd_v_gqa 分配、写入、读取 KV Cache。

5. llama-graph.cpp
   用 q、k、v 调 build_attn_mha，执行 attention。
```

这条链路很重要，因为 GQA 不是只改一个 kernel 参数，而是贯穿：

- 模型配置。
- 权重形状。
- 运行时张量形状。
- KV Cache layout。
- attention 计算。

## 四、模型配置：n_head 与 n_head_kv

llama.cpp 从 GGUF 模型元数据中读取 attention head 数。

关键逻辑在 `llama-model.cpp`：

```cpp
ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT,
                  hparams.n_head_arr,
                  hparams.n_layer,
                  false);

// n_head_kv is optional, default to n_head
hparams.n_head_kv_arr = hparams.n_head_arr;

ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV,
                  hparams.n_head_kv_arr,
                  hparams.n_layer,
                  false);
```

这段代码说明：

```text
如果模型没有单独声明 n_head_kv，则默认 n_head_kv = n_head。
```

也就是说：

- 没有 `attention.head_count_kv` 时，是普通 MHA。
- 有 `attention.head_count_kv` 且小于 `n_head` 时，是 GQA/MQA。

例如：

```text
n_head = 32
没有 n_head_kv -> n_head_kv = 32 -> MHA

n_head = 32
n_head_kv = 8 -> GQA

n_head = 32
n_head_kv = 1 -> MQA
```

llama.cpp 启动时也会打印这些信息：

```cpp
LLAMA_LOG_INFO("%s: n_head = %s\n", ...);
LLAMA_LOG_INFO("%s: n_head_kv = %s\n", ...);
LLAMA_LOG_INFO("%s: n_gqa = %s\n", ...);
LLAMA_LOG_INFO("%s: n_embd_k_gqa = %s\n", ...);
LLAMA_LOG_INFO("%s: n_embd_v_gqa = %s\n", ...);
```

如果加载一个 LLaMA 类 GQA 模型，日志里通常能看到 `n_head_kv` 小于 `n_head`。

## 五、n_gqa 怎么算

`llama-hparams.cpp` 中有直接定义：

```cpp
uint32_t llama_hparams::n_gqa(uint32_t il) const {
    const uint32_t n_head    = this->n_head(il);
    const uint32_t n_head_kv = this->n_head_kv(il);

    if (n_head_kv == 0) {
        return 0;
    }

    return n_head / n_head_kv;
}
```

这里 `il` 是 layer id，因为一些模型可能每层 head 配置不同。

`n_gqa` 的含义是：

```text
每个 KV head 被多少个 Q heads 共享。
```

例如：

```text
n_head = 32
n_head_kv = 8
n_gqa = 32 / 8 = 4
```

映射关系：

```text
Q head 0,1,2,3    -> KV head 0
Q head 4,5,6,7    -> KV head 1
...
Q head 28,29,30,31 -> KV head 7
```

代码里虽然 `n_gqa` 不是每处都直接用于 attention kernel，但它决定了很多和 GQA 宽度相关的判断、日志和量化粒度。

## 六、Q/K/V 投影后的形状

GQA 的一个关键点是 Q 和 K/V 的输出维度不同。

在 `llama-graph.cpp` 的 `build_qkv` 中：

```cpp
const int64_t n_embd_q  = n_embd_head * n_head;
const int64_t n_embd_kv = n_embd_head * n_head_kv;
```

如果是 fused QKV 权重，llama.cpp 从一个 `qkv` tensor 中切出 Q/K/V：

```cpp
Qcur = ggml_view_3d(ctx0, qkv,
    n_embd_head, n_head, n_tokens,
    ...,
    0);

Kcur = ggml_view_3d(ctx0, qkv,
    n_embd_head, n_head_kv, n_tokens,
    ...,
    ggml_row_size(qkv->type, n_embd_q));

Vcur = ggml_view_3d(ctx0, qkv,
    n_embd_head, n_head_kv, n_tokens,
    ...,
    ggml_row_size(qkv->type, n_embd_q + n_embd_kv));
```

如果是 separate Q/K/V 权重，则分别 matmul 后 reshape：

```cpp
Qcur = ggml_reshape_3d(ctx0, Qcur,
    n_embd_head, n_head, n_tokens);

Kcur = ggml_reshape_3d(ctx0, Kcur,
    n_embd_head, n_head_kv, n_tokens);

Vcur = ggml_reshape_3d(ctx0, Vcur,
    n_embd_head, n_head_kv, n_tokens);
```

这段代码已经把 GQA 的形状讲清楚了：

```text
Qcur: [head_dim, n_head,    n_tokens]
Kcur: [head_dim, n_head_kv, n_tokens]
Vcur: [head_dim, n_head_kv, n_tokens]
```

注意：K/V 没有 reshape 到 `n_head`。它们始终只保留 `n_head_kv` 个 heads。

### 数字例子

假设：

```text
n_embd = 4096
n_head = 32
n_head_kv = 8
n_embd_head = 128
n_tokens = 1
```

那么：

```text
Qcur: [128, 32, 1] -> 4096 elements
Kcur: [128, 8,  1] -> 1024 elements
Vcur: [128, 8,  1] -> 1024 elements
```

Q projection 输出 4096 维，K/V projection 各输出 1024 维。

如果是 fused QKV，总输出宽度是：

```text
Q + K + V = 4096 + 1024 + 1024 = 6144
```

而 MHA 下是：

```text
Q + K + V = 4096 + 4096 + 4096 = 12288
```

所以 GQA 不只减少 KV Cache，也减少 K/V projection 输出和后续写 cache 的数据量。

## 七、KV Cache 只按 KV heads 分配

`llama-hparams.cpp` 中定义：

```cpp
uint32_t llama_hparams::n_embd_k_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);
    return n_embd_head_k(il) * n_head_kv;
}

uint32_t llama_hparams::n_embd_v_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);
    return n_embd_head_v(il) * n_head_kv;
}
```

这两个值就是每个 token 在 KV Cache 中的 K/V 宽度。

KV Cache 分配在 `llama-kv-cache.cpp`：

```cpp
const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
const uint32_t n_embd_v_gqa = !v_trans
    ? hparams.n_embd_v_gqa(il)
    : hparams.n_embd_v_gqa_max();

ggml_tensor * k = ggml_new_tensor_3d(
    ctx, type_k, n_embd_k_gqa, kv_size, n_stream);

ggml_tensor * v = ggml_new_tensor_3d(
    ctx, type_v, n_embd_v_gqa, kv_size, n_stream);
```

这里最重要的是：

```text
k cache shape: [n_embd_k_gqa, kv_size, n_stream]
v cache shape: [n_embd_v_gqa, kv_size, n_stream]
```

而：

```text
n_embd_k_gqa = head_dim * n_head_kv
```

不是：

```text
head_dim * n_head
```

所以 llama.cpp 的 KV Cache 真实节省了显存。

### 数字例子

```text
head_dim = 128
n_head = 32
n_head_kv = 8
kv_size = 8192
type = FP16
```

GQA：

```text
n_embd_k_gqa = 128 * 8 = 1024
K cache = 1024 * 8192 * 2 bytes = 16 MB
V cache = 16 MB
K + V = 32 MB / layer
```

MHA：

```text
n_embd_k = 128 * 32 = 4096
K cache = 4096 * 8192 * 2 bytes = 64 MB
V cache = 64 MB
K + V = 128 MB / layer
```

GQA 是 MHA 的 1/4。

## 八、写入 KV Cache

Decode 每生成一个 token，都要把当前 token 的 K/V 写入 KV Cache。

llama.cpp 中写 K 的函数是 `cpy_k`：

```cpp
ggml_tensor * llama_kv_cache::cpy_k(
    ggml_context * ctx,
    ggml_tensor * k_cur,
    ggml_tensor * k_idxs,
    int32_t il,
    const slot_info & sinfo) const {

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head * n_head;

    k_cur = ggml_view_2d(ctx, k_cur,
        n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}
```

这里的变量名 `n_head` 是从 `k_cur->ne[1]` 读出来的。对 K 来说，`k_cur->ne[1]` 实际就是 `n_head_kv`。

也就是说，K 写 cache 时：

```text
k_cur: [head_dim, n_head_kv, n_tokens]
reshape -> [head_dim * n_head_kv, n_tokens]
写入 cache 的对应 rows
```

V 写入类似：

```cpp
const int64_t n_embd_head = v_cur->ne[0];
const int64_t n_head      = v_cur->ne[1];
const int64_t n_tokens    = v_cur->ne[2];

const int64_t n_embd_gqa = n_embd_head * n_head;

v_cur = ggml_view_2d(ctx, v_cur,
    n_embd_gqa, n_tokens, v_cur->nb[2], 0);

return ggml_set_rows(ctx, v, v_cur, v_idxs);
```

这说明写入 cache 时没有 K/V repeat。

## 九、读取 KV Cache 并参与 attention

读取 K 的函数是 `get_k`：

```cpp
return ggml_view_4d(ctx, k,
    hparams.n_embd_head_k(il),
    hparams.n_head_kv(il),
    n_kv,
    ns,
    ...);
```

读取 V 的函数是 `get_v`：

```cpp
return ggml_view_4d(ctx, v,
    hparams.n_embd_head_v(il),
    hparams.n_head_kv(il),
    n_kv,
    ns,
    ...);
```

因此 attention 看到的 K/V 形状是：

```text
K: [head_dim, n_head_kv, n_kv, ns]
V: [head_dim, n_head_kv, n_kv, ns]
```

在 `llama-graph.cpp` 的带 KV cache attention 路径中：

```cpp
ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));

ggml_tensor * q = q_cur;
ggml_tensor * k = mctx_cur->get_k(ctx0, il);
ggml_tensor * v = mctx_cur->get_v(ctx0, il);

ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
```

`build_attn_mha` 内部先调整维度：

```cpp
q = ggml_permute(ctx0, q, 0, 2, 1, 3);
k = ggml_permute(ctx0, k, 0, 2, 1, 3);
v = ggml_permute(ctx0, v, 0, 2, 1, 3);
```

然后有两条路径：

### Flash Attention 路径

```cpp
cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, ...);
```

底层 flash attention op 根据 Q heads 和 KV heads 的形状完成 GQA 映射。

### 非 Flash Attention 路径

```cpp
ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, ...);
ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
```

从语义上看：

```text
K/Q 做 attention score；
softmax 后再乘 V；
K/V 的 head 维度少于 Q 时，底层 op 按 GQA 规则广播/映射。
```

llama.cpp 在高层图里保持 `K/V` 的 `n_head_kv` 形状，没有显式把它扩成 `n_head`。

## 十、手搓一个 llama.cpp 风格的 GQA

下面写一个接近 llama.cpp 变量命名的简化版。

### 参数结构

```cpp
struct HParams {
    int n_head;        // Q heads
    int n_head_kv;     // K/V heads
    int head_dim;

    int n_gqa() const {
        return n_head / n_head_kv;
    }

    int n_embd_k_gqa() const {
        return head_dim * n_head_kv;
    }

    int n_embd_v_gqa() const {
        return head_dim * n_head_kv;
    }
};
```

这对应 llama.cpp 的：

```cpp
n_gqa()
n_embd_k_gqa()
n_embd_v_gqa()
```

### Q/K/V reshape

假设输入投影后是一维连续数组：

```cpp
// q_proj: [n_tokens, n_head    * head_dim]
// k_proj: [n_tokens, n_head_kv * head_dim]
// v_proj: [n_tokens, n_head_kv * head_dim]
```

可以按下面方式理解 reshape：

```cpp
float q_at(const float* q_proj, int token, int q_head, int d,
           int n_head, int head_dim) {
    return q_proj[token * n_head * head_dim + q_head * head_dim + d];
}

float k_at(const float* k_proj, int token, int kv_head, int d,
           int n_head_kv, int head_dim) {
    return k_proj[token * n_head_kv * head_dim + kv_head * head_dim + d];
}
```

### KV Cache layout

llama.cpp 的 cache 宽度类似：

```text
n_embd_k_gqa = n_head_kv * head_dim
```

简化成 C++：

```cpp
struct KVCache {
    int max_seq;
    int n_head_kv;
    int head_dim;

    std::vector<float> k; // [max_seq, n_head_kv, head_dim]
    std::vector<float> v; // [max_seq, n_head_kv, head_dim]

    float& K(int pos, int kv_head, int d) {
        return k[(pos * n_head_kv + kv_head) * head_dim + d];
    }

    float& V(int pos, int kv_head, int d) {
        return v[(pos * n_head_kv + kv_head) * head_dim + d];
    }
};
```

注意：这里没有 `n_head`。

### 写入当前 token 的 K/V

```cpp
void write_kv_cache(
    KVCache& cache,
    int pos,
    const float* k_cur, // [n_head_kv, head_dim]
    const float* v_cur  // [n_head_kv, head_dim]
) {
    for (int hkv = 0; hkv < cache.n_head_kv; ++hkv) {
        for (int d = 0; d < cache.head_dim; ++d) {
            cache.K(pos, hkv, d) = k_cur[hkv * cache.head_dim + d];
            cache.V(pos, hkv, d) = v_cur[hkv * cache.head_dim + d];
        }
    }
}
```

### Decode attention

```cpp
void gqa_decode(
    const HParams& hp,
    const KVCache& cache,
    const float* q_cur, // [n_head, head_dim]
    float* out,         // [n_head, head_dim]
    int seq_len
) {
    int group_size = hp.n_gqa(); // n_head / n_head_kv

    for (int hq = 0; hq < hp.n_head; ++hq) {
        int hkv = hq / group_size;

        std::vector<float> score(seq_len);

        for (int t = 0; t < seq_len; ++t) {
            float dot = 0.0f;
            for (int d = 0; d < hp.head_dim; ++d) {
                float q = q_cur[hq * hp.head_dim + d];
                float k = cache.k[(t * hp.n_head_kv + hkv) * hp.head_dim + d];
                dot += q * k;
            }
            score[t] = dot / std::sqrt((float)hp.head_dim);
        }

        float max_score = score[0];
        for (int t = 1; t < seq_len; ++t) {
            max_score = std::max(max_score, score[t]);
        }

        float denom = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            score[t] = std::exp(score[t] - max_score);
            denom += score[t];
        }

        for (int d = 0; d < hp.head_dim; ++d) {
            float acc = 0.0f;
            for (int t = 0; t < seq_len; ++t) {
                float prob = score[t] / denom;
                float v = cache.v[(t * hp.n_head_kv + hkv) * hp.head_dim + d];
                acc += prob * v;
            }
            out[hq * hp.head_dim + d] = acc;
        }
    }
}
```

这一版代码有三个关键点：

```cpp
int group_size = hp.n_head / hp.n_head_kv;
int hkv = hq / group_size;
cache layout = [seq_len, n_head_kv, head_dim]
```

这就是 GQA 的最小实现。

## 十一、具体数字例子

设：

```text
n_head = 8
n_head_kv = 2
head_dim = 4
seq_len = 3
```

那么：

```text
n_gqa = 8 / 2 = 4
n_embd_k_gqa = 2 * 4 = 8
n_embd_v_gqa = 2 * 4 = 8
```

Q shape：

```text
q_cur: [8, 4]
```

KV Cache shape：

```text
k_cache: [3, 2, 4]
v_cache: [3, 2, 4]
```

head 映射：

| q_head | kv_head |
| ---: | ---: |
| 0 | 0 |
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 1 |
| 5 | 1 |
| 6 | 1 |
| 7 | 1 |

如果是 MHA：

```text
k_cache: [3, 8, 4]
v_cache: [3, 8, 4]
```

GQA 的 KV Cache 元素数：

```text
K + V = 3 * 2 * 4 * 2 = 48 elements
```

MHA 的 KV Cache 元素数：

```text
K + V = 3 * 8 * 4 * 2 = 192 elements
```

GQA 正好是 MHA 的 1/4。

## 十二、工程实现要点

### 1. 配置读取时要给 n_head_kv 默认值

llama.cpp 的设计很稳妥：

```cpp
hparams.n_head_kv_arr = hparams.n_head_arr;
ml.get_key_or_arr(... HEAD_COUNT_KV ...);
```

这样 MHA 模型也能走同一套逻辑。

### 2. Q/K/V reshape 必须用不同 head 数

正确：

```cpp
Q: [head_dim, n_head,    n_tokens]
K: [head_dim, n_head_kv, n_tokens]
V: [head_dim, n_head_kv, n_tokens]
```

错误：

```cpp
K/V 也 reshape 成 [head_dim, n_head, n_tokens]
```

那会破坏 GQA 的显存收益。

### 3. KV Cache 宽度看 n_embd_k_gqa / n_embd_v_gqa

llama.cpp 明确用：

```cpp
n_embd_k_gqa = n_embd_head_k * n_head_kv
n_embd_v_gqa = n_embd_head_v * n_head_kv
```

KV Cache 分配、读取、保存都围绕这两个值。

### 4. Attention kernel 里做 head 映射

在手写 kernel 中最直接的映射是：

```cpp
kv_head = q_head / (n_head / n_head_kv);
```

高性能框架可能把映射藏在 flash attention kernel 或 tensor shape/broadcast 规则里，但语义相同。

### 5. 不要物理 repeat KV Cache

物理 repeat 会变成：

```text
[seq_len, n_head_kv, head_dim] -> [seq_len, n_head, head_dim]
```

这样就失去 GQA 的主要意义。训练代码中为了简单可能临时 repeat，但推理 cache 不应该这样做。

## 十三、面试表达

可以这样讲 llama.cpp 的 GQA 实现：

```text
llama.cpp 从模型元数据里读取 n_head 和 n_head_kv。
如果没有 n_head_kv，就默认等于 n_head，因此 MHA 和 GQA 可以共用一套逻辑。
它定义 n_gqa = n_head / n_head_kv，表示每个 KV head 被多少个 Q heads 共享。
在 build_qkv 中，Q reshape 成 [head_dim, n_head, tokens]，K/V reshape 成 [head_dim, n_head_kv, tokens]。
KV Cache 分配时使用 n_embd_k_gqa = head_dim * n_head_kv，因此 cache 只存 KV heads，不会 repeat 到 Q heads。
attention 计算时通过 q_head 到 kv_head 的映射完成共享。
```

如果面试官追问怎么手搓：

```text
核心是 cache layout 和 head mapping。
cache 存 [seq_len, num_kv_heads, head_dim]。
每个 query head hq 对应 kv_head = hq / (num_q_heads / num_kv_heads)。
然后对该 kv_head 的历史 K 做 dot，softmax 后加权该 kv_head 的历史 V。
```

如果面试官问 GQA 为什么省显存：

```text
KV Cache 大小和 num_kv_heads 成正比。
MHA 中 num_kv_heads = num_q_heads；
GQA 中 num_kv_heads 更少。
例如 32 个 Q heads、8 个 KV heads，KV Cache 是 MHA 的 1/4。
```

## 十四、总结

llama.cpp 的 GQA 实现体现了一个高性能推理框架该有的做法：

```text
参数层：读取 n_head_kv，默认等于 n_head。
形状层：Q 用 n_head，K/V 用 n_head_kv。
缓存层：KV Cache 宽度是 head_dim * n_head_kv。
计算层：attention 中让多个 Q heads 共享同一个 KV head。
```

GQA 的代码核心不复杂，难点在于所有环节都不能把 `n_head` 和 `n_head_kv` 混用。

最应该记住的源码级公式是：

```cpp
n_gqa = n_head / n_head_kv;
n_embd_k_gqa = n_embd_head_k * n_head_kv;
n_embd_v_gqa = n_embd_head_v * n_head_kv;
```

最应该记住的手搓公式是：

```cpp
kv_head = q_head / n_gqa;
```

其中：

```cpp
n_gqa = num_q_heads / num_kv_heads;
```

这就是 GQA 在工程实现里的主线。
