---
title: '内存层级：Cache Line、伪共享与性能优化'
description: '从 CPU Cache 和 Cache Line 出发，理解内存层级、局部性、伪共享问题，以及多线程程序中如何减少缓存争用。'
pubDate: '2026-06-07'
updatedDate: '2026-06-07'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [为什么内存层级重要](#二为什么内存层级重要)
3. [CPU Cache 和 Cache Line](#三cpu-cache-和-cache-line)
4. [局部性原理](#四局部性原理)
5. [伪共享是什么](#五伪共享是什么)
6. [如何避免伪共享](#六如何避免伪共享)
7. [工程优化思路](#七工程优化思路)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

内存层级决定了程序访问数据的真实成本。

- CPU 访问寄存器最快，访问内存慢得多。
- Cache 用来缓解 CPU 和内存之间的速度差距。
- Cache Line 是 CPU Cache 和内存之间搬运数据的基本单位，常见大小是 64 字节。
- 程序访问连续内存通常更快，因为能更好利用缓存局部性。
- 伪共享指多个线程修改不同变量，但这些变量落在同一个 Cache Line 中，导致缓存行频繁失效。
- 避免伪共享常用 padding、alignas、按线程分片统计等方法。

## 二、为什么内存层级重要

现代 CPU 很快，内存相对很慢。如果每次读写都直接访问主存，CPU 会大量等待。

典型层级可以简化为：

```text
Register  -> fastest, smallest
L1 Cache
L2 Cache
L3 Cache
Memory
Disk      -> slowest, largest
```

越靠近 CPU，容量越小、速度越快。性能优化很多时候不是减少“指令条数”，而是减少“慢内存访问”。

## 三、CPU Cache 和 Cache Line

CPU Cache 不是按单个字节加载数据，而是按 Cache Line 加载。常见 Cache Line 大小是 64 字节。

如果读取一个 `int`，CPU 可能会把它附近连续 64 字节都加载进 Cache。

```text
Memory
+------------------------------------------------+
| int | int | int | int | ... total 64 bytes ... |
+------------------------------------------------+
             ^
             read one int, load whole cache line
```

这解释了为什么连续数组遍历通常很快。

```cpp
#include <vector>

int sum_vector(const std::vector<int>& v) {
    int sum = 0;
    for (int x : v) {
        sum += x;
    }
    return sum;
}
```

`std::vector` 连续存储，顺序访问时硬件预取和 Cache Line 都能发挥作用。

链表则不同：

```cpp
struct Node {
    int value;
    Node* next;
};
```

链表节点可能分散在堆上，遍历时容易产生 Cache Miss。

## 四、局部性原理

### 时间局部性

如果一个数据刚被访问过，短时间内可能还会被访问。

```cpp
for (int i = 0; i < n; ++i) {
    acc += data[i];
    acc += data[i];
}
```

同一个元素重复使用，容易命中 Cache。

### 空间局部性

如果一个地址被访问，它附近地址也可能很快被访问。

```cpp
for (int i = 0; i < n; ++i) {
    acc += data[i];
}
```

连续访问数组能充分利用 Cache Line。

### 行主序访问

C/C++ 二维数组按行连续存储。

```cpp
int a[1024][1024];
long long sum = 0;

for (int i = 0; i < 1024; ++i) {
    for (int j = 0; j < 1024; ++j) {
        sum += a[i][j];  // 连续访问
    }
}
```

如果交换循环顺序，访问步长变大，Cache 命中率可能下降。

## 五、伪共享是什么

伪共享发生在多线程写入不同变量，但这些变量位于同一个 Cache Line。

```cpp
#include <atomic>

struct Counters {
    std::atomic<long long> a;
    std::atomic<long long> b;
};

Counters counters;
```

假设 `a` 和 `b` 落在同一个 Cache Line：

```text
Cache Line 64B
+------------------+------------------+
| counters.a       | counters.b       |
+------------------+------------------+
Thread 1 writes a   Thread 2 writes b
```

即使两个线程没有写同一个变量，硬件缓存一致性协议也以 Cache Line 为单位维护状态。线程 1 写 `a` 会让其他核心中包含 `b` 的同一缓存行失效；线程 2 写 `b` 又会让线程 1 的缓存行失效。

结果是：数据逻辑上没有共享，硬件层面却在争同一条 Cache Line，这就是“伪共享”。

### 一个容易出问题的计数器数组

```cpp
#include <atomic>
#include <thread>

std::atomic<long long> counters[2];

void worker(int id) {
    for (int i = 0; i < 10000000; ++i) {
        counters[id].fetch_add(1, std::memory_order_relaxed);
    }
}
```

`counters[0]` 和 `counters[1]` 很可能相邻，落入同一个 Cache Line。两个线程分别更新不同元素，也可能互相拖慢。

## 六、如何避免伪共享

### 1. 使用 alignas 对齐

C++17 提供 `std::hardware_destructive_interference_size`，表示避免破坏性干扰的推荐间隔。实际使用时也常见直接按 64 字节对齐。

```cpp
#include <atomic>

struct alignas(64) PaddedCounter {
    std::atomic<long long> value{0};
};

PaddedCounter counters[2];
```

这样每个计数器尽量独占一个 Cache Line。

### 2. 手动 padding

```cpp
#include <atomic>

struct CounterSlot {
    std::atomic<long long> value{0};
    char padding[64 - sizeof(std::atomic<long long>)];
};
```

这种写法直观，但要注意不同平台 Cache Line 大小、对齐和结构体布局问题。

### 3. 按线程本地累加，最后归并

```cpp
#include <vector>

long long sum_all(const std::vector<int>& data, int thread_count) {
    std::vector<long long> local_sum(thread_count, 0);

    // 每个线程只写自己的 local_sum[id]
    // 最后单线程归并 local_sum

    long long total = 0;
    for (long long x : local_sum) {
        total += x;
    }
    return total;
}
```

如果 `local_sum` 写入频繁，也要注意它自身可能伪共享，因此可以配合 padding。

## 七、工程优化思路

### 1. 优先使用连续内存

`vector`、数组、结构体数组在顺序访问时通常比链表更缓存友好。

### 2. 区分读共享和写共享

只读共享通常问题较小；频繁写共享会引发缓存一致性流量。

### 3. 减少跨线程写同一数据结构

让每个线程处理自己的分片，最后归并，通常比所有线程抢同一个全局计数器更快。

### 4. 注意结构体布局

热点字段和冷字段可以拆开，频繁写的字段避免和其他线程热点字段放在同一 Cache Line。

```cpp
struct HotData {
    int frequently_updated;
};

struct ColdData {
    std::string name;
    std::string debug_info;
};
```

### 5. 用性能工具验证

伪共享和缓存问题不能只靠猜。Linux 上可以用 `perf` 观察 cache miss、LLC miss 等指标；Intel 平台也可以用 VTune 做更细分析。

## 八、面试回答模板

如果问题是“什么是 Cache Line”，可以这样回答：

1. Cache Line 是 CPU Cache 和内存之间传输数据的基本单位，常见大小是 64 字节。
2. CPU 访问某个地址时，通常会把该地址所在的一整行加载进 Cache。
3. 连续内存访问能更好利用空间局部性，提高 Cache 命中率。
4. 多线程写入同一 Cache Line 上的不同变量，可能引发伪共享。

如果问题是“什么是伪共享，如何避免”，可以回答：

1. 伪共享是多个线程访问不同变量，但这些变量位于同一个 Cache Line，导致缓存行频繁失效。
2. 它的根因是缓存一致性协议以 Cache Line 为单位工作。
3. 避免方式包括 padding、`alignas(64)`、按线程分片统计、减少共享写。
4. 最终应通过性能工具验证，而不是只凭代码结构猜测。

## 九、总结

内存层级是高性能代码的基础：

- CPU 不是按单个变量和内存交互，而是按 Cache Line 搬运数据。
- 连续访问通常比随机访问更快。
- 多线程写共享数据时，缓存一致性成本可能比锁本身更隐蔽。
- 伪共享的变量逻辑上不同，硬件上却共享同一缓存行。
- 数据布局、线程分片和对齐是优化缓存性能的重要手段。
