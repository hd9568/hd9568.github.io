---
title: 'C++11 并发：std::thread、std::atomic 与内存序'
description: '用简单示例理解 C++11 std::thread、std::atomic，以及 Relaxed、Acquire-Release、Sequential Consistency 三类基础内存序。'
category: '操作系统与多线程'
pubDate: '2026-05-29T10:50:00+08:00'
updatedDate: '2026-05-29T10:50:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [std::thread 基础](#二stdthread-基础)
3. [数据竞争与互斥](#三数据竞争与互斥)
4. [std::atomic 解决什么问题](#四stdatomic-解决什么问题)
5. [为什么需要内存序](#五为什么需要内存序)
6. [Relaxed 内存序](#六relaxed-内存序)
7. [Acquire-Release 内存序](#七acquire-release-内存序)
8. [Sequential Consistency](#八sequential-consistency)
9. [工程使用建议](#九工程使用建议)
10. [面试回答模板](#十面试回答模板)
11. [总结](#十一总结)

## 一、核心结论

C++11 标准库提供了跨平台线程和原子操作能力。

- `std::thread` 用于创建线程，线程对象必须 `join` 或 `detach`。
- 多线程读写同一普通变量且没有同步，会产生数据竞争，属于未定义行为。
- `std::atomic<T>` 提供原子读写和读改写操作，可用于无锁计数、标志位等场景。
- 内存序控制原子操作周围的可见性和重排序约束。
- `memory_order_relaxed` 只保证原子性，不保证跨线程同步顺序。
- Acquire-Release 常用于“一个线程发布数据，另一个线程读取数据”的同步模式。
- Sequential Consistency 是默认内存序，语义最强，也最容易理解。

## 二、std::thread 基础

创建线程：

```cpp
#include <iostream>
#include <thread>

void worker() {
    std::cout << "hello from worker\n";
}

int main() {
    std::thread t(worker);
    t.join();
}
```

`join` 表示等待线程结束。

如果一个 `std::thread` 对象在析构时仍然是 joinable 状态，程序会调用 `std::terminate`。

```cpp
std::thread t(worker);
// 如果这里既不 join 也不 detach，t 析构时会 terminate
```

### detach

```cpp
std::thread t(worker);
t.detach();
```

`detach` 后线程独立运行，线程对象不再管理它。工程中要谨慎使用，因为 detached 线程生命周期不容易控制，可能访问已经销毁的对象。

### 传参注意引用

```cpp
#include <functional>
#include <thread>

void add(int& x) {
    ++x;
}

int main() {
    int value = 0;
    std::thread t(add, std::ref(value));
    t.join();
}
```

`std::thread` 默认会拷贝参数。如果需要传引用，需要使用 `std::ref`。

## 三、数据竞争与互斥

下面代码有数据竞争：

```cpp
#include <thread>

int counter = 0;

void add() {
    for (int i = 0; i < 100000; ++i) {
        ++counter;
    }
}

int main() {
    std::thread t1(add);
    std::thread t2(add);
    t1.join();
    t2.join();
}
```

`++counter` 不是原子操作，两个线程可能同时读写同一变量。

使用互斥锁：

```cpp
#include <mutex>

std::mutex mu;
int counter = 0;

void add() {
    for (int i = 0; i < 100000; ++i) {
        std::lock_guard<std::mutex> lock(mu);
        ++counter;
    }
}
```

如果只是计数器，也可以使用原子变量。

## 四、std::atomic 解决什么问题

`std::atomic` 保证对变量的操作是原子的。

```cpp
#include <atomic>
#include <thread>

std::atomic<int> counter{0};

void add() {
    for (int i = 0; i < 100000; ++i) {
        counter.fetch_add(1, std::memory_order_relaxed);
    }
}
```

`fetch_add` 是读-改-写原子操作，不会出现多个线程更新丢失的问题。

常见原子操作：

- `load`：原子读取。
- `store`：原子写入。
- `exchange`：替换值并返回旧值。
- `compare_exchange_weak/strong`：CAS。
- `fetch_add/fetch_sub`：原子加减。

## 五、为什么需要内存序

多线程程序中，编译器和 CPU 都可能为了性能重排序指令。单线程看结果不变，但跨线程观察时，顺序可能影响正确性。

例如一个线程写数据后设置标志，另一个线程看到标志后读取数据：

```cpp
int data = 0;
std::atomic<bool> ready{false};
```

期望顺序：

```text
Thread A: data = 42; ready = true;
Thread B: if ready == true, read data == 42;
```

如果没有合适同步，另一个线程可能看到 `ready`，却没有正确看到 `data` 的写入。内存序就是用来描述这些可见性和顺序约束的。

## 六、Relaxed 内存序

`memory_order_relaxed` 只保证原子操作本身不会被撕裂，不提供线程间同步关系。

适合只关心计数准确性、不关心其他数据顺序的场景。

```cpp
#include <atomic>

std::atomic<long long> request_count{0};

void on_request() {
    request_count.fetch_add(1, std::memory_order_relaxed);
}
```

这里每次加一是原子的，但不要求它和其他内存读写建立顺序关系。

### 不适合发布数据

```cpp
int data = 0;
std::atomic<bool> ready{false};

void producer() {
    data = 42;
    ready.store(true, std::memory_order_relaxed);
}

void consumer() {
    if (ready.load(std::memory_order_relaxed)) {
        // 不应依赖这里一定能看到 data == 42
    }
}
```

如果要用标志位发布数据，通常需要 Acquire-Release。

## 七、Acquire-Release 内存序

Release 用于发布数据，Acquire 用于获取数据。

```cpp
#include <atomic>
#include <iostream>
#include <thread>

int data = 0;
std::atomic<bool> ready{false};

void producer() {
    data = 42;
    ready.store(true, std::memory_order_release);
}

void consumer() {
    while (!ready.load(std::memory_order_acquire)) {
    }
    std::cout << data << '\n';  // 能看到 producer 中 ready 之前的写入
}
```

含义：

- `release store` 之前的写入，不能被重排到 release 之后。
- `acquire load` 之后的读取，不能被重排到 acquire 之前。
- 当 acquire load 读到 release store 写入的值时，二者建立同步关系。

这常用于生产者发布数据、消费者等待标志的场景。

## 八、Sequential Consistency

`memory_order_seq_cst` 是默认内存序，也是最强、最直观的内存序。

```cpp
std::atomic<int> x{0};

x.store(1);  // 默认 seq_cst
int v = x.load();
```

它提供一个全局一致顺序，让所有线程对原子操作的观察更接近“按某个单一顺序执行”。

优点：

- 容易理解。
- 不容易写错。
- 适合大多数非极致性能场景。

代价：

- 约束更强，可能限制编译器和 CPU 优化。
- 在某些架构上成本可能更高。

## 九、工程使用建议

### 1. 默认先用锁

锁的语义清晰，适合保护复杂不变量。原子操作适合简单状态或计数器。

### 2. 默认原子用 seq_cst

除非明确知道为什么需要更弱内存序，否则使用默认内存序更安全。

```cpp
std::atomic<bool> stop{false};
stop.store(true);       // seq_cst
bool s = stop.load();   // seq_cst
```

### 3. 计数器可用 relaxed

只做统计、不参与同步时，`relaxed` 通常足够。

```cpp
metrics.fetch_add(1, std::memory_order_relaxed);
```

### 4. 发布-订阅用 acquire-release

一个线程写数据并发布标志，另一个线程读标志后读数据，适合 acquire-release。

### 5. 避免手写复杂无锁结构

无锁队列、栈、内存回收涉及 ABA、内存序、对象生命周期等复杂问题。工程中应优先使用成熟库或经过充分验证的实现。

## 十、面试回答模板

如果问题是“std::thread 使用时要注意什么”，可以这样回答：

1. `std::thread` 创建后必须 `join` 或 `detach`，否则析构时会 `terminate`。
2. `join` 是等待线程结束，`detach` 是让线程独立运行。
3. 传引用参数需要 `std::ref`。
4. 多线程访问共享数据必须同步，否则会有数据竞争。

如果问题是“atomic 和内存序怎么理解”，可以这样回答：

1. `std::atomic` 保证单个变量操作的原子性，避免数据竞争。
2. 内存序描述原子操作和周围普通内存访问之间的可见性与重排序约束。
3. `relaxed` 只保证原子性，不建立同步关系，适合计数器。
4. release-acquire 用于发布和获取数据，一个线程 release 写标志，另一个线程 acquire 读到标志后能看到之前写入。
5. seq_cst 是默认最强内存序，提供全局一致顺序，最容易理解。

## 十一、总结

C++11 并发的关键是区分三个层次：

- `std::thread` 负责并发执行。
- `mutex`、条件变量等负责线程协调。
- `std::atomic` 和内存序负责更底层的原子访问与可见性。

实际工程中，优先选择清晰同步模型。只有在性能瓶颈明确、语义完全理解时，才应使用更弱内存序和复杂无锁代码。
