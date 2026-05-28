---
title: '并发控制：Mutex、读写锁与自旋锁'
description: '从临界区、阻塞等待和忙等三个角度理解互斥锁、读写锁、自旋锁的使用场景与性能差异。'
category: '操作系统与多线程'
pubDate: '2026-06-05'
updatedDate: '2026-06-05'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [为什么需要锁](#二为什么需要锁)
3. [互斥锁 Mutex](#三互斥锁-mutex)
4. [读写锁 Reader-Writer Lock](#四读写锁-reader-writer-lock)
5. [自旋锁 Spinlock](#五自旋锁-spinlock)
6. [三类锁的性能差异](#六三类锁的性能差异)
7. [常见优化思路](#七常见优化思路)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

锁的本质是保护共享状态，避免多个线程同时访问临界区导致数据竞争。

- `Mutex` 适合通用互斥场景，竞争失败的线程通常会阻塞或让出 CPU。
- 读写锁适合读多写少，允许多个读线程并发进入，但写线程需要独占。
- 自旋锁适合临界区极短、等待时间极短的场景，等待时不会睡眠，而是忙等。
- 锁不是越高级越好，关键要看临界区长度、竞争强度、读写比例和线程调度成本。
- 大多数业务代码优先使用 `std::mutex`，确认瓶颈后再考虑读写锁或自旋锁。

## 二、为什么需要锁

多个线程同时修改共享变量时，会产生数据竞争。

```cpp
#include <thread>
#include <vector>

int counter = 0;

void add() {
    for (int i = 0; i < 100000; ++i) {
        ++counter;  // 数据竞争
    }
}

int main() {
    std::thread t1(add);
    std::thread t2(add);
    t1.join();
    t2.join();
}
```

`++counter` 看起来是一条语句，但底层通常包含读取、加一、写回多个步骤。两个线程交错执行时，更新可能丢失。

用锁保护临界区：

```cpp
#include <mutex>
#include <thread>

int counter = 0;
std::mutex m;

void add() {
    for (int i = 0; i < 100000; ++i) {
        std::lock_guard<std::mutex> lock(m);
        ++counter;
    }
}
```

`std::lock_guard` 利用 RAII 自动加锁和解锁，避免忘记释放锁。

## 三、互斥锁 Mutex

互斥锁保证同一时刻只有一个线程进入临界区。

```cpp
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

class CounterMap {
public:
    void inc(const std::string& key) {
        std::lock_guard<std::mutex> lock(mu_);
        ++data_[key];
    }

    int get(const std::string& key) {
        std::lock_guard<std::mutex> lock(mu_);
        return data_[key];
    }

private:
    std::mutex mu_;
    std::unordered_map<std::string, int> data_;
};
```

特点：

- 简单可靠。
- 适合读写都需要互斥的场景。
- 竞争激烈时，线程可能阻塞并发生上下文切换。

### 临界区要尽量短

```cpp
void bad() {
    std::lock_guard<std::mutex> lock(mu_);
    slow_io();          // 不建议在锁内做慢 I/O
    update_shared();
}
```

更好的方式是把慢操作移出临界区：

```cpp
void good() {
    auto result = slow_io();

    std::lock_guard<std::mutex> lock(mu_);
    update_shared(result);
}
```

## 四、读写锁 Reader-Writer Lock

读写锁允许多个读者同时访问，但写者独占。

C++17 提供 `std::shared_mutex`。

```cpp
#include <shared_mutex>
#include <string>
#include <unordered_map>

class Cache {
public:
    int get(const std::string& key) const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        auto it = data_.find(key);
        return it == data_.end() ? -1 : it->second;
    }

    void put(const std::string& key, int value) {
        std::unique_lock<std::shared_mutex> lock(mu_);
        data_[key] = value;
    }

private:
    mutable std::shared_mutex mu_;
    std::unordered_map<std::string, int> data_;
};
```

- `shared_lock`：共享读锁，多个读线程可以同时持有。
- `unique_lock`：独占写锁，写入时不能有其他读者或写者。

### 适用场景

读写锁适合读多写少。例如配置中心、本地缓存、模型元信息查询等。

如果写很多，读写锁不一定比互斥锁快，因为它维护读写状态本身也有成本。

### 饥饿问题

某些读写锁实现可能偏向读者或写者。如果读请求源源不断，写线程可能长期等不到锁；如果偏向写者，读线程延迟可能变大。具体行为取决于实现。

## 五、自旋锁 Spinlock

自旋锁在获取失败时不会睡眠，而是在循环中不断尝试。

```cpp
#include <atomic>

class SpinLock {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // busy wait
        }
    }

    void unlock() {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};
```

使用：

```cpp
class Guard {
public:
    explicit Guard(SpinLock& lock) : lock_(lock) {
        lock_.lock();
    }

    ~Guard() {
        lock_.unlock();
    }

private:
    SpinLock& lock_;
};
```

特点：

- 不发生睡眠和唤醒，避免上下文切换。
- 等待期间持续占用 CPU。
- 适合临界区非常短且竞争时间极短的场景。

如果锁持有时间长，自旋锁会浪费大量 CPU。

## 六、三类锁的性能差异

| 锁类型 | 等待方式 | 适合场景 | 风险 |
| --- | --- | --- | --- |
| Mutex | 阻塞或睡眠 | 通用临界区 | 上下文切换开销 |
| 读写锁 | 读共享、写独占 | 读多写少 | 写饥饿或维护成本 |
| 自旋锁 | 忙等 | 极短临界区 | 浪费 CPU |

### 选择思路

- 临界区较长：优先 `mutex`，避免自旋浪费 CPU。
- 读远多于写：考虑读写锁。
- 临界区极短且在内核/运行时底层：可以考虑自旋锁。
- 高频共享计数：可能用原子变量比锁更合适。

## 七、常见优化思路

### 1. 缩小临界区

只把真正需要保护的共享状态放进锁里。

### 2. 分片锁

把一把大锁拆成多把小锁，减少竞争。

```cpp
static constexpr int kShardCount = 16;

struct Shard {
    std::mutex mu;
    std::unordered_map<int, int> data;
};
```

根据 key 选择不同分片，不同 key 可以并发操作。

### 3. 避免锁内调用未知代码

锁内调用回调、虚函数、外部模块函数，可能导致不可控耗时或死锁。

### 4. 使用 RAII 管理锁

```cpp
std::lock_guard<std::mutex> lock(mu);
```

相比手写 `lock()` 和 `unlock()`，RAII 更安全，异常或提前返回时也能释放锁。

## 八、面试回答模板

如果问题是“Mutex、读写锁、自旋锁有什么区别”，可以这样回答：

1. `Mutex` 是互斥锁，同一时刻只允许一个线程进入临界区，通用但竞争时可能阻塞和上下文切换。
2. 读写锁区分读和写，多个读者可以并发，写者独占，适合读多写少。
3. 自旋锁获取失败时忙等，不让出 CPU，适合临界区极短、等待时间极短的场景。
4. 如果锁持有时间长，自旋锁会浪费 CPU；如果写多，读写锁可能不如普通 mutex。
5. 选择锁要看临界区长度、竞争强度、读写比例和调度成本。

## 九、总结

锁的选择不是背 API，而是理解等待成本：

- `Mutex` 用阻塞换 CPU，不适合过度竞争。
- 读写锁用更复杂的状态管理换读并发。
- 自旋锁用 CPU 忙等换低延迟。
- 工程上优先保持临界区短、锁粒度合理、所有权清晰。
