---
title: '并发原语：条件变量、信号量与死锁'
description: '用生产者消费者模型理解条件变量和信号量，梳理死锁的四个必要条件与常见预防方法。'
pubDate: '2026-06-06'
updatedDate: '2026-06-06'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [条件变量解决什么问题](#二条件变量解决什么问题)
3. [生产者消费者示例](#三生产者消费者示例)
4. [信号量解决什么问题](#四信号量解决什么问题)
5. [条件变量和信号量的区别](#五条件变量和信号量的区别)
6. [死锁的四个必要条件](#六死锁的四个必要条件)
7. [死锁预防方法](#七死锁预防方法)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

并发原语用于协调线程之间的执行顺序和资源访问。

- 条件变量用于“等待某个条件成立”，通常要和互斥锁一起使用。
- 条件变量必须用循环检查条件，防止虚假唤醒和条件被其他线程抢先消费。
- 信号量维护一个计数，适合控制资源数量或并发额度。
- 死锁通常需要同时满足互斥、占有且等待、不可抢占、循环等待四个条件。
- 预防死锁的核心是破坏其中至少一个条件，常见方法是固定加锁顺序、减少锁持有范围、使用超时和 RAII。

## 二、条件变量解决什么问题

条件变量让线程在条件不满足时睡眠，等条件变化后再被唤醒。

典型场景：队列为空时，消费者不应一直空转，而应该等待生产者放入任务。

```text
Consumer: queue empty -> wait
Producer: push task -> notify
Consumer: wake up -> pop task
```

条件变量需要和互斥锁搭配，因为“检查条件”和“修改共享状态”必须是同步的。

## 三、生产者消费者示例

```cpp
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

std::mutex mu;
std::condition_variable cv;
std::queue<int> tasks;
bool done = false;

void producer() {
    for (int i = 0; i < 10; ++i) {
        {
            std::lock_guard<std::mutex> lock(mu);
            tasks.push(i);
        }
        cv.notify_one();
    }

    {
        std::lock_guard<std::mutex> lock(mu);
        done = true;
    }
    cv.notify_all();
}

void consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [] {
            return !tasks.empty() || done;
        });

        if (tasks.empty() && done) {
            break;
        }

        int task = tasks.front();
        tasks.pop();
        lock.unlock();

        // process task
        (void)task;
    }
}
```

### 为什么 wait 要带条件

```cpp
cv.wait(lock, [] { return !tasks.empty() || done; });
```

这等价于循环检查：

```cpp
while (!condition) {
    cv.wait(lock);
}
```

原因有两个：

- 可能发生虚假唤醒，即没有明确通知也被唤醒。
- 多个消费者被唤醒后，任务可能已经被其他线程取走。

### wait 会自动释放锁

`wait` 在睡眠前会释放互斥锁，被唤醒后会重新加锁，然后再检查条件。否则生产者无法拿到锁修改队列，消费者也就永远等不到条件变化。

## 四、信号量解决什么问题

信号量维护一个计数，用来表示可用资源数量。

C++20 提供 `std::counting_semaphore`。

```cpp
#include <semaphore>
#include <thread>
#include <vector>

std::counting_semaphore<3> slots(3);  // 最多允许 3 个线程同时进入

void work() {
    slots.acquire();

    // 访问有限资源

    slots.release();
}
```

如果计数大于 0，`acquire` 会减少计数并继续执行；如果计数为 0，线程会等待。`release` 增加计数并可能唤醒等待线程。

### 二值信号量

二值信号量的计数只有 0 和 1，功能类似互斥控制，但语义不完全等同于 mutex。

```cpp
#include <semaphore>

std::binary_semaphore sem(0);
```

它常用于线程间通知：一个线程 `release`，另一个线程 `acquire`。

## 五、条件变量和信号量的区别

| 对比项 | 条件变量 | 信号量 |
| --- | --- | --- |
| 核心语义 | 等待某个条件成立 | 管理资源计数 |
| 是否保存通知 | 不保存，通知时没人等可能丢失 | 保存计数 |
| 是否需要配合锁 | 通常需要 | 不一定 |
| 典型场景 | 队列非空、状态变化 | 限流、资源池、并发额度 |

条件变量更像“状态变化通知”，信号量更像“资源许可证”。

## 六、死锁的四个必要条件

死锁指多个线程互相等待对方释放资源，导致所有线程都无法继续执行。

经典例子：

```cpp
#include <mutex>
#include <thread>

std::mutex a;
std::mutex b;

void t1() {
    std::lock_guard<std::mutex> lock_a(a);
    std::lock_guard<std::mutex> lock_b(b);
}

void t2() {
    std::lock_guard<std::mutex> lock_b(b);
    std::lock_guard<std::mutex> lock_a(a);
}
```

`t1` 先拿 `a` 再等 `b`，`t2` 先拿 `b` 再等 `a`，就可能死锁。

死锁的四个必要条件：

1. 互斥：资源同一时刻只能被一个线程持有。
2. 占有且等待：线程持有资源的同时等待其他资源。
3. 不可抢占：资源不能被强制从持有者手中夺走。
4. 循环等待：多个线程形成等待环。

## 七、死锁预防方法

### 1. 固定加锁顺序

所有线程都按同一顺序加锁，破坏循环等待。

```cpp
void safe() {
    std::lock_guard<std::mutex> lock_a(a);
    std::lock_guard<std::mutex> lock_b(b);
}
```

### 2. 使用 std::lock 一次性加多把锁

```cpp
#include <mutex>

void safe_transfer() {
    std::lock(a, b);
    std::lock_guard<std::mutex> lock_a(a, std::adopt_lock);
    std::lock_guard<std::mutex> lock_b(b, std::adopt_lock);
}
```

更现代的写法是 `std::scoped_lock`：

```cpp
void safe_transfer() {
    std::scoped_lock lock(a, b);
}
```

`std::scoped_lock` 能同时管理多把锁，并减少死锁风险。

### 3. 缩短锁持有时间

锁内只做必要操作，不在锁内执行慢 I/O、网络请求或未知回调。

### 4. 使用 try_lock 或超时

无法获得锁时主动放弃并重试，可以避免永久等待。

```cpp
std::timed_mutex mu;

if (mu.try_lock_for(std::chrono::milliseconds(10))) {
    // critical section
    mu.unlock();
}
```

### 5. 减少共享资源

不可变数据、线程局部存储、消息传递都可以减少锁需求。

## 八、面试回答模板

如果问题是“条件变量怎么用”，可以这样回答：

1. 条件变量用于线程等待某个条件成立，通常和互斥锁一起使用。
2. 等待线程先持有锁检查条件，不满足则调用 `wait`。
3. `wait` 会原子地释放锁并睡眠，被唤醒后重新加锁。
4. 必须用循环或带谓词版本检查条件，防止虚假唤醒。
5. 修改共享条件后调用 `notify_one` 或 `notify_all` 唤醒等待线程。

如果问题是“死锁产生条件和预防”，可以回答：

1. 死锁需要互斥、占有且等待、不可抢占、循环等待四个条件。
2. 预防死锁就是破坏其中一个条件。
3. 工程中常用固定加锁顺序、`std::scoped_lock`、缩小临界区、超时重试和减少共享资源。

## 九、总结

条件变量、信号量和死锁是并发协调的基础：

- 条件变量等待状态变化，重点是“条件”。
- 信号量管理资源数量，重点是“计数”。
- 死锁来自资源等待环，重点是“加锁顺序”和“锁持有范围”。
- 并发代码的可靠性比技巧更重要，优先写清晰、可证明的同步逻辑。
