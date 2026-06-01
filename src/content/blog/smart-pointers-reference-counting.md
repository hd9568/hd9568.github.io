---
title: '智能指针：unique_ptr、shared_ptr、weak_ptr 与引用计数'
description: '用所有权模型理解 C++ 智能指针，讲清 unique_ptr、shared_ptr、weak_ptr 的底层思路、引用计数和循环引用问题。'
category: 'C++'
pubDate: '2026-05-29T09:10:00+08:00'
updatedDate: '2026-05-29T09:10:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [为什么需要智能指针](#二为什么需要智能指针)
3. [unique_ptr：独占所有权](#三unique_ptr独占所有权)
4. [shared_ptr：共享所有权与引用计数](#四shared_ptr共享所有权与引用计数)
5. [weak_ptr：观察 shared_ptr 管理的对象](#五weak_ptr观察-shared_ptr-管理的对象)
6. [循环引用问题](#六循环引用问题)
7. [常见误区](#七常见误区)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

智能指针的核心不是“把裸指针包装一下”，而是把资源生命周期和对象生命周期绑定起来。

- `std::unique_ptr` 表示独占所有权，同一时刻只有一个对象负责释放资源。
- `std::shared_ptr` 表示共享所有权，通过控制块中的引用计数决定何时释放资源。
- `std::weak_ptr` 不拥有对象，只观察 `shared_ptr` 管理的对象，用来打破循环引用。
- `shared_ptr` 的引用计数通常是线程安全增减的，但被管理对象本身并不会自动线程安全。
- 优先使用 `make_unique`、`make_shared`，避免裸 `new` 和重复托管同一个裸指针。

## 二、为什么需要智能指针

裸指针最大的问题是：指针只保存地址，不表达“谁负责释放”。

```cpp
void work() {
    int* p = new int(42);

    if (*p > 0) {
        return;  // 忘记 delete，内存泄漏
    }

    delete p;
}
```

这段代码在提前 `return` 时没有释放 `p`。真实工程中，异常、多个分支、复杂对象关系都会让手写 `delete` 变得很脆弱。

智能指针利用 RAII：对象构造时获得资源，析构时释放资源。

```cpp
#include <memory>

void work() {
    auto p = std::make_unique<int>(42);

    if (*p > 0) {
        return;  // 函数返回时 p 析构，资源自动释放
    }
}
```

`p` 离开作用域时自动析构，被管理的 `int` 也会被释放。

## 三、unique_ptr：独占所有权

`std::unique_ptr<T>` 表示只有一个智能指针拥有对象。

```cpp
#include <iostream>
#include <memory>

int main() {
    std::unique_ptr<int> p = std::make_unique<int>(10);
    std::cout << *p << '\n';
}
```

### 不能拷贝

独占所有权不能被复制，否则就会出现两个对象都想释放同一块内存的问题。

```cpp
std::unique_ptr<int> p1 = std::make_unique<int>(10);
// std::unique_ptr<int> p2 = p1;  // 编译错误
```

### 可以移动

所有权可以转移。

```cpp
#include <memory>

std::unique_ptr<int> p1 = std::make_unique<int>(10);
std::unique_ptr<int> p2 = std::move(p1);

// p1 变为空，p2 拥有原来的 int
```

`std::move` 本身不移动数据，它只是把表达式转换为右值，允许调用移动构造或移动赋值。

### 典型使用场景

`unique_ptr` 适合表达明确的单一所有权：

```cpp
#include <memory>
#include <vector>

struct KernelTask {
    int id;
};

int main() {
    std::vector<std::unique_ptr<KernelTask>> tasks;
    tasks.push_back(std::make_unique<KernelTask>(KernelTask{1}));
}
```

容器拥有这些任务对象，任务不需要被多个地方共同释放。

## 四、shared_ptr：共享所有权与引用计数

`std::shared_ptr<T>` 允许多个智能指针共同拥有同一个对象。当最后一个拥有者消失时，对象才会被释放。

```cpp
#include <iostream>
#include <memory>

int main() {
    auto p1 = std::make_shared<int>(10);
    auto p2 = p1;

    std::cout << p1.use_count() << '\n';  // 2
}
```

### shared_ptr 的底层结构

一个 `shared_ptr` 通常包含两部分信息：

- 指向被管理对象的指针。
- 指向控制块的指针。

控制块通常保存：

- 强引用计数：有多少个 `shared_ptr` 拥有对象。
- 弱引用计数：有多少个 `weak_ptr` 观察对象。
- 删除器：对象该如何释放。
- 分配器等额外信息。

可以用简化模型理解：

```cpp
struct ControlBlock {
    long shared_count;
    long weak_count;
};

template <typename T>
struct SimpleSharedPtr {
    T* ptr;
    ControlBlock* control;
};
```

真实标准库实现更复杂，但大方向就是“对象指针 + 控制块”。

### 为什么推荐 make_shared

```cpp
auto p = std::make_shared<int>(42);
```

`make_shared` 通常可以把对象和控制块放在一次内存分配里，减少分配次数，提高局部性。

不推荐这样写：

```cpp
std::shared_ptr<int> p(new int(42));
```

这不是一定错误，但更容易在复杂表达式中引入异常安全问题，也不如 `make_shared` 简洁。

## 五、weak_ptr：观察 shared_ptr 管理的对象

`std::weak_ptr<T>` 不增加强引用计数，因此不会延长对象生命周期。

```cpp
#include <iostream>
#include <memory>

int main() {
    std::weak_ptr<int> observer;

    {
        auto owner = std::make_shared<int>(42);
        observer = owner;

        if (auto p = observer.lock()) {
            std::cout << *p << '\n';
        }
    }

    if (observer.expired()) {
        std::cout << "object destroyed\n";
    }
}
```

`weak_ptr::lock()` 会尝试获得一个 `shared_ptr`：

- 对象还活着：返回有效 `shared_ptr`。
- 对象已销毁：返回空 `shared_ptr`。

这种设计比直接保存裸指针安全，因为它能判断对象是否还存在。

## 六、循环引用问题

`shared_ptr` 的典型陷阱是循环引用。

```cpp
#include <memory>
#include <string>

struct Node {
    std::string name;
    std::shared_ptr<Node> next;
};

int main() {
    auto a = std::make_shared<Node>();
    auto b = std::make_shared<Node>();

    a->next = b;
    b->next = a;
}
```

函数结束时，局部变量 `a` 和 `b` 销毁，但两个节点内部还互相持有 `shared_ptr`：

- `a` 指向的对象被 `b->next` 持有。
- `b` 指向的对象被 `a->next` 持有。
- 强引用计数不会归零。
- 析构函数不会执行，形成内存泄漏。

### 用 weak_ptr 打破循环

通常把“非拥有关系”改成 `weak_ptr`。

```cpp
#include <memory>
#include <string>

struct Node {
    std::string name;
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;
};

int main() {
    auto a = std::make_shared<Node>();
    auto b = std::make_shared<Node>();

    a->next = b;  // a 拥有 b
    b->prev = a;  // b 只观察 a，不拥有 a
}
```

这样强引用不会形成闭环，对象可以正常释放。

## 七、常见误区

### 1. shared_ptr 不是越多越好

共享所有权越多，对象生命周期越不清晰。多数场景优先考虑 `unique_ptr`，只有确实需要共享所有权时再使用 `shared_ptr`。

### 2. 不要用两个 shared_ptr 托管同一个裸指针

```cpp
int* raw = new int(42);

std::shared_ptr<int> p1(raw);
std::shared_ptr<int> p2(raw);  // 错误：两个独立控制块
```

`p1` 和 `p2` 各自有独立控制块，最终会对同一个 `raw` 执行两次 `delete`。

正确做法是复制已有的 `shared_ptr`：

```cpp
auto p1 = std::make_shared<int>(42);
auto p2 = p1;
```

### 3. shared_ptr 的引用计数线程安全，不等于对象线程安全

多个线程复制、销毁不同的 `shared_ptr` 实例时，引用计数操作通常是安全的。但多个线程同时修改 `*p` 指向的对象，仍然需要锁或其他同步机制。

```cpp
// 引用计数安全，不代表 data.value++ 安全
struct Data {
    int value = 0;
};
```

## 八、面试回答模板

如果问题是“智能指针有哪些，区别是什么”，可以这样回答：

1. `unique_ptr` 表示独占所有权，不能拷贝，只能移动，适合明确单一拥有者的资源。
2. `shared_ptr` 表示共享所有权，通过控制块里的强引用计数决定对象释放时机。
3. `weak_ptr` 不拥有对象，不增加强引用计数，常用于观察对象和解决循环引用。
4. `shared_ptr` 控制块一般包含强引用计数、弱引用计数、删除器和分配器。
5. 循环引用会导致强引用计数无法归零，需要把非拥有关系改为 `weak_ptr`。
6. 工程实践中优先用 `make_unique`、`make_shared`，避免裸 `new/delete` 和重复托管裸指针。

## 九、总结

智能指针本质上是在表达所有权：

- 独占所有权：`unique_ptr`。
- 共享所有权：`shared_ptr`。
- 非拥有观察关系：`weak_ptr`。
- 引用计数能解决共享释放问题，但解决不了循环引用。
- 生命周期越清晰，内存错误越少，接口也越容易维护。
