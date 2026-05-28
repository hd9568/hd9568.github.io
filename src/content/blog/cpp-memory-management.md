---
title: '内存管理：new/delete、malloc/free、内存泄漏排查与内存池'
description: '从 C++ 对象生命周期出发，讲清 new/delete 与 malloc/free 的区别，理解内存泄漏排查和内存池设计思想。'
category: 'C++'
pubDate: '2026-05-30'
updatedDate: '2026-05-30'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [new/delete 做了什么](#二newdelete-做了什么)
3. [malloc/free 做了什么](#三mallocfree-做了什么)
4. [new/delete 与 malloc/free 的区别](#四newdelete-与-mallocfree-的区别)
5. [内存泄漏与常见错误](#五内存泄漏与常见错误)
6. [ASAN 和 Valgrind 排查思路](#六asan-和-valgrind-排查思路)
7. [内存池设计思想](#七内存池设计思想)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

C++ 内存管理需要同时关注“内存”和“对象生命周期”。

- `new` 会分配内存并调用构造函数。
- `delete` 会调用析构函数并释放内存。
- `malloc` 只分配原始内存，不会调用构造函数。
- `free` 只释放原始内存，不会调用析构函数。
- 内存泄漏指不再使用的内存没有被释放。
- 内存池通过批量申请和复用小块内存，降低频繁分配释放的开销。

## 二、new/delete 做了什么

`new` 不是简单地申请一块内存，它还会构造对象。

```cpp
#include <iostream>

struct Task {
    Task() {
        std::cout << "construct\n";
    }

    ~Task() {
        std::cout << "destruct\n";
    }
};

int main() {
    Task* p = new Task();
    delete p;
}
```

输出大致是：

```text
construct
destruct
```

可以把 `new Task()` 理解为两步：

1. 申请足够容纳 `Task` 的内存。
2. 在这块内存上调用 `Task` 的构造函数。

`delete p` 也可以理解为两步：

1. 调用 `Task` 的析构函数。
2. 释放这块内存。

## 三、malloc/free 做了什么

`malloc` 是 C 风格内存分配函数，只负责申请一段原始内存。

```cpp
#include <cstdlib>

int* p = static_cast<int*>(std::malloc(sizeof(int)));
*p = 10;
std::free(p);
```

对于普通 `int`，这种写法看起来还能工作。但对于有构造函数和析构函数的 C++ 对象，问题会变明显。

```cpp
#include <cstdlib>
#include <iostream>

struct Task {
    Task() {
        std::cout << "construct\n";
    }

    ~Task() {
        std::cout << "destruct\n";
    }
};

int main() {
    Task* p = static_cast<Task*>(std::malloc(sizeof(Task)));
    std::free(p);
}
```

这里不会输出 `construct` 和 `destruct`，因为 `malloc/free` 不知道 C++ 构造和析构语义。

## 四、new/delete 与 malloc/free 的区别

| 对比项 | new/delete | malloc/free |
| --- | --- | --- |
| 语言来源 | C++ 运算符 | C 标准库函数 |
| 构造函数 | `new` 会调用 | `malloc` 不会调用 |
| 析构函数 | `delete` 会调用 | `free` 不会调用 |
| 返回类型 | 返回具体类型指针 | 返回 `void*` |
| 失败行为 | 默认抛出 `std::bad_alloc` | 返回 `nullptr` |
| 是否可重载 | 可以重载 `operator new/delete` | 不属于 C++ 运算符重载 |

### 不要混用

```cpp
int* p = new int(10);
// std::free(p);  // 错误：new 分配的内存不能用 free 释放
```

```cpp
int* q = static_cast<int*>(std::malloc(sizeof(int)));
// delete q;      // 错误：malloc 分配的内存不能用 delete 释放
```

分配和释放必须配对：

- `new` 配 `delete`。
- `new[]` 配 `delete[]`。
- `malloc` 配 `free`。

## 五、内存泄漏与常见错误

### 1. 忘记释放

```cpp
void leak() {
    int* p = new int(42);
    // 忘记 delete
}
```

函数返回后，`p` 这个指针变量消失，但堆上的 `int` 没有释放。

### 2. 提前返回导致释放逻辑被跳过

```cpp
bool process(bool failed) {
    int* buffer = new int[1024];

    if (failed) {
        return false;  // 泄漏
    }

    delete[] buffer;
    return true;
}
```

这种代码适合用 RAII 改写：

```cpp
#include <memory>

bool process(bool failed) {
    auto buffer = std::make_unique<int[]>(1024);

    if (failed) {
        return false;
    }

    return true;
}
```

### 3. 重复释放

```cpp
int* p = new int(1);
delete p;
// delete p;  // double free，未定义行为
```

### 4. 释放后继续使用

```cpp
int* p = new int(1);
delete p;
// int x = *p;  // use-after-free，未定义行为
```

## 六、ASAN 和 Valgrind 排查思路

### AddressSanitizer

ASAN 是编译期插桩工具，运行时能发现越界访问、释放后使用、重复释放等问题。

```bash
g++ -std=c++17 -fsanitize=address -g main.cpp -o main
./main
```

示例代码：

```cpp
int main() {
    int* p = new int[2];
    p[2] = 10;  // 越界
    delete[] p;
}
```

ASAN 通常会指出错误类型、出错位置、分配位置和释放位置。

### Valgrind

Valgrind 不需要重新编译插桩，但运行速度通常更慢。

```bash
g++ -std=c++17 -g main.cpp -o main
valgrind --leak-check=full ./main
```

它常用于检查：

- 内存泄漏。
- 非法读写。
- 未初始化内存使用。

在 macOS 上，Valgrind 支持可能不如 Linux 完整；工程实践中 ASAN 更常用。

## 七、内存池设计思想

频繁调用系统分配器申请和释放小对象，可能带来明显开销和内存碎片。内存池的思路是：一次申请一大块内存，再切成固定大小的小块复用。

### 一个极简固定块内存池

下面示例只展示核心思想，不处理对齐、线程安全和异常安全等工程细节。

```cpp
#include <cstddef>
#include <vector>

class FixedBlockPool {
public:
    FixedBlockPool(std::size_t block_size, std::size_t block_count)
        : block_size_(block_size), storage_(block_size * block_count) {
        for (std::size_t i = 0; i < block_count; ++i) {
            free_list_.push_back(storage_.data() + i * block_size_);
        }
    }

    void* allocate() {
        if (free_list_.empty()) {
            return nullptr;
        }
        void* p = free_list_.back();
        free_list_.pop_back();
        return p;
    }

    void deallocate(void* p) {
        free_list_.push_back(static_cast<char*>(p));
    }

private:
    std::size_t block_size_;
    std::vector<char> storage_;
    std::vector<char*> free_list_;
};
```

使用方式：

```cpp
FixedBlockPool pool(sizeof(int), 1024);

void* mem = pool.allocate();
int* p = new (mem) int(42);  // placement new，在已有内存上构造对象

p->~int();                   // 对平凡类型这行实际不需要，仅用于表达析构语义
pool.deallocate(p);
```

更完整的内存池需要考虑：

- 内存对齐。
- 不同大小对象的分级管理。
- 线程安全。
- 空闲链表管理。
- 扩容策略。
- 对象构造和析构。

### 为什么内存池常用于系统和高性能场景

- 减少系统分配器调用次数。
- 提升小对象分配性能。
- 降低内存碎片。
- 让内存生命周期更可控。

在推理服务、网络框架、任务调度系统中，如果存在大量短生命周期小对象，内存池会比较常见。

## 八、面试回答模板

如果问题是“`new/delete` 和 `malloc/free` 的区别”，可以这样回答：

1. `new/delete` 是 C++ 运算符，`malloc/free` 是 C 库函数。
2. `new` 会分配内存并调用构造函数，`delete` 会调用析构函数并释放内存。
3. `malloc/free` 只处理原始内存，不处理 C++ 对象生命周期。
4. `new` 默认失败会抛异常，`malloc` 失败返回 `nullptr`。
5. 分配和释放必须配对，不能混用。

如果问题是“怎么排查内存泄漏”，可以回答：

1. 优先用 RAII 和智能指针减少手写释放。
2. 使用 ASAN 检查越界、use-after-free、double free。
3. 使用 Valgrind 或平台工具检查泄漏。
4. 对复杂服务，结合日志、压测、内存曲线和对象池统计定位。

## 九、总结

C++ 内存管理的核心是“内存分配”和“对象生命周期”不能混在一起看：

- `new/delete` 处理对象生命周期。
- `malloc/free` 只处理原始内存。
- 内存泄漏通常来自所有权不清晰或释放路径不完整。
- ASAN 和 Valgrind 是定位问题的常用工具。
- 内存池通过复用内存块提升性能，但需要更严格的生命周期管理。
