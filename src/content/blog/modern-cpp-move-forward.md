---
title: '现代 C++：右值引用、移动语义与完美转发'
description: '用资源转移的视角理解右值引用、std::move 和 std::forward，讲清现代 C++ 中减少拷贝和保持值类别的关键机制。'
category: 'C++'
pubDate: '2026-05-29'
updatedDate: '2026-05-29'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [左值和右值](#二左值和右值)
3. [右值引用解决什么问题](#三右值引用解决什么问题)
4. [移动语义：把资源转移出去](#四移动语义把资源转移出去)
5. [std::move 到底做了什么](#五stdmove-到底做了什么)
6. [完美转发与 std::forward](#六完美转发与-stdforward)
7. [常见误区](#七常见误区)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

现代 C++ 的移动语义主要解决“昂贵拷贝”的问题。

- 左值有稳定身份，通常可以取地址。
- 右值多表示临时对象或即将不再使用的对象。
- 右值引用 `T&&` 可以绑定右值，是移动语义的基础。
- 移动构造不是深拷贝资源，而是把资源所有权转移给新对象。
- `std::move` 不移动任何东西，只把表达式转换成右值引用。
- `std::forward` 用于模板中保持参数原来的值类别，是完美转发的核心。

## 二、左值和右值

一个简化理解：

- 左值：有名字、有稳定位置的表达式。
- 右值：临时值，通常很快就不再使用。

```cpp
int a = 10;     // a 是左值
int b = a + 1;  // a + 1 是右值
```

函数返回的临时对象也是右值：

```cpp
#include <string>

std::string make_name() {
    return "kernel";
}

std::string s = make_name();
```

`make_name()` 返回的是一个临时 `std::string`。现代 C++ 会尽量通过返回值优化或移动语义避免不必要拷贝。

## 三、右值引用解决什么问题

右值引用写作 `T&&`，可以绑定右值。

```cpp
int&& x = 10;
```

它的意义在于：如果一个对象是临时的，或者明确不再需要原来的值，就可以“偷走”它内部的资源，而不是重新分配和拷贝。

例如 `std::vector` 内部有一块堆内存。拷贝一个大 vector 可能需要复制大量元素，而移动一个 vector 通常只需要转移内部指针、大小和容量。

## 四、移动语义：把资源转移出去

下面用一个极简字符串类说明拷贝和移动的区别。

```cpp
#include <cstring>
#include <iostream>

class Buffer {
public:
    explicit Buffer(const char* s) {
        size_ = std::strlen(s);
        data_ = new char[size_ + 1];
        std::strcpy(data_, s);
    }

    ~Buffer() {
        delete[] data_;
    }

    Buffer(const Buffer& other) {
        size_ = other.size_;
        data_ = new char[size_ + 1];
        std::strcpy(data_, other.data_);
        std::cout << "copy\n";
    }

    Buffer(Buffer&& other) noexcept {
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
        std::cout << "move\n";
    }

private:
    char* data_ = nullptr;
    std::size_t size_ = 0;
};
```

拷贝构造会重新分配内存并复制内容。移动构造只转移指针，并把源对象置为安全的空状态。

```cpp
Buffer a("hello");
Buffer b(std::move(a));  // 调用移动构造
```

移动后，`a` 仍然是一个有效对象，但它的具体值通常只保证“可析构、可赋值”，不应继续依赖原内容。

## 五、std::move 到底做了什么

`std::move` 的名字容易误导。它不移动对象，也不释放资源，只做类型转换。

可以把它理解成：

```cpp
static_cast<T&&>(x)
```

示例：

```cpp
#include <iostream>
#include <string>

void use(const std::string&) {
    std::cout << "copy-like path\n";
}

void use(std::string&&) {
    std::cout << "move-like path\n";
}

int main() {
    std::string s = "hello";

    use(s);            // s 是左值
    use(std::move(s)); // 转成右值，允许移动
}
```

输出大致是：

```text
copy-like path
move-like path
```

`std::move(s)` 之后，是否真的发生移动，取决于后续调用的函数或构造函数是否实现了移动逻辑。

## 六、完美转发与 std::forward

模板函数中常见需求是：接收到什么样的参数，就原样转交给另一个函数。

先看一个问题：

```cpp
#include <iostream>

void sink(int&) {
    std::cout << "lvalue\n";
}

void sink(int&&) {
    std::cout << "rvalue\n";
}

template <typename T>
void wrapper(T&& x) {
    sink(x);  // x 有名字，因此 x 是左值
}

int main() {
    int a = 1;
    wrapper(a);
    wrapper(2);
}
```

即使 `wrapper(2)` 传入的是右值，函数体里的 `x` 因为有名字，也是左值，所以会调用 `sink(int&)`。

使用 `std::forward` 可以保持原始值类别：

```cpp
#include <iostream>
#include <utility>

void sink(int&) {
    std::cout << "lvalue\n";
}

void sink(int&&) {
    std::cout << "rvalue\n";
}

template <typename T>
void wrapper(T&& x) {
    sink(std::forward<T>(x));
}
```

这就是完美转发：模板函数把参数传给下游函数时，尽量保持它原本是左值还是右值。

### 转发引用

`T&&` 在模板参数推导场景下叫转发引用：

```cpp
template <typename T>
void f(T&& x);
```

它既能接收左值，也能接收右值。注意，普通的 `int&&` 不是转发引用：

```cpp
void g(int&& x);  // 只能绑定右值
```

## 七、常见误区

### 1. std::move 后对象不是销毁了

```cpp
std::string s = "hello";
std::string t = std::move(s);

s = "world";  // 可以重新赋值
```

移动后的对象仍然有效，但值处于“有效但未指定”状态。

### 2. const 对象通常无法真正移动

```cpp
const std::string s = "hello";
std::string t = std::move(s);  // 通常调用拷贝构造
```

移动构造一般需要修改源对象，所以参数通常是 `T&&`，而不是 `const T&&`。`std::move(const T)` 得到的是 `const T&&`，很难匹配正常移动构造。

### 3. 移动构造建议标记 noexcept

标准容器在扩容时，如果元素移动构造可能抛异常，可能会选择拷贝以保证异常安全。

```cpp
class Data {
public:
    Data(Data&& other) noexcept {
        // move fields
    }
};
```

对于资源管理类，移动构造和移动赋值通常应该考虑 `noexcept`。

## 八、面试回答模板

如果问题是“右值引用和移动语义有什么用”，可以这样回答：

1. 右值引用 `T&&` 可以绑定右值，是移动语义和完美转发的基础。
2. 移动语义用于把临时对象或不再使用对象的资源转移出去，减少深拷贝成本。
3. `std::move` 不移动对象，只把表达式转换成右值引用，是否移动取决于后续函数。
4. 移动后对象仍然有效，但值通常未指定。
5. `std::forward` 用于模板转发，保持参数原始值类别。
6. 转发引用只出现在类型推导场景中的 `T&&`，普通 `int&&` 不是转发引用。

## 九、总结

现代 C++ 的这组机制服务于两个目标：

- 通过移动语义减少不必要的资源拷贝。
- 通过完美转发写出通用且不丢失参数属性的模板代码。

关键点是：`std::move` 是强制转换，`std::forward` 是条件转换。前者表达“可以把资源转走”，后者表达“保持传入时的左值/右值属性”。
