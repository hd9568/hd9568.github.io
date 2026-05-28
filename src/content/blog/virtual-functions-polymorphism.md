---
title: '虚函数与多态：vptr、vtbl、纯虚函数与动态绑定'
description: '从对象内存布局和调用过程理解 C++ 虚函数、多态、虚函数表、纯虚函数、动态绑定与静态绑定。'
category: 'C++'
pubDate: '2026-06-01'
updatedDate: '2026-06-01'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [什么是多态](#二什么是多态)
3. [静态绑定与动态绑定](#三静态绑定与动态绑定)
4. [vptr 和 vtbl 的基本机制](#四vptr-和-vtbl-的基本机制)
5. [虚函数调用过程](#五虚函数调用过程)
6. [纯虚函数与抽象类](#六纯虚函数与抽象类)
7. [虚析构函数为什么重要](#七虚析构函数为什么重要)
8. [虚函数的成本与注意点](#八虚函数的成本与注意点)
9. [面试回答模板](#九面试回答模板)
10. [总结](#十总结)

## 一、核心结论

C++ 运行时多态的核心是虚函数。

- 普通函数调用通常在编译期确定目标函数，属于静态绑定。
- 虚函数通过基类指针或引用调用时，运行期根据对象真实类型决定调用哪个函数，属于动态绑定。
- 含虚函数的类对象通常会有一个隐藏的 `vptr`，指向该类型的虚函数表 `vtbl`。
- `vtbl` 中保存虚函数入口地址。
- 基类析构函数如果用于多态删除，应该声明为 `virtual`。

## 二、什么是多态

多态的直观含义是：同一个接口，根据对象真实类型表现出不同行为。

```cpp
#include <iostream>

struct Op {
    virtual void run() {
        std::cout << "base op\n";
    }
};

struct MatMulOp : Op {
    void run() override {
        std::cout << "matmul op\n";
    }
};

struct ConvOp : Op {
    void run() override {
        std::cout << "conv op\n";
    }
};

void execute(Op& op) {
    op.run();
}

int main() {
    MatMulOp matmul;
    ConvOp conv;

    execute(matmul);
    execute(conv);
}
```

`execute` 只知道参数是 `Op&`，但运行时会分别调用 `MatMulOp::run` 和 `ConvOp::run`。

## 三、静态绑定与动态绑定

### 静态绑定

非虚函数调用通常在编译期就确定。

```cpp
#include <iostream>

struct Base {
    void run() {
        std::cout << "Base\n";
    }
};

struct Derived : Base {
    void run() {
        std::cout << "Derived\n";
    }
};

int main() {
    Derived d;
    Base* p = &d;
    p->run();  // Base
}
```

这里 `run` 不是虚函数，`p` 的静态类型是 `Base*`，所以调用 `Base::run`。

### 动态绑定

加上 `virtual` 后，通过基类指针或引用调用会发生动态绑定。

```cpp
#include <iostream>

struct Base {
    virtual void run() {
        std::cout << "Base\n";
    }
};

struct Derived : Base {
    void run() override {
        std::cout << "Derived\n";
    }
};

int main() {
    Derived d;
    Base* p = &d;
    p->run();  // Derived
}
```

## 四、vptr 和 vtbl 的基本机制

主流编译器通常用 `vptr + vtbl` 实现虚函数。

可以用简化模型理解对象布局：

```text
Derived object
+----------------+
| vptr --------- | ----> Derived vtbl
+----------------+       +------------------+
| data members   |       | &Derived::run    |
+----------------+       | &Derived::other  |
                         +------------------+
```

- `vptr`：对象内部隐藏指针，指向虚函数表。
- `vtbl`：虚函数表，保存该类型的虚函数地址。
- 每个多态对象通常有自己的 `vptr`。
- 同一类型的多个对象通常共享同一张 `vtbl`。

标准并没有强制所有编译器必须用这种实现，但这是主流 ABI 的常见实现方式。

## 五、虚函数调用过程

以下代码：

```cpp
Base* p = new Derived();
p->run();
```

调用过程可以简化理解为：

1. 通过 `p` 找到对象。
2. 从对象中读取隐藏的 `vptr`。
3. 通过 `vptr` 找到 `Derived` 的虚函数表。
4. 在表中找到 `run` 对应的函数地址。
5. 调用 `Derived::run`。

因此，虚函数调用通常比普通函数调用多一次间接寻址，也可能影响内联优化。

## 六、纯虚函数与抽象类

纯虚函数写法：

```cpp
struct Op {
    virtual void run() = 0;
};
```

含有纯虚函数的类叫抽象类，不能直接实例化。

```cpp
struct MatMulOp : Op {
    void run() override {
        // 实现具体逻辑
    }
};
```

纯虚函数用于定义接口，派生类负责提供具体实现。

也可以给纯虚函数提供函数体，但类仍然是抽象类：

```cpp
struct Base {
    virtual void f() = 0;
};

void Base::f() {
    // 公共逻辑
}
```

这种写法较少见，但在需要派生类显式调用基类公共逻辑时可能出现。

## 七、虚析构函数为什么重要

多态基类几乎都应该有虚析构函数。

```cpp
#include <iostream>

struct Base {
    virtual ~Base() {
        std::cout << "Base dtor\n";
    }
};

struct Derived : Base {
    ~Derived() override {
        std::cout << "Derived dtor\n";
    }
};

int main() {
    Base* p = new Derived();
    delete p;
}
```

输出：

```text
Derived dtor
Base dtor
```

如果基类析构函数不是虚函数，通过基类指针删除派生类对象会导致未定义行为，派生类资源可能无法正确释放。

## 八、虚函数的成本与注意点

### 1. 对象体积可能变大

含虚函数的对象通常需要一个隐藏的 `vptr`。在 64 位系统中，`vptr` 通常占 8 字节。

```cpp
#include <iostream>

struct A {
    int x;
};

struct B {
    virtual void f() {}
    int x;
};

int main() {
    std::cout << sizeof(A) << '\n';
    std::cout << sizeof(B) << '\n';
}
```

`B` 往往比 `A` 更大，具体结果还受对齐影响。

### 2. 调用可能更难内联

虚函数是运行期决定目标函数的，编译器不一定能直接内联。不过现代编译器在能推断真实类型时，也可能做去虚拟化优化。

### 3. 构造和析构期间的虚函数行为特殊

在构造函数和析构函数中调用虚函数，不会按“最终派生类”动态分发，而是按当前正在构造或析构的类处理。

```cpp
#include <iostream>

struct Base {
    Base() {
        f();
    }

    virtual void f() {
        std::cout << "Base\n";
    }
};

struct Derived : Base {
    void f() override {
        std::cout << "Derived\n";
    }
};

int main() {
    Derived d;  // 构造 Base 部分时输出 Base
}
```

这是因为派生类部分还没有构造完成，调用派生类虚函数可能访问未初始化成员。

## 九、面试回答模板

如果问题是“虚函数表怎么工作”，可以这样回答：

1. 含虚函数的类对象通常有隐藏的 `vptr`，指向该类型的虚函数表 `vtbl`。
2. `vtbl` 中保存虚函数地址，不同派生类有不同表项。
3. 通过基类指针或引用调用虚函数时，程序运行期根据对象的 `vptr` 找到实际函数地址。
4. 这就是动态绑定，也是 C++ 运行时多态的基础。
5. 成本包括对象可能多一个指针、调用多一次间接寻址、可能影响内联。
6. 多态基类析构函数应该是虚函数，否则通过基类指针删除派生类对象会有问题。

## 十、总结

虚函数把“调用哪个函数”的决定从编译期推迟到运行期：

- 静态绑定看表达式静态类型。
- 动态绑定看对象真实类型。
- `vptr/vtbl` 是主流实现机制。
- 纯虚函数用于定义接口，抽象类不能直接实例化。
- 虚析构函数是多态基类的基本安全要求。
