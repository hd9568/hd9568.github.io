---
title: 'STL 容器底层：vector、unordered_map 与 map'
description: '从扩容、哈希冲突、负载因子和红黑树角度理解 std::vector、std::unordered_map、std::map 的底层机制和面试重点。'
pubDate: '2026-06-03'
updatedDate: '2026-06-03'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [核心结论](#一核心结论)
2. [vector：连续内存与扩容机制](#二vector连续内存与扩容机制)
3. [vector 的迭代器和引用失效](#三vector-的迭代器和引用失效)
4. [unordered_map：哈希表](#四unordered_map哈希表)
5. [哈希冲突与负载因子](#五哈希冲突与负载因子)
6. [map：红黑树](#六map红黑树)
7. [三个容器如何选择](#七三个容器如何选择)
8. [面试回答模板](#八面试回答模板)
9. [总结](#九总结)

## 一、核心结论

STL 容器底层结构决定了它们的复杂度、内存特征和失效规则。

- `std::vector` 使用连续内存，随机访问快，尾部追加均摊 `O(1)`。
- `vector` 扩容会重新分配内存并搬迁元素，旧指针、引用、迭代器可能失效。
- `std::unordered_map` 基于哈希表，平均查找 `O(1)`，但最坏可能退化。
- 哈希冲突通常通过链地址法或类似结构解决，负载因子影响冲突概率和扩容时机。
- `std::map` 通常基于红黑树，查找、插入、删除都是 `O(log n)`，并保持 key 有序。

## 二、vector：连续内存与扩容机制

`std::vector` 的核心特点是连续内存。

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3};

    std::cout << &v[0] << '\n';
    std::cout << &v[1] << '\n';
    std::cout << &v[2] << '\n';
}
```

相邻元素地址通常连续递增，这让 `vector` 的随机访问非常快，也有利于 CPU 缓存。

### size 和 capacity

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v;

    for (int i = 0; i < 10; ++i) {
        v.push_back(i);
        std::cout << "size=" << v.size()
                  << ", capacity=" << v.capacity() << '\n';
    }
}
```

- `size`：当前元素个数。
- `capacity`：当前已分配空间最多能放多少元素。

当 `size == capacity` 且继续插入时，`vector` 通常会扩容。

### 扩容过程

扩容可以简化理解为：

1. 申请一块更大的连续内存。
2. 把旧元素移动或拷贝到新内存。
3. 销毁旧元素并释放旧内存。
4. 更新内部指针、大小和容量。

常见扩容倍数可能是 1.5 倍或 2 倍，标准不强制规定具体倍数。

## 三、vector 的迭代器和引用失效

扩容后，旧内存被释放，因此指向旧元素的指针、引用、迭代器会失效。

```cpp
#include <vector>

int main() {
    std::vector<int> v;
    v.push_back(1);

    int* p = &v[0];

    for (int i = 0; i < 100; ++i) {
        v.push_back(i);
    }

    // *p 可能已经悬垂，不能再使用
}
```

如果提前知道元素数量，可以用 `reserve` 减少扩容次数。

```cpp
std::vector<int> v;
v.reserve(1000);

for (int i = 0; i < 1000; ++i) {
    v.push_back(i);
}
```

`reserve` 只改变容量，不改变元素数量。

```cpp
std::vector<int> v;
v.reserve(10);
// v.size() 仍然是 0
```

## 四、unordered_map：哈希表

`std::unordered_map` 是无序关联容器，底层通常是哈希表。

```cpp
#include <iostream>
#include <string>
#include <unordered_map>

int main() {
    std::unordered_map<std::string, int> score;

    score["matmul"] = 10;
    score["conv"] = 20;

    std::cout << score["matmul"] << '\n';
}
```

插入或查找时，大致流程是：

1. 对 key 计算哈希值。
2. 根据哈希值定位桶 bucket。
3. 在桶中查找对应 key。

简化公式：

```text
bucket_index = hash(key) % bucket_count
```

真实实现可能会用更复杂方式，但核心思想类似。

## 五、哈希冲突与负载因子

不同 key 可能映射到同一个 bucket，这叫哈希冲突。

常见解决方式是链地址法：每个 bucket 里挂一个链表或节点集合。

```text
bucket[0] -> (key1, value1) -> (key9, value9)
bucket[1] -> empty
bucket[2] -> (key3, value3)
```

查找时先定位 bucket，再在 bucket 内比较 key。

### 负载因子

负载因子表示元素数量和 bucket 数量的比例。

```cpp
#include <iostream>
#include <unordered_map>

int main() {
    std::unordered_map<int, int> m;

    for (int i = 0; i < 100; ++i) {
        m[i] = i;
    }

    std::cout << m.load_factor() << '\n';
    std::cout << m.bucket_count() << '\n';
}
```

负载因子越高，冲突概率通常越大。`unordered_map` 会在负载因子超过阈值时 rehash，重新分配 bucket 并搬迁节点。

可以主动设置：

```cpp
std::unordered_map<int, int> m;
m.reserve(1000);
m.max_load_factor(0.7f);
```

`reserve` 能减少 rehash 次数，对大量插入有帮助。

### 平均 O(1) 不是永远 O(1)

哈希表平均查找是 `O(1)`，但如果大量 key 冲突到同一桶，查找会退化。实际性能取决于：

- 哈希函数质量。
- bucket 数量。
- 负载因子。
- key 比较成本。

## 六、map：红黑树

`std::map` 是有序关联容器，通常基于红黑树实现。

```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> score;

    score["conv"] = 20;
    score["matmul"] = 10;
    score["attention"] = 30;

    for (const auto& [key, value] : score) {
        std::cout << key << ':' << value << '\n';
    }
}
```

输出会按 key 有序排列。

红黑树是一种近似平衡二叉搜索树。它通过颜色和旋转保持树高在 `O(log n)` 级别。

因此 `map` 的：

- 查找：`O(log n)`。
- 插入：`O(log n)`。
- 删除：`O(log n)`。
- 有序遍历：天然支持。

### map 的迭代器稳定性

`map` 的节点通常单独分配，插入新元素一般不会让已有迭代器失效。删除某个元素只会让指向该元素的迭代器失效。

这和 `vector` 扩容后大面积失效不同。

## 七、三个容器如何选择

| 场景 | 推荐容器 | 原因 |
| --- | --- | --- |
| 需要连续存储、随机访问、遍历性能好 | `vector` | 缓存友好，随机访问 `O(1)` |
| 需要按 key 快速查找，不关心顺序 | `unordered_map` | 平均查找 `O(1)` |
| 需要 key 有序、范围查询、稳定顺序遍历 | `map` | 红黑树有序，操作 `O(log n)` |

### 一个简单例子

```cpp
#include <map>
#include <unordered_map>
#include <vector>

std::vector<int> values;                 // 存一批连续数据
std::unordered_map<int, int> id_to_pos;   // 根据 id 快速查位置
std::map<int, int> ordered_counter;       // 需要按 key 有序遍历
```

容器选择不只看复杂度，也要看内存布局、缓存局部性、是否有序、迭代器是否稳定。

## 八、面试回答模板

如果问题是“vector 扩容机制”，可以这样回答：

1. `vector` 底层是连续内存，维护 size 和 capacity。
2. 当插入元素超过 capacity 时，会申请更大内存。
3. 旧元素会移动或拷贝到新内存，然后释放旧内存。
4. 扩容后旧指针、引用、迭代器通常失效。
5. `push_back` 均摊 `O(1)`，因为不是每次插入都扩容。
6. 可以用 `reserve` 预留容量减少扩容次数。

如果问题是“unordered_map 和 map 区别”，可以这样回答：

1. `unordered_map` 基于哈希表，平均查找 `O(1)`，不保证顺序。
2. `map` 基于红黑树，查找 `O(log n)`，按 key 有序。
3. `unordered_map` 受哈希函数、冲突和负载因子影响，最坏可能退化。
4. `map` 支持有序遍历和范围查询，迭代器稳定性通常更好。
5. 需要快速点查选 `unordered_map`，需要有序和范围操作选 `map`。

## 九、总结

STL 容器不是只有接口差异，底层结构也很重要：

- `vector` 胜在连续内存和缓存友好，但扩容会导致失效。
- `unordered_map` 胜在平均快速查找，但要关注哈希冲突和 rehash。
- `map` 胜在有序性和稳定复杂度，但单次查找通常比哈希表慢。
- 工程中容器选择要结合访问模式、数据规模、顺序需求和内存局部性。
