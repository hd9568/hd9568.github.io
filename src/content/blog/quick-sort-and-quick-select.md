---
title: 'Quick Sort 与 Quick Select：从分区不变量到完整 C++ 实现'
description: '用同一个三路分区核心讲清快速排序与快速选择，包括 Pivot、循环不变量、重复元素、随机化、复杂度、递归栈控制及完整可运行的 C++17 代码。'
category: '数据结构和算法'
pubDate: '2026-09-15T10:00:00+08:00'
updatedDate: '2026-09-15T10:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---
## 目录
1. [两个算法其实只差一步](#一两个算法其实只差一步)
2. [核心操作：三路分区](#二核心操作三路分区)
3. [Quick Sort 如何递归排序](#三quick-sort-如何递归排序)
4. [Quick Select 如何只找第 k 小](#四quick-select-如何只找第-k-小)
5. [完整 C++17 实现](#五完整-c17-实现)
6. [正确性与复杂度](#六正确性与复杂度)
7. [边界条件与使用方法](#七边界条件与使用方法)
## 一、两个算法其实只差一步
Quick Sort 和 Quick Select 都从同一个操作开始：选择一个基准值 `pivot`，再将数组分成三部分。
```text
[小于 pivot] [等于 pivot] [大于 pivot]
```
区别只在分区结束后：
```text
Quick Sort:
  左右两边都要继续处理，因为整个数组都要有序。
Quick Select:
  只处理第 k 小元素所在的一边，另一边可以直接丢弃。
```
例如：
```text
数组:  [7, 2, 5, 3, 5, 1, 5, 9]
pivot: 5
分区:  [2, 3, 1] [5, 5, 5] [9, 7]
下标:   0  1  2   3  4  5   6  7
```
分区后，三个区间内部不一定有序，但区间之间已有确定的大小关系：
```text
任意左区元素 < 任意中区元素 < 任意右区元素
```
因此：
```text
Quick Sort:
  只需递归排序左区和右区。
Quick Select:
  k < 3       -> 去左区
  3 <= k < 6  -> 答案就是 pivot
  k >= 6      -> 去右区
```
本文的 `k` 使用 **0-based 下标**：
```text
k = 0 表示最小值
k = 1 表示第二小
k = n - 1 表示最大值
```
## 二、核心操作：三路分区
### 为什么不只分成两路
经典二路分区只保证：
```text
[<= pivot] [> pivot]
```
当数组包含大量相同元素时，例如：
```text
[5, 5, 5, 5, 5, 5]
```
二路分区可能每次只排除一个元素，递归不断处理几乎相同的区间，性能退化到：
```text
T(n) = T(n - 1) + O(n) = O(n^2)
```
三路分区一次跳过所有等于 `pivot` 的元素。这对重复值很多的数据更稳健。
### 三个指针表示什么
对半开区间 `[left, right)` 分区：
```cpp
int lt = left;
int i  = left;
int gt = right;
```
循环过程中始终保持以下不变量：
```text
[left, lt)  < pivot
[lt, i)     = pivot
[i, gt)     尚未检查
[gt, right) > pivot
```
开始时：
```text
lt = i = left, gt = right
```
三个已分类区间都是空的，全部元素位于 `[i, gt)`。
### 每次只处理 `a[i]`
情况一，`a[i] < pivot`：
```cpp
std::swap(a[lt], a[i]);
++lt;
++i;
```
`a[i]` 被放进小于区。交换前的 `a[lt]` 属于等于区，因此交换后它位于新的 `[lt, i)` 内。

情况二，`a[i] > pivot`：
```cpp
--gt;
std::swap(a[i], a[gt]);
```
大元素被放到 `[gt, right)`。此时不能执行 `++i`，因为从右边交换到 `a[i]` 的元素还没有检查。

情况三，`a[i] == pivot`：
```cpp
++i;
```
等于区向右扩展一格。

循环在 `i == gt` 时结束，未检查区间变为空：
```text
[left, lt)  < pivot
[lt, gt)    = pivot
[gt, right) > pivot
```
完整分区函数：
```cpp
struct EqualRange {
  int begin;  // 等于 pivot 区间的起点
  int end;    // 等于 pivot 区间的终点，不包含 end
};
EqualRange partition_three_way(
    std::vector<int>& a,
    int left,
    int right,
    std::mt19937& rng) {
  std::uniform_int_distribution<int> dist(left, right - 1);
  const int pivot = a[dist(rng)];
  int lt = left;
  int i = left;
  int gt = right;
  while (i < gt) {
    if (a[i] < pivot) {
      std::swap(a[lt], a[i]);
      ++lt;
      ++i;
    } else if (a[i] > pivot) {
      --gt;
      std::swap(a[i], a[gt]);
    } else {
      ++i;
    }
  }
  return {lt, gt};
}
```
这里随机选择的是 `pivot` 的值，不要求先把 Pivot 交换到数组末尾。三路分区只比较元素与该值的大小。
## 三、Quick Sort 如何递归排序
分区返回：
```text
[left, equal.begin)        < pivot
[equal.begin, equal.end)   = pivot
[equal.end, right)         > pivot
```
中间区已经位于最终正确位置，不必继续排序。最直接的递归是：
```cpp
quick_sort(left, equal.begin);
quick_sort(equal.end, right);
```
### 为什么优先递归较小区间
如果每次 Pivot 都是最小值，普通递归的调用栈会达到 `O(n)`，大数组可能栈溢出。算法仍可能执行 `O(n^2)` 次比较，但可以把栈空间限制为 `O(log n)`：
```text
较小区间:
  使用递归。
较大区间:
  在当前函数中通过 while 继续处理。
```
因为递归进入的区间大小最多是当前区间的一半：
```text
n -> n/2 -> n/4 -> ... -> 1
```
递归深度不超过 `log2(n)`。

实现：
```cpp
void quick_sort_impl(
    std::vector<int>& a,
    int left,
    int right,
    std::mt19937& rng) {
  while (right - left > 1) {
    EqualRange equal =
        partition_three_way(a, left, right, rng);
    int left_size = equal.begin - left;
    int right_size = right - equal.end;
    if (left_size < right_size) {
      quick_sort_impl(a, left, equal.begin, rng);
      left = equal.end;
    } else {
      quick_sort_impl(a, equal.end, right, rng);
      right = equal.begin;
    }
  }
}
```
注意 `while` 更新的是当前待排序区间：
```text
递归左区后:
  left = equal.end
  当前函数继续处理右区
递归右区后:
  right = equal.begin
  当前函数继续处理左区
```
等于区不会再次进入任何待排序范围。
## 四、Quick Select 如何只找第 k 小
Quick Select 不需要整个数组有序。一次分区后，只需判断 `k` 落在哪个区间。
```cpp
if (k < equal.begin) {
  right = equal.begin;
} else if (k >= equal.end) {
  left = equal.end;
} else {
  return a[k];
}
```
它可以完全用循环实现，不需要递归。
### 手算一次
输入：
```text
a = [7, 2, 5, 3, 5, 1, 5, 9]
k = 6
```
假设第一次选择 `pivot=5`：
```text
分区后: [2, 3, 1] [5, 5, 5] [9, 7]
equal = [3, 6)
```
因为：
```text
k = 6 >= equal.end = 6
```
第 7 小元素一定在右区 `[6, 8)`。左区和中区共有 6 个元素，而且都不大于右区，因此不可能包含答案，可以永久丢弃。

右区若选择 `pivot=9`：
```text
[7] [9]
```
`k=6` 落在值为 7 的区间，答案为 7。

这里 `k` 始终是原数组中的绝对下标，不需要在进入右区后执行 `k -= right_begin`。因为函数只移动 `[left, right)` 边界，没有创建子数组。
## 五、完整 C++17 实现
下面的程序可以直接编译运行。Quick Sort 原地修改数组；Quick Select 也会改变元素顺序，因此示例传入副本。
```cpp
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
class QuickAlgorithms {
 public:
  explicit QuickAlgorithms(std::uint32_t seed = std::random_device{}())
      : rng_(seed) {}
  void sort(std::vector<int>& a) {
    quick_sort(a, 0, static_cast<int>(a.size()));
  }
  int select(std::vector<int>& a, int k) {
    if (k < 0 || k >= static_cast<int>(a.size())) {
      throw std::out_of_range("k is outside [0, a.size())");
    }
    int left = 0;
    int right = static_cast<int>(a.size());
    while (true) {
      EqualRange equal = partition(a, left, right);
      if (k < equal.begin) {
        right = equal.begin;
      } else if (k >= equal.end) {
        left = equal.end;
      } else {
        return a[k];
      }
    }
  }
 private:
  struct EqualRange {
    int begin;
    int end;
  };
  std::mt19937 rng_;
  EqualRange partition(std::vector<int>& a, int left, int right) {
    std::uniform_int_distribution<int> dist(left, right - 1);
    const int pivot = a[dist(rng_)];
    int lt = left;
    int i = left;
    int gt = right;
    while (i < gt) {
      if (a[i] < pivot) {
        std::swap(a[lt], a[i]);
        ++lt;
        ++i;
      } else if (a[i] > pivot) {
        --gt;
        std::swap(a[i], a[gt]);
      } else {
        ++i;
      }
    }
    return {lt, gt};
  }
  void quick_sort(
      std::vector<int>& a,
      int left,
      int right) {
    while (right - left > 1) {
      EqualRange equal = partition(a, left, right);
      const int left_size = equal.begin - left;
      const int right_size = right - equal.end;
      if (left_size < right_size) {
        quick_sort(a, left, equal.begin);
        left = equal.end;
      } else {
        quick_sort(a, equal.end, right);
        right = equal.begin;
      }
    }
  }
};
int main() {
  std::vector<int> input{7, 2, 5, 3, 5, 1, 5, 9};
  QuickAlgorithms algorithms(42);  // 固定 Seed，方便复现实验
  std::vector<int> sorted = input;
  algorithms.sort(sorted);
  std::cout << "sorted:";
  for (int value : sorted) {
    std::cout << ' ' << value;
  }
  std::cout << '\n';
  std::vector<int> selected = input;
  int k = 6;
  std::cout << "index " << k
            << " after sorting = "
            << algorithms.select(selected, k)
            << '\n';
}
```
编译：
```bash
g++ -std=c++17 -O2 -Wall -Wextra quick_algorithms.cpp -o quick_algorithms
./quick_algorithms
```
输出：
```text
sorted: 1 2 3 5 5 5 7 9
index 6 after sorting = 7
```
若只需要找第 `x` 小，传入：
```cpp
int answer = algorithms.select(a, x - 1);
```
因为自然语言“第 1 小”对应 0-based 的 `k=0`。
## 六、正确性与复杂度
### 分区为什么正确
每轮循环前保持：
```text
[left, lt)  < pivot
[lt, i)     = pivot
[i, gt)     unknown
[gt, right) > pivot
```
三种分支都只把一个未知元素移动到正确区间，并让 `[i, gt)` 缩短一格。循环必然结束；结束时未知区间为空，因此三个分区成立。
### Quick Sort 为什么正确
分区后：
```text
左区所有元素 < 中区
中区所有元素 < 右区
```
递归使左区和右区各自有序，中间区本身全相等，所以拼接后的整个区间有序。长度为 0 或 1 的区间天然有序，是递归终点。
### Quick Select 为什么正确
设等于区为 `[p, q)`：
```text
k < p:
  第 k 小一定在左区。
k >= q:
  左区和中区已有 q 个更小或相等的元素，
  第 k 小一定在右区。
p <= k < q:
  排序后该位置必然等于 pivot。
```
每轮至少排除等于区，搜索区间严格缩小，所以算法一定终止。
### 时间复杂度
一次分区扫描当前区间一次，成本是 `O(n)`。

随机 Pivot 下，Quick Sort 的期望递推近似为：
```text
T(n) = 2T(n/2) + O(n)
     = O(n log n)
```
最坏情况下每次只排除一个元素：
```text
T(n) = T(n - 1) + O(n)
     = O(n^2)
```
Quick Select 只进入一边，期望成本为：
```text
T(n) = T(n/2) + O(n)
     = n + n/2 + n/4 + ...
     = O(n)
```
最坏情况仍为 `O(n^2)`。

空间复杂度：
```text
Quick Sort:
  原地分区 O(1)
  递归栈 O(log n)
Quick Select:
  原地分区 O(1)
  循环实现无递归栈
```
这里 Quick Sort 的 `O(log n)` 栈空间由“只递归较小区间”保证，与 Pivot 是否均匀无关。
## 七、边界条件与使用方法
### 空数组与单元素
```cpp
std::vector<int> a;
algorithms.sort(a);  // 合法，什么也不做
```
`sort()` 只有在区间长度大于 1 时才调用分区，因此不会为 `uniform_int_distribution` 构造非法范围。

Quick Select 对空数组没有合法的 `k`，会抛出：
```cpp
std::out_of_range
```
### 重复元素
```text
[4, 4, 4, 4]
```
一次分区得到：
```text
equal = [0, 4)
```
Quick Sort 立即结束；任意合法 `k` 的 Quick Select 也立即返回 4。
### 已排序或逆序输入
固定选择首元素或尾元素会稳定地产生极不均匀分区。本文从当前区间均匀随机选择 Pivot，使任何输入排列都具有相同的期望复杂度。

随机化不能消除理论上的 `O(n^2)` 最坏情况，但能防止“已排序数组必然最坏”。
### 是否稳定
Quick Sort 不是稳定排序。若两个元素的 Key 相等，交换可能改变它们的相对顺序：
```text
(score=5, id=A)
(score=5, id=B)
```
排序后 `B` 可能出现在 `A` 前。如果业务要求稳定性，应使用 `std::stable_sort` 或 Merge Sort。
### 何时使用哪个接口
```text
需要整个数组有序:
  Quick Sort，实际工程通常直接用 std::sort。
只需要中位数、Top-K 边界或第 k 小:
  Quick Select，避免排序所有元素。
只需要 STL:
  std::nth_element 实现与 Quick Select 相同的接口语义。
```
例如找中位数：
```cpp
int middle = static_cast<int>(a.size() / 2);
int median = algorithms.select(a, middle);
```
若数组长度为偶数，“中位数”可能定义为中间两个数的平均值。这时需分别选择：
```text
k1 = n / 2 - 1
k2 = n / 2
```
两次选择会修改数组但不影响答案；也可以先对 `k2` 执行一次选择，再在左侧找最大值。

最终只需记住一个核心：
> 分区负责建立大小关系；Quick Sort 处理两边，Quick Select 只处理包含 `k` 的一边。
