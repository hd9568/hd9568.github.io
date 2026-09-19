---
title: 'Quick Sort 与 Quick Select：从 Partition 到完整 C++ 实现'
description: '用 LeetCode 风格的简单 C++ 代码讲清 Partition、Quick Sort 和 Quick Select，并逐步展示指针移动、递归区间与第 k 小/第 k 大的计算过程。'
category: '数据结构和算法'
pubDate: '2026-09-15T10:00:00+08:00'
updatedDate: '2026-09-19T10:00:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---
## 目录
1. [Quick Sort 和 Quick Select 的关系](#一quick-sort-和-quick-select-的关系)
2. [先彻底理解 Partition](#二先彻底理解-partition)
3. [Quick Sort 的完整流程](#三quick-sort-的完整流程)
4. [Quick Select 的完整流程](#四quick-select-的完整流程)
5. [完整 C++ 代码](#五完整-c-代码)
6. [复杂度与常见问题](#六复杂度与常见问题)
## 一、Quick Sort 和 Quick Select 的关系
两个算法都依赖同一个核心操作：`partition`，即分区。

分区从数组中选择一个基准元素 `pivot`，重新排列当前区间，使得：
```text
[小于等于 pivot 的元素] pivot [大于 pivot 的元素]
```
假设分区后 `pivot` 位于下标 `p`：
```text
left ... p - 1 | p | p + 1 ... right
    <= pivot   |pivot|    > pivot
```
此时可以确定：
> `pivot` 已经位于最终排序结果中的正确下标 `p`。

注意，左右区间内部还没有完全有序。例如：
```text
[4, 2, 3] 6 [7]
```
左边的 `4、2、3` 都小于 6，但它们之间仍然无序。

两个算法的区别只在分区之后：
```text
Quick Sort:
  为了让整个数组有序，继续处理左区间和右区间。
Quick Select:
  为了寻找某个下标，只处理目标下标所在的一边。
```
可以先记成：
```text
Quick Sort   = partition + 递归两边
Quick Select = partition + 只递归一边
```
本文所有区间都使用闭区间：
```text
[left, right]
```
因此区间包含 `nums[left]` 和 `nums[right]`。
## 二、先彻底理解 Partition
使用 LeetCode 中常见的写法：选择当前区间最后一个元素作为 `pivot`。
```cpp
int partitionArray(vector<int>& nums, int left, int right) {
    int pivot = nums[right];
    int i = left;
    for (int j = left; j < right; ++j) {
        if (nums[j] <= pivot) {
            swap(nums[i], nums[j]);
            ++i;
        }
    }
    swap(nums[i], nums[right]);
    return i;
}
```
代码只有两个指针：
```text
j:
  从左到右检查每个元素。
i:
  下一个“小于等于 pivot 的元素”应该放置的位置。
```
### 循环过程中数组被分成什么
每次开始检查 `nums[j]` 前，都有：
```text
[left, i - 1]  <= pivot
[i, j - 1]     > pivot
[j, right - 1] 还没有检查
[right]         pivot
```
初始时：
```text
i = left
j = left
```
前两个区间都是空的，所有元素都还没有检查。

每次只判断 `nums[j]`。

如果：
```text
nums[j] <= pivot
```
就把它交换到位置 `i`，然后执行：
```cpp
++i;
```
如果：
```text
nums[j] > pivot
```
什么都不做，只让 `j` 继续向右移动。这个大元素自然会留在 `[i, j]` 区间中。

扫描结束后：
```text
[left, i - 1]  <= pivot
[i, right - 1] > pivot
[right]         pivot
```
最后交换：
```cpp
swap(nums[i], nums[right]);
```
就得到：
```text
[left, i - 1] <= pivot
[i]            pivot
[i + 1, right] > pivot
```
所以 `i` 就是 Pivot 的最终下标。
### 完整手算一次
输入：
```text
nums = [4, 2, 7, 3, 6]
left = 0
right = 4
pivot = nums[4] = 6
i = 0
```
检查 `j=0`：
```text
nums[0] = 4 <= 6
swap(nums[0], nums[0])
i = 1
数组: [4, 2, 7, 3, 6]
```
检查 `j=1`：
```text
nums[1] = 2 <= 6
swap(nums[1], nums[1])
i = 2
数组: [4, 2, 7, 3, 6]
```
检查 `j=2`：
```text
nums[2] = 7 > 6
不交换
i 仍然为 2
数组: [4, 2, 7, 3, 6]
```
检查 `j=3`：
```text
nums[3] = 3 <= 6
swap(nums[2], nums[3])
i = 3
数组: [4, 2, 3, 7, 6]
```
扫描结束，将 Pivot 放到 `i=3`：
```text
swap(nums[3], nums[4])
数组: [4, 2, 3, 6, 7]
```
返回：
```text
p = 3
```
现在可以确定：
```text
nums[0..2] <= 6
nums[3]     = 6
nums[4]     > 6
```
`6` 已经处在整个数组排序后的正确位置，但 `[4,2,3]` 仍需继续处理。
## 三、Quick Sort 的完整流程
Quick Sort 分区后递归处理两边：
```cpp
void quickSort(vector<int>& nums, int left, int right) {
    if (left >= right) {
        return;
    }
    int p = partitionArray(nums, left, right);
    quickSort(nums, left, p - 1);
    quickSort(nums, p + 1, right);
}
```
递归终止条件是：
```cpp
left >= right
```
此时区间中最多只有一个元素，不需要排序。
### 用同一个数组走完整过程
初始数组：
```text
[4, 2, 7, 3, 6]
```
第一次分区：
```text
[4, 2, 3] 6 [7]
             p=3
```
然后递归：
```text
左区间 [0, 2]: [4, 2, 3]
右区间 [4, 4]: [7]
```
右区间只有一个元素，直接结束。

处理左区间 `[4,2,3]`，选择最后的 3 作为 Pivot：
```text
检查 4:
  4 > 3，不交换
检查 2:
  2 <= 3，与位置 0 的 4 交换
数组变为 [2, 4, 3, 6, 7]
```
最后将 Pivot 3 放到下标 1：
```text
[2] 3 [4] 6 [7]
     p=1
```
接下来两个子区间：
```text
[0, 0] 只有元素 2
[2, 2] 只有元素 4
```
都直接结束。最终结果：
```text
[2, 3, 4, 6, 7]
```
递归结构可以画成：
```text
[4, 2, 7, 3, 6]
        |
      pivot 6
      /     \
[4, 2, 3]  [7]
     |
   pivot 3
   /     \
 [2]     [4]
```
Quick Sort 必须处理左右两边，因为目标是让每个元素都有序。
## 四、Quick Select 的完整流程
Quick Select 用于寻找：
```text
第 k 小
第 k 大
中位数
Top-K 的边界值
```
它不需要把整个数组排好序。
### 先把“第 k 小”转成目标下标
数组下标从 0 开始：
```text
第 1 小 -> target = 0
第 2 小 -> target = 1
第 k 小 -> target = k - 1
```
如果数组长度为 `n`：
```text
第 1 大 -> target = n - 1
第 2 大 -> target = n - 2
第 k 大 -> target = n - k
```
例如：
```text
排序后: [2, 3, 4, 6, 7]
下标:     0  1  2  3  4
第 3 小: target = 3 - 1 = 2，答案是 4
第 2 大: target = 5 - 2 = 3，答案是 6
```
### 分区后只进入一边
```cpp
int quickSelect(
    vector<int>& nums,
    int left,
    int right,
    int target) {
    int p = partitionArray(nums, left, right);
    if (p == target) {
        return nums[p];
    }
    if (target < p) {
        return quickSelect(nums, left, p - 1, target);
    }
    return quickSelect(nums, p + 1, right, target);
}
```
判断逻辑是：
```text
p == target:
  Pivot 正好位于目标下标，直接返回。
target < p:
  目标位于 Pivot 左侧，只搜索左边。
target > p:
  目标位于 Pivot 右侧，只搜索右边。
```
另一边可以直接丢弃，因为分区已经确定 Pivot 的最终排名。
### 手算第 3 小
输入：
```text
nums = [4, 2, 7, 3, 6]
k = 3
target = k - 1 = 2
```
第一次分区选择 6：
```text
[4, 2, 3] 6 [7]
             p=3
```
因为：
```text
target=2 < p=3
```
只需搜索左区间 `[0,2]`，右边的 6 和 7 不可能是第 3 小。

左区间当前为：
```text
[4, 2, 3]
```
选择 3 分区：
```text
[2] 3 [4]
     p=1
```
因为：
```text
target=2 > p=1
```
只需搜索右区间 `[2,2]`。这个区间只有一个元素 4，因此答案是：
```text
第 3 小 = 4
```
Quick Select 没有继续排序下标 0 和 1 的元素，也没有排序下标 3 和 4 的元素。它只保证目标下标上的值正确。
## 五、完整 C++ 代码
下面只使用 `vector`、函数、循环、递归和 `swap`。
```cpp
#include <iostream>
#include <utility>
#include <vector>
using namespace std;
int partitionArray(vector<int>& nums, int left, int right) {
    int pivot = nums[right];
    int i = left;
    for (int j = left; j < right; ++j) {
        if (nums[j] <= pivot) {
            swap(nums[i], nums[j]);
            ++i;
        }
    }
    swap(nums[i], nums[right]);
    return i;
}
void quickSort(vector<int>& nums, int left, int right) {
    if (left >= right) {
        return;
    }
    int p = partitionArray(nums, left, right);
    quickSort(nums, left, p - 1);
    quickSort(nums, p + 1, right);
}
int quickSelect(
    vector<int>& nums,
    int left,
    int right,
    int target) {
    if (left == right) {
        return nums[left];
    }
    int p = partitionArray(nums, left, right);
    if (p == target) {
        return nums[p];
    }
    if (target < p) {
        return quickSelect(nums, left, p - 1, target);
    }
    return quickSelect(nums, p + 1, right, target);
}
int findKthSmallest(vector<int>& nums, int k) {
    int target = k - 1;
    return quickSelect(nums, 0, nums.size() - 1, target);
}
int findKthLargest(vector<int>& nums, int k) {
    int target = nums.size() - k;
    return quickSelect(nums, 0, nums.size() - 1, target);
}
int main() {
    vector<int> input{4, 2, 7, 3, 6};
    vector<int> sorted = input;
    quickSort(sorted, 0, sorted.size() - 1);
    cout << "sorted:";
    for (int x : sorted) {
        cout << ' ' << x;
    }
    cout << '\n';
    vector<int> a = input;
    cout << "3rd smallest: " << findKthSmallest(a, 3) << '\n';
    vector<int> b = input;
    cout << "2nd largest: " << findKthLargest(b, 2) << '\n';
}
```
编译运行：
```bash
g++ -std=c++17 -O2 quick_algorithms.cpp -o quick_algorithms
./quick_algorithms
```
输出：
```text
sorted: 2 3 4 6 7
3rd smallest: 4
2nd largest: 6
```
Quick Sort 和 Quick Select 都会通过交换改变原数组。示例创建 `sorted`、`a`、`b` 三个副本，是为了让三个操作都从相同输入开始。

LeetCode 215“数组中的第 K 个最大元素”对应的核心入口就是：
```cpp
int findKthLargest(vector<int>& nums, int k) {
    int target = nums.size() - k;
    return quickSelect(nums, 0, nums.size() - 1, target);
}
```
题目保证 `k` 合法时，可以省略参数检查。
## 六、复杂度与常见问题
### 时间复杂度
一次 `partition` 只扫描当前区间一次：
```text
Partition: O(n)
```
Quick Sort 平均每次把数组分成大小接近的两半：
```text
T(n) = 2T(n/2) + O(n)
     = O(n log n)
```
如果每次 Pivot 都是最大值或最小值：
```text
T(n) = T(n - 1) + O(n)
     = O(n^2)
```
Quick Select 每次只进入一边，平均情况为：
```text
T(n) = T(n/2) + O(n)
     = n + n/2 + n/4 + ...
     = O(n)
```
最坏情况同样是 `O(n^2)`。
### 为什么已排序数组可能很慢
本文总是选择最后一个元素为 Pivot。对于：
```text
[1, 2, 3, 4, 5]
```
每次 Pivot 都是最大值，分区只能排除一个元素：
```text
长度 5 -> 4 -> 3 -> 2 -> 1
```
LeetCode 中常见的改进是在分区前随机选择 Pivot：
```cpp
#include <cstdlib>
int randomIndex = left + rand() % (right - left + 1);
swap(nums[randomIndex], nums[right]);
```
后面的分区代码完全不变。随机化不能消除理论上的最坏情况，但能避免已排序输入稳定触发最坏情况。
### 空数组和非法 k
Quick Sort 的调用：
```cpp
quickSort(nums, 0, nums.size() - 1);
```
空数组时右边界转换后需要谨慎处理。更安全的外层写法是：
```cpp
if (!nums.empty()) {
    quickSort(nums, 0, nums.size() - 1);
}
```
Quick Select 必须保证：
```text
1 <= k <= nums.size()
```
否则不存在第 k 小或第 k 大。LeetCode 题目会保证 `k` 合法，所以完整代码没有加入异常处理。
### 重复元素
本文使用：
```cpp
nums[j] <= pivot
```
所以等于 Pivot 的元素会被放到左边。这不会影响正确性，但如果数组中大量元素相等，分区可能很不均匀，性能退化到 `O(n^2)`。

处理大量重复元素时可以使用三路分区：
```text
[< pivot] [= pivot] [> pivot]
```
但三路分区需要更多指针。应先完全理解本文的二路分区，再学习该优化。
### Quick Sort 是否稳定
不稳定。`swap` 可能改变相等元素原来的相对顺序。

需要稳定排序时使用：
```cpp
std::stable_sort
```
实际工程中需要完整排序时，通常直接使用：
```cpp
std::sort
```
只需要第 k 小/大时可使用：
```cpp
std::nth_element
```
最后只需记住：
> `partition` 确定一个 Pivot 的最终位置；Quick Sort 继续处理两边，Quick Select 只处理目标所在的一边。
