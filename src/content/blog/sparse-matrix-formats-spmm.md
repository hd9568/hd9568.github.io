---
title: '稀疏矩阵乘法中的常见稀疏格式：COO、CSR、CSC、ELL、BSR 与 HYB'
description: '系统梳理稀疏矩阵乘法中常见的稀疏存储格式，用具体矩阵说明表示方式、乘法访问路径、实现思路和优缺点。'
category: 'Research & Work'
pubDate: '2026-06-24T18:15:00+08:00'
updatedDate: '2026-06-24T18:15:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [为什么需要稀疏格式](#一为什么需要稀疏格式)
2. [贯穿全文的矩阵例子](#二贯穿全文的矩阵例子)
3. [COO：坐标格式](#三coo坐标格式)
4. [CSR：按行压缩格式](#四csr按行压缩格式)
5. [CSC：按列压缩格式](#五csc按列压缩格式)
6. [DIA：对角线格式](#六dia对角线格式)
7. [ELL：固定行宽格式](#七ell固定行宽格式)
8. [SELL：切片 ELL 格式](#八sell切片-ell-格式)
9. [BSR：块稀疏格式](#九bsr块稀疏格式)
10. [HYB：混合格式](#十hyb混合格式)
11. [DOK、LIL 与 Bitmap](#十一doklil-与-bitmap)
12. [稀疏矩阵乘法怎么做](#十二稀疏矩阵乘法怎么做)
13. [格式选择建议](#十三格式选择建议)
14. [总结](#十四总结)

## 一、为什么需要稀疏格式

稀疏矩阵的核心特征是大量元素为 0。直接按 dense matrix 存储会浪费显存/内存，也会做大量无意义乘法。

例如一个 `m x n` 矩阵，如果只有 `nnz` 个非零元素：

```text
dense 存储空间: O(mn)
sparse 存储空间: O(nnz) + 索引开销
```

稀疏格式要解决三个问题：

- 非零值存在哪里。
- 非零值的行列位置怎么恢复。
- 做矩阵乘法时如何高效遍历这些非零值。

在 AI Infra 和高性能计算中，常见乘法形式包括：

- `SpMV`：Sparse Matrix Vector Multiplication，稀疏矩阵乘向量，`y = A x`。
- `SpMM`：Sparse Matrix Dense Matrix Multiplication，稀疏矩阵乘稠密矩阵，`C = A X`。
- `SpGEMM`：Sparse Matrix Sparse Matrix Multiplication，稀疏矩阵乘稀疏矩阵，`C = A B`。

不同稀疏格式本质上是在“存储紧凑性、索引开销、访存连续性、并行友好性、构建/更新成本”之间取舍。

## 二、贯穿全文的矩阵例子

后文用同一个矩阵 `A` 举例：

```text
A = 4 x 5

[10  0  0  0  2]
[ 0  3  0  0  0]
[ 4  0  5  0  0]
[ 0  0  0  7  8]
```

非零元素共有 7 个：

```text
(0,0)=10, (0,4)=2
(1,1)=3
(2,0)=4,  (2,2)=5
(3,3)=7,  (3,4)=8
```

如果做 `SpMV`：

```text
x = [1, 2, 3, 4, 5]^T

y = A x
  = [10*1 + 2*5,
     3*2,
     4*1 + 5*3,
     7*4 + 8*5]^T
  = [20, 6, 19, 68]^T
```

如果做 `SpGEMM`，再给一个稀疏矩阵 `B`：

```text
B = 5 x 3

[0  1  0]
[2  0  0]
[0  0  3]
[4  0  0]
[0  5  6]
```

则：

```text
C = A B = 4 x 3

[ 0 20 12]
[ 6  0  0]
[ 0  4 15]
[28 40 48]
```

## 三、COO：坐标格式

`COO` 是 Coordinate Format，直接记录每个非零元素的 `(row, col, value)`。

对矩阵 `A`，COO 表示为：

```text
row = [0, 0, 1, 2, 2, 3, 3]
col = [0, 4, 1, 0, 2, 3, 4]
val = [10,2, 3, 4, 5, 7, 8]
```

每个下标位置对应一个非零元素：

```text
k=0 -> A[0][0] = 10
k=1 -> A[0][4] = 2
k=2 -> A[1][1] = 3
```

### COO 如何参与 SpMV

计算 `y = A x` 时，遍历所有非零元素：

```cpp
for (int k = 0; k < nnz; ++k) {
    int i = row[k];
    int j = col[k];
    y[i] += val[k] * x[j];
}
```

对例子中的前两个非零元素：

```text
k=0: y[0] += 10 * x[0] = 10
k=1: y[0] +=  2 * x[4] = 10
```

最终 `y[0] = 20`。

### 优点

- 格式直观，容易构建。
- 适合从日志、边列表、三元组数据中导入。
- 插入新非零元素比较简单。
- 适合作为中间格式，再转换为 CSR/CSC。

### 缺点

- 每个非零元素都要存 row 和 col，索引开销大。
- SpMV 中多个线程可能同时写同一个 `y[i]`，并行时需要原子加或按行分组规约。
- 如果没有按行排序，访存局部性较差。

COO 更像“原始稀疏数据格式”，适合构建和交换，不一定适合高性能乘法内核的最终格式。

## 四、CSR：按行压缩格式

`CSR` 是 Compressed Sparse Row，按行压缩存储，是 SpMV/SpMM 中最常见的格式之一。

它用三个数组：

- `values`：按行存储所有非零值。
- `col_indices`：每个非零值对应的列号。
- `row_ptr`：每一行非零元素在 `values` 中的起止位置。

对矩阵 `A`：

```text
values      = [10, 2, 3, 4, 5, 7, 8]
col_indices = [ 0, 4, 1, 0, 2, 3, 4]
row_ptr     = [ 0, 2, 3, 5, 7]
```

`row_ptr` 的长度是 `num_rows + 1`。第 `i` 行的非零元素位于：

```text
values[row_ptr[i] ... row_ptr[i+1]-1]
```

例如第 2 行：

```text
row_ptr[2] = 3
row_ptr[3] = 5

values[3:5]      = [4, 5]
col_indices[3:5] = [0, 2]

所以 A[2][0]=4, A[2][2]=5
```

### CSR 如何参与 SpMV

CSR 的 SpMV 非常自然：每一行独立算一个输出 `y[i]`。

```cpp
#include <vector>

std::vector<float> csr_spmv(
    int rows,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_indices,
    const std::vector<float>& values,
    const std::vector<float>& x
) {
    std::vector<float> y(rows, 0.0f);

    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
            int j = col_indices[p];
            sum += values[p] * x[j];
        }
        y[i] = sum;
    }

    return y;
}
```

对第 3 行：

```text
row_ptr[3]=5, row_ptr[4]=7

y[3] = values[5] * x[col_indices[5]]
     + values[6] * x[col_indices[6]]
     = 7 * x[3] + 8 * x[4]
     = 68
```

### CSR 如何参与 SpMM

如果右侧是稠密矩阵 `X`，例如 `X` 是 `5 x N`，输出 `Y = A X` 是 `4 x N`：

```cpp
void csr_spmm(
    int rows,
    int out_cols,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_indices,
    const std::vector<float>& values,
    const std::vector<float>& X,
    std::vector<float>& Y
) {
    // X 的形状是 K x out_cols，按行主序存储。
    for (int i = 0; i < rows; ++i) {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
            int k = col_indices[p];
            float a = values[p];

            for (int n = 0; n < out_cols; ++n) {
                Y[i * out_cols + n] += a * X[k * out_cols + n];
            }
        }
    }
}
```

这段代码的访问逻辑是：

```text
A 的一行中每个非零元素 A[i,k]，都会把 X 的第 k 行加权累加到 Y 的第 i 行。
```

### 优点

- 按行计算很方便，SpMV/SpMM 的主流格式。
- 行边界明确，适合一行一个线程、一个 warp 或一个 block。
- 比 COO 少存一个 `row` 数组，索引开销更低。
- 对按行访问的算法非常友好。

### 缺点

- 插入或删除非零元素代价高，可能需要移动后续数组。
- 行非零数量差异大时，并行负载不均衡。
- 访问 `x[col_indices[p]]` 或 `X[k, :]` 可能是不连续的，容易受 cache 命中率影响。
- 对按列访问不友好。

CSR 是工程中最常见的“默认选择”，尤其适合行方向遍历的稀疏乘法。

## 五、CSC：按列压缩格式

`CSC` 是 Compressed Sparse Column，可以理解为 CSR 的转置版本。它按列压缩存储。

三个数组：

- `values`：按列存储非零值。
- `row_indices`：每个非零值对应的行号。
- `col_ptr`：每一列非零元素在 `values` 中的起止位置。

对矩阵 `A`，按列列出非零元素：

```text
col 0: (0,0)=10, (2,0)=4
col 1: (1,1)=3
col 2: (2,2)=5
col 3: (3,3)=7
col 4: (0,4)=2, (3,4)=8
```

CSC 表示为：

```text
values      = [10,4, 3, 5, 7, 2,8]
row_indices = [ 0,2, 1, 2, 3, 0,3]
col_ptr     = [ 0,2, 3, 4, 5, 7]
```

第 `j` 列的非零元素位于：

```text
values[col_ptr[j] ... col_ptr[j+1]-1]
```

### CSC 如何参与 SpMV

计算 `y = A x` 时，CSC 的思路是按列展开：

```cpp
for (int j = 0; j < cols; ++j) {
    float xj = x[j];
    for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
        int i = row_indices[p];
        y[i] += values[p] * xj;
    }
}
```

例如第 4 列：

```text
col 4: A[0][4]=2, A[3][4]=8

y[0] += 2 * x[4]
y[3] += 8 * x[4]
```

### 优点

- 按列访问非常自然。
- 适合求 `A^T x`、列切片、列归约等操作。
- 在 SpGEMM 中，如果需要频繁访问右矩阵的列，CSC 会更方便。

### 缺点

- 对普通 `y = A x`，会向多个 `y[i]` 分散写，CPU/GPU 并行时可能需要处理写冲突。
- 按行遍历不方便。
- 和 CSR 一样，不适合频繁插入删除。

简单理解：CSR 适合“拿出一行去乘”，CSC 适合“拿出一列去乘”。

## 六、DIA：对角线格式

`DIA` 是 Diagonal Format，适合非零元素集中在少数几条对角线上的矩阵。

例如矩阵 `A` 的非零元素分布在这些对角线上：

```text
offset = col - row

(0,0), (1,1), (2,2), (3,3) -> offset 0
(0,4)                     -> offset 4
(2,0)                     -> offset -2
(3,4)                     -> offset 1
```

所以对角线偏移为：

```text
diagonal_offsets = [-2, 0, 1, 4]
```

按这些 offset 存储每条对角线的值。为了便于说明，用 `*` 表示 padding：

```text
offset -2: [*, *, 4, *]
offset  0: [10,3,5,7]
offset  1: [*, *, *, 8]
offset  4: [2, *, *, *]
```

也可以写成二维数组：

```text
diag_values =
[
  [*, *, 4, *],
  [10,3,5,7],
  [*, *, *, 8],
  [2, *, *, *]
]
```

### DIA 如何参与 SpMV

对每条对角线遍历行号：

```cpp
for (int d = 0; d < num_diags; ++d) {
    int offset = diagonal_offsets[d];

    for (int i = 0; i < rows; ++i) {
        int j = i + offset;
        if (0 <= j && j < cols) {
            y[i] += diag_values[d][i] * x[j];
        }
    }
}
```

例如主对角线 `offset=0`：

```text
y[0] += 10 * x[0]
y[1] +=  3 * x[1]
y[2] +=  5 * x[2]
y[3] +=  7 * x[3]
```

### 优点

- 对规则带状矩阵非常高效。
- 每条对角线连续访问，访存模式简单。
- 索引开销低，只需要存对角线 offset。
- GPU 上容易做向量化和并行。

### 缺点

- 如果非零元素不集中在少数对角线，padding 很多，空间浪费严重。
- 不适合非结构化稀疏矩阵。
- 插入不在已有对角线上的元素，会改变格式结构。

DIA 适合有限差分、图像处理、PDE 离散化这类具有规则邻接结构的矩阵，不适合任意稀疏图。

## 七、ELL：固定行宽格式

`ELL` 或 `ELLPACK` 会给每一行分配相同数量的槽位。这个槽位数通常是所有行的最大非零数量。

矩阵 `A` 每行非零数量是：

```text
row 0: 2
row 1: 1
row 2: 2
row 3: 2
```

最大值是 2，所以 ELL 每行存 2 个位置：

```text
ell_values =
[
  [10,2],
  [ 3,*],
  [ 4,5],
  [ 7,8]
]

ell_cols =
[
  [0,4],
  [1,*],
  [0,2],
  [3,4]
]
```

### ELL 如何参与 SpMV

```cpp
for (int i = 0; i < rows; ++i) {
    float sum = 0.0f;
    for (int t = 0; t < max_nnz_per_row; ++t) {
        int j = ell_cols[i][t];
        if (j != -1) {
            sum += ell_values[i][t] * x[j];
        }
    }
    y[i] = sum;
}
```

对第 1 行：

```text
ell_values[1] = [3,*]
ell_cols[1]   = [1,*]

y[1] = 3 * x[1]
```

### 优点

- 每一行循环次数相同，GPU 并行友好。
- 结构规则，容易向量化。
- 适合每行非零数量接近的矩阵。

### 缺点

- 行非零数量差异大时 padding 多。
- 如果某一行特别长，会把所有行的宽度都拉大。
- 不适合极端不均匀稀疏矩阵。

ELL 的优势是“整齐”，问题也是“过于整齐”。在稀疏神经网络或图计算中，如果 degree 分布长尾明显，单纯 ELL 往往浪费较大。

## 八、SELL：切片 ELL 格式

`SELL` 是 Sliced ELL。它把矩阵按若干行切成 slice，每个 slice 内部使用 ELL。

假设 slice 高度为 2：

```text
slice 0: row 0, row 1
slice 1: row 2, row 3
```

slice 0 的行非零数量是 `[2,1]`，局部最大宽度是 2：

```text
slice 0 values =
[
  [10,2],
  [ 3,*]
]

slice 0 cols =
[
  [0,4],
  [1,*]
]
```

slice 1 的行非零数量是 `[2,2]`，局部最大宽度也是 2：

```text
slice 1 values =
[
  [4,5],
  [7,8]
]

slice 1 cols =
[
  [0,2],
  [3,4]
]
```

如果矩阵行长度差异更大，SELL 会比全局 ELL 少很多 padding，因为每个 slice 只按局部最大行宽补齐。

### SELL 如何参与 SpMV

```cpp
for (int s = 0; s < num_slices; ++s) {
    int row_begin = s * slice_height;
    int width = slice_width[s];

    for (int local_i = 0; local_i < slice_height; ++local_i) {
        int i = row_begin + local_i;
        float sum = 0.0f;

        for (int t = 0; t < width; ++t) {
            int j = sell_cols[s][local_i][t];
            if (j != -1) {
                sum += sell_values[s][local_i][t] * x[j];
            }
        }

        y[i] = sum;
    }
}
```

### SELL-C-sigma

工程里经常看到 `SELL-C-sigma`：

- `C`：slice 高度，常和 SIMD/warp 宽度相关。
- `sigma`：排序窗口大小，在窗口内按行非零数量重排，减少 padding。

它的思想是：

```text
先把相似长度的行放到一起，再按 slice 做 ELL。
```

### 优点

- 比 ELL 更能适应行长度不均匀的矩阵。
- 保留较规则的内存访问模式。
- 适合 SIMD/SIMT 架构。

### 缺点

- 格式比 CSR/ELL 更复杂。
- 如果做了行重排，需要维护原始行号映射。
- 对动态更新不友好。

SELL 常用于高性能 SpMV 库，是 CSR 与 ELL 之间的折中。

## 九、BSR：块稀疏格式

`BSR` 是 Block Sparse Row，也叫 Blocked CSR。它把矩阵按固定大小的 dense block 划分，只存非零块。

为了更清楚，换一个适合块稀疏的矩阵 `M`：

```text
M = 4 x 4，block size = 2 x 2

[1 2 0 0]
[3 4 0 0]
[0 0 5 6]
[0 0 7 8]
```

按 `2 x 2` 切块：

```text
block(0,0) = [1 2; 3 4]
block(0,1) = zero
block(1,0) = zero
block(1,1) = [5 6; 7 8]
```

BSR 表示类似 CSR，只不过值数组里存的是 block：

```text
block_values =
[
  [[1,2],
   [3,4]],

  [[5,6],
   [7,8]]
]

block_col_indices = [0, 1]
block_row_ptr     = [0, 1, 2]
```

这里有 2 个 block row，每个 block row 对应原矩阵中的 2 行。

### BSR 如何参与 SpMV

```cpp
for (int br = 0; br < block_rows; ++br) {
    for (int p = block_row_ptr[br]; p < block_row_ptr[br + 1]; ++p) {
        int bc = block_col_indices[p];
        const float* block = block_values[p]; // block_dim x block_dim

        for (int bi = 0; bi < block_dim; ++bi) {
            int row = br * block_dim + bi;
            for (int bj = 0; bj < block_dim; ++bj) {
                int col = bc * block_dim + bj;
                y[row] += block[bi * block_dim + bj] * x[col];
            }
        }
    }
}
```

对第一个块：

```text
[1 2]   [x0]
[3 4] * [x1]
```

它一次处理一个小 dense GEMV，而不是逐元素处理。

### 优点

- 对块结构稀疏非常高效。
- 块内可以使用 dense 计算，提高数据复用。
- 索引按 block 存储，索引开销比逐元素 CSR 更低。
- 适合结构化剪枝、有限元、图中多维特征聚合等场景。

### 缺点

- 如果块内也很稀疏，会存很多块内 0。
- block size 选择很关键，太小收益有限，太大浪费空间。
- 不适合完全无结构的随机稀疏矩阵。

在深度学习里，硬件友好的稀疏往往更偏 block sparse，而不是任意 unstructured sparse，因为块内 dense 计算更容易吃满 Tensor Core 或 SIMD 单元。

## 十、HYB：混合格式

`HYB` 是 Hybrid Format，常见组合是 `ELL + COO`。

思想很简单：

```text
大多数行用规则的 ELL 存。
超出固定宽度的长尾非零元素，用 COO 补充。
```

假设某个矩阵每行大多只有 2 个非零元素，但有一行有 5 个非零元素。可以设置：

```text
ELL width = 2
```

每行前 2 个非零元素进入 ELL，额外的 3 个元素放入 COO。

对本文矩阵 `A`，如果设置 `ELL width = 1`，则：

```text
ELL 部分：

ell_values =
[
  [10],
  [ 3],
  [ 4],
  [ 7]
]

ell_cols =
[
  [0],
  [1],
  [0],
  [3]
]
```

剩下元素进入 COO：

```text
coo_row = [0, 2, 3]
coo_col = [4, 2, 4]
coo_val = [2, 5, 8]
```

### HYB 如何参与 SpMV

```cpp
// 先计算 ELL 部分。
for (int i = 0; i < rows; ++i) {
    for (int t = 0; t < ell_width; ++t) {
        int j = ell_cols[i][t];
        if (j != -1) {
            y[i] += ell_values[i][t] * x[j];
        }
    }
}

// 再累加 COO 尾部。
for (int k = 0; k < coo_nnz; ++k) {
    y[coo_row[k]] += coo_val[k] * x[coo_col[k]];
}
```

### 优点

- 保留 ELL 的规则访存。
- 避免 ELL 被少数超长行拖垮。
- 对 GPU SpMV 比较友好。

### 缺点

- 有两套格式，内核和数据管理更复杂。
- ELL width 需要调参。
- COO 尾部仍可能有原子写或规约问题。

HYB 是典型的工程折中：常规部分走高效路径，异常部分单独处理。

## 十一、DOK、LIL 与 Bitmap

除了高性能乘法格式，还有一些更适合构建、更新或特殊稀疏模式的格式。

### DOK

`DOK` 是 Dictionary of Keys，用哈希表存 `(row, col) -> value`。

对矩阵 `A`：

```text
{
  (0,0): 10,
  (0,4): 2,
  (1,1): 3,
  (2,0): 4,
  (2,2): 5,
  (3,3): 7,
  (3,4): 8
}
```

优点：

- 插入、删除、查找单个元素方便。
- 适合逐步构建稀疏矩阵。

缺点：

- 哈希表开销大。
- 遍历顺序不稳定，cache locality 差。
- 不适合作为高性能 SpMV/SpMM 的最终格式。

常见用法是：先用 DOK 构建，再转换为 CSR/CSC。

### LIL

`LIL` 是 List of Lists，每一行一个列表，存该行的 `(col, value)`。

对矩阵 `A`：

```text
row 0: [(0,10), (4,2)]
row 1: [(1,3)]
row 2: [(0,4), (2,5)]
row 3: [(3,7), (4,8)]
```

优点：

- 按行插入比较方便。
- 和 CSR 结构接近，容易转换。

缺点：

- 每行列表可能带来额外指针开销。
- 不如 CSR 连续紧凑。
- 乘法性能通常不如 CSR。

### Bitmap

Bitmap 格式用 bitmask 表示某个位置是否非零，再单独存非零值。

对矩阵 `A` 的非零 mask：

```text
1 0 0 0 1
0 1 0 0 0
1 0 1 0 0
0 0 0 1 1
```

如果按行存 bitmask：

```text
mask =
[
  10001,
  01000,
  10100,
  00011
]

values = [10,2,3,4,5,7,8]
```

优点：

- 适合中等稀疏或结构较固定的矩阵。
- 判断某个位置是否非零很快。
- 可利用位运算加速扫描。

缺点：

- 极稀疏时 bitmask 本身也可能浪费空间。
- 需要额外逻辑把 bit 位置映射到 values 下标。
- 对通用 SpMV 不一定比 CSR 更合适。

Bitmap 在一些 tile/block 内部很常见，例如一个小块内用 mask 描述哪些元素有效。

## 十二、稀疏矩阵乘法怎么做

### SpMV：`y = A x`

SpMV 的核心是：

```text
只遍历 A 的非零元素，并把 A[i,j] * x[j] 累加到 y[i]。
```

CSR 版本最清晰：

```cpp
for (int i = 0; i < rows; ++i) {
    float sum = 0.0f;
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
        sum += values[p] * x[col_indices[p]];
    }
    y[i] = sum;
}
```

复杂度是：

```text
O(nnz)
```

真正影响性能的通常不是乘加次数，而是访存：

- `values` 和 `col_indices` 通常连续读取。
- `x[col]` 是间接访问，cache 命中率取决于列分布。
- `y[i]` 写入通常连续，但 COO/CSC 可能分散写。

### SpMM：`Y = A X`

SpMM 中右侧是 dense matrix。CSR 的计算逻辑是：

```text
对 A 的每个非零 A[i,k]：
    Y[i, :] += A[i,k] * X[k, :]
```

伪代码：

```cpp
for (int i = 0; i < rows; ++i) {
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
        int k = col_indices[p];
        float a = values[p];

        for (int n = 0; n < N; ++n) {
            Y[i][n] += a * X[k][n];
        }
    }
}
```

和 SpMV 相比，SpMM 对 `X[k, :]` 的访问有更多连续数据，算术强度更高，通常更容易利用 SIMD/GPU。

### SpGEMM：`C = A B`

SpGEMM 比 SpMV/SpMM 更难，因为输出矩阵 `C` 的稀疏结构通常事先不知道。

以 CSR 为例，计算 `C = A B` 的一行：

```text
C[i, :] = sum over k where A[i,k] != 0:
          A[i,k] * B[k, :]
```

伪代码：

```cpp
for (int i = 0; i < A_rows; ++i) {
    // accumulator: col -> value
    std::unordered_map<int, float> acc;

    for (int ap = A_row_ptr[i]; ap < A_row_ptr[i + 1]; ++ap) {
        int k = A_col[ap];
        float a = A_val[ap];

        // 需要快速拿到 B 的第 k 行，因此 B 用 CSR 很自然。
        for (int bp = B_row_ptr[k]; bp < B_row_ptr[k + 1]; ++bp) {
            int j = B_col[bp];
            float b = B_val[bp];
            acc[j] += a * b;
        }
    }

    // 把 acc 中的非零结果写入 C 的第 i 行。
}
```

用前面的 `A` 和 `B` 看第 0 行：

```text
A row 0: A[0,0]=10, A[0,4]=2

C[0,:] = 10 * B[0,:] + 2 * B[4,:]
       = 10 * [0,1,0] + 2 * [0,5,6]
       = [0,20,12]
```

SpGEMM 的难点：

- 输出 `C` 每行有多少非零元素不固定。
- 不同乘积可能落到同一个输出列，需要合并。
- accumulator 可以用 hash、排序归并、dense scratchpad、heap 等，不同选择影响很大。
- 并行负载不均衡更明显。

## 十三、格式选择建议

下面是常见格式的快速对比：

| 格式 | 适合场景 | 主要优点 | 主要缺点 |
| --- | --- | --- | --- |
| COO | 构建、导入、临时格式 | 简单直观，插入方便 | 索引开销大，并行写冲突 |
| CSR | 行遍历、SpMV、SpMM | 紧凑，按行计算自然 | 行长度不均会负载不均 |
| CSC | 列遍历、`A^T x`、列切片 | 按列访问自然 | 普通 SpMV 写入分散 |
| DIA | 带状矩阵、规则 stencil | 索引少，访存规则 | 非规则稀疏会大量 padding |
| ELL | 每行非零数接近 | 结构整齐，GPU 友好 | 长尾行导致浪费 |
| SELL | 行长度中等不均 | 比 ELL 少 padding，仍较规则 | 需要切片和可能的行重排 |
| BSR | 块结构稀疏 | 块内 dense 计算效率高 | 块内 0 会浪费 |
| HYB | 大多数行规则，少数行异常 | 兼顾规则路径和长尾 | 实现复杂，需要调参数 |
| DOK/LIL | 动态构建和更新 | 插入删除方便 | 乘法性能通常较差 |
| Bitmap | tile 内 mask、中等稀疏 | 位运算友好，查询快 | 极稀疏时 mask 开销明显 |

实际工程里可以按这条思路判断：

```text
1. 只是构建矩阵：DOK / LIL / COO
2. 通用 SpMV/SpMM：CSR 优先
3. 需要频繁按列访问：CSC
4. 非零集中在固定对角线：DIA
5. 每行非零数接近并且跑 GPU：ELL / SELL
6. 存在明显块结构：BSR
7. 大多数行规则但有长尾：HYB
```

对 AI Infra 场景还要额外关注硬件：

- GPU 喜欢规则访存和批量计算，不喜欢随机访存和细粒度分支。
- 任意稀疏不一定快，稀疏带来的计算减少可能被索引开销和不规则访存抵消。
- 块稀疏通常比完全非结构化稀疏更容易获得真实加速。
- SpMM 通常比 SpMV 更容易跑快，因为右侧 dense matrix 提高了数据复用。

## 十四、总结

稀疏格式不是单纯的“压缩存储”，它直接决定稀疏矩阵乘法的访问模式和并行方式。

最重要的判断是：

```text
数据按行用得多，优先看 CSR。
数据按列用得多，优先看 CSC。
结构规则，考虑 DIA、ELL、SELL。
存在块结构，考虑 BSR。
需要动态构建，先用 COO、DOK、LIL，再转计算格式。
```

在面试或工作讨论中，可以这样表达：

```text
稀疏矩阵格式的选择取决于稀疏模式和计算模式。
CSR 适合按行 SpMV/SpMM，是通用默认选择；
ELL/SELL 更适合 GPU 上行长度相对均匀的场景；
BSR 适合块稀疏，因为块内可以走 dense 计算；
COO/DOK/LIL 更适合构建或中间表示，不一定适合最终高性能乘法。
```
