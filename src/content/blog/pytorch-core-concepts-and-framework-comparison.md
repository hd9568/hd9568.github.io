---
title: 'PyTorch 核心理念与框架对比：从动态图、Autograd 到 PyTorch / TensorFlow / Paddle API 兼容'
description: '结合框架 API 兼容和模型迁移场景，系统讲解 PyTorch 的核心设计、张量、动态图、Autograd、Module、DataLoader、编译路径，并用代码对比 PyTorch、TensorFlow 和 PaddlePaddle。'
category: 'Research & Work'
pubDate: '2026-07-13T16:41:00+08:00'
updatedDate: '2026-07-13T16:41:00+08:00'
heroImage: '../../assets/blog-placeholder-1.jpg'
---

## 目录

1. [为什么要理解 PyTorch](#一为什么要理解-pytorch)
2. [PyTorch 的核心理念](#二pytorch-的核心理念)
3. [Tensor：PyTorch 的基础数据结构](#三tensorpytorch-的基础数据结构)
4. [动态图：define-by-run](#四动态图define-by-run)
5. [Autograd：自动求导怎么理解](#五autograd自动求导怎么理解)
6. [nn.Module：模型组织方式](#六nnmodule模型组织方式)
7. [DataLoader：数据输入管线](#七dataloader数据输入管线)
8. [训练循环：PyTorch 为什么容易调试](#八训练循环pytorch-为什么容易调试)
9. [PyTorch 和 TensorFlow 代码对比](#九pytorch-和-tensorflow-代码对比)
10. [PyTorch 和 Paddle API 兼容为什么重要](#十pytorch-和-paddle-api-兼容为什么重要)
11. [从框架开发角度看 API 映射](#十一从框架开发角度看-api-映射)
12. [PyTorch 2.x：从 eager 到 compile](#十二pytorch-2x从-eager-到-compile)
13. [面试表达](#十三面试表达)
14. [总结](#十四总结)

## 一、为什么要理解 PyTorch

PyTorch 是当前深度学习研究和工程中最常见的框架之一。理解 PyTorch 不只是会写模型，还要理解它为什么适合研究、为什么容易调试、为什么和 TensorFlow / Paddle 之间存在 API 兼容和迁移问题。

在框架兼容场景中，常见任务是：

```text
把 PyTorch 代码迁移到 PaddlePaddle；
或让 Paddle API 尽量兼容 PyTorch 参数命名和行为；
或自动生成 PyTorch-Paddle API 映射文档。
```

这类工作要求理解：

- PyTorch API 的语义。
- 参数命名和默认值。
- Tensor 维度约定。
- inplace 行为。
- dtype / device 规则。
- Autograd 行为。
- 模块和函数式 API 的区别。

本文从这些角度系统梳理 PyTorch。

## 二、PyTorch 的核心理念

PyTorch 的核心理念可以概括为：

```text
Python-first
动态图
命令式执行
易调试
张量计算 + 自动求导 + 神经网络模块
```

### Python-first

PyTorch 尽量让用户写普通 Python 代码：

```python
if condition:
    y = model_a(x)
else:
    y = model_b(x)
```

这对研究很重要，因为模型结构经常变化。

### 命令式执行

PyTorch 默认是 eager mode。每一行代码都会立即执行。

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = x * 2
print(y)
```

`y` 会马上得到结果，不需要先构建静态图再 session run。

### 动态图

PyTorch 的计算图是在运行时根据实际 Python 执行路径动态构建的。每次 forward 都可以构建不同图。

这让 PyTorch 很适合：

- 变长输入。
- 条件分支。
- 递归模型。
- 动态 MoE。
- 调试复杂模型。

## 三、Tensor：PyTorch 的基础数据结构

Tensor 是 PyTorch 的核心数据结构。它可以理解为带 dtype、device、shape、stride 的多维数组。

```python
import torch

x = torch.randn(2, 3, device="cuda", dtype=torch.float16)
print(x.shape)   # torch.Size([2, 3])
print(x.dtype)   # torch.float16
print(x.device)  # cuda:0
```

Tensor 的几个重要属性：

| 属性 | 含义 |
| --- | --- |
| `shape` | 每个维度大小 |
| `dtype` | 数据类型，如 fp32、fp16、bf16 |
| `device` | CPU 或 GPU |
| `stride` | 多维索引到内存地址的步长 |
| `requires_grad` | 是否参与 Autograd |

### stride 为什么重要

例如：

```python
x = torch.arange(12).reshape(3, 4)
y = x.t()
print(y.shape)
print(y.stride())
```

`transpose` 通常不复制数据，只改变 stride。很多性能问题和算子兼容问题都与 non-contiguous tensor 有关。

如果某个底层 kernel 要求连续内存，就可能需要：

```python
y = y.contiguous()
```

框架 API 兼容时，不能只看 shape，还要注意是否保持 view / copy 语义。

## 四、动态图：define-by-run

PyTorch 的动态图可以用下面例子理解。

```python
import torch

def f(x):
    if x.sum() > 0:
        return x * x
    else:
        return x + 1

x1 = torch.tensor([1.0, 2.0], requires_grad=True)
y1 = f(x1)

x2 = torch.tensor([-1.0, -2.0], requires_grad=True)
y2 = f(x2)
```

`x1` 和 `x2` 走了不同分支，构建的计算图也不同。

这就是 define-by-run：

```text
程序怎么跑，图就怎么建。
```

优点：

- 调试方便。
- 控制流自然。
- 研究代码表达力强。

代价：

- eager mode 每步有 Python overhead。
- 静态优化空间不如完整图编译。
- 大规模部署时通常需要 TorchScript、ONNX、`torch.compile` 或专门推理引擎。

## 五、Autograd：自动求导怎么理解

Autograd 是 PyTorch 自动求导系统。

只要 Tensor 设置：

```python
requires_grad=True
```

PyTorch 会在 forward 时记录操作，形成反向图。

简单例子：

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x * x + 3 * x
y.backward()

print(x.grad)  # dy/dx = 2x + 3 = 7
```

### 计算图中的 grad_fn

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.sum()

print(y.grad_fn)
print(z.grad_fn)
```

`grad_fn` 记录了如何反向传播。

### no_grad

推理时通常不需要构建反向图：

```python
with torch.no_grad():
    y = model(x)
```

这样可以减少显存和计算图开销。

### detach

`detach` 会把 Tensor 从计算图中分离：

```python
y = x.detach()
```

常见于：

- 停止梯度。
- teacher-student。
- logging。
- 避免保存不必要图。

## 六、nn.Module：模型组织方式

PyTorch 用 `nn.Module` 组织模型。

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

model = MLP(128, 256, 10).cuda()
```

`nn.Module` 的职责：

- 管理参数。
- 管理子模块。
- 支持 `.cuda()`、`.to(dtype)`。
- 支持 `state_dict()` 保存和加载。
- 支持 train/eval 模式。

```python
print(model.state_dict().keys())
model.eval()
model.train()
```

### Module API 与 functional API

PyTorch 同时提供：

```python
nn.ReLU()
torch.nn.functional.relu(x)
```

区别：

- `nn.Module` 有状态，可以注册参数或 buffer。
- `functional` 通常是无状态函数。

例如 Dropout：

```python
self.dropout = nn.Dropout(0.1)
```

Module 会根据 `model.train()` / `model.eval()` 自动切换行为。

## 七、DataLoader：数据输入管线

训练通常使用 `Dataset` 和 `DataLoader`。

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

loader = DataLoader(
    MyDataset(data, labels),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
```

关键参数：

- `batch_size`：batch 大小。
- `shuffle`：是否打乱。
- `num_workers`：数据加载子进程数量。
- `pin_memory`：使用 pinned memory，加速 CPU 到 GPU 拷贝。

在 AI Infra 中，DataLoader 可能成为瓶颈。GPU 利用率低时，需要排查：

- CPU decode 是否慢。
- 数据增强是否慢。
- H2D 拷贝是否慢。
- `num_workers` 是否合适。
- NUMA 亲和性是否合理。

## 八、训练循环：PyTorch 为什么容易调试

典型训练循环：

```python
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for x, y in loader:
    x = x.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)

    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

PyTorch 的优点在这里很明显：

- forward 是普通 Python。
- loss 是普通 Tensor。
- `loss.backward()` 触发反向。
- 可以随时 `print`、debug、断点。

如果出现梯度异常，可以直接检查：

```python
for name, p in model.named_parameters():
    if p.grad is not None:
        print(name, p.grad.norm())
```

## 九、PyTorch 和 TensorFlow 代码对比

### 线性层模型

PyTorch：

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

model = Net()
x = torch.randn(3, 4)
y = model(x)
```

TensorFlow / Keras：

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_shape=(4,))
])

x = tf.random.normal([3, 4])
y = model(x)
```

### 自定义训练

PyTorch：

```python
pred = model(x)
loss = loss_fn(pred, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

TensorFlow：

```python
with tf.GradientTape() as tape:
    pred = model(x)
    loss = loss_fn(target, pred)

grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

对比：

| 项目 | PyTorch | TensorFlow |
| --- | --- | --- |
| 默认执行 | eager | TF2 默认 eager，常配合 `tf.function` |
| 自动求导 | `loss.backward()` | `tf.GradientTape()` |
| 模型组织 | `nn.Module` | `tf.keras.Model` |
| 部署生态 | TorchScript / ONNX / torch.compile / ExecuTorch | SavedModel / TensorFlow Serving / TFLite |
| 研究体验 | 更 Pythonic | Keras 高层 API 友好 |

### 静态图和编译

TensorFlow 早期更偏静态图：

```text
先构图，再执行。
```

PyTorch 早期更偏动态图：

```text
边执行边构图。
```

现在两边都在融合：

- TensorFlow 2 默认 eager，但可以 `tf.function` 编译。
- PyTorch 默认 eager，但可以 `torch.compile`。

## 十、PyTorch 和 Paddle API 兼容为什么重要

在模型迁移中，经常需要把 PyTorch 代码转换为 PaddlePaddle。

问题不只是函数名不同，还包括：

- 参数名不同。
- 默认值不同。
- Tensor layout 不同。
- 返回值类型不同。
- inplace 行为不同。
- dtype 推断不同。
- device 语义不同。
- 训练 / 推理模式切换不同。

比如 PyTorch：

```python
torch.nn.functional.softmax(input, dim=-1)
```

Paddle 可能更常见：

```python
paddle.nn.functional.softmax(x, axis=-1)
```

`dim` 和 `axis` 语义类似，但参数名不同。

一个兼容层可能要做：

```python
def softmax(input=None, dim=None, axis=None, **kwargs):
    if axis is None:
        axis = dim
    return paddle.nn.functional.softmax(input, axis=axis, **kwargs)
```

这类工作看似简单，但大规模 API 映射时很容易出现边界问题。

## 十一、从框架开发角度看 API 映射

假设要做一个 PyTorch 参数兼容装饰器，目标是让 Paddle API 支持 PyTorch 风格参数名。

一个简化例子：

```python
import functools

def alias_kwargs(alias_map):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for old_name, new_name in alias_map.items():
                if old_name in kwargs:
                    if new_name in kwargs:
                        raise TypeError(
                            f"got both {old_name} and {new_name}"
                        )
                    kwargs[new_name] = kwargs.pop(old_name)
            return fn(*args, **kwargs)
        return wrapper
    return deco

@alias_kwargs({"dim": "axis"})
def paddle_softmax_compat(x, axis=-1):
    # return paddle.nn.functional.softmax(x, axis=axis)
    return f"softmax(axis={axis})"

print(paddle_softmax_compat("x", dim=-1))
```

关键点：

- 不能简单覆盖已有参数。
- 要处理同义参数同时出现的报错。
- 要保留函数签名或文档。
- 要覆盖默认值和 None 行为。
- 要有 CI 测试。

API 映射文档自动化通常需要：

```text
读取 PyTorch API 信息
读取 Paddle API 信息
比较参数签名
维护映射规则
渲染文档模板
CI 校验文档是否过期
自动部署
```

这就是为什么框架 API 兼容是一项工程工作，而不是简单写几行 wrapper。

## 十二、PyTorch 2.x：从 eager 到 compile

PyTorch 2.x 引入 `torch.compile`，把 eager 程序捕获并编译优化。

```python
model = torch.compile(model)
```

大致流程：

```text
Python eager code
-> TorchDynamo 捕获 graph
-> AOTAutograd 处理 forward/backward
-> Inductor 生成优化 kernel
```

这说明 PyTorch 的方向不是放弃动态图，而是：

```text
保留动态图易用性；
在可捕获区域自动编译优化。
```

对推理和训练性能来说，这很重要。但编译也有约束：

- 动态 shape 可能导致重新编译。
- Python side effect 可能影响捕获。
- 某些自定义 op 无法自动优化。
- 调试编译图更复杂。

## 十三、面试表达

可以这样讲 PyTorch：

```text
PyTorch 是 Python-first 的深度学习框架，默认 eager execution。
它通过 Tensor 表达多维数组，通过 Autograd 在运行时记录计算图并自动求导，通过 nn.Module 组织模型参数和子模块。
它的动态图机制让研究和调试非常方便，但 eager 模式有 Python overhead，因此 PyTorch 2.x 又通过 torch.compile 把可捕获图交给编译器优化。
```

如果问 PyTorch 和 TensorFlow：

```text
PyTorch 早期更强调动态图和命令式编程，TensorFlow 早期更强调静态图和部署。
TF2 之后默认 eager，PyTorch 2.x 之后也有 compile，两者都在向易用性和编译性能结合的方向发展。
```

如果问 API 兼容：

```text
跨框架 API 兼容不只是函数名映射，还要处理参数命名、默认值、dtype、device、inplace、返回值和边界行为。
例如 PyTorch 的 dim 和 Paddle 的 axis 语义类似，但兼容层需要处理参数冲突、None 行为和文档自动生成。
```

## 十四、总结

PyTorch 的核心可以压缩成：

```text
Tensor + eager execution + dynamic graph + Autograd + nn.Module
```

它的优势是易用、灵活、易调试；它的工程挑战是如何在动态图灵活性和高性能编译之间平衡。

对于框架兼容和模型迁移来说，理解 PyTorch API 的真实语义非常重要。函数名可以自动替换，但参数行为、默认值、Tensor layout、梯度和 inplace 语义必须逐项确认。
