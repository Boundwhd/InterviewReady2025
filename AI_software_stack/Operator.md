# 什么是算子？
AI 框架中对张量计算的种类有很多，比如加法、乘法、矩阵相乘、矩阵转置等，这些计算被称为算子（Operator），它们是 AI 框架的核心组件。为了更加方便的描述计算图中的算子，现在来对**算子**这一概念进行定义：
- **狭义的算子（Kernel）**：对张量 Tensor 执行的基本操作集合，包括四则运算，数学函数，甚至是对张量元数据的修改，如维度压缩（Squeeze），维度修改（reshape）等。
- **广义的算子（Function）**：AI 框架中对算子模块的具体实现，涉及到调度模块，Kernel 模块，求导模块以及代码自动生成模块。狭义的算子，统一称之为核（Kernel），在 AI 框架中，使用 C++ 实现层里的算子指的就是这里的 Kernel，而这里的 Kernel 实现并不支持自动梯度计算（Autograd）模块，也不感知微分的概念。广义的算子我们将其称之为函数或方法（Function），这也是我们平时经常接触到的PyTorch API，包括Python API和C++ API，其配合PyTorch Autograd模块后就可以支持自动梯度求导计算。


# AI编译器后端优化

### 前端优化和后端优化的区别
- 前端优化：
输入计算图，关注计算图整体拓扑结构，不关系算子的具体实现。对算子节点进行融合、消除、化简等操作，使计算图的计算和存储开销最小。

- 后端优化：
关注算子节点的内部具体实现，针对具体实现使得性能达到最优。重点关心节点的输入、输出，内存循环方式和计算的逻辑。

计算图->TensorIR->LowerIR->代码生成->硬件平台执行

### 算子类型
1. 访存密集型
2. 计算密集型

目的：对于一个算子，要实现正确的逻辑计算或许不难，但要结合硬件能力达到高性能就比较难。例如CuDNN。

## 算子的计算与调度
算子：<br>
深度学习算法由一个个计算单元组成，我们称这些计算单元为算子。算子是一个函数空间到函数空间上的映射；从广义上讲，对任何函数进行某一项操作都可以认为是一个算子。对于AI框架而言，所开发的算子是网络模型中涉及到的计算函数。

调度：算子的具体实现和执行策略。同一个算子会有不同的实现方式。

计算实现和计算在硬件单元上的调度是分离的。

### 算子的调度
在硬件层面上极致优化结果，用到了SIMD、平铺(Tiling)、展开(Unrolling)和向量化(Vectorization)等常用技术。充分利用硬件性能。

#### 例如卷积算子的实现，3个计算特征
1. 多重循环为特点；
2. 没有复杂控制流；
3. 以多维张量计算为主要数据；

### 算子调度优化方法
#### 循环优化
- 循环展开
- 循环分块
- 循环重排
- 循环融合
- 循环拆分

#### 指令优化
- 向量化

#### 内存优化
- 访存延迟
- 存储分配