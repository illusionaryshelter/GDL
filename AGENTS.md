

# @GeoEmbodied

**背景上下文 (Context)：** 你当前正在 `GeoEmbodied` 的代码库中运行。这是一个专为机器人学 (Robotics)、自动驾驶 (Autonomous Driving)、SLAM 以及世界模型 (World Models) 设计的**几何深度学习 (Geometric Deep Learning, GDL) 核心库**。本代码库严格遵循群论、李群 (Lie Groups) 和微分几何的严格数学原理。

**最高指令 (The Prime Directive)：**
**物理规律和几何一致性绝对凌驾于常规的深度学习编程直觉之上。** 如果一个标准的 PyTorch 操作（例如普通的 `.add()` 或 `.mean()`）破坏了张量的拓扑约束或物理意义，**你绝对不能使用它**。

**test要求**
本地可用内存只有4GB，任何涉及pytest的测试必须禁止，交由用户进行远端测试。

## 数学与几何刚性约束 (Mathematical & Geometric Constraints)

* **规则 1：绝对禁止使用欧拉角 (NEVER use Euler Angles)。**
  欧拉角存在万向节死锁 (Gimbal Lock) 且在空间中不连续。所有 3D 旋转必须使用四元数 (Quaternions，优化内存) 或 $SO(3)$ 旋转矩阵表示。
* **规则 2：尊重流形与切空间 (Respect the Manifold)。**
  在更新位姿变量或编写自定义优化器时，**绝对禁止**对 $SE(3)$ 或 $SO(3)$ 对象直接应用加法梯度（严禁出现 `T = T + delta_T` 或 `loss.backward()` 后的直接 `w.data += grad`）。
  你必须使用指数映射 (Exponential Map) 在切空间（李代数 $\mathfrak{se}(3)$）上应用更新：$T_{new} = \exp(\xi^\wedge) \circ T_{old}$。
* **规则 3：维持等变性 (Maintain Equivariance)。**
  当在 `equivariant/` 目录下实现新的网络层时，该层函数 $f$ 必须满足 $f(\mathfrak{g}x) = \mathfrak{g}f(x)$（$\mathfrak{g}$ 为指定的对称群）。绝不能引入破坏这种对称性的操作（例如在空间维度上进行普通的 Batch Normalization 是禁止的，必须使用 `EquivariantBatchNorm`）。

## 数值稳定性防卫协议 (Numerical Stability Protocol)

GDL 极其容易发生梯度爆炸或 NaN。在生成代码时，必须强制包含以下保护：
* **规则 4：危险函数的安全截断 (Safe Clamp for Dangerous Ops)。**
  在使用 `torch.acos`、`torch.asin` 或 `torch.sqrt` 时，输入值必须进行 `clamp` 操作（例如 `x.clamp(-1.0 + eps, 1.0 - eps)`），以防止在反向传播时在奇点处产生 `NaN` 梯度。
* **规则 5：正交化与归一化修复 (Orthogonal & Normalization Projection)。**
  如果对旋转矩阵进行了多次连乘或插值，不可避免会产生浮点误差。必须定期调用内部的 `project_to_SO3()` 或 `normalize_quaternion()` 函数将其强行拉回流形。

## 新增：架构边界与依赖限制 (Architecture & Dependency Limits)

* **规则 6：不可侵犯的类抽象 (Inviolable Abstractions)。**
  如果上下文中存在自定义的 `LieTensor` 或 `GroupAction` 类，**必须使用它们的方法**。不允许为了代码简短而绕过封装，直接使用 `torch.matmul` 或 `einops.rearrange` 来处理带有物理意义的张量。
* **规则 7：严格的依赖控制 (Strict Dependency Control)。**
  除了 `torch`, `numpy`, `triton` 以及项目内置模块外，**禁止**在未获得用户明确允许的情况下引入外部复杂的数学库（如 `scipy.spatial.transform` 或其他未经优化的 Python 几何库），因为这会破坏计算图 (Autograd Graph) 或降低推理速度。

## 代码生成与测试规范 (Code Generation & Testing Standards)

* **算子性能要求：**
  * 涉及到高频调用的空间聚合（如球谐函数计算、半径图消息传递），请优先使用 **Triton** (`triton.jit`) 编写定制 Kernel。
* **类型提示与文档 (Typing & Docstrings)：**
  强制要求严格的 Python 类型提示 (Type Hinting)。对于表示物理量的张量，必须在 Docstring 或注释中**同时标注形状和群表示 (Group Representation)**。
  *示例：* `x: torch.Tensor # shape: [B, N, C], representation: SO(3) spherical harmonics up to l=2`
* **强制性等变测试 (Testing Mandates)：**
  当被要求为新模块编写测试时，**必须**包含“等变性误差测试 (Equivariance Error Test)”。
  逻辑如下：
  1. 生成随机输入 $x$。
  2. 生成随机群作用 $g \in G$。
  3. 计算 $y_1 = f(gx)$ 和 $y_2 = g(f(x))$。
  4. 断言相对误差 `torch.allclose(y1, y2, atol=1e-5)`。

## 领域特定词汇表 (Domain-Specific Vocabulary)

当你与用户沟通或生成 Git 提交信息时，请准确理解和使用本代码库的专属词汇：
* `Pose` = 位于 $SE(3)$ 群的元素（包含旋转和平移）。
* `Twist` = 位于李代数 $\mathfrak{se}(3)$ 的元素（切空间上的速度/微小运动）。
* `Feature` = 符合 $SO(3)$ 等变表示的 Type-$l$ 张量。
* `VLA` = Vision-Language-Action（视觉-语言-动作）模型融合。

