# CLAUDE.md — cnc5x 协作规范

cnc5x 是五轴数控插补的教学与论文复现库。本文件规定代码风格、设计与协作方式，以它为准。改代码前先读一遍，改完按第 12 节自查。数学公式与约定的细节见 [docs/数学约定.md](docs/数学约定.md)。

## 1. 定位

- 服务于教学和论文复现，不做工业级插补器。优先级依次是：可读性、数值正确、速度。
- 库里只放两类东西：
  - 教科书级的基础构件；
  - 至少两篇论文会复用的公共部分。
- 论文特有的方法写在 `papers/<作者年份>/` 里，用库里的构件搭起来。
- 不为"以后可能用到"预先增加功能、参数或抽象层。等第二处真正需要时再抽取。

## 2. 架构

主流程的每一步都对应一个名词：

```
Curve ─▶ ToolPath ─▶ bidirectional_scan ─▶ schedule ─▶ interpolate ─▶ metrics
曲线      刀路         前瞻                  速度规划     插补           评价
```

| 模块 | 职责 | 可以依赖 |
|---|---|---|
| `tolerances.py` | 全库的数值容差，每个值旁边写明理由 | 无 |
| `calculus.py` | 三阶链式法则 `compose`、Leibniz `product`，以及单位化、反函数、arccos、辐角的导数 | numpy |
| `geometry.py` | 单位化、夹角、折线切向与转角、旋转矩阵、曲率 | numpy |
| `curves.py` | `Curve` 基类、自适应弧长表，Line、Bezier、BSpline、NURBS、SubCurve、Reparameterized | calculus、geometry、tolerances、scipy |
| `orientation.py` | 刀轴曲线：GreatCircle、UnitDirection、DualCurveDirection、SphericalCurve | curves |
| `fitting.py` | 参数化、B 样条插值与（带约束）最小二乘、Hermite、单调插值、进给修正多项式、球坐标刀轴 | curves、orientation、scipy |
| `limits.py` | 几何限速、DriveLimits、各轴约束区间、时间缩放倍数 | numpy |
| `toolpath.py` | PoseCurve、Block、ToolPath 及 PolylinePath、LinearPath、HermiteCornerPath、CurvePath | curves、orientation、fitting、limits |
| `kinematics.py` | 双转台正逆解与机床轴解析导数 | calculus、geometry |
| `profiles.py` | 分段恒 jerk 进给轮廓，七段、五段 S 曲线 | scipy |
| `look_ahead.py`、`scheduler.py` | 双向扫描、进给包络、整条刀路的速度规划 | profiles、limits |
| `interpolator.py` | 周期插补 `interpolate`、Taylor 参数插补、进给修正插补 | calculus、profiles |
| `metrics.py` | 评价指标 | calculus、geometry、scipy |
| `io.py`、`datasets/` | 刀位文件读取、内置数据 | geometry |
| `plotting.py` | notebook 里反复出现的图（进给四联图、机床轴图、刀轴箭头）和统一配色；matplotlib 是可选依赖，不在 `__init__` 中导入 | matplotlib |

- 依赖只能从上往下，下层模块不能 import 上层模块。
- 新增模块前，先确认它确实不能并入现有模块。

## 3. 核心约定（改动时必须保持）

1. **导数栈**：形状 `(4, ..., dim)`，`d[k]` 是第 k 阶导数，不是 Taylor 系数。对参数、对弧长、对时间的导数一律用这个形状。
2. **链式法则只有一处实现**，即 `calculus.compose`。需要复合求导时调用它，不要另写一份。
3. **曲线接口**：
   - 子类只需设定 `domain`，并实现 `_derivatives(u, order)`，一次返回 0 到 `order` 阶导数叠成的 `(order + 1, ..., dim)`。复合曲线（NURBS、单位化、复合函数）求高阶导数要用到全部低阶导数，一次求一叠才不会重复计算；
   - 外部通过 `curve(u)`、`curve(u, k)`、`curve.derivatives(u)`、`curve.derivatives_by_length(u)`（对弧长的导数栈）使用；
   - `u` 可以是标量或任意形状的数组，返回形状 `(..., dim)`。
4. **参数域**保持用户给定的值，不归一化到 [0, 1]。`breaks` 必须包含所有不光滑的点。
5. **进给轮廓接口**：`profile(t)` 返回 `(4, ...)`，即 `[s, v, a, j]`（也是导数栈），另有 `duration`、`length` 两个属性。插补器只依赖这三样。
6. **刀路接口**：`ToolPath` 的子类负责生成 `self.blocks`，并实现 `get_v_limit(Ts, v_max, a_max, j_max)`，返回 `(n_blocks + 1,)`。`schedule` 返回 `(profile, 连接点的弧长位置, 连接点速度)`。
7. **五轴**：`PoseCurve` 求值得到 6 维 `[p, o]`，6 维的路径一律按五轴处理。进给默认沿刀尖弧长计量，只转刀轴时用 `along="axis"`。
8. **单位**：内部统一用 mm、s、rad，刀轴由刀尖指向刀柄。
   - 输入输出如果用 degree、mm/min，在边界处显式换算。
   - 毫米和弧度不能直接相加或比较。
9. **机床约定**：`R_AC = R_x(A) R_z(C)`，`R_BC = R_y(B) R_z(C)`，`[X, Y, Z] = R p + b`，`o = Rᵀ e_z`。约定一变就是另一台机床，要新写一个类，不能改现有的类。
10. **容差**集中在 `tolerances.py`，公式里不写字面量容差。

## 4. 代码风格

- **语法从简**：以函数、少量普通类和 dataclass 为主。不用 ABC、Protocol、TypeVar、overload、元类，不写装饰器技巧，不设别名参数（`**aliases`）。
- 一个函数只做一件事，一般不超过 30 行，超过就拆。
- 不写多层推导式，不用 itertools 技巧。0 到 3 阶、两个分支这类小循环直接写 `for`。
- **向量化**：用广播（如 `u[..., None] * delta`），矩阵乘法用 `@`。尽量不用 `einsum`；确实要用时，注释写明各下标的含义。
- **命名**：变量名贴近论文符号（`d2`、`c1`、`beta`、`N0`…`N8`），公式旁标注 `# Eq. (n)` 或定理编号。标识符用英文，与 CIRP / IJMTM 的术语一致（见第 7 节）。
- **docstring 用中文**：
  - 第一行说清做什么；
  - 有公式就用 Unicode 数学符号写出来；
  - 写明数组形状和单位；
  - 有前提或局限也要写上。
- 注释写"为什么"，不复述代码在做什么。
- 类型注解只写简单的，不强求。
- 格式由 ruff 统一（行宽 120）：`ruff format` 和 `ruff check` 都必须通过。
- docstring 的正文不要全是缩进行，否则 ruff 会去掉缩进。公式块后面补一句普通说明即可。

## 5. 数值规范

- **哪些交给 SciPy、哪些自己写**，判断标准是：这段代码是否承载某篇论文或教材的算法思想。
  - 交给 SciPy：B 样条求值与基函数、节点插入、数值积分的节点权重、求根、线性方程组、KD 树。它们是数值内核，自己重写不增加理解，还要处理边界情况。
  - 自己写：参数化、节点选择、拟合的方程组（插值、最小二乘、约束）、弧长映射、前瞻与速度规划。调用 SciPy 的整套拟合函数会把思想藏进黑盒。
  - 同一类操作两种处理并存时写明理由（例如 `Bezier.split` 手写 de Casteljau，`BSpline.split` 调 `scipy.interpolate.insert`）。
- **分母上不加 epsilon。** 遇到零向量、零速度、极点等退化情况，二选一：
  - 显式报错，并说明原因；
  - 按数学定义分情况处理（例如直行的"拐角"、极点处取相邻值）。
- 输入检查只放在公开入口（构造函数、`Curve.derivative`），公式内部不写防御代码。
- 导数一律解析求得。差分只出现在测试的参考值里。
- 舍入容差集中管理，并注释原因（例如参数域的 1e-9 相对容差）。
- 在样本上检查通过不等于连续域上的证明。文档和报告里要如实写"在样本上"。

## 6. 测试

- 每个新公式都要有独立的参考值，不能只和自己比。可用的参考：
  - SymPy 符号求导；
  - mpmath 高精度数值求导；
  - 教材例题；
  - 解析解（圆、直线、多项式）；
  - 与已审阅的旧实现对拍。
- 测试不依赖网络；用到随机数时每个测试自己建 `np.random.default_rng(种子)`，不共用模块级的随机数生成器。整套测试保持在 10 秒以内，慢的符号计算改用 mpmath。
- 修 bug 时补一条回归测试，并确认它在修之前会失败。
- 容差贴近真实精度，断言要有区分度。例如写"二阶比一阶小一个数量级"，而不是让结果贴着阈值过关。
- 常用命令：

```bash
/Users/theigrams/miniforge3/envs/gpt/bin/python -m pytest tests -q
```

```bash
ruff format . && ruff check .
```

## 7. 术语对照

| 概念（CIRP / IJMTM） | 代码中的名字 | 中文 |
|---|---|---|
| interpolation period | `Ts` | 插补周期 |
| chord error | `chord_error` | 弓高误差 |
| look-ahead, bidirectional scanning | `bidirectional_scan` | 前瞻、双向扫描 |
| feedrate profile, jerk-limited S-curve | `Profile`、`seven_phase`、`five_phase` | 进给轮廓、S 曲线 |
| feedrate fluctuation | `feedrate_fluctuation` | 进给速度波动 |
| tool tip, tool axis orientation | `tip`、`axis` | 刀尖、刀轴 |
| table-tilting machine (AC / BC) | `TableTilting` | 双转台 |
| local corner smoothing | `corner_blocks`、`papers/*` | 局部拐角光顺 |
| dual-spline tool path | `dual_spline`、`DualCurveDirection` | 双样条刀路 |
| kinematic singularity (pole) | `pole` | 奇异点（极点） |
| time scaling | `Profile.scaled`、`time_scale_factor` | 时间缩放 |
| nonlinear error | `nonlinear_error` | 非线性误差（周期内各轴线性插值引起的刀位偏差） |
| approximation error / contour error | `corner_error` / — | 逼近误差（光顺偏差）/ 轮廓误差（伺服跟随） |

引入新概念前，先查它在 CIRP Annals、IJMTM 等期刊中的通行叫法，再给代码命名。

## 8. 论文复现（`papers/`）

- 目录结构为 `papers/<第一作者><年份>/`，包含 `algorithm.py`、`README.md`。只有这篇论文用的数据放在它的 `data/`；与其他论文共用的数据登记到 `cnc5x/datasets`。
- `algorithm.py`：继承 `PolylinePath`、`ToolPath` 等库类，只写论文自己的数学，公式编号与原文保持一致。
- `README.md` 要包含：
  - 完整题录和 DOI（用 paper2 技能核对）；
  - 算法步骤与代码的对照表；
  - 与原文的差异、已知勘误。
- 写清楚是否与原文的图表对上了。没对上就写"未复现"，不要含糊。
- **不照抄参考代码**，包括 MATLAB 参考库 virtualfiveaxisCNCmachining：其中有已确认的错误。它只能当功能清单，实现一律按论文公式自己写。

## 9. 数据

- 内置数据登记在 `cnc5x/datasets/registry.py`。
  - `source` 写明文件来自哪里；
  - `reference` 只填核对过的文献，没核对的一律为 `None`。
- 不编造数据出处或文献，不修改原始数据点。发现数据与所称来源不符时，在 registry 里如实写明。

## 10. 文档

- 用 Markdown 写数学时：行内公式用 `\(...\)`；独立公式用 `$$`，且前后两个 `$$` 各自单独占一行。向量和矩阵用 `\mathbf{}`，单位向量加 `\hat{}`，标量不加粗。
- `docs/数学约定.md` 与代码的 docstring 保持一致，改公式时两处一起改。
- 用中文写，先讲物理或几何直觉，再给公式，最后对应到代码。

## 11. Notebook

- 库里每一块设计（新模块、新算法、新刀路类）都要配一本教学 notebook，放在 `notebooks/<编号>_<主题>.ipynb`，提交时带着执行后的输出。pytest 负责守住正确性；notebook 负责讲清原理，也方便人类调试时直接看图、看数字。
- notebook 由 `notebooks/_build/build_<编号>.py` 生成并执行，写作规范见 `notebooks/_build/README.md`。改 notebook 内容时改对应的 build 脚本再重建，不要直接编辑 `.ipynb`。
- 开头第一个代码单元格固定为：

```python
import os.path as osp
import sys
from pathlib import Path

import matplotlib.pyplot as plt

root_path = Path(osp.abspath("")).parents[0]
sys.path.append(str(root_path))

%config InlineBackend.figure_format='retina'
```

  第二个单元格是 `import numpy as np`、`import cnc5x as cx`、`from cnc5x import plotting`、`plotting.use_style()`。
- 结构：标题单元格列出本节要回答的问题。每一节依次是：直觉、公式、代码、图或数字、要点。结尾是"练习"和"延伸阅读"。
- **图里的文字（标题、坐标轴、图例、标注）一律用英文**，避免中文字体缺失。Markdown 用中文。
- notebook 里行内公式用 `$...$`，因为有的渲染器会把 `\(` 当转义吃掉；独立公式仍用 `$$` 且各自单独占一行。
- 画图规则：
  - 颜色按 `plotting.COLORS` 的固定顺序分配，同一实体在整本 notebook 里颜色不变；
  - 参考数据用 `GRAY`，上限用 `LIMIT` 虚线；
  - 不用双 y 轴；
  - 两个及以上系列要有图例。
- 每个主要结论配一个带独立参考值的检查单元格（打印误差并 `assert`）。Markdown 里引用的数字必须和执行输出一致。
- 单本 notebook 的执行时间不超过 60 秒。ruff 同样检查 notebook。

## 12. 完成改动前自查

- [ ] 一个没读过这段代码的人能在 5 分钟内看懂
- [ ] 没有引入新的抽象层、全局状态或 epsilon
- [ ] 公开接口的形状、单位和前提都写在 docstring 里
- [ ] 新公式有独立参考的测试；pytest 与 ruff 全部通过
- [ ] 改了公共接口（签名、返回值、形状）时，在全仓库 grep 旧用法，包括 README、docs、examples、notebooks、papers
- [ ] README、`docs/数学约定.md`、论文 README、相关 notebook 已同步更新
- [ ] 暂不实现的想法记进 `docs/路线图.md`
