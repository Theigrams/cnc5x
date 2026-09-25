# cnc5x：五轴数控插补教学库

cnc5x 用于学习和复现五轴数控插补论文。它覆盖一条完整的流水线：参数曲线、刀路光顺、前瞻、S 曲线速度规划、周期插补、机床运动学和评价指标。代码以可读性为先，数值计算交给 NumPy 和 SciPy，每个公式都有独立的测试参考。内部单位为 mm、s、rad，刀轴由刀尖指向刀柄。

```
Curve ─▶ ToolPath ─▶ bidirectional_scan ─▶ schedule ─▶ interpolate ─▶ metrics
曲线      刀路         前瞻                  速度规划     插补           评价
```

## 安装

使用现有的 `gpt` 环境做可编辑安装：

```bash
/Users/theigrams/miniforge3/envs/gpt/bin/python -m pip install --no-deps --no-build-isolation -e /Users/theigrams/Desktop/cnc5x
```

运行依赖只有 NumPy 和 SciPy。测试另需 SymPy 与 mpmath，画图和 notebook 需要 Matplotlib 与 Jupyter。

## 快速上手

三轴：G01 点列直接插补。

```python
import numpy as np
import cnc5x as cx

points = cx.datasets.load_dataset("rhombic").points
path = cx.LinearPath(points)                       # 每段直线一个 block
profile, s, v = cx.schedule(path, v_max=100, a_max=3000, j_max=60000, Ts=0.0005)
commands = cx.interpolate(path, profile, Ts=0.0005)
commands.position                                  # (N, 2) 每个插补周期的刀尖位置
commands.tip                                       # (4, N, 2) 位置及其 1–3 阶时间导数
```

五轴：离散刀位 → 五轴样条 → AC 双转台。

```python
data = cx.datasets.load_dataset("horseshoe_planar_sweep")
path = cx.CurvePath(cx.pose_spline(data.points, data.axes), chord_error=1e-3)
profile, s, v = cx.schedule(path, v_max=50, a_max=500, j_max=5000, Ts=0.001)
commands = cx.interpolate(path, profile, Ts=0.001, machine=cx.TableTilting("AC"))
commands.q                                         # (4, N, 5) 机床轴 X Y Z A C 及其时间导数
```

各轴约束：沿弧长采样进给上限（进给包络），速度规划保证 block 内部的进给也不超出它。包络只含匀速通过时的部分，切向加减速引起的轴加速度、jerk 要插补后用 `time_scale_factor` 检查（`examples/five_axis.py` 第 6、7 步有一个包络管不住 C 轴 jerk 的例子）。

```python
machine = cx.TableTilting("AC")
drives = cx.DriveLimits(velocity=[60, 60, 60, 1, 2], acceleration=[600, 600, 600, 5, 10], jerk=[6e3, 6e3, 6e3, 50, 100])
s_env = np.linspace(0, path.length, 5000)
v_env = cx.feed_envelope(path, s_env, 0.001, 50, 500, 5000, chord_error=1e-3, machine=machine, drives=drives)
profile, s, v = cx.schedule(path, 50, 500, 5000, 0.001, envelope=(s_env, v_env))
```

五轴 G01 拐角光顺：刀尖与刀轴一起用五次 Hermite 过渡（教学基线），刀尖 G²、刀轴对刀尖弧长 C² 连续。

```python
points = np.array([[0, 0, 0], [30, 0, 0], [42, 18, 3], [60, 18, 0], [78, -6, 6]], float)  # 刀尖 (N, 3)
axes = cx.unit([[0, 0, 1], [0.3, 0, 1], [0.3, 0.4, 1], [-0.2, 0.3, 1], [0, 0, 1]])       # 刀轴 (N, 3)
path = cx.HermiteCornerPath(points, tolerance=0.05, chord_error=1e-3, axes=axes)
cx.metrics.junction_jumps(path.curves).max()                                          # ≈ 1e-14
```

全库只有一种导数表示：导数栈，形状为 `(4, ..., dim)`，第 k 层是第 k 阶导数。曲线对参数、刀路对弧长、指令对时间，都用这个形状。

## 模块

| 模块 | 内容 |
|---|---|
| `curves` | `Line`、`Bezier`、`BSpline`、`NURBS`、`SubCurve`、`Reparameterized`；误差受控的自适应弧长表 `ArcLengthTable` |
| `orientation` | 刀轴曲线：`GreatCircle`（slerp）、`UnitDirection`、`DualCurveDirection`（双样条刀轴）、`SphericalCurve`（球坐标） |
| `fitting` | 弦长与角度参数化；B 样条插值、（带约束）最小二乘，方程自己列；`hermite`（任意阶两点 Hermite）、`monotone_interpolate`（C² 单调插值，用于参数同步）、`feed_correction`（进给修正多项式）、`spherical_spline` |
| `toolpath` | `PoseCurve`（五轴刀位）、`Block`、`ToolPath` 基类，内置 `LinearPath`、`HermiteCornerPath`、`CurvePath`，以及 `pose_spline`（可让刀轴按自己的角度参数化再同步）、`dual_spline`、`hermite_transition` |
| `kinematics` | `TableTilting("AC" / "BC")`：正逆解、C 角连续展开、机床轴解析导数、极点处理 |
| `profiles` | 分段恒 jerk 轮廓 `Profile`，七段、五段 S 曲线，时间缩放与周期对齐 |
| `limits` | 弓高误差（含定义域外的饱和）、法向加速度与法向 jerk 限速；`DriveLimits`、各轴匀速限速、时间缩放倍数；各轴约束下切向加速度与 jerk 的可行区间（供逐周期调度用，`schedule` 不用） |
| `look_ahead`、`scheduler` | 双向扫描；进给包络 `feed_envelope`；整条刀路的速度规划 `schedule` |
| `interpolator` | `interpolate`（向量化周期插补）、`taylor_interpolate`（经典 Taylor 参数插补）、`correction_interpolate`（进给修正多项式插补） |
| `metrics` | 路径偏差、拐角误差、弓高误差、进给波动、切向速度（及加速度、jerk）、轴峰值、连接处的连续性（五轴时刀尖与刀轴分列） |
| `tolerances` | 全库的数值容差，每个值旁边写明理由 |
| `io`、`datasets` | 读 APT（`GOTO/`）与纯数字刀位文件；内置 11 条平面轮廓、9 组五轴刀位 |

公式、约定与推导见 [docs/数学约定.md](docs/数学约定.md)，代码风格与设计规范见 [CLAUDE.md](CLAUDE.md)。

## 论文复现

每篇论文一个目录，把它的刀路写成 `ToolPath` 的子类，只写论文自己的数学：

| 目录 | 论文 | 内容 |
|---|---|---|
| [papers/Zhao2013](papers/Zhao2013/README.md) | Zhao, Zhu & Ding, IJMTM 2013 | 曲率连续的双三次 Bézier 拐角过渡、五段 S 曲线前瞻 |
| [papers/Xu2018](papers/Xu2018/README.md) | Xu & Sun, IJAMT 2018 | 双三次 B 样条外切圆角 |

这两篇由 `cnc_interpolation` 移植而来。与原实现对拍（rhombic、butterfly、griffen 三组数据）的结果：

- **几何**：Xu2018 的 block 长度吻合到 2e-11。Zhao2013 原代码在式 (5) 的分母上加了 `1e-6`，影响**所有**拐角：rhombic 上 block 长度差 3e-7；butterfly 上转角小于 0.01 rad 的 7 个拐角过渡长度 d2 差到 0.23 mm；griffen 上式 (13) 的逐次减半从略有不同的初值出发，走了不同的分支，276 个拐角的 d2 相差最多一倍，block 长度差到 6e-3 mm。本库按原式计算，两者都满足论文的约束。
- **速度规划**：弓高限速在弓高容差超过曲率半径两倍处，原代码给出 v = 0，本库改为饱和于 2ρ/Ts（见 `limits.chord_error_limit`），不再无谓停车：Zhao2013 在 butterfly 上少停 6 处、griffen 上少停 15 处，Xu2018 在 butterfly 上少停 2 处。这些拐角的法向加速度限速本来就更小，总时长变化不到 0.1%。原代码还把曲率下限截为 1e-6：griffen 上 Zhao2013 的 104 个直行拐角因此限速为 0，Xu2018 的 104 个近直行拐角限速约 0.008 mm/s；本库对直行拐角按曲率为零、不受弓高约束处理。两篇的弓高容差都沿用原复现、取各拐角的实际逼近误差，近直行拐角上它只有 1e-13 mm 量级，会把限速压得很低（本库的 Xu2018 在 griffen 上仍有 19 个近直行拐角因此降到 39–98 mm/s），这一取法是否符合原文有待核对。

新增一篇论文的写法：

```python
from cnc5x import Bezier, PolylinePath, geometric_limit

class MyCornerPath(PolylinePath):
    def __init__(self, points, tolerance):
        super().__init__(points)                      # 已算好 tangents、L、turning_angles
        ...                                           # 论文的过渡参数
        self.blocks = self.corner_blocks(transitions) # 每个拐角的 (前半段, 后半段)

    def get_v_limit(self, Ts, v_max, a_max, j_max):   # 连接点速度上限 (n_blocks + 1,)
        ...
```

五轴 G01（给出 `axes`）时，`corner_blocks` 用刀尖直线加刀轴大圆补上直线部分；过渡曲线要与接点处的导数栈吻合，可用 `self.corner_ends(i, 裁去长度, 裁去长度)` 取出，再交给 `hermite_transition`（见 `HermiteCornerPath`）。

## 示例与测试

在项目根目录运行：

```bash
/Users/theigrams/miniforge3/envs/gpt/bin/python -m examples.curves
```

```bash
/Users/theigrams/miniforge3/envs/gpt/bin/python -m examples.corner_smoothing butterfly
```

```bash
/Users/theigrams/miniforge3/envs/gpt/bin/python -m examples.five_axis
```

```bash
/Users/theigrams/miniforge3/envs/gpt/bin/python -m pytest tests -q
```

测试约 140 项，5 秒左右跑完。参考值都独立于库本身：

- SymPy 符号求导，以及 mpmath 40 位精度的数值求导；
- SciPy 自带的样条插值与最小二乘（另一套实现）、SLSQP 数值优化、`scipy.integrate.quad` 求弧长；
- 逐轴直接检查不等式（各轴约束的区间与缩放倍数）、圆上弦长与弓高的几何关系；
- Biagiotti & Melchiorri 教材中的 S 曲线例题；
- 解析解（圆、直线、多项式）；
- 与 cnc_interpolation 原有测试数据对拍。

## 数据来源

内置数据的来源登记在 [cnc5x/datasets/registry.py](cnc5x/datasets/registry.py)：

- `reference` 只填核对过的文献，没核对的一律留空；
- 五轴刀位来自 `测试刀位数据/`，由 virtualfiveaxisCNCmachining MATLAB 工程的输入数据转换而来。

旧版 cnc5x 里名为 `iso_s_shape`、`fleisig_fan` 的两组数据，经核对其实是同一个 MATLAB 输入文件（前者还改动过一行），与所称的出处不符，没有迁移。

## 现状与局限

- 机床只实现了理想相交轴的 AC/BC 双转台，固定平移 `b` 不代表旋转轴线之间的偏置。
- 不给进给包络时，`CurvePath` 只在曲率峰值处限速；曲率近乎恒定的长区段（例如整圆）要用 `feed_envelope`。
- block 加 S 曲线的结构在连接点处加速度必须回到零，贴不住大段缓慢起伏的包络（五轴各轴约束常常如此），只能加连接点或整体放慢，离时间最优还有距离（见 `docs/数学约定.md` 第 3 节）。
- 限速和各轴约束的检查都是在插补样本上做的，不等于连续域上的证明。
- G01 刀路里刀尖不动、只转刀轴的段暂不支持，会报错。
- 暂不实现的功能与方案记在 [docs/路线图.md](docs/路线图.md)。
- 旧版的 20 篇教学 notebook 和论文 notebook 尚未迁移到新接口。
- 历史版本：`五轴复现/cnc5`（最早的平铺版）→ `五轴复现/cnc5x`（上一版）→ 本库。Gemini 生成的空骨架已备份为 `五轴复现/gemini骨架_cnc5x_20260925.tar.gz`。
