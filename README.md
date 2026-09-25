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

运行依赖只有 NumPy 和 SciPy。测试另需 SymPy，画图需要 Matplotlib。

## 快速上手

三轴：G01 点列直接插补。

```python
import cnc5x as cx

points = cx.datasets.load_dataset("rhombic").points
path = cx.LinearPath(points)                       # 每段直线一个 block
profile, v = cx.schedule(path, v_max=100, a_max=3000, j_max=60000, Ts=0.0005)
commands = cx.interpolate(path, profile, Ts=0.0005)
commands.position                                  # (N, 2) 每个插补周期的刀尖位置
commands.tip                                       # (4, N, 2) 位置及其 1–3 阶时间导数
```

五轴：离散刀位 → 五轴样条 → AC 双转台。

```python
data = cx.datasets.load_dataset("horseshoe_planar_sweep")
path = cx.CurvePath(cx.pose_spline(data.points, data.axes), chord_error=1e-3)
profile, v = cx.schedule(path, v_max=50, a_max=500, j_max=5000, Ts=0.001)
commands = cx.interpolate(path, profile, Ts=0.001, machine=cx.TableTilting("AC"))
commands.q                                         # (4, N, 5) 机床轴 X Y Z A C 及其时间导数
```

全库只有一种导数表示：导数栈，形状为 `(4, ..., dim)`，第 k 层是第 k 阶导数。曲线对参数、刀路对弧长、指令对时间，都用这个形状。

## 模块

| 模块 | 内容 |
|---|---|
| `curves` | `Line`、`Bezier`、`BSpline`、`NURBS`、`SubCurve`、`Reparameterized`；弧长表 `ArcLengthTable` |
| `orientation` | 刀轴曲线：`GreatCircle`（slerp）、`UnitDirection`、`DualCurveDirection`（双样条刀轴） |
| `fitting` | `chord_parameters`、`interpolate_bspline`（参数由调用者给出）、`fit_bspline`（最小二乘） |
| `toolpath` | `PoseCurve`（五轴刀位）、`Block`、`ToolPath` 基类，内置 `LinearPath`、`CurvePath`，以及 `pose_spline`、`dual_spline` |
| `kinematics` | `TableTilting("AC" / "BC")`：正逆解、C 角连续展开、机床轴解析导数、极点处理 |
| `profiles` | 分段恒 jerk 轮廓 `Profile`，七段、五段 S 曲线，时间缩放与周期对齐 |
| `limits` | 弓高误差、法向加速度与法向 jerk 限速，`DriveLimits`，各轴约束区间，时间缩放倍数 |
| `look_ahead`、`scheduler` | 双向扫描；整条刀路的速度规划 |
| `interpolator` | `interpolate`（向量化周期插补）、`taylor_interpolate`（经典 Taylor 参数插补） |
| `metrics` | 路径偏差、拐角误差、弓高误差、进给波动、切向量、轴峰值、连接处 G² 检查 |
| `io`、`datasets` | 读 APT（`GOTO/`）与纯数字刀位文件；内置 11 条平面轮廓、9 组五轴刀位 |

公式、约定与推导见 [docs/数学约定.md](docs/数学约定.md)，代码风格与设计规范见 [CLAUDE.md](CLAUDE.md)。

## 论文复现

每篇论文一个目录，把它的刀路写成 `ToolPath` 的子类，只写论文自己的数学：

| 目录 | 论文 | 内容 |
|---|---|---|
| [papers/Zhao2013](papers/Zhao2013/README.md) | Zhao, Zhu & Ding, IJMTM 2013 | 曲率连续的双三次 Bézier 拐角过渡、五段 S 曲线前瞻 |
| [papers/Xu2018](papers/Xu2018/README.md) | Xu & Sun, IJAMT 2018 | 双三次 B 样条外切圆角 |

这两篇由 `cnc_interpolation` 移植而来，与原实现逐项对拍：Xu2018 吻合到 1e-9；Zhao2013 只在原代码加 `1e-6` 分母补丁的近直行拐角处有差异，block 长度仍吻合到 1e-9。

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

测试共 91 项，约 5 秒跑完。参考值都独立于库本身：

- SymPy 符号求导，以及 mpmath 40 位精度的数值求导；
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
- `CurvePath` 只在曲率峰值处限速；曲率近乎恒定的长区段（例如整圆）需要另加限制。
- 限速和各轴约束的检查都是在插补样本上做的，不等于连续域上的证明。
- 旧版的 20 篇教学 notebook 和论文 notebook 尚未迁移到新接口。
- 历史版本：`五轴复现/cnc5`（最早的平铺版）→ `五轴复现/cnc5x`（上一版）→ 本库。Gemini 生成的空骨架已备份为 `五轴复现/gemini骨架_cnc5x_20260925.tar.gz`。
