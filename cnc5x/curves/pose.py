"""五轴刀位：刀尖曲线与刀轴曲线配成的一对（PoseCurve），以及由刀位点、拐角导数构造它的函数。

单独成一个文件，是为了让五轴特有的东西都在一处：刀尖曲线是普通的 Curve（spline.py、fitting.py），
刀轴曲线在 orientation.py，把两者配对、判断 6 维输入的只有这里。
"""

import numpy as np

from ..utils.calculus import compose, inverse_derivatives, speed_derivatives
from ..utils.geometry import unit
from .curve import Reparameterized
from .fitting import chord_parameters, hermite, interpolate_bspline, monotone_interpolate
from .orientation import DualCurveDirection, UnitDirection


class PoseCurve:
    """五轴刀位曲线：刀尖 p(u) 与单位刀轴 o(u) 共用参数 u，求值为 6 维 [p, o]。

    为什么是两条曲线的组合，而不是 Curve 的子类：6 维的 [p, o] 不是空间中的点——弧长只由其中一条分量
    决定，曲率只是刀尖的曲率——Curve 的约定它一条都不满足。以前靠在基类上开口子（measured 切片）勉强
    继承，包装类稍不留神就会把 mm 和 rad 混在一起取模（待修复问题 P7）。导数栈仍拼成 6 维，是为了让
    calculus.compose 对刀尖和刀轴一起换元（CLAUDE.md 第 3.7 条）。刀路按鸭子类型使用它（第 3.6 条）。

    Args:
        tip (Curve): Tool tip curve p(u), 3-D, mm.
        axis (Curve): Unit tool axis curve o(u), 3-D, pointing from the tip to the holder.
        along (str): Component whose arc length measures the feed: "tip" (mm), or "axis"
            (rad, for motion that only rotates the tool axis).
    """

    def __init__(self, tip, axis, along="tip"):
        if tip.domain != axis.domain:
            raise ValueError("tip and axis curves must share the same parameter domain")
        if along not in ("tip", "axis"):
            raise ValueError(f'along must be "tip" or "axis", got {along!r}')
        self.tip, self.axis, self.along = tip, axis, along
        self.domain = tip.domain

    @property
    def _measured_curve(self):
        return self.tip if self.along == "tip" else self.axis

    @property
    def breaks(self):
        return np.unique(np.concatenate([self.tip.breaks, self.axis.breaks]))

    def derivatives(self, u, order=3):
        """[p, o] 对 u 的导数栈，只求到 `order` 阶。

        Args:
            u (...): Parameter values in `domain`.
            order (int): Highest derivative order, 0 to 3.

        Returns:
            d (order + 1, ..., 6): Tip derivatives in d[..., :3], axis derivatives in d[..., 3:].
        """
        return np.concatenate([self.tip.derivatives(u, order), self.axis.derivatives(u, order)], axis=-1)

    def __call__(self, u, order=0):
        """刀位 [p, o]（order = 0），或第 order 阶导数；形状 (..., 6)。"""
        return self.derivatives(u, order)[order]

    def derivatives_by_length(self, u):
        """[p, o] 对计量弧长 s（见 along）的导数栈。

        s 的导数只由计量的那条分量求出，再用同一组 (u_s, u_ss, u_sss) 对两条分量一起换元。按刀尖计量时，
        o_s 就是"刀尖每走 1 mm，刀轴转多少"，插补和各轴限速都要用它。

        Args:
            u (...): Parameter values in `domain`.

        Returns:
            d (4, ..., 6): d[k] = dᵏ[p, o]/dsᵏ.
        """
        d = self.derivatives(u)
        measured = d[..., :3] if self.along == "tip" else d[..., 3:]
        return compose(d, *inverse_derivatives(*speed_derivatives(measured)))

    @property
    def length(self):
        """计量分量的弧长：按刀尖为 mm，按刀轴为 rad。"""
        return self._measured_curve.length

    def length_at(self, u):
        return self._measured_curve.length_at(u)

    def u_at_length(self, s):
        return self._measured_curve.u_at_length(s)

    def curvature(self, u):
        """刀尖曲线的曲率，1/mm。刀路在曲率峰值处切 block、限速都只看刀尖。"""
        return self.tip.curvature(u)

    @property
    def start_point(self):
        return self(self.domain[0])

    @property
    def end_point(self):
        return self(self.domain[1])

    def restrict(self, a, b):
        """[a, b] 上的一段：两条分量分别取子段，参数保持原值。

        不能写成 SubCurve(pose)：SubCurve 会把 6 维一起当成一条曲线算弧长，结果错误却不报错。
        """
        return PoseCurve(self.tip.restrict(a, b), self.axis.restrict(a, b), self.along)

    def __repr__(self):
        return f"PoseCurve(tip={self.tip!r}, axis={self.axis!r}, along={self.along!r})"


# ---------- 由刀位点构造 ----------


def pose_spline(points, axes, parameters=None, degree=3, orientation_parameters=None):
    """过全部刀位点的五轴样条：刀尖、刀轴各插值一条 B 样条，刀轴再单位化。

    刀轴先当普通三维向量插值再单位化（UnitDirection），|o| ≡ 1 严格成立，各阶导数也能解析求出。
    默认刀轴与刀尖共用刀尖的弦长参数，刀位点之间刀轴转得快慢就被刀尖的弦长牵着走。给出
    orientation_parameters 时（例如 fitting.angle_parameters，Yuen et al. 2013），刀轴 ô(w) 按自己的
    参数拟合，再经严格递增的映射 g 同步回刀尖参数：o(u) = ô(g(u))，g 过节点 (uₖ, wₖ)、C² 连续
    （fitting.monotone_interpolate）。刀轴在刀位点处的取值不变，只改变点与点之间的转动分布。

    Args:
        points (N, 3): Tool tip points, mm.
        axes (N, 3): Tool axis vectors; normalized on input.
        parameters (N,): Tip parameters; defaults to cumulative chord length.
        degree (int): Spline degree.
        orientation_parameters (N,): Parameters for fitting the axis; defaults to the tip parameters.

    Returns:
        pose (PoseCurve): Tip and axis curves over the tip parameter domain.
    """
    u = chord_parameters(points) if parameters is None else np.asarray(parameters, dtype=float)
    tip = interpolate_bspline(points, u, degree)
    if orientation_parameters is None:
        return PoseCurve(tip, UnitDirection(interpolate_bspline(unit(axes), u, degree)))
    w = np.asarray(orientation_parameters, dtype=float)
    axis = UnitDirection(interpolate_bspline(unit(axes), w, degree))
    return PoseCurve(tip, Reparameterized(axis, monotone_interpolate(u, w)))


def dual_spline(points, axes, height, parameters=None, degree=3):
    """双样条刀路（Langeron et al. 2004）：刀尖点和沿刀轴偏移 height 的第二点各插值一条 B 样条。

    两条样条共用参数，刀轴取两者之差的方向（DualCurveDirection）。刀轴的变化由两条普通的三维样条
    决定，三轴样条的工具（拟合、节点插入、误差分析）对刀轴同样适用。

    Args:
        points (N, 3): Tool tip points, mm.
        axes (N, 3): Tool axis vectors; normalized on input.
        height (float): Distance from the tip to the second point along the axis, mm.
        parameters (N,): Shared parameters; defaults to cumulative chord length.
        degree (int): Spline degree.

    Returns:
        pose (PoseCurve): Tip curve and the direction of (top − tip).
    """
    u = chord_parameters(points) if parameters is None else parameters
    tip = interpolate_bspline(points, u, degree)
    top = interpolate_bspline(np.asarray(points, dtype=float) + height * unit(axes), u, degree)
    return PoseCurve(tip, DualCurveDirection(tip, top))


# ---------- 拐角过渡 ----------


def hermite_transition(start, end, h, order=2):
    """拐角过渡：两端直到 order 阶导数都与相邻段吻合的 Hermite 曲线（次数 2·order + 1）。

    端点条件取 dᵏC/duᵏ = hᵏ · dᵏC/dsᵏ（k = 0 … order），相当于接点处 ds/du = h、更高阶导数为零，
    所以接点处对弧长 order 阶连续。
    五轴时刀轴取单位化的 Hermite 曲线 o = r/|r|。端点处 r 与相邻段的单位刀轴直到 order 阶导数都相同，
    即 r = o_相邻 + O(u^{order+1})；单位化是光滑映射，又把单位向量映到自己，所以单位化后刀轴对刀尖
    弧长同样 order 阶连续（见 docs/数学约定.md）。

    Args:
        start (4, dim): Derivative stack by tip arc length at the start joint, from
            PolylinePath.corner_ends; dim is 3, or 6 for [p, o].
        end (4, dim): Derivative stack at the end joint.
        h (float): Parametric speed ds/du imposed at both ends, mm; usually the total trimmed length.
        order (int): Highest derivative order matched at the joints.

    Returns:
        curve (Bezier or PoseCurve): Transition over u ∈ [0, 1]; a PoseCurve when dim is 6.
    """
    scale = (h ** np.arange(order + 1))[:, None]
    start, end = start[: order + 1] * scale, end[: order + 1] * scale
    if start.shape[-1] != 6:
        return hermite(start, end)
    return PoseCurve(hermite(start[:, :3], end[:, :3]), UnitDirection(hermite(start[:, 3:], end[:, 3:])))
