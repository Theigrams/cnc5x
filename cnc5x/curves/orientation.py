"""刀轴曲线（取值在单位球面上），以及把刀尖曲线和刀轴曲线配成一对的五轴刀位曲线。

GreatCircle         两方向之间的大圆弧（slerp）
UnitDirection       三维向量曲线单位化
DualCurveDirection  两条曲线之差的方向（双样条刀路）
SphericalCurve      球坐标 (θ, φ) 曲线映到球面
PoseCurve           五轴刀位 [p, o]：刀尖曲线与刀轴曲线共用参数
"""

import numpy as np

from ..utils import tolerances
from ..utils.calculus import compose, inverse_derivatives, product, speed_derivatives, unit_derivatives
from ..utils.geometry import angle_between, unit
from .curve import Curve


class GreatCircle(Curve):
    """两方向之间的大圆弧（球面线性插值 slerp），u ∈ [0, 1]：

        o(u)    = cos(uθ) o₀ + sin(uθ) t₀
        o⁽ᵏ⁾(u) = θᵏ [cos(uθ + kπ/2) o₀ + sin(uθ + kπ/2) t₀]

    θ = ∠(o₀, o₁)，t₀ 是 o₀ 处指向 o₁ 的单位切向。刀轴沿大圆匀速转动，是五轴 G01 段刀轴插值的常用做法。
    |o'(u)| ≡ θ，弧长有解析式 s = θu，所以不建弧长表。

    Args:
        start (3,): Start direction o₀; normalized on input.
        end (3,): End direction o₁; must not be opposite to o₀.

    Note:
        两方向相反时大圆不唯一，报错；两方向相同（θ = 0）时允许构造，但求弧长参数时报错。
    """

    domain = (0.0, 1.0)

    def __init__(self, start, end):
        self.start, self.end = unit(start), unit(end)
        if self.start.shape != (3,) or self.end.shape != (3,):
            raise ValueError("start and end must be 3-D directions")
        self.theta = float(angle_between(self.start, self.end))
        if np.pi - self.theta < tolerances.ANTIPODAL:
            raise ValueError("start and end directions are opposite, so the great-circle arc is not unique")
        perpendicular = self.end - np.dot(self.start, self.end) * self.start
        self.tangent = unit(perpendicular) if self.theta > 0 else np.zeros(3)

    @property
    def length(self):
        return self.theta

    def length_at(self, u):
        return self._check(u) * self.theta

    def u_at_length(self, s):
        return self._check_length(s) / self.theta

    def _derivatives(self, u, order):
        stack = []
        for k in range(order + 1):
            phase = u[..., None] * self.theta + k * np.pi / 2
            stack.append(self.theta**k * (np.cos(phase) * self.start + np.sin(phase) * self.tangent))
        return np.stack(stack)


class UnitDirection(Curve):
    """三维向量曲线 r(u) 单位化：o = r / |r|。

    刀轴拟合时先对刀轴向量插值一条普通样条，再单位化：任意阶导数都能解析求出，|o| ≡ 1 也严格成立。
    o 的 k 阶导数要用到 r 的 0..k 阶导数和 o 的全部低阶导数（calculus.unit_derivatives），所以总是
    一次求出 r 的整个导数栈。

    Args:
        curve (Curve): Vector curve r(u), 3-D; r = 0 anywhere raises on evaluation.
    """

    def __init__(self, curve):
        self.curve, self.domain = curve, curve.domain

    @property
    def breaks(self):
        return self.curve.breaks

    def _derivatives(self, u, order):
        return unit_derivatives(self.curve.derivatives(u))[: order + 1]


class DualCurveDirection(Curve):
    """双样条刀轴（Langeron et al. 2004）：o = (Q − C) / |Q − C|。

    刀尖点 C 和刀轴上另一点 Q 各拟合一条样条、共用参数，刀轴取两者之差的方向。这样刀轴的变化由两条
    三维曲线决定，三轴样条的全部工具都能直接用上。

    Args:
        tip (Curve): Tool tip curve C(u).
        top (Curve): Curve Q(u) of a second point on the tool axis, same parameter domain.
    """

    def __init__(self, tip, top):
        if tip.domain != top.domain:
            raise ValueError("tip and top curves must share the same parameter domain")
        self.tip, self.top = tip, top
        self.domain = tip.domain

    @property
    def breaks(self):
        return np.unique(np.concatenate([self.tip.breaks, self.top.breaks]))

    def _derivatives(self, u, order):
        return unit_derivatives(self.top.derivatives(u) - self.tip.derivatives(u))[: order + 1]


class SphericalCurve(Curve):
    """球坐标表示的刀轴曲线（Yuen et al. 2013）：o = (sinθ cosφ, sinθ sinφ, cosθ)。

    |o| ≡ 1 由三角恒等式保证，不必再单位化。把前两个分量写成一个复数 o_x + i·o_y = sinθ·e^{iφ}，
    各阶导数就只要标量函数的链式法则（compose）和乘积的 Leibniz 法则（product），不必对分量逐个展开。

    Args:
        angles (Curve): 2-D curve [θ(u), φ(u)], rad; θ is the angle from the z axis, φ the azimuth.

    Note:
        θ = 0 或 π（极点）处 φ 没有定义，拟合时要避开（见 fitting.spherical_spline）。
    """

    def __init__(self, angles):
        self.angles, self.domain = angles, angles.domain

    @property
    def breaks(self):
        return self.angles.breaks

    def _derivatives(self, u, order):
        a = self.angles.derivatives(u)
        theta, phi = a[..., 0], a[..., 1]  # 各为 (4, ...) 的导数栈
        t, e = theta[0], np.exp(1j * phi[0])
        # 外函数的 0..3 阶导数：sin → cos → −sin → −cos；cos → −sin → −cos → sin；e^{iφ} 每阶乘 i
        sin_theta = _outer(np.stack([np.sin(t), np.cos(t), -np.sin(t), -np.cos(t)]), theta)
        cos_theta = _outer(np.stack([np.cos(t), -np.sin(t), -np.cos(t), np.sin(t)]), theta)
        exp_phi = _outer(np.stack([e, 1j * e, -e, -1j * e]), phi)
        xy = product(sin_theta, exp_phi)
        return np.stack([xy.real, xy.imag, cos_theta], axis=-1)[: order + 1]


def _outer(f, x):
    """标量复合函数 f(x(u)) 的导数栈：f 为外函数在 x 处的 0..3 阶导数，x 为 x(u) 的导数栈。"""
    return compose(f[..., None], x[1], x[2], x[3])[..., 0]


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
