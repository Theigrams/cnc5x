"""刀轴方向曲线：取值在单位球面上的曲线（五轴刀位的方向部分）。"""

import numpy as np

from .calculus import unit_derivatives
from .curves import Curve
from .geometry import angle_between, unit


class GreatCircle(Curve):
    """两方向之间的大圆弧（球面线性插值 slerp），u ∈ [0, 1]：

        o(u)   = cos(uθ) o₀ + sin(uθ) t₀
        o⁽ᵏ⁾(u) = θᵏ [cos(uθ + kπ/2) o₀ + sin(uθ + kπ/2) t₀]

    θ = ∠(o₀, o₁)，t₀ 是 o₀ 处指向 o₁ 的单位切向。两方向相反时大圆不唯一，报错。
    """

    def __init__(self, start, end):
        self.start, self.end = unit(start), unit(end)
        if self.start.shape != (3,) or self.end.shape != (3,):
            raise ValueError("需要三维方向")
        self.theta = float(angle_between(self.start, self.end))
        if np.isclose(self.theta, np.pi):
            raise ValueError("两方向相反，大圆弧不唯一")
        perpendicular = self.end - np.dot(self.start, self.end) * self.start
        self.tangent = unit(perpendicular) if self.theta > 0 else np.zeros(3)

    def _derivative(self, u, order):
        phase = u[..., None] * self.theta + order * np.pi / 2
        return self.theta**order * (np.cos(phase) * self.start + np.sin(phase) * self.tangent)


class UnitDirection(Curve):
    """把三维向量曲线 r(u) 单位化：o = r / |r|。r 在某处为零时报错。"""

    def __init__(self, curve):
        self.curve, self.domain = curve, curve.domain

    @property
    def breaks(self):
        return self.curve.breaks

    def _derivative(self, u, order):
        return unit_derivatives(self.curve.derivatives(u))[order]


class DualCurveDirection(Curve):
    """双曲线刀轴（Langeron et al. 2004）：o = (Q − C) / |Q − C|。

    C(u) 为刀尖曲线，Q(u) 为刀轴上另一点（如刀具上方某点）的曲线，两者共用参数。
    """

    def __init__(self, tip, top):
        if tip.domain != top.domain:
            raise ValueError("两条曲线的参数域必须相同")
        self.tip, self.top = tip, top
        self.domain = tip.domain

    @property
    def breaks(self):
        return np.unique(np.concatenate([self.tip.breaks, self.top.breaks]))

    def _derivative(self, u, order):
        return unit_derivatives(self.top.derivatives(u) - self.tip.derivatives(u))[order]
