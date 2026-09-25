"""刀轴方向曲线：取值在单位球面上的曲线（五轴刀位的方向部分）。

GreatCircle         两方向之间的大圆弧（slerp）
UnitDirection       任意三维向量曲线单位化
DualCurveDirection  两条曲线之差的方向（双样条刀路）
SphericalCurve      球坐标 (θ, φ) 曲线映到球面
"""

import numpy as np

from . import tolerances
from .calculus import compose, product, unit_derivatives
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
        if np.pi - self.theta < tolerances.ANTIPODAL:
            raise ValueError("两方向相反，大圆弧不唯一")
        perpendicular = self.end - np.dot(self.start, self.end) * self.start
        self.tangent = unit(perpendicular) if self.theta > 0 else np.zeros(3)

    # 弧长有解析式：|o'(u)| ≡ θ，所以 s(u) = θu，不必建弧长表
    @property
    def length(self):
        return self.theta

    def length_at(self, u):
        return self._check(u) * self.theta

    def u_at_length(self, s):
        return np.asarray(s, dtype=float) / self.theta

    def _derivatives(self, u, order):
        stack = []
        for k in range(order + 1):
            phase = u[..., None] * self.theta + k * np.pi / 2
            stack.append(self.theta**k * (np.cos(phase) * self.start + np.sin(phase) * self.tangent))
        return np.stack(stack)


class UnitDirection(Curve):
    """把三维向量曲线 r(u) 单位化：o = r / |r|。r 在某处为零时报错。

    o 的 k 阶导数要用到 r 的 0..k 阶导数和 o 的全部低阶导数（calculus.unit_derivatives），
    所以总是一次求出 r 的整个导数栈。
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

    def _derivatives(self, u, order):
        return unit_derivatives(self.top.derivatives(u) - self.tip.derivatives(u))[: order + 1]


class SphericalCurve(Curve):
    """球坐标表示的刀轴曲线（Yuen et al. 2013）：o = (sinθ cosφ, sinθ sinφ, cosθ)。

    angles(u) = [θ(u), φ(u)] 是一条二维曲线（θ 为刀轴与 z 轴的夹角，φ 为方位角，单位 rad）。
    |o| ≡ 1 由三角恒等式保证，不必再单位化。把前两个分量写成复数 o_x + i·o_y = sinθ·e^{iφ}，
    各阶导数就只要标量函数的链式法则（compose）和乘积的 Leibniz 法则（product）。
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
