"""Tool axis curves on the unit sphere, and the five-axis pose curve that pairs one with a tip curve.

GreatCircle         great-circle arc between two directions (slerp)
UnitDirection       a 3-D vector curve normalized to unit length
DualCurveDirection  direction of the difference of two curves (dual-spline tool path)
SphericalCurve      a curve in spherical coordinates (θ, φ) mapped onto the sphere
PoseCurve           five-axis tool pose [p, o]: a tip curve and a tool axis curve sharing one parameter
"""

import numpy as np

from . import tolerances
from .calculus import compose, inverse_derivatives, product, speed_derivatives, unit_derivatives
from .curves import Curve
from .geometry import angle_between, unit


class GreatCircle(Curve):
    """Great-circle arc between two directions (spherical linear interpolation, slerp), u ∈ [0, 1].

        o(u)    = cos(uθ) o₀ + sin(uθ) t₀
        o⁽ᵏ⁾(u) = θᵏ [cos(uθ + kπ/2) o₀ + sin(uθ + kπ/2) t₀]

    θ = ∠(o₀, o₁), and t₀ is the unit tangent at o₀ pointing towards o₁. The arc length is analytic,
    |o'(u)| ≡ θ, so s(u) = θu and no arc-length table is built.

    Args:
        start (3,): Start direction o₀; normalized on input.
        end (3,): End direction o₁; must not be opposite to o₀ (the great circle would not be unique).
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
    """A 3-D vector curve r(u) normalized to unit length, o = r / |r|.

    The k-th derivative of o needs derivatives 0 … k of r and all lower derivatives of o
    (calculus.unit_derivatives), so the whole stack of r is always computed.

    Args:
        curve (Curve): The vector curve r(u); r = 0 anywhere raises on evaluation.
    """

    def __init__(self, curve):
        self.curve, self.domain = curve, curve.domain

    @property
    def breaks(self):
        return self.curve.breaks

    def _derivatives(self, u, order):
        return unit_derivatives(self.curve.derivatives(u))[: order + 1]


class DualCurveDirection(Curve):
    """Dual-spline tool axis (Langeron et al. 2004): o = (Q − C) / |Q − C|.

    Args:
        tip (Curve): Tool tip curve C(u).
        top (Curve): Curve Q(u) of a second point on the tool axis (e.g. above the tip), same parameter.
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
    """Tool axis in spherical coordinates (Yuen et al. 2013): o = (sinθ cosφ, sinθ sinφ, cosθ).

    |o| ≡ 1 holds by a trigonometric identity, so no normalization is needed. Writing the first two
    components as one complex number o_x + i·o_y = sinθ·e^{iφ}, every derivative needs only the
    scalar chain rule (compose) and Leibniz's rule (product).

    Args:
        angles (Curve): 2-D curve [θ(u), φ(u)], rad; θ is the angle from the z axis, φ the azimuth.
            At θ = 0 or π (the poles) φ is undefined; fitting must avoid them (see fitting.spherical_spline).
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
    """Derivative stack of a scalar composition f(x(u)): f holds the outer derivatives at x, x the stack of x(u)."""
    return compose(f[..., None], x[1], x[2], x[3])[..., 0]


class PoseCurve:
    """Five-axis tool pose: tool tip p(u) and unit tool axis o(u) sharing one parameter u.

    It is made of two curves rather than being one: the 6-D value [p, o] is not a point in space,
    its arc length comes from one component only, and its curvature is the tip's. The derivative
    stack is still concatenated to 6-D so that calculus.compose converts tip and axis together
    (CLAUDE.md §3.7). It provides the tool path segment interface (CLAUDE.md §3.6).

    Args:
        tip (Curve): Tool tip curve p(u), 3-D, mm.
        axis (Curve): Unit tool axis curve o(u), 3-D, pointing from the tip to the holder.
        along (str): Which component measures arc length (the feed): "tip" (mm, default), or
            "axis" (rad, the angle turned by the axis) for motion that only rotates the axis.
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
        """Derivative stack [x, x', x'', x'''] of x = [p, o] up to `order`, shape (order + 1, ..., 6)."""
        return np.concatenate([self.tip.derivatives(u, order), self.axis.derivatives(u, order)], axis=-1)

    def __call__(self, u, order=0):
        """Pose [p, o] for order = 0, otherwise its derivative of that order; shape (..., 6)."""
        return self.derivatives(u, order)[order]

    def derivatives_by_length(self, u):
        """Derivative stack of [p, o] with respect to the measured arc length s (see `along`), shape (4, ..., 6).

        The derivatives of s come from the measured component alone; both components are then
        converted with the same inverse derivatives (u_s, u_ss, u_sss).
        """
        d = self.derivatives(u)
        measured = d[..., :3] if self.along == "tip" else d[..., 3:]
        return compose(d, *inverse_derivatives(*speed_derivatives(measured)))

    @property
    def length(self):
        """Arc length of the measured component: mm along the tip, or rad along the axis."""
        return self._measured_curve.length

    def length_at(self, u):
        return self._measured_curve.length_at(u)

    def u_at_length(self, s):
        return self._measured_curve.u_at_length(s)

    def curvature(self, u):
        """Curvature of the tool tip curve, 1/mm."""
        return self.tip.curvature(u)

    @property
    def start_point(self):
        return self(self.domain[0])

    @property
    def end_point(self):
        return self(self.domain[1])

    def restrict(self, a, b):
        """The piece over [a, b]: each component is restricted separately, keeping the parameter."""
        return PoseCurve(self.tip.restrict(a, b), self.axis.restrict(a, b), self.along)

    def __repr__(self):
        return f"PoseCurve(tip={self.tip!r}, axis={self.axis!r}, along={self.along!r})"
