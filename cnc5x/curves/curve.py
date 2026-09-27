"""曲线的抽象：Curve 基类，以及两种包装——取子段 SubCurve、换参数 Reparameterized。

具体的曲线（Line、Bezier、BSpline、NURBS）在 spline.py，弧长表在 arclength.py。
"""

from functools import cached_property

import numpy as np
from scipy.optimize import brentq

from ..utils import calculus, geometry, tolerances
from .arclength import ArcLengthTable


class Curve:
    """Base class of parametric curves C(u), u ∈ domain = (a, b).

    A subclass sets `domain` and implements `_derivatives(u, order)`, which returns the derivatives
    of orders 0 … order stacked into shape (order + 1, ..., dim); `u` arrives already checked.
    Evaluation, arc length and derivatives by arc length are all built on it here.

    Why a whole stack at once instead of one order per call: for NURBS, unit vectors and
    compositions, the k-th derivative needs every lower one. Asking for each order separately would
    recompute the lower ones again and again; one stack computes each quantity once. `order` lets
    callers that need only low orders (a point, the parametric speed) skip the high ones.

    Note:
        A curve is treated as immutable after construction: its arc-length table is built on first
        use and then cached.
    """

    def _derivatives(self, u, order):
        raise NotImplementedError

    # ---------- evaluation ----------
    def derivatives(self, u, order=3):
        """Derivative stack [C, C', C'', C'''] up to `order`.

        Args:
            u (...): Parameter values in `domain`.
            order (int): Highest derivative order, 0 to 3.

        Returns:
            d (order + 1, ..., dim): d[k] = dᵏC/duᵏ.
        """
        if order not in (0, 1, 2, 3):
            raise ValueError(f"order must be 0, 1, 2 or 3, got {order}")
        return self._derivatives(self._check(u), order)

    def __call__(self, u, order=0):
        """Point C(u) for order = 0, otherwise the derivative dᵏC/duᵏ with k = order; shape (..., dim)."""
        return self.derivatives(u, order)[order]

    @property
    def breaks(self):
        """Parameters where the curve may be non-smooth, both ends included, shape (m,).

        The curve is C^∞ between two breaks; at a break some derivative may jump. The arc-length
        table splits its quadrature here.
        """
        return np.array(self.domain)

    @property
    def start_point(self):
        return self(self.domain[0])

    @property
    def end_point(self):
        return self(self.domain[1])

    def curvature(self, u):
        """Curvature κ(u), 1/mm; shape of u."""
        d = self.derivatives(u, 2)
        return geometry.curvature(d[1], d[2])

    def restrict(self, a, b):
        """The piece over [a, b] ⊂ domain, keeping the original parameter (see SubCurve)."""
        return SubCurve(self, a, b)

    def sample(self, n=100, by_length=False):
        """n points equally spaced in u, or in arc length when by_length is True; shape (n, dim)."""
        if by_length:
            return self(self.u_at_length(np.linspace(0.0, self.length, n)))
        lo, hi = self.domain
        return self(np.linspace(lo, hi, n))

    # ---------- arc length ----------
    def parametric_speed(self, u):
        """Parametric speed σ(u) = |C'(u)| = ds/du, mm per unit of u; shape of u.

        A geometric quantity of the curve, not the feedrate ds/dt.
        """
        return np.linalg.norm(self(u, 1), axis=-1)

    def derivatives_by_length(self, u):
        """Derivative stack with respect to arc length, [C, C_s, C_ss, C_sss], at parameter u.

        From one stack over u: (s', s'', s''') by calculus.speed_derivatives, the inverse function
        derivatives (u_s, u_ss, u_sss) by calculus.inverse_derivatives, then the chain rule
        calculus.compose.

        Args:
            u (...): Parameter values in `domain`.

        Returns:
            d (4, ..., dim): d[k] = dᵏC/dsᵏ.
        """
        d = self.derivatives(u)
        return calculus.compose(d, *calculus.inverse_derivatives(*calculus.speed_derivatives(d)))

    @cached_property
    def arc(self):
        """Arc-length table (ArcLengthTable), built on first use."""
        return ArcLengthTable(self)

    @property
    def length(self):
        """Total arc length, mm."""
        return self.arc.total

    def length_at(self, u):
        """Arc length s(u) from the start to parameter u, mm; shape of u."""
        return self.arc.s(self._check(u))

    def u_at_length(self, s):
        """Parameter u(s) at arc length s ∈ [0, length], the inverse of length_at; shape of s."""
        return self.arc.u(self._check_length(s))

    def _check(self, u):
        return _clip_to(u, self.domain, "parameter")

    def _check_length(self, s):
        if self.length == 0:
            raise ValueError("curve has zero length, so arc length cannot parameterize it")
        return _clip_to(s, (0.0, self.length), "arc length")


def _clip_to(x, interval, name):
    """x as a float array clipped into [lo, hi]; allows a relative rounding error of tolerances.ROUNDING."""
    x = np.asarray(x, dtype=float)
    lo, hi = interval
    tol = tolerances.ROUNDING * max(1.0, hi - lo)
    if not np.all((x >= lo - tol) & (x <= hi + tol)):
        raise ValueError(f"{name} outside [{lo}, {hi}]")
    return np.clip(x, lo, hi)


class SubCurve(Curve):
    """The piece of a curve over [a, b], keeping the original parameter (e.g. to cut a curve into blocks).

    Create it through `curve.restrict(a, b)`. A piece of a piece points straight at the original
    curve, so wrappers never nest.

    Args:
        curve (Curve): The original curve.
        a, b (float): Parameter interval, domain[0] ≤ a < b ≤ domain[1].
    """

    def __init__(self, curve, a, b):
        if isinstance(curve, SubCurve):
            curve = curve.curve
        lo, hi = curve.domain
        if not lo <= a < b <= hi:
            raise ValueError(f"sub-interval [{a}, {b}] must lie inside the domain [{lo}, {hi}]")
        self.curve, self.domain = curve, (float(a), float(b))

    @property
    def breaks(self):
        a, b = self.domain
        inner = self.curve.breaks
        return np.unique(np.concatenate([[a], inner[(inner > a) & (inner < b)], [b]]))

    def _derivatives(self, u, order):
        return self.curve.derivatives(u, order)

    def parametric_speed(self, u):
        return self.curve.parametric_speed(self._check(u))  # 原曲线可能有更省的求法（例如 BSpline）


class Reparameterized(Curve):
    """Composite curve C(g(w)): g is a strictly increasing 1-D mapping curve, w the new parameter.

    Derivatives come from the third-order chain rule (calculus.compose). The preimages under g of
    the inner breaks of C join the breaks, otherwise the arc-length quadrature would cross
    non-smooth points; they are found by solving g(w) = b with brentq.

    Args:
        curve (Curve): The curve C.
        mapping (Curve): The mapping g(w), evaluating to shape (..., 1); its range must lie in
            curve.domain. Monotonicity is checked on samples only.
    """

    def __init__(self, curve, mapping):
        lo, hi = mapping.domain
        w = np.unique(np.concatenate([np.linspace(lo, hi, 201), mapping.breaks]))
        if np.any(mapping(w, 1)[..., 0] <= 0):
            raise ValueError("mapping g must be strictly increasing (checked on samples)")
        g_lo, g_hi = float(mapping(lo)[0]), float(mapping(hi)[0])
        _clip_to([g_lo, g_hi], curve.domain, "range of the mapping")
        preimages = []
        for b in curve.breaks[(curve.breaks > g_lo) & (curve.breaks < g_hi)]:
            preimages.append(brentq(lambda x: float(mapping(x)[0]) - b, lo, hi, xtol=tolerances.ROOT_XTOL))
        self.curve, self.mapping, self.domain = curve, mapping, mapping.domain
        self._breaks = np.unique(np.concatenate([mapping.breaks, preimages]))

    @property
    def breaks(self):
        return self._breaks

    def _derivatives(self, w, order):
        # compose 要求完整的四层导数栈，所以总是求满三阶再截取，order 在这里省不了计算
        g = self.mapping.derivatives(w)[..., 0]  # (4, ...)
        d = self.curve.derivatives(g[0])  # (4, ..., dim)
        return calculus.compose(d, g[1], g[2], g[3])[: order + 1]
