"""Parametric curves: Line, BSpline, Bezier, NURBS, SubCurve, Reparameterized, and the arc-length table.

All curves share one interface (see Curve):
    curve(u), curve(u, k)           point, or the k-th derivative dᵏC/duᵏ (k ≤ 3); shape (..., dim)
    curve.derivatives(u)            derivative stack [C, C', C'', C'''], shape (4, ..., dim)
    curve.derivatives_by_length(u)  the same stack taken with respect to arc length s
    curve.length, curve.length_at(u), curve.u_at_length(s)
    curve.restrict(a, b)            the piece over [a, b]
"""

from functools import cached_property
from math import comb

import numpy as np
from scipy import interpolate as si
from scipy.optimize import brentq

from . import calculus, geometry, tolerances


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


class Line(Curve):
    """Line segment C(u) = P₀ + u (P₁ − P₀), u ∈ [0, 1].

    The arc length is analytic, s = u·|P₁ − P₀|, so no arc-length table is built.

    Args:
        start (dim,): Start point P₀.
        end (dim,): End point P₁.
    """

    domain = (0.0, 1.0)

    def __init__(self, start, end):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        if self.start.ndim != 1 or self.start.shape != self.end.shape:
            raise ValueError("start and end must be vectors of the same dimension")

    def _derivatives(self, u, order):
        delta = self.end - self.start
        point = self.start + u[..., None] * delta
        slope = np.broadcast_to(delta, point.shape)
        zero = np.zeros_like(point)
        return np.stack([point, slope, zero, zero][: order + 1])

    @property
    def length(self):
        return float(np.linalg.norm(self.end - self.start))

    def length_at(self, u):
        return self._check(u) * self.length

    def u_at_length(self, s):
        return self._check_length(s) / self.length

    def __repr__(self):
        return f"Line({self.start.tolist()} -> {self.end.tolist()})"


class BSpline(Curve):
    """B-spline curve C(u) = Σᵢ Nᵢ,ₚ(u) Pᵢ, evaluated by scipy.interpolate.BSpline.

    The domain is [knots[p], knots[n]] as given, not normalized. At an interior knot the
    derivatives are right limits; at the end of the domain, left limits.

    Args:
        control_points (n, dim): Control points Pᵢ.
        degree (int): Degree p, 1 ≤ p < n.
        knots (n + p + 1,): Non-decreasing knot vector; defaults to the uniform clamped one on [0, 1].
    """

    def __init__(self, control_points, degree=3, knots=None):
        P = np.asarray(control_points, dtype=float)
        n = len(P)
        if P.ndim != 2 or not 1 <= degree < n:
            raise ValueError(f"control_points must have shape (n, dim) with 1 <= degree < n, got {P.shape}")
        if knots is None:
            knots = np.concatenate([np.zeros(degree), np.linspace(0.0, 1.0, n - degree + 1), np.ones(degree)])
        knots = np.asarray(knots, dtype=float)
        if len(knots) != n + degree + 1 or np.any(np.diff(knots) < 0):
            raise ValueError(f"need {n + degree + 1} non-decreasing knots, got {len(knots)}")
        if not knots[degree] < knots[n]:
            raise ValueError("empty parameter domain: knots[degree] must be less than knots[n]")
        self.control_points, self.degree, self.knots = P, degree, knots
        self.domain = (float(knots[degree]), float(knots[n]))
        self._spline = si.BSpline(knots, P, degree, extrapolate=False)

    @property
    def breaks(self):
        lo, hi = self.domain
        return np.unique(self.knots[(self.knots >= lo) & (self.knots <= hi)])

    def parametric_speed(self, u):
        # 弧长表只要 |C'|：直接求一阶导数，不必像基类那样连带求出曲线上的点
        return np.linalg.norm(self._spline(self._check(u), nu=1), axis=-1)

    def _derivatives(self, u, order):
        zero = np.zeros(u.shape + self.control_points.shape[1:])
        return np.stack([self._spline(u, nu=k) if k <= self.degree else zero for k in range(order + 1)])

    def split(self, u):
        """Split into two B-splines at u by knot insertion up to multiplicity p (The NURBS Book §5.2).

        Knot insertion (Boehm's algorithm) is left to scipy.interpolate.insert for the same reason
        as evaluation: it is a numerical kernel that carries no idea of any paper. Bezier.split, by
        contrast, writes de Casteljau by hand, because those few lines are the principle itself.

        Args:
            u (float): Split parameter, strictly inside the domain.

        Returns:
            (left, right) (BSpline, BSpline): Pieces over [lo, u] and [u, hi], same parameter.
        """
        lo, hi = self.domain
        if not lo < u < hi:
            raise ValueError(f"split parameter must lie strictly inside ({lo}, {hi}), got {u}")
        p = self.degree
        spline = self._spline
        insertions = p - np.count_nonzero(self.knots == u)
        if insertions > 0:
            spline = si.insert(u, spline, insertions)
        t = spline.t
        c = spline.c[: len(t) - p - 1]  # scipy 把系数补零到与节点等长，有效的只有 len(t) − p − 1 个
        # u 的重数为 p 时，曲线在 u 处恰好经过控制点 c[j−1]（j 为 u 在节点里第一次出现的位置）。
        # 左段取 c[0..j−1]，节点末尾再补一个 u 成为夹持端；右段取 c[j−1..]，节点开头补一个 u。
        j = np.searchsorted(t, u, side="left")
        left = BSpline(c[:j], p, np.concatenate([t[: j + p], [u]]))
        right = BSpline(c[j - 1 :], p, np.concatenate([[u], t[j:]]))
        return left, right

    def __repr__(self):
        return f"BSpline(degree={self.degree}, {len(self.control_points)} control points, domain={self.domain})"


class Bezier(BSpline):
    """Bézier curve: the B-spline with knots [0]·(p+1) + [1]·(p+1), u ∈ [0, 1].

    Args:
        control_points (p + 1, dim): Control points; the degree p is their count minus one.
    """

    def __init__(self, control_points):
        degree = len(control_points) - 1
        knots = np.concatenate([np.zeros(degree + 1), np.ones(degree + 1)])
        super().__init__(control_points, degree, knots)

    def split(self, u):
        """Split into two Bézier curves at u by de Casteljau's algorithm; each keeps u ∈ [0, 1]."""
        if not 0 < u < 1:
            raise ValueError(f"split parameter must lie strictly inside (0, 1), got {u}")
        points = self.control_points
        left, right = [points[0]], [points[-1]]
        while len(points) > 1:
            points = (1 - u) * points[:-1] + u * points[1:]
            left.append(points[0])
            right.append(points[-1])
        return Bezier(left), Bezier(right[::-1])

    def __repr__(self):
        return f"Bezier(degree={self.degree}, control points={self.control_points.tolist()})"


class NURBS(Curve):
    """NURBS curve: a B-spline in homogeneous coordinates H(u) = Σᵢ Nᵢ,ₚ(u) [wᵢPᵢ, wᵢ], C = A / w.

    With A = wC (the first dim components of H), Leibniz's rule on A = wC gives the recurrence
    (The NURBS Book §4.3)

        C⁽ᵏ⁾ = (A⁽ᵏ⁾ − Σᵢ₌₁ᵏ C(k, i) w⁽ⁱ⁾ C⁽ᵏ⁻ⁱ⁾) / w.

    The k-th derivative needs all lower ones, so the whole stack is computed in one recurrence.

    Args:
        control_points (n, dim): Control points Pᵢ.
        degree (int): Degree p, 1 ≤ p < n.
        knots (n + p + 1,): Knot vector; defaults as in BSpline.
        weights (n,): Positive weights wᵢ; defaults to all ones.
    """

    def __init__(self, control_points, degree=3, knots=None, weights=None):
        P = np.asarray(control_points, dtype=float)
        w = np.ones(len(P)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (len(P),) or np.any(w <= 0):
            raise ValueError("need one positive weight per control point")
        self.control_points, self.weights = P, w
        self.homogeneous = BSpline(np.column_stack([P * w[:, None], w]), degree, knots)
        self.degree, self.knots = degree, self.homogeneous.knots
        self.domain = self.homogeneous.domain

    @property
    def breaks(self):
        return self.homogeneous.breaks

    def _derivatives(self, u, order):
        H = self.homogeneous.derivatives(u, order)
        A, w = H[..., :-1], H[..., -1:]
        C = []
        for k in range(order + 1):
            value = A[k]
            for i in range(1, k + 1):
                value = value - comb(k, i) * w[i] * C[k - i]
            C.append(value / w[0])
        return np.stack(C)

    def __repr__(self):
        return f"NURBS(degree={self.degree}, {len(self.control_points)} control points, domain={self.domain})"


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


# ---------- 弧长表 ----------


class ArcLengthTable:
    """Arc-length table: s(u) and its inverse u(s), error-controlled by adaptive bisection.

    The domain is cut into pieces. The arc length of each piece comes from Gauss–Legendre
    quadrature; inside a piece both s(u) and u(s) are cubic Hermite interpolants, with slopes
    σ = |C'(u)| and 1/σ. A piece is accepted when its quadrature error and the interpolation errors
    in both directions (see _piece_errors) are within `tolerance`; otherwise it is halved.

    The initial grid first splits at curve.breaks (quadrature across a non-smooth point converges
    slowly), then cuts every knot span into `initial_pieces` equal parts. Checking the interpolation
    only at the midpoint misses a piece that is symmetric about its midpoint (its error vanishes
    there); after the first cut the pieces are generally no longer symmetric.

    Args:
        curve (Curve): The curve; its parametric speed must be nonzero everywhere.
        tolerance (float): Error allowed per piece, in units of length.
        initial_pieces (int): Equal parts per knot span in the initial grid.

    Attributes:
        u_grid (m + 1,): Piece ends in u.
        s_grid (m + 1,): Arc length at u_grid, starting from 0.
        total (float): Total arc length.
    """

    def __init__(self, curve, tolerance=tolerances.ARC_LENGTH, initial_pieces=4):
        self.u_grid, self.s_grid, speed = _adaptive_grid(curve, tolerance, initial_pieces)
        self.total = float(self.s_grid[-1])
        # 建好的表用 SciPy 的 CubicHermiteSpline；_piece_errors 里另写 _hermite，是为了对几千个小段一次向量化检查
        self._s_of_u = si.CubicHermiteSpline(self.u_grid, self.s_grid, speed)
        self._u_of_s = si.CubicHermiteSpline(self.s_grid, self.u_grid, 1 / speed)

    def s(self, u):
        """Arc length s(u); u must already lie in the curve domain (Curve.length_at checks it)."""
        return self._s_of_u(u)

    def u(self, s):
        """Parameter u(s); s must already lie in [0, total] (Curve.u_at_length checks it)."""
        return self._u_of_s(s)


def _adaptive_grid(curve, tolerance, initial_pieces):
    """Bisect pieces until each is within tolerance; returns (u_grid, s_grid, parametric speed at u_grid)."""
    breaks = curve.breaks
    starts = (breaks[:-1, None] + np.diff(breaks)[:, None] * np.arange(initial_pieces) / initial_pieces).ravel()
    ends = np.append(starts[1:], breaks[-1])
    done_starts, done_lengths, done_speeds = [], [], []
    for _ in range(tolerances.ARC_LENGTH_MAX_DEPTH):
        length, error, speed = _piece_errors(curve, starts, ends)
        good = error <= tolerance
        done_starts.append(starts[good])
        done_lengths.append(length[good])
        done_speeds.append(speed[good])
        middles = (starts[~good] + ends[~good]) / 2
        starts = np.concatenate([starts[~good], middles])
        ends = np.concatenate([middles, ends[~good]])
        if len(starts) == 0:
            break
    else:
        raise ValueError(
            f"arc-length table did not converge after {tolerances.ARC_LENGTH_MAX_DEPTH} bisections: "
            "the curve probably has a point of near-zero parametric speed"
        )
    starts = np.concatenate(done_starts)
    by_u = np.argsort(starts)
    u_grid = np.append(starts[by_u], breaks[-1])
    s_grid = np.concatenate([[0.0], np.cumsum(np.concatenate(done_lengths)[by_u])])
    speed = np.append(np.concatenate(done_speeds)[by_u], curve.parametric_speed(breaks[-1]))
    return u_grid, s_grid, speed


# 8 点 Gauss–Legendre 求积：节点 X 与权重 W（区间 [−1, 1]），对 15 次以内的多项式精确。
_X, _W = np.polynomial.legendre.leggauss(8)


def _piece_errors(curve, a, b):
    """Arc length and error estimate of each piece [a, b]; a, b have shape (n,).

    The length is the sum of Gauss quadratures over the two halves. The error, in units of length,
    is the largest of:
        quadrature error: whole-piece quadrature minus the sum of the halves;
        error of s(u): Hermite value at the parameter midpoint m minus the true s(m);
        error of u(s): Hermite parameter at s(m) minus m, times σ(m) to convert it to length.

    Returns:
        (length, error, speed) (n,) each: speed is σ at the piece starts a.
    """
    m = (a + b) / 2
    lo, hi = np.stack([a, a, m]), np.stack([b, m, b])  # (3, n)：整段、左半、右半
    nodes = (lo + hi)[..., None] / 2 + (hi - lo)[..., None] / 2 * _X  # (3, n, 8)
    # 所有 Gauss 节点与三个端点的速度一次求出：每次调用 SciPy 都有固定开销，
    # 拐角光顺的刀路要为几千条短过渡曲线各建一张表，调用次数比点数更要紧。
    speed = curve.parametric_speed(np.concatenate([nodes.ravel(), a, m, b]))
    whole, left, right = (hi - lo) / 2 * (speed[: nodes.size].reshape(nodes.shape) @ _W)
    sa, sm, sb = speed[nodes.size :].reshape(3, -1)
    if np.any(sa == 0) or np.any(sm == 0) or np.any(sb == 0):
        raise ValueError("the curve has a point of zero parametric speed, so arc length cannot parameterize it")
    length = left + right
    s_mid = _hermite(0.5, 0.0, length, sa, sb, b - a)
    u_mid = _hermite(left / length, a, b, 1 / sa, 1 / sb, length)
    error = np.maximum(np.abs(whole - length), np.abs(s_mid - left))
    return length, np.maximum(error, sm * np.abs(u_mid - m)), sa


def _hermite(t, y0, y1, dy0, dy1, h):
    """Cubic Hermite interpolant at relative position t ∈ [0, 1] of an interval of length h.

    y0, y1 are the end values and dy0, dy1 the end slopes.
    """
    return (
        (1 + 2 * t) * (1 - t) ** 2 * y0
        + t**2 * (3 - 2 * t) * y1
        + h * t * (1 - t) ** 2 * dy0
        + h * t**2 * (t - 1) * dy1
    )
