"""参数曲线：Line、Bezier、BSpline、NURBS，以及弧长表。

所有曲线共用一套接口（见 Curve）：
    curve(u)              曲线上的点；u 可为标量或数组，返回 (..., dim)
    curve(u, k)           第 k 阶导数 dᵏC/duᵏ（k ≤ 3）
    curve.derivatives(u)  导数栈 [C, C', C'', C''']，形状 (4, ..., dim)
    curve.length          弧长；curve.u_at_length(s) 是弧长的反函数
"""

from functools import cached_property
from math import comb

import numpy as np
from scipy import interpolate as si

from . import geometry
from .calculus import compose, speed_derivatives


class Curve:
    """参数曲线基类。子类设定 domain，并实现 _derivative(u, order)（传入的 u 已检查过）。"""

    domain = (0.0, 1.0)

    def _derivative(self, u, order):
        raise NotImplementedError

    # ---------- 求值 ----------
    def derivative(self, u, order=1):
        """第 order 阶导数 dᵏC/duᵏ；order = 0 就是曲线上的点。"""
        if order not in (0, 1, 2, 3):
            raise ValueError("order 只能是 0、1、2、3")
        return self._derivative(self._check(u), order)

    def __call__(self, u, order=0):
        return self.derivative(u, order)

    def derivatives(self, u):
        """导数栈 [C, C', C'', C''']，形状 (4, ..., dim)。"""
        u = self._check(u)
        return np.stack([self._derivative(u, k) for k in range(4)])

    @property
    def breaks(self):
        """参数域内的节点（含两端）。节点处高阶导数可能跳变，弧长积分在这里分段。"""
        return np.array(self.domain)

    @property
    def start_point(self):
        return self.derivative(self.domain[0], 0)

    @property
    def end_point(self):
        return self.derivative(self.domain[1], 0)

    def curvature(self, u):
        """曲率 κ(u)。"""
        return geometry.curvature(self.derivative(u, 1), self.derivative(u, 2))

    # ---------- 弧长 ----------
    def speed(self, u):
        """ds/du = |C'(u)|。"""
        return np.linalg.norm(self.derivative(u, 1), axis=-1)

    def arc_derivatives(self, u):
        """弧长 s(u) 的 1..3 阶导数 (s', s'', s''')。"""
        return speed_derivatives(self.derivatives(u))

    @cached_property
    def arc(self):
        """弧长表，第一次用到时才建立。"""
        return ArcLengthTable(self)

    @property
    def length(self):
        return self.arc.total

    def length_at(self, u):
        """从起点到参数 u 的弧长 s(u)。"""
        return self.arc.s(self._check(u))

    def u_at_length(self, s):
        """弧长 s 处的参数 u(s)。"""
        return self.arc.u(s)

    def sample(self, n=100, by_length=False):
        """取 n 个点：默认参数等分；by_length=True 时弧长等分。"""
        if by_length:
            return self(self.u_at_length(np.linspace(0.0, self.length, n)))
        lo, hi = self.domain
        return self(np.linspace(lo, hi, n))

    def _check(self, u):
        """转成浮点数组；允许超出参数域 1e-9（相对）以内的舍入误差，并截回域内。"""
        u = np.asarray(u, dtype=float)
        lo, hi = self.domain
        tol = 1e-9 * max(1.0, hi - lo)
        if not np.all((u >= lo - tol) & (u <= hi + tol)):
            raise ValueError(f"参数超出定义域 [{lo}, {hi}]")
        return np.clip(u, lo, hi)


class ArcLengthTable:
    """弧长表：s(u) 与反函数 u(s)。

    参数域先在 breaks 处分开（保证每一小段内曲线光滑），再均分成约 pieces 段；
    每段弧长用 8 点 Gauss–Legendre 积分。节点之间用三次 Hermite 插值：
    s(u) 在节点处的斜率取 |C'(u)|，u(s) 的斜率取 1/|C'(u)|。
    """

    def __init__(self, curve, pieces=256):
        lo, hi = curve.domain
        grid = np.unique(np.concatenate([np.linspace(lo, hi, pieces + 1), curve.breaks]))
        x, w = np.polynomial.legendre.leggauss(8)
        a, b = grid[:-1], grid[1:]
        nodes = (a + b)[:, None] / 2 + (b - a)[:, None] / 2 * x  # (段数, 8)
        piece_lengths = (b - a) / 2 * (curve.speed(nodes) @ w)
        speed = curve.speed(grid)
        if np.any(speed <= 0):
            raise ValueError("曲线上有速度为零的点，不能用弧长作参数")
        self.u_grid = grid
        self.s_grid = np.concatenate([[0.0], np.cumsum(piece_lengths)])
        self.total = float(self.s_grid[-1])
        self._s_of_u = si.CubicHermiteSpline(grid, self.s_grid, speed)
        self._u_of_s = si.CubicHermiteSpline(self.s_grid, grid, 1 / speed)

    def s(self, u):
        return self._s_of_u(u)

    def u(self, s):
        return self._u_of_s(np.clip(s, 0.0, self.total))


class Line(Curve):
    """直线段 C(u) = P₀ + u (P₁ − P₀)，u ∈ [0, 1]。"""

    def __init__(self, start, end):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        if self.start.ndim != 1 or self.start.shape != self.end.shape:
            raise ValueError("起点和终点必须是维数相同的向量")

    def _derivative(self, u, order):
        delta = self.end - self.start
        if order == 0:
            return self.start + u[..., None] * delta
        value = delta if order == 1 else np.zeros_like(delta)
        return np.broadcast_to(value, u.shape + value.shape).copy()

    @property
    def length(self):
        return float(np.linalg.norm(self.end - self.start))

    def length_at(self, u):
        return self._check(u) * self.length

    def u_at_length(self, s):
        return np.asarray(s, dtype=float) / self.length

    def __repr__(self):
        return f"Line({self.start.tolist()} -> {self.end.tolist()})"


class BSpline(Curve):
    """B 样条曲线（数值计算交给 scipy.interpolate.BSpline）。

    control_points (n, dim)、次数 p、knots (n + p + 1,)；不给 knots 时用均匀夹持节点。
    参数域为 [knots[p], knots[n]]，不做归一化。内部节点处导数取右极限，终点取左极限。
    """

    def __init__(self, control_points, degree=3, knots=None):
        P = np.asarray(control_points, dtype=float)
        n = len(P)
        if P.ndim != 2 or not 1 <= degree < n:
            raise ValueError("control_points 应为 (n, dim)，且 1 ≤ degree < n")
        if knots is None:
            knots = np.concatenate([np.zeros(degree), np.linspace(0.0, 1.0, n - degree + 1), np.ones(degree)])
        knots = np.asarray(knots, dtype=float)
        if len(knots) != n + degree + 1 or np.any(np.diff(knots) < 0):
            raise ValueError(f"需要 {n + degree + 1} 个非降的节点")
        if not knots[degree] < knots[n]:
            raise ValueError("参数域为空")
        self.control_points, self.degree, self.knots = P, degree, knots
        self.domain = (float(knots[degree]), float(knots[n]))
        self._spline = si.BSpline(knots, P, degree, extrapolate=False)

    @property
    def breaks(self):
        lo, hi = self.domain
        return np.unique(self.knots[(self.knots >= lo) & (self.knots <= hi)])

    def _derivative(self, u, order):
        if order > self.degree:
            return np.zeros(u.shape + self.control_points.shape[1:])
        return self._spline(u, nu=order)

    def split(self, u):
        """在参数 u 处分成两条 B 样条：插入节点使 u 的重数达到 p（The NURBS Book §5.2）。"""
        lo, hi = self.domain
        if not lo < u < hi:
            raise ValueError("分割点必须在参数域内部")
        p = self.degree
        spline = self._spline
        insertions = p - np.count_nonzero(self.knots == u)
        if insertions > 0:
            spline = si.insert(u, spline, insertions)
        t = spline.t
        c = spline.c[: len(t) - p - 1]
        j = np.searchsorted(t, u, side="left")  # u 第一次出现的位置
        left = BSpline(c[:j], p, np.concatenate([t[: j + p], [u]]))
        right = BSpline(c[j - 1 :], p, np.concatenate([[u], t[j:]]))
        return left, right

    def __repr__(self):
        return f"BSpline(degree={self.degree}, {len(self.control_points)} control points, domain={self.domain})"


class Bezier(BSpline):
    """Bézier 曲线：节点为 [0]*(p+1) + [1]*(p+1) 的 B 样条，u ∈ [0, 1]。"""

    def __init__(self, control_points):
        degree = len(control_points) - 1
        knots = np.concatenate([np.zeros(degree + 1), np.ones(degree + 1)])
        super().__init__(control_points, degree, knots)

    def split(self, u):
        """de Casteljau 算法：在 u 处一分为二。"""
        if not 0 < u < 1:
            raise ValueError("分割点必须在 (0, 1) 内")
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
    """NURBS 曲线：齐次坐标下的 B 样条 H(u) = Σ Nᵢ(u) [wᵢPᵢ, wᵢ]，C = H 的前 dim 维 / w。

    记 A = wC，由 A⁽ᵏ⁾ 的 Leibniz 展开递推（The NURBS Book §4.3）：
        C⁽ᵏ⁾ = (A⁽ᵏ⁾ − Σᵢ₌₁ᵏ C(k, i) w⁽ⁱ⁾ C⁽ᵏ⁻ⁱ⁾) / w
    """

    def __init__(self, control_points, degree=3, knots=None, weights=None):
        P = np.asarray(control_points, dtype=float)
        w = np.ones(len(P)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (len(P),) or np.any(w <= 0):
            raise ValueError("每个控制点需要一个正权重")
        self.control_points, self.weights = P, w
        self.homogeneous = BSpline(np.column_stack([P * w[:, None], w]), degree, knots)
        self.degree, self.knots = degree, self.homogeneous.knots
        self.domain = self.homogeneous.domain

    @property
    def breaks(self):
        return self.homogeneous.breaks

    def _derivative(self, u, order):
        H = [self.homogeneous.derivative(u, k) for k in range(order + 1)]
        A = [h[..., :-1] for h in H]
        w = [h[..., -1:] for h in H]
        C = []
        for k in range(order + 1):
            value = A[k]
            for i in range(1, k + 1):
                value = value - comb(k, i) * w[i] * C[k - i]
            C.append(value / w[0])
        return C[order]

    def __repr__(self):
        return f"NURBS(degree={self.degree}, {len(self.control_points)} control points, domain={self.domain})"


class SubCurve(Curve):
    """曲线在 [a, b] 上的一段，沿用原来的参数（用于在临界点处把曲线切成 block）。"""

    def __init__(self, curve, a, b):
        lo, hi = curve.domain
        if not lo <= a < b <= hi:
            raise ValueError("子区间必须落在原参数域内")
        self.curve, self.domain = curve, (float(a), float(b))

    @property
    def breaks(self):
        a, b = self.domain
        inner = self.curve.breaks
        return np.unique(np.concatenate([[a], inner[(inner > a) & (inner < b)], [b]]))

    def _derivative(self, u, order):
        return self.curve.derivative(u, order)

    def speed(self, u):
        return self.curve.speed(u)

    def arc_derivatives(self, u):
        return self.curve.arc_derivatives(u)

    def curvature(self, u):
        return self.curve.curvature(u)


class Reparameterized(Curve):
    """复合曲线 C(g(w))：g 是一维映射曲线（求值形状 (..., 1)），w 是新参数。

    导数由三阶链式法则得到。C 若有内部节点，要把它们在 w 下的原像由 breaks 传入，
    否则弧长积分会跨过不光滑的点。
    """

    def __init__(self, curve, mapping, breaks=()):
        self.curve, self.mapping = curve, mapping
        self.domain = mapping.domain
        self._breaks = np.unique(np.concatenate([mapping.breaks, np.asarray(breaks, dtype=float)]))

    @property
    def breaks(self):
        return self._breaks

    def _derivative(self, w, order):
        g = self.mapping.derivatives(w)[..., 0]  # (4, ...)
        d = self.curve.derivatives(g[0])  # (4, ..., dim)
        return compose(d, g[1], g[2], g[3])[order]
