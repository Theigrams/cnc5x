"""具体的参数曲线：Line、Bezier、BSpline、NURBS。

Line 就是一次 Bézier 曲线，单独写成一类是因为它的弧长有解析式，不必建弧长表。
"""

from math import comb

import numpy as np
from scipy import interpolate as si

from .curve import Curve


class Line(Curve):
    """直线段 C(u) = P₀ + u (P₁ − P₀)，u ∈ [0, 1]。

    弧长有解析式 s = u·|P₁ − P₀|，所以覆盖了 length、length_at、u_at_length，不建弧长表。零长度的
    线段允许构造（五轴里刀尖不动、只转刀轴），但求弧长参数时报错。

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
    """B 样条曲线 C(u) = Σᵢ Nᵢ,ₚ(u) Pᵢ。

    求值交给 scipy.interpolate.BSpline：基函数求值是数值内核，自己重写不增加理解（CLAUDE.md 第 5 节）。
    参数域保持 [knots[p], knots[n]]，不归一化到 [0, 1]，这样切开、拼接、重参数化之后参数都能对上。
    内部节点处导数取右极限，参数域末端取左极限。

    Args:
        control_points (n, dim): Control points Pᵢ.
        degree (int): Degree p, 1 ≤ p < n.
        knots (n + p + 1,): Non-decreasing knot vector; defaults to uniform clamped on [0, 1].
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
        """在参数 u 处把曲线分成两条 B 样条（The NURBS Book §5.2）。

        插入节点使 u 的重数达到 p，曲线在 u 处恰好经过一个控制点，沿它剪开，两段的几何与原曲线完全相同。
        节点插入（Boehm 算法）交给 scipy.interpolate.insert，理由与求值相同：它是数值内核，不承载论文思想。
        Bezier.split 则手写 de Casteljau，因为那几行本身就是原理。

        Args:
            u (float): Split parameter, strictly inside the domain.

        Returns:
            left (BSpline): Piece over [domain[0], u], same parameter.
            right (BSpline): Piece over [u, domain[1]], same parameter.
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
    """Bézier 曲线：节点为 [0]·(p+1) + [1]·(p+1) 的 B 样条，u ∈ [0, 1]。

    作为 BSpline 的子类，求值和求导都沿用 B 样条的实现，只有 split 换成手写的 de Casteljau。

    Args:
        control_points (p + 1, dim): Control points; degree p = count − 1.
    """

    def __init__(self, control_points):
        degree = len(control_points) - 1
        knots = np.concatenate([np.zeros(degree + 1), np.ones(degree + 1)])
        super().__init__(control_points, degree, knots)

    def split(self, u):
        """de Casteljau 算法：在 u 处一分为二，两段的参数都是 [0, 1]。

        逐层做线性插值，每一层的首尾两点恰好是左、右两段的控制点。

        Args:
            u (float): Split parameter in (0, 1).

        Returns:
            left (Bezier): Piece over [0, u], reparameterized to [0, 1].
            right (Bezier): Piece over [u, 1], reparameterized to [0, 1].
        """
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
    """NURBS 曲线：齐次坐标下的 B 样条 H(u) = Σᵢ Nᵢ,ₚ(u) [wᵢPᵢ, wᵢ]，C = A / w（A 为 H 的前 dim 维）。

    对 A = wC 用 Leibniz 法则，得到递推（The NURBS Book §4.3）

        C⁽ᵏ⁾ = (A⁽ᵏ⁾ − Σᵢ₌₁ᵏ C(k, i) w⁽ⁱ⁾ C⁽ᵏ⁻ⁱ⁾) / w

    k 阶导数要用到全部低阶导数，这正是 Curve 要求子类一次返回整个导数栈的原因。

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
