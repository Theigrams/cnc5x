"""参数曲线：Line、Bezier、BSpline、NURBS、SubCurve、Reparameterized，以及弧长表。

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
from scipy.optimize import brentq

from . import geometry, tolerances
from .calculus import compose, inverse_derivatives, speed_derivatives


class Curve:
    """参数曲线基类。

    子类设定 domain，并实现 _derivatives(u, order)：返回 0 到 order 阶导数叠成的数组，
    形状 (order + 1, ..., dim)；传入的 u 已经检查过。

    为什么一次返回"一叠"导数，而不是一次只求一阶：NURBS、单位化、复合函数这类曲线，
    求 k 阶导数要先求出全部低阶导数。逐阶单独求的话，求到三阶时低阶部分已经重复算了好几遍
    （五轴刀位曲线的一次求值要多调用两倍多的 B 样条）。一次算出整叠，每个量只算一遍；
    order 参数让只要低阶导数的调用（求点、求速度）不必去算高阶。
    """

    domain = (0.0, 1.0)

    # 导数栈里用来计量弧长的分量。五轴刀位曲线 [p, o] 只量刀尖（见 toolpath.PoseCurve）。
    measured = slice(None)

    def _derivatives(self, u, order):
        raise NotImplementedError

    # ---------- 求值 ----------
    def derivatives(self, u, order=3):
        """导数栈 [C, C', C'', C''']，形状 (4, ..., dim)；给出 order 时只取前 order + 1 层。"""
        if order not in (0, 1, 2, 3):
            raise ValueError("order 只能是 0、1、2、3")
        return self._derivatives(self._check(u), order)

    def derivative(self, u, order=1):
        """第 order 阶导数 dᵏC/duᵏ；order = 0 就是曲线上的点。"""
        return self.derivatives(u, order)[order]

    def __call__(self, u, order=0):
        return self.derivative(u, order)

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
        d = self.derivatives(u, 2)
        return geometry.curvature(d[1], d[2])

    # ---------- 弧长 ----------
    def speed(self, u):
        """ds/du = |C'(u)|，只量 measured 分量。"""
        return np.linalg.norm(self.derivative(u, 1)[..., self.measured], axis=-1)

    def speed_derivatives(self, d):
        """弧长 s(u) 的 1..3 阶导数 (s', s'', s''')，d 是本曲线在 u 处的导数栈 (4, ..., dim)。

        传入现成的导数栈而不是 u：调用者手里通常已经有它，不必为了弧长再求一遍导数。
        """
        return speed_derivatives(d[..., self.measured])

    def derivatives_by_length(self, u):
        """参数 u 处对弧长 s 的导数栈 [C, C_s, C_ss, C_sss]，形状 (4, ..., dim)。

        先由导数栈求弧长的导数 (s', s'', s''')，再用反函数求导得 (u_s, u_ss, u_sss)，
        最后用链式法则换元。整个过程只求一次导数栈。
        """
        d = self.derivatives(u)
        return compose(d, *inverse_derivatives(*self.speed_derivatives(d)))

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
        """转成浮点数组；允许超出参数域的舍入误差（相对 ROUNDING），并截回域内。"""
        u = np.asarray(u, dtype=float)
        lo, hi = self.domain
        tol = tolerances.ROUNDING * max(1.0, hi - lo)
        if not np.all((u >= lo - tol) & (u <= hi + tol)):
            raise ValueError(f"参数超出定义域 [{lo}, {hi}]")
        return np.clip(u, lo, hi)


# 8 点 Gauss–Legendre 求积：节点 X 与权重 W（区间 [−1, 1]），对 15 次以内的多项式精确。
_X, _W = np.polynomial.legendre.leggauss(8)


def _hermite(t, y0, y1, dy0, dy1, h):
    """区间长 h 上的三次 Hermite 插值在相对位置 t ∈ [0, 1] 处的值（端点值 y、斜率 dy）。"""
    return (
        (1 + 2 * t) * (1 - t) ** 2 * y0
        + t**2 * (3 - 2 * t) * y1
        + h * t * (1 - t) ** 2 * dy0
        + h * t**2 * (t - 1) * dy1
    )


def _piece_errors(curve, a, b):
    """每一小段 [a, b] 的弧长与误差估计，a、b 形状 (n,)。

    弧长取两个半段的 Gauss 积分之和；误差取以下三者的最大值（单位都是长度）：
        积分误差：整段积分与两半之和的差；
        s(u) 的插值误差：Hermite 在参数中点 m 的值与真值 s(m) 的差；
        u(s) 的插值误差：Hermite 在 s(m) 处给出的参数与 m 的差，乘 σ(m) 换算成长度。
    """
    m = (a + b) / 2
    lo, hi = np.stack([a, a, m]), np.stack([b, m, b])  # (3, n)：整段、左半、右半
    nodes = (lo + hi)[..., None] / 2 + (hi - lo)[..., None] / 2 * _X  # (3, n, 8)
    # 所有 Gauss 节点与三个端点的速度一次求出：每次调用 SciPy 都有固定开销，
    # 拐角光顺的刀路要为几千条短过渡曲线各建一张表，调用次数比点数更要紧。
    speed = curve.speed(np.concatenate([nodes.ravel(), a, m, b]))
    whole, left, right = (hi - lo) / 2 * (speed[: nodes.size].reshape(nodes.shape) @ _W)
    sa, sm, sb = speed[nodes.size :].reshape(3, -1)
    if np.any(sa == 0) or np.any(sm == 0) or np.any(sb == 0):
        raise ValueError("曲线上有速度为零的点，不能用弧长作参数")
    length = left + right
    h = b - a
    s_mid = _hermite(0.5, 0.0, length, sa, sb, h)
    u_mid = _hermite(left / length, a, b, 1 / sa, 1 / sb, length)
    error = np.maximum(np.abs(whole - length), np.abs(s_mid - left))
    return length, np.maximum(error, sm * np.abs(u_mid - m))


class ArcLengthTable:
    """弧长表：s(u) 与反函数 u(s)，误差受控（自适应对分）。

    把参数域切成小段，段端点处的弧长 sₖ 由 Gauss 积分算出，段内的 s(u)、u(s) 都用三次
    Hermite 插值：s(u) 的斜率取 σ = |C'(u)|，u(s) 的斜率取 1/σ。每一段检查积分误差和
    两个方向的插值误差（见 _piece_errors），超过 tolerance 的段对半分后再查，直到全部达标。

    起始网格：先在 breaks 处分开（跨过不光滑点的积分收敛很慢），再把每个节点区间四等分。
    只在中点检查插值误差，会漏掉关于自身中点对称的段（那种段的误差恰好在中点为零），
    四等分之后每一小段一般不再对称。
    """

    def __init__(self, curve, tolerance=tolerances.ARC_LENGTH, split=4):
        breaks = curve.breaks
        starts = (breaks[:-1, None] + np.diff(breaks)[:, None] * np.arange(split) / split).ravel()
        ends = np.append(starts[1:], breaks[-1])
        done_starts, done_lengths = [], []
        for _ in range(tolerances.ARC_LENGTH_MAX_DEPTH):
            length, error = _piece_errors(curve, starts, ends)
            good = error <= tolerance
            done_starts.append(starts[good])
            done_lengths.append(length[good])
            middles = (starts[~good] + ends[~good]) / 2
            starts = np.concatenate([starts[~good], middles])
            ends = np.concatenate([middles, ends[~good]])
            if len(starts) == 0:
                break
        else:
            raise ValueError("弧长表对分 40 次仍不收敛：曲线上可能有速度接近零的点")
        starts = np.concatenate(done_starts)
        order = np.argsort(starts)
        self.u_grid = np.append(starts[order], breaks[-1])
        self.s_grid = np.concatenate([[0.0], np.cumsum(np.concatenate(done_lengths)[order])])
        self.total = float(self.s_grid[-1])
        speed = curve.speed(self.u_grid)
        self._s_of_u = si.CubicHermiteSpline(self.u_grid, self.s_grid, speed)
        self._u_of_s = si.CubicHermiteSpline(self.s_grid, self.u_grid, 1 / speed)

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

    def speed(self, u):
        # 弧长表只要 |C'|：直接求一阶导数，不必像 Curve.speed 那样连带求出曲线上的点
        return np.linalg.norm(self._spline(self._check(u), nu=1), axis=-1)

    def _derivatives(self, u, order):
        zero = np.zeros(u.shape + self.control_points.shape[1:])
        return np.stack([self._spline(u, nu=k) if k <= self.degree else zero for k in range(order + 1)])

    def split(self, u):
        """在参数 u 处分成两条 B 样条：插入节点使 u 的重数达到 p（The NURBS Book §5.2）。

        节点插入（Boehm 算法）交给 scipy.interpolate.insert，理由与求值相同：它是数值内核，
        不承载论文的算法思想。Bézier.split 则手写 de Casteljau，因为那几行本身就是原理。
        """
        lo, hi = self.domain
        if not lo < u < hi:
            raise ValueError("分割点必须在参数域内部")
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
    k 阶导数用到全部低阶导数，所以一次递推出整个导数栈。
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
    """曲线在 [a, b] 上的一段，沿用原来的参数（用于在临界点处把曲线切成 block）。

    子段的子段直接指向原曲线，免得层层嵌套。
    """

    def __init__(self, curve, a, b):
        if isinstance(curve, SubCurve):
            curve = curve.curve
        lo, hi = curve.domain
        if not lo <= a < b <= hi:
            raise ValueError("子区间必须落在原参数域内")
        self.curve, self.domain = curve, (float(a), float(b))
        self.measured = curve.measured

    @property
    def breaks(self):
        a, b = self.domain
        inner = self.curve.breaks
        return np.unique(np.concatenate([[a], inner[(inner > a) & (inner < b)], [b]]))

    def _derivatives(self, u, order):
        return self.curve.derivatives(u, order)

    def speed(self, u):
        return self.curve.speed(u)  # 原曲线可能有更省的求法（例如五轴刀位只求刀尖）

    def curvature(self, u):
        return self.curve.curvature(u)


class Reparameterized(Curve):
    """复合曲线 C(g(w))：g 是严格递增的一维映射曲线（求值形状 (..., 1)），w 是新参数。

    导数由三阶链式法则得到（calculus.compose）。C 的内部节点在 w 下的原像要并入 breaks，
    否则弧长积分会跨过不光滑的点；原像由 brentq 在 g 上反解得到。
    """

    def __init__(self, curve, mapping):
        lo, hi = mapping.domain
        w = np.unique(np.concatenate([np.linspace(lo, hi, 201), mapping.breaks]))
        if np.any(mapping.derivative(w, 1)[..., 0] <= 0):
            raise ValueError("映射 g 必须严格递增（在样本上检查）")
        g_lo, g_hi = float(mapping(lo)[0]), float(mapping(hi)[0])
        c_lo, c_hi = curve.domain
        tol = tolerances.ROUNDING * max(1.0, c_hi - c_lo)
        if g_lo < c_lo - tol or g_hi > c_hi + tol:
            raise ValueError("映射的值域超出了曲线的参数域")
        inner = curve.breaks[(curve.breaks > g_lo) & (curve.breaks < g_hi)]
        preimages = [brentq(lambda x: float(mapping(x)[0]) - b, lo, hi, xtol=tolerances.ROOT_XTOL) for b in inner]
        self.curve, self.mapping, self.domain = curve, mapping, mapping.domain
        self.measured = curve.measured
        self._breaks = np.unique(np.concatenate([mapping.breaks, preimages]))

    @property
    def breaks(self):
        return self._breaks

    def _derivatives(self, w, order):
        g = self.mapping.derivatives(w)[..., 0]  # (4, ...)
        d = self.curve.derivatives(g[0])  # (4, ..., dim)
        return compose(d, g[1], g[2], g[3])[: order + 1]
