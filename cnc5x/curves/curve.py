"""曲线的抽象：Curve 基类，以及两种包装——取子段 SubCurve、换参数 Reparameterized。

具体的曲线（Line、Bezier、BSpline、NURBS）在 spline.py，弧长表在 arclength.py。
"""

from functools import cached_property

import numpy as np
from scipy.optimize import brentq

from ..utils import calculus, geometry, tolerances
from .arclength import ArcLengthTable


class Curve:
    """参数曲线 C(u) 的基类，u ∈ domain。

    子类一次返回一整叠导数，而不是每次只求一阶：NURBS、单位化、复合函数这类曲线，k 阶导数要用到全部
    低阶导数，逐阶单独求会把低阶部分反复重算；一次求一叠，每个量只算一遍。`order` 让只要低阶的调用
    （求点、求参数速度）不必去算高阶。

    弧长表很贵（每张要求几千次速度），很多曲线又用不到（刀轴曲线、只求值的曲线），所以第一次用到时才建，
    建好后缓存。这要求曲线构造后不再修改。
    """

    def _derivatives(self, u, order):
        raise NotImplementedError

    # ---------- 求值 ----------
    def derivatives(self, u, order=3):
        """导数栈 [C, C', C'', C''']，只求到 `order` 阶。

        参数检查只在这里做一次，子类的 `_derivatives` 拿到的 u 已经合法，公式里不必再写防御代码。

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
        """曲线上的点 C(u)（order = 0），或第 order 阶导数；形状 (..., dim)。"""
        return self.derivatives(u, order)[order]

    @property
    def breaks(self):
        """参数域内可能不光滑的点（含两端），形状 (m,)。

        相邻两点之间曲线无穷次可导，到了这些点上某阶导数可能跳变。弧长表在这里切开积分：Gauss 求积
        跨过不光滑点时收敛很慢，误差估计也会失准。基类默认整段光滑，分段的曲线必须覆盖它。
        """
        return np.array(self.domain)

    @property
    def start_point(self):
        return self(self.domain[0])

    @property
    def end_point(self):
        return self(self.domain[1])

    def curvature(self, u):
        """曲率 κ(u)，1/mm，形状同 u。"""
        d = self.derivatives(u, 2)
        return geometry.curvature(d[1], d[2])

    def restrict(self, a, b):
        """[a, b] 上的一段，参数保持原值。

        做成方法而不是让调用者直接写 SubCurve(...)，是为了让五轴的 PoseCurve 能改成对刀尖、刀轴两条
        分量分别取子段（CLAUDE.md 第 3.6 条）。
        """
        return SubCurve(self, a, b)

    def sample(self, n=100, by_length=False):
        """取 n 个点画图用：默认参数等分，by_length 为真时弧长等分；形状 (n, dim)。"""
        if by_length:
            return self(self.u_at_length(np.linspace(0.0, self.length, n)))
        lo, hi = self.domain
        return self(np.linspace(lo, hi, n))

    # ---------- 弧长 ----------
    def parametric_speed(self, u):
        """参数速度 σ(u) = |C'(u)| = ds/du，单位为 mm 每单位 u，形状同 u。

        名字里带 parametric，是为了和进给速度 v = ds/dt 区分：σ 只是曲线的几何量，与时间无关。
        弧长表积分的就是它，子类可以覆盖成更省的求法（例如 BSpline 只求一阶导数）。
        """
        return np.linalg.norm(self(u, 1), axis=-1)

    def derivatives_by_length(self, u):
        """参数 u 处对弧长 s 的导数栈 [C, C_s, C_ss, C_sss]。

        刀路和插补都按弧长计量，而曲线只会对 u 求导。换元分三步，全部从同一个导数栈算出：由 |C'| 及其
        导数得 (s', s'', s''')（calculus.speed_derivatives），反函数求导得 (u_s, u_ss, u_sss)
        （calculus.inverse_derivatives），再用链式法则 calculus.compose。它不经过弧长表，所以
        |C_s| = 1、C_s·C_ss = 0 精确到舍入误差，不受弧长表精度的影响。

        Args:
            u (...): Parameter values in `domain`.

        Returns:
            d (4, ..., dim): d[k] = dᵏC/dsᵏ.
        """
        d = self.derivatives(u)
        return calculus.compose(d, *calculus.inverse_derivatives(*calculus.speed_derivatives(d)))

    @cached_property
    def arc(self):
        """弧长表（ArcLengthTable），第一次用到时才建立。"""
        return ArcLengthTable(self)

    @property
    def length(self):
        """总弧长，mm。"""
        return self.arc.total

    def length_at(self, u):
        """从起点到参数 u 的弧长 s(u)，mm，形状同 u。"""
        return self.arc.s(self._check(u))

    def u_at_length(self, s):
        """弧长 s ∈ [0, length] 处的参数 u(s)，即 length_at 的反函数，形状同 s。"""
        return self.arc.u(self._check_length(s))

    def _check(self, u):
        return _clip_to(u, self.domain, "parameter")

    def _check_length(self, s):
        """弧长越界的规则与参数相同；零长度的曲线没有弧长参数，显式报错，而不是算出 0/0。"""
        if self.length == 0:
            raise ValueError("curve has zero length, so arc length cannot parameterize it")
        return _clip_to(s, (0.0, self.length), "arc length")


def _clip_to(x, interval, name):
    """转成浮点数组并截回 [lo, hi]；越界不超过 tolerances.ROUNDING（相对）时截回，超出就报错。

    参数和弧长常由累加、求根算出，端点处可能差 1e-16 量级，直接报错会误伤；越界再多就是调用者的错。
    """
    x = np.asarray(x, dtype=float)
    lo, hi = interval
    tol = tolerances.ROUNDING * max(1.0, hi - lo)
    if not np.all((x >= lo - tol) & (x <= hi + tol)):
        raise ValueError(f"{name} outside [{lo}, {hi}]")
    return np.clip(x, lo, hi)


class SubCurve(Curve):
    """曲线在 [a, b] 上的一段，参数保持原值。

    不重新参数化到 [0, 1]：与原曲线在同一参数处取值完全相同，切开的各段首尾严丝合缝，不引入任何
    逼近误差。子段的子段直接指向原曲线，免得包装层层嵌套。

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
    """复合曲线 C(g(w))：g 是严格递增的一维映射曲线，w 是新参数。

    用来换参数而不改几何，例如让刀轴按自己转过的角度参数化，再同步到刀尖的参数上（Yuen et al. 2013，
    见 pose.pose_spline）。导数由三阶链式法则 calculus.compose 得到。C 的内部节点在 w 下的原像
    要并入 breaks，否则弧长积分会跨过不光滑的点；原像用 brentq 在 g 上反解。

    Args:
        curve (Curve): The curve C.
        mapping (Curve): Mapping g(w), evaluating to shape (..., 1); its range must lie in curve.domain.

    Note:
        g 的单调性只在 201 个样本和 g 的节点上检查，不是连续域上的证明。
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
