"""弧长表：曲线的 s(u) 与反函数 u(s)，误差由自适应对分控制。

为什么不用教科书上的固定分段：固定 N 段弦长累加、再线性插值反查，u(s) 是折线，du/ds 在每个节点处跳变，
恒速指令下进给也跟着跳；在节点不均匀的样条上，512 段时进给波动仍可达约 12%。这里用自适应 Gauss 求积加
三次 Hermite 插值：被接受的小段直接成为表的网格，误差判据同时覆盖积分和两个方向的插值，所以最终用到的
u(s) 也受控。
"""

import numpy as np
from scipy import interpolate as si

from ..utils import tolerances


class ArcLengthTable:
    """弧长表：s(u) 与反函数 u(s)。

    参数域切成小段：段端点的弧长由 Gauss–Legendre 求积得到；段内 s(u)、u(s) 都用三次 Hermite 插值，
    斜率取 σ = |C'(u)| 和 1/σ，所以 u(s) 是 C¹ 的，du/ds 没有跳变。一段的积分误差和两个方向的插值误差
    （见 _piece_errors）都不超过 tolerance 才接受，否则对半分。

    初始网格先在 curve.breaks 处切开（跨过不光滑点的求积收敛很慢），再把每个节点区间等分成
    initial_pieces 段：只在中点检查插值误差，会漏掉关于自身中点对称的段（误差恰好在中点为零），
    等分之后各段一般不再对称。

    Args:
        curve (Curve): The curve; its parametric speed must be nonzero everywhere.
        tolerance (float): Error allowed per piece, in units of length.
        initial_pieces (int): Equal parts per knot span in the initial grid.

    Attributes:
        u_grid (m + 1,): Piece ends in u.
        s_grid (m + 1,): Arc length at u_grid, starting from 0.
        total (float): Total arc length.

    Note:
        容差是每一段的，不是全表的，总误差理论上是各段之和。但"整段减两半"的估计非常保守，实测总误差
        远小于这个和（见 tests/test_curves.py）。
    """

    def __init__(self, curve, tolerance=tolerances.ARC_LENGTH, initial_pieces=4):
        self.u_grid, self.s_grid, speed = _adaptive_grid(curve, tolerance, initial_pieces)
        self.total = float(self.s_grid[-1])
        # 建好的表用 SciPy 的 CubicHermiteSpline；_piece_errors 里另写 _hermite，是为了对几千个小段一次向量化检查
        self._s_of_u = si.CubicHermiteSpline(self.u_grid, self.s_grid, speed)
        self._u_of_s = si.CubicHermiteSpline(self.s_grid, self.u_grid, 1 / speed)

    def s(self, u):
        """弧长 s(u)。u 由调用者检查（Curve.length_at），这里不再重复。"""
        return self._s_of_u(u)

    def u(self, s):
        """参数 u(s)。s 由调用者检查（Curve.u_at_length），这里不再重复。"""
        return self._u_of_s(s)


def _adaptive_grid(curve, tolerance, initial_pieces):
    """逐层对分，直到每一段都达标；返回 (u_grid, s_grid, u_grid 处的参数速度)。

    按层（广度优先）而不是递归：同一层所有待查的段一次向量化求值，调用 SciPy 的次数只有对分的层数那么多。
    拐角光顺的刀路要为几千条短过渡曲线各建一张表，调用次数比点数更要紧。
    """
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
    """每一小段 [a, b] 的弧长与误差估计，a、b 形状 (n,)。

    弧长取两个半段的 Gauss 积分之和。误差（单位都是长度）取三者的最大值：
        积分误差：整段积分与两半之和的差；
        s(u) 的插值误差：Hermite 在参数中点 m 的值与真值 s(m) 的差；
        u(s) 的插值误差：Hermite 在 s(m) 处给出的参数与 m 的差，乘 σ(m) 换算成长度。
    后两项不能省：只查积分误差，总长是准的，但反查 u(s) 的误差不受控。

    Returns:
        length (n,): Arc length of each piece.
        error (n,): Error estimate of each piece, in units of length.
        speed (n,): Parametric speed σ at the piece starts a.
    """
    m = (a + b) / 2
    lo, hi = np.stack([a, a, m]), np.stack([b, m, b])  # (3, n)：整段、左半、右半
    nodes = (lo + hi)[..., None] / 2 + (hi - lo)[..., None] / 2 * _X  # (3, n, 8)
    # 所有 Gauss 节点与三个端点的速度一次求出：每次调用 SciPy 都有固定开销
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
    """区间长 h 上的三次 Hermite 插值在相对位置 t ∈ [0, 1] 处的值；y0、y1 为端点值，dy0、dy1 为端点斜率。"""
    return (
        (1 + 2 * t) * (1 - t) ** 2 * y0
        + t**2 * (3 - 2 * t) * y1
        + h * t * (1 - t) ** 2 * dy0
        + h * t**2 * (t - 1) * dy1
    )
