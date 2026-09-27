"""Arc-length table of a curve: s(u) and its inverse u(s), error-controlled by adaptive bisection."""

import numpy as np
from scipy import interpolate as si

from ..utils import tolerances


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
