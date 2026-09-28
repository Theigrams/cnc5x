"""由离散数据构造 B 样条：参数化、节点、插值、最小二乘逼近、Hermite、单调插值、进给修正、球坐标刀轴。

曲线 C(u) = Σⱼ Nⱼ,ₚ(u) Pⱼ 对控制点 P 是线性的。参数 ū 处的一个条件 C⁽ᵏ⁾(ū) = D，
就是基函数矩阵 N⁽ᵏ⁾[i, j] = Nⱼ,ₚ⁽ᵏ⁾(ūᵢ) 的一行，所以各种构造都归结为线性方程组：
    插值            N P = Q                           方阵，唯一解
    最小二乘        min Σ wᵢ |(N P − Q)ᵢ|²              法方程 Nᵀ W N P = Nᵀ W Q
    带约束最小二乘  另要求 A P = b 精确成立             Lagrange 乘子（KKT 方程组）
基函数的值交给 SciPy 计算；方程怎么列、怎么解，都写在这里。
"""

from math import comb, factorial

import numpy as np
from scipy import interpolate as si
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..utils import tolerances
from ..utils.calculus import inverse_derivatives, speed_derivatives
from ..utils.geometry import angle_between, unit
from .orientation import SphericalCurve
from .spline import Bezier, BSpline

# ---------- 参数化与节点 ----------


def chord_parameters(points, exponent=1.0):
    """累积弦长参数，归一化到 [0, 1]；exponent = 0.5 为向心参数化。"""
    if len(points) < 2:
        raise ValueError("至少需要两个点")
    d = np.linalg.norm(np.diff(np.asarray(points, dtype=float), axis=0), axis=1)
    if np.any(d == 0):
        raise ValueError("相邻点重合，无法用弦长参数化")
    u = np.concatenate([[0.0], np.cumsum(d**exponent)])
    return u / u[-1]


def angle_parameters(axes, exponent=1.0):
    """刀轴的累积角度参数（刀轴在单位球面上的"弦长"），归一化到 [0, 1]；exponent = 0.5 为向心参数化。

    刀轴曲线用自己的参数拟合时用它（Yuen et al. 2013），再与刀尖参数同步（见 monotone_interpolate）。
    相邻刀轴相同时角度为零，参数不再严格递增，这时报错。
    """
    if len(axes) < 2:
        raise ValueError("至少需要两个刀轴")
    o = unit(axes)
    angle = angle_between(o[:-1], o[1:])
    if np.any(angle == 0):
        raise ValueError("相邻刀轴相同，无法用角度参数化")
    w = np.concatenate([[0.0], np.cumsum(angle**exponent)])
    return w / w[-1]


def averaged_knots(parameters, degree):
    """插值用的平均节点（The NURBS Book 式 (9.8)）：u_{j+p} = (ū_j + … + ū_{j+p−1}) / p。"""
    u = np.asarray(parameters, dtype=float)
    p = degree
    inner = [np.mean(u[j : j + p]) for j in range(1, len(u) - p)]
    return np.concatenate([np.full(p + 1, u[0]), inner, np.full(p + 1, u[-1])])


def approximation_knots(parameters, n_controls, degree):
    """最小二乘逼近用的节点（The NURBS Book 式 (9.68)–(9.69)），保证每个节点区间里都有数据点。

    m + 1 个数据点、n + 1 个控制点：d = (m + 1)/(n − p + 1)，
    u_{p+j} = (1 − α) ū_{i−1} + α ū_i，其中 i = ⌊jd⌋，α = jd − i，j = 1 … n − p。
    n_controls = degree + 1 时没有内部节点，得到的就是一段多项式（Bézier）。
    """
    u = np.asarray(parameters, dtype=float)
    m, n, p = len(u) - 1, n_controls - 1, degree
    if not p < n_controls <= len(u):
        raise ValueError("需要 degree < n_controls ≤ 数据点数")
    d = (m + 1) / (n - p + 1)
    inner = []
    for j in range(1, n - p + 1):
        i = int(j * d)
        alpha = j * d - i
        inner.append((1 - alpha) * u[i - 1] + alpha * u[i])
    return np.concatenate([np.full(p + 1, u[0]), inner, np.full(p + 1, u[-1])])


def basis_matrix(knots, degree, parameters, order=0):
    """基函数矩阵 N[i, j] = Nⱼ,ₚ⁽ᵏ⁾(ūᵢ)，形状 (m, n)，k = order。

    第 j 个基函数就是"第 j 个控制点为 1、其余为 0"的 B 样条，所以把单位矩阵当作控制点
    交给 scipy 求值，一次得到全部基函数。每行最多 p + 1 个非零元（局部支撑）。
    """
    n = len(knots) - degree - 1
    return si.BSpline(knots, np.eye(n), degree)(np.asarray(parameters, dtype=float), nu=order)


def condition_rows(knots, degree, conditions):
    """条件 C⁽ᵏ⁾(ū) = D 的线性方程 A P = b：A 的每行是 k 阶基函数在 ū 处的值，形状 (条件数, n)。

    conditions：[(ū, k, D), ...]。ū 必须落在参数域 [t_p, t_{n}] 内（允许相对 ROUNDING 的舍入），
    否则 scipy 会按端点多项式外推，条件其实加在了曲线之外。
    """
    lo, hi = knots[degree], knots[-degree - 1]
    tol = tolerances.ROUNDING * max(1.0, hi - lo)
    x = np.array([c[0] for c in conditions], dtype=float)
    if np.any((x < lo - tol) | (x > hi + tol)):
        raise ValueError(f"条件的参数 ū = {x} 超出参数域 [{lo}, {hi}]")
    x = np.clip(x, lo, hi)
    A = np.vstack([basis_matrix(knots, degree, [xi], k) for xi, (_, k, _) in zip(x, conditions)])
    b = np.vstack([np.reshape(D, (1, -1)) for _, _, D in conditions])
    return A, b


# ---------- 插值与逼近 ----------


def interpolate_bspline(points, parameters, degree=3, knots=None, derivatives=()):
    """过所有点的 B 样条：C(ūᵢ) = Qᵢ，另可加导数条件 C⁽ᵏ⁾(ū) = D。

    points (m, dim)；parameters (m,) 严格递增，由调用者给出（五轴刀位的位置和刀轴要共用同一组参数）。
    knots 缺省时取平均节点，恰好 m 个控制点，这时不能再加导数条件：每多一个条件就多一个
    控制点，节点要由调用者按论文给出。derivatives：[(ū, k, D), ...]。

    方程组 [N; A] P = [Q; D] 按稀疏矩阵求解：N 每行只有 p + 1 个非零元，几千个点也只要几毫秒。
    """
    Q = np.asarray(points, dtype=float)
    u = np.asarray(parameters, dtype=float)
    if np.any(np.diff(u) <= 0):
        raise ValueError("参数必须严格递增")
    knots = averaged_knots(u, degree) if knots is None else np.asarray(knots, dtype=float)
    rows, values = [si.BSpline.design_matrix(u, knots, degree)], [Q.reshape(len(u), -1)]
    if derivatives:
        A, b = condition_rows(knots, degree, derivatives)
        rows.append(sparse.csr_array(A))
        values.append(b)
    M = sparse.vstack(rows).tocsc()
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"条件数 {M.shape[0]} 与控制点数 {M.shape[1]} 不相等")
    P = spsolve(M, np.vstack(values))
    return BSpline(P.reshape(M.shape[1], -1), degree, knots)


def fit_bspline(points, parameters, knots, degree=3, weights=None, constraints=()):
    """最小二乘 B 样条：min Σ wᵢ |C(ūᵢ) − Qᵢ|²，并精确满足约束 C⁽ᵏ⁾(ū) = D。

    knots 由调用者给出（例如 approximation_knots）；constraints：[(ū, k, D), ...]，
    例如 [(ū₀, 0, Q₀), (ūₘ, 0, Qₘ)] 让首末点精确插值。points 可以是 (m,) 的一维数据。
    无约束时解法方程 NᵀWN P = NᵀWQ；有约束时引入 Lagrange 乘子 λ，解 KKT 方程组
        [NᵀWN  Aᵀ] [P]   [NᵀWQ]
        [A     0 ] [λ] = [b   ]
    （The NURBS Book §9.4.2）。约束必须线性无关（因而不多于控制点数），否则 KKT 矩阵奇异。
    """
    u = np.asarray(parameters, dtype=float)
    Q = np.asarray(points, dtype=float).reshape(len(u), -1)
    N = basis_matrix(knots, degree, u)
    w = np.ones(len(u)) if weights is None else np.asarray(weights, dtype=float)
    NtW = N.T * w
    G, rhs = NtW @ N, NtW @ Q
    if constraints:
        A, b = condition_rows(knots, degree, constraints)
        if np.linalg.matrix_rank(A) < len(A):
            raise ValueError(f"{len(A)} 条约束线性相关（或多于控制点数 {N.shape[1]}），KKT 方程组奇异")
        G = np.block([[G, A.T], [A, np.zeros((len(A), len(A)))]])
        rhs = np.vstack([rhs, b])
    P = np.linalg.solve(G, rhs)[: N.shape[1]]
    return BSpline(P, degree, knots)


# ---------- 由端点导数构造 ----------


def hermite(start, end):
    """两点 Hermite 插值：Bézier 曲线在 u = 0、1 处的 0..k 阶导数分别等于 start、end。

    start、end 是 k + 1 层的导数栈 (k + 1, dim)，对 u ∈ [0, 1] 求导；曲线次数 n = 2k + 1
    （k = 1 三次、k = 2 五次、k = 3 七次）。Bézier 端点的 r 阶导数只取决于端点附近的 r + 1 个控制点：
        C⁽ʳ⁾(0) = n!/(n−r)! · Δʳb₀，  Δʳb₀ = Σᵢ (−1)ʳ⁻ⁱ C(r, i) bᵢ
        C⁽ʳ⁾(1) = n!/(n−r)! · ∇ʳbₙ，  ∇ʳbₙ = Σᵢ (−1)ⁱ C(r, i) bₙ₋ᵢ
    从 r = 0 起逐个解出 bᵣ 与 bₙ₋ᵣ。五次时即 b₁ = b₀ + C'/5，b₂ = 2b₁ − b₀ + C''/20。
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    k = len(start) - 1
    n = 2 * k + 1
    b = np.zeros((n + 1,) + start.shape[1:])
    for r in range(k + 1):
        scale = factorial(n - r) / factorial(n)
        b[r] = scale * start[r] - sum((-1) ** (r - i) * comb(r, i) * b[i] for i in range(r))
        b[n - r] = (-1) ** r * (scale * end[r] - sum((-1) ** i * comb(r, i) * b[n - i] for i in range(r)))
    return Bezier(b)


def join_beziers(pieces, breaks):
    """把首尾相接的同次 Bézier 段拼成一条 B 样条，第 i 段占参数区间 [breaks[i], breaks[i+1]]。

    内部节点重数取 p：相邻段只共用衔接处的一个控制点，各段的控制点原样保留（Bézier 的控制点
    与参数区间的伸缩无关）。这种表示本身只保证 C⁰，更高阶的连续性由各段的端点条件保证。
    """
    p = pieces[0].degree
    control = [pieces[0].control_points]
    for left, right in zip(pieces[:-1], pieces[1:]):
        gap = np.abs(right.control_points[0] - left.control_points[-1]).max()
        if right.degree != p or gap > tolerances.ROUNDING * (1 + np.abs(left.control_points).max()):
            raise ValueError("各段必须同次并且首尾相接")
        control.append(right.control_points[1:])
    breaks = np.asarray(breaks, dtype=float)
    knots = np.concatenate([np.full(p + 1, breaks[0]), np.repeat(breaks[1:-1], p), np.full(p + 1, breaks[-1])])
    return BSpline(np.concatenate(control), p, knots)


def monotone_interpolate(x, y):
    """严格递增数据 (xₖ, yₖ) 的 C² 单调插值，返回一维 B 样条 y = g(x)，常用作参数同步映射。

    每段是一条五次 Hermite 曲线：节点斜率 mₖ 取两侧割线斜率 Δ 的调和平均（Fritsch & Butland 1984，
    与 SciPy 的 PCHIP 同一思路），节点二阶导数取 0，所以整条曲线 C² 且精确过每个数据点。
    单调性：Bézier 的控制点严格递增时曲线严格递增（导数的 Bernstein 系数全为正）。五次 Hermite 的
    中间两个控制点之差为 h(Δ − 0.4(m₀ + m₁))，所以要求每段 m₀ + m₁ ≤ 2Δ；超了就把这段两端的斜率
    按比例缩小。缩小只会让相邻段的条件更容易满足，一遍扫描即可。直线数据会被原样还原。
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError("x 与 y 都必须严格递增")
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.concatenate([[delta[0]], 2 / (1 / delta[:-1] + 1 / delta[1:]), [delta[-1]]])
    for i in range(len(h)):
        excess = (m[i] + m[i + 1]) / (2 * delta[i])
        if excess > 1:
            m[i] /= excess
            m[i + 1] /= excess
    pieces = []
    for i in range(len(h)):
        start = [[y[i]], [h[i] * m[i]], [0.0]]  # 对段内参数 t ∈ [0, 1] 求导：dy/dt = h·dy/dx
        end = [[y[i + 1]], [h[i] * m[i + 1]], [0.0]]
        pieces.append(hermite(start, end))
    return join_beziers(pieces, x)


# ---------- 进给修正多项式 ----------


def feed_correction(curve, degree=9, continuity=3, tolerance=1e-6, samples=32):
    """进给修正多项式（Erkorkmaz & Altintas 2001；Yuen et al. 2013）：用分段多项式 ũ(s) 逼近弧长的反函数。

    在曲线的每个节点区间上，由弧长表取 samples 个样本 (sₖ, uₖ)，用 degree 次多项式做最小二乘拟合；
    两端的 u 及其对 s 的 1..continuity 阶导数取精确值（反函数求导），所以相邻段之间 C^continuity。
    Yuen 2013 取 9 次、C³：10 个系数里 8 个由端点定下，只剩 2 个由最小二乘决定。
    拟合后检查段内的进给误差 |σ(ũ)·ũ_s − 1|（σ = |C'|；这是实际进给与指令进给之比减 1）与单调性，
    不合格的段对半分后重拟合。返回一维 B 样条 s → u，参数域 [0, 弧长]。
    """
    if 2 * (continuity + 1) > degree + 1:
        raise ValueError("端点条件 2(continuity + 1) 个，不能多于系数 degree + 1 个")
    lo, hi = curve.domain
    pieces, breaks = [], [0.0]
    stack = list(zip(curve.breaks[:-1], curve.breaks[1:]))[::-1]  # 从左到右处理：最左边的区间在栈顶
    while stack:
        a, b = stack.pop()
        piece = _correction_span(curve, a, b, degree, continuity, tolerance, samples)
        if piece is not None:
            pieces.append(piece)
            breaks.append(float(curve.length_at(b)))
        elif b - a > tolerances.ROUNDING * (hi - lo):
            m = (a + b) / 2
            stack += [(m, b), (a, m)]  # 对半分，左半在栈顶
        else:
            raise ValueError("进给修正多项式在极短的区间上仍达不到精度")
    return join_beziers(pieces, breaks)


def _correction_span(curve, a, b, degree, continuity, tolerance, samples):
    """在参数区间 [a, b] 上拟合 ũ(r)，r = (s − sₐ)/h ∈ [0, 1]，h 为区间弧长；不合格时返回 None。"""
    sa, sb = curve.length_at([a, b])
    h = sb - sa
    u = np.linspace(a, b, samples)
    r = (curve.length_at(u) - sa) / h
    constraints = []
    for x, end in ((0.0, a), (1.0, b)):
        stack = [end, *inverse_derivatives(*speed_derivatives(curve.derivatives(end)))]  # u, u_s, u_ss, u_sss
        constraints += [(x, k, stack[k] * h**k) for k in range(continuity + 1)]  # 对 r 求导要乘 hᵏ
    knots = np.concatenate([np.zeros(degree + 1), np.ones(degree + 1)])
    piece = fit_bspline(u, r, knots, degree, constraints=constraints)
    check = np.linspace(0.0, 1.0, 4 * samples)
    u_fit, u_r = piece(check)[:, 0], piece(check, 1)[:, 0]
    feed_error = np.abs(curve.parametric_speed(u_fit) * u_r / h - 1)
    if np.any(u_r <= 0) or feed_error.max() > tolerance:
        return None
    return Bezier(piece.control_points)


# ---------- 刀轴 ----------


def spherical_spline(axes, parameters, degree=3):
    """刀轴的球坐标样条（Yuen et al. 2013）：θ = arccos(o_z)、φ = atan2(o_y, o_x)（展开成连续值），
    对二维数据 [θ, φ] 插值一条 B 样条，再映回单位球面（orientation.SphericalCurve）。

    刀轴平行于 z 轴（极点）时 φ 没有定义，报错；靠近极点时 φ 变化剧烈，样条会振荡。
    """
    o = unit(axes)
    if np.any(np.hypot(o[:, 0], o[:, 1]) < tolerances.POLE):
        raise ValueError("刀轴平行于 z 轴（极点），方位角 φ 没有定义")
    theta = np.arccos(np.clip(o[:, 2], -1.0, 1.0))
    phi = np.unwrap(np.arctan2(o[:, 1], o[:, 0]))
    return SphericalCurve(interpolate_bspline(np.column_stack([theta, phi]), parameters, degree))
