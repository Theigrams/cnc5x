"""由离散点构造 B 样条：参数化、插值、最小二乘逼近。"""

import numpy as np
from scipy import interpolate as si

from .curves import BSpline


def chord_parameters(points, exponent=1.0):
    """累积弦长参数，归一化到 [0, 1]；exponent = 0.5 为向心参数化。"""
    d = np.linalg.norm(np.diff(np.asarray(points, dtype=float), axis=0), axis=1)
    if np.any(d == 0):
        raise ValueError("相邻点重合，无法用弦长参数化")
    u = np.concatenate([[0.0], np.cumsum(d**exponent)])
    return u / u[-1]


def averaged_knots(parameters, degree):
    """插值用的平均节点（The NURBS Book §9.2.1）：u_{j+p} = (ū_j + … + ū_{j+p−1}) / p。"""
    u = np.asarray(parameters, dtype=float)
    p = degree
    inner = [np.mean(u[j : j + p]) for j in range(1, len(u) - p)]
    return np.concatenate([np.full(p + 1, u[0]), inner, np.full(p + 1, u[-1])])


def interpolate_bspline(points, parameters, degree=3):
    """过所有点的 B 样条，C(ūᵢ) = Pᵢ。

    参数 ū 由调用者给出：五轴刀位的位置和刀轴要共用同一组参数，不能各算各的。
    """
    u = np.asarray(parameters, dtype=float)
    if np.any(np.diff(u) <= 0):
        raise ValueError("参数必须严格递增")
    knots = averaged_knots(u, degree)
    spline = si.make_interp_spline(u, np.asarray(points, dtype=float), k=degree, t=knots)
    return BSpline(spline.c, degree, knots)


def approximation_knots(parameters, n_controls, degree):
    """最小二乘逼近用的节点（The NURBS Book §9.4.1），保证每个节点区间里都有数据点。

    m + 1 个数据点、n + 1 个控制点：d = (m + 1)/(n − p + 1)，
    u_{p+j} = (1 − α) ū_{i−1} + α ū_i，其中 i = ⌊jd⌋，α = jd − i，j = 1 … n − p。
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


def fit_bspline(points, parameters, n_controls, degree=3, weights=None):
    """最小二乘 B 样条逼近：min Σ wᵢ |C(ūᵢ) − Pᵢ|²，控制点数为 n_controls。

    不强制通过首末点；需要时把首末点的权重设大。
    """
    u = np.asarray(parameters, dtype=float)
    knots = approximation_knots(u, n_controls, degree)
    spline = si.make_lsq_spline(u, np.asarray(points, dtype=float), knots, k=degree, w=weights)
    return BSpline(spline.c, degree, knots)
