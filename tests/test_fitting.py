"""fitting：插值、最小二乘、Hermite、单调插值、进给修正、球坐标刀轴。

参考值都独立于被测代码：SciPy 自带的样条拟合（另一套实现）、SLSQP 数值优化、
cnc5 中已审阅的五次 Hermite 闭式公式、SymPy 符号求导、scipy.integrate.quad 求弧长。
"""

import numpy as np
import pytest
import sympy as sp
from scipy import interpolate as si
from scipy.integrate import quad
from scipy.optimize import minimize

import cnc5x as cx
from cnc5x import fitting


@pytest.mark.parametrize("degree", [3, 5])
def test_interpolation_matches_scipy(degree):
    rng = np.random.default_rng(7)
    u = np.sort(np.r_[0, rng.uniform(0, 1, 18), 1])
    Q = rng.normal(size=(20, 3))
    curve = fitting.interpolate_bspline(Q, u, degree)
    reference = si.make_interp_spline(u, Q, k=degree, t=fitting.averaged_knots(u, degree))
    x = np.linspace(0, 1, 101)
    assert np.abs(curve(x) - reference(x)).max() < 1e-10
    assert np.abs(curve(u) - Q).max() < 1e-12


def test_weighted_least_squares_matches_scipy():
    rng = np.random.default_rng(7)
    u = np.linspace(0, 1, 60)
    Q = np.column_stack([np.sin(3 * u), np.cos(2 * u)]) + 0.01 * rng.normal(size=(60, 2))
    w = rng.uniform(0.5, 2, 60)
    knots = fitting.approximation_knots(u, 12, 3)
    curve = fitting.fit_bspline(Q, u, knots, weights=w)
    # SciPy 的权重乘在残差上：min Σ (wᵢ rᵢ)²，所以传入 √w
    reference = si.make_lsq_spline(u, Q, knots, k=3, w=np.sqrt(w))
    x = np.linspace(0, 1, 101)
    assert np.abs(curve(x) - reference(x)).max() < 1e-12


def test_constrained_least_squares_matches_slsqp():
    rng = np.random.default_rng(7)
    u = np.linspace(0, 1, 40)
    Q = np.column_stack([np.sin(3 * u), u**2]) + 0.01 * rng.normal(size=(40, 2))
    knots = fitting.approximation_knots(u, 9, 3)
    constraints = [(0.0, 0, Q[0]), (1.0, 0, Q[-1]), (0.0, 1, [3.0, 0.0])]
    curve = fitting.fit_bspline(Q, u, knots, constraints=constraints)
    N = fitting.basis_matrix(knots, 3, u)
    A = np.vstack([fitting.basis_matrix(knots, 3, [x], k) for x, k, _ in constraints])
    b = np.array([np.asarray(D, dtype=float) for *_, D in constraints])
    n = N.shape[1]
    result = minimize(
        lambda p: np.sum((N @ p.reshape(n, 2) - Q) ** 2),
        np.zeros(2 * n),
        constraints={"type": "eq", "fun": lambda p: (A @ p.reshape(n, 2) - b).ravel()},
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    assert np.abs(result.x.reshape(n, 2) - curve.control_points).max() < 1e-6
    assert np.abs(curve(0.0) - Q[0]).max() < 1e-12 and np.abs(curve(0.0, 1) - [3, 0]).max() < 1e-10


def test_redundant_constraints_are_rejected_with_reason():
    u = np.linspace(0, 1, 20)
    Q = np.column_stack([u, u**2])
    knots = fitting.approximation_knots(u, 5, 3)
    duplicated = [(0.0, 0, Q[0]), (0.0, 0, Q[0])]
    too_many = [(x, 0, [x, x**2]) for x in np.linspace(0, 1, 6)]  # 6 条约束，只有 5 个控制点
    for constraints in (duplicated, too_many):
        with pytest.raises(ValueError, match="线性相关"):
            fitting.fit_bspline(Q, u, knots, constraints=constraints)


def test_conditions_outside_the_parameter_domain_are_rejected():
    knots = np.r_[[0.0] * 6, [1.0] * 6]
    conditions = [(0, 1, [1, 0]), (0, 2, [0, 0]), (1, 2, [0, 0]), (1.5, 1, [0, 1])]  # 最后一个在 [0, 1] 之外
    with pytest.raises(ValueError, match="参数域"):
        fitting.interpolate_bspline([[0, 0], [1, 1]], [0, 1], 5, knots=knots, derivatives=conditions)
    u = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="参数域"):
        fitting.fit_bspline(np.c_[u, u], u, fitting.approximation_knots(u, 6, 3), constraints=[(-0.5, 0, [0, 0])])


def test_quintic_hermite_matches_closed_form():
    rng = np.random.default_rng(7)
    p0, p1, v0, v1, a0, a1 = rng.normal(size=(6, 3))
    curve = fitting.hermite([p0, v0, a0], [p1, v1, a1])
    b1, b4 = p0 + v0 / 5, p1 - v1 / 5
    expected = [p0, b1, a0 / 20 + 2 * b1 - p0, a1 / 20 + 2 * b4 - p1, b4, p1]  # cnc5/smoothing.py
    assert np.allclose(curve.control_points, expected, atol=1e-14)


def test_septic_hermite_end_derivatives():
    rng = np.random.default_rng(7)
    start, end = rng.normal(size=(2, 4, 2))
    curve = fitting.hermite(start, end)
    assert curve.degree == 7
    assert np.allclose(curve.derivatives(0.0), start, atol=1e-12)
    assert np.allclose(curve.derivatives(1.0), end, atol=1e-12)


def test_interpolation_with_derivative_conditions_is_hermite():
    rng = np.random.default_rng(7)
    p0, p1, v0, v1, a0, a1 = rng.normal(size=(6, 3))
    knots = np.r_[[0.0] * 6, [1.0] * 6]
    conditions = [(0, 1, v0), (1, 1, v1), (0, 2, a0), (1, 2, a1)]
    curve = fitting.interpolate_bspline([p0, p1], [0, 1], 5, knots=knots, derivatives=conditions)
    assert np.allclose(curve.control_points, fitting.hermite([p0, v0, a0], [p1, v1, a1]).control_points)


def test_monotone_interpolation():
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(0, 10, 15))
    y = np.cumsum(rng.uniform(0.01, 3, 15) ** 3)  # 斜率相差几个数量级，普通样条会过冲
    g = fitting.monotone_interpolate(x, y)
    t = np.linspace(x[0], x[-1], 20001)
    assert np.abs(g(x)[:, 0] - y).max() < 1e-12
    assert g(t, 1).min() > 0
    h = 1e-7
    for k in (1, 2):  # 节点处 C²：两侧相距 2h 的差不超过 2h·max|g⁽ᵏ⁺¹⁾|（若有跳变，差会是 O(1)）
        bound = 2 * h * np.abs(g(t, k + 1)).max()
        assert np.abs(g(x[1:-1] - h, k) - g(x[1:-1] + h, k)).max() <= bound
    line = fitting.monotone_interpolate(x, 2 * x + 1)
    assert np.abs(line(t)[:, 0] - (2 * t + 1)).max() < 1e-12
    ordinary = fitting.interpolate_bspline(y[:, None], x)
    assert ordinary(t, 1).min() < 0  # 作为对照：普通插值样条在这组数据上不单调


def test_feed_correction():
    curve = cx.BSpline([[0, 0], [10, 25], [25, -10], [40, 30], [55, 0], [70, 20]], 3)
    mapping = fitting.feed_correction(curve, degree=9, continuity=3, tolerance=1e-6)
    s = np.linspace(0, curve.length, 41)[1:-1]
    u = mapping(s)[:, 0]
    reference = [quad(lambda x: curve.parametric_speed(x), 0, ui, epsabs=1e-13, limit=200)[0] for ui in u]
    assert np.abs(reference - s).max() < 1e-5  # 弧长误差 = ∫ 进给误差 ds，远小于 1e-6·弧长
    x = np.linspace(0, curve.length, 20001)
    assert np.abs(curve.parametric_speed(mapping(x)[:, 0]) * mapping(x, 1)[:, 0] - 1).max() < 1e-6
    h = 1e-6
    inner = mapping.breaks[1:-1]
    spline = si.BSpline(mapping.knots, mapping.control_points[:, 0], mapping.degree)  # 直接用 SciPy 求 4 阶导数
    for k in range(4):  # 段间 C³：k 阶导数两侧相距 2h 之差不超过 2h·max|ũ⁽ᵏ⁺¹⁾|，有跳变时差是 O(1)
        bound = 2 * h * np.abs(spline(x, nu=k + 1)).max()
        assert np.abs(spline(inner - h, nu=k) - spline(inner + h, nu=k)).max() <= bound


def test_spherical_curve_matches_sympy():
    x = sp.symbols("x", real=True)
    theta, phi = 0.5 + 0.3 * x + 0.2 * x**2, 1.0 - 0.7 * x + 0.4 * x**3
    o = [sp.sin(theta) * sp.cos(phi), sp.sin(theta) * sp.sin(phi), sp.cos(theta)]
    u = np.linspace(0, 1, 50)
    data = np.column_stack([0.5 + 0.3 * u + 0.2 * u**2, 1.0 - 0.7 * u + 0.4 * u**3])
    angles = fitting.fit_bspline(data, u, np.r_[[0.0] * 4, [1.0] * 4])  # 三次多项式在一段三次 Bézier 里精确
    curve = cx.SphericalCurve(angles)
    x0 = 0.37
    expected = np.array([[float(sp.diff(e, x, k).subs(x, x0)) for e in o] for k in range(4)])
    assert np.allclose(curve.derivatives(x0), expected, rtol=1e-11, atol=1e-12)


def test_spherical_spline_hits_data_and_rejects_poles():
    data = cx.datasets.load_dataset("horseshoe_planar_sweep")
    w = fitting.angle_parameters(data.axes)
    curve = fitting.spherical_spline(data.axes, w)
    assert np.abs(curve(w) - data.axes).max() < 1e-12
    with pytest.raises(ValueError):
        fitting.spherical_spline([[0, 0, 1], [0.1, 0, 1], [0.2, 0, 1], [0.3, 0, 1]], np.linspace(0, 1, 4))


def test_pose_spline_with_synchronized_orientation():
    data = cx.datasets.load_dataset("semicircle_arch_normal")
    u = fitting.chord_parameters(data.points)
    w = fitting.angle_parameters(data.axes)
    pose = cx.pose_spline(data.points, data.axes, orientation_parameters=w)
    assert np.abs(pose(u)[:, 3:] - data.axes).max() < 1e-12  # 刀位点处刀轴不变
    t = np.linspace(0, 1, 2001)
    assert np.allclose(np.linalg.norm(pose(t)[:, 3:], axis=1), 1)
