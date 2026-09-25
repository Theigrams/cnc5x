import numpy as np
import pytest
import sympy as sp

from cnc5x import (
    Bezier,
    DualCurveDirection,
    GreatCircle,
    Line,
    UnitDirection,
    chord_parameters,
    fit_bspline,
    interpolate_bspline,
)
from cnc5x.fitting import approximation_knots, averaged_knots


def test_great_circle():
    start, end = np.array([0.0, 0.0, 1.0]), np.array([0.6, 0.0, 0.8])
    arc = GreatCircle(start, end)
    u = np.linspace(0, 1, 11)
    assert np.allclose(np.linalg.norm(arc(u), axis=1), 1)
    assert np.allclose(arc(1.0), end)
    theta = np.arccos(0.8)
    # o(u) = (sin θu, 0, cos θu)，各阶导数可以直接写出
    assert np.allclose(arc(u, 1), theta * np.column_stack([np.cos(theta * u), 0 * u, -np.sin(theta * u)]))
    assert np.allclose(arc(u, 3), theta**3 * np.column_stack([-np.cos(theta * u), 0 * u, np.sin(theta * u)]))
    assert np.isclose(arc.length, theta)


def test_great_circle_rejects_opposite_directions():
    with pytest.raises(ValueError):
        GreatCircle([0, 0, 1], [0, 0, -1])


def test_unit_direction_matches_sympy():
    x = sp.symbols("x", real=True)
    r = sp.Matrix([1 + x, 2 * x**2, 1 - x])  # 与下面的二次 Bézier 相同
    o = r / sp.sqrt(r.dot(r))
    curve = UnitDirection(Bezier([[1, 0, 1], [1.5, 0, 0.5], [2, 2, 0]]))
    x0 = 0.3
    expected = np.array([[float(sp.diff(e, x, k).subs(x, x0)) for e in o] for k in range(4)])
    assert np.allclose(curve.derivatives(x0), expected, rtol=1e-12)


def test_dual_curve_direction_is_normalized_difference():
    tip = Line([0, 0, 0], [10, 0, 0])
    top = Bezier([[0, 0, 5], [5, 3, 5], [10, 0, 6]])
    difference = Bezier([[0, 0, 5], [0, 3, 5], [0, 0, 6]])  # top − tip：直线升阶为二次 Bézier 后逐个控制点相减
    u = np.linspace(0, 1, 7)
    assert np.allclose(DualCurveDirection(tip, top).derivatives(u), UnitDirection(difference).derivatives(u))


def test_chord_parameters():
    points = np.array([[0, 0], [3, 4], [3, 5], [3, 9]])
    assert np.allclose(chord_parameters(points), [0, 0.5, 0.6, 1.0])
    root5 = np.sqrt(5)
    assert np.allclose(chord_parameters(points, exponent=0.5), np.array([0, root5, root5 + 1, root5 + 3]) / (root5 + 3))
    with pytest.raises(ValueError):
        chord_parameters([[0, 0], [0, 0], [1, 1]])


def cubic_a(t):
    return np.column_stack([t**3 - t, 2 * t**2 + 1, np.ones_like(t)])


def cubic_b(t):
    return np.column_stack([t**3, 1 - t, t**2])


def test_interpolation_passes_through_points_and_reproduces_cubics():
    u = np.array([0.0, 0.1, 0.35, 0.5, 0.8, 0.9, 1.0])
    curve = interpolate_bspline(cubic_a(u), u, degree=3)
    assert len(curve.knots) == len(averaged_knots(u, 3)) == len(u) + 4
    assert np.allclose(curve(u), cubic_a(u))
    t = np.linspace(0, 1, 23)
    assert np.allclose(curve(t), cubic_a(t))  # 三次多项式在样条空间里，插值会精确还原


def test_least_squares_fit_reproduces_cubics():
    u = np.linspace(0, 1, 40)
    curve = fit_bspline(cubic_b(u), u, approximation_knots(u, 8, 3))
    assert len(curve.control_points) == 8
    t = np.linspace(0, 1, 17)
    assert np.allclose(curve(t), cubic_b(t))


def test_great_circle_length_is_analytic():
    arc = GreatCircle([0, 0, 1], [0.6, 0, 0.8])
    theta = np.arccos(0.8)
    assert arc.length == pytest.approx(theta, rel=1e-15)
    assert np.isclose(arc.u_at_length(theta / 4), 0.25) and np.isclose(arc.length_at(0.5), theta / 2)
    assert np.allclose(np.linalg.norm(arc(np.linspace(0, 1, 5), 1), axis=-1), theta)  # |o'| ≡ θ


def test_parameters_need_two_points():
    from cnc5x import angle_parameters

    with pytest.raises(ValueError):
        chord_parameters([[1, 2]])
    with pytest.raises(ValueError):
        angle_parameters([[0, 0, 1]])
