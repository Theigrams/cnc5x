import numpy as np
import pytest
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import brentq

from cnc5x import NURBS, Bezier, BSpline, Line, Reparameterized, SubCurve
from cnc5x.utils.tolerances import ARC_LENGTH

# 与 cnc_interpolation/tests/test_curve_bspline.py 相同的曲线
CONTROL_POINTS = np.array([[5.0, 5.0], [10.0, 10.0], [20.0, 15.0], [35.0, 15.0], [45.0, 10.0], [50.0, 5.0]])
KNOTS = [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0]


@pytest.fixture
def bspline():
    return BSpline(CONTROL_POINTS, 3, KNOTS)


def reference_length(curve, a, b):
    def sigma(x):
        return np.linalg.norm(curve(x, 1))

    return quad(sigma, a, b, points=[0.33, 0.66], epsabs=1e-13, epsrel=1e-13, limit=200)[0]


@pytest.mark.parametrize(
    "u, point",
    [(0.0, (5.0, 5.0)), (0.3, (18.617, 13.377)), (0.5, (27.645, 14.691)), (0.6, (32.143, 14.328)), (1.0, (50.0, 5.0))],
)
def test_bspline_points(bspline, u, point):
    assert np.allclose(bspline(u), point, atol=1e-3)


@pytest.mark.parametrize(
    "u, tangent",
    [
        (0.0, (45.45454545, 45.45454545)),
        (0.3, (45.26671675, 13.52366642)),
        (0.5, (45.02416338, -0.25500369)),
        (0.6, (44.93369634, -7.00602307)),
        (1.0, (44.11764706, -44.11764706)),
    ],
)
def test_bspline_first_derivative(bspline, u, tangent):
    assert np.allclose(bspline(u, 1), tangent, atol=1e-6)


def test_vectorized_shapes(bspline):
    assert bspline(0.2).shape == (2,)
    assert bspline(np.linspace(0, 1, 7)).shape == (7, 2)
    assert bspline(np.zeros((3, 4)), 2).shape == (3, 4, 2)
    assert bspline.derivatives(np.linspace(0, 1, 5)).shape == (4, 5, 2)
    assert np.allclose(bspline(0.4, 3), bspline.derivatives(0.4)[3])


def test_arc_length_table(bspline):
    assert np.isclose(bspline.length, reference_length(bspline, 0, 1), rtol=1e-12)
    for s in np.linspace(0.5, bspline.length - 0.5, 7):
        u_exact = brentq(lambda x: reference_length(bspline, 0, x) - s, 0, 1, xtol=1e-15)
        # 弧长表的约定：每一段的误差（换算成长度）不超过 tolerances.ARC_LENGTH
        assert abs(bspline.u_at_length(s) - u_exact) * bspline.parametric_speed(u_exact) < ARC_LENGTH
        assert abs(bspline.length_at(u_exact) - s) < ARC_LENGTH


def test_equal_length_samples(bspline):
    points = bspline.sample(21, by_length=True)
    chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.ptp(chords) / chords.mean() < 0.01  # 等弧长取点，弦长几乎相同


def test_parameter_outside_domain_raises(bspline):
    with pytest.raises(ValueError):
        bspline(1.1)
    with pytest.raises(ValueError):
        bspline(np.nan)


def test_domain_is_not_normalized():
    curve = BSpline(CONTROL_POINTS, 3, np.array(KNOTS) * 10 + 2)
    assert curve.domain == (2.0, 12.0)
    assert np.allclose(curve(12.0), CONTROL_POINTS[-1])


def test_line():
    line = Line([0, 0, 0], [3, 4, 0])
    assert line.length == 5.0
    assert np.allclose(line(0.5), [1.5, 2, 0])
    assert np.allclose(line(np.array([0.1, 0.9]), 1), [[3, 4, 0], [3, 4, 0]])
    assert np.allclose(line(0.3, 2), 0)
    assert line.u_at_length(2.5) == 0.5


def test_bspline_split(bspline):
    left, right = bspline.split(0.5)
    assert left.domain == (0.0, 0.5) and right.domain == (0.5, 1.0)
    u = np.linspace(0, 0.5, 9)
    assert np.allclose(left(u), bspline(u)) and np.allclose(left(u, 2), bspline(u, 2))
    u = np.linspace(0.5, 1, 9)
    assert np.allclose(right(u), bspline(u)) and np.allclose(right(u, 1), bspline(u, 1))


def test_bezier_split():
    curve = Bezier([[0, 0], [1, 2], [3, 3], [4, 0]])
    left, right = curve.split(0.3)
    assert np.allclose(left(np.linspace(0, 1, 5)), curve(0.3 * np.linspace(0, 1, 5)))
    assert np.allclose(right(np.linspace(0, 1, 5)), curve(0.3 + 0.7 * np.linspace(0, 1, 5)))


def test_nurbs_quarter_circle():
    w = np.sqrt(0.5)
    circle = NURBS([[1, 0], [1, 1], [0, 1]], 2, [0, 0, 0, 1, 1, 1], [1, w, 1])
    u = np.linspace(0, 1, 11)
    assert np.allclose(np.linalg.norm(circle(u), axis=1), 1)
    assert np.allclose(circle.curvature(u), 1)
    assert np.isclose(circle.length, np.pi / 2, rtol=1e-12)


def test_nurbs_full_circle_with_repeated_knots():
    # 节点重数等于次数（节点处只有 C⁰ 的齐次样条），高阶导数仍要能求
    w = np.sqrt(0.5)
    circle = NURBS(
        [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]],
        degree=2,
        knots=[0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1],
        weights=[1, w, 1, w, 1, w, 1, w, 1],
    )
    u = np.linspace(0, 1, 101)
    assert np.allclose(np.linalg.norm(circle(u), axis=1), 1)
    assert np.allclose(circle.curvature(u), 1)
    assert np.isclose(circle.length, 2 * np.pi, rtol=1e-12)
    assert np.all(np.isfinite(circle.derivatives(u)))


def test_nurbs_third_derivative_matches_sympy():
    # 二次有理 Bézier：C(u) = Σ Bᵢ wᵢ Pᵢ / Σ Bᵢ wᵢ
    x = sp.symbols("x", real=True)
    P = [(0, 0), (1, 2), (3, 1)]
    w = [1, 3, sp.Rational(1, 2)]
    B = [(1 - x) ** 2, 2 * x * (1 - x), x**2]
    denominator = sum(b * wi for b, wi in zip(B, w))
    C = [sum(b * wi * p[k] for b, wi, p in zip(B, w, P)) / denominator for k in range(2)]
    curve = NURBS(P, 2, [0, 0, 0, 1, 1, 1], [float(wi) for wi in w])
    x0 = 0.35
    expected = np.array([[float(sp.diff(c, x, k).subs(x, x0)) for c in C] for k in range(4)])
    assert np.allclose(curve.derivatives(x0), expected, rtol=1e-11)


def test_subcurve(bspline):
    sub = bspline.restrict(0.2, 0.7)
    assert isinstance(sub, SubCurve)
    assert np.isclose(sub.length, reference_length(bspline, 0.2, 0.7), rtol=1e-12)
    assert np.isclose(sub.u_at_length(0.0), 0.2) and np.isclose(sub.u_at_length(sub.length), 0.7)
    assert np.allclose(sub.breaks, [0.2, 0.33, 0.66, 0.7])


def test_reparameterized_matches_sympy():
    x = sp.symbols("x", real=True)
    curve = Bezier([[0, 0], [1, 3], [4, 2]])
    mapping = Bezier([[0.0], [0.2], [0.9], [1.0]])  # 单调的一维映射
    B2 = [(1 - x) ** 2, 2 * x * (1 - x), x**2]
    B3 = [(1 - x) ** 3, 3 * x * (1 - x) ** 2, 3 * x**2 * (1 - x), x**3]
    g = sum(b * c for b, c in zip(B3, [0, sp.Rational(1, 5), sp.Rational(9, 10), 1]))
    C = [sum(b * p[k] for b, p in zip(B2, [(0, 0), (1, 3), (4, 2)])).subs(x, g) for k in range(2)]
    x0 = 0.45
    expected = np.array([[float(sp.diff(c, x, k).subs(x, x0)) for c in C] for k in range(4)])
    assert np.allclose(Reparameterized(curve, mapping).derivatives(x0), expected, rtol=1e-12)


def test_arc_length_table_on_uneven_spline():
    # 点距相差上千倍的插值样条：速度在 [~1, ~1e3] 间剧烈变化，固定分段的弧长表在这里会差好几毫米
    x = np.cumsum(np.r_[0, np.geomspace(0.01, 30, 12), np.geomspace(30, 0.01, 12)])
    points = np.column_stack([x, np.sin(x / 7) * 5])
    from cnc5x import chord_parameters, interpolate_bspline

    curve = interpolate_bspline(points, chord_parameters(points))
    rng = np.random.default_rng(3)
    for u in np.sort(rng.uniform(0, 1, 6)):
        inner = curve.breaks[(curve.breaks > 0) & (curve.breaks < u)]
        exact = quad(lambda t: curve.parametric_speed(t), 0, u, points=inner, limit=500, epsabs=1e-12, epsrel=1e-13)[0]
        assert abs(curve.length_at(u) - exact) < 10 * ARC_LENGTH  # 各段积分误差会累加，但远小于逐段容差之和
        assert abs(curve.u_at_length(exact) - u) * curve.parametric_speed(u) < 10 * ARC_LENGTH  # 反函数误差换算成长度


class CountingBSpline(BSpline):
    calls = 0

    def _derivatives(self, u, order):
        CountingBSpline.calls += 1
        return super()._derivatives(u, order)


def test_composite_curves_evaluate_components_once():
    # 导数栈一次求出：五轴刀位曲线求一次导数栈，刀尖、刀轴两条 B 样条各只求值一次
    from cnc5x import PoseCurve, UnitDirection

    tip = CountingBSpline(CONTROL_POINTS[:, [0, 1, 1]] * [1, 1, 0], 3, KNOTS)
    axis = CountingBSpline(np.c_[np.linspace(0, 0.3, 6), np.zeros(6), np.ones(6)], 3, KNOTS)
    pose = PoseCurve(tip, UnitDirection(axis))
    CountingBSpline.calls = 0
    pose.derivatives(np.linspace(0, 1, 50))
    assert CountingBSpline.calls == 2
    CountingBSpline.calls = 0
    pose.derivatives_by_length(np.linspace(0, 1, 50))  # 弧长的导数从同一个导数栈里算
    assert CountingBSpline.calls == 2


def test_arc_length_outside_range_raises(bspline):
    line = Line([0, 0], [3, 4])
    for curve in (bspline, line):
        assert np.isclose(curve.u_at_length(curve.length * (1 + 1e-12)), curve.domain[1])  # 舍入误差以内截回
        with pytest.raises(ValueError, match="arc length"):
            curve.u_at_length(curve.length + 1)
        with pytest.raises(ValueError, match="arc length"):
            curve.u_at_length(-0.5)


def test_zero_length_curve_has_no_arc_length_parameter():
    from cnc5x import GreatCircle

    for curve in (Line([1, 2, 3], [1, 2, 3]), GreatCircle([0, 0, 1], [0, 0, 1])):
        assert curve.length == 0
        with pytest.raises(ValueError, match="zero length"):
            curve.u_at_length(0.0)


def test_derivatives_by_length_on_circle():
    # 半径 R 的圆按弧长的解析导数：C_s = T，C_ss = κN = −C/R²，C_sss = −κ²T（κ = 1/R）
    R, w = 2.5, np.sqrt(0.5)
    circle = NURBS(R * np.array([[1, 0], [1, 1], [0, 1]]), 2, [0, 0, 0, 1, 1, 1], [1, w, 1])
    d = circle.derivatives_by_length(np.linspace(0, 1, 11))
    C, T = d[0], d[1]
    assert np.allclose(np.linalg.norm(T, axis=-1), 1, atol=1e-12)
    assert np.allclose(np.sum(C * T, axis=-1), 0, atol=1e-12)  # 切向垂直于半径
    assert np.allclose(d[2], -C / R**2, atol=1e-12)
    assert np.allclose(d[3], -T / R**2, atol=1e-12)


def test_pose_curve_piece_measures_only_one_component():
    from cnc5x import Curve, GreatCircle, PoseCurve

    tip = BSpline(CONTROL_POINTS[:, [0, 1, 1]] * [1, 1, 0], 3, KNOTS)
    axis = GreatCircle([0, 0, 1], [0, 0.6, 0.8])
    assert not isinstance(PoseCurve(tip, axis), Curve)  # 6 维 [p, o] 不是空间中的点，组合而非继承
    u = np.linspace(0.2, 0.7, 9)
    for along, length in (("tip", reference_length(tip, 0.2, 0.7)), ("axis", 0.5 * axis.theta)):
        piece = PoseCurve(tip, axis, along).restrict(0.2, 0.7)
        assert np.isclose(piece.length, length, rtol=1e-12)
        d = piece.derivatives_by_length(u)
        measured = d[1, :, :3] if along == "tip" else d[1, :, 3:]
        assert np.allclose(np.linalg.norm(measured, axis=-1), 1, atol=1e-12)  # 对计量分量的弧长求导，模长为 1
