import numpy as np
import pytest
import sympy as sp

from cnc5x import calculus

u = sp.symbols("u", real=True)


def sympy_stack(expr, u0):
    """sympy 表达式（标量或向量）在 u0 处的 0..3 阶导数栈。"""
    expr = sp.Matrix(expr) if isinstance(expr, (list, sp.Matrix)) else sp.Matrix([expr])
    return np.array([[float(sp.diff(e, u, k).subs(u, u0)) for e in expr] for k in range(4)])


def test_compose_matches_sympy():
    x = u**2 + sp.Rational(3, 10) * u
    X = sp.symbols("X")
    f = [sp.sin(X), X**3]
    u0 = 0.7
    x0 = float(x.subs(u, u0))
    outer = np.array([[float(sp.diff(e, X, k).subs(X, x0)) for e in f] for k in range(4)])
    x_stack = sympy_stack(x, u0)[:, 0]
    result = calculus.compose(outer, x_stack[1], x_stack[2], x_stack[3])
    expected = sympy_stack([e.subs(X, x) for e in f], u0)
    assert np.allclose(result, expected, rtol=1e-12)


def test_product_matches_sympy():
    f, g = u**2 + 1, sp.sin(u)
    u0 = 0.4
    result = calculus.product(sympy_stack(f, u0), sympy_stack(g, u0))
    assert np.allclose(result, sympy_stack(f * g, u0), rtol=1e-12)


def test_unit_derivatives_matches_sympy():
    r = sp.Matrix([1, u, u**2])
    u0 = 0.3
    result = calculus.unit_derivatives(sympy_stack(r, u0))
    assert np.allclose(result, sympy_stack(r / sp.sqrt(r.dot(r)), u0), rtol=1e-12)


def test_speed_derivatives_are_arc_length_derivatives():
    C = sp.Matrix([u, u**2, u**3])
    u0 = 0.6
    speed = sp.sqrt(C.diff(u).dot(C.diff(u)))
    s1, s2, s3 = calculus.speed_derivatives(sympy_stack(C, u0))
    expected = [float(sp.diff(speed, u, k).subs(u, u0)) for k in range(3)]
    assert np.allclose([s1, s2, s3], expected, rtol=1e-12)


def test_inverse_derivatives_of_exponential():
    # s(u) = e^u，反函数 u(s) = ln s：u' = 1/s，u'' = −1/s²，u''' = 2/s³
    s = np.exp(0.8)
    u1, u2, u3 = calculus.inverse_derivatives(s, s, s)
    assert np.allclose([u1, u2, u3], [1 / s, -1 / s**2, 2 / s**3], rtol=1e-14)


def test_acos_and_arg_derivatives_match_sympy():
    raw = sp.Matrix([sp.Rational(3, 10) + u + u**2 / 3, sp.Rational(1, 2) - u**2 + u**3 / 5, 1 + u / 4])
    o = raw / sp.sqrt(raw.dot(raw))
    u0 = 0.37
    stack = sympy_stack(o, u0)
    tilt = calculus.acos_derivatives(stack[:, 2])
    c = calculus.arg_derivatives(stack[:, 1] + 1j * stack[:, 0])
    assert np.allclose(tilt, sympy_stack(sp.acos(o[2]), u0)[:, 0], rtol=1e-12)
    assert np.allclose(c, sympy_stack(sp.atan2(o[0], o[1]), u0)[:, 0], rtol=1e-12)


def test_unit_derivatives_rejects_zero_vector():
    with pytest.raises(ValueError):
        calculus.unit_derivatives(np.zeros((4, 3)))
