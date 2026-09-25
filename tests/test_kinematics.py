import mpmath as mp
import numpy as np
import pytest
import sympy as sp

from cnc5x import GreatCircle, TableTilting

x = sp.symbols("x", real=True)


def stack(expr, x0):
    return np.array([[float(sp.diff(e, x, k).subs(x, x0)) for e in expr] for k in range(4)])


@pytest.mark.parametrize("kind", ["AC", "BC"])
def test_physical_anchors(kind):
    machine = TableTilting(kind)
    p, o = machine.forward([0, 0, 0, 0, 0])
    assert np.allclose(o, [0, 0, 1])
    _, o = machine.forward([0, 0, 0, 0, 1.2])  # 只转 C，机床 +Z 在工件系中不变
    assert np.allclose(o, [0, 0, 1])
    _, o = machine.forward([0, 0, 0, np.pi / 2, 0])  # 只倾转 90°
    assert np.allclose(o, [0, 1, 0] if kind == "AC" else [-1, 0, 0])


@pytest.mark.parametrize("kind", ["AC", "BC"])
@pytest.mark.parametrize("branch", [1, -1])
def test_inverse_then_forward(kind, branch):
    rng = np.random.default_rng(0)
    p = rng.normal(size=(50, 3)) * 20
    o = rng.normal(size=(50, 3))
    o[:, 2] = np.abs(o[:, 2]) + 0.1
    o /= np.linalg.norm(o, axis=1, keepdims=True)
    machine = TableTilting(kind, offset=[1.0, -2.0, 30.0])
    q = machine.inverse(p, o, branch)
    assert np.all(np.sign(q[:, 3]) == branch)
    p_back, o_back = machine.forward(q)
    assert np.allclose(p_back, p) and np.allclose(o_back, o)


@pytest.mark.parametrize("kind", ["AC", "BC"])
@pytest.mark.parametrize("branch", [1, -1])
def test_axis_motion_matches_high_precision_derivatives(kind, branch):
    """参考值：把 q(x) 直接写成复合函数，用 mpmath 以 40 位精度数值求导。"""

    def q_of(t):
        p = mp.matrix([10 + 3 * t, -5 + t**2, 2 * t**3 - t])
        raw = [mp.mpf(3) / 10 + t + t**2 / 3, mp.mpf(1) / 2 - t**2 + t**3 / 5, 1 + t / 4]
        norm = mp.sqrt(sum(r**2 for r in raw))
        o = [r / norm for r in raw]
        tilt = branch * mp.acos(o[2])
        c = mp.atan2(o[0], o[1]) if kind == "AC" else mp.atan2(o[1], -o[0])
        if branch < 0:
            c += mp.pi
        R = mp_rotation("x" if kind == "AC" else "y", tilt) * mp_rotation("z", c)
        xyz = R * p + mp.matrix([1, 2, 3])
        return [xyz[0], xyz[1], xyz[2], tilt, c]

    x0 = 0.37
    with mp.workdps(40):
        expected = np.array([[float(mp.diff(lambda t: q_of(t)[i], x0, k)) for i in range(5)] for k in range(4)])
    p = sp.Matrix([10 + 3 * x, -5 + x**2, 2 * x**3 - x])
    raw = sp.Matrix([sp.Rational(3, 10) + x + x**2 / 3, sp.Rational(1, 2) - x**2 + x**3 / 5, 1 + x / 4])
    o = raw / sp.sqrt(raw.dot(raw))
    machine = TableTilting(kind, offset=[1, 2, 3])
    q = machine.axis_motion(stack(p, x0)[:, None, :], stack(o, x0)[:, None, :], branch)
    assert np.allclose(q[:, 0, :], expected, rtol=1e-11)


def mp_rotation(axis, a):
    c, s = mp.cos(a), mp.sin(a)
    if axis == "x":
        return mp.matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return mp.matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return mp.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_c_is_unwrapped_across_pi():
    a = 0.4
    C = np.linspace(2.9, 3.6, 30)  # 越过 π
    o = np.column_stack([np.sin(a) * np.sin(C), np.sin(a) * np.cos(C), np.full_like(C, np.cos(a))])
    q = TableTilting("AC").inverse_path(np.zeros((30, 3)), o)
    assert np.allclose(q[:, 4], C)
    q = TableTilting("AC").inverse_path(np.zeros((30, 3)), o, c_start=C[0] + 2 * np.pi)
    assert np.allclose(q[:, 4], C + 2 * np.pi)


def test_pole_takes_neighbouring_values():
    # 刀轴从竖直（极点）出发沿大圆倾斜：C 恒为 π/2，倾角 θu 的导数恒为 θ
    arc = GreatCircle([0, 0, 1], [0.6, 0, 0.8])
    u = np.linspace(0, 1, 11)
    axis = arc.derivatives(u)
    tip = np.zeros_like(axis)
    q = TableTilting("AC").axis_motion(tip, axis)
    theta = np.arccos(0.8)
    assert np.allclose(q[0, :, 4], np.pi / 2)
    assert np.allclose(q[1, :, 3], theta)
    assert np.all(np.isfinite(q))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("kind", ["AC", "BC"])
def test_vertical_tool_axis_everywhere(kind):
    # 平面轮廓配竖直刀轴：每个样本都在极点上。以前 axis_motion 在这里抛 IndexError
    import cnc5x as cx

    points = cx.datasets.load_dataset("rhombic").points
    points = np.column_stack([points, np.zeros(len(points))])
    path = cx.LinearPath(points, np.tile([0.0, 0.0, 1.0], (len(points), 1)))
    profile, _, _ = cx.schedule(path, 50, 500, 5000, 0.001)
    commands = cx.interpolate(path, profile, 0.001, machine=TableTilting(kind))
    assert np.all(np.isfinite(commands.q))
    assert np.allclose(commands.q[:, :, :3], commands.tip)  # 转台不动：机床坐标就是刀尖坐标
    assert np.all(commands.q[:, :, 3:] == 0)
