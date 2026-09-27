import numpy as np

from cnc5x import (
    BSpline,
    CurvePath,
    GreatCircle,
    Line,
    LinearPath,
    PoseCurve,
    TableTilting,
    ToolPath,
    align_period,
    interpolate,
    metrics,
    pose_spline,
    schedule,
    seven_phase,
    taylor_interpolate,
)
from cnc5x.toolpath import Block
from cnc5x.utils.calculus import compose

CURVE = BSpline([[0, 0], [10, 25], [25, -10], [40, 30], [55, 0], [70, 20]], 3)


class SingleCurve(ToolPath):
    """整条曲线一个 block 的最简刀路。"""

    def __init__(self, curve):
        self.blocks = [Block([curve])]

    def get_v_limit(self, Ts, v_max, a_max, j_max):
        return np.array([0.0, 0.0])


def test_straight_line_feed():
    path = LinearPath([[0, 0, 0], [30, 40, 0]])
    profile, _, _ = schedule(path, v_max=20, a_max=200, j_max=4000, Ts=0.001)
    commands = interpolate(path, profile, Ts=0.001)
    assert np.allclose(np.diff(commands.t), 0.001)
    assert np.allclose(commands.position[0], [0, 0, 0]) and np.allclose(commands.position[-1], [30, 40, 0])
    direction = np.array([0.6, 0.8, 0.0])
    for k in range(4):  # 直线上刀尖的各阶导数 = 进给的各阶导数 × 方向
        assert np.allclose(commands.tip[k], commands.feed[k][:, None] * direction, atol=1e-9)
    assert 1.0 <= commands.scale < 1.001


def test_tangential_motion_equals_feed_profile():
    # 对弧长求导后 |C_s| = 1、C_s·C_ss = 0 严格成立，所以刀尖的切向量就是进给轮廓
    path = CurvePath(CURVE, chord_error=1e-3)
    profile, _, _ = schedule(path, v_max=100, a_max=1000, j_max=20000, Ts=0.001)
    commands = interpolate(path, profile, Ts=0.001)
    feed, acceleration, jerk = metrics.tangential(commands.tip)
    moving = commands.feed[1] > 1e-3
    assert np.allclose(feed, commands.feed[1], atol=1e-9)
    assert np.allclose(acceleration[moving], commands.feed[2][moving], atol=1e-6)
    assert np.allclose(jerk[moving], commands.feed[3][moving], rtol=1e-6, atol=1e-3)
    assert np.nanmax(np.abs(acceleration)) <= 1000 * (1 + 1e-9)
    assert metrics.chord_error(path, commands.feed[0]).max() <= 1e-3


def test_arc_length_position_is_accurate():
    path = SingleCurve(CURVE)
    s = np.linspace(0, path.length, 37)
    points = path.derivatives(s)[0]
    u = CURVE.u_at_length(s)
    assert np.allclose(points, CURVE(u))
    assert np.allclose(CURVE.length_at(u), s, atol=1e-9)


def test_five_axis_forward_kinematics_and_derivatives():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 1, 9)
    points = np.column_stack([60 * t, 20 * np.sin(3 * t), 5 * t**2])
    axes = np.column_stack([0.3 * np.sin(2 * t) + 0.05, 0.2 * t + 0.1, np.ones_like(t)]) + rng.normal(0, 0.01, (9, 3))
    path = CurvePath(pose_spline(points, axes), chord_error=1e-3)
    machine = TableTilting("BC", offset=[0, 0, 20])
    profile, _, _ = schedule(path, v_max=40, a_max=400, j_max=4000, Ts=0.002)
    commands = interpolate(path, profile, Ts=0.002, machine=machine)
    p, o = machine.forward(commands.q[0])
    assert np.allclose(p, commands.position, atol=1e-9) and np.allclose(o, commands.orientation, atol=1e-12)

    # 在同一条流水线上取 t ± h 三个时刻，用中心差分检查机床轴导数
    aligned = align_period(profile, 0.002)
    h = 1e-5
    for t0 in commands.t[[37, 150, 300]]:
        feed = aligned(np.array([t0 - h, t0, t0 + h]))
        d = compose(path.derivatives(feed[0]), feed[1], feed[2], feed[3])
        q = machine.axis_motion(d[..., :3], d[..., 3:])
        for k in range(3):
            numeric = (q[k][2] - q[k][0]) / (2 * h)
            assert np.allclose(numeric, q[k + 1][1], rtol=1e-4, atol=1e-6 * np.abs(q[k + 1][1]).max() + 1e-6)


def test_pure_orientation_move():
    # 刀尖不动、只转刀轴：按刀轴转角计进给（rad/s）
    tip = Line([5.0, 5.0, 0.0], [5.0, 5.0, 0.0])
    pose = PoseCurve(tip, GreatCircle([0, 0, 1], [0, 0.6, 0.8]), along="axis")
    path = SingleCurve(pose)
    profile = seven_phase(path.length, 0, 0, 1.0, 10.0, 100.0)
    commands = interpolate(path, profile, Ts=0.001, machine=TableTilting("AC"))
    assert np.allclose(commands.position, [5, 5, 0])
    omega, _, _ = metrics.tangential(commands.axis)
    assert np.allclose(omega, commands.feed[1], atol=1e-9)  # 刀轴角速度就是进给速度
    assert np.isclose(commands.q[0, -1, 3], np.arccos(0.8))


def test_taylor_interpolation_fluctuation():
    profile = seven_phase(CURVE.length, 0, 0, 100, 1000, 20000)
    aligned = align_period(profile, 0.001)
    s = aligned(np.arange(round(aligned.duration / 0.001) + 1) * 0.001)[0]  # 指令弧长
    errors = []
    for order in (1, 2):
        u, points = taylor_interpolate(CURVE, profile, Ts=0.001, order=order)
        assert np.all(np.diff(u) >= -1e-12)  # 末周期速度≈0 时截断误差可能让步长略小于零
        fluctuation = metrics.feedrate_fluctuation(points, s)
        errors.append(np.nanmax(np.abs(fluctuation[20:-20])))
    assert errors[1] < 2e-3 and errors[0] > 10 * errors[1]  # 二阶展开的进给波动比一阶小一个数量级以上
