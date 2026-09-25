"""五轴拐角（corner_blocks、hermite_transition）、进给包络与速度规划、进给修正插补。

参考：顶点距离用稠密采样 + KD 树独立量出；圆弧的法向加速度用解析式 v²/R；
机床轴约束用插补点上的解析导数检查；进给波动与 Taylor 插补对照。
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

import cnc5x as cx
from cnc5x import metrics

POINTS = np.array([[0, 0, 0], [10, 0, 0], [14, 6, 1], [20, 6, 0], [26, -2, 2]], float) * 3
AXES = cx.unit([[0, 0, 1], [0.3, 0, 1], [0.3, 0.4, 1], [-0.2, 0.3, 1], [0, 0, 1]])
LIMITS = cx.DriveLimits([60, 60, 60, 1.0, 2.0], [600, 600, 600, 5, 10], [6000, 6000, 6000, 50, 100])


def arc_length_jumps(curves, components):
    """相邻曲线段连接处，指定分量对刀尖弧长的 0..2 阶导数的最大跳变。"""
    jumps = np.zeros(3)
    for left, right in zip(curves[:-1], curves[1:]):
        a = left.derivatives_by_length(left.domain[1])[..., components]
        b = right.derivatives_by_length(right.domain[0])[..., components]
        jumps = np.maximum(jumps, np.abs(b[:3] - a[:3]).max(axis=-1))
    return jumps


@pytest.mark.parametrize("axes", [None, AXES])
def test_hermite_corner_path(axes):
    path = cx.HermiteCornerPath(POINTS, tolerance=0.05, chord_error=1e-3, axes=axes)
    distance, _ = cKDTree(path.sample(400001)[:, :3]).query(POINTS[1:-1])
    assert np.allclose(distance, 0.05, atol=1e-5)  # 采样间距约 3e-4，最近点误差为其平方量级
    assert arc_length_jumps(path.curves, slice(0, 3)).max() < 1e-12  # 刀尖 C²（对弧长）
    for curve in path.transitions:  # 曲率峰值在中点，即 block 的分界
        u = np.linspace(0, 1, 2001)
        assert np.argmax(curve.curvature(u)) == 1000
    if axes is not None:
        assert arc_length_jumps(path.curves, slice(3, 6)).max() < 1e-12  # 刀轴对刀尖弧长 C²
        o = path.sample(20001)[:, 3:]
        assert np.allclose(np.linalg.norm(o, axis=1), 1)


def test_corner_blocks_keep_orientation_on_straight_parts():
    path = cx.HermiteCornerPath(POINTS, tolerance=0.05, chord_error=1e-3, axes=AXES)
    middle = path.blocks[1].curves[1]  # 第二个 block 中间的 G01
    assert isinstance(middle, cx.PoseCurve) and isinstance(middle.orientation, cx.GreatCircle)
    # 刀轴在直线部分沿原 G01 的大圆转动：接点处的刀轴与原 G01 在该处的大圆插值一致
    seg = path.segment(1)
    fraction = path.trim[0] / path.L[1]
    assert np.allclose(middle.start_point[3:], seg(fraction)[3:], atol=1e-12)


def test_block_rejects_rotation_without_tip_motion():
    rotation = cx.PoseCurve(cx.Line([1, 2, 3], [1, 2, 3]), cx.GreatCircle([0, 0, 1], [0, 0.6, 0.8]))
    with pytest.raises(ValueError):
        cx.toolpath.Block([rotation])
    cx.toolpath.Block([cx.Line([1, 2], [1, 2]), cx.Line([1, 2], [3, 4])])  # 纯粹的零长度段照常去掉


def test_envelope_bounds_feed_on_full_circle():
    # 整圆只有一个 block，曲率处处相同，没有峰值：不给包络时速度会冲到 v_max
    w = np.sqrt(0.5)
    ring = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]]
    circle = cx.NURBS(np.array(ring) * 5.0, 2, [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], [1, w] * 4 + [1])
    path = cx.CurvePath(circle, chord_error=1e-3)
    Ts, V, A, J = 0.001, 100.0, 500.0, 20000.0
    s = np.linspace(0, path.length, 2000)
    envelope = (s, cx.feed_envelope(path, s, Ts, V, A, J, chord_error=1e-3))
    profile, _, _ = cx.schedule(path, V, A, J, Ts, envelope=envelope)
    feed, _, _ = metrics.tangential(cx.interpolate(path, profile, Ts).tip)
    assert np.nanmax(feed) ** 2 / 5.0 <= A * (1 + 1e-6)  # 法向加速度 v²/R ≤ A
    assert np.nanmax(feed) > 0.99 * np.sqrt(A * 5.0)  # 但也不保守：巡航速度就是 √(AR)
    profile, _, _ = cx.schedule(path, V, A, J, Ts)
    feed, _, _ = metrics.tangential(cx.interpolate(path, profile, Ts).tip)
    assert np.nanmax(feed) ** 2 / 5.0 > 3 * A  # 对照：不给包络时严重超限


def test_envelope_with_drive_limits():
    path = cx.HermiteCornerPath(POINTS, tolerance=0.05, chord_error=1e-3, axes=AXES)
    machine, Ts, V, A, J = cx.TableTilting("AC"), 0.001, 50, 500, 5000
    s = np.linspace(0, path.length, int(path.length / (V * Ts)) + 2)
    v_env = cx.feed_envelope(path, s, Ts, V, A, J, 1e-3, machine=machine, drives=LIMITS)
    profile, knots, v = cx.schedule(path, V, A, J, Ts, envelope=(s, v_env))
    assert len(knots) > len(path.blocks) + 1  # block 内部加了连接点
    assert cx.envelope_scale(profile, (s, v_env), Ts) <= 1 + 1e-9
    commands = cx.interpolate(path, profile, Ts, machine=machine)
    velocity = metrics.axis_report(commands.q, LIMITS)["velocity"]["ratio"]
    assert velocity.max() <= 1 + 1e-3  # 轴速度只与进给速度有关，包络保证它（容差来自样本间的线性插值）
    plain, _, _ = cx.schedule(path, V, A, J, Ts)
    unconstrained = metrics.axis_report(cx.interpolate(path, plain, Ts, machine=machine).q, LIMITS)
    assert unconstrained["velocity"]["ratio"].max() > 2  # 对照：只看刀尖时旋转轴严重超速


def test_correction_interpolation_beats_taylor():
    curve = cx.BSpline([[0, 0], [10, 25], [25, -10], [40, 30], [55, 0], [70, 20]], 3)
    profile = cx.seven_phase(curve.length, 0, 0, 100, 1000, 20000)
    aligned = cx.align_period(profile, 0.001)
    s = aligned(np.arange(round(aligned.duration / 0.001) + 1) * 0.001)[0]
    mapping = cx.feed_correction(curve)
    _, points = cx.correction_interpolate(curve, profile, 0.001, mapping)
    correction = np.nanmax(np.abs(metrics.feedrate_fluctuation(points, s)[20:-20]))
    _, points = cx.taylor_interpolate(curve, profile, 0.001, order=2)
    taylor = np.nanmax(np.abs(metrics.feedrate_fluctuation(points, s)[20:-20]))
    assert correction < 1e-4 and taylor > 10 * correction
