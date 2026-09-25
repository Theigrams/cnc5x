"""评价指标：路径偏差、拐角误差、弓高误差、进给波动、切向量、轴峰值、连接处的几何连续性。"""

import numpy as np
from scipy.spatial import cKDTree

from .calculus import compose, inverse_derivatives, speed_derivatives


def path_deviation(points, vertices):
    """每个点到折线（有限线段组成）的最近距离，points (M, D)，vertices (N, D)，返回 (M,)。"""
    points = np.asarray(points, dtype=float)
    a = np.asarray(vertices, dtype=float)[:-1]  # (N−1, D) 各线段起点
    ab = np.diff(np.asarray(vertices, dtype=float), axis=0)  # (N−1, D) 各线段方向
    result = np.empty(len(points))
    for start in range(0, len(points), 2048):  # 分块，避免 M×N 的中间数组太大
        p = points[start : start + 2048, None, :]  # (m, 1, D)
        t = np.clip(np.sum((p - a) * ab, axis=-1) / np.sum(ab * ab, axis=-1), 0.0, 1.0)  # (m, N−1)
        result[start : start + 2048] = np.linalg.norm(p - a - t[..., None] * ab, axis=-1).min(axis=1)
    return result


def corner_error(vertices, points):
    """每个 G01 顶点到插补点的最近距离 (N,)，衡量拐角被“切”掉多少（受采样间隔影响）。"""
    distance, _ = cKDTree(points).query(vertices)
    return distance


def chord_error(path, s):
    """相邻两个插补点之间的弓高误差 (N−1,)：弧长中点处的刀尖到这两点连线的距离。"""
    s = np.asarray(s, dtype=float)
    p = _tip(path.derivatives(s)[0])
    mid = _tip(path.derivatives((s[:-1] + s[1:]) / 2)[0])
    a, ab = p[:-1], np.diff(p, axis=0)
    length2 = np.sum(ab * ab, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(length2 > 0, np.sum((mid - a) * ab, axis=-1) / length2, 0.0)
    return np.linalg.norm(mid - a - np.clip(t, 0.0, 1.0)[:, None] * ab, axis=-1)


def feedrate_fluctuation(positions, s):
    """进给波动 (N−1,)：实际步长（相邻点距离）相对指令步长 Δs 的偏差 (|ΔP| − Δs)/Δs。

    弦长与弧长之差是 O(κ²Δs³) 的小量，这里忽略。指令步长为零处给 nan。
    """
    actual = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    command = np.diff(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(command > 0, (actual - command) / command, np.nan)


def tangential(d):
    """由时间导数栈 (4, N, dim) 求切向速度、切向加速度、切向 jerk（各 (N,)）。

    刀尖栈得到进给速度等；刀轴栈（单位向量）得到刀轴角速度 ω = |ȯ| 及其导数。速度为零处为 nan。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return speed_derivatives(d)


def axis_report(q, limits):
    """各轴速度、加速度、jerk 的峰值 (n_axes,) 与超限比 = 峰值 / 上限。q 为时间导数栈 (4, N, n_axes)。"""
    report = {}
    for k, name, bound in (
        (1, "velocity", limits.velocity),
        (2, "acceleration", limits.acceleration),
        (3, "jerk", limits.jerk),
    ):
        peak = np.max(np.abs(q[k]), axis=0)
        report[name] = {"peak": peak, "ratio": peak / bound}
    return report


def junction_jumps(curves):
    """相邻曲线段连接处的跳变 (n−1, 3)：位置、单位切向、曲率向量（对弧长的 0、1、2 阶导数）。

    三列都为零即 G² 连续。
    """
    jumps = []
    for left, right in zip(curves[:-1], curves[1:]):
        a = _arc_length_derivatives(left, left.domain[1])
        b = _arc_length_derivatives(right, right.domain[0])
        jumps.append([np.linalg.norm(_tip(b[k] - a[k])) for k in range(3)])
    return np.array(jumps)


def rms(values):
    """均方根。"""
    return float(np.sqrt(np.mean(np.square(values))))


def _arc_length_derivatives(curve, u):
    """曲线在参数 u 处对弧长 s 的导数栈 [C, C_s, C_ss, C_sss]。"""
    s1, s2, s3 = curve.arc_derivatives(u)
    u1, u2, u3 = inverse_derivatives(s1, s2, s3)
    return compose(curve.derivatives(u), u1, u2, u3)


def _tip(x):
    """五轴 6 维 [p, o] 只取刀尖 p；其余原样返回。"""
    return x[..., :3] if x.shape[-1] == 6 else x
