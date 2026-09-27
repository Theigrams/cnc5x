"""评价指标：路径偏差、拐角误差、弓高误差、进给波动、切向速度（及加速度、jerk）、轴峰值、连接处的连续性、
五轴非线性误差。"""

import numpy as np
from scipy.spatial import cKDTree

from .utils.calculus import speed_derivatives
from .utils.geometry import angle_between


def path_deviation(points, vertices):
    """每个点到折线（有限线段组成）的最近距离，points (M, D)，vertices (N, D)，返回 (M,)。

    与 corner_error 方向相反：这里是"插补点离折线多远"，那里是"折线顶点离插补点多远"。
    折线可以有重复顶点（read_cl 不去重）：零长度的线段退化为一个点，投影参数取 0。
    """
    points = np.asarray(points, dtype=float)
    a = np.asarray(vertices, dtype=float)[:-1]  # (N−1, D) 各线段起点
    ab = np.diff(np.asarray(vertices, dtype=float), axis=0)  # (N−1, D) 各线段方向
    length2 = np.sum(ab * ab, axis=-1)
    result = np.empty(len(points))
    for start in range(0, len(points), 2048):  # 分块，避免 M×N 的中间数组太大
        p = points[start : start + 2048, None, :]  # (m, 1, D)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(length2 > 0, np.sum((p - a) * ab, axis=-1) / length2, 0.0)  # (m, N−1)
        t = np.clip(t, 0.0, 1.0)
        result[start : start + 2048] = np.linalg.norm(p - a - t[..., None] * ab, axis=-1).min(axis=1)
    return result


def corner_error(vertices, points):
    """每个 G01 顶点到插补点的最近距离 (N,)，衡量拐角被“切”掉多少（受采样间隔影响）。

    与 path_deviation 方向相反，参数顺序也相反：先顶点、后插补点。
    """
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


def nonlinear_error(machine, path, s, q, fractions=(0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)):
    """五轴非线性误差（nonlinear error）：相邻插补点之间机床轴按直线插值时，刀位偏离目标刀路多少。

    伺服层在两个插补点之间对各轴线性插值：q(f) = q_k + f·(q_{k+1} − q_k)，f ∈ [0, 1]。
    正解 p(f)、o(f) 后，转动轴让刀尖沿弧线而不是直线走，这就是非线性误差，量级约 R·Δθ²/8
    （R 为刀尖到转轴的距离，Δθ 为一个周期的转角）。
    目标点取刀路上离 p(f) 最近的点：从同进度 s = s_k + f·Δs_k 出发沿单位切向 T 投影一步，
    s* = s + (p − C(s))·T。同进度配对里有进给不匀带来的切向偏差（量级 a·Ts²，与非线性误差相当），
    投影把它去掉，剩下的误差是 O(κe²)。
    s：各插补点的刀尖弧长 (N,)（Commands.feed[0]），刀路须按刀尖计量；q：机床轴 (N, 5)（Commands.q[0]）。
    返回 (刀尖距离 (N−1,) mm, 刀轴夹角 (N−1,) rad)，是每个周期在 fractions 上的最大值，只在这些样本上检查。
    """
    f = np.asarray(fractions, dtype=float)
    q_lin = q[:-1, None] + f[:, None] * np.diff(q, axis=0)[:, None]  # (N−1, F, 5)
    p, o = machine.forward(q_lin)
    s_same = np.clip(s[:-1, None] + f * np.diff(s)[:, None], 0.0, path.length)  # 同进度的弧长 (N−1, F)
    d = path.derivatives(s_same)
    step = np.sum((p - d[0, ..., :3]) * d[1, ..., :3], axis=-1)
    target = path.derivatives(np.clip(s_same + step, 0.0, path.length))[0]
    tip = np.linalg.norm(p - target[..., :3], axis=-1).max(axis=1)
    return tip, angle_between(o, target[..., 3:]).max(axis=1)


def junction_jumps(curves):
    """相邻曲线段连接处的跳变：对弧长的 0、1、2 阶导数在两侧之差的模。

    三轴返回 (n−1, 3)：位置、单位切向、曲率向量的跳变，三列都为零即 G² 连续。
    五轴（6 维刀位曲线）返回 (n−1, 6)：前三列同上，是刀尖的；后三列是刀轴 o、o_s、o_ss 的跳变，
    导数也是对刀尖弧长求的（单位 1、rad/mm、rad/mm²），全为零即刀轴对刀尖弧长 C² 连续。
    刀尖与刀轴分开列出，不合成一个数：毫米和弧度不能相加。
    """
    jumps = []
    for left, right in zip(curves[:-1], curves[1:]):
        jump = right.derivatives_by_length(right.domain[0])[:3] - left.derivatives_by_length(left.domain[1])[:3]
        row = np.linalg.norm(_tip(jump), axis=-1)
        if jump.shape[-1] == 6:
            row = np.concatenate([row, np.linalg.norm(jump[..., 3:], axis=-1)])
        jumps.append(row)
    return np.array(jumps)


def _tip(x):
    """五轴 6 维 [p, o] 只取刀尖 p；其余原样返回。"""
    return x[..., :3] if x.shape[-1] == 6 else x
