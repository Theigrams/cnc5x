"""插补：按插补周期 Ts 对进给轮廓采样，得到每个周期的刀尖位置（五轴还有刀轴与机床轴）。

    t_k = k·Ts → [s, v, a, j] = profile(t_k) → 刀路在 s 处对 s 的导数 → 链式法则得到对 t 的导数

所有周期一次性向量化求值，没有逐周期的循环（taylor_interpolate 除外，它本身就是递推）。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .feedrate.profiles import align_period
from .utils.calculus import compose


@dataclass
class Commands:
    """每个插补周期一个样本。导数栈 [x, ẋ, ẍ, x⃛] 的形状为 (4, N, dim)。"""

    t: np.ndarray  # (N,) 时间
    feed: np.ndarray  # (4, N) 进给状态 [s, v, a, j]：沿刀尖弧长（或刀轴转角）的位置、速度、加速度、jerk
    tip: np.ndarray  # (4, N, D) 刀尖位置及其时间导数（工件坐标）
    axis: Optional[np.ndarray] = None  # (4, N, 3) 刀轴方向及其时间导数（五轴）
    q: Optional[np.ndarray] = None  # (4, N, 5) 机床轴坐标及其时间导数（给了机床模型时）
    scale: float = 1.0  # 为使总时长是 Ts 的整数倍而整体放慢的倍数 λ

    @property
    def position(self):
        return self.tip[0]

    @property
    def orientation(self):
        return None if self.axis is None else self.axis[0]


def interpolate(path, profile, Ts, machine=None, branch=None):
    """按周期 Ts 插补整条刀路。

    进给轮廓先整体放慢到总时长是 Ts 的整数倍（λ 略大于 1，记在 Commands.scale）。
    路径求值为 6 维时按五轴 [刀尖, 刀轴] 处理；再给出 machine 时同时算机床轴。
    branch 缺省时由机床按倾转轴行程选择（没有行程时取正分支），见 TableTilting.choose_branch。
    """
    aligned = align_period(profile, Ts)
    n = int(round(aligned.duration / Ts))
    t = np.arange(n + 1) * Ts
    feed = aligned(t)  # (4, N)
    s = np.clip(feed[0], 0.0, path.length)
    d = compose(path.derivatives(s), feed[1], feed[2], feed[3])  # (4, N, dim)
    scale = aligned.duration / profile.duration
    if d.shape[-1] != 6:
        return Commands(t, feed, d, scale=scale)
    tip, axis = d[..., :3], d[..., 3:]
    q = None if machine is None else machine.axis_motion(tip, axis, branch)
    return Commands(t, feed, tip, axis, q, scale)


def taylor_interpolate(curve, profile, Ts, order=2):
    """经典参数插补：每个周期用 Taylor 展开推进参数 u，不做弧长反算。

        u̇ = v / |C'|，  ü = a / |C'| − v² (C'·C'') / |C'|⁴
        u_{k+1} = u_k + u̇ Ts + ü Ts²/2（order = 1 时只保留前两项）

    返回 (u, 位置)。截断误差会让实际进给偏离指令，可用 metrics.feedrate_fluctuation 查看。
    """
    aligned = align_period(profile, Ts)
    n = int(round(aligned.duration / Ts))
    feed = aligned(np.arange(n + 1) * Ts)
    lo, hi = curve.domain
    u = np.empty(n + 1)
    u[0] = lo
    for k in range(n):
        v, a = feed[1, k], feed[2, k]
        d = curve.derivatives(u[k], order)  # 一阶展开只要 C'，不必求 C''
        speed = np.linalg.norm(d[1])
        step = v * Ts / speed
        if order == 2:
            step += Ts**2 / 2 * (a / speed - v**2 * np.dot(d[1], d[2]) / speed**4)
        u[k + 1] = min(u[k] + step, hi)
    return u, curve(u)


def correction_interpolate(curve, profile, Ts, mapping):
    """进给修正多项式插补（Erkorkmaz & Altintas 2001；Yuen et al. 2013）：u_k = ũ(s_k)。

    mapping 是 fitting.feed_correction 拟合出的 s → u 映射。每个周期直接代入多项式，
    不查弧长表、也不递推，所以没有 Taylor 插补那样的累积误差；进给波动只来自拟合误差
    |σ(ũ)·ũ_s − 1|（拟合时已控制在容差以内）。返回 (u, 位置)。
    """
    aligned = align_period(profile, Ts)
    n = int(round(aligned.duration / Ts))
    s = np.clip(aligned(np.arange(n + 1) * Ts)[0], 0.0, mapping.domain[1])
    u = mapping(s)[:, 0]
    return u, curve(u)
