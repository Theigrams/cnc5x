"""速度规划：连接点限速 → 双向扫描 → 每个 block 一段 S 曲线 → 拼成整条进给轮廓。

给出进给包络（沿弧长采样的进给上限，见 feed_envelope）时，block 内部也要满足它（见 schedule）。
"""

import numpy as np

from ..utils import tolerances
from .limits import drive_limit, geometric_limit
from .look_ahead import bidirectional_scan
from .profiles import concatenate, five_phase, seven_phase


def feed_envelope(path, s, Ts, v_max, a_max, j_max, chord_error, machine=None, drives=None):
    """沿全局弧长 s 采样的进给上限 v_lim(s)，形状同 s；是匀速通过每一点时的上限。

    取以下各项的最小值：
        v_max；
        刀尖曲率 κ = |p_ss| 给出的弓高误差、法向加速度、法向 jerk 限速（limits.geometric_limit）；
        给出 machine 与 drives 时，各轴速度、加速度、jerk 的限速（limits.drive_limit）。
    机床轴对 s 的导数由 machine.axis_motion 从刀位对 s 的导数算出：换元公式与对时间时相同。
    包络只含匀速的部分，切向加减速带来的轴加速度 q_s·a、轴 jerk 3q_ss·v·a + q_s·j 不在其中，
    插补后用 limits.time_scale_factor 检查。s 的间距不要大于一个插补周期走过的长度 v_max·Ts。
    """
    d = path.derivatives(s)
    tip = d[..., :3] if d.shape[-1] == 6 else d
    kappa = np.linalg.norm(tip[2], axis=-1)  # 对弧长参数化的曲线，曲率就是 |C_ss|
    v = np.minimum(v_max, geometric_limit(kappa, chord_error, Ts, a_max, j_max))
    if drives is not None:
        q = machine.axis_motion(d[..., :3], d[..., 3:])
        v = np.minimum(v, drive_limit(q, drives))
    return v


def schedule(path, v_max, a_max, j_max, Ts, phases=7, envelope=None, split_rounds=10):
    """为刀路规划进给轮廓，返回 (profile, 连接点的弧长位置 s, 连接点速度 v)，后两者形状 (n + 1,)。

    连接点起初是 block 的分界，速度上限由 path.get_v_limit 给出。phases = 7 用七段 S 曲线，
    phases = 5 用五段 S 曲线（Zhao et al. 2013）。

    envelope = (s_env, v_env)：沿弧长采样的进给上限（见 feed_envelope），样本之间线性插值。
    给出时 block 内部也要满足它，分三步：
    1. 连接点的上限再与包络取小；每个 block 的巡航上限取块内包络的最大值（不再是 v_max），
       包络是平台时（例如整圆的法向加速度限速）这一步就够了；
    2. 按插补周期采样检查，每个超出包络的 block 在 v/v_env 最大处加一个连接点（上限取包络值），
       重新规划，最多 split_rounds 轮；
    3. 仍然超出的，整体放慢 λ = max(v/v_env) 倍（见 envelope_scale）。
    第 2 步可能收敛得慢，这是 block 结构本身的局限：S 曲线在连接点处加速度为零，
    贴不住大段缓慢变化的包络（五轴的各轴约束常常如此），只能靠加连接点一级一级逼近。
    所有结论只对样本成立。
    """
    if phases not in (5, 7):
        raise ValueError("phases 只能是 5 或 7")
    plan = seven_phase if phases == 7 else five_phase
    knots = np.concatenate([[0.0], np.cumsum(path.lengths)])
    limit = np.minimum(path.get_v_limit(Ts, v_max, a_max, j_max), v_max)
    if envelope is not None:
        limit = np.minimum(limit, np.interp(knots, *envelope))
    for round_ in range(split_rounds + 1):
        v = bidirectional_scan(np.diff(knots), limit, a_max, j_max, phases)
        caps = np.full(len(knots) - 1, v_max) if envelope is None else _block_caps(knots, envelope, v_max)
        profiles = []
        for i, length in enumerate(np.diff(knots)):
            profiles.append(plan(length, v[i], v[i + 1], caps[i], a_max, j_max))
        profile = concatenate(profiles)
        if envelope is None:
            return profile, knots, v
        extra = _worst_violations(profile, knots, envelope, Ts) if round_ < split_rounds else []
        if len(extra) == 0:
            break
        order = np.argsort(np.concatenate([knots, extra]))
        knots = np.concatenate([knots, extra])[order]
        limit = np.concatenate([limit, np.interp(extra, *envelope)])[order]
    scale = envelope_scale(profile, envelope, Ts)
    return profile.scaled(scale), knots, v / scale


def envelope_scale(profile, envelope, Ts):
    """进给轮廓超出包络的最大倍数 λ = max(v / v_env)，至少为 1；在每个插补周期的样本上计算。

    整体放慢 λ 倍（Profile.scaled）后，速度处处除以 λ、走过的位置不变，样本上就不再超出包络。
    """
    ratio = _envelope_ratio(profile, envelope, Ts)[2]
    return float(max(1.0, np.max(ratio)))


def _envelope_ratio(profile, envelope, Ts):
    """每个插补周期的样本：弧长 s、进给 v、超出比 v / v_env。"""
    s, v = profile(np.arange(0.0, profile.duration, Ts))[:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = v / np.interp(s, *envelope)
    return s, v, np.nan_to_num(ratio, nan=0.0)


def _block_caps(knots, envelope, v_max):
    """每个 block 的巡航上限：块内包络样本与两端插值的最大值，再与 v_max 取小。"""
    s_env, v_env = envelope
    ends = np.interp(knots, s_env, v_env)
    first = np.searchsorted(s_env, knots[:-1], side="right")
    last = np.searchsorted(s_env, knots[1:], side="left")
    caps = np.maximum(ends[:-1], ends[1:])
    for i, (a, b) in enumerate(zip(first, last)):
        if a < b:
            caps[i] = max(caps[i], v_env[a:b].max())
    return np.minimum(caps, v_max)


def _worst_violations(profile, knots, envelope, Ts):
    """每个超出包络的 block 里 v / v_env 最大的样本位置，作为新的连接点。

    离已有连接点不到一个插补周期行程（v·Ts）的样本不加：更密的连接点按周期采样已分辨不出，
    太短的 block 还会让速度的舍入误差超过它能容纳的速度变化。
    """
    s, v, ratio = _envelope_ratio(profile, envelope, Ts)
    over = ratio > 1 + tolerances.ROUNDING
    block = np.searchsorted(knots, s, side="right") - 1
    extra = []
    for b in np.unique(block[over]):
        inside = np.flatnonzero(over & (block == b))
        k = inside[np.argmax(ratio[inside])]
        if min(s[k] - knots[b], knots[b + 1] - s[k]) > v[k] * Ts:
            extra.append(s[k])
    return np.array(extra)
