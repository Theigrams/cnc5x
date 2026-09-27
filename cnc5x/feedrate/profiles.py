"""进给轮廓 s(t)：分段恒 jerk 的 S 曲线。

插补器对进给轮廓只要求三样东西：
    profile(t) → (4, ...)，即 [s, v, a, j]，与全库的导数栈同一形状（s 对 t 的 0..3 阶导数）
    profile.duration      总时长
    profile.length        总路程
"""

import numpy as np
from scipy.optimize import brentq

from ..utils import tolerances


def advance(state, jerk, T):
    """从 state = [s, v, a] 出发，以恒定 jerk 运动 T 秒后的 [s, v, a]。"""
    s, v, a = state
    return np.array([s + v * T + a * T**2 / 2 + jerk * T**3 / 6, v + a * T + jerk * T**2 / 2, a + jerk * T])


class Profile:
    """分段恒 jerk 的进给轮廓。

    durations (m,) 为各阶段时长，jerks (m,) 为各阶段 jerk，从 [s, v, a] = [0, v0, a0] 出发。
    阶段内经过 τ 秒：s = s₀ + v₀τ + a₀τ²/2 + jτ³/6，v = v₀ + a₀τ + jτ²/2，a = a₀ + jτ。
    """

    def __init__(self, durations, jerks, v0=0.0, a0=0.0):
        durations = np.asarray(durations, dtype=float)
        jerks = np.asarray(jerks, dtype=float)
        keep = durations > 0  # 去掉零时长阶段，免得时间节点重复
        if not np.any(keep):
            raise ValueError("至少需要一个时长为正的阶段")
        self.durations, self.jerks = durations[keep], jerks[keep]
        self.times = np.concatenate([[0.0], np.cumsum(self.durations)])
        states = [np.array([0.0, v0, a0])]
        for T, j in zip(self.durations, self.jerks):
            states.append(advance(states[-1], j, T))
        self.states = np.array(states)  # (m + 1, 3)，每个阶段起点的 [s, v, a]
        self.duration = float(self.times[-1])
        self.length = float(self.states[-1, 0])

    def __call__(self, t):
        """进给状态 [s, v, a, j]，形状 (4, ...)，也就是 s(t) 的导数栈。"""
        t = np.clip(np.asarray(t, dtype=float), 0.0, self.duration)
        k = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.durations) - 1)
        tau = t - self.times[k]
        s0, v0, a0 = self.states[k, 0], self.states[k, 1], self.states[k, 2]
        j = self.jerks[k]
        s = s0 + v0 * tau + a0 * tau**2 / 2 + j * tau**3 / 6
        v = v0 + a0 * tau + j * tau**2 / 2
        a = a0 + j * tau
        return np.stack([s, v, a, j])

    def scaled(self, factor):
        """整体放慢 λ 倍：时长 ×λ，速度 ÷λ，加速度 ÷λ²，jerk ÷λ³，路程不变。"""
        v0, a0 = self.states[0, 1], self.states[0, 2]
        return Profile(self.durations * factor, self.jerks / factor**3, v0 / factor, a0 / factor**2)

    def __repr__(self):
        return f"Profile(duration={self.duration:.6g} s, length={self.length:.6g}, phases={len(self.durations)})"


def transition(v_start, v_end, a_max, j_max):
    """端点加速度为零、从 v_start 变到 v_end 的最快过渡：加加速、匀加速、减加速三段。

    Δv ≤ A²/J 时没有匀加速段（加速度是三角形，峰值 √(JΔv)）；否则峰值为 A。
    返回三段的时长和 jerk。速度曲线关于中点对称，平均速度恰为 (v_start + v_end)/2，
    因此过渡距离 = (v_start + v_end)/2 × 总时长。
    """
    dv = v_end - v_start
    if dv == 0:
        return np.zeros(3), np.zeros(3)
    Tj = min(a_max / j_max, np.sqrt(abs(dv) / j_max))
    Ta = abs(dv) / (j_max * Tj) - Tj
    return np.array([Tj, Ta, Tj]), np.sign(dv) * np.array([j_max, 0.0, -j_max])


def transition_distance(v_start, v_end, a_max, j_max):
    """transition 走过的距离。"""
    durations, _ = transition(v_start, v_end, a_max, j_max)
    return (v_start + v_end) / 2 * durations.sum()


def seven_phase(length, v_start, v_end, v_max, a_max, j_max):
    """七段 S 曲线：加速到峰值 v_p、匀速、再减速到 v_end，在限制内用时最短。

    两段过渡所需距离 D(v_p) 随 v_p 单调增加：D(v_max) ≤ length 时用匀速段补足剩余距离；
    否则解 D(v_p) = length 求峰值速度。
    """
    low = max(v_start, v_end)

    def distance(peak):
        return transition_distance(v_start, peak, a_max, j_max) + transition_distance(peak, v_end, a_max, j_max)

    if distance(low) > length * (1 + tolerances.ROUNDING):
        raise ValueError(
            f"长度不够：从 v_start = {v_start:.6g} 变到 v_end = {v_end:.6g} 至少要走 {distance(low):.6g}，"
            f"只有 {length:.6g}（前瞻 bidirectional_scan 会保证相邻速度来得及衔接）"
        )
    if distance(v_max) <= length:
        peak = v_max
    elif distance(low) >= length:
        peak = low  # 只够完成一次过渡（差距在舍入以内）
    else:
        peak = brentq(lambda v: distance(v) - length, low, v_max, xtol=tolerances.ROOT_XTOL)
    up, up_jerks = transition(v_start, peak, a_max, j_max)
    down, down_jerks = transition(peak, v_end, a_max, j_max)
    cruise = max(length - distance(peak), 0.0) / peak
    durations = np.concatenate([up, [cruise], down])
    jerks = np.concatenate([up_jerks, [0.0], down_jerks])
    return Profile(durations, jerks, v0=v_start)


def five_phase(length, v_start, v_end, v_max, a_max, j_max):
    """五段 S 曲线（Lin et al. 2007；Zhao et al. 2013 采用）：没有匀加速段。

    过渡的峰值加速度是 √(JΔv)；为使它不超过 A，峰值速度不超过 min(v_start, v_end) + A²/J。
    前提：|v_end − v_start| ≤ A²/J。否则没有匀加速段就无法在 a_max 以内完成从 v_start 到 v_end
    的过渡，这时报错（用 seven_phase，或先经 bidirectional_scan(..., phases=5) 把相邻速度差截到 A²/J）。
    """
    if abs(v_end - v_start) > a_max**2 / j_max * (1 + tolerances.ROUNDING):
        raise ValueError(
            f"|v_end − v_start| = {abs(v_end - v_start):.6g} 超过 a_max²/j_max = {a_max**2 / j_max:.6g}："
            "五段 S 曲线没有匀加速段，峰值加速度会超过 a_max"
        )
    cap = min(v_max, min(v_start, v_end) + a_max**2 / j_max)
    return seven_phase(length, v_start, v_end, max(cap, v_start, v_end), np.inf, j_max)


def concatenate(profiles):
    """把各 block 的轮廓首尾相接成一条（接点处速度相同、加速度都为零）。"""
    durations = np.concatenate([p.durations for p in profiles])
    jerks = np.concatenate([p.jerks for p in profiles])
    v0, a0 = profiles[0].states[0, 1], profiles[0].states[0, 2]
    return Profile(durations, jerks, v0, a0)


def align_period(profile, Ts):
    """把总时长向上取整为 Ts 的整数倍（整体放慢 λ = nTs/T ≥ 1），保证每个插补周期等长。"""
    n = max(1, int(np.ceil(profile.duration / Ts - tolerances.ROUNDING)))
    return profile.scaled(n * Ts / profile.duration)
