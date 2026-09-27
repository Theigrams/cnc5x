"""速度上限与各轴约束。

几何限速（前瞻时用于 block 连接点）：
    弓高误差    v ≤ (2/Ts)·√(2ρδ − δ²)，ρ = 1/κ           Yeh & Hsu 2002
    法向加速度  v ≤ √(A/κ)；法向 jerk  v ≤ (J/κ²)^(1/3)     Lai et al. 2008

各轴约束（五轴）：q 对弧长 s 的导数记为 q_s、q_ss、q_sss，进给 v = ṡ、a = s̈、j = s⃛，
    q̇ = q_s v，  q̈ = q_ss v² + q_s a，  q⃛ = q_sss v³ + 3 q_ss v a + q_s j。
"""

from dataclasses import dataclass

import numpy as np


def chord_error_limit(curvature, chord_error, Ts):
    """弓高误差限速：一个插补周期走过的弦长 c = vTs 所对应的弓高不超过 δ。

    半径 ρ = 1/κ 的圆上，弦长 c 与弓高 δ 的关系为 (c/2)² = 2ρδ − δ²，所以
        v = c/Ts = (2/Ts)·√(2ρδ − δ²)，  κδ ≤ 1；
        v = 2ρ/Ts，                     κδ > 1（饱和）；
        v = ∞，                         κ = 0（直线上没有弓高误差）。
    第一式只在 δ ≤ ρ 时成立：弦长随 δ 增大而增大，δ = ρ 时弦长达到直径 2ρ。δ > ρ 时公式里的
    δ 已经是优弧一侧的"弓高"，弦长反而随 δ 变小，δ ≥ 2ρ 时甚至给出 v = 0，让刀具在急弯处停下。
    弦长不可能超过直径，容差再放宽也只能达到 c = 2ρ，所以饱和在 2ρ/Ts。这种急弯处法向加速度的
    限速通常小得多（见 curvature_limit），饱和值一般不起作用，它只是不再让速度无谓地降到零。
    curvature (...)、单位 1/mm；chord_error 单位 mm（可与 curvature 广播）；返回 mm/s。
    """
    kappa = np.asarray(curvature, dtype=float)
    x = np.minimum(kappa * chord_error, 1.0)  # κδ，饱和于 1
    with np.errstate(divide="ignore", invalid="ignore"):
        half_chord = np.sqrt(2 * x - x**2) / kappa  # (c/2)² = (2κδ − κ²δ²)/κ² = 2ρδ − δ²；x = 1 时即 ρ
    return np.where(kappa > 0, 2.0 * half_chord / Ts, np.inf)


def curvature_limit(curvature, a_max, j_max):
    """匀速过弯时的法向加速度 v²κ ≤ A 与法向 jerk v³κ² ≤ J 给出的速度上限。

    法向 jerk 只保留 v³κ²，忽略了曲率变化率 κ_s 与挠率的贡献。
    """
    kappa = np.asarray(curvature, dtype=float)
    with np.errstate(divide="ignore"):
        return np.minimum(np.sqrt(a_max / kappa), np.cbrt(j_max / kappa**2))


def geometric_limit(curvature, chord_error, Ts, a_max, j_max):
    """弓高误差、法向加速度、法向 jerk 三者给出的速度上限取小。"""
    return np.minimum(chord_error_limit(curvature, chord_error, Ts), curvature_limit(curvature, a_max, j_max))


@dataclass
class DriveLimits:
    """各轴速度、加速度、jerk 上限，形状 (n_axes,)。直线轴用 mm/s…，旋转轴用 rad/s…。"""

    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray

    def __post_init__(self):
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.acceleration = np.asarray(self.acceleration, dtype=float)
        self.jerk = np.asarray(self.jerk, dtype=float)
        for limit in (self.velocity, self.acceleration, self.jerk):
            if limit.shape != self.velocity.shape or np.any(limit <= 0):
                raise ValueError("三组上限必须是形状相同的正数数组")


def drive_limit(dq_ds, limits):
    """匀速通过时各轴约束给出的进给上限：|q_s|v ≤ V，|q_ss|v² ≤ A，|q_sss|v³ ≤ J。

    dq_ds: (4, ..., n_axes)，q 对弧长 s 的导数栈（不是对时间）；返回 (...)。
    """
    q1, q2, q3 = np.abs(dq_ds[1]), np.abs(dq_ds[2]), np.abs(dq_ds[3])
    with np.errstate(divide="ignore"):
        v1 = np.min(limits.velocity / q1, axis=-1)
        v2 = np.min(np.sqrt(limits.acceleration / q2), axis=-1)
        v3 = np.min(np.cbrt(limits.jerk / q3), axis=-1)
    return np.minimum(np.minimum(v1, v2), v3)


def acceleration_interval(dq_ds, v, limits):
    """给定进给速度 v，切向加速度 a 的可行区间 [lo, hi]：各轴 |q_ss v² + q_s a| ≤ A。无解处为 nan。

    逐周期调度（例如 Beudaert et al. 2012）每一步要知道 a 能取多大，就用它；库内的 schedule 不用。
    """
    v = np.asarray(v, dtype=float)[..., None]
    return _affine_interval(dq_ds[1], dq_ds[2] * v**2, limits.acceleration)


def jerk_interval(dq_ds, v, a, limits):
    """给定 v、a，切向 jerk j 的可行区间 [lo, hi]：各轴 |q_sss v³ + 3 q_ss v a + q_s j| ≤ J。无解处为 nan。"""
    v = np.asarray(v, dtype=float)[..., None]
    a = np.asarray(a, dtype=float)[..., None]
    return _affine_interval(dq_ds[1], dq_ds[3] * v**3 + 3 * dq_ds[2] * v * a, limits.jerk)


def time_scale_factor(dq_dt, limits):
    """整体放慢 λ 倍（t → λt）使各轴都不超限所需的最小 λ ≥ 1。

    放慢后 q̇、q̈、q⃛ 分别变为 1/λ、1/λ²、1/λ³ 倍，所以 λ = max(1, r_v, √r_a, ∛r_j)，
    r 是最大超限比。dq_dt: (4, N, n_axes) 为 q 对时间的导数栈（插补结果 Commands.q）；
    结论只对这些样本成立。
    """
    r_v = np.max(np.abs(dq_dt[1]) / limits.velocity)
    r_a = np.max(np.abs(dq_dt[2]) / limits.acceleration)
    r_j = np.max(np.abs(dq_dt[3]) / limits.jerk)
    return float(max(1.0, r_v, np.sqrt(r_a), np.cbrt(r_j)))


def _affine_interval(slope, offset, bound):
    """所有轴 |slope·x + offset| ≤ bound 的公共区间（沿最后一维求交）。无解处为 nan。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        x1 = (-bound - offset) / slope
        x2 = (bound - offset) / slope
    active = slope != 0
    lo = np.max(np.where(active, np.minimum(x1, x2), -np.inf), axis=-1)
    hi = np.min(np.where(active, np.maximum(x1, x2), np.inf), axis=-1)
    feasible = np.all(active | (np.abs(offset) <= bound), axis=-1) & (lo <= hi)
    return np.where(feasible, lo, np.nan), np.where(feasible, hi, np.nan)
