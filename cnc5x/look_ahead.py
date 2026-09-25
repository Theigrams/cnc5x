"""前瞻：双向扫描，保证相邻连接点的速度能在 block 长度内衔接上（Lin et al. 2007；Zhao et al. 2013）。"""

import numpy as np
from scipy.optimize import brentq

from .profiles import transition_distance


def reachable_velocity(v0, length, a_max, j_max, phases=7):
    """从 v0（加速度为零）出发、走过 length 后能达到的最高速度（终点加速度也为零）。

    解 D(v) = length，D 是 profiles.transition_distance。不出现匀加速段时它就是
    (2v₀ + Δv)·√(Δv/J) = L，即 Δv³ + 4v₀Δv² + 4v₀²Δv − L²J = 0。
    五段 S 曲线没有匀加速段，另外要求 Δv ≤ A²/J，使峰值加速度 √(JΔv) 不超过 A。
    """
    accel = a_max if phases == 7 else np.inf
    dv_max = np.cbrt(length**2 * j_max)  # 过渡距离至少为 Δv^1.5/√J，所以 Δv ≤ ∛(L²J)
    v = brentq(lambda v: transition_distance(v0, v, accel, j_max) - length, v0, v0 + 2 * dv_max)
    if phases == 5:
        v = min(v, v0 + a_max**2 / j_max)
    return v


def bidirectional_scan(lengths, v_limit, a_max, j_max, phases=7):
    """双向扫描。

    lengths (n,) 为各 block 长度，v_limit (n+1,) 为各连接点的速度上限（首尾通常为 0）。
    反向扫描保证每个 block 来得及减速到下一个连接点，正向扫描保证来得及加速，
    返回修正后的连接点速度 (n+1,)。
    """
    v = np.array(v_limit, dtype=float)
    n = len(lengths)
    for i in range(n - 1, -1, -1):
        v[i] = min(v[i], reachable_velocity(v[i + 1], lengths[i], a_max, j_max, phases))
    for i in range(n):
        v[i + 1] = min(v[i + 1], reachable_velocity(v[i], lengths[i], a_max, j_max, phases))
    return v
