"""速度规划：连接点限速 → 双向扫描 → 每个 block 一段 S 曲线 → 拼成整条进给轮廓。"""

import numpy as np

from .look_ahead import bidirectional_scan
from .profiles import concatenate, five_phase, seven_phase


def schedule(path, v_max, a_max, j_max, Ts, phases=7):
    """为刀路规划进给轮廓，返回 (profile, 连接点速度 (n_blocks + 1,))。

    phases = 7 用七段 S 曲线，phases = 5 用五段 S 曲线（Zhao et al. 2013）。
    """
    if phases not in (5, 7):
        raise ValueError("phases 只能是 5 或 7")
    v_limit = np.minimum(path.get_v_limit(Ts, v_max, a_max, j_max), v_max)
    v = bidirectional_scan(path.lengths, v_limit, a_max, j_max, phases)
    plan = seven_phase if phases == 7 else five_phase
    profiles = []
    for i, length in enumerate(path.lengths):
        profiles.append(plan(length, v[i], v[i + 1], v_max, a_max, j_max))
    return concatenate(profiles), v
