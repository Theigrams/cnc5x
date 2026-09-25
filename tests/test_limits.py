"""limits：弓高限速的定义域与饱和。

参考：圆上弦长 c 对应的弓高由几何直接算出，δ = ρ − √(ρ² − (c/2)²)，与限速公式的推导方向相反。
"""

import numpy as np

from cnc5x import DriveLimits, chord_error_limit, drive_limit, time_scale_factor
from cnc5x.limits import acceleration_interval, jerk_interval

TS = 0.001


def sagitta(rho, chord):
    return rho - np.sqrt(rho**2 - (chord / 2) ** 2)


def test_chord_limit_below_saturation_matches_sagitta():
    rho = np.array([0.5, 2.0, 10.0, 1e4])
    for delta in (1e-4, 1e-2, 0.4):
        v = chord_error_limit(1 / rho, delta, TS)
        assert np.allclose(sagitta(rho, v * TS), delta, rtol=1e-9)  # 一个周期的弦长恰好产生弓高 δ


def test_chord_limit_saturates_at_diameter():
    rho = 0.5
    deltas = np.array([0.5, 0.6, 0.99, 1.0, 1.5, 10.0])  # δ ≥ ρ；旧公式在 δ ≥ 2ρ = 1 时给出 0
    v = chord_error_limit(1 / rho, deltas, TS)
    assert np.allclose(v, 2 * rho / TS, rtol=1e-12)
    below = chord_error_limit(1 / rho, rho * (1 - 1e-9), TS)
    assert np.isclose(below, 2 * rho / TS, rtol=1e-4)  # 在 δ = ρ 处连续


def test_chord_limit_on_straight_line_is_unbounded():
    v = chord_error_limit(np.array([0.0, 0.0]), np.array([0.0, 0.02]), TS)
    assert np.all(np.isinf(v))  # 直线上没有弓高误差，容差为零也不限速
    assert chord_error_limit(1.0, 0.0, TS) == 0.0  # 有曲率而容差为零：只能停下


# ---------- 各轴约束：参考值是逐轴直接检查 |q̇| ≤ V、|q̈| ≤ A、|q⃛| ≤ J，看边界是否恰好卡住 ----------

LIMITS = DriveLimits([60.0, 60.0, 1.0], [600.0, 600.0, 5.0], [6000.0, 6000.0, 50.0])


def feasible(q_s, q_ss, q_sss, v, a, j):
    """各轴 |q̇| ≤ V、|q̈| ≤ A、|q⃛| ≤ J 是否都成立（q 对 s 的导数换成对 t 的导数）。"""
    qd = q_s * v
    qdd = q_ss * v**2 + q_s * a
    qddd = q_sss * v**3 + 3 * q_ss * v * a + q_s * j
    ok = (np.abs(qd) <= LIMITS.velocity) & (np.abs(qdd) <= LIMITS.acceleration) & (np.abs(qddd) <= LIMITS.jerk)
    return bool(np.all(ok))


def test_drive_limit_is_largest_uniform_feed():
    rng = np.random.default_rng(11)
    for _ in range(20):
        dq = np.vstack([np.zeros(3), rng.normal(size=(3, 3)) * [1, 1, 0.05]])
        v = drive_limit(dq, LIMITS)
        assert feasible(*dq[1:], v * (1 - 1e-9), 0, 0) and not feasible(*dq[1:], v * (1 + 1e-6), 0, 0)


def test_acceleration_and_jerk_intervals_are_tight():
    rng = np.random.default_rng(12)
    for _ in range(20):
        dq = np.vstack([np.zeros(3), rng.normal(size=(3, 3)) * [1, 1, 0.05]])
        v = 0.5 * drive_limit(dq, LIMITS)
        lo, hi = acceleration_interval(dq, v, LIMITS)
        eps = 1e-6 * (hi - lo)
        for a in (lo + eps, (lo + hi) / 2, hi - eps):  # 区间内可行（jerk 取 0 时 jerk 约束另说，这里只看加速度）
            assert np.all(np.abs(dq[2] * v**2 + dq[1] * a) <= LIMITS.acceleration)
        for a in (lo - eps, hi + eps):  # 区间外至少一个轴超限
            assert np.any(np.abs(dq[2] * v**2 + dq[1] * a) > LIMITS.acceleration)
        a = (lo + hi) / 2
        lo, hi = jerk_interval(dq, v, a, LIMITS)
        eps = 1e-6 * (hi - lo)
        for j in (lo + eps, hi - eps):
            assert feasible(*dq[1:], v, a, j)
        for j in (lo - eps, hi + eps):
            assert not feasible(*dq[1:], v, a, j)


def test_acceleration_interval_is_nan_when_infeasible():
    dq = np.zeros((4, 3))
    dq[2] = [0.0, 0.0, 10.0]  # 第三轴 q_s = 0：切向加速度帮不上忙，而 q_ss v² = 10·v² 已超过 5
    lo, hi = acceleration_interval(dq, 1.0, LIMITS)
    assert np.isnan(lo) and np.isnan(hi)


def test_time_scale_factor_is_the_smallest_sufficient_slowdown():
    rng = np.random.default_rng(13)
    dq_dt = rng.normal(size=(4, 50, 3)) * np.array([1, 80, 900, 12000])[:, None, None]
    lam = time_scale_factor(dq_dt, LIMITS)
    bounds = (LIMITS.velocity, LIMITS.acceleration, LIMITS.jerk)

    def worst(factor):  # 放慢 factor 倍后第 k 阶导数乘 factor⁻ᵏ，返回最大超限比
        return max(np.max(np.abs(dq_dt[k]) * factor**-k / bounds[k - 1]) for k in (1, 2, 3))

    assert worst(lam) <= 1 + 1e-12
    assert worst(lam * (1 - 1e-6)) > 1  # 再少放慢一点就不够
