import numpy as np
import pytest

from cnc5x import (
    LinearPath,
    Profile,
    align_period,
    bidirectional_scan,
    concatenate,
    five_phase,
    reachable_velocity,
    schedule,
    seven_phase,
)
from cnc5x.feedrate.profiles import transition_distance


def dense(profile, n=20001):
    t = np.linspace(0, profile.duration, n)
    return t, profile(t)


def phase_blocks(profile):
    """教材的记法：加速段 T_a、匀速段 T_v、减速段 T_d。"""
    d = profile.durations
    return d[:3].sum(), d[3], d[4:].sum()


def test_biagiotti_example_3_9():
    # Biagiotti & Melchiorri《Trajectory Planning for Automatic Machines and Robots》例 3.9
    profile = seven_phase(10, v_start=1, v_end=0, v_max=5, a_max=10, j_max=30)
    Ta, Tv, Td = phase_blocks(profile)
    assert np.allclose([Ta, Tv, Td], [0.7333, 1.1433, 0.8333], atol=1e-4)
    assert np.isclose(profile.durations[0], 0.3333, atol=1e-4)


def test_biagiotti_example_3_10():
    profile = seven_phase(10, v_start=1, v_end=0, v_max=10, a_max=10, j_max=30)
    d = profile.durations
    assert np.isclose(d[:3].sum(), 1.0747, atol=1e-4) and np.isclose(d[3:].sum(), 1.1747, atol=1e-4)
    _, state = dense(profile)
    assert np.isclose(state[1].max(), 8.4136, atol=1e-4)


def test_biagiotti_example_3_11_is_not_slower_than_textbook():
    # 教材让 a_max 每次乘 0.99 近似求解，总时间 1.93837 s；这里直接解峰值速度，不应更慢
    profile = seven_phase(10, v_start=7, v_end=0, v_max=10, a_max=10, j_max=30)
    assert profile.duration <= 1.93837
    _, state = dense(profile)
    assert np.abs(state[2]).max() <= 10 + 1e-9


@pytest.mark.parametrize(
    "length, v0, v1", [(10, 0, 0), (10, 1, 0), (10, 4.5, 0), (0.8, 2, 2.5), (100, 0, 3), (0.2, 0, 0), (3, 4, 4)]
)
def test_profile_properties(length, v0, v1):
    v_max, a_max, j_max = 5, 10, 30
    profile = seven_phase(length, v0, v1, v_max, a_max, j_max)
    _, state = dense(profile)
    assert np.isclose(profile.length, length, rtol=1e-9)
    assert np.allclose(state[:3, 0], [0, v0, 0]) and np.allclose(state[1:3, -1], [v1, 0], atol=1e-9)
    assert state[1].max() <= v_max + 1e-9
    assert np.abs(state[2]).max() <= a_max + 1e-9
    assert np.abs(state[3]).max() <= j_max + 1e-9
    assert np.all(np.diff(state[0]) >= -1e-12)  # 不倒退


def test_five_phase_respects_acceleration():
    profile = five_phase(10, 1, 0, 10, a_max=10, j_max=30)
    _, state = dense(profile)
    assert np.abs(state[2]).max() <= 10 + 1e-9
    assert np.count_nonzero(profile.jerks == 0) <= 1  # 只有匀速段 jerk 为零
    edge = five_phase(10, 10**2 / 30, 0, 5, a_max=10, j_max=30)  # Δv 恰好等于 A²/J：峰值加速度恰好是 A
    assert np.isclose(np.abs(dense(edge)[1][2]).max(), 10, rtol=1e-9)


def test_five_phase_rejects_large_speed_change():
    # Δv = 5 > A²/J = 10/3：没有匀加速段时峰值加速度 √(JΔv) = 12.2 > 10，以前会静默超限
    with pytest.raises(ValueError):
        five_phase(10, 5, 0, 5, a_max=10, j_max=30)


def test_infeasible_boundary_speeds_raise():
    with pytest.raises(ValueError):
        seven_phase(0.01, 5, 0, 5, 10, 30)


def test_profile_is_continuous_across_phases():
    profile = concatenate([seven_phase(3, 0, 2, 4, 10, 30), seven_phase(2, 2, 1, 4, 10, 30)])
    t = profile.times[1:-1]
    left, right = profile(t - 1e-9), profile(t + 1e-9)
    assert np.allclose(left[:3], right[:3], atol=1e-7)
    assert np.isclose(profile.length, 5)


def test_scaled_and_aligned():
    profile = seven_phase(10, 0, 0, 5, 10, 30)
    slow = profile.scaled(2.0)
    t = np.linspace(0, profile.duration, 50)
    assert np.allclose(slow(2 * t), profile(t) / np.array([1, 2, 4, 8])[:, None])
    aligned = align_period(profile, 0.001)
    n = aligned.duration / 0.001
    assert np.isclose(n, round(n)) and aligned.duration >= profile.duration


def test_reachable_velocity_solves_distance_equation():
    for phases in (5, 7):
        v = reachable_velocity(0.3, 0.8, a_max=10, j_max=30, phases=phases)
        accel = 10 if phases == 7 else np.inf
        distance = transition_distance(0.3, v, accel, 30)
        assert np.isclose(distance, 0.8) or np.isclose(v, 0.3 + 10**2 / 30)
    # 不出现匀加速段时即 Δv³ + 4v₀Δv² + 4v₀²Δv − L²J = 0
    v0, L, J = 0.3, 0.05, 30
    dv = reachable_velocity(v0, L, 1e9, J) - v0
    assert np.isclose(dv**3 + 4 * v0 * dv**2 + 4 * v0**2 * dv, L**2 * J)


@pytest.mark.parametrize("phases", [5, 7])
def test_scan_makes_every_block_feasible(phases):
    rng = np.random.default_rng(1)
    lengths = rng.uniform(0.05, 3.0, 30)
    v_limit = np.r_[0, rng.uniform(0.5, 5, 29), 0]
    v = bidirectional_scan(lengths, v_limit, a_max=10, j_max=30, phases=phases)
    assert np.all(v <= v_limit + 1e-12) and v[0] == 0 and v[-1] == 0
    plan = seven_phase if phases == 7 else five_phase
    for i, L in enumerate(lengths):
        profile = plan(L, v[i], v[i + 1], 5, 10, 30)
        _, state = dense(profile, 2001)
        assert np.abs(state[2]).max() <= 10 + 1e-9


def test_schedule_on_polyline():
    path = LinearPath([[0, 0], [10, 0], [10, 10], [0, 10]])
    profile, s, v = schedule(path, v_max=50, a_max=1000, j_max=20000, Ts=0.001)
    assert isinstance(profile, Profile)
    assert np.isclose(profile.length, path.length)
    assert v.shape == (4,) and v[0] == 0 and v[-1] == 0
