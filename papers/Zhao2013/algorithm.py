"""Zhao, Zhu & Ding (2013)：短线段的曲率连续 B 样条拐角过渡与实时前瞻插补。

每个拐角用两段对称的三次 Bézier 过渡（Theorem 1），曲率峰值在两段的连接处，
这个连接点就是 block 的分界。进给规划用五段 S 曲线：schedule(..., phases=5)。

由 cnc_interpolation/papers/Zhao2013/algorithm.py 移植，公式与原复现一致。
"""

import numpy as np

from cnc5x import Bezier, PolylinePath, geometric_limit, unit


def compute_d2(L, c1, beta, chord_error):
    """每个拐角的过渡长度 d2（自适应细分，满足式 (13)）。

    L (N,) 各段长度；c1 曲线形状参数；beta (N−1,) 各拐角的半转角；chord_error 拐角逼近误差上限。
    返回 d2 (N−1,)。
    """
    N = len(L)
    # Step 1：由式 (5) 按误差上限取初值；直行的“拐角”（β = 0）不受误差约束，取两侧较短段长
    with np.errstate(divide="ignore"):
        d2 = 2 * chord_error / np.sin(beta)
    d2 = np.where(np.isinf(d2), np.minimum(L[:-1], L[1:]), d2)
    # Step 2：首末两段只被一个过渡占用
    d2[0] = min(d2[0], L[0] / (c1 + 1))
    d2[-1] = min(d2[-1], L[-1] / (c1 + 1))
    # Step 3：d2 不超过相邻两段长度；相邻两个过渡不能在同一段上重叠（式 (11)）
    for i in range(N - 1):
        while d2[i] > L[i] or d2[i] > L[i + 1]:
            d2[i] /= 2
    for i in range(N - 2):
        while d2[i] + d2[i + 1] > L[i + 1] / (c1 + 1):
            d2[i], d2[i + 1] = d2[i] / 2, d2[i + 1] / 2
    return d2


class SmoothedPath(PolylinePath):
    """拐角用两段三次 Bézier 做曲率连续过渡的 G01 刀路。"""

    def __init__(self, points, chord_error, c1=0.5):
        super().__init__(points)
        self.chord_error, self.c1 = chord_error, c1
        self.beta = self.turning_angles / 2
        self.d2 = compute_d2(self.L, c1, self.beta, chord_error)
        self.curvature_peaks = 4 * np.sin(self.beta) / (3 * self.d2 * np.cos(self.beta) ** 2)
        self.chord_errors = self.d2 * np.sin(self.beta) / 2  # 各拐角的实际逼近误差，式 (5)
        self.blocks = self.generate_blocks()

    def generate_ctrlpts(self, P, d2):
        """拐角 P[1] 处两段三次 Bézier 的控制点（Theorem 1）。

        P (3, dim) 为 P0、P1、P2；d2 是 P1 到 Q1 的距离。返回 (4, dim)、(4, dim)。
        """
        dir1 = unit(P[0] - P[1])
        dir2 = unit(P[2] - P[1])
        Q0 = P[1] + (1 + self.c1) * d2 * dir1
        Q1 = P[1] + d2 * dir1
        Q2 = (P[1] + Q1) / 2
        Q5 = P[1] + d2 * dir2
        Q6 = P[1] + (1 + self.c1) * d2 * dir2
        Q4 = (Q5 + P[1]) / 2
        Q3 = (Q2 + Q4) / 2
        return np.array([Q0, Q1, Q2, Q3]), np.array([Q3, Q4, Q5, Q6])

    def generate_blocks(self):
        transitions = []
        for i in range(self.N - 1):
            first, second = self.generate_ctrlpts(self.points[i : i + 3], self.d2[i])
            transitions.append((Bezier(first), Bezier(second)))
        return self.corner_blocks(transitions)

    def get_v_limit(self, Ts, v_max, a_max, j_max):
        """连接点（曲率峰值处）的速度上限，式 (16)。

        沿用原复现的做法：弓高误差的容差取各拐角的实际逼近误差 chord_errors。
        """
        v = geometric_limit(self.curvature_peaks, self.chord_errors, Ts, a_max, j_max)
        return np.minimum(np.concatenate([[0.0], v, [0.0]]), v_max)
