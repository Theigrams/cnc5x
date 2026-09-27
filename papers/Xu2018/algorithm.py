"""Xu & Sun (2018)：基于双三次 B 样条的外切圆角（circumscribed corner rounding）。

每个拐角用一条 9 个控制点的三次 B 样条过渡，从中点 u = 0.5 处分成两半，
中点是曲率峰值，也是 block 的分界。

由 cnc_interpolation/papers/Xu2018/algorithm.py 移植，公式与原复现一致；
另外补上了直行拐角（转角为 0）的处理。
"""

import numpy as np

from cnc5x import BSpline, PolylinePath, geometric_limit, rodrigues, unit
from cnc5x.utils.geometry import to_3d

KNOTS = np.array([0, 0, 0, 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1, 1, 1, 1])


class CcrPath(PolylinePath):
    """拐角用双三次 B 样条外切圆角的 G01 刀路。"""

    def __init__(self, points, chord_error):
        super().__init__(points)
        self.chord_error = chord_error
        self.theta = np.pi - self.turning_angles  # 拐角处两段之间的夹角
        # 直行（θ = π）时 Eq. (7) 的分母 cos(θ/2) 为零，取极限 α → π：过渡曲线是一段直线，
        # 没有逼近误差，Eq. (4) 不起约束。显式处理，不依赖 cos(π/2) 在浮点下恰好不为零。
        straight = self.turning_angles == 0
        alpha = 2 * np.arctan((1 / 3 + np.sin(self.theta / 2)) / np.cos(self.theta / 2))  # Eq. (7)
        self.alpha = np.where(straight, np.pi, alpha)
        self.N0N4_list = self.adjustment_length()
        N3N4_limit1 = np.where(straight, np.inf, 3 * chord_error / np.cos(self.alpha / 2))  # Eq. (4)
        N3N4_limit2 = self.N0N4_list / (4 * np.cos((self.alpha - self.theta) / 2) + 1)  # Eq. (11)
        self.N3N4_list = np.minimum(N3N4_limit1, N3N4_limit2)
        self.chord_errors = np.where(straight, 0.0, self.N3N4_list * np.cos(self.alpha / 2) / 3)
        self.blocks = self.generate_blocks()

    def adjustment_length(self):
        """各拐角可用的过渡长度：首末段整段可用，中间段两个拐角各分一半，再留 10% 余量。"""
        L = self.L
        N = len(L)
        L_T = np.zeros(N - 1)
        for i in range(N - 1):
            if i == 0:
                L_T[i] = min(L[i], L[i + 1] / 2)
            elif i == N - 2:
                L_T[i] = min(L[i] / 2, L[i + 1])
            else:
                L_T[i] = min(L[i] / 2, L[i + 1] / 2)
        return L_T * 0.9

    def generate_ctrlpts(self, P, L, i):
        """第 i 个拐角的 9 个控制点 N0…N8。P (3, dim) 为 P1、P2、P3；L 是 N3 到 N4 的距离。"""
        theta, alpha = self.theta[i], self.alpha[i]
        P1, P2, P3 = to_3d(P)
        e1 = unit(P1 - P2)
        e2 = unit(P3 - P2)
        n = np.cross(e1, e2)
        n = unit(n) if np.linalg.norm(n) > 0 else np.array([0.0, 0.0, 1.0])  # 直行时转角为 0，法向任取
        rot_angle = (alpha - theta) / 2
        e3 = rodrigues(n, -rot_angle) @ e1
        e4 = rodrigues(n, rot_angle) @ e2
        k = 4 * np.cos((alpha - theta) / 2)
        N4 = P2
        N3 = N4 + L * e3
        N5 = N4 + L * e4
        N1 = N2 = N4 + k * L * e1
        N6 = N7 = N4 + k * L * e2
        N0 = N4 + (k + 1) * L * e1
        N8 = N4 + (k + 1) * L * e2
        ctrlpts = np.array([N0, N1, N2, N3, N4, N5, N6, N7, N8])
        return ctrlpts[:, : P.shape[1]]

    def generate_blocks(self):
        splines = []
        for i in range(self.N - 1):
            ctrlpts = self.generate_ctrlpts(self.points[i : i + 3], self.N3N4_list[i], i)
            splines.append(BSpline(ctrlpts, 3, KNOTS))
        self.bsplines = splines
        # 直行拐角的过渡是直线，曲率为零；数值求出的是 1e-13 量级的舍入噪声，配上为零的逼近误差会给出 v = 0
        peaks = np.array([spline.curvature(0.5) for spline in splines])
        self.curvature_peaks = np.where(self.turning_angles == 0, 0.0, peaks)
        return self.corner_blocks([spline.split(0.5) for spline in splines])

    def get_v_limit(self, Ts, v_max, a_max, j_max):
        """连接点（曲率峰值处）的速度上限；弓高误差的容差沿用原复现，取各拐角的实际逼近误差。"""
        v = geometric_limit(self.curvature_peaks, self.chord_errors, Ts, a_max, j_max)
        return np.minimum(np.concatenate([[0.0], v, [0.0]]), v_max)
