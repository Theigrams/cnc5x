"""双转台五轴机床（理想相交轴）的正逆运动学与机床轴导数。

约定（主动旋转，工件坐标 → 机床坐标）：
    R_AC = R_x(A) R_z(C)，  R_BC = R_y(B) R_z(C)
    [X, Y, Z] = R p + b，    o = Rᵀ e_z（刀轴由刀尖指向刀柄）
机床轴 q = [X, Y, Z, A 或 B, C]，单位 mm 与 rad。b 只是机床坐标系里的固定平移，
不代表旋转轴线之间的偏置。

逆解：刀轴 o 的第三个分量给出倾角 = ±arccos(o_z)；C 是复数 w 的辐角，
AC 机床 w = o_y + i·o_x，BC 机床 w = −o_x + i·o_y；负倾角分支的 C 再加 π。
w = 0（刀轴平行于 C 轴）时 C 不定，这是运动学奇异点（极点）。
"""

import numpy as np

from .calculus import acos_derivatives, arg_derivatives, compose, product
from .geometry import rotation, unit

# 绕 x、y、z 轴旋转的生成元 K：dR/dθ = K R
GENERATORS = {
    "x": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "y": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    "z": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
}


class TableTilting:
    """双转台五轴机床，kind = "AC" 或 "BC"。"""

    def __init__(self, kind="AC", offset=(0.0, 0.0, 0.0)):
        if kind not in ("AC", "BC"):
            raise ValueError('kind 只能是 "AC" 或 "BC"')
        self.kind = kind
        self.tilt_axis = "x" if kind == "AC" else "y"
        self.offset = np.asarray(offset, dtype=float)
        self.axis_names = ("X", "Y", "Z", kind[0], "C")

    def rotation(self, tilt, c):
        """R = R_tilt(倾角) R_z(C)，形状 (..., 3, 3)。"""
        return rotation(self.tilt_axis, tilt) @ rotation("z", c)

    def forward(self, q):
        """正解：机床轴 q (..., 5) → 刀尖 p (..., 3) 与刀轴 o (..., 3)（工件坐标）。"""
        q = np.asarray(q, dtype=float)
        R = self.rotation(q[..., 3], q[..., 4])
        p = _matvec(np.swapaxes(R, -1, -2), q[..., :3] - self.offset)
        return p, R[..., 2, :]

    def inverse(self, p, o, branch=1):
        """逆解（主值）：刀位 p、o (..., 3) → 机床轴 q (..., 5)。branch = ±1 选倾角的正负分支。"""
        o = unit(o)
        tilt = branch * np.arccos(np.clip(o[..., 2], -1.0, 1.0))
        c = np.angle(self._c_complex(o)) + (np.pi if branch < 0 else 0.0)
        xyz = _matvec(self.rotation(tilt, c), np.asarray(p, dtype=float)) + self.offset
        return np.concatenate([xyz, tilt[..., None], c[..., None]], axis=-1)

    def inverse_path(self, p, o, branch=1, c_start=None):
        """沿一串刀位 (N, 3) 逐点逆解，返回 (N, 5)。

        C 角展开成连续值（相邻点跳变不超过 π）；极点处 C 不定，沿用最近的非极点值。
        给出 c_start 时再整体平移 2kπ，使起点的 C 最接近 c_start。
        """
        o = unit(o)
        tilt = branch * np.arccos(np.clip(o[:, 2], -1.0, 1.0))
        c = self._continuous_c(self._c_complex(o), branch, c_start)
        xyz = _matvec(self.rotation(tilt, c), np.asarray(p, dtype=float)) + self.offset
        return np.column_stack([xyz, tilt, c])

    def axis_motion(self, tip, axis, branch=1, c_start=None):
        """由刀尖、刀轴的导数栈 (4, N, 3) 求机床轴的导数栈 (4, N, 5)，N 按时间顺序排列。

        倾角 = ±arccos(o_z)、C = arg w 的导数由 calculus 中的链式公式给出，
        XYZ = R(倾角, C) p + b 的导数由 Leibniz 法则给出，全部是解析式、没有差分。
        极点处 C 不定，C 与倾角的各阶导数取最近的非极点样本（即从该侧逼近的极限）。
        """
        tip, axis = np.asarray(tip, dtype=float), np.asarray(axis, dtype=float)
        w = self._c_complex(axis)  # (4, N)，w 对刀轴分量是线性的，所以可以直接作用在导数栈上
        pole = np.abs(w[0]) < 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            tilt = branch * acos_derivatives(axis[..., 2])
            c = arg_derivatives(w)
        c[0] = self._continuous_c(w[0], branch, c_start)
        if np.any(pole):
            nearest = _nearest_regular(pole)
            tilt[1:, pole] = tilt[1:, nearest[pole]]
            c[1:, pole] = c[1:, nearest[pole]]
        R = product(_rotation_derivatives(self.tilt_axis, tilt), _rotation_derivatives("z", c), np.matmul)
        xyz = product(R, tip, _matvec)
        xyz[0] += self.offset
        return np.concatenate([xyz, tilt[..., None], c[..., None]], axis=-1)

    def _c_complex(self, o):
        """C 角对应的复数 w（C = arg w）。"""
        if self.kind == "AC":
            return o[..., 1] + 1j * o[..., 0]
        return -o[..., 0] + 1j * o[..., 1]

    def _continuous_c(self, w, branch, c_start):
        """C 角序列：取主值 → 极点处用最近的非极点值 → 展开成连续 → 靠近 c_start。"""
        c = np.angle(w) + (np.pi if branch < 0 else 0.0)
        pole = np.abs(w) < 1e-9
        if np.all(pole):
            return np.full(c.shape, 0.0 if c_start is None else float(c_start))
        c = np.unwrap(c[_nearest_regular(pole)])
        if c_start is not None:
            c += 2 * np.pi * np.round((c_start - c[0]) / (2 * np.pi))
        return c

    def __repr__(self):
        return f"TableTilting({self.kind!r}, offset={self.offset.tolist()})"


def _rotation_derivatives(axis, angle):
    """R(θ(t)) 的导数栈 (4, N, 3, 3)，angle 为 θ 的导数栈 (4, N)。

    dᵏR/dθᵏ = Kᵏ R，再用 compose 与 θ 的导数复合（把 3×3 矩阵当作 9 维向量）。
    """
    K = GENERATORS[axis]
    R = rotation(axis, angle[0])
    outer = np.stack([R, K @ R, K @ K @ R, K @ K @ K @ R])  # (4, N, 3, 3)
    flat = outer.reshape(outer.shape[:-2] + (9,))
    return compose(flat, angle[1], angle[2], angle[3]).reshape(outer.shape)


def _matvec(R, p):
    """批量矩阵乘向量：(..., 3, 3) @ (..., 3) → (..., 3)。"""
    return (R @ p[..., None])[..., 0]


def _nearest_regular(pole):
    """每个样本对应的非极点样本下标：本身不是极点就是自己，否则取其后（或最后一个）非极点样本。"""
    regular = np.flatnonzero(~pole)
    position = np.searchsorted(regular, np.arange(len(pole)))
    return regular[np.minimum(position, len(regular) - 1)]
