"""双转台五轴机床（理想相交轴）的正逆运动学与机床轴导数。

约定（主动旋转，工件坐标 → 机床坐标）：
    R_AC = R_x(A) R_z(C)，  R_BC = R_y(B) R_z(C)
    [X, Y, Z] = R p + b，    o = Rᵀ e_z（刀轴由刀尖指向刀柄）
机床轴 q = [X, Y, Z, A 或 B, C]，单位 mm 与 rad。b 只是机床坐标系里的固定平移，
不代表旋转轴线之间的偏置。

逆解：刀轴 o 的第三个分量给出倾角 = ±arccos(o_z)；C 是复数 w 的辐角，
AC 机床 w = o_y + i·o_x，BC 机床 w = −o_x + i·o_y；负倾角分支的 C 再加 π。
w = 0（刀轴平行于 C 轴）时 C 不定，这是运动学奇异点（极点）。

同一个刀位有两个倾角分支，C 还可以差任意整圈。行程限位（tilt_range、c_range）排除一部分解：
沿刀路逆解时，先选倾角全程在行程内的分支，再把展开后的 C 平移整圈，放进 C 轴行程并尽量靠近 c_start。
"""

import numpy as np

from .utils import tolerances
from .utils.calculus import acos_derivatives, arg_derivatives, compose, product
from .utils.geometry import rotation, unit

# 绕 x、y、z 轴旋转的生成元 K：dR/dθ = K R
GENERATORS = {
    "x": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "y": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    "z": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
}


class TableTilting:
    """双转台五轴机床，kind = "AC" 或 "BC"。

    tilt_range、c_range：倾转轴（A 或 B）与 C 轴的行程 (lower, upper)，单位 rad；None 表示不限
    （C 轴常可无限回转）。行程只在沿刀路逆解（inverse_path、axis_motion）时起作用。
    """

    def __init__(self, kind="AC", offset=(0.0, 0.0, 0.0), tilt_range=None, c_range=None):
        if kind not in ("AC", "BC"):
            raise ValueError('kind 只能是 "AC" 或 "BC"')
        for bounds in (tilt_range, c_range):
            if bounds is not None and not bounds[0] < bounds[1]:
                raise ValueError("行程必须写成 (lower, upper) 且 lower < upper")
        self.kind = kind
        self.tilt_axis = "x" if kind == "AC" else "y"
        self.offset = np.asarray(offset, dtype=float)
        self.axis_names = ("X", "Y", "Z", kind[0], "C")
        self.tilt_range = None if tilt_range is None else (float(tilt_range[0]), float(tilt_range[1]))
        self.c_range = None if c_range is None else (float(c_range[0]), float(c_range[1]))

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
        """逆解（主值，不看行程）：刀位 p、o (..., 3) → 机床轴 q (..., 5)。branch = ±1 选倾角的正负分支。"""
        o = unit(o)
        tilt = branch * np.arccos(np.clip(o[..., 2], -1.0, 1.0))
        c = np.angle(self._c_complex(o)) + (np.pi if branch < 0 else 0.0)
        xyz = _matvec(self.rotation(tilt, c), np.asarray(p, dtype=float)) + self.offset
        return np.concatenate([xyz, tilt[..., None], c[..., None]], axis=-1)

    def choose_branch(self, o):
        """整条刀路用哪个倾角分支：倾角全程在 tilt_range 内的那个，两个都行时取正分支。

        倾角换号只能经过极点（刀轴竖直），所以一条刀路中途不换分支。C 轴行程不参与选择，
        放不进时由 C 的整圈平移报错。o：(N, 3) 单位刀轴。
        """
        tilt = np.arccos(np.clip(unit(o)[..., 2], -1.0, 1.0))
        for branch in (1, -1):
            if _inside(branch * tilt, self.tilt_range):
                return branch
        lo, hi = np.degrees(self.tilt_range)
        raise ValueError(
            f"倾角 {np.degrees(tilt.min()):.1f}°–{np.degrees(tilt.max()):.1f}° 的正负两个分支"
            f"都超出 {self.kind[0]} 轴行程 [{lo:.1f}°, {hi:.1f}°]"
        )

    def inverse_path(self, p, o, branch=None, c_start=None):
        """沿一串刀位 (N, 3) 逐点逆解，返回 (N, 5)。

        branch 缺省时由 choose_branch 按倾转轴行程选择。C 角展开成连续值（相邻点跳变不超过 π），
        极点处 C 不定，沿用最近的非极点值；再整体平移整圈，放进 C 轴行程并使起点最接近 c_start。
        """
        o = unit(o)
        branch = self.choose_branch(o) if branch is None else branch
        tilt = self._checked_tilt(branch * np.arccos(np.clip(o[:, 2], -1.0, 1.0)))
        c = self._continuous_c(self._c_complex(o), branch, c_start)
        xyz = _matvec(self.rotation(tilt, c), np.asarray(p, dtype=float)) + self.offset
        return np.column_stack([xyz, tilt, c])

    def axis_motion(self, tip, axis, branch=None, c_start=None):
        """由刀尖、刀轴的导数栈 (4, N, 3) 求机床轴的导数栈 (4, N, 5)，N 按时间顺序排列。

        倾角 = ±arccos(o_z)、C = arg w 的导数由 calculus 中的链式公式给出，
        XYZ = R(倾角, C) p + b 的导数由 Leibniz 法则给出，全部是解析式、没有差分。
        极点处 C 不定，C 与倾角的各阶导数取最近的非极点样本（即从该侧逼近的极限）。
        branch、c_start 与行程的处理同 inverse_path。
        """
        tip, axis = np.asarray(tip, dtype=float), np.asarray(axis, dtype=float)
        branch = self.choose_branch(axis[0]) if branch is None else branch
        w = self._c_complex(axis)  # (4, N)，w 对刀轴分量是线性的，所以可以直接作用在导数栈上
        pole = np.abs(w[0]) < tolerances.POLE
        with np.errstate(divide="ignore", invalid="ignore"):
            tilt = branch * acos_derivatives(axis[..., 2], rho=np.abs(w[0]))  # |w| = √(o_x² + o_y²)，无相消
            c = arg_derivatives(w)
        self._checked_tilt(tilt[0])
        c[0] = self._continuous_c(w[0], branch, c_start)  # 只有 C 本身要展开：导数对 C + 2kπ 的平移不变
        if np.all(pole):
            # 刀轴在全部样本上都平行于 C 轴：倾角恒为 0（或 π），C 保持初值，各阶导数都为零。
            # 公式里的 1/|w| 在这里是 0·∞，只能按定义给出（结论只对这些样本成立）。
            tilt[1:] = 0.0
            c[1:] = 0.0
        elif np.any(pole):
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
        """C 角序列：取主值 → 极点处用最近的非极点值 → 展开成连续 → 平移整圈（进 C 轴行程、靠近 c_start）。"""
        c = np.angle(w) + (np.pi if branch < 0 else 0.0)
        pole = np.abs(w) < tolerances.POLE
        if np.all(pole):
            c = np.full(c.shape, 0.0 if c_start is None else float(c_start))
        else:
            c = np.unwrap(c[_nearest_regular(pole)])
        return c + 2 * np.pi * self._whole_turns(c, c_start)

    def _whole_turns(self, c, c_start):
        """C 序列要平移的圈数 k：取最接近 (c_start − c₀)/2π 的整数（没给 c_start 时为 0），
        再限制到让全程落进 C 轴行程的范围 [k_lo, k_hi] 内；这个范围为空时报错。
        """
        k = 0.0 if c_start is None else np.round((c_start - c[0]) / (2 * np.pi))
        if self.c_range is None:
            return k
        lo, hi = self.c_range
        tol = tolerances.ROUNDING * max(1.0, hi - lo)
        k_lo, k_hi = np.ceil((lo - tol - c.min()) / (2 * np.pi)), np.floor((hi + tol - c.max()) / (2 * np.pi))
        if k_lo > k_hi:
            raise ValueError(f"C 角全程跨 [{c.min():.3f}, {c.max():.3f}] rad，平移整圈也放不进 C 轴行程 [{lo}, {hi}]")
        return np.clip(k, k_lo, k_hi)

    def _checked_tilt(self, tilt):
        """倾角 (N,) 超出倾转轴行程时报错，否则原样返回。"""
        if not _inside(tilt, self.tilt_range):
            lo, hi = np.degrees(self.tilt_range)
            raise ValueError(
                f"倾角范围 [{np.degrees(tilt.min()):.1f}°, {np.degrees(tilt.max()):.1f}°] "
                f"超出 {self.kind[0]} 轴行程 [{lo:.1f}°, {hi:.1f}°]"
            )
        return tilt

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


def _inside(values, bounds):
    """values 是否全部落在行程 bounds = (lower, upper) 内（允许相对 ROUNDING 的舍入）；bounds 为 None 时不限。"""
    if bounds is None:
        return True
    lo, hi = bounds
    tol = tolerances.ROUNDING * max(1.0, hi - lo)
    return bool(np.all((values >= lo - tol) & (values <= hi + tol)))


def _matvec(R, p):
    """批量矩阵乘向量：(..., 3, 3) @ (..., 3) → (..., 3)。"""
    return (R @ p[..., None])[..., 0]


def _nearest_regular(pole):
    """每个样本对应的非极点样本下标：本身不是极点就是自己，否则取其后（或最后一个）非极点样本。"""
    regular = np.flatnonzero(~pole)
    position = np.searchsorted(regular, np.arange(len(pole)))
    return regular[np.minimum(position, len(regular) - 1)]
