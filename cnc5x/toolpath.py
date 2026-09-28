"""刀具路径：由 block 组成，是前瞻与速度规划的对象。

    ToolPath ──get_v_limit──▶ look_ahead ──▶ scheduler ──▶ interpolator

block 是若干首尾相连的曲线段，速度规划在每个 block 内生成一段 S 曲线；相邻 block 的
连接点就是速度控制点。复现一篇论文时，把它的刀路写成 ToolPath 的子类：生成
self.blocks，并实现 get_v_limit。
"""

import numpy as np
from scipy.optimize import minimize_scalar

from .curves.orientation import GreatCircle
from .curves.pose import PoseCurve, hermite_transition
from .curves.spline import Line
from .feedrate.limits import geometric_limit  # 反向依赖：刀路不该知道限速，P4 消除
from .utils import tolerances
from .utils.geometry import polyline_tangents, turning_angles, unit


class Block:
    """进给率规划单元：若干首尾相连的曲线段。

    零长度的段自动去掉（例如拐角过渡恰好占满整段直线，中间剩下的那一段）。但刀尖不动、
    刀轴却在转的五轴段不能去掉：按刀尖弧长计量时它的进给没有定义，这里直接报错。
    """

    def __init__(self, curves):
        self.curves = []
        for curve in curves:
            if curve.length > 0:
                self.curves.append(curve)
            elif np.max(np.abs(curve.end_point - curve.start_point)) > tolerances.ROUNDING:
                raise ValueError(
                    "曲线段的弧长为零但端点不同（刀尖不动、只转刀轴？），按刀尖弧长计量时进给没有定义。"
                    '单独的纯转刀轴运动可以用 PoseCurve(..., along="axis") 自成一条刀路；'
                    "混在 G01 刀路里的纯转刀轴段尚未支持，方案见 docs/路线图.md"
                )
        self.length = float(sum(curve.length for curve in self.curves))

    def __repr__(self):
        return f"Block({len(self.curves)} curves, length={self.length:.6g})"


class ToolPath:
    """刀具路径基类：一串 block。子类负责生成 self.blocks 并实现 get_v_limit。"""

    blocks = ()

    def get_v_limit(self, Ts, v_max, a_max, j_max):
        """各连接点的速度上限，形状 (n_blocks + 1,)；首尾两个连接点通常为 0。"""
        raise NotImplementedError

    @property
    def lengths(self):
        return np.array([block.length for block in self.blocks])

    @property
    def length(self):
        return float(self.lengths.sum())

    @property
    def curves(self):
        """所有 block 的曲线段，按走刀顺序排成一列。"""
        return [curve for block in self.blocks for curve in block.curves]

    def derivatives(self, s):
        """全局弧长 s 处的导数栈 [C, C_s, C_ss, C_sss]，形状 (4, ..., dim)。

        先找出 s 落在哪一段、段内参数 u 是多少，再由 Curve.derivatives_by_length
        把对 u 的导数换成对 s 的导数（反函数求导加链式法则）。
        """
        s = np.asarray(s, dtype=float)
        flat = s.ravel()
        curves = self.curves
        ends = np.cumsum([curve.length for curve in curves])
        starts = np.concatenate([[0.0], ends[:-1]])
        index = np.minimum(np.searchsorted(ends, flat, side="right"), len(curves) - 1)
        result = None
        for i in np.unique(index):
            mask = index == i
            curve = curves[i]
            d = curve.derivatives_by_length(curve.u_at_length(flat[mask] - starts[i]))
            if result is None:
                result = np.empty((4, len(flat), d.shape[-1]))
            result[:, mask] = d
        return result.reshape((4,) + s.shape + result.shape[-1:])

    def sample(self, n=1000):
        """沿弧长等分取 n 个点（画图用）。"""
        return self.derivatives(np.linspace(0.0, self.length, n))[0]


class PolylinePath(ToolPath):
    """G01 折线刀路的基类。points (N+1, D)；五轴时再给出每个点的刀轴 axes (N+1, 3)。

    预先算好子类常用的量：tangents (N, D) 各段单位切向，L (N,) 段长，
    turning_angles (N−1,) 各拐角的转角（直行为 0）。
    """

    def __init__(self, points, axes=None):
        self.points = np.asarray(points, dtype=float)
        self.axes = None if axes is None else unit(axes)
        repeated = np.all(self.points[1:] == self.points[:-1], axis=-1)
        if np.any(repeated) and self.axes is not None:
            changed = np.any(self.axes[1:] != self.axes[:-1], axis=-1)
            if np.any(repeated & changed):
                raise ValueError(
                    f"第 {np.flatnonzero(repeated & changed)[0]} 段刀尖不动、只转刀轴：G01 刀路按刀尖弧长计量进给，"
                    "这种段尚未支持，方案见 docs/路线图.md；刀尖与刀轴都重复的点请先去掉"
                )
        self.poses = self.points if axes is None else np.hstack([self.points, self.axes])
        self.N = len(self.points) - 1
        self.tangents, self.L = polyline_tangents(self.points)
        self.turning_angles = turning_angles(self.tangents)

    def g01(self, start, end):
        """从位姿 start 到 end 的一段 G01。

        三轴：直线。五轴（start、end 为 6 维 [p, o]）：刀尖走直线，刀轴沿大圆匀速转动，
        两者共用参数 u ∈ [0, 1]。
        """
        if self.axes is None:
            return Line(start, end)
        return PoseCurve(Line(start[:3], end[:3]), GreatCircle(start[3:], end[3:]))

    def segment(self, i):
        """第 i 段 G01（从 poses[i] 到 poses[i+1]）。"""
        return self.g01(self.poses[i], self.poses[i + 1])

    def corner_ends(self, i, before, after):
        """第 i 个拐角（顶点 points[i+1]）处过渡曲线的两个接点：进入段上距顶点 before，离开段上距顶点 after。

        返回两个对刀尖弧长的导数栈 (4, dim)；五轴时含刀轴，刀轴按 G01 的大圆插值取值。
        过渡曲线在接点处要与这两个导数栈吻合（例如 hermite_transition）。
        """
        start = self.segment(i).derivatives_by_length(1 - before / self.L[i])
        end = self.segment(i + 1).derivatives_by_length(after / self.L[i + 1])
        return start, end

    def corner_blocks(self, transitions):
        """由各拐角的过渡曲线拼出 block（拐角光顺类论文的公共部分）。

        transitions[i] = (前半段, 后半段) 是第 i 个拐角（顶点 points[i+1]）的过渡，
        五轴时它们是 6 维的刀位曲线。第 i 个 block = 上一拐角的后半段 + G01 + 下一拐角的前半段，
        中间的 G01 用 g01 补上（五轴时刀轴也一起沿大圆过渡），所以 block 的连接点正好落在
        各拐角过渡的中点（通常是曲率峰值处）。
        """
        blocks = []
        for i in range(self.N):
            curves = []
            start, end = self.poses[i], self.poses[i + 1]
            if i > 0:
                curves.append(transitions[i - 1][1])
                start = transitions[i - 1][1].end_point
            if i < self.N - 1:
                end = transitions[i][0].start_point
            curves.append(self.g01(start, end))
            if i < self.N - 1:
                curves.append(transitions[i][0])
            blocks.append(Block(curves))
        return blocks


class LinearPath(PolylinePath):
    """不做光顺的 G01 路径：每段直线自成一个 block。

    拐角处速度方向突变 2β（β 为半转角）。要求一个插补周期内速度矢量的变化 2v·sinβ
    不超过 a_max·Ts，得到 v ≤ a_max·Ts / (2 sinβ)（Zhao et al. 2013）。
    """

    def __init__(self, points, axes=None):
        super().__init__(points, axes)
        self.beta = self.turning_angles / 2
        self.blocks = [Block([self.segment(i)]) for i in range(self.N)]

    def get_v_limit(self, Ts, v_max, a_max, j_max):
        with np.errstate(divide="ignore"):
            corner = a_max * Ts / (2 * np.sin(self.beta))
        return np.minimum(np.concatenate([[0.0], corner, [0.0]]), v_max)


class HermiteCornerPath(PolylinePath):
    """拐角用五次 Hermite 曲线过渡的 G01 刀路（三轴或五轴）。这是拐角光顺的教学基线，不是某篇论文的方法。

    第 i 个拐角两侧各裁去 ℓᵢ，用 hermite_transition（h = 2ℓᵢ）连接：刀尖在接点处 G² 连续，
    五轴时刀轴对刀尖弧长 C² 连续。对称的五次过渡离顶点最近的是它的中点，距离为
        ε = (3/8)·ℓ·sin(φ/2)（φ 为转角），
    所以由拐角逼近误差上限 tolerance 取 ℓ = 8ε / (3 sin(φ/2))，再限制 ℓ 不超过相邻段长的一半
    （首末两段只有一端过渡，可用整段）。只限制了刀尖的逼近误差，刀轴的偏差没有限制。
    过渡在中点一分为二，block 的连接点就在中点；限速取中点曲率（对称过渡的曲率峰值）。
    """

    def __init__(self, points, tolerance, chord_error, axes=None):
        super().__init__(points, axes)
        if np.any(np.pi - self.turning_angles < tolerances.ANTIPODAL):
            raise ValueError("有掉头（转角为 π）的拐角：Hermite 过渡在中点速度为零")
        self.tolerance, self.chord_error = tolerance, chord_error
        room = self.L / 2
        room[[0, -1]] = self.L[[0, -1]]
        with np.errstate(divide="ignore"):
            trim = 8 * tolerance / (3 * np.sin(self.turning_angles / 2))  # 直行时为 ∞，由段长决定
        self.trim = np.minimum(trim, np.minimum(room[:-1], room[1:]))
        self.transitions = []
        for i in range(self.N - 1):
            start, end = self.corner_ends(i, self.trim[i], self.trim[i])
            self.transitions.append(hermite_transition(start, end, 2 * self.trim[i]))
        self.curvature_peaks = np.array([curve.curvature(0.5) for curve in self.transitions])
        halves = [(curve.restrict(0.0, 0.5), curve.restrict(0.5, 1.0)) for curve in self.transitions]
        self.blocks = self.corner_blocks(halves)

    def get_v_limit(self, Ts, v_max, a_max, j_max):
        """连接点（过渡中点）的速度上限：弓高误差、法向加速度、法向 jerk 三者取小。"""
        v = geometric_limit(self.curvature_peaks, self.chord_error, Ts, a_max, j_max)
        return np.minimum(np.concatenate([[0.0], v, [0.0]]), v_max)


class CurvePath(ToolPath):
    """一条光滑曲线（B 样条、NURBS 或五轴 PoseCurve）作为刀路。

    在曲率的局部极大处把曲线切成 block，连接点的速度上限由弓高误差、法向加速度与
    法向 jerk 决定（Lin et al. 2007 的做法）。只在这些临界点限速：两个峰值之间曲率不超过
    两端的较大者；曲率近乎恒定的长区段（例如整圆）没有峰值，需要子类另加限制。
    """

    def __init__(self, curve, chord_error, samples=2001):
        self.curve, self.chord_error = curve, chord_error
        lo, hi = curve.domain
        u = np.linspace(lo, hi, samples)
        kappa = curve.curvature(u)
        inner = kappa[1:-1]
        # 严格高于左邻（留舍入余量，滤掉恒曲率上的噪声），不低于右邻，且不是近乎直线处
        higher = inner > kappa[:-2] * (1 + tolerances.ROUNDING)
        is_peak = higher & (inner >= kappa[2:]) & (inner > tolerances.FLAT_CURVATURE * kappa.max())
        peaks = []
        for i in np.flatnonzero(is_peak) + 1:
            result = minimize_scalar(lambda x: -curve.curvature(x), bounds=(u[i - 1], u[i + 1]), method="bounded")
            peaks.append(result.x)
        self.split_parameters = np.array(peaks)
        self.curvature_peaks = curve.curvature(self.split_parameters)
        bounds = np.concatenate([[lo], self.split_parameters, [hi]])
        self.blocks = [Block([curve.restrict(a, b)]) for a, b in zip(bounds[:-1], bounds[1:])]

    def get_v_limit(self, Ts, v_max, a_max, j_max):
        """连接点（曲率峰值处）的速度上限：弓高误差、法向加速度、法向 jerk 三者取小。"""
        v = geometric_limit(self.curvature_peaks, self.chord_error, Ts, a_max, j_max)
        return np.minimum(np.concatenate([[0.0], v, [0.0]]), v_max)
