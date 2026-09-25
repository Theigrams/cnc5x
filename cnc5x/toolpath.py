"""刀具路径：由 block 组成，是前瞻与速度规划的对象。

    ToolPath ──get_v_limit──▶ look_ahead ──▶ scheduler ──▶ interpolator

block 是若干首尾相连的曲线段，速度规划在每个 block 内生成一段 S 曲线；相邻 block 的
连接点就是速度控制点。复现一篇论文时，把它的刀路写成 ToolPath 的子类：生成
self.blocks，并实现 get_v_limit。
"""

import numpy as np
from scipy.optimize import minimize_scalar

from .calculus import compose, inverse_derivatives
from .curves import Curve, Line, SubCurve
from .fitting import chord_parameters, interpolate_bspline
from .geometry import polyline_tangents, turning_angles, unit
from .limits import geometric_limit
from .orientation import DualCurveDirection, GreatCircle, UnitDirection


class PoseCurve(Curve):
    """五轴刀位曲线：刀尖 p(u) 与单位刀轴 o(u) 共用参数 u，求值得到 6 维向量 [p, o]。

    弧长（进给的度量）默认只量刀尖（along="tip"，单位 mm）；刀尖不动、只转刀轴时用
    along="axis"，按刀轴转过的角度计量（单位 rad）。
    """

    def __init__(self, position, orientation, along="tip"):
        if position.domain != orientation.domain:
            raise ValueError("刀尖曲线与刀轴曲线的参数域必须相同")
        if along not in ("tip", "axis"):
            raise ValueError('along 只能是 "tip" 或 "axis"')
        self.position, self.orientation = position, orientation
        self.domain = position.domain
        self.measure = position if along == "tip" else orientation

    @property
    def breaks(self):
        return np.unique(np.concatenate([self.position.breaks, self.orientation.breaks]))

    def _derivative(self, u, order):
        return np.concatenate([self.position.derivative(u, order), self.orientation.derivative(u, order)], axis=-1)

    def curvature(self, u):
        return self.position.curvature(u)

    def speed(self, u):
        return self.measure.speed(u)

    def arc_derivatives(self, u):
        return self.measure.arc_derivatives(u)

    @property
    def length(self):
        return self.measure.length

    def length_at(self, u):
        return self.measure.length_at(u)

    def u_at_length(self, s):
        return self.measure.u_at_length(s)


def pose_spline(points, axes, parameters=None, degree=3):
    """过全部刀位点的五轴样条：刀尖与刀轴各插值一条 B 样条（共用参数），刀轴再单位化。

    parameters 默认用刀尖的累积弦长。
    """
    u = chord_parameters(points) if parameters is None else parameters
    position = interpolate_bspline(points, u, degree)
    orientation = UnitDirection(interpolate_bspline(unit(axes), u, degree))
    return PoseCurve(position, orientation)


def dual_spline(points, axes, height, parameters=None, degree=3):
    """双样条刀路（Langeron et al. 2004）：刀尖点与沿刀轴偏移 height 的第二点
    各插值一条 B 样条（共用参数），刀轴取两条曲线之差的方向。
    """
    u = chord_parameters(points) if parameters is None else parameters
    tip = interpolate_bspline(points, u, degree)
    top = interpolate_bspline(np.asarray(points, dtype=float) + height * unit(axes), u, degree)
    return PoseCurve(tip, DualCurveDirection(tip, top))


class Block:
    """进给率规划单元：若干首尾相连的曲线段（零长度的段自动去掉）。"""

    def __init__(self, curves):
        self.curves = [curve for curve in curves if curve.length > 0]
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

        先找出 s 落在哪一段、段内参数 u 是多少，再用反函数求导（u 对 s）和链式法则，
        把对 u 的导数换成对 s 的导数。
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
            u = curve.u_at_length(flat[mask] - starts[i])
            s1, s2, s3 = curve.arc_derivatives(u)
            u1, u2, u3 = inverse_derivatives(s1, s2, s3)
            d = compose(curve.derivatives(u), u1, u2, u3)
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
        self.N = len(self.points) - 1
        self.tangents, self.L = polyline_tangents(self.points)
        self.turning_angles = turning_angles(self.tangents)

    def segment(self, i):
        """第 i 段 G01：三轴是直线；五轴是直线 + 刀轴大圆插值。"""
        line = Line(self.points[i], self.points[i + 1])
        if self.axes is None:
            return line
        return PoseCurve(line, GreatCircle(self.axes[i], self.axes[i + 1]))

    def corner_blocks(self, transitions):
        """由各拐角的过渡曲线拼出 block（拐角光顺类论文的公共部分）。

        transitions[i] = (前半段, 后半段) 是第 i 个拐角（顶点 points[i+1]）的过渡。
        第 i 个 block = 上一拐角的后半段 + 直线 + 下一拐角的前半段，
        所以 block 的连接点正好落在各拐角过渡的中点（通常是曲率峰值处）。
        """
        blocks = []
        for i in range(self.N):
            curves = []
            start, end = self.points[i], self.points[i + 1]
            if i > 0:
                curves.append(transitions[i - 1][1])
                start = transitions[i - 1][1].end_point
            if i < self.N - 1:
                end = transitions[i][0].start_point
            curves.append(Line(start, end))
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
        # 严格高于左邻（留 1e-9 相对余量，滤掉恒曲率上的舍入噪声），不低于右邻，且不是近乎直线处
        is_peak = (inner > kappa[:-2] * (1 + 1e-9)) & (inner >= kappa[2:]) & (inner > 1e-6 * kappa.max())
        peaks = []
        for i in np.flatnonzero(is_peak) + 1:
            result = minimize_scalar(lambda x: -curve.curvature(x), bounds=(u[i - 1], u[i + 1]), method="bounded")
            peaks.append(result.x)
        self.split_parameters = np.array(peaks)
        self.curvature_peaks = curve.curvature(self.split_parameters)
        bounds = np.concatenate([[lo], self.split_parameters, [hi]])
        self.blocks = [Block([SubCurve(curve, a, b)]) for a, b in zip(bounds[:-1], bounds[1:])]

    def get_v_limit(self, Ts, v_max, a_max, j_max):
        """连接点（曲率峰值处）的速度上限：弓高误差、法向加速度、法向 jerk 三者取小。"""
        v = geometric_limit(self.curvature_peaks, self.chord_error, Ts, a_max, j_max)
        return np.minimum(np.concatenate([[0.0], v, [0.0]]), v_max)
