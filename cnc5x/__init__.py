"""cnc5x：五轴数控插补的教学库。单位 mm、s、rad；刀轴由刀尖指向刀柄。

主流程（每一步都是一个名词）：

    曲线 Curve ─▶ 刀路 ToolPath ─▶ 前瞻 bidirectional_scan ─▶ 速度规划 schedule ─▶ 插补 interpolate

全库约定：导数栈的形状为 (4, ..., dim)，第 k 层是第 k 阶导数。
"""

from . import calculus, datasets, geometry, metrics
from .curves import NURBS, ArcLengthTable, Bezier, BSpline, Curve, Line, Reparameterized, SubCurve
from .fitting import chord_parameters, fit_bspline, interpolate_bspline
from .geometry import rodrigues, rotation, unit
from .interpolator import Commands, interpolate, taylor_interpolate
from .io import read_cl
from .kinematics import TableTilting
from .limits import DriveLimits, chord_error_limit, curvature_limit, drive_limit, geometric_limit, time_scale_factor
from .look_ahead import bidirectional_scan, reachable_velocity
from .orientation import DualCurveDirection, GreatCircle, UnitDirection
from .profiles import Profile, align_period, concatenate, five_phase, seven_phase, transition
from .scheduler import schedule
from .toolpath import Block, CurvePath, LinearPath, PolylinePath, PoseCurve, ToolPath, dual_spline, pose_spline

__version__ = "0.3.0"

__all__ = [
    # 曲线
    "Curve",
    "Line",
    "Bezier",
    "BSpline",
    "NURBS",
    "SubCurve",
    "Reparameterized",
    "ArcLengthTable",
    "GreatCircle",
    "UnitDirection",
    "DualCurveDirection",
    "chord_parameters",
    "interpolate_bspline",
    "fit_bspline",
    # 刀路
    "PoseCurve",
    "pose_spline",
    "dual_spline",
    "Block",
    "ToolPath",
    "PolylinePath",
    "LinearPath",
    "CurvePath",
    # 机床
    "TableTilting",
    # 速度规划
    "Profile",
    "transition",
    "seven_phase",
    "five_phase",
    "concatenate",
    "align_period",
    "chord_error_limit",
    "curvature_limit",
    "geometric_limit",
    "DriveLimits",
    "drive_limit",
    "time_scale_factor",
    "reachable_velocity",
    "bidirectional_scan",
    "schedule",
    # 插补
    "Commands",
    "interpolate",
    "taylor_interpolate",
    # 工具与数据
    "unit",
    "rotation",
    "rodrigues",
    "read_cl",
    "calculus",
    "geometry",
    "metrics",
    "datasets",
]
