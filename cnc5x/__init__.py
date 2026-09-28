"""cnc5x：五轴数控插补的教学库。单位 mm、s、rad；刀轴由刀尖指向刀柄。

主流程（每一步都是一个名词）：

    曲线 Curve ─▶ 刀路 ToolPath ─▶ 前瞻 bidirectional_scan ─▶ 速度规划 schedule ─▶ 插补 interpolate

全库约定：导数栈的形状为 (4, ..., dim)，第 k 层是第 k 阶导数。
"""

from . import datasets, metrics
from .curves.arclength import ArcLengthTable
from .curves.curve import Curve, Reparameterized, SubCurve
from .curves.fitting import (
    angle_parameters,
    approximation_knots,
    chord_parameters,
    feed_correction,
    fit_bspline,
    hermite,
    interpolate_bspline,
    monotone_interpolate,
    spherical_spline,
)
from .curves.orientation import DualCurveDirection, GreatCircle, SphericalCurve, UnitDirection
from .curves.pose import PoseCurve, dual_spline, hermite_transition, pose_spline
from .curves.spline import NURBS, Bezier, BSpline, Line
from .datasets.io import read_cl
from .feedrate.limits import (
    DriveLimits,
    chord_error_limit,
    curvature_limit,
    drive_limit,
    geometric_limit,
    time_scale_factor,
)
from .feedrate.look_ahead import bidirectional_scan, reachable_velocity
from .feedrate.profiles import Profile, align_period, concatenate, five_phase, seven_phase, transition
from .feedrate.scheduler import envelope_scale, feed_envelope, schedule
from .interpolator import Commands, correction_interpolate, interpolate, taylor_interpolate
from .kinematics import TableTilting
from .toolpath import Block, CurvePath, HermiteCornerPath, LinearPath, PolylinePath, ToolPath
from .utils import calculus, geometry, tolerances
from .utils.geometry import rodrigues, rotation, unit

__version__ = "0.4.0"

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
    "SphericalCurve",
    "chord_parameters",
    "angle_parameters",
    "approximation_knots",
    "interpolate_bspline",
    "fit_bspline",
    "hermite",
    "monotone_interpolate",
    "spherical_spline",
    "feed_correction",
    # 刀路
    "PoseCurve",
    "pose_spline",
    "dual_spline",
    "Block",
    "ToolPath",
    "PolylinePath",
    "LinearPath",
    "HermiteCornerPath",
    "hermite_transition",
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
    "feed_envelope",
    "schedule",
    "envelope_scale",
    # 插补
    "Commands",
    "interpolate",
    "taylor_interpolate",
    "correction_interpolate",
    # 工具与数据
    "unit",
    "rotation",
    "rodrigues",
    "read_cl",
    "calculus",
    "geometry",
    "metrics",
    "datasets",
    "tolerances",
]
