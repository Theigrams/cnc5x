"""几何小工具：单位化、夹角、折线的切向与转角、旋转矩阵、曲率。"""

import numpy as np


def unit(v):
    """沿最后一维单位化。零向量没有方向，直接报错。"""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("零向量没有方向")
    return v / norm


def to_3d(v):
    """二维点或向量补 z = 0；三维原样返回。"""
    v = np.asarray(v, dtype=float)
    if v.shape[-1] == 2:
        return np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], axis=-1)
    return v


def angle_between(a, b):
    """两向量的夹角 ∈ [0, π]。用 atan2(|a×b|, a·b)，小角度时比 arccos 准。"""
    a, b = to_3d(a), to_3d(b)
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


def polyline_tangents(points):
    """折线各段的单位切向 (N, D) 与段长 (N,)，points 形状 (N+1, D)。"""
    delta = np.diff(np.asarray(points, dtype=float), axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths == 0):
        raise ValueError("折线含有重复点")
    return delta / lengths[:, None], lengths


def turning_angles(tangents):
    """相邻两段切向的夹角 (N−1,)：0 表示直行，π 表示掉头。"""
    return angle_between(tangents[:-1], tangents[1:])


def rotation(axis, angle):
    """绕 x、y 或 z 轴旋转 angle 的主动旋转矩阵，形状 (..., 3, 3)。"""
    c, s = np.cos(angle), np.sin(angle)
    zero, one = np.zeros_like(c), np.ones_like(c)
    if axis == "x":
        rows = [[one, zero, zero], [zero, c, -s], [zero, s, c]]
    elif axis == "y":
        rows = [[c, zero, s], [zero, one, zero], [-s, zero, c]]
    elif axis == "z":
        rows = [[c, -s, zero], [s, c, zero], [zero, zero, one]]
    else:
        raise ValueError("axis 只能是 'x'、'y'、'z'")
    return np.moveaxis(np.array(rows), (0, 1), (-2, -1))  # (3, 3, ...) → (..., 3, 3)


def rodrigues(axis, angle):
    """绕任意单位轴 k 旋转 θ 的矩阵（Rodrigues 公式）：R = I + sinθ K + (1 − cosθ) K²。"""
    x, y, z = unit(axis)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = np.asarray(angle, dtype=float)[..., None, None]
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def curvature(d1, d2):
    """曲率 κ = |C' × C''| / |C'|³（二维曲线按 z = 0 处理）。"""
    cross = np.linalg.norm(np.cross(to_3d(d1), to_3d(d2)), axis=-1)
    return cross / np.linalg.norm(d1, axis=-1) ** 3
