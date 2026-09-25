"""三阶链式法则：把“对参数的导数”换成“对弧长或时间的导数”。

全库约定：导数栈 d 的形状为 (4, ..., dim)，d[k] 是第 k 阶导数（不是 Taylor 系数）。
"""

import numpy as np


def compose(d, x1, x2, x3):
    """复合函数 f(x(t)) 对 t 的 0..3 阶导数（三阶 Faà di Bruno 公式）：

        f'   = f_x x'
        f''  = f_xx x'² + f_x x''
        f''' = f_xxx x'³ + 3 f_xx x' x'' + f_x x'''

    d: (4, ..., dim)，f 对 x 的导数栈 [f, f_x, f_xx, f_xxx]
    x1, x2, x3: (...)，x 对 t 的 1..3 阶导数
    返回 (4, ..., dim)。
    """
    f0, f1, f2, f3 = d
    x1 = np.asarray(x1)[..., None]
    x2 = np.asarray(x2)[..., None]
    x3 = np.asarray(x3)[..., None]
    return np.stack([f0, f1 * x1, f2 * x1**2 + f1 * x2, f3 * x1**3 + 3 * f2 * x1 * x2 + f1 * x3])


def product(f, g, multiply=np.multiply):
    """乘积 f·g 的 0..3 阶导数（Leibniz 法则）。f、g 为导数栈，multiply 指定乘法（如矩阵乘）。"""
    f0, f1, f2, f3 = f
    g0, g1, g2, g3 = g
    return np.stack(
        [
            multiply(f0, g0),
            multiply(f1, g0) + multiply(f0, g1),
            multiply(f2, g0) + 2 * multiply(f1, g1) + multiply(f0, g2),
            multiply(f3, g0) + 3 * multiply(f2, g1) + 3 * multiply(f1, g2) + multiply(f0, g3),
        ]
    )


def speed_derivatives(d):
    """速度 σ = |x'| 及其 1、2 阶导数 (σ, σ', σ'')，d 为 x 的导数栈 (4, ..., dim)。

        σ   = |x'|
        σ'  = x'·x'' / σ
        σ'' = (|x''|² + x'·x''' − σ'²) / σ

    x 对参数 u 求导时，(σ, σ', σ'') 就是弧长的导数 (s', s'', s''')；
    x 对时间 t 求导时，它们是切向速度、切向加速度、切向 jerk。
    """
    _, x1, x2, x3 = d
    speed = np.linalg.norm(x1, axis=-1)
    speed1 = np.sum(x1 * x2, axis=-1) / speed
    speed2 = (np.sum(x2 * x2, axis=-1) + np.sum(x1 * x3, axis=-1) - speed1**2) / speed
    return speed, speed1, speed2


def inverse_derivatives(s1, s2, s3):
    """反函数 u(s) 的 1..3 阶导数，已知 s(u) 的导数 s'、s''、s'''：

        u'   = 1 / s'
        u''  = −s'' / s'³
        u''' = (3 s''² − s' s''') / s'⁵

    由 u(s(u)) = u 两边逐阶求导得到。
    """
    return 1 / s1, -s2 / s1**3, (3 * s2**2 - s1 * s3) / s1**5


def unit_derivatives(d):
    """单位向量 o = r / |r| 的 0..3 阶导数，d 为 r 的导数栈 (4, ..., dim)。

    由 r = ρ o（ρ = |r|）逐阶求导：
        o'   = (r' − ρ' o) / ρ
        o''  = (r'' − ρ'' o − 2ρ' o') / ρ
        o''' = (r''' − ρ''' o − 3ρ'' o' − 3ρ' o'') / ρ
    """
    r0, r1, r2, r3 = d
    rho = np.linalg.norm(r0, axis=-1, keepdims=True)
    if np.any(rho == 0):
        raise ValueError("方向向量为零，单位化没有定义")
    rho1 = _dot(r0, r1) / rho
    rho2 = (_dot(r1, r1) + _dot(r0, r2) - rho1**2) / rho
    rho3 = (3 * _dot(r1, r2) + _dot(r0, r3) - 3 * rho1 * rho2) / rho
    o0 = r0 / rho
    o1 = (r1 - rho1 * o0) / rho
    o2 = (r2 - rho2 * o0 - 2 * rho1 * o1) / rho
    o3 = (r3 - rho3 * o0 - 3 * rho2 * o1 - 3 * rho1 * o2) / rho
    return np.stack([o0, o1, o2, o3])


def acos_derivatives(d):
    """θ = arccos(z) 的 0..3 阶导数，d = [z, z', z'', z''']，形状 (4, ...)。

    外函数 arccos 的导数为 −1/ρ、−z/ρ³、−(1 + 2z²)/ρ⁵（ρ = √(1 − z²)），再用 compose 复合。
    z = ±1 时 ρ = 0，导数没有定义。
    """
    z = d[0]
    rho = np.sqrt(1 - z**2)
    outer = np.stack([np.arccos(z), -1 / rho, -z / rho**3, -(1 + 2 * z**2) / rho**5])
    return compose(outer[..., None], d[1], d[2], d[3])[..., 0]


def arg_derivatives(d):
    """辐角 φ = arg(w) 的 0..3 阶导数，d = [w, w', w'', w''']，w = x + iy 为复数，形状 (4, ...)。

    φ = Im(log w)，外函数 log 的导数为 1/w、−1/w²、2/w³。w = 0 时辐角没有定义。
    """
    w = d[0]
    outer = np.stack([np.log(w), 1 / w, -1 / w**2, 2 / w**3])
    return compose(outer[..., None], d[1], d[2], d[3])[..., 0].imag


def _dot(a, b):
    return np.sum(a * b, axis=-1, keepdims=True)
