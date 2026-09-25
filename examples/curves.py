"""曲线与弧长：B 样条、NURBS 整圆、导数栈、按弧长等分取点。

在项目根目录运行：python -m examples.curves
"""

import numpy as np

import cnc5x as cx

# B 样条：参数等分取点时弦长不均匀，按弧长等分才均匀
curve = cx.BSpline([[0, 0], [10, 25], [25, -10], [40, 30], [55, 0], [70, 20]], degree=3)
print(curve, f"弧长 {curve.length:.6f} mm")
for label, points in (("参数等分", curve.sample(9)), ("弧长等分", curve.sample(9, by_length=True))):
    chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
    print(f"  {label}的弦长：", chords.round(2))

# 导数栈：d[k] 是第 k 阶导数，形状 (4, ..., dim)
d = curve.derivatives(np.linspace(0, 1, 5))
print("导数栈形状", d.shape, "；u = 0.5 处的曲率", float(curve.curvature(0.5)))

# NURBS 精确表示整圆：四段二次有理 Bézier，权重 1、√2/2、1
w = np.sqrt(0.5)
circle = cx.NURBS(
    [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]],
    degree=2,
    knots=[0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1],
    weights=[1, w, 1, w, 1, w, 1, w, 1],
)
u = np.linspace(0, 1, 101)
radius = np.linalg.norm(circle(u), axis=1)
print(f"NURBS 整圆：半径 ∈ [{radius.min():.15f}, {radius.max():.15f}]，周长 / 2π = {circle.length / (2 * np.pi):.12f}")

# 弧长的反函数：给定弧长，找参数
s = 0.25 * circle.length
print(f"弧长 {s:.6f} 处的参数 u = {float(circle.u_at_length(s)):.12f}（四分之一圆应为 0.25）")
