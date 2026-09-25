"""拐角光顺对比：G01 直接插补、Zhao 2013、Xu 2018，走同一条流水线。

    刀路（ToolPath 子类）→ schedule（前瞻 + S 曲线）→ interpolate → metrics

在项目根目录运行：python -m examples.corner_smoothing [数据名]，默认用 butterfly。
装了 matplotlib 时，轨迹与进给速度图保存到 output/corner_smoothing.png。
"""

import sys
from pathlib import Path

import numpy as np

import cnc5x as cx
from cnc5x import metrics
from papers.Xu2018.algorithm import CcrPath
from papers.Zhao2013.algorithm import SmoothedPath

V_MAX, A_MAX, J_MAX, TS = 100.0, 3000.0, 60000.0, 0.0005  # mm/s、mm/s²、mm/s³、s
TOLERANCE = 0.02  # 拐角逼近误差上限，mm

name = sys.argv[1] if len(sys.argv) > 1 else "butterfly"
points = cx.datasets.load_dataset(name).points
methods = {  # 名字: (刀路, S 曲线段数)
    "G01": (cx.LinearPath(points), 7),
    "Zhao2013": (SmoothedPath(points, TOLERANCE), 5),
    "Xu2018": (CcrPath(points, TOLERANCE), 7),
}

print(f"数据 {name}：{len(points)} 个点；V = {V_MAX} mm/s，A = {A_MAX} mm/s²，J = {J_MAX} mm/s³，Ts = {TS} s")
results = {}
for label, (path, phases) in methods.items():
    profile, _ = cx.schedule(path, V_MAX, A_MAX, J_MAX, TS, phases=phases)
    commands = cx.interpolate(path, profile, TS)
    _, acceleration, jerk = metrics.tangential(commands.tip)
    deviation = metrics.path_deviation(commands.position, points).max()
    results[label] = commands
    print(
        f"{label:9s} 用时 {commands.t[-1]:7.3f} s | 偏离 G01 折线最大 {deviation:.4f} mm | "
        f"切向加速度最大 {np.nanmax(np.abs(acceleration)):6.1f} | 切向 jerk 最大 {np.nanmax(np.abs(jerk)):7.1f}"
    )

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
axes[0].plot(points[:, 0], points[:, 1], "o-", color="0.7", ms=2, lw=0.8, label="G01 points")
for label, commands in results.items():
    if label != "G01":
        axes[0].plot(commands.position[:, 0], commands.position[:, 1], lw=0.8, label=label)
    axes[1].plot(commands.t, commands.feed[1], lw=0.8, label=label)
axes[0].set(aspect="equal", xlabel="X (mm)", ylabel="Y (mm)", title=name)
axes[1].set(xlabel="t (s)", ylabel="feedrate (mm/s)", title="feedrate profile")
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
output = Path("output")
output.mkdir(exist_ok=True)
fig.savefig(output / "corner_smoothing.png", dpi=150)
print("图已保存到", output / "corner_smoothing.png")
