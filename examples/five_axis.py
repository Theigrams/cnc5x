"""五轴全流程：离散刀位 → 五轴样条 → 前瞻与速度规划 → 插补 → 机床轴检查 → 时间缩放。

在项目根目录运行：python -m examples.five_axis
"""

import numpy as np

import cnc5x as cx
from cnc5x import metrics

TS = 0.001
data = cx.datasets.load_dataset("horseshoe_planar_sweep")
print(f"数据 {data.name}：{len(data.points)} 个刀位；{data.info['description']}")

# 1. 刀尖与刀轴共用一组参数（刀尖弦长），各插值一条三次 B 样条
pose = cx.pose_spline(data.points, data.axes)
path = cx.CurvePath(pose, chord_error=1e-3)
print(f"刀尖弧长 {path.length:.2f} mm，在 {len(path.curvature_peaks)} 个曲率峰值处切成 {len(path.blocks)} 个 block")

# 2. 前瞻 + 七段 S 曲线（沿刀尖弧长），再插补并求 AC 双转台的机床轴
profile, v = cx.schedule(path, v_max=50, a_max=500, j_max=5000, Ts=TS)
machine = cx.TableTilting("AC")
commands = cx.interpolate(path, profile, TS, machine=machine)
print(f"插补 {len(commands.t)} 个周期，用时 {commands.t[-1]:.3f} s；连接点速度 {np.round(v, 1)}")

# 3. 正解回代：机床轴指令应当精确回到刀位曲线
p, o = machine.forward(commands.q[0])
print(f"正解残差：位置 {np.abs(p - commands.position).max():.1e} mm，刀轴 {np.abs(o - commands.orientation).max():.1e}")

# 4. 机床轴约束：刀尖进给满足限制，并不代表各轴也满足
limits = cx.DriveLimits(
    velocity=[60, 60, 60, 1.0, 2.0],  # X Y Z (mm/s)，A C (rad/s)
    acceleration=[600, 600, 600, 5.0, 10.0],
    jerk=[6000, 6000, 6000, 50.0, 100.0],
)
names = machine.axis_names
report = metrics.axis_report(commands.q, limits)
print("各轴峰值 / 上限：" + "".join(f"{axis:>8s}" for axis in names))
for quantity, label in (("velocity", "速度"), ("acceleration", "加速度"), ("jerk", "jerk")):
    print(f"  {label:6s}" + "".join(f"{ratio:8.3f}" for ratio in report[quantity]["ratio"]))

# 5. 超限时整体放慢：λ = max(1, r_v, √r_a, ∛r_j)，放慢后再插补一次
factor = cx.time_scale_factor(commands.q, limits)
if factor > 1:
    commands = cx.interpolate(path, profile.scaled(factor * 1.001), TS, machine=machine)
    report = metrics.axis_report(commands.q, limits)
    ratio = max(np.max(value["ratio"]) for value in report.values())
    print(f"整体放慢 λ = {factor:.4f} 倍后，用时 {commands.t[-1]:.3f} s，最大超限比 {ratio:.4f}（样本上）")
