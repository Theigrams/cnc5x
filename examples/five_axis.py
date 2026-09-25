"""五轴全流程：离散刀位 → 五轴样条 → 前瞻与速度规划 → 插补 → 机床轴检查 → 时间缩放或进给包络。

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
profile, _, v = cx.schedule(path, v_max=50, a_max=500, j_max=5000, Ts=TS)
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

# 6. 另一种做法：把各轴约束写进进给包络，只在需要的地方减速（block 内部也保证进给不超过包络）
s_env = np.linspace(0, path.length, 20000)
v_env = cx.feed_envelope(path, s_env, TS, 50, 500, 5000, chord_error=1e-3, machine=machine, drives=limits)
profile, knots, _ = cx.schedule(path, v_max=50, a_max=500, j_max=5000, Ts=TS, envelope=(s_env, v_env))
commands = cx.interpolate(path, profile, TS, machine=machine)
jerk_ratio = metrics.axis_report(commands.q, limits)["jerk"]["ratio"]
print(f"进给包络：连接点 {len(knots)} 个，用时 {commands.t[-1]:.3f} s，各轴 jerk 超限比 {np.round(jerk_ratio, 2)}")

# 7. 包络管不到的部分：C 轴 jerk q⃛ = q_sss v³ + 3 q_ss v a + q_s j，包络只限制了第一项（匀速的部分）
k = np.argmax(np.abs(commands.q[3, :, 4]))
s, v, a, j = commands.feed[:, k]
d = path.derivatives(s)
q = machine.axis_motion(d[:, None, :3], d[:, None, 3:])[:, 0, 4]  # C 轴对弧长的导数栈
terms = q[3] * v**3, 3 * q[2] * v * a, q[1] * j
print("C 轴 jerk 最大处的三项：匀速 {:.0f}，交叉 {:.0f}，切向 jerk {:.0f}（上限 100 rad/s³）".format(*terms))
print("  超限主要来自进给本身的 jerk（j_max = 5000 mm/s³ 对 C 轴太大），降低进给速度解决不了；")
factor = cx.time_scale_factor(commands.q, limits)
print(f"  再整体放慢 λ = {factor:.3f} 倍，用时 {commands.t[-1] * factor:.3f} s，反而比第 5 步只整体放慢更慢")
