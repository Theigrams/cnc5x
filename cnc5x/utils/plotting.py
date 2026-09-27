"""画图辅助：教学 notebook 里反复出现的几种图。

matplotlib 是可选依赖，所以这个模块不在 cnc5x/__init__ 里导入，用时写 `from cnc5x.utils import plotting`。
这里只放 notebook 之间重复的图（进给四联图、机床轴图、刀轴箭头）；一次性的图直接在 notebook 里画，
读者能看到每一行 matplotlib 代码。图中文字一律用英文，避免中文字体缺失。
"""

import matplotlib.pyplot as plt

# 分类配色：固定顺序取用，不循环（已用色觉缺陷模拟检查过相邻两色的区分度）
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#9a9994"  # 参考线、原始数据点
LIMIT = "#52514e"  # 上限虚线

FEED_LABELS = ("s (mm)", "v (mm/s)", "a (mm/s²)", "j (mm/s³)")


def use_style():
    """统一的图面风格：细线、浅网格、去掉上右边框、按固定顺序取色。"""
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(color=COLORS),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.3,
            "lines.markersize": 4,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 100,
            "figure.constrained_layout.use": True,
        }
    )


def plot_feed(t, feed, limits=None, label=None, axes=None):
    """进给四联图：s、v、a、j 对时间，上下排列、共用时间轴。

    feed: (4, N) 导数栈 [s, v, a, j]（Commands.feed 或 profile(t)）；limits = (v_max, a_max, j_max)
    时画 ±上限虚线。axes 给出时画在已有的四个子图上（用来叠加多条曲线），返回 axes。
    """
    if axes is None:
        _, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(t, feed[k], label=label)
        ax.set_ylabel(FEED_LABELS[k])
        if limits is not None and k > 0:
            bound = limits[k - 1]
            ax.axhline(bound, color=LIMIT, ls="--", lw=0.8)
            if k > 1:
                ax.axhline(-bound, color=LIMIT, ls="--", lw=0.8)
    axes[-1].set_xlabel("t (s)")
    return axes


def plot_axes(t, q, limits=None, names=("X", "Y", "Z", "A", "C"), orders=(1, 2, 3)):
    """机床轴运动：每行一个轴，每列一阶导数（速度、加速度、jerk），有上限时画 ±上限虚线。

    q: (4, N, n_axes) 机床轴对时间的导数栈（Commands.q）；limits 为 DriveLimits。orders 只能取 1、2、3
    （位置直接画 q[0]）。直线轴单位 mm/s…，旋转轴 rad/s…。返回 (fig, axes)。
    """
    rows, cols = q.shape[-1], len(orders)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 1.5 * rows), sharex=True, squeeze=False)
    titles = {1: "velocity (unit/s)", 2: "acceleration (unit/s²)", 3: "jerk (unit/s³)"}
    for i in range(rows):
        unit = "rad" if names[i] in ("A", "B", "C") else "mm"
        axes[i, 0].set_ylabel(f"{names[i]} ({unit})")
        for c, k in enumerate(orders):
            ax = axes[i, c]
            ax.plot(t, q[k, :, i], color=COLORS[0])
            if limits is not None:
                bound = (limits.velocity, limits.acceleration, limits.jerk)[k - 1][i]
                ax.axhline(bound, color=LIMIT, ls="--", lw=0.8)
                ax.axhline(-bound, color=LIMIT, ls="--", lw=0.8)
    for c, k in enumerate(orders):
        axes[0, c].set_title(titles[k])
    for ax in axes[-1]:
        ax.set_xlabel("t (s)")
    return fig, axes


def plot_tool_axes(ax, points, axes, every=1, length=5.0, color=None):
    """在 3D 子图上画刀尖轨迹与刀轴箭头。points、axes (N, 3)；每 every 个点画一支长 length 的箭头。"""
    color = COLORS[0] if color is None else color
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color)
    p, o = points[::every], axes[::every]
    ax.quiver(p[:, 0], p[:, 1], p[:, 2], o[:, 0], o[:, 1], o[:, 2], length=length, color=COLORS[1], lw=0.8)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    return ax
