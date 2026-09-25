# Zhao2013：曲率连续 B 样条拐角过渡 + 实时前瞻插补

Zhao H., Zhu L., Ding H. A real-time look-ahead interpolation methodology with curvature-continuous B-spline transition scheme for CNC machining of short line segments. *International Journal of Machine Tools and Manufacture*, 2013, 65: 88–98. DOI: [10.1016/j.ijmachtools.2012.10.005](https://doi.org/10.1016/j.ijmachtools.2012.10.005)

## 算法流程与代码对应

| 步骤 | 内容 | 代码 |
|---|---|---|
| 路径光顺 | 每个拐角用两段对称三次 Bézier 过渡，与直线 G² 连接；曲率峰值在两段连接处 | `SmoothedPath.generate_ctrlpts`（Theorem 1） |
| 过渡长度 | 由误差上限取初值，再用自适应细分避免相邻过渡重叠 | `compute_d2`（式 (5)、(11)、(13)） |
| 分 block | block = 上一拐角后半段 + 直线 + 下一拐角前半段，连接点在曲率峰值处 | `PolylinePath.corner_blocks` |
| 连接点限速 | 弓高误差、法向加速度、法向 jerk 取小 | `SmoothedPath.get_v_limit`（式 (16)） |
| 前瞻 | 双向扫描 | `cnc5x.bidirectional_scan` |
| 速度规划 | 五段 S 曲线 | `cnc5x.schedule(..., phases=5)` |
| 插补 | 按插补周期对轮廓采样 | `cnc5x.interpolate` |

```python
import cnc5x as cx
from papers.Zhao2013.algorithm import SmoothedPath

points = cx.datasets.load_dataset("butterfly").points
path = SmoothedPath(points, chord_error=0.02, c1=0.5)
profile, s, v = cx.schedule(path, v_max=100, a_max=3000, j_max=60000, Ts=0.0005, phases=5)
commands = cx.interpolate(path, profile, Ts=0.0005)
```

## 说明

- 由 `cnc_interpolation/papers/Zhao2013/algorithm.py` 移植，公式不变。原代码在式 (5) 分母上加的 `1e-6` 已去掉：转角为 0 的“拐角”改为显式处理，过渡长度取两侧较短的段长，曲率峰值与逼近误差都为 0，连接点不受弓高约束（`limits.chord_error_limit` 对 κ = 0 返回 ∞）。
- 与原实现对拍：`1e-6` 影响所有拐角，rhombic 上 block 长度差 3e-7；griffen 上式 (13) 的逐次减半从略有不同的初值出发走了不同分支，276 个拐角的 d2 相差最多一倍，block 长度差到 6e-3 mm。两者都满足论文的约束。连接点限速的差异见仓库 README 的"论文复现"一节：弓高限速饱和于 2ρ/Ts 后，butterfly 上少停 6 处、griffen 上少停 15 处；原代码把直行拐角的限速截为 0（griffen 上 104 处），本库不再停车。
- 连接点的弓高误差限速沿用原复现，容差取各拐角的实际逼近误差 `chord_errors`，而不是插补的弓高容差。两者是否应当分开，需要对照原文确认。
- 原复现按 Lin 2007 表 3 的七种类型分别求解五段 S 曲线，并记录了一处勘误：Type (III) 应为

$$
f\left(T_{\text{end}}\right)=-J_{\max} T_{\text{end}}^3+2 V_{\text{str}} T_{\text{end}}-L_{\text{seg}}=0
$$

  cnc5x 改为统一求解峰值速度（`cnc5x.profiles.five_phase`），不再逐类型写方程，结果是同一条时间最短的五段曲线。
- 原目录中的 Butterfly、Dolphin、Fig6、Math_Validation 等 notebook 尚未迁移到新接口。

## 参考文献

```bibtex
@article{zhao2013RealtimeLookaheadInterpolation,
  title = {A Real-Time Look-Ahead Interpolation Methodology with Curvature-Continuous {{B-spline}} Transition Scheme for {{CNC}} Machining of Short Line Segments},
  author = {Zhao, Huan and Zhu, LiMin and Ding, Han},
  year = {2013},
  journal = {International Journal of Machine Tools and Manufacture},
  volume = {65},
  pages = {88--98},
  doi = {10.1016/j.ijmachtools.2012.10.005}
}

@article{lin2007DevelopmentDynamicsbasedNURBS,
  title = {Development of a Dynamics-Based {{NURBS}} Interpolator with Real-Time Look-Ahead Algorithm},
  author = {Lin, Ming-Tzong and Tsai, Meng-Shiun and Yau, Hong-Tzong},
  year = {2007},
  journal = {International Journal of Machine Tools and Manufacture},
  volume = {47},
  number = {15},
  pages = {2246--2262},
  doi = {10.1016/j.ijmachtools.2007.06.005}
}
```
