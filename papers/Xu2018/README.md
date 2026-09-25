# Xu2018：双三次 B 样条外切圆角

Xu F., Sun Y. A circumscribed corner rounding method based on double cubic B-splines for a five-axis linear tool path. *The International Journal of Advanced Manufacturing Technology*, 2018, 94(1–4): 451–462. DOI: [10.1007/s00170-017-0869-x](https://doi.org/10.1007/s00170-017-0869-x)

## 算法与代码对应

| 步骤 | 内容 | 代码 |
|---|---|---|
| 过渡曲线 | 每个拐角一条 9 个控制点的三次 B 样条，外切于拐角，与直线 G² 连接 | `CcrPath.generate_ctrlpts` |
| 控制点角度 | α 由拐角夹角 θ 决定 | 式 (7) |
| 过渡长度 | 同时满足误差上限与长度约束，长度不够时按比例缩小 | 式 (4)、(11)，`adjustment_length` |
| 分 block | 过渡样条从中点（曲率峰值）一分为二，与直线拼成 block | `BSpline.split(0.5)`、`PolylinePath.corner_blocks` |
| 连接点限速 | 弓高误差、法向加速度、法向 jerk 取小 | `CcrPath.get_v_limit` |

```python
import cnc5x as cx
from papers.Xu2018.algorithm import CcrPath

points = cx.datasets.load_dataset("rhombic").points
path = CcrPath(points, chord_error=0.2)
profile, s, v = cx.schedule(path, v_max=100, a_max=3000, j_max=60000, Ts=0.0005)
commands = cx.interpolate(path, profile, Ts=0.0005)
```

## 说明

- 由 `cnc_interpolation/papers/Xu2018/algorithm.py` 移植，公式不变；未使用的手工分割函数 `split_bspline` 没有迁移。
- 补上了直行拐角（转角为 0）的处理：式 (7) 取极限 α = π，过渡是一段直线，逼近误差与曲率都取 0，法向任取；不依赖 cos(π/2) 在浮点下恰好不为零。
- 与原实现对拍：block 长度吻合到 2e-11（rhombic、butterfly、griffen）。连接点限速有两处不同：弓高限速在急弯处饱和于 2ρ/Ts 而不是降到 0（butterfly 上原实现有 2 处停车）；原实现把曲率下限截为 1e-6，griffen 上 104 个近直行拐角（转角不到 2e-9 rad）的限速因此只有约 0.008 mm/s，本库没有这一截断。
- 目前只复现了刀尖轨迹的圆角。论文中的刀轴光顺（对底部、顶部两条轨迹分别圆角）尚未实现。
- 原目录中的 rhombic、Math_Validation notebook 尚未迁移到新接口。

## 参考文献

```bibtex
@article{xu2018CircumscribedCornerRounding,
  title = {A Circumscribed Corner Rounding Method Based on Double Cubic {{B-splines}} for a Five-Axis Linear Tool Path},
  author = {Xu, Fuyang and Sun, Yuwen},
  year = {2018},
  journal = {The International Journal of Advanced Manufacturing Technology},
  volume = {94},
  number = {1-4},
  pages = {451--462},
  doi = {10.1007/s00170-017-0869-x}
}
```
