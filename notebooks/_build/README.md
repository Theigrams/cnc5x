# cnc5x 教学 notebook：共同要求（每个 notebook 作者必读）

## 背景

cnc5x 是五轴数控插补的**教学与论文复现库**。你要为它写一本教学 notebook。这本 notebook 有两个用途：

- 给初学者看懂原理（直觉 → 公式 → 代码 → 图）；
- 给人类调试时用：每个关键结论都有一个可以直接看数字的检查单元格。

动手前先读完下面这些：

- `CLAUDE.md`：协作规范、术语、约定，尤其是导数栈形状 `(4, ..., dim)`；
- `docs/数学约定.md` 中与你这本相关的章节；
- 你要讲的那些模块的源码；
- `examples/` 下相关的示例。

**API 以源码为准，不要凭印象调用。**

## 工具（已经写好，直接用）

- Python：`/Users/theigrams/miniforge3/envs/gpt/bin/python`（numpy、scipy、matplotlib、nbformat、nbclient、mpmath、sympy 都已装好）。
- 构建脚本写在本目录的 `build_<编号>.py`，形式如下：

```python
from nbtools import md, code, build
cells = [md("# 01 曲线与导数栈\n..."), md("## 1. ..."), code("..."), ...]
build("01_曲线与导数栈", cells)
```

  在本目录下运行 `python build_<编号>.py` 即可重建对应的 notebook。

- `build` 做的事：
  - 在第一个（标题）单元格之后，自动插入用户指定的开头单元格（`sys.path` 设置、`%config InlineBackend.figure_format='retina'`、`import matplotlib.pyplot as plt`），以及 `import numpy as np`、`import cnc5x as cx`、`from cnc5x import plotting`、`plotting.use_style()`。**不要再重复这些 import。** 需要别的模块时，在用到它的单元格开头 import，例如 `from cnc5x import metrics`、`from papers.Zhao2013.algorithm import SmoothedPath`；`papers` 可以直接 import，因为根目录已在 sys.path 里。
  - 写出 `notebooks/<name>.ipynb`，用 ruff format 格式化，用 nbclient 执行（工作目录是 `notebooks/`），并把输出存进 notebook。
  - 跑 ruff check（必须通过）。
  - 把所有图导出到系统临时目录下的 `cnc5x_nb/<name>/*.png`。
- 画图辅助在 `cnc5x/plotting.py`（先读一遍）：
  - `COLORS`：分类色，按固定顺序使用，不循环；
  - `GRAY`：原始数据、参考线；
  - `LIMIT`：上限虚线；
  - `plot_feed(t, feed, limits)`：s/v/a/j 四联图；
  - `plot_axes(t, q, limits, names)`：机床轴速度、加速度、jerk；
  - `plot_tool_axes(ax3d, points, axes, every, length)`。

## 写作规范

- **Markdown 用中文。** 每一节的顺序是：
  1. 物理或几何直觉，一两段；
  2. 公式；
  3. 代码；
  4. 图或数字；
  5. 一句"要点"。

  语气是讲给初学者听的教材，不是 API 文档。术语与 CLAUDE.md 第 7 节一致，第一次出现时给出英文，例如"弓高误差（chord error）"。
- **数学格式**：
  - 行内公式用 `$...$`（notebook 的渲染器都支持；不要用 `\(...\)`）；
  - 独立公式用 `$$`，前后两个 `$$` 各自单独占一行；
  - 向量和矩阵用 `\mathbf{}`，单位向量加 `\hat{}`，标量不加粗。
- **图里的一切文字都用英文**：title、xlabel、ylabel、legend、annotate、text，防止中文字体缺失。print 输出可以用中文。
- **代码单元格要短**（一般 ≤ 25 行），一个单元格只做一件事。
  - 注释写"为什么"。
  - 变量名贴近论文符号。
  - 讲原理的地方，先用几行 numpy 白盒写一遍，再和库函数对照，并打印差值。
- **检查单元格**：每个主要结论都配一个带独立参考值的数值检查。参考值可以来自：解析解、`scipy.integrate.quad`、有限差分（在 notebook 里当参考值可以用）、mpmath、SciPy 的同类函数。
  - 打印误差，并用 `assert` 断言到合理的量级。
  - 断言要有区分度，不要让结果贴着阈值过关。
- **数字必须来自实际运行输出。** Markdown 里引用的数值要和执行结果一致：先跑一遍，再回填，或者写成"见上面的输出"。不写没有验证过的结论。
- 结尾两节：
  - "练习"：2 到 4 道，可以改参数观察现象，或者指出一个值得动手验证的结论；
  - "延伸阅读"：链接 `../docs/数学约定.md` 的对应章节、`../docs/路线图.md`、相关论文目录。
- 开头的标题单元格写：标题、本节要回答的问题（3 到 5 条）、前置知识（指向前面的 notebook）。

## 画图规范（dataviz 规则的 matplotlib 版）

- 颜色按 `plotting.COLORS` 的顺序分配给系列，同一个实体在整本 notebook 里颜色保持一致。例如"G01 永远是 COLORS[0]"。
- 原始数据点、参考折线用 `plotting.GRAY`；上限用 `plotting.LIMIT` 虚线。
- **一个子图只有一个 y 轴，禁止 twinx。** 量纲不同的两个量画成两个子图。
- 有两个及以上系列就加 legend；单个系列不加 legend，用标题说明。
- 线条要细，不要每个点都标数字，只标关键的几个点。
- 几何图用 `ax.set_aspect("equal")`；误差跨越数量级时用对数坐标。
- figsize 宽度不超过 10 英寸，一本 notebook 大约 8 到 14 张图。
- 每张图导出后都要用 Read 工具打开看一遍，检查：
  - 标签有没有重叠；
  - 有没有空图；
  - 图例有没有挡住数据；
  - 放大的局部是不是真的看得清。

## 约束

- 整本 notebook 的执行时间不超过 60 秒（build 会打印耗时）。
- **只准写** `notebooks/<你的文件>.ipynb` 和本目录下的文件。
  - 不要修改 `cnc5x/`、`tests/`、`docs/`、`papers/`、README 等任何其他文件。
  - 发现库的 bug、文档与代码不一致、API 不顺手，记下来写进最终报告，不要自己修。
- 生成的 `.ipynb`（带执行输出）随库一起提交；临时试验脚本（probe、参数试验等）不要放进仓库。

## 最终报告（≤ 300 字，中文）

1. 各节标题列表；
2. 执行耗时、图的数量；
3. 发现的库问题（文件:行号，以及现象）；
4. 你没把握的论断，以及文档与实测不一致之处。
