"""进给规划：几何与各轴限速、S 曲线进给轮廓、前瞻双向扫描、整条刀路的速度规划。

只依赖 utils。刀路和机床只通过接口使用（CLAUDE.md 第 3.6、3.9 条），不导入 curves、toolpath、kinematics。
"""
