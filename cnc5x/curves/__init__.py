"""曲线：全库的数学核心，与数控无关的几何对象。

curve.py 定义所有曲线共用的接口（导数栈、对弧长求导、取子段、换参数），先读它；spline.py 是具体的
曲线，arclength.py 是弧长表，orientation.py 是单位球面上的刀轴曲线，fitting.py 由离散点构造曲线，
pose.py 把刀尖、刀轴配成五轴刀位曲线。只依赖 utils。
"""
