"""辅助工具：数值容差、导数运算（三阶链式法则等）、向量几何、画图。

第一次读库时可以先跳过这里，用到哪个函数再回来看。这里的模块只依赖 numpy、scipy（plotting 另依赖可选的
matplotlib），不依赖 cnc5x 的其他包，所以任何地方都能用它们，也不会形成循环依赖。plotting 不在
cnc5x/__init__ 中导入，用时写 `from cnc5x.utils import plotting`。
"""
