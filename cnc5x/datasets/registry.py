"""内置数据的登记表。

source 写明文件从哪里来；reference 只填核对过的文献，未核实的一律为 None。
"""

_PLANAR_SOURCE = "旧 cnc5x 数据集（cnc5x/datasets/data/cad_curves）"
_HUST_SOURCE = (
    "测试刀位数据/（由 virtualfiveaxisCNCmachining MATLAB 工程 Data/Input/{} 转换，"
    "HUST iRobotCNC，ICIRA 2015 配套工程）"
)

REGISTRY = {
    # ---------- 平面 G01 轮廓（mm） ----------
    "rhombic": {
        "file": "rhombic.txt",
        "description": "菱形，4 个 90° 拐角，拐角光顺最小算例",
        "source": _PLANAR_SOURCE + "；与 cnc_interpolation 中 Zhao2013、Xu2018 复现所用的菱形相同",
        "reference": None,
    },
    "square": {"file": "square.txt", "description": "方框，含 90° 直角", "source": _PLANAR_SOURCE, "reference": None},
    "butterfly": {
        "file": "butterfly.txt",
        "description": "蝴蝶轮廓，100 点",
        "source": _PLANAR_SOURCE + "；与 cnc_interpolation/papers/Zhao2013/data/butterfly.txt 相同",
        "reference": None,
    },
    "dolphin": {
        "file": "dolphin.txt",
        "description": "海豚轮廓，200 点",
        "source": _PLANAR_SOURCE + "；与 cnc_interpolation/papers/Zhao2013/data/dolphin.txt 相同",
        "reference": None,
    },
    "golden_fish": {"file": "golden_fish.txt", "description": "金鱼轮廓", "source": _PLANAR_SOURCE, "reference": None},
    "griffen": {
        "file": "griffen.txt",
        "description": "狮鹫轮廓，约 1100 点",
        "source": _PLANAR_SOURCE,
        "reference": None,
    },
    "manta_ray": {"file": "manta_ray.txt", "description": "蝠鲼轮廓", "source": _PLANAR_SOURCE, "reference": None},
    "mermaid": {
        "file": "mermaid.txt",
        "description": "美人鱼轮廓，约 600 点",
        "source": _PLANAR_SOURCE,
        "reference": None,
    },
    "seahorse": {"file": "seahorse.txt", "description": "海马轮廓", "source": _PLANAR_SOURCE, "reference": None},
    "shark": {"file": "shark.txt", "description": "鲨鱼轮廓", "source": _PLANAR_SOURCE, "reference": None},
    "unicorn": {"file": "unicorn.txt", "description": "独角兽轮廓", "source": _PLANAR_SOURCE, "reference": None},
    # ---------- 五轴刀位（刀尖 mm + 单位刀轴，刀轴由刀尖指向刀柄） ----------
    "horseshoe_planar_sweep": {
        "file": "horseshoe_planar_sweep.npz",
        "description": "25 点，平面马蹄形/S 形曲线，刀轴 10°–42° 连续变倾（MATLAB 工程的默认演示刀路）",
        "source": _HUST_SOURCE.format("tool path data.txt"),
        "reference": None,
    },
    "semicircle_arch_normal": {
        "file": "semicircle_arch_normal.npz",
        "description": "76 点，半圆拱，刀轴沿法向扇形展开（拱顶约 0°，两腿约 80°）",
        "source": _HUST_SOURCE.format("tool path data3.txt"),
        "reference": None,
    },
    "impeller_blade_coarse": {
        "file": "impeller_blade_coarse.npz",
        "description": "52 点，叶轮叶片单道走刀，刀轴 20°→90°",
        "source": _HUST_SOURCE.format("yelun.txt"),
        "reference": None,
    },
    "impeller_blade_dense": {
        "file": "impeller_blade_dense.npz",
        "description": "326 点，与 impeller_blade_coarse 同一条刀路的密采样版",
        "source": _HUST_SOURCE.format("CurvePoints.mat"),
        "reference": None,
    },
    "closed_loop_smooth": {
        "file": "closed_loop_smooth.npz",
        "description": "395 点，闭合圆角三角形环，起终点重合，刀轴 8°–38°",
        "source": _HUST_SOURCE.format("pathdata2.txt"),
        "reference": None,
    },
    "pocket_planar_down_axis": {
        "file": "pocket_planar_down_axis.npz",
        "description": "101 点，z = 0 平面开口袋 S 曲线；注意刀轴朝下（k ≈ −0.95），与其余数据的符号约定相反",
        "source": _HUST_SOURCE.format("Open pocket curve.txt"),
        "reference": None,
    },
    "line_gradual_sweep72": {
        "file": "line_gradual_sweep72.npz",
        "description": "39 点，对角直线，刀轴全程匀速扫掠 8°→72°",
        "source": _HUST_SOURCE.format("PathPointsFile.cls"),
        "reference": None,
    },
    "line_end_snap_sweep90": {
        "file": "line_end_snap_sweep90.npz",
        "description": "39 点，同一条直线；刀轴前 95% 行程竖直，末端 0.5 mm 内急转到 90°（逆解分支压力测试）",
        "source": _HUST_SOURCE.format("Commands.mat"),
        "reference": None,
    },
    "blade_sample": {
        "file": "blade_sample.npz",
        "description": "2363 点，叶片精加工刀轨（大规模插补测试）",
        "source": "旧 cnc5x 数据集 five_axis/blade_sample.clean.npz；原始出处未核实",
        "reference": None,
    },
}
