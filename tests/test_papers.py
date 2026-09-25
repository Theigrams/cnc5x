"""论文复现（papers/）的回归测试：几何性质与整条流水线。"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

import cnc5x as cx
from cnc5x import metrics
from papers.Xu2018.algorithm import CcrPath
from papers.Zhao2013.algorithm import SmoothedPath

V_MAX, A_MAX, J_MAX, TS = 100, 3000, 60000, 0.0005


def make_paths():
    rhombic = cx.datasets.load_dataset("rhombic").points
    butterfly = cx.datasets.load_dataset("butterfly").points
    return [
        ("Zhao2013-rhombic", SmoothedPath(rhombic, 0.2, 0.5), 5),
        ("Xu2018-rhombic", CcrPath(rhombic, 0.2), 7),
        ("Zhao2013-butterfly", SmoothedPath(butterfly, 0.02, 0.5), 5),
        ("Xu2018-butterfly", CcrPath(butterfly, 0.02), 7),
    ]


@pytest.mark.parametrize("name, path, phases", make_paths())
def test_corner_error_and_g2(name, path, phases):
    # 过渡曲线关于角平分线对称，离顶点最近的是两半过渡的连接点，也就是 block 的分界点
    junctions = np.array([block.curves[-1].end_point for block in path.blocks[:-1]])
    distance = np.linalg.norm(junctions - path.points[1:-1], axis=1)
    assert np.allclose(distance, path.chord_errors, rtol=1e-9)
    assert np.all(path.chord_errors <= (0.2 if "rhombic" in name else 0.02) * (1 + 1e-9))
    # 所有曲线段之间 G² 连续：位置、单位切向、曲率向量都没有跳变
    assert metrics.junction_jumps(path.curves).max() < 1e-6


def test_junction_is_the_closest_point_to_the_corner():
    rhombic = cx.datasets.load_dataset("rhombic").points
    for path in (SmoothedPath(rhombic, 0.2, 0.5), CcrPath(rhombic, 0.2)):
        distance, _ = cKDTree(path.sample(200001)).query(path.points[1:-1])  # 菱形路径短，采样间距约 6e-5
        assert np.allclose(distance, path.chord_errors, atol=1e-4)


@pytest.mark.parametrize("name, path, phases", make_paths())
def test_pipeline_respects_limits(name, path, phases):
    profile, v = cx.schedule(path, V_MAX, A_MAX, J_MAX, TS, phases=phases)
    commands = cx.interpolate(path, profile, TS)
    feed, acceleration, jerk = metrics.tangential(commands.tip)
    assert np.nanmax(feed) <= V_MAX * (1 + 1e-9)
    assert np.nanmax(np.abs(acceleration)) <= A_MAX * (1 + 1e-6)
    assert np.nanmax(np.abs(jerk)) <= J_MAX * (1 + 1e-6)
    assert np.all(v <= path.get_v_limit(TS, V_MAX, A_MAX, J_MAX) + 1e-9)
    assert np.allclose(commands.position[[0, -1]], path.points[[0, -1]])
