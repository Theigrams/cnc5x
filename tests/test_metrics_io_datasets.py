import numpy as np
import pytest

from cnc5x import NURBS, CurvePath, DriveLimits, Line, datasets, interpolate, metrics, read_cl, schedule


def test_path_deviation():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], float)
    points = np.array([[5, 1], [5, 5], [12, 5], [-1, -1], [10, 10]], float)
    assert np.allclose(metrics.path_deviation(points, square), [1, 5, 2, np.sqrt(2), 0])


def test_corner_error():
    vertices = np.array([[0, 0], [1, 0]])
    points = np.array([[0, 0.3], [1.0, 0.1], [5, 5]])
    assert np.allclose(metrics.corner_error(vertices, points), [0.3, 0.1])


def test_chord_error_on_circle_equals_sagitta():
    w = np.sqrt(0.5)
    circle = NURBS([[10, 0], [10, 10], [0, 10]], 2, [0, 0, 0, 1, 1, 1], [1, w, 1])  # 半径 10 的四分之一圆
    path = CurvePath(circle, chord_error=1e-3)
    s = np.linspace(0, path.length, 25)
    chord = 2 * 10 * np.sin((s[1] - s[0]) / 20)
    sagitta = 10 - np.sqrt(10**2 - (chord / 2) ** 2)
    assert np.allclose(metrics.chord_error(path, s), sagitta, rtol=1e-8)


def test_junction_jumps_detect_corner():
    curves = [Line([0, 0], [1, 0]), Line([1, 0], [1, 1]), Line([1, 1], [2, 1])]
    jumps = metrics.junction_jumps(curves)
    assert jumps.shape == (2, 3)
    assert np.allclose(jumps[:, 0], 0) and np.allclose(jumps[:, 1], np.sqrt(2)) and np.allclose(jumps[:, 2], 0)


def test_axis_report():
    q = np.zeros((4, 3, 2))
    q[1] = [[1, -2], [3, 0], [0, 1]]
    limits = DriveLimits([2, 4], [1, 1], [1, 1])
    report = metrics.axis_report(q, limits)
    assert np.allclose(report["velocity"]["peak"], [3, 2]) and np.allclose(report["velocity"]["ratio"], [1.5, 0.5])


def test_drive_limits_validation():
    with pytest.raises(ValueError):
        DriveLimits([1, 2], [1, 1], [1, -1])


def test_read_cl_formats(tmp_path):
    apt = tmp_path / "path.cls"
    apt.write_text("PARTNO/TEST\nGOTO/1.0,2.0,3.0,0,0,2\nFEDRAT/100\nGOTO/4,5,6,0,1,0\nFINI\n")
    points, axes = read_cl(apt)
    assert np.allclose(points, [[1, 2, 3], [4, 5, 6]]) and np.allclose(axes, [[0, 0, 1], [0, 1, 0]])
    plain = tmp_path / "path.txt"
    plain.write_text("x,y\n0.5\t1.5\n-2 3e-1\n")
    points, axes = read_cl(plain)
    assert np.allclose(points, [[0.5, 1.5], [-2, 0.3]]) and axes is None


def test_all_datasets_load():
    for name in datasets.list_datasets():
        data = datasets.load_dataset(name)
        assert data.points.ndim == 2 and len(data.points) >= 4
        if data.axes is not None:
            assert data.axes.shape == data.points.shape
            assert np.allclose(np.linalg.norm(data.axes, axis=1), 1)
    with pytest.raises(KeyError):
        datasets.load_dataset("no_such_data")


def test_five_axis_dataset_runs_through_pipeline():
    import cnc5x as cx

    data = datasets.load_dataset("horseshoe_planar_sweep")
    path = cx.CurvePath(cx.pose_spline(data.points, data.axes), chord_error=1e-3)
    profile, _ = schedule(path, v_max=50, a_max=500, j_max=5000, Ts=0.001)
    commands = interpolate(path, profile, Ts=0.001, machine=cx.TableTilting("AC"))
    assert commands.q.shape == (4, len(commands.t), 5) and np.all(np.isfinite(commands.q))
