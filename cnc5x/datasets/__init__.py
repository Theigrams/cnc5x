"""内置数据：平面 G01 基准轮廓与五轴刀位。

    list_datasets()        所有数据名
    load_dataset(name)     → Dataset(name, points, axes, info)

每组数据的来源写在 registry.py 里。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..geometry import unit
from ..io import read_cl
from .registry import REGISTRY

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Dataset:
    """一组刀位：points (N, 2) 或 (N, 3)；五轴数据另有单位刀轴 axes (N, 3)。"""

    name: str
    points: np.ndarray
    axes: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)


def list_datasets():
    return sorted(REGISTRY)


def load_dataset(name):
    if name not in REGISTRY:
        raise KeyError(f"没有数据 {name!r}，可选：{', '.join(list_datasets())}")
    info = REGISTRY[name]
    path = DATA_DIR / info["file"]
    if path.suffix == ".npz":
        data = np.load(path)
        axes = data["orientations"] if "orientations" in data.files else data["axes"]
        return Dataset(name, data["positions"], unit(axes), info)
    points, axes = read_cl(path)
    return Dataset(name, points, axes, info)
