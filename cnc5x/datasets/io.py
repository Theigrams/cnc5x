"""读取刀位文件（CL data）。"""

import re
from pathlib import Path

import numpy as np

from ..utils.geometry import unit


def read_cl(path):
    """读刀位文件，返回 (points, axes)：points (N, D)；有刀轴时 axes (N, 3)，否则为 None。

    支持两种常见格式：
    - APT：GOTO/x,y,z,i,j,k（刀轴可省略），其余语句忽略
    - 纯数字：每行 2、3 或 6 个数，以空格、制表符或逗号分隔；文字开头的行（如表头）忽略
    """
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        goto = re.match(r"GOTO\s*/", line, re.IGNORECASE)  # "GOTO/" 与 "GOTO / " 两种写法都有
        if goto:
            line = line[goto.end() :]
        elif not line or not (line[0].isdigit() or line[0] in "+-."):
            continue
        rows.append([float(x) for x in line.replace(",", " ").split()])
    data = np.array(rows)
    if data.ndim != 2 or data.shape[1] not in (2, 3, 6):
        raise ValueError("每行应有 2、3 或 6 个数")
    if data.shape[1] == 6:
        return data[:, :3], unit(data[:, 3:])
    return data, None
