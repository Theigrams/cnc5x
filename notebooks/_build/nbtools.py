"""Build + execute a teaching notebook for cnc5x, then dump its figures for inspection.

Usage in a builder script (run from this directory so `nbtools` is importable):
    from nbtools import md, code, build
    cells = [md("# 标题"), code("x = 1"), ...]
    build("01_曲线与导数栈", cells)

The header cell (user-mandated) is inserted automatically as the first code cell after the title cell.
"""

import base64
import re
import subprocess
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "notebooks"
RUFF = "ruff"

HEADER = """import os.path as osp
import sys
from pathlib import Path

import matplotlib.pyplot as plt

root_path = Path(osp.abspath("")).parents[0]
sys.path.append(str(root_path))

%config InlineBackend.figure_format='retina'"""

IMPORTS = """import numpy as np

import cnc5x as cx
from cnc5x import plotting

plotting.use_style()"""


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbformat.v4.new_code_cell(text.strip("\n"))


def build(name, cells, timeout=600):
    """cells[0] must be the title markdown cell; header + imports cells are inserted after it."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [cells[0], code(HEADER), code(IMPORTS)] + list(cells[1:])
    for cell in nb.cells:  # 画图单元格统一以 plt.show() 结尾，免得输出里多出一行 Text(...) 或 Legend
        src = cell.source
        draws = "plt.subplots" in src or "plotting.plot_" in src or "plt.figure" in src
        if cell.cell_type == "code" and draws and "plt.show()" not in src:
            cell.source = src.rstrip().rstrip(";") + "\nplt.show()"

    path = NB_DIR / f"{name}.ipynb"
    nbformat.write(nb, path)
    subprocess.run([RUFF, "format", str(path)], cwd=ROOT, check=False)
    nb = nbformat.read(path, as_version=4)
    start = time.time()
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)
        print(f"executed in {time.time() - start:.1f} s -> {path}")
    lint = subprocess.run([RUFF, "check", str(path)], cwd=ROOT, capture_output=True, text=True)
    print("ruff check:", lint.stdout.strip() or "ok")
    dump_figures(nb, name)
    return nb


def dump_figures(nb, name):
    out = Path(tempfile.gettempdir()) / "cnc5x_nb" / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    count = 0
    for i, cell in enumerate(nb.cells):
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                count += 1
                (out / f"cell{i:02d}_{count:02d}.png").write_bytes(base64.b64decode(data["image/png"]))
            if output.get("output_type") == "stream" and re.search(r"Warning", output.get("text", "")):
                print(f"cell {i}: warning in output:", output["text"][:300])
    print(f"{count} figures dumped to {out}")
