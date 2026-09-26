"""打印 notebook 各单元格的源码与文字输出，用于审阅。

用法：python dump.py <文件名（不带 .ipynb）> [起始格 结束格]
"""

import sys
from pathlib import Path

import nbformat

NB_DIR = Path(__file__).resolve().parents[1]

nb = nbformat.read(NB_DIR / f"{sys.argv[1]}.ipynb", 4)
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, len(nb.cells))
for i, c in enumerate(nb.cells[lo:hi], lo):
    kind = "MD" if c.cell_type == "markdown" else "CODE"
    print(f"--- [{i}] {kind}")
    print(c.source)
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            print("   >>>", o["text"].rstrip())
        elif o.get("output_type") == "execute_result" and "text/plain" in o["data"]:
            print("   >>>", o["data"]["text/plain"][:300])
        elif "image/png" in o.get("data", {}):
            print("   >>> [figure]")
