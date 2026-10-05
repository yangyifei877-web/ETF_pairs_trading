"""Regenerate the Jupyter notebook from the VS Code percent-format source."""

from pathlib import Path
import re

import nbformat as nbf


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "etf_pairs_reproduction.py"
TARGET = HERE / "Yahoo_Finance_ETF_Pairs_Reproduced.ipynb"


def build_notebook():
    text = SOURCE.read_text(encoding="utf-8")
    parts = re.split(r"^# %%.*$", text, flags=re.MULTILINE)
    markers = re.findall(r"^# %%(.*)$", text, flags=re.MULTILINE)

    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3 (.venv)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python", "version": "3"}
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# Yahoo Finance ETF pairs-trading reproduction\n\n"
        "Reconstructed from the supplied JupyterLab PDF. Long lines clipped in the "
        "printout were restored and duplicate exploratory cells were consolidated. "
        "Start with the offline smoke test at the bottom; use `quick` or `full` only "
        "after installing dependencies and when internet access is available."
    ))

    preamble = parts[0].strip()
    if preamble:
        nb.cells.append(nbf.v4.new_code_cell(preamble))

    for marker, body in zip(markers, parts[1:]):
        body = body.strip()
        if not body:
            continue
        title = marker.strip()
        if title:
            nb.cells.append(nbf.v4.new_markdown_cell(f"## {title}"))
        nb.cells.append(nbf.v4.new_code_cell(body))

    nb.cells.append(nbf.v4.new_markdown_cell("## Run an analysis"))
    nb.cells.append(nbf.v4.new_code_cell(
        "# Offline validation (about a minute after dependencies are installed)\n"
        "config = RunConfig(mode='smoke', output_dir=Path('results'))\n"
        "results = run(config)\n"
        "results['summary']"
    ))
    nbf.write(nb, TARGET)
    print(f"Wrote {TARGET} with {len(nb.cells)} cells")


if __name__ == "__main__":
    build_notebook()

