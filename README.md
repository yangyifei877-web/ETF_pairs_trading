# ETF pairs trading

STAT 486 project - Yahoo Finance ETF pairs-trading reproduction.

This project reconstructs the code shown in the supplied 76-page JupyterLab PDF.
The PDF's long lines were clipped by the page boundary, so the code was restored
semantically and consolidated into a clean pipeline. It includes:

- Yahoo Finance ETF downloads with a local cache
- return-correlation clustering with HDBSCAN
- optional first-principal-component removal
- cointegration-, distance-, and covariance-based pair selection
- leakage-aware walk-forward formation/holding windows
- z-score entry/exit/stop rules and volatility-scaled returns
- PC1 comparison, block-bootstrap/permutation tests, and window sensitivity

## Run in VS Code on macOS

Open this folder in VS Code. In its Terminal, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Choose `.venv/bin/python` with **Python: Select Interpreter**. Then open
`etf_pairs_reproduction.py`; every `# %%` block has a **Run Cell** button.

## Terminal modes

```bash
# Offline deterministic validation (recommended first run)
python etf_pairs_reproduction.py --mode smoke

# Smaller live Yahoo Finance run
python etf_pairs_reproduction.py --mode quick

# PDF-scale universe and all research tables (slow)
python etf_pairs_reproduction.py --mode full --analyses pc1 bootstrap sensitivity
```

Live modes cache downloaded data under the chosen results directory. Delete the
specific `prices_*.pkl` file only when you intentionally want fresh market data.

The PDF was printed on 2026-04-17. The default end date is therefore fixed to
2026-04-17 so that later data is not silently added. Yahoo Finance can revise
historical prices, delist symbols, or change availability, so exact numerical
outputs may still differ from the printout.

## Notebook

`Yahoo_Finance_ETF_Pairs_Reproduced.ipynb` contains the same code split into
Jupyter cells. Select the `.venv` kernel, then run cells from top to bottom. The
last cell contains an explicit example; it is not run automatically.
