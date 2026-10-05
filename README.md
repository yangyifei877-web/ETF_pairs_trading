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

## Reproduced results

The smaller live-data validation run (`--mode quick`) completed successfully on
2026-10-05. It downloaded 290,636 daily observations for 51 ETFs and produced a
common backtest sample from 2012-10-25 through 2026-04-16 (3,386 trading days).
The default walk-forward configuration used a 252-day formation window, a
126-day holding window, the top 10 pairs, and removal of the first principal
component before HDBSCAN clustering.

| Pair-selection method | Backtest days | Total return | Annualized Sharpe | Max drawdown | Positive-return days |
| --- | ---: | ---: | ---: | ---: | ---: |
| Distance | 3,024 | 301.75% | 1.514 | -10.47% | 50.26% |
| Covariance | 3,024 | 55.87% | 0.525 | -17.04% | 45.30% |
| Cointegration | 2,772 | -26.98% | -0.245 | -38.40% | 33.69% |

In this validation sample, distance-based selection produced the strongest
risk-adjusted result. These figures are a reproducibility check rather than an
investment claim: the backtest does not model transaction costs, bid-ask
spreads, market impact, taxes, or the full effects of survivorship and data
availability. Yahoo Finance can also revise historical observations. Re-running
the project may therefore produce somewhat different values.

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
