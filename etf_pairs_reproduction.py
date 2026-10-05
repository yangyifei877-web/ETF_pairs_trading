"""ETF pairs-trading study reconstructed from the supplied JupyterLab PDF.

The PDF was a browser printout, so several long lines were clipped at the right
margin.  This version restores those lines, removes duplicated notebook cells,
and keeps the original research flow in a reusable, testable form.

VS Code: open this file and use "Run Cell" on the # %% blocks.
Terminal examples:
    python etf_pairs_reproduction.py --mode smoke
    python etf_pairs_reproduction.py --mode quick
    python etf_pairs_reproduction.py --mode full --analyses pc1 bootstrap sensitivity
"""

# %% Imports and reproducibility settings
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)
RNG_SEED = 486
np.random.seed(RNG_SEED)


# %% ETF universe reconstructed from the PDF
# Duplicates are removed below while preserving order.  The printed PDF clipped
# a few symbols at page boundaries; only complete, valid symbols are retained.
PDF_TICKERS = """
IVV VTI ITOT SCHB DIA QQQ QQQM IJH MDY VO IWR IJR VB IWM IWC
IWF IWD VUG VTV IWP IWS VOE VBR IWO IWN USMV SPLV MTUM QUAL SIZE VLUE RPV RZG RZV
VIG VYM SCHD DVY SDY NOBL HDV SPYD RSP SCHA SCHG SCHV SCHM SCHX VV VONE VTHR OEF
XLB XLE XLF XLI XLK XLP XLU XLV XLY XLC XLRE RYT RYE RYF RGI RTH RYH RHS RYU RCD RTM EWRE
SMH SOXX SOXQ XSD FDN IGV HACK CIBR SKYY BOTZ ROBO IBB XBI IHE IHI ITA XAR IYT IYJ
XME SLX XOP OIH ITB XHB XRT IYR VNQ ICF SCHH KBE KRE IAI IYF IYZ XTL FTEC VGT XT PAVE
ACWI ACWX VT VEU VEA IEFA EFA VGK IEV VPL EPP VSS SCZ IEUS DLS EFV IDEV IQLT IXUS
EWJ EWG EWU EWQ EWC EWA EWL EWH EWS EWP EWD EWN EWI EIRL EWO EWK ENZL PGAL EDEN EFNL ENOR GREK ECH
EEM VWO IEMG EWZ EWT EWY EWW EWM EZA FXI MCHI ASHR KWEB CQQQ INDA EPI TUR THD EIDO EPHE VNM EGPT EIS FM
KSA QAT UAE KWT PAK ARGT GXG GULF GXC KBA NGE EPU
DXJ HEFA HEDJ
BND AGG SCHZ BNDW GOVT SGOV SHY SCHO VGSH SPTS IEI IEF SPTI VGIT TLT SPTL VGLT ZROZ EDV
LQD IGSB IGIB VCIT VCSH HYG JNK USHY SJNK ANGL HYLB PHB HYS SPHY SPLB SPSB SPBO EMB
TIP SCHP STIP VTIP MBB MUB TFI VTEB HYD SHM SMMU SUB CMF NYF TFSA SHYD BKLN SRLN
PFF PFFD PFXF PFFA PREF PGX PGF VRP IUSB JPST ICSH NEAR MINT GSY ULST
REET RWR VNQI FREL PPTY NETL HOMZ RWO REM
DBC PDBC COMT BCI CMDY GSP GSG GCC IAU GLD SLV PPLT PALL OUNZ GLTR SGOL USO BNO UNG UGA
DBB CPER JJM DBA WEAT CORN SOYB CANE JO NIB SGG COW KRBN GRN
GDX GDXJ SIL SILJ COPX LIT REMX URA
ARKK ARKG ARKW ICLN TAN PBW FAN PHO FIW CLOU WCLD FIVG IBUY FINX IPAY JETS BETZ ESPO HERO UFO
MOAT OMFL PPA QQQJ QQQS BUG
VFMO VFQY VFMF VFMV AVUV AVUS AVLV AVDV AVDE AVEM DFAU DFUS DFIV DFHY DFEM DFAT DFUV
JQUA JMIN JVAL JHML JPSE JPRE COWZ CALF BBJP
UUP FXE FXB FXY FXF CEW
IAK IHF IYG IYE IYM IEO IEZ IETC IUSG IUSV XTN XHE XHS XPH XNTK XWEB
KBWB KBWY KBWR PBJ PEJ PXJ PXE PBE PSP PKW PSCT PSCD PSCM PSCC PSCE PSCQ
SCHK ESGV SUSA VHT VIOO VIOV EUSA FBT IFRA ILCG ILCV IYW PCEF PSJ
""".split()
PDF_TICKERS = list(dict.fromkeys(PDF_TICKERS))

QUICK_TICKERS = """
SPY IVV VTI SCHB QQQ DIA IWM IJR MDY VUG VTV IWF IWD RSP USMV
XLB XLE XLF XLI XLK XLP XLU XLV XLY SMH SOXX IBB XBI ITA XAR
EFA VEA IEFA VWO EEM IEMG VNQ IYR GLD IAU SLV GDX
BND SCHZ IEF TLT LQD HYG SHY VGSH TIP
""".split()


@dataclass(frozen=True)
class RunConfig:
    mode: str = "smoke"
    start: str = "2000-01-03"
    end: str = "2026-04-17"  # PDF was generated on 2026-04-17.
    min_obs: int = 3000
    top_n: int = 10
    formation_days: int = 252
    holding_days: int = 126
    pc1_mode: str = "remove"
    output_dir: Path = Path("results")


# %% Data acquisition and panel construction
def tidy_from_yf(df: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance output (either MultiIndex orientation) to tidy prices."""
    empty = pd.DataFrame(columns=["date", "ticker", "price"])
    if df is None or df.empty:
        return empty

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        if "Adj Close" in lvl0 or "Close" in lvl0:
            field = "Adj Close" if "Adj Close" in lvl0 else "Close"
            prices = df[field]
        elif "Adj Close" in lvl1 or "Close" in lvl1:
            field = "Adj Close" if "Adj Close" in lvl1 else "Close"
            prices = df.xs(field, axis=1, level=1)
        else:
            return empty
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        prices.index.name = "date"
        out = prices.stack(dropna=True).rename("price").reset_index()
        out.columns = ["date", "ticker", "price"]
    else:
        field = "Adj Close" if "Adj Close" in df.columns else "Close" if "Close" in df.columns else None
        if field is None:
            return empty
        out = df[[field]].rename(columns={field: "price"}).reset_index()
        out["ticker"] = "UNKNOWN"

    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = out["ticker"].astype(str)
    return out[["date", "ticker", "price"]].sort_values(["ticker", "date"]).reset_index(drop=True)


def download_panel(tickers: Iterable[str], start: str, end: str, cache_file: Path) -> pd.DataFrame:
    """Download adjusted prices once, then reuse a local pickle cache."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        panel = pd.read_pickle(cache_file)
        print(f"Loaded cache: {cache_file} ({len(panel):,} rows)")
        return panel

    import yfinance as yf

    requested = list(dict.fromkeys(tickers))
    raw = yf.download(
        tickers=requested,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=True,
        group_by="column",
        threads=True,
    )
    panel = tidy_from_yf(raw)
    panel = panel[panel["ticker"].isin(requested)].copy()
    panel["pct_change"] = panel.groupby("ticker")["price"].pct_change(fill_method=None)
    panel.to_pickle(cache_file)
    print(f"Saved cache: {cache_file} ({len(panel):,} rows)")
    return panel


def make_synthetic_panel(n_days: int = 900, n_groups: int = 4, per_group: int = 5) -> pd.DataFrame:
    """Deterministic offline panel for installation and pipeline verification."""
    rng = np.random.default_rng(RNG_SEED)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    rows = []
    market = rng.normal(0.00025, 0.007, n_days)
    for group in range(n_groups):
        factor = market + rng.normal(0, 0.0035, n_days)
        common_level = 80 + 15 * group + np.cumsum(factor)
        for j in range(per_group):
            ticker = f"G{group + 1}E{j + 1}"
            stationary = np.zeros(n_days)
            noise = rng.normal(0, 0.15, n_days)
            for i in range(1, n_days):
                stationary[i] = 0.82 * stationary[i - 1] + noise[i]
            price = np.maximum(5, common_level * (1 + 0.03 * j) + stationary)
            frame = pd.DataFrame({"date": dates, "ticker": ticker, "price": price})
            rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel["pct_change"] = panel.groupby("ticker")["price"].pct_change(fill_method=None)
    return panel


def common_sample(panel: pd.DataFrame, min_obs: int) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Find a large ticker set with at least min_obs common trading dates."""
    clean = panel.dropna(subset=["date", "ticker", "price", "pct_change"]).copy()
    ranges = clean.groupby("ticker")["date"].agg(first="min", last="max")
    keep = ranges.index.tolist()
    all_dates = np.sort(clean["date"].unique())

    while keep:
        cur = ranges.loc[keep]
        start_common = pd.Timestamp(cur["first"].max())
        end_common = pd.Timestamp(cur["last"].min())
        n_days = int(((all_dates >= start_common) & (all_dates <= end_common)).sum())
        if start_common <= end_common and n_days >= min_obs:
            break
        latest_starter = cur["first"].idxmax()
        earliest_ender = cur["last"].idxmin()
        # Remove whichever boundary is more restrictive relative to the median.
        start_gap = (cur.loc[latest_starter, "first"] - cur["first"].median()).days
        end_gap = (cur["last"].median() - cur.loc[earliest_ender, "last"]).days
        keep.remove(latest_starter if start_gap >= end_gap else earliest_ender)
    else:
        raise ValueError("No common sample satisfies min_obs")

    out = clean[
        clean["ticker"].isin(keep)
        & clean["date"].between(start_common, end_common)
    ].copy()
    print(f"Common sample: {len(keep)} tickers | {start_common.date()} -> {end_common.date()} | {n_days} days")
    return out, start_common, end_common


# %% Formation-window clustering and pair features
def get_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))


def build_formation_data(panel, f_start, f_end, min_coverage=0.8):
    formation = panel[panel["date"].between(f_start, f_end)].copy()
    if formation.empty:
        return formation, []
    n_days = formation["date"].nunique()
    counts = formation.groupby("ticker")["date"].count()
    eligible = counts[counts >= int(np.ceil(min_coverage * n_days))].index.tolist()
    return formation[formation["ticker"].isin(eligible)].copy(), eligible


def cluster_pairs_for_period(formation_panel, pc1_mode="remove", min_cluster_size=3, min_samples=2):
    empty_pairs = pd.DataFrame(columns=["cluster", "ticker1", "ticker2"])
    empty_clusters = pd.DataFrame(columns=["ticker", "cluster"])
    if formation_panel.empty:
        return empty_pairs, empty_clusters

    X = formation_panel.pivot(index="ticker", columns="date", values="pct_change").fillna(0.0)
    if X.shape[0] < max(3, min_cluster_size):
        return empty_pairs, empty_clusters

    X_scaled = StandardScaler().fit_transform(X)
    X_proc = X_scaled.copy()
    if pc1_mode == "remove" and min(X_scaled.shape) >= 2:
        ncomp = min(10, *X_scaled.shape)
        pca = PCA(n_components=ncomp, random_state=RNG_SEED)
        scores = pca.fit_transform(X_scaled)
        X_proc = X_scaled - scores[:, [0]] @ pca.components_[[0], :]
    elif pc1_mode != "keep":
        raise ValueError("pc1_mode must be 'remove' or 'keep'")

    distance = pairwise_distances(X_proc, metric="correlation")
    distance = np.nan_to_num(distance, nan=1.0, posinf=2.0, neginf=0.0)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0.0)

    import hdbscan

    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
    ).fit_predict(distance)
    clusters = pd.DataFrame({"ticker": X.index, "cluster": labels})
    pairs = []
    for cluster_id, group in clusters[clusters["cluster"] != -1].groupby("cluster"):
        for t1, t2 in combinations(group["ticker"], 2):
            pairs.append({"cluster": int(cluster_id), "ticker1": t1, "ticker2": t2})
    return pd.DataFrame(pairs, columns=empty_pairs.columns), clusters


def hurst_exponent(series):
    x = pd.Series(series).dropna().values
    if len(x) < 50:
        return np.nan
    lags = np.arange(2, min(40, len(x) // 2))
    tau = np.maximum([np.std(x[lag:] - x[:-lag]) for lag in lags], 1e-8)
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0)


def ou_half_life(spread):
    x = pd.Series(spread).dropna()
    if len(x) < 50:
        return np.inf
    ds = x.diff().dropna()
    lag = x.shift(1).loc[ds.index]
    beta = float(OLS(ds.values, lag.values).fit().params.item())
    return np.inf if beta >= 0 else float(-np.log(2) / beta)


def returns_beta(p1, p2):
    r1, r2 = p1.pct_change().dropna(), p2.pct_change().dropna()
    common = r1.index.intersection(r2.index)
    if len(common) < 50:
        return np.nan, np.nan
    r1, r2 = r1.loc[common], r2.loc[common]
    beta = float(OLS(r2.values, sm.add_constant(r1.values)).fit().params[1])
    w1, w2 = abs(beta) / (abs(beta) + 1), 1 / (abs(beta) + 1)
    return beta, float((w2 * r2 - w1 * r1).std())


def estimate_pair_features(p1, p2):
    model = OLS(p2, sm.add_constant(p1)).fit()
    intercept, beta = float(model.params.iloc[0]), float(model.params.iloc[1])
    spread = p2 - (intercept + beta * p1)
    spread_std = float(spread.std())
    beta_ret, vol_spread_ret = returns_beta(p1, p2)
    if not np.isfinite(spread_std) or spread_std <= 0 or not np.isfinite(beta_ret) or vol_spread_ret <= 0:
        return None
    return {
        "beta_price": beta,
        "intercept_price": intercept,
        "spread_mean": float(spread.mean()),
        "spread_std": spread_std,
        "beta_ret": beta_ret,
        "vol_spread_ret": vol_spread_ret,
        "half_life": ou_half_life(spread),
        "hurst": hurst_exponent(spread),
        "adf_pval": float(adfuller(spread, autolag="AIC")[1]),
        "spread_series": spread,
    }


def _pair_inputs(px, row, min_overlap=0.8):
    t1, t2 = row.ticker1, row.ticker2
    if t1 not in px.columns or t2 not in px.columns:
        return None
    pair = px[[t1, t2]].dropna()
    if len(pair) < max(50, int(min_overlap * len(px))):
        return None
    return pair.iloc[:, 0], pair.iloc[:, 1]


def select_pairs_for_period(formation_panel, pairs_df, method="cointegration", entry_z=0.75, top_n=10, cov_window=60):
    if formation_panel.empty or pairs_df.empty:
        return pd.DataFrame()
    px = formation_panel.pivot(index="date", columns="ticker", values="price")
    selected = []
    for row in pairs_df.itertuples(index=False):
        inputs = _pair_inputs(px, row)
        if inputs is None:
            continue
        p1, p2 = inputs
        feat = estimate_pair_features(p1, p2)
        if feat is None:
            continue

        if method == "cointegration":
            z = (feat["spread_series"] - feat["spread_mean"]) / feat["spread_std"]
            if feat["adf_pval"] >= 0.10 or (z.abs() > entry_z).mean() < 0.01:
                continue
            if not (0.20 < feat["hurst"] < 0.50 and 2 < feat["half_life"] < 60):
                continue
            score, ascending = feat["half_life"], True
        elif method == "distance":
            r = pd.concat([p1.pct_change(), p2.pct_change()], axis=1).dropna()
            score, ascending = 1.0 - float(r.corr().iloc[0, 1]), True
        elif method == "covariance":
            r = pd.concat([p1.pct_change(), p2.pct_change()], axis=1).dropna()
            cov = r.iloc[:, 0].rolling(cov_window).cov(r.iloc[:, 1]).dropna()
            if len(cov) < 30 or not np.isfinite(cov.std()):
                continue
            score, ascending = abs(float(cov.mean())) / (float(cov.std()) + 1e-8), False
        else:
            raise ValueError(f"Unknown method: {method}")

        selected.append({
            "ticker1": row.ticker1,
            "ticker2": row.ticker2,
            "cluster": row.cluster,
            "score": score,
            "selector_method": method,
            **{k: v for k, v in feat.items() if k != "spread_series"},
        })
    if not selected:
        return pd.DataFrame()
    return pd.DataFrame(selected).sort_values("score", ascending=ascending).head(top_n).reset_index(drop=True)


# %% Trading and walk-forward backtest
def trade_pair(prices, beta_price, intercept_price, spread_mean, spread_std, beta_ret, vol_spread_ret,
               entry_z=2.0, exit_z=1.5, stop_z=3.0):
    prices = prices.dropna()
    if len(prices) < 10:
        return pd.Series(0.0, index=prices.index)
    p1, p2 = prices.iloc[:, 0], prices.iloc[:, 1]
    z = ((p2 - (intercept_price + beta_price * p1)) - spread_mean) / spread_std
    signal = z.shift(1)  # strict no-lookahead: today's trade uses yesterday's z-score
    r1, r2 = p1.pct_change().fillna(0.0), p2.pct_change().fillna(0.0)
    w1, w2 = abs(beta_ret) / (abs(beta_ret) + 1), 1 / (abs(beta_ret) + 1)
    spread_ret = ((w2 * r2 - w1 * r1) / vol_spread_ret) * (0.10 / np.sqrt(252))

    pos, out = 0, np.zeros(len(prices))
    for i, zi in enumerate(signal):
        if np.isfinite(zi):
            if pos == 0:
                if zi <= -entry_z:
                    pos = 1
                elif zi >= entry_z:
                    pos = -1
            elif (zi <= -stop_z and pos == 1) or (zi >= stop_z and pos == -1) or abs(zi) <= exit_z:
                pos = 0
        out[i] = pos * spread_ret.iloc[i]
    return pd.Series(out, index=prices.index)


def trade_period(panel, selected_pairs, t_start, t_end, entry_z=2.0, exit_z=1.5):
    if selected_pairs.empty:
        return pd.Series(dtype=float), 0.0
    hold = panel[panel["date"].between(t_start, t_end)]
    px = hold.pivot(index="date", columns="ticker", values="price")
    pair_returns = []
    for row in selected_pairs.itertuples(index=False):
        if row.ticker1 not in px or row.ticker2 not in px:
            continue
        prices = px[[row.ticker1, row.ticker2]].dropna()
        if len(prices) < 10:
            continue
        pair_returns.append(trade_pair(
            prices, row.beta_price, row.intercept_price, row.spread_mean, row.spread_std,
            row.beta_ret, row.vol_spread_ret, entry_z, exit_z, 3.0,
        ))
    if not pair_returns:
        return pd.Series(dtype=float), 0.0
    mat = pd.concat(pair_returns, axis=1).fillna(0.0)
    active = (mat != 0).astype(float)
    weights = active.div(active.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return (mat * weights).sum(axis=1), float(active.sum(axis=1).mean())


def compute_perf_summary(port):
    if port is None or len(pd.Series(port).dropna()) == 0:
        return {"days": 0, "total_ret": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "win_rate": np.nan}
    r = pd.Series(port).dropna()
    wealth = (1 + r).cumprod()
    return {
        "days": int(len(r)),
        "total_ret": float(wealth.iloc[-1] - 1),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
        "win_rate": float((r > 0).mean()),
    }


def backtest_walkforward(panel, method="cointegration", top_n=10, formation_days=252, holding_days=126,
                         entry_z=2.0, exit_z=1.5, min_coverage=0.8, pc1_mode="remove",
                         min_cluster_size=3, min_samples=2, verbose=True):
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    dates = get_dates(panel)
    portfolio, stats = [], []
    i, period_idx = 0, 0
    while i + formation_days + holding_days <= len(dates):
        period_idx += 1
        f_start, f_end = dates[i], dates[i + formation_days - 1]
        t_start, t_end = dates[i + formation_days], dates[i + formation_days + holding_days - 1]
        assert f_end < t_start
        formation, eligible = build_formation_data(panel, f_start, f_end, min_coverage)
        pairs, clusters = cluster_pairs_for_period(formation, pc1_mode, min_cluster_size, min_samples)
        selected = select_pairs_for_period(formation, pairs, method, 0.75, top_n)
        clustered = clusters.loc[clusters["cluster"] != -1, "ticker"].nunique() if not clusters.empty else 0
        if verbose:
            print(f"[{method}/{pc1_mode}] {period_idx}: eligible={len(eligible)} clustered={clustered} "
                  f"pairs={len(pairs)} selected={len(selected)}")
        if not selected.empty:
            port, avg_active = trade_period(panel, selected, t_start, t_end, entry_z, exit_z)
            if not port.empty:
                perf = compute_perf_summary(port)
                stats.append({
                    "method": method, "period": period_idx,
                    "formation_start": f_start, "formation_end": f_end,
                    "holding_start": t_start, "holding_end": t_end,
                    "eligible_tickers": len(eligible), "candidate_pairs": len(pairs),
                    "selected_pairs": len(selected), "avg_active_pairs": avg_active,
                    "ret": float(port.sum()), "vol": float(port.std()),
                    "sharpe": perf["sharpe"], "win_rate": perf["win_rate"],
                })
                portfolio.append(port)
        i += holding_days
    if not portfolio:
        return None, pd.DataFrame(stats)
    return pd.concat(portfolio).sort_index(), pd.DataFrame(stats)


# %% Comparisons, bootstrap/permutation null, and sensitivity analysis
def compare_three_methods(panel, methods=("cointegration", "distance", "covariance"), **kwargs):
    portfolios, period_stats, rows = {}, {}, []
    for method in methods:
        port, stats = backtest_walkforward(panel, method=method, **kwargs)
        portfolios[method], period_stats[method] = port, stats
        rows.append({"method": method, **compute_perf_summary(port)})
    return portfolios, period_stats, pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def moving_block_bootstrap_zero_mean(r, block_size=20, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    values = pd.Series(r).dropna().to_numpy(float)
    if len(values) == 0:
        return values
    centered = values - values.mean()
    if len(values) <= block_size:
        return centered[rng.integers(0, len(values), size=len(values))]
    starts, out = np.arange(len(values) - block_size + 1), []
    while len(out) < len(values):
        s = rng.choice(starts)
        out.extend(centered[s:s + block_size])
    return np.asarray(out[:len(values)])


def sign_permutation_blocks_zero_mean(r, block_size=20, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    values = pd.Series(r).dropna().to_numpy(float)
    out = values - values.mean()
    for start in range(0, len(out), block_size):
        out[start:start + block_size] *= rng.choice([-1.0, 1.0])
    return out


def empirical_pvalue_greater(observed, simulations):
    sims = np.asarray(simulations, float)
    sims = sims[np.isfinite(sims)]
    return np.nan if not np.isfinite(observed) or len(sims) == 0 else float((1 + (sims >= observed).sum()) / (len(sims) + 1))


def evaluate_null_baselines_for_portfolio(port, n_boot=500, block_size=20, seed=123):
    obs = compute_perf_summary(port)
    rng, boot, perm = np.random.default_rng(seed), [], []
    for _ in range(n_boot):
        boot.append(compute_perf_summary(moving_block_bootstrap_zero_mean(port, block_size, rng)))
        perm.append(compute_perf_summary(sign_permutation_blocks_zero_mean(port, block_size, rng)))
    return {
        "Obs Sharpe": obs["sharpe"], "Obs Total Return": obs["total_ret"],
        "Obs Max Drawdown": obs["max_drawdown"], "Days": obs["days"],
        "Boot Sharpe Mean": np.nanmean([x["sharpe"] for x in boot]),
        "Perm Sharpe Mean": np.nanmean([x["sharpe"] for x in perm]),
        "p_boot_sharpe": empirical_pvalue_greater(obs["sharpe"], [x["sharpe"] for x in boot]),
        "p_boot_total_ret": empirical_pvalue_greater(obs["total_ret"], [x["total_ret"] for x in boot]),
        "p_perm_sharpe": empirical_pvalue_greater(obs["sharpe"], [x["sharpe"] for x in perm]),
        "p_perm_total_ret": empirical_pvalue_greater(obs["total_ret"], [x["total_ret"] for x in perm]),
    }


def run_pc1_comparison(panel, **kwargs):
    rows, portfolios_by_mode, stats_by_mode = [], {}, {}
    for mode in ("remove", "keep"):
        portfolios, stats, _ = compare_three_methods(panel, pc1_mode=mode, **kwargs)
        portfolios_by_mode[mode], stats_by_mode[mode] = portfolios, stats
        for method, port in portfolios.items():
            rows.append({"PC1 Mode": mode, "Method": method.title(), **compute_perf_summary(port)})
    return portfolios_by_mode, stats_by_mode, pd.DataFrame(rows)


def sensitivity_analysis(panel, pc1_mode="keep", **kwargs):
    configs = [("6m/3m", 126, 63), ("12m/6m", 252, 126), ("18m/6m", 378, 126), ("12m/3m", 252, 63)]
    rows = []
    for label, formation, holding in configs:
        for method in ("cointegration", "distance", "covariance"):
            port, _ = backtest_walkforward(
                panel, method=method, formation_days=formation, holding_days=holding,
                pc1_mode=pc1_mode, verbose=False, **kwargs,
            )
            rows.append({"Window": label, "Method": method.title(), **compute_perf_summary(port)})
    return pd.DataFrame(rows)


# %% Reporting
def save_equity_plot(portfolios: dict[str, pd.Series], path: Path, title: str):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, port in portfolios.items():
        if port is not None and len(port):
            (1 + port.dropna()).cumprod().plot(ax=ax, label=name.title(), linewidth=1.8)
    ax.set(title=title, xlabel="Date", ylabel="Growth of $1")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(config: RunConfig, analyses: tuple[str, ...] = ()) -> dict:
    out = config.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if config.mode == "smoke":
        panel = make_synthetic_panel()
        panel_bt, _, _ = common_sample(panel, min_obs=700)
        formation_days, holding_days, top_n = 252, 126, 5
    else:
        tickers = QUICK_TICKERS if config.mode == "quick" else PDF_TICKERS
        cache = out / f"prices_{config.mode}_{config.start}_{config.end}.pkl"
        panel = download_panel(tickers, config.start, config.end, cache)
        requested_min = min(config.min_obs, max(252, panel["date"].nunique() - 5)) if config.mode == "quick" else config.min_obs
        panel_bt, _, _ = common_sample(panel, requested_min)
        formation_days, holding_days, top_n = config.formation_days, config.holding_days, config.top_n

    portfolios, period_stats, summary = compare_three_methods(
        panel_bt,
        top_n=top_n,
        formation_days=formation_days,
        holding_days=holding_days,
        entry_z=2.0,
        exit_z=1.5,
        min_coverage=0.8,
        pc1_mode=config.pc1_mode,
        min_cluster_size=3,
        min_samples=2,
        verbose=True,
    )
    summary.to_csv(out / "method_comparison.csv", index=False)
    save_equity_plot(portfolios, out / "equity_curves.png", f"ETF pairs strategies ({config.mode})")
    for method, stats in period_stats.items():
        stats.to_csv(out / f"period_stats_{method}.csv", index=False)

    result = {"summary": summary.to_dict(orient="records")}
    common_kwargs = dict(
        top_n=top_n, formation_days=formation_days, holding_days=holding_days,
        entry_z=2.0, exit_z=1.5, min_coverage=0.8,
        min_cluster_size=3, min_samples=2, verbose=False,
    )
    if "pc1" in analyses:
        modes, _, pc1_table = run_pc1_comparison(panel_bt, **common_kwargs)
        pc1_table.to_csv(out / "pc1_comparison.csv", index=False)
        result["pc1"] = pc1_table.to_dict(orient="records")
    else:
        modes = {config.pc1_mode: portfolios}

    if "bootstrap" in analyses:
        rows = []
        for mode, method_ports in modes.items():
            for method, port in method_ports.items():
                if port is not None:
                    rows.append({"PC1 Mode": mode, "Method": method.title(), **evaluate_null_baselines_for_portfolio(port)})
        baseline = pd.DataFrame(rows)
        baseline.to_csv(out / "bootstrap_significance.csv", index=False)
        result["bootstrap"] = baseline.to_dict(orient="records")

    if "sensitivity" in analyses:
        sensitivity = sensitivity_analysis(
            panel_bt, pc1_mode="keep", top_n=top_n, entry_z=2.0, exit_z=1.5,
            min_coverage=0.8, min_cluster_size=3, min_samples=2,
        )
        sensitivity.to_csv(out / "window_sensitivity.csv", index=False)
        result["sensitivity"] = sensitivity.to_dict(orient="records")

    (out / "run_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("\n", summary.to_string(index=False))
    print(f"\nResults written to {out}")
    return result


# %% Terminal entry point (does not auto-run inside Jupyter)
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quick", "full"), default="smoke")
    parser.add_argument("--start", default="2000-01-03")
    parser.add_argument("--end", default="2026-04-17")
    parser.add_argument("--min-obs", type=int, default=3000)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--formation-days", type=int, default=252)
    parser.add_argument("--holding-days", type=int, default=126)
    parser.add_argument("--pc1-mode", choices=("remove", "keep"), default="remove")
    parser.add_argument("--analyses", nargs="*", choices=("pc1", "bootstrap", "sensitivity"), default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


if "__file__" in globals() and __name__ == "__main__":
    args = parse_args()
    run(
        RunConfig(
            mode=args.mode,
            start=args.start,
            end=args.end,
            min_obs=args.min_obs,
            top_n=args.top_n,
            formation_days=args.formation_days,
            holding_days=args.holding_days,
            pc1_mode=args.pc1_mode,
            output_dir=args.output_dir,
        ),
        analyses=tuple(args.analyses),
    )

