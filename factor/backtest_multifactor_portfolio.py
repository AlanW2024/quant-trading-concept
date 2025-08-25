# -*- coding: utf-8 -*-
"""
Offline Multi-Factor Backtest (ASCII-only logs)
Momentum (12M-1M) + QualityValue, monthly rebalance, daily holdings,
volatility targeting (10% annual, 20-day window), leverage cap 3x.
Benchmark: QQQ. Outputs: CSVs and a log-scale equity curve PNG.

Run with defaults (place caches next to this script):
  price_cache_1995-01-01_to_2024-12-31_504_tickers.csv
  fundamentals_cache_503_tickers.json
  python backtest_multifactor_portfolio_fixed.py

Or specify paths:
  python backtest_multifactor_portfolio_fixed.py \
    --prices "/path/price_cache_1995-01-01_to_2024-12-31_504_tickers.csv" \
    --fundamentals "/path/fundamentals_cache_503_tickers.json" \
    --out "/path/out"
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
@dataclass
class BacktestConfig:
    price_cache_path: str
    fundamentals_cache_path: str
    out_dir: str

    benchmark: str = "QQQ"
    start_date: str = "1999-01-01"
    end_date: str = "2024-12-31"

    factors: Tuple[str, str] = ("Momentum", "QualityValue")
    factor_weights: Tuple[float, float] = (0.5, 0.5)

    mom_lookback_days: int = 252      # 12M
    mom_skip_days: int = 20           # skip last 1M
    min_history_days: int = 252       # at least one year of history

    target_vol: float = 0.10
    vol_window: int = 20
    leverage_cap: float = 3.0
    eps: float = 1e-8

    top_n_per_factor: int = 50        # if <=0, use top 20% by quantile


# =========================
# Helpers
# =========================
def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

def _month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Use "ME" (month end) to avoid deprecation of "M"
    df = pd.DataFrame(index=index, data={"x": 1})
    me = df.resample("ME").last().index
    return pd.DatetimeIndex([d for d in me if d in index])

def _annualize_return(d: pd.Series) -> float:
    return (1 + d.mean()) ** 252 - 1

def _ann_vol(d: pd.Series) -> float:
    return d.std(ddof=0) * math.sqrt(252)

def _sharpe(d: pd.Series, rf: float = 0.0) -> float:
    s = d.std(ddof=0)
    if s < 1e-12:
        return 0.0
    excess = d - rf / 252.0
    return (excess.mean() / (excess.std(ddof=0) + 1e-12)) * math.sqrt(252)

def _mdd(eq: pd.Series) -> float:
    return (eq / eq.cummax() - 1.0).min()

def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _guess_existing_file(preferred: str, patterns: List[str], roots: List[Path]) -> str:
    """
    If preferred path does not exist, search roots with glob patterns and
    choose the largest file (likely the full cache).
    """
    if preferred:
        p = Path(preferred)
        if p.exists():
            return str(p)

    cands = []
    for root in roots:
        for pat in patterns:
            cands.extend(root.glob(pat))
    cands = [c for c in cands if c.is_file()]

    if cands:
        best = max(cands, key=lambda x: x.stat().st_size)
        print(f"[WARN] Preferred path not found. Using discovered file: {best}")
        return str(best)

    raise FileNotFoundError(
        "Cache file not found. Provide --prices / --fundamentals with valid paths, "
        "or place the expected caches next to this script. "
        f"Patterns tried: {patterns} ; Roots: {[str(r) for r in roots]}"
    )


# =========================
# Data Loading
# =========================
def load_prices(price_csv: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_csv(price_csv)
    date_cols = [c for c in df.columns if c.lower() in ("date", "timestamp")]
    if not date_cols:
        date_cols = [df.columns[0]]
    df.rename(columns={date_cols[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    df = df.dropna(axis=1, how="all")
    return df.sort_index()

def load_fundamentals(fund_json: str) -> Dict[str, dict]:
    with open(fund_json, "r") as f:
        data = json.load(f)
    return {k: (v if isinstance(v, dict) else {}) for k, v in data.items()}


# =========================
# Factors
# =========================
def momentum_top(prices: pd.DataFrame, universe: List[str], date: pd.Timestamp,
                 lookback: int, skip: int, top_n: int) -> List[str]:
    idx = prices.index.get_loc(date)
    start_idx = idx - lookback - skip
    end_idx = idx - skip
    if start_idx < 0 or end_idx <= start_idx:
        return []
    window = prices.iloc[start_idx:end_idx + 1]
    if len(window) < 2:
        return []
    perf = (window.iloc[-1] / window.iloc[0] - 1.0)
    perf = perf.dropna()
    perf = perf[perf.index.isin(universe)]
    if perf.empty:
        return []
    if top_n and top_n > 0:
        return perf.sort_values(ascending=False).head(top_n).index.tolist()
    ranks = perf.rank(method="first") / len(perf)
    qcut = pd.qcut(ranks, 5, labels=False, duplicates="drop")
    return perf[qcut == 4].index.tolist()

def qvalue_top(fund_map: Dict[str, dict], universe: List[str], top_n: int) -> List[str]:
    rows = []
    for t in universe:
        f = fund_map.get(t, {})
        if not isinstance(f, dict):
            continue
        rows.append({
            "ticker": t,
            "roe": f.get("returnOnEquity"),
            "gross": f.get("grossMargins"),
            "pm": f.get("profitMargins"),
            "pb": f.get("priceToBook"),
            "ps": f.get("priceToSalesTrailing12Months") or f.get("priceToSales"),
            "evebitda": f.get("enterpriseToEbitda") or f.get("enterpriseToEbitdaratio"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    df["roe"] = _coerce_num(df["roe"])
    df["gross"] = _coerce_num(df["gross"])
    df["pm"] = _coerce_num(df["pm"])
    df["pb"] = _coerce_num(df["pb"])
    df["ps"] = _coerce_num(df["ps"])
    df["evebitda"] = _coerce_num(df["evebitda"])
    for col in ["pb", "ps", "evebitda"]:
        df.loc[(df[col] <= 0) | (df[col].isna()), col] = np.nan
    df = df.set_index("ticker")
    q = _zscore(df["roe"]).fillna(0) + _zscore(df["gross"]).fillna(0) + _zscore(df["pm"]).fillna(0)
    v = -_zscore(np.log(df["pb"])).fillna(0) + -_zscore(np.log(df["ps"])).fillna(0) + -_zscore(np.log(df["evebitda"])).fillna(0)
    score = (q + v).replace([np.inf, -np.inf], np.nan)
    score = score.fillna(score.median())
    if top_n and top_n > 0:
        return score.sort_values(ascending=False).head(top_n).index.tolist()
    ranks = score.rank(method="first") / len(score)
    qcut = pd.qcut(ranks, 5, labels=False, duplicates="drop")
    return score[qcut == 4].index.tolist()


# =========================
# Backtest
# =========================
def backtest(cfg: BacktestConfig):
    prices = load_prices(cfg.price_cache_path, cfg.start_date, cfg.end_date)
    print(f"[INFO] Loaded prices shape: {prices.shape} ({prices.index.min().date()} ~ {prices.index.max().date()})")
    if cfg.benchmark not in prices.columns:
        raise ValueError(f"Benchmark {cfg.benchmark} is not in price columns. Please ensure CSV includes it.")

    fundamentals = load_fundamentals(cfg.fundamentals_cache_path)
    print(f"[INFO] Loaded fundamentals for {len(fundamentals)} tickers")

    universe = sorted(set(prices.columns) & set(fundamentals.keys()))
    if cfg.benchmark in universe:
        universe.remove(cfg.benchmark)
    universe = [t for t in universe if prices[t].dropna().shape[0] >= cfg.min_history_days]
    print(f"[INFO] Universe size after filters: {len(universe)}")

    me = _month_ends(prices.index)
    me = me[(me >= pd.to_datetime(cfg.start_date)) & (me <= pd.to_datetime(cfg.end_date))]
    print(f"[INFO] Rebalance dates: {len(me)} months")

    target_portfolios = pd.DataFrame(index=me, columns=list(cfg.factors), dtype=object)

    for d in me:
        try:
            picks = momentum_top(prices, universe, d, cfg.mom_lookback_days, cfg.mom_skip_days, cfg.top_n_per_factor)
        except Exception:
            picks = []
        target_portfolios.loc[d, "Momentum"] = picks

    qv_picks = qvalue_top(fundamentals, universe, cfg.top_n_per_factor)
    for d in me:
        target_portfolios.loc[d, "QualityValue"] = list(qv_picks)

    # Daily weights from last rebalance; equal-weight inside each factor; combine and renormalize
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i, d in enumerate(me):
        days = prices.index[(prices.index >= d)] if i == len(me)-1 else prices.index[(prices.index >= d) & (prices.index < me[i+1])]
        row = target_portfolios.loc[d]
        mom_list = row.get("Momentum", []) or []
        qv_list = row.get("QualityValue", []) or []

        w = {}
        if mom_list:
            wm = cfg.factor_weights[0] / len(mom_list)
            for t in mom_list:
                if t in prices.columns:
                    w[t] = w.get(t, 0.0) + wm
        if qv_list:
            wq = cfg.factor_weights[1] / len(qv_list)
            for t in qv_list:
                if t in prices.columns:
                    w[t] = w.get(t, 0.0) + wq

        if not w:
            continue
        s = sum(w.values())
        if s > 0:
            for k in list(w.keys()):
                w[k] = w[k] / s
        for day in days:
            weights.loc[day, list(w.keys())] = list(w.values())

    rets = prices.pct_change().fillna(0.0)
    strat_ret_raw = (weights.shift(1) * rets).sum(axis=1)

    roll_vol = strat_ret_raw.rolling(cfg.vol_window).std(ddof=0) * math.sqrt(252)
    scaler = (cfg.target_vol / (roll_vol + cfg.eps)).clip(upper=cfg.leverage_cap).shift(1).fillna(0.0)
    strat_ret = strat_ret_raw * scaler

    bench_ret = rets[cfg.benchmark].copy()
    strat_eq = (1 + strat_ret).cumprod()
    bench_eq = (1 + bench_ret).cumprod()

    def _stats(series: pd.Series):
        eq = (1 + series).cumprod()
        return {
            "Total Return": f"{(eq.iloc[-1]-1)*100:,.2f}%",
            "Annualized Return": f"{_annualize_return(series)*100:,.2f}%",
            "Annualized Volatility": f"{_ann_vol(series)*100:,.2f}%",
            "Sharpe Ratio": f"{_sharpe(series):.2f}",
            "Max Drawdown": f"{_mdd(eq)*100:,.2f}%",
        }

    s_stat = _stats(strat_ret)
    b_stat = _stats(bench_ret)
    perf = pd.DataFrame([s_stat, b_stat], index=["Multi-Factor Strategy", f"Benchmark ({cfg.benchmark})"])

    print("\n--- Final Backtest Results ---")
    print(perf)

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "strategy_daily_returns.csv").write_text(strat_ret.to_csv(header=["ret"]))
    (out_dir / "benchmark_daily_returns.csv").write_text(bench_ret.to_csv(header=["ret"]))

    plt.figure(figsize=(12, 6))
    plt.plot(strat_eq, label="Multi-Factor Strategy (Momentum + Q-Value)")
    plt.plot(bench_eq, label=f"Benchmark (Buy and Hold {cfg.benchmark})")
    plt.yscale("log"); plt.xlabel("Date"); plt.ylabel("Equity Curve (Log Scale)")
    plt.title("Final Portfolio: Volatility-Targeted Multi-Factor (Momentum + Q-Value) Strategy")
    plt.legend(title="Strategy")
    fig_path = out_dir / "final_performance.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"[INFO] Performance chart saved to: {fig_path}")

    return {"perf_table": perf, "chart_path": str(fig_path)}


# =========================
# CLI (with defaults and auto-discovery)
# =========================
def parse_args_with_defaults() -> BacktestConfig:
    base = Path(__file__).resolve().parent
    cwd = Path.cwd()
    roots = [base, cwd, base.parent, base / "data", cwd / "data"]

    default_prices = base / "price_cache_1995-01-01_to_2024-12-31_504_tickers.csv"
    default_funds = base / "fundamentals_cache_503_tickers.json"
    default_out = base / "out"

    p = argparse.ArgumentParser(
        description="Offline Multifactor Backtest (Momentum + Q-Value)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--prices", default=str(default_prices), help="Price CSV cache path (auto-discovery if not found)")
    p.add_argument("--fundamentals", default=str(default_funds), help="Fundamentals JSON cache path (auto-discovery if not found)")
    p.add_argument("--out", default=str(default_out), help="Output directory")
    p.add_argument("--start", default="1999-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    p.add_argument("--target-vol", type=float, default=0.10, help="Target annual volatility")
    p.add_argument("--vol-window", type=int, default=20, help="Rolling window (trading days)")
    p.add_argument("--leverage-cap", type=float, default=3.0, help="Leverage cap")
    p.add_argument("--top-n", type=int, default=50, help="Top N per factor (<=0 uses top 20% quantile)")
    args = p.parse_args()

    price_path = _guess_existing_file(
        args.prices,
        patterns=[
            "price_cache*1995*504*tickers*.csv",
            "price_cache*1995*503*tickers*.csv",
            "price_cache*1995*507*tickers*.csv",
            "*price*cache*.csv",
        ],
        roots=roots,
    )
    fund_path = _guess_existing_file(
        args.fundamentals,
        patterns=[
            "fundamentals_cache*503*tickers*.json",
            "fundamentals_cache*.json",
            "*fundamentals*.json",
        ],
        roots=roots,
    )

    return BacktestConfig(
        price_cache_path=price_path,
        fundamentals_cache_path=fund_path,
        out_dir=args.out,
        start_date=args.start,
        end_date=args.end,
        target_vol=args.target_vol,
        vol_window=args.vol_window,
        leverage_cap=args.leverage_cap,
        top_n_per_factor=args.top_n,
    )


if __name__ == "__main__":
    cfg = parse_args_with_defaults()
    backtest(cfg)
