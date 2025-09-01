# -*- coding: utf-8 -*-
"""
Professional Multi-Factor Portfolio Backtesting Engine (with Auto-Pipeline)
Factors: Momentum (12M-1M) + QualityValue (Multi-metric Quality + Value)
Core Features: Monthly Rebalancing, Volatility Targeting, Professional Performance Reporting.
"""

# ==============================================================================
# Import Necessary Libraries
# ==============================================================================
import argparse
import json
import math
from dataclasses import dataclass, field # *** 核心修正：匯入 field ***
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# Configuration Hub - Our "Design Blueprint"
# ==============================================================================
@dataclass
class BacktestConfig:
    # --- Basic Settings ---
    start_date: str = "1999-01-01"
    end_date: str = "2024-12-31"
    benchmark: str = "QQQ"
    
    # --- Data Path Settings ---
    price_cache_path: str = "price_cache.csv"
    fundamentals_cache_path: str = "fundamentals_cache.json"
    output_dir: str = "output"

    # --- Portfolio Construction Settings ---
    factors: Tuple[str, ...] = ("Momentum", "QualityValue")
    top_n_per_factor: int = 50 

    # --- Risk Management Settings ---
    target_volatility: float = 0.10
    volatility_window: int = 20
    max_leverage: float = 3.0

    # --- Factor Definitions ---
    momentum_lookback_days: int = 252
    momentum_skip_days: int = 20
    min_history_days: int = 252
    
    # This field will be populated by the program, so we use field()
    sp500_tickers: List[str] = field(default_factory=list)

# ==============================================================================
# Data Pipeline Lego Block
# ==============================================================================
def get_sp500_tickers() -> List[str]:
    """Fetches the list of S&P 500 constituents from Wikipedia."""
    print("Fetching S&P 500 tickers from Wikipedia...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)
        tickers = [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
        print(f"Successfully fetched {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        print(f"Failed to fetch tickers: {e}. Using a small backup list.")
        return ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']

def load_data(cfg: BacktestConfig) -> Tuple[pd.DataFrame, Dict]:
    """Intelligent data loader."""
    # --- Price Data ---
    sp500 = get_sp500_tickers()
    cfg.sp500_tickers = sp500
    all_tickers = sorted(list(set(sp500 + [cfg.benchmark])))
    
    price_cache_path = Path(cfg.price_cache_path)
    prices = None
    if price_cache_path.exists():
        print(f"Loading prices from cache: {price_cache_path}...")
        prices = pd.read_csv(price_cache_path, index_col=0, parse_dates=True)
        missing = [t for t in all_tickers if t not in prices.columns]
        if not missing:
            print("Price data loaded successfully.")
        else:
            print(f"Cache is incomplete. Forcing fresh download...")
            prices = None
    
    if prices is None:
        print(f"Downloading prices for {len(all_tickers)} tickers...")
        prices = yf.download(all_tickers, start=cfg.start_date, end=cfg.end_date)['Close'].dropna(axis=1, how='all')
        print(f"Saving price data to cache: {price_cache_path}...")
        price_cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(price_cache_path)

    # --- Fundamentals Data ---
    fundamentals_cache_path = Path(cfg.fundamentals_cache_path)
    fundamentals = {}
    if fundamentals_cache_path.exists():
        print(f"Loading fundamentals from cache: {fundamentals_cache_path}...")
        with open(fundamentals_cache_path, 'r') as f:
            fundamentals = json.load(f)
    else:
        print(f"Fundamentals cache not found. This will take a long time (15-30 minutes)...")
        for i, ticker_str in enumerate(sp500):
            print(f"Fetching fundamentals for {ticker_str} ({i+1}/{len(sp500)})...")
            try:
                ticker_obj = yf.Ticker(ticker_str)
                info = ticker_obj.info
                fundamentals[ticker_str] = {
                    'priceToBook': info.get('priceToBook'),
                    'returnOnEquity': info.get('returnOnEquity'),
                    'grossMargins': info.get('grossMargins'),
                    'profitMargins': info.get('profitMargins'),
                    'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months'),
                    'enterpriseToEbitda': info.get('enterpriseToEbitda')
                }
            except Exception: continue
        print(f"Saving fundamentals data to cache: {fundamentals_cache_path}...")
        fundamentals_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fundamentals_cache_path, 'w') as f:
            json.dump(fundamentals, f)
            
    return prices, fundamentals

# ==============================================================================
# Factor Calculation Lego Blocks
# ==============================================================================
def calculate_momentum_portfolio(prices: pd.DataFrame, universe: List[str], date: pd.Timestamp,
                                 lookback: int, skip: int, top_n: int) -> List[str]:
    """Calculates the momentum portfolio for a given date."""
    # *** 核心修正：使用 get_indexer 搭配 method='pad' 來取代過時的 get_loc ***
    try:
        # 找到 date 當天或之前最近的一個交易日的索引位置
        idx = prices.index.get_indexer([date], method='pad')[0]
    except KeyError:
        return []

    start_idx = idx - lookback - skip
    end_idx = idx - skip
    if start_idx < 0 or end_idx <= start_idx: return []
    
    window = prices.iloc[start_idx:end_idx + 1]
    if len(window) < 2: return []
    
    perf = (window.iloc[-1] / window.iloc[0] - 1.0).dropna()
    perf = perf[perf.index.isin(universe)]
    if perf.empty: return []
    
    if top_n > 0:
        return perf.nlargest(top_n).index.tolist()
    else:
        ranks = perf.rank(pct=True)
        return perf[ranks > 0.8].index.tolist()

def calculate_qval_portfolio(fundamentals: Dict, universe: List[str], top_n: int) -> List[str]:
    """Calculates the Quality+Value portfolio (static calculation)."""
    metrics = []
    for t in universe:
        f = fundamentals.get(t, {})
        if isinstance(f, dict):
            metrics.append({
                'ticker': t,
                'roe': f.get('returnOnEquity'),
                'gross': f.get('grossMargins'),
                'pm': f.get('profitMargins'),
                'pb': f.get('priceToBook'),
                'ps': f.get('priceToSalesTrailing12Months'),
                'evebitda': f.get('enterpriseToEbitda')
            })
    
    df = pd.DataFrame(metrics).set_index('ticker').apply(pd.to_numeric, errors='coerce')
    df = df[(df['pb'] > 0) & (df['ps'] > 0) & (df['evebitda'] > 0) & (df['roe'] > 0)]
    if df.empty: return []

    q_score = df[['roe', 'gross', 'pm']].rank(pct=True).mean(axis=1)
    v_score = df[['pb', 'ps', 'evebitda']].rank(pct=True, ascending=False).mean(axis=1)
    score = (q_score + v_score).dropna()

    if top_n > 0:
        return score.nlargest(top_n).index.tolist()
    else:
        ranks = score.rank(pct=True)
        return score[ranks > 0.8].index.tolist()

# ==============================================================================
# Performance Reporting Lego Block
# ==============================================================================
def analyze_performance(strategy_returns: pd.Series, benchmark_returns: pd.Series, cfg: BacktestConfig):
    """Calculates and displays a full performance report and chart."""
    def _stats(series: pd.Series):
        equity = (1 + series).cumprod()
        ann_ret = (1 + series.mean()) ** 252 - 1
        ann_vol = series.std() * np.sqrt(252)
        return {
            "Total Return": equity.iloc[-1] - 1,
            "Annualized Return": ann_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": ann_ret / ann_vol if ann_vol > 0 else 0,
            "Max Drawdown": (equity / equity.cummax() - 1.0).min(),
        }

    s_stat = _stats(strategy_returns)
    b_stat = _stats(benchmark_returns)
    perf = pd.DataFrame([s_stat, b_stat], index=["Multi-Factor Strategy", f"Benchmark ({cfg.benchmark})"])

    print("\n--- Final Backtest Results ---")
    print(perf.to_string(formatters={
        'Total Return': '{:,.2%}'.format, 'Annualized Return': '{:,.2%}'.format,
        'Annualized Volatility': '{:,.2%}'.format, 'Sharpe Ratio': '{:,.2f}'.format,
        'Max Drawdown': '{:,.2%}'.format
    }))

    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    strategy_returns.to_csv(out_dir / "strategy_daily_returns.csv", header=["ret"])
    benchmark_returns.to_csv(out_dir / "benchmark_daily_returns.csv", header=["ret"])

    print("\nGenerating performance chart...")
    plt.figure(figsize=(16, 9))
    plt.plot((1 + strategy_returns).cumprod(), label="Multi-Factor Strategy (Momentum + Q-Value)")
    plt.plot((1 + benchmark_returns).cumprod(), label=f"Benchmark (Buy and Hold {cfg.benchmark})")
    plt.yscale("log"); plt.xlabel("Date"); plt.ylabel("Equity Curve (Log Scale)")
    plt.title("Final Portfolio: Volatility-Targeted Multi-Factor Strategy")
    plt.legend(title="Strategy"); plt.grid(True, which="both", ls="--")
    fig_path = out_dir / "final_performance.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"[INFO] Performance chart saved to: {fig_path}")
    plt.show()

# ==============================================================================
# Backtesting Engine Lego Block
# ==============================================================================
def run_backtest(cfg: BacktestConfig):
    """The main backtest process that assembles all the lego blocks."""
    prices, fundamentals = load_data(cfg)
    
    universe = [t for t in cfg.sp500_tickers if t in prices.columns and prices[t].dropna().shape[0] >= cfg.min_history_days]
    print(f"[INFO] Final universe size after filters: {len(universe)}")

    rebalance_dates = prices.resample('M').last().index
    rebalance_dates = rebalance_dates[(rebalance_dates >= pd.to_datetime(cfg.start_date)) & (rebalance_dates <= pd.to_datetime(cfg.end_date))]
    
    print("Calculating monthly target portfolios...")
    target_portfolios = pd.DataFrame(index=rebalance_dates, columns=list(cfg.factors), dtype=object)
    
    for d in rebalance_dates:
        target_portfolios.loc[d, "Momentum"] = calculate_momentum_portfolio(
            prices, universe, d, cfg.momentum_lookback_days, cfg.momentum_skip_days, cfg.top_n_per_factor
        )
    qval_picks = calculate_qval_portfolio(fundamentals, universe, cfg.top_n_per_factor)
    for d in rebalance_dates:
        target_portfolios.loc[d, "QualityValue"] = list(qval_picks)

    print("Constructing daily multi-factor positions...")
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i, d in enumerate(rebalance_dates):
        start_day = d
        end_day = prices.index[-1] if i == len(rebalance_dates)-1 else rebalance_dates[i+1]
        days = prices.index[(prices.index >= start_day) & (prices.index < end_day)]
        
        row = target_portfolios.loc[d]
        mom_list = row.get("Momentum", []) or []
        qv_list = row.get("QualityValue", []) or []
        
        combined_portfolio = set(mom_list) | set(qv_list)
        valid_portfolio = [t for t in combined_portfolio if t in prices.columns]

        if not valid_portfolio: continue
        
        weight = 1.0 / len(valid_portfolio)
        for day in days:
            weights.loc[day, valid_portfolio] = weight

    print("Applying volatility targeting and calculating performance...")
    daily_returns = prices.pct_change()
    strat_ret_raw = (weights.shift(1) * daily_returns).sum(axis=1).fillna(0.0)
    
    roll_vol = strat_ret_raw.rolling(cfg.volatility_window).std() * np.sqrt(252)
    scaler = (cfg.target_volatility / (roll_vol + 1e-8)).clip(upper=cfg.max_leverage).shift(1).fillna(1.0)
    strat_ret = strat_ret_raw * scaler

    benchmark_ret = daily_returns[cfg.benchmark].fillna(0.0)
    
    analyze_performance(strat_ret, benchmark_ret, cfg)

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    # We create a default config object and pass it to the backtest engine.
    # This separates our research (the config) from the execution (the engine).
    default_cfg = BacktestConfig(
        price_cache_path="price_cache_1995-01-01_to_2024-12-31_504_tickers.csv",
        fundamentals_cache_path="fundamentals_cache_503_tickers.json"
    )
    
    run_backtest(default_cfg)
