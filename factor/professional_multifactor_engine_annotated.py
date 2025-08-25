# 解釋：原作者註解（保留）
# -*- coding: utf-8 -*-
# 解釋：多行字串（通常是檔頭/函式說明）
"""
# 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
Professional Multi-Factor Portfolio Backtesting Engine (with Auto-Pipeline)
# 解釋：動量相關邏輯（挑最近一段時間表現較強的股票）
Factors: Momentum (12M-1M) + QualityValue (Multi-metric Quality + Value)
# 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
Core Features: Monthly Rebalancing, Volatility Targeting, Professional Performance Reporting.
# 解釋：多行字串（通常是檔頭/函式說明）
"""

# 解釋：原作者註解（保留）
# ==============================================================================
# 解釋：原作者註解（保留）
# Import Necessary Libraries
# 解釋：原作者註解（保留）
# ==============================================================================
# 解釋：匯入外部模組/函式，供後續使用
import argparse
# 解釋：匯入外部模組/函式，供後續使用
import json
# 解釋：匯入外部模組/函式，供後續使用
import math
# 解釋：匯入外部模組/函式，供後續使用
from dataclasses import dataclass
# 解釋：匯入外部模組/函式，供後續使用
from pathlib import Path
# 解釋：匯入外部模組/函式，供後續使用
from typing import Dict, List, Tuple

# 解釋：匯入外部模組/函式，供後續使用
import numpy as np
# 解釋：匯入外部模組/函式，供後續使用
import pandas as pd
# 解釋：匯入外部模組/函式，供後續使用
import matplotlib.pyplot as plt
# 解釋：匯入外部模組/函式，供後續使用
import yfinance as yf
# 解釋：匯入外部模組/函式，供後續使用
import warnings
# 解釋：匯入外部模組/函式，供後續使用
import os

# 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
warnings.simplefilter(action='ignore', category=FutureWarning)

# 解釋：原作者註解（保留）
# ==============================================================================
# 解釋：原作者註解（保留）
# Configuration Hub
# 解釋：原作者註解（保留）
# ==============================================================================
# 解釋：將下方 class 標記為資料類（自動生成 __init__ 等）
@dataclass
# 解釋：定義一個類別（結構/引擎/設定容器）
class BacktestConfig:
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    start_date: str = "1999-01-01"
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    end_date: str = "2024-12-31"
    # 解釋：設定基準（例如 QQQ）用來比較績效
    benchmark: str = "QQQ"
    
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    price_cache_path: str = "price_cache.csv"
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    fundamentals_cache_path: str = "fundamentals_cache.json"
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    output_dir: str = "output"

    # 解釋：動量相關邏輯（挑最近一段時間表現較強的股票）
    factors: Tuple[str, ...] = ("Momentum", "QualityValue")
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    top_n_per_factor: int = 50

    # 解釋：設定目標年化波動，用來調整整體倉位大小
    target_volatility: float = 0.10
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    volatility_window: int = 20
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    max_leverage: float = 3.0

    # 解釋：動量相關邏輯（挑最近一段時間表現較強的股票）
    momentum_lookback_days: int = 252
    # 解釋：動量相關邏輯（挑最近一段時間表現較強的股票）
    momentum_skip_days: int = 20
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    min_history_days: int = 252
    
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    sp500_tickers: List[str] = field(default_factory=list)

# 解釋：原作者註解（保留）
# ==============================================================================
# 解釋：原作者註解（保留）
# Data Pipeline
# 解釋：原作者註解（保留）
# ==============================================================================
# 解釋：定義函式 get_sp500_tickers（封裝一段可重用的邏輯）
def get_sp500_tickers() -> List[str]:
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    print("Fetching S&P 500 tickers from Wikipedia...")
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    try:
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        table = pd.read_html(url)
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        tickers = [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        print(f"Successfully fetched {len(tickers)} tickers.")
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        return tickers
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    except Exception as e:
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        print(f"Failed to fetch tickers: {e}. Using a small backup list.")
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        return ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']

# 解釋：定義函式 load_data（封裝一段可重用的邏輯）
def load_data(cfg: BacktestConfig) -> Tuple[pd.DataFrame, Dict]:
    # 解釋：多行字串（通常是檔頭/函式說明）
    """
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    Intelligent data loader:
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    1. Automatically checks if cache exists.
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    2. If cache is incomplete or missing, it downloads automatically.
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    3. Returns prices and fundamentals data.
    # 解釋：多行字串（通常是檔頭/函式說明）
    """
    # 解釋：原作者註解（保留）
    # --- Price Data ---
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    sp500 = get_sp500_tickers()
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    cfg.sp500_tickers = sp500
    # 解釋：設定基準（例如 QQQ）用來比較績效
    all_tickers = sorted(list(set(sp500 + [cfg.benchmark])))
    
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    price_cache_path = Path(cfg.price_cache_path)
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    prices = None
    # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
    if price_cache_path.exists():
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        print(f"Loading prices from cache: {price_cache_path}...")
        # 解釋：讀取 CSV 價格/資料到 DataFrame
        prices = pd.read_csv(price_cache_path, index_col=0, parse_dates=True)
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        missing = [t for t in all_tickers if t not in prices.columns]
        # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
        if not missing:
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            print("Price data loaded successfully.")
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        else:
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            print(f"Cache is incomplete. Forcing fresh download...")
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            prices = None
    
    # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
    if prices is None:
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        print(f"Downloading prices for {len(all_tickers)} tickers...")
        # 解釋：丟棄缺失值（NaN），避免影響計算
        prices = yf.download(all_tickers, start=cfg.start_date, end=cfg.end_date)['Close'].dropna(axis=1, how='all')
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        print(f"Saving price data to cache: {price_cache_path}...")
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        price_cache_path.parent.mkdir(parents=True, exist_ok=True)
        # 解釋：把結果輸出成 CSV 檔以便檢視/留檔
        prices.to_csv(price_cache_path)

    # 解釋：原作者註解（保留）
    # --- Fundamentals Data ---
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    fundamentals_cache_path = Path(cfg.fundamentals_cache_path)
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    fundamentals = {}
    # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
    if fundamentals_cache_path.exists():
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        print(f"Loading fundamentals from cache: {fundamentals_cache_path}...")
        # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
        with open(fundamentals_cache_path, 'r') as f:
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            fundamentals = json.load(f)
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    else:
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        print(f"Fundamentals cache not found. This will take a long time (15-30 minutes)...")
        # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
        for i, ticker_str in enumerate(sp500):
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            print(f"Fetching fundamentals for {ticker_str} ({i+1}/{len(sp500)})...")
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            try:
                # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                ticker_obj = yf.Ticker(ticker_str)
                # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                info = ticker_obj.info
                # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                fundamentals[ticker_str] = {
                    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                    'priceToBook': info.get('priceToBook'),
                    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                    'returnOnEquity': info.get('returnOnEquity'),
                    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                    'grossMargins': info.get('grossMargins'),
                    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                    'profitMargins': info.get('profitMargins'),
                    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                    'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months'),
                    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                    'enterpriseToEbitda': info.get('enterpriseToEbitda')
                # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                }
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            except Exception: continue
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        print(f"Saving fundamentals data to cache: {fundamentals_cache_path}...")
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        fundamentals_cache_path.parent.mkdir(parents=True, exist_ok=True)
        # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
        with open(fundamentals_cache_path, 'w') as f:
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            json.dump(fundamentals, f)
            
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    return prices, fundamentals

# 解釋：原作者註解（保留）
# ==============================================================================
# 解釋：原作者註解（保留）
# Factor Calculation
# 解釋：原作者註解（保留）
# ==============================================================================
# 解釋：定義函式 calculate_momentum_portfolio（封裝一段可重用的邏輯）
def calculate_momentum_portfolio(prices: pd.DataFrame, universe: List[str], date: pd.Timestamp,
                                 # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                                 lookback: int, skip: int, top_n: int) -> List[str]:
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    idx = prices.index.get_loc(date)
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    start_idx = idx - lookback - skip
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    end_idx = idx - skip
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    if start_idx < 0 or end_idx <= start_idx: return []
    
    # 解釋：以整數位置選取資料
    window = prices.iloc[start_idx:end_idx + 1]
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    if len(window) < 2: return []
    
    # 解釋：以整數位置選取資料
    perf = (window.iloc[-1] / window.iloc[0] - 1.0).dropna()
    # 解釋：定義可投資股票池（過濾出符合條件的代碼）
    perf = perf[perf.index.isin(universe)]
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    if perf.empty: return []
    
    # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
    if top_n > 0:
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        return perf.nlargest(top_n).index.tolist()
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    else:
        # 解釋：對數值做排序名次（配合分位篩選）
        ranks = perf.rank(pct=True)
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        return perf[ranks > 0.8].index.tolist()

# 解釋：定義函式 calculate_qval_portfolio（封裝一段可重用的邏輯）
def calculate_qval_portfolio(fundamentals: Dict, universe: List[str], top_n: int) -> List[str]:
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    metrics = []
    # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
    for t in universe:
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        f = fundamentals.get(t, {})
        # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
        if isinstance(f, dict):
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            metrics.append({
                # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                'ticker': t,
                # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                'roe': f.get('returnOnEquity'),
                # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                'gross': f.get('grossMargins'),
                # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                'pm': f.get('profitMargins'),
                # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                'pb': f.get('priceToBook'),
                # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                'ps': f.get('priceToSalesTrailing12Months'),
                # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
                'evebitda': f.get('enterpriseToEbitda')
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            })
    
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    df = pd.DataFrame(metrics).set_index('ticker').apply(pd.to_numeric, errors='coerce')
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    df = df[(df['pb'] > 0) & (df['ps'] > 0) & (df['evebitda'] > 0) & (df['roe'] > 0)]
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    if df.empty: return []

    # 解釋：對數值做排序名次（配合分位篩選）
    q_score = df[['roe', 'gross', 'pm']].rank(pct=True).mean(axis=1)
    # 解釋：對數值做排序名次（配合分位篩選）
    v_score = df[['pb', 'ps', 'evebitda']].rank(pct=True, ascending=False).mean(axis=1)
    # 解釋：丟棄缺失值（NaN），避免影響計算
    score = (q_score + v_score).dropna()

    # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
    if top_n > 0:
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        return score.nlargest(top_n).index.tolist()
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    else:
        # 解釋：對數值做排序名次（配合分位篩選）
        ranks = score.rank(pct=True)
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        return score[ranks > 0.8].index.tolist()

# 解釋：原作者註解（保留）
# ==============================================================================
# 解釋：原作者註解（保留）
# Main Backtest Function
# 解釋：原作者註解（保留）
# ==============================================================================
# 解釋：定義函式 run_backtest（封裝一段可重用的邏輯）
def run_backtest(cfg: BacktestConfig):
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    prices, fundamentals = load_data(cfg)
    
    # 解釋：丟棄缺失值（NaN），避免影響計算
    universe = [t for t in cfg.sp500_tickers if t in prices.columns and prices[t].dropna().shape[0] >= cfg.min_history_days]
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    print(f"[INFO] Final universe size after filters: {len(universe)}")

    # 解釋：重取樣（例如聚合到每月月末）
    rebalance_dates = prices.resample('M').last().index
    # 解釋：把字串日期轉成 pandas 的日期型別
    rebalance_dates = rebalance_dates[(rebalance_dates >= pd.to_datetime(cfg.start_date)) & (rebalance_dates <= pd.to_datetime(cfg.end_date))]
    
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    print("Calculating monthly target portfolios...")
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    target_portfolios = pd.DataFrame(index=rebalance_dates, columns=list(cfg.factors), dtype=object)
    
    # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
    for d in rebalance_dates:
        # 解釋：以標籤（日期/欄位名）選取資料
        target_portfolios.loc[d, "Momentum"] = calculate_momentum_portfolio(
            # 解釋：動量相關邏輯（挑最近一段時間表現較強的股票）
            prices, universe, d, cfg.momentum_lookback_days, cfg.momentum_skip_days, cfg.top_n_per_factor
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        )
    # 解釋：定義可投資股票池（過濾出符合條件的代碼）
    qval_picks = calculate_qval_portfolio(fundamentals, universe, cfg.top_n_per_factor)
    # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
    for d in rebalance_dates:
        # 解釋：以標籤（日期/欄位名）選取資料
        target_portfolios.loc[d, "QualityValue"] = list(qval_picks)

    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    print("Constructing daily multi-factor positions...")
    # 解釋：初始化持倉權重表（每日每檔的配置）
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
    for i, d in enumerate(rebalance_dates):
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        start_day = d
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        end_day = prices.index[-1] if i == len(rebalance_dates)-1 else rebalance_dates[i+1]
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        days = prices.index[(prices.index >= start_day) & (prices.index < end_day)]
        
        # 解釋：以標籤（日期/欄位名）選取資料
        row = target_portfolios.loc[d]
        # 解釋：動量相關邏輯（挑最近一段時間表現較強的股票）
        mom_list = row.get("Momentum", []) or []
        # 解釋：品質/價值因子相關邏輯（需避免未來資訊）
        qv_list = row.get("QualityValue", []) or []
        
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        combined_portfolio = set(mom_list) | set(qv_list)
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        valid_portfolio = [t for t in combined_portfolio if t in prices.columns]

        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        if not valid_portfolio: continue
        
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        weight = 1.0 / len(valid_portfolio)
        # 解釋：流程控制區塊（條件/迴圈/資源管理）開始
        for day in days:
            # 解釋：以標籤（日期/欄位名）選取資料
            weights.loc[day, valid_portfolio] = weight

    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    print("Applying volatility targeting and calculating performance...")
    # 解釋：把價格序列轉成報酬率（今日/昨日 - 1）
    daily_returns = prices.pct_change()
    # 解釋：以固定值或方法填補缺失值
    strat_ret_raw = (weights.shift(1) * daily_returns).sum(axis=1).fillna(0.0)
    
    # 解釋：以滾動窗口計算標準差（常用來估波動）
    roll_vol = strat_ret_raw.rolling(cfg.volatility_window).std() * np.sqrt(252)
    # 解釋：以固定值或方法填補缺失值
    scaler = (cfg.target_volatility / (roll_vol + 1e-8)).clip(upper=cfg.max_leverage).shift(1).fillna(1.0)
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    strat_ret = strat_ret_raw * scaler

    # 解釋：以固定值或方法填補缺失值
    benchmark_ret = daily_returns[cfg.benchmark].fillna(0.0)
    
    # 解釋：原作者註解（保留）
    # --- Performance Analysis ---
    # 解釋：定義函式 _stats（封裝一段可重用的邏輯）
    def _stats(series: pd.Series):
        # 解釋：累乘（用 1+日報酬 的累積乘積 → 權益曲線）
        equity = (1 + series).cumprod()
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        ann_ret = (1 + series.mean()) ** 252 - 1
        # 解釋：把日波動換算成年化（交易日約 252 天）
        ann_vol = series.std() * np.sqrt(252)
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        return {
            # 解釋：以整數位置選取資料
            "Total Return": equity.iloc[-1] - 1,
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            "Annualized Return": ann_ret,
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            "Annualized Volatility": ann_vol,
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            "Sharpe Ratio": ann_ret / ann_vol if ann_vol > 0 else 0,
            # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
            "Max Drawdown": (equity / equity.cummax() - 1.0).min(),
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        }

    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    s_stat = _stats(strat_ret)
    # 解釋：設定基準（例如 QQQ）用來比較績效
    b_stat = _stats(benchmark_ret)
    # 解釋：設定基準（例如 QQQ）用來比較績效
    perf = pd.DataFrame([s_stat, b_stat], index=["Multi-Factor Strategy", f"Benchmark ({cfg.benchmark})"])

    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    print("\n--- Final Backtest Results ---")
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    print(perf.to_string(formatters={
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        'Total Return': '{:,.2%}'.format, 'Annualized Return': '{:,.2%}'.format,
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        'Annualized Volatility': '{:,.2%}'.format, 'Sharpe Ratio': '{:,.2f}'.format,
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        'Max Drawdown': '{:,.2%}'.format
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    }))

    # 解釋：原作者註解（保留）
    # --- Output & Plotting ---
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # 解釋：把結果輸出成 CSV 檔以便檢視/留檔
    strat_ret.to_csv(out_dir / "strategy_daily_returns.csv", header=["ret"])
    # 解釋：把結果輸出成 CSV 檔以便檢視/留檔
    benchmark_ret.to_csv(out_dir / "benchmark_daily_returns.csv", header=["ret"])

    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    print("\nGenerating performance chart...")
    # 解釋：繪圖：畫權益曲線或其他圖表
    plt.figure(figsize=(16, 9))
    # 解釋：累乘（用 1+日報酬 的累積乘積 → 權益曲線）
    plt.plot((1 + strat_ret).cumprod(), label="Multi-Factor Strategy (Momentum + Q-Value)")
    # 解釋：累乘（用 1+日報酬 的累積乘積 → 權益曲線）
    plt.plot((1 + benchmark_ret).cumprod(), label=f"Benchmark (Buy and Hold {cfg.benchmark})")
    # 解釋：繪圖：畫權益曲線或其他圖表
    plt.yscale("log"); plt.xlabel("Date"); plt.ylabel("Equity Curve (Log Scale)")
    # 解釋：繪圖：畫權益曲線或其他圖表
    plt.title("Final Portfolio: Volatility-Targeted Multi-Factor Strategy")
    # 解釋：繪圖：畫權益曲線或其他圖表
    plt.legend(title="Strategy"); plt.grid(True, which="both", ls="--")
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    fig_path = out_dir / "final_performance.png"
    # 解釋：繪圖：畫權益曲線或其他圖表
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    print(f"[INFO] Performance chart saved to: {fig_path}")
    # 解釋：繪圖：畫權益曲線或其他圖表
    plt.show()

# 解釋：原作者註解（保留）
# ==============================================================================
# 解釋：原作者註解（保留）
# Command-Line Interface
# 解釋：原作者註解（保留）
# ==============================================================================
# 解釋：定義函式 parse_args_with_aults（封裝一段可重用的邏輯）
def parse_args_with_defaults() -> BacktestConfig:
    # 解釋：原作者註解（保留）
    # This function allows running the script from the command line with custom paths
    # 解釋：原作者註解（保留）
    # For simplicity in our interactive environment, we will use the default config directly.
    # 解釋：原作者註解（保留）
    # This code is here to show the professional way of making scripts configurable.
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    base = Path(__file__).resolve().parent
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    default_prices = base / "price_cache.csv"
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    default_funds = base / "fundamentals_cache.json"
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    default_out = base / "output"

    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    p = argparse.ArgumentParser(description="Professional Multi-Factor Backtest")
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    p.add_argument("--prices", default=str(default_prices))
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    p.add_argument("--fundamentals", default=str(default_funds))
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    p.add_argument("--out", default=str(default_out))
    # 解釋：原作者註解（保留）
    # Add other arguments if needed...
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    args, _ = p.parse_known_args() # Use parse_known_args to avoid issues in some environments

    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    return BacktestConfig(
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        price_cache_path=args.prices,
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        fundamentals_cache_path=args.fundamentals,
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        output_dir=args.out
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    )

# 解釋：程式入口：直接執行此檔時，會從這裡開始跑
if __name__ == "__main__":
    # 解釋：原作者註解（保留）
    # We directly create the config object instead of parsing args
    # 解釋：原作者註解（保留）
    # This makes it easier to run in environments like this canvas.
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    default_cfg = BacktestConfig(
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        price_cache_path="price_cache_1995-01-01_to_2024-12-31_504_tickers.csv",
        # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
        fundamentals_cache_path="fundamentals_cache_503_tickers.json"
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    )
    # 解釋：此行是邏輯流程的一部分（一般計算/指派/函式呼叫）
    run_backtest(default_cfg)