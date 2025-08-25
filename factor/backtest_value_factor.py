# ==============================================================================
# 匯入必要的函式庫 (Import Necessary Libraries)
# ==============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import json # 用於儲存和讀取我們的基本面數據快取
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# 策略配置中心 (Configuration Hub)
# ==============================================================================
config = {
    "start_date": "1995-01-01",
    "end_date": "2024-12-31",
    "rebalance_freq": "M",
    "data": {
        "price_cache_file": "price_cache_1995_2024_503_tickers.csv",
        "fundamentals_cache_file": "fundamentals_cache.json" # 新增：基本面數據的快取檔案
    },
    "factor": {
        "name": "Value (Price-to-Book)",
        "num_quantiles": 5,
    }
}

# ==============================================================================
# 專業級因子回測引擎 V2 (Professional Factor Backtesting Engine V2)
# ==============================================================================
class FactorBacktester:
    def __init__(self, config):
        self.config = config
        self.sp500_tickers = []
        self.price_data = None
        self.fundamentals_data = {} # 用於儲存公司的基本面數據
        self.factor_returns = None

    def _get_sp500_tickers(self):
        # 從維基百科獲取 S&P 500 成分股列表
        print("Fetching S&P 500 tickers...")
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)
            self.sp500_tickers = [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
            print(f"Successfully fetched {len(self.sp500_tickers)} tickers.")
        except Exception as e:
            print(f"Failed to fetch tickers: {e}. Using a small backup list.")
            self.sp500_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']

    def _get_data(self):
        # --- 價格數據處理 (核心修正) ---
        price_cache = self.config['data']['price_cache_file']
        if os.path.exists(price_cache):
            print(f"Loading prices from cache: {price_cache}...")
            self.price_data = pd.read_csv(price_cache, index_col=0, parse_dates=True)
        else:
            # 如果快取不存在，則自己下載
            print(f"Price cache file not found. Downloading data...")
            self._get_sp500_tickers()
            print(f"Downloading prices for {len(self.sp500_tickers)} tickers...")
            self.price_data = yf.download(self.sp500_tickers, start=self.config['start_date'], end=self.config['end_date'])['Close'].dropna(axis=1, how='all')
            print(f"Saving price data to cache: {price_cache}...")
            self.price_data.to_csv(price_cache)


        # --- 基本面數據處理 ---
        fundamentals_cache = self.config['data']['fundamentals_cache_file']
        if os.path.exists(fundamentals_cache):
            print(f"Loading fundamentals from cache: {fundamentals_cache}...")
            with open(fundamentals_cache, 'r') as f:
                self.fundamentals_data = json.load(f)
        else:
            print("Fundamentals cache not found. Fetching data from yfinance...")
            if not self.sp500_tickers: # 確保在下載基本面數據前有名單
                self._get_sp500_tickers()
            
            for ticker_str in self.sp500_tickers:
                try:
                    print(f"Fetching fundamentals for {ticker_str}...")
                    ticker_obj = yf.Ticker(ticker_str)
                    info = ticker_obj.info
                    if 'priceToBook' in info and info['priceToBook'] is not None:
                        self.fundamentals_data[ticker_str] = {'priceToBook': info['priceToBook']}
                except Exception as e:
                    print(f"Could not fetch fundamentals for {ticker_str}: {e}")
            
            print(f"Saving fundamentals data to cache: {fundamentals_cache}...")
            with open(fundamentals_cache, 'w') as f:
                json.dump(self.fundamentals_data, f)
    
    def run(self):
        # 執行完整的回測流程
        self._get_data()
        
        print("Calculating monthly returns...")
        monthly_returns = self.price_data.resample(self.config['rebalance_freq']).last().pct_change()
        
        portfolio_returns = pd.DataFrame()
        rebalance_dates = monthly_returns.index

        print("Starting monthly rebalancing backtest for Value factor...")
        for date in rebalance_dates:
            # --- a. 獲取當期的因子值 ---
            factor_values = {}
            for ticker, data in self.fundamentals_data.items():
                if 'priceToBook' in data and ticker in self.price_data.columns:
                    factor_values[ticker] = data['priceToBook']
            
            value_factor = pd.Series(factor_values)
            value_factor = value_factor[value_factor > 0].dropna()
            
            if value_factor.empty:
                continue

            # --- b. 排序與分組 ---
            num_quantiles = self.config['factor']['num_quantiles']
            labels = [f'Q{j+1}' for j in range(num_quantiles)]
            try:
                factor_quantiles = pd.qcut(value_factor, num_quantiles, labels=labels, duplicates='drop')
            except ValueError:
                continue
            
            # --- c. 計算下個月各投資組合的報酬 ---
            if date in monthly_returns.index:
                next_month_returns = monthly_returns.loc[date]
                for q in factor_quantiles.unique():
                    stocks_in_quantile = factor_quantiles[factor_quantiles == q].index
                    valid_stocks = stocks_in_quantile.intersection(next_month_returns.dropna().index)
                    portfolio_returns.loc[date, q] = next_month_returns[valid_stocks].mean()

        # --- d. 計算多空組合 ---
        if 'Q1' in portfolio_returns.columns and f'Q{num_quantiles}' in portfolio_returns.columns:
            portfolio_returns['Long-Short'] = portfolio_returns['Q1'] - portfolio_returns[f'Q{num_quantiles}']
        else:
            portfolio_returns['Long-Short'] = 0
        
        self.factor_returns = portfolio_returns.fillna(0)
        print("Backtest complete.")

    def analyze_and_plot(self):
        # 計算並顯示績效
        if self.factor_returns is None:
            print("Please run the backtest first.")
            return

        equity_curves = (1 + self.factor_returns).cumprod()

        def calculate_metrics(returns):
            if returns.abs().sum() == 0: return {"Annualized Return": 0, "Annualized Volatility": 0, "Sharpe Ratio": 0}
            annualized_return = (1 + returns).prod() ** (12 / len(returns)) - 1
            annualized_volatility = returns.std() * np.sqrt(12)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
            return {"Annualized Return": annualized_return, "Annualized Volatility": annualized_volatility, "Sharpe Ratio": sharpe_ratio}

        long_short_metrics = calculate_metrics(self.factor_returns['Long-Short'])

        print("\n--- Value Factor Performance (Long Cheapest / Short Most Expensive) ---")
        for key, value in long_short_metrics.items():
            print(f"{key}: {value:.2%}" if "Return" in key or "Volatility" in key else f"{key}: {value:.2f}")
        print("-" * 50)

        print("Generating performance chart...")
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(16, 9))

        num_quantiles = self.config['factor']['num_quantiles']
        if 'Q1' in equity_curves.columns:
            ax.plot(equity_curves.index, equity_curves['Q1'], label='Q1 (Cheapest Stocks)')
        if f'Q{num_quantiles}' in equity_curves.columns:
            ax.plot(equity_curves.index, equity_curves[f'Q{num_quantiles}'], label=f'Q{num_quantiles} (Most Expensive)')
        
        ax.plot(equity_curves.index, equity_curves['Long-Short'], label=f'Long-Short (Q1 - Q{num_quantiles})', color='black', linewidth=3, linestyle='--')

        ax.set_title(f"{self.config['factor']['name']} Factor Quintile Backtest (S&P 500)")
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity Curve (Log Scale, Initial Capital=1)')
        ax.set_yscale('log')
        ax.legend()
        plt.show()

# --- 主執行流程 ---
if __name__ == '__main__':
    engine = FactorBacktester(config)
    engine.run()
    engine.analyze_and_plot()
