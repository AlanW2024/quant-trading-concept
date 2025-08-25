# 匯入必要的函式庫
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import warnings
import os

# 忽略 pandas 在未來版本中的一些警告，讓輸出更乾淨
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. 策略配置中心 (Configuration Hub) ---
# 專業的做法是將所有可變參數集中管理
config = {
    "universe": "SP500",
    "start_date": "1995-01-01", 
    "end_date": "2024-12-31",
    "rebalance_freq": "M",
    "factor": {
        "name": "Momentum",
        "lookback_period": 12,
        "skip_period": 1,
    },
    "backtest": {
        "num_quantiles": 5,
    },
    "data": {
        "cache_file": "sp500_prices_cache_1995_2024.csv",
    },
    "plotting": {
        "font_path": 'C:/Windows/Fonts/msjh.ttc',
    }
}

# --- 2. 物件導向重構：專業級回測引擎 ---

class FactorBacktester:
    """
    一個模組化的、可重複使用的因子回測引擎。
    """
    def __init__(self, config):
        self.config = config
        self.tickers = []
        self.price_data = None
        self.factor_returns = None
        self.equity_curves = None
        self._setup_fonts()

    def _setup_fonts(self):
        """設定圖表字體"""
        try:
            font_path = self.config['plotting']['font_path']
            self.my_font = FontProperties(fname=font_path, size=12)
            self.title_font = FontProperties(fname=font_path, size=18)
        except FileNotFoundError:
            print(f"Warning: Font file not found at {self.config['plotting']['font_path']}. Using default.")
            self.my_font = FontProperties(size=12)
            self.title_font = FontProperties(size=18)

    def _get_sp500_tickers(self):
        """從維基百科抓取 S&P 500 的成分股列表"""
        print("Fetching S&P 500 tickers from Wikipedia...")
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)
            tickers = table[0]['Symbol'].tolist()
            self.tickers = [t.replace('.', '-') for t in tickers]
            print(f"Successfully fetched {len(self.tickers)} tickers.")
        except Exception as e:
            print(f"Failed to fetch tickers: {e}. Using a small backup list.")
            self.tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']

    def _get_price_data(self):
        """下載或從快取載入價格數據"""
        cache_file = self.config['data']['cache_file']
        if os.path.exists(cache_file):
            print(f"Loading prices from cache file: {cache_file}...")
            self.price_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            print("Price data loaded from cache.")
        else:
            self._get_sp500_tickers()
            print(f"Downloading historical prices for {len(self.tickers)} tickers...")
            self.price_data = yf.download(self.tickers, 
                                          start=self.config['start_date'], 
                                          end=self.config['end_date'])['Close']
            print("Price data download complete.")
            print(f"Saving data to cache file: {cache_file}...")
            self.price_data.to_csv(cache_file)

    def run(self):
        """執行完整的回測流程"""
        self._get_price_data()
        
        print("Calculating monthly returns...")
        monthly_returns = self.price_data.resample(self.config['rebalance_freq']).last().pct_change()
        
        portfolio_returns = pd.DataFrame()
        rebalance_dates = monthly_returns.index
        
        lookback = self.config['factor']['lookback_period']
        skip = self.config['factor']['skip_period']
        num_quantiles = self.config['backtest']['num_quantiles']

        print("Starting monthly rebalancing backtest...")
        for i in range(lookback + skip, len(rebalance_dates)):
            current_date = rebalance_dates[i]
            
            price_lookback_start = rebalance_dates[i - lookback - skip]
            price_lookback_end = rebalance_dates[i - skip]
            
            lookback_prices = self.price_data.loc[price_lookback_start:price_lookback_end]
            if lookback_prices.empty or len(lookback_prices) < 2: continue
                
            momentum_factor = lookback_prices.iloc[-1] / lookback_prices.iloc[0] - 1
            momentum_factor = momentum_factor.dropna()
            if momentum_factor.empty: continue

            labels = [f'Q{j+1}' for j in range(num_quantiles)]
            try:
                factor_quantiles = pd.qcut(momentum_factor, num_quantiles, labels=labels, duplicates='drop')
            except ValueError: continue
            
            next_month_returns = monthly_returns.loc[current_date]
            for q in factor_quantiles.unique():
                stocks_in_quantile = factor_quantiles[factor_quantiles == q].index
                valid_stocks = stocks_in_quantile.intersection(next_month_returns.dropna().index)
                portfolio_returns.loc[current_date, q] = next_month_returns[valid_stocks].mean()

        if 'Q1' in portfolio_returns.columns and f'Q{num_quantiles}' in portfolio_returns.columns:
            portfolio_returns['Long-Short'] = portfolio_returns['Q1'] - portfolio_returns[f'Q{num_quantiles}']
        else:
            portfolio_returns['Long-Short'] = 0
        
        self.factor_returns = portfolio_returns.fillna(0)
        print("Backtest complete.")

    def analyze_performance(self):
        """計算並顯示績效指標"""
        if self.factor_returns is None:
            print("Please run the backtest first.")
            return

        self.equity_curves = (1 + self.factor_returns).cumprod()

        def calculate_metrics(returns):
            if returns.abs().sum() == 0: return {"Annualized Return": 0, "Annualized Volatility": 0, "Sharpe Ratio": 0}
            annualized_return = (1 + returns).prod() ** (12 / len(returns)) - 1
            annualized_volatility = returns.std() * np.sqrt(12)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
            return {"Annualized Return": annualized_return, "Annualized Volatility": annualized_volatility, "Sharpe Ratio": sharpe_ratio}

        long_short_metrics = calculate_metrics(self.factor_returns['Long-Short'])

        print("\n--- Factor Performance (Long Winners / Short Losers) ---")
        for key, value in long_short_metrics.items():
            print(f"{key}: {value:.2%}" if "Return" in key or "Volatility" in key else f"{key}: {value:.2f}")
        print("-" * 50)

    def plot_results(self):
        """繪製績效圖表"""
        if self.equity_curves is None:
            print("Please analyze performance first.")
            return

        print("Generating performance chart...")
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(16, 9))

        num_quantiles = self.config['backtest']['num_quantiles']
        if 'Q1' in self.equity_curves.columns:
            ax.plot(self.equity_curves.index, self.equity_curves['Q1'], label='Q1 (Winners)')
        if f'Q{num_quantiles}' in self.equity_curves.columns:
            ax.plot(self.equity_curves.index, self.equity_curves[f'Q{num_quantiles}'], label=f'Q{num_quantiles} (Losers)')

        ax.plot(self.equity_curves.index, self.equity_curves['Long-Short'], label=f'Long-Short (Q1 - Q{num_quantiles})', color='black', linewidth=3, linestyle='--')

        ax.set_title(f"{self.config['factor']['name']} Factor Quintile Backtest ({self.config['universe']})", fontproperties=self.title_font)
        ax.set_xlabel('Date', fontproperties=self.my_font)
        ax.set_ylabel('Equity Curve (Log Scale, Initial Capital=1)', fontproperties=self.my_font)
        ax.set_yscale('log')
        legend = ax.legend(prop=self.my_font)
        plt.setp(legend.get_texts(), fontproperties=self.my_font)
        plt.show()

# --- 5. 主執行流程 ---
if __name__ == '__main__':
    backtester = FactorBacktester(config)
    backtester.run()
    backtester.analyze_performance()
    backtester.plot_results()
