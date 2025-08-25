# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

# Suppress future warnings for a cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. Strategy Configuration ---
config = {
    "start_date": "1995-01-01",
    "end_date": "2024-12-31",
    "rebalance_freq": "M",
    "data": {
        "tickers": ['QQQ', '^VIX'], # QQQ for benchmark
        "cache_file_prefix": "price_cache"
    },
    "risk_management": {
        "volatility_target": 0.15, 
        "volatility_lookback": 60,
        "max_leverage": 1.5
    },
    "factor_selection": {
        "name": "Momentum",
        "lookback_period": 12,
        "skip_period": 1,
        "num_quantiles": 5,
        "selected_quantile": "Q1" # We are long the top quintile
    }
}

# --- 2. Professional Backtesting Engine V5 ---

class VolTargetedFactorBacktester:
    def __init__(self, config):
        self.config = config
        self.sp500_tickers = []
        self.price_data = None
        self.strategy_returns = None

    def _get_sp500_tickers(self):
        print("Fetching S&P 500 tickers...")
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)
            self.sp500_tickers = [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
        except Exception as e:
            print(f"Failed to fetch tickers: {e}. Using a small backup list.")
            self.sp500_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']

    def _get_price_data(self):
        self._get_sp500_tickers()
        all_tickers = sorted(list(set(self.sp500_tickers + self.config['data']['tickers'])))
        
        start_str = self.config['start_date']
        end_str = self.config['end_date']
        cache_file = f"{self.config['data']['cache_file_prefix']}_{start_str}_to_{end_str}_{len(all_tickers)}_tickers.csv"

        if os.path.exists(cache_file):
            print(f"Loading prices from cache: {cache_file}...")
            self.price_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            missing = [t for t in all_tickers if t not in self.price_data.columns]
            if not missing:
                print("Price data loaded successfully.")
                return
            else:
                print(f"Cache is incomplete. Forcing fresh download...")
        
        print(f"Downloading prices for {len(all_tickers)} tickers...")
        self.price_data = yf.download(all_tickers, start=start_str, end=end_str)['Close']
        self.price_data = self.price_data.dropna(axis=1, how='all')
        print(f"Saving data to cache: {cache_file}...")
        self.price_data.to_csv(cache_file)

    def run(self):
        self._get_price_data()
        
        # --- a. Calculate Monthly Factor Portfolio Returns (The Alpha Engine) ---
        print("Calculating base factor portfolio returns...")
        monthly_returns = self.price_data.resample(self.config['rebalance_freq']).last().pct_change()
        rebalance_dates = monthly_returns.index
        
        factor_portfolio_returns = pd.Series(index=rebalance_dates, dtype=float)
        
        lookback = self.config['factor_selection']['lookback_period']
        skip = self.config['factor_selection']['skip_period']

        for date in rebalance_dates[lookback+skip:]:
            try:
                start_loc = rebalance_dates.get_loc(date) - lookback - skip
                end_loc = rebalance_dates.get_loc(date) - skip
                
                prices = self.price_data.loc[rebalance_dates[start_loc]:rebalance_dates[end_loc]]
                if len(prices) > 1:
                    factor = prices.iloc[-1] / prices.iloc[0] - 1
                    valid_factor = factor.dropna()
                    valid_factor = valid_factor[valid_factor.index.isin(self.sp500_tickers)]
                    if not valid_factor.empty:
                        quantiles = pd.qcut(valid_factor, 5, labels=False, duplicates='drop')
                        winners = valid_factor[quantiles == 4].index.tolist()
                        
                        if winners:
                            valid_winners = [t for t in winners if t in monthly_returns.columns]
                            factor_portfolio_returns[date] = monthly_returns.loc[date, valid_winners].mean()
            except (KeyError, ValueError):
                continue
        
        factor_portfolio_returns = factor_portfolio_returns.fillna(0)
        
        # --- b. Convert Monthly Returns to Daily Returns for Volatility Targeting ---
        daily_factor_returns = factor_portfolio_returns.resample('D').bfill().reindex(self.price_data.index).fillna(0) / 21 # Approximate daily returns

        # --- c. Apply Volatility Targeting Engine (The Risk Shield) ---
        print("Applying volatility targeting engine...")
        vol_target = self.config['risk_management']['volatility_target']
        vol_lookback = self.config['risk_management']['volatility_lookback']
        max_leverage = self.config['risk_management']['max_leverage']
        
        realized_vol = daily_factor_returns.rolling(window=vol_lookback).std() * np.sqrt(252)
        
        leverage = vol_target / realized_vol
        leverage = leverage.fillna(1.0).clip(upper=max_leverage) 

        self.strategy_returns = (daily_factor_returns * leverage.shift(1)).fillna(0)

    def analyze_and_plot(self):
        print("\n--- Final Backtest Results ---")
        
        benchmark_returns = self.price_data['QQQ'].pct_change().fillna(0)
        
        results = {
            "Vol-Targeted Factor": self._calculate_metrics(self.strategy_returns),
            "Benchmark (QQQ)": self._calculate_metrics(benchmark_returns)
        }
        
        results_df = pd.DataFrame(results).T
        print(results_df.to_string(formatters={
            'Total Return': '{:,.2%}'.format,
            'Annualized Return': '{:,.2%}'.format,
            'Annualized Volatility': '{:,.2%}'.format,
            'Sharpe Ratio': '{:,.2f}'.format,
            'Max Drawdown': '{:,.2%}'.format
        }))
        
        print("\nGenerating performance chart...")
        equity_curves = pd.DataFrame({
            "Volatility-Targeted Factor Strategy": (1 + self.strategy_returns).cumprod(),
            "Benchmark (Buy and Hold QQQ)": (1 + benchmark_returns).cumprod()
        })
        
        fig, ax = plt.subplots(figsize=(16, 9))
        equity_curves.plot(ax=ax, logy=True,
                           title='Final Strategy: Volatility-Targeted Momentum Factor',
                           linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity Curve (Log Scale)')
        ax.legend(title='Strategy')
        plt.show()

    def _calculate_metrics(self, returns):
        if returns.abs().sum() == 0: return {"Total Return": 0, "Annualized Return": 0, "Annualized Volatility": 0, "Sharpe Ratio": 0, "Max Drawdown": 0}
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        equity = (1 + returns).cumprod()
        max_dd = (equity / equity.cummax() - 1).min()
        return {"Total Return": total_return, "Annualized Return": annualized_return, "Annualized Volatility": annualized_volatility, "Sharpe Ratio": sharpe, "Max Drawdown": max_dd}

# --- 3. Main Execution ---
if __name__ == '__main__':
    engine = VolTargetedFactorBacktester(config)
    engine.run()
    engine.analyze_and_plot()
