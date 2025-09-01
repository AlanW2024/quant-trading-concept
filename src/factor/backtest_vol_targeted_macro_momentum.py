# 匯入必要的函式庫
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

# 忽略 pandas 在未來版本中的一些警告，讓輸出更乾淨
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. 策略配置中心 ---
config = {
    "universe": "SP500",
    "start_date": "1995-01-01",
    "end_date": "2024-12-31",
    "rebalance_freq": "M",
    "data": {
        "tickers": ['QQQ', 'GLD', '^VIX'] 
    },
    "risk_management": {
        "volatility_target": 0.15, 
        "volatility_lookback": 60,
        "max_leverage": 1.5
    },
    "macro_timing": {
        "vix_enter_threshold": 25,
        "defense_asset": "GLD",
        "defense_holding_period": 50,
        "fed_hike_dates": [
            '1997-03-25', '1999-06-30', '1999-08-24', '1999-10-05', '1999-11-16', 
            '2000-02-02', '2000-03-21', '2000-05-16', '2004-06-30', '2004-08-10', 
            '2004-09-21', '2004-11-10', '2004-12-14', '2005-02-02', '2005-03-22', 
            '2005-05-03', '2005-06-30', '2005-08-09', '2005-09-20', '2005-11-01', 
            '2005-12-13', '2006-01-31', '2006-03-28', '2006-05-10', '2006-06-29', 
            '2015-12-17', '2016-12-15', '2017-03-16', '2017-06-15', '2017-12-14',
            '2018-03-22', '2018-06-14', '2018-09-27', '2018-12-20', '2022-03-17',
            '2022-05-05', '2022-06-16', '2022-07-28', '2022-09-22', '2022-11-03',
            '2022-12-15', '2023-02-02', '2023-03-23', '2023-05-04', '2023-07-27'
        ]
    },
    "factor_selection": {
        "name": "Momentum",
        "lookback_period": 12,
        "skip_period": 1,
        "num_quantiles": 5,
        "selected_quantile": "Q1"
    }
}

# --- 2. 終極回測引擎 V2 ---

class VolatilityTargetingBacktester:
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
        cache_file = f"price_cache_{start_str}_to_{end_str}_{len(all_tickers)}_tickers.csv"

        if os.path.exists(cache_file):
            print(f"Loading prices from cache: {cache_file}...")
            self.price_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            missing_tickers = [t for t in all_tickers if t not in self.price_data.columns]
            if not missing_tickers:
                print("Price data loaded from cache successfully.")
                return
            else:
                print(f"Cache is incomplete. Forcing fresh download...")
        
        print(f"Downloading prices for {len(all_tickers)} tickers...")
        downloaded_data = yf.download(all_tickers, 
                                      start=self.config['start_date'], 
                                      end=self.config['end_date'])
        
        self.price_data = downloaded_data['Close']
        self.price_data = self.price_data.dropna(axis=1, how='all')
        print("Price data download complete.")
        
        print(f"Saving data to cache: {cache_file}...")
        self.price_data.to_csv(cache_file)

    def run(self):
        self._get_price_data()
        
        daily_returns = self.price_data.pct_change()

        hike_dates = pd.to_datetime(self.config['macro_timing']['fed_hike_dates'])
        vix_enter = self.config['macro_timing']['vix_enter_threshold']
        
        macro_signals = pd.DataFrame(index=self.price_data.index)
        macro_signals['vix'] = self.price_data['^VIX']
        macro_signals['is_hike_day'] = macro_signals.index.isin(hike_dates)
        macro_signals['is_high_vix'] = macro_signals['vix'] > vix_enter
        macro_signals['defense_trigger'] = (macro_signals['is_hike_day']) & (macro_signals['is_high_vix'])

        rebalance_dates = self.price_data.resample(self.config['rebalance_freq']).last().index
        
        base_positions = pd.DataFrame(0.0, index=self.price_data.index, columns=self.price_data.columns)
        current_winners = []
        in_defense_mode = False
        defense_end_date = pd.NaT

        print("Generating base strategy positions...")
        for i in range(1, len(self.price_data)):
            today = self.price_data.index[i]
            yesterday = self.price_data.index[i-1]
            
            if in_defense_mode and today > defense_end_date:
                in_defense_mode = False
            if yesterday in macro_signals.index and macro_signals.loc[yesterday, 'defense_trigger']:
                if not in_defense_mode:
                    in_defense_mode = True
                    defense_period = self.config['macro_timing']['defense_holding_period']
                    end_idx = i + defense_period
                    defense_end_date = self.price_data.index[end_idx] if end_idx < len(self.price_data) else self.price_data.index[-1]

            if yesterday in rebalance_dates:
                lookback = self.config['factor_selection']['lookback_period']
                skip = self.config['factor_selection']['skip_period']
                try:
                    start_idx_loc = rebalance_dates.get_loc(yesterday)
                    start_idx = start_idx_loc - lookback - skip
                    end_idx = start_idx_loc - skip
                    if start_idx >= 0:
                        lookback_prices = self.price_data.loc[rebalance_dates[start_idx]:rebalance_dates[end_idx]]
                        if len(lookback_prices) > 1:
                            momentum_factor = lookback_prices.iloc[-1] / lookback_prices.iloc[0] - 1
                            momentum_factor = momentum_factor.dropna()
                            if not momentum_factor.empty:
                                factor_quantiles = pd.qcut(momentum_factor, 5, labels=[f'Q{j+1}' for j in range(5)], duplicates='drop')
                                current_winners = factor_quantiles[factor_quantiles == 'Q1'].index.tolist()
                except (KeyError, ValueError): pass

            base_positions.loc[today] = 0.0
            if in_defense_mode:
                base_positions.loc[today, self.config['macro_timing']['defense_asset']] = 1.0
            else:
                if current_winners:
                    valid_winners = [t for t in current_winners if t in self.price_data.columns]
                    if valid_winners:
                        weight = 1.0 / len(valid_winners)
                        base_positions.loc[today, valid_winners] = weight
        
        base_strategy_returns = (base_positions.shift(1) * daily_returns).sum(axis=1).fillna(0)

        print("Applying volatility targeting engine...")
        vol_target = self.config['risk_management']['volatility_target']
        vol_lookback = self.config['risk_management']['volatility_lookback']
        max_leverage = self.config['risk_management']['max_leverage']
        
        realized_vol = base_strategy_returns.rolling(window=vol_lookback).std() * np.sqrt(252)
        
        leverage = vol_target / realized_vol
        leverage = leverage.fillna(1.0).clip(upper=max_leverage) 

        self.strategy_returns = (base_strategy_returns * leverage.shift(1)).fillna(0)


    def analyze_and_plot(self):
        benchmark_returns = self.price_data['QQQ'].pct_change().fillna(0)
        
        strategy_equity = (1 + self.strategy_returns).cumprod()
        benchmark_equity = (1 + benchmark_returns).cumprod()

        def calculate_metrics(returns):
            if returns.abs().sum() == 0: return {"Total Return": 0, "Annualized Return": 0, "Annualized Volatility": 0, "Sharpe Ratio": 0, "Max Drawdown": 0}
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            annualized_volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
            equity = (1 + returns).cumprod()
            max_dd = (equity / equity.cummax() - 1).min()
            return {"Total Return": total_return, "Annualized Return": annualized_return, "Annualized Volatility": annualized_volatility, "Sharpe Ratio": sharpe_ratio, "Max Drawdown": max_dd}

        strategy_metrics = calculate_metrics(self.strategy_returns)
        benchmark_metrics = calculate_metrics(benchmark_returns)

        print("\n--- Backtest Results ---")
        print("\n[Volatility Targeting Strategy]")
        for key, value in strategy_metrics.items(): print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")
        print("\n[Benchmark: Buy and Hold QQQ]")
        for key, value in benchmark_metrics.items(): print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")
        print("-" * 30)

        print("Generating performance chart...")
        fig, ax = plt.subplots(figsize=(16, 9))
        plt.style.use('seaborn-v0_8-darkgrid')

        strategy_equity.plot(ax=ax, label='Volatility Targeting Strategy', color='teal')
        benchmark_equity.plot(ax=ax, label='Benchmark (Buy and Hold QQQ)', color='grey', linestyle='--')

        ax.set_title('Final Strategy Performance: Volatility Targeting Engine')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity Curve (Log Scale)')
        ax.set_yscale('log')
        ax.legend()
        plt.show()

# --- 5. 主執行流程 ---
if __name__ == '__main__':
    engine = VolatilityTargetingBacktester(config)
    engine.run()
    engine.analyze_and_plot()
