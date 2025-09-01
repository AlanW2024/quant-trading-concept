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
        "tickers": ['QQQ', 'GLD', '^VIX', 'SPY'], # SPY for market context
        "cache_file_prefix": "price_cache"
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
        "lookback_period": 12,
        "skip_period": 1,
        "num_quantiles": 5,
    }
}

# --- 2. Professional Backtesting Engine V4 ---

class AttributionBacktester:
    def __init__(self, config):
        self.config = config
        self.sp500_tickers = []
        self.price_data = None
        self.all_returns = pd.DataFrame()

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

    def run_attribution(self):
        self._get_price_data()
        
        daily_returns = self.price_data.pct_change()
        rebalance_dates = self.price_data.resample(self.config['rebalance_freq']).last().index

        hike_dates = pd.to_datetime(self.config['macro_timing']['fed_hike_dates'])
        macro_signals = pd.DataFrame(index=self.price_data.index)
        macro_signals['vix'] = self.price_data['^VIX']
        macro_signals['is_hike_day'] = macro_signals.index.isin(hike_dates)
        macro_signals['is_high_vix'] = macro_signals['vix'] > self.config['macro_timing']['vix_enter_threshold']
        macro_signals['defense_trigger'] = (macro_signals['is_hike_day']) & (macro_signals['is_high_vix'])

        print("Calculating monthly momentum winners...")
        momentum_winners = pd.Series(index=rebalance_dates, dtype=object)
        lookback = self.config['factor_selection']['lookback_period']
        skip = self.config['factor_selection']['skip_period']
        for date in rebalance_dates:
            try:
                start_loc = rebalance_dates.get_loc(date) - lookback - skip
                end_loc = rebalance_dates.get_loc(date) - skip
                if start_loc >= 0:
                    prices = self.price_data.loc[rebalance_dates[start_loc]:rebalance_dates[end_loc]]
                    if len(prices) > 1:
                        factor = prices.iloc[-1] / prices.iloc[0] - 1
                        valid_factor = factor.dropna()
                        valid_factor = valid_factor[valid_factor.index.isin(self.sp500_tickers)]
                        if not valid_factor.empty:
                            quantiles = pd.qcut(valid_factor, 5, labels=False, duplicates='drop')
                            momentum_winners[date] = valid_factor[quantiles == 4].index.tolist()
            except (KeyError, ValueError): continue
        
        print("Generating daily positions for all strategies...")
        
        in_defense_mode = False
        defense_end_date = pd.NaT
        current_winners = []
        for i in range(1, len(self.price_data)):
            today = self.price_data.index[i]
            yesterday = self.price_data.index[i-1]

            if in_defense_mode and today > defense_end_date: in_defense_mode = False
            if yesterday in macro_signals.index and macro_signals.loc[yesterday, 'defense_trigger']:
                if not in_defense_mode:
                    in_defense_mode = True
                    period = self.config['macro_timing']['defense_holding_period']
                    end_idx = i + period
                    defense_end_date = self.price_data.index[end_idx] if end_idx < len(self.price_data) else self.price_data.index[-1]
            
            if yesterday in rebalance_dates:
                if isinstance(momentum_winners.get(yesterday), list):
                    current_winners = momentum_winners.get(yesterday)

            # Macro-Only Strategy
            if in_defense_mode:
                self.all_returns.loc[today, 'Macro-Only'] = daily_returns.loc[today, self.config['macro_timing']['defense_asset']]
            else:
                self.all_returns.loc[today, 'Macro-Only'] = daily_returns.loc[today, 'QQQ']

            # Factor-Only Strategy
            if current_winners:
                valid_winners = [t for t in current_winners if t in daily_returns.columns]
                if valid_winners:
                    self.all_returns.loc[today, 'Factor-Only'] = daily_returns.loc[today, valid_winners].mean()

            # Macro-Momentum (Fusion) Strategy
            if in_defense_mode:
                self.all_returns.loc[today, 'Macro-Momentum'] = daily_returns.loc[today, self.config['macro_timing']['defense_asset']]
            else:
                if current_winners:
                    valid_winners = [t for t in current_winners if t in daily_returns.columns]
                    if valid_winners:
                        self.all_returns.loc[today, 'Macro-Momentum'] = daily_returns.loc[today, valid_winners].mean()
        
        self.all_returns['Benchmark (QQQ)'] = daily_returns['QQQ']
        self.all_returns = self.all_returns.fillna(0)

    def analyze_and_plot(self):
        print("\n--- Performance Attribution Results ---")
        
        results = {}
        for col in self.all_returns.columns:
            results[col] = self._calculate_metrics(self.all_returns[col])
        
        results_df = pd.DataFrame(results).T
        print(results_df.to_string(formatters={
            'Total Return': '{:,.2%}'.format,
            'Annualized Return': '{:,.2%}'.format,
            'Annualized Volatility': '{:,.2%}'.format,
            'Sharpe Ratio': '{:,.2f}'.format,
            'Max Drawdown': '{:,.2%}'.format
        }))
        
        print("\nGenerating performance chart...")
        equity_curves = (1 + self.all_returns).cumprod()
        
        fig, ax = plt.subplots(figsize=(16, 9))
        equity_curves.plot(ax=ax, logy=True,
                           title='Performance Attribution: Deconstructing Alpha',
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
    engine = AttributionBacktester(config)
    engine.run_attribution()
    engine.analyze_and_plot()
