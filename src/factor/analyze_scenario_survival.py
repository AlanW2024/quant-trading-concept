# ==============================================================================
# 匯入必要的函式庫 (Import Necessary Libraries)
# ==============================================================================
# 這些是我們執行分析所需的工具，就像廚師的刀具和鍋具。
import yfinance as yf         # 用於從 Yahoo Finance 下載股票數據
import pandas as pd           # 數據分析的核心工具，可以把它想像成一個超強的 Excel
import numpy as np            # 進行科學計算，特別是數學運算
import matplotlib.pyplot as plt # 用於繪製圖表，將數據視覺化
import warnings               # 用於管理程式運行時可能出現的警告訊息
import os                     # 用於和操作系統互動，例如檢查檔案是否存在

# 忽略 pandas 在未來版本中可能出現的一些警告訊息，讓我們的輸出結果更乾淨。
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# 策略配置中心 (Configuration Hub)
# ==============================================================================
# 這是我們最終、最優的 Level 15 策略設定
config = {
    "start_date": "1995-01-01",
    "end_date": "2024-12-31",
    "rebalance_freq": "M",
    "data": { "tickers": ['QQQ', 'GLD', '^VIX'] },
    "costs": {
        "transaction_pct": 0.001,  # 0.1% 的交易成本 (手續費+稅)
        "slippage_pct": 0.0005     # 0.05% 的滑價成本
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
        "lookback_period": 12,
        "skip_period": 1,
        "num_quantiles": 5,
    }
}

# ==============================================================================
# 專業級回測引擎 (The Professional Backtesting Engine)
# ==============================================================================
# 我們將 Level 15 的引擎作為一個可重複使用的模組
class ProfessionalBacktester:
    def __init__(self, config):
        self.config = config
        self.sp500_tickers = []
        self.price_data = None
        self.strategy_returns_net = None

    def _get_sp500_tickers(self):
        # 這個函式負責獲取 S&P 500 的成分股列表
        # print("Fetching S&P 500 tickers...")
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)
            self.sp500_tickers = [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
        except Exception:
            self.sp500_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']

    def _get_price_data(self):
        # 這個函式負責下載或從快取讀取價格數據
        if self.price_data is not None: return
        self._get_sp500_tickers()
        all_tickers = sorted(list(set(self.sp500_tickers + self.config['data']['tickers'])))
        cache_file = f"price_cache_{self.config['start_date']}_to_{self.config['end_date']}_{len(all_tickers)}_tickers.csv"
        if os.path.exists(cache_file):
            # print(f"Loading prices from cache: {cache_file}...")
            self.price_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return
        print(f"Downloading prices for {len(all_tickers)} tickers...")
        self.price_data = yf.download(all_tickers, start=self.config['start_date'], end=self.config['end_date'])['Close'].dropna(axis=1, how='all')
        self.price_data.to_csv(cache_file)

    def run(self, external_returns=None):
        # 這個函式是回測的核心邏輯
        if external_returns is not None:
             # 如果提供了外部報酬率（用於情境分析），則直接使用
            base_strategy_returns = external_returns
            # 為了成本計算，我們需要一個與 price_data 結構相同的虛擬 positions DataFrame
            if self.price_data is None: self._get_price_data()
            positions = pd.DataFrame(0.0, index=self.price_data.index, columns=self.price_data.columns)
        else:
            # 否則，正常執行回測以產生基礎策略報酬率
            self._get_price_data()
            daily_returns = self.price_data.pct_change()
            rebalance_dates = self.price_data.resample(self.config['rebalance_freq']).last().index
            hike_dates = pd.to_datetime(self.config['macro_timing']['fed_hike_dates'])
            macro_signals = pd.DataFrame({'vix': self.price_data['^VIX']}, index=self.price_data.index)
            macro_signals['defense_trigger'] = (macro_signals.index.isin(hike_dates)) & (macro_signals['vix'] > self.config['macro_timing']['vix_enter_threshold'])
            
            positions = pd.DataFrame(0.0, index=self.price_data.index, columns=self.price_data.columns)
            current_winners, in_defense_mode, defense_end_date = [], False, pd.NaT
            
            # print("Generating base strategy positions...")
            for i in range(1, len(self.price_data)):
                today, yesterday = self.price_data.index[i], self.price_data.index[i-1]
                if in_defense_mode and today > defense_end_date: in_defense_mode = False
                if yesterday in macro_signals.index and macro_signals.loc[yesterday, 'defense_trigger']:
                    if not in_defense_mode:
                        in_defense_mode = True
                        period = self.config['macro_timing']['defense_holding_period']
                        end_idx = i + period
                        defense_end_date = self.price_data.index[end_idx] if end_idx < len(self.price_data) else self.price_data.index[-1]

                if yesterday in rebalance_dates:
                    try:
                        start_loc = rebalance_dates.get_loc(yesterday) - self.config['factor_selection']['lookback_period'] - self.config['factor_selection']['skip_period']
                        end_loc = rebalance_dates.get_loc(yesterday) - self.config['factor_selection']['skip_period']
                        if start_loc >= 0:
                            prices = self.price_data.loc[rebalance_dates[start_loc]:rebalance_dates[end_loc]]
                            if len(prices) > 1:
                                factor = (prices.iloc[-1] / prices.iloc[0] - 1).dropna()
                                valid_factor = factor[factor.index.isin(self.sp500_tickers)]
                                if not valid_factor.empty:
                                    quantiles = pd.qcut(valid_factor, 5, labels=False, duplicates='drop')
                                    current_winners = valid_factor[quantiles == 4].index.tolist()
                    except (KeyError, ValueError): pass

                positions.loc[today] = 0.0
                if in_defense_mode:
                    positions.loc[today, self.config['macro_timing']['defense_asset']] = 1.0
                elif current_winners:
                    valid_winners = [t for t in current_winners if t in self.price_data.columns]
                    if valid_winners: positions.loc[today, valid_winners] = 1.0 / len(valid_winners)
            
            base_strategy_returns = (positions.shift(1) * daily_returns).sum(axis=1).fillna(0)
        
        # --- 應用成本模擬與波動率目標引擎 ---
        turnover = positions.diff().abs().sum(axis=1)
        total_cost_rate = self.config['costs']['transaction_pct'] + self.config['costs']['slippage_pct']
        daily_costs = turnover * total_cost_rate
        base_strategy_returns_net = base_strategy_returns - daily_costs

        vol_target = self.config['risk_management']['volatility_target']
        vol_lookback = self.config['risk_management']['volatility_lookback']
        max_leverage = self.config['risk_management']['max_leverage']
        realized_vol = base_strategy_returns.rolling(window=vol_lookback).std() * np.sqrt(252)
        leverage = (vol_target / realized_vol).fillna(1.0).clip(upper=max_leverage)
        self.strategy_returns_net = (base_strategy_returns_net * leverage.shift(1)).fillna(0)
        return self.strategy_returns_net

    def get_performance_metrics(self, returns):
        # 這個函式只負責計算績效，並返回結果
        if returns.abs().sum() == 0: return {"Sharpe Ratio": 0, "Max Drawdown": 0, "Annualized Return": 0}
        annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        equity = (1 + returns).cumprod()
        max_dd = (equity / equity.cummax() - 1).min()
        return {"Sharpe Ratio": sharpe, "Max Drawdown": max_dd, "Annualized Return": annualized_return}

# ==============================================================================
# 情境分析實驗室 (The Scenario Analysis Lab)
# ==============================================================================
def run_scenario_analysis(base_config):
    print("\n" + "="*50)
    print("Starting Strategy Stress Test: Scenario Analysis")
    print("="*50)

    # --- 準備基礎數據 ---
    print("\nRunning baseline backtest (Normal World)...")
    engine = ProfessionalBacktester(base_config)
    baseline_returns = engine.run()
    baseline_metrics = engine.get_performance_metrics(baseline_returns)

    # --- 劇本一：閃電崩盤 ---
    print("\n--- Scenario 1: Flash Crash Simulation ---")
    flash_crash_returns = baseline_returns.copy()
    crash_date = '2017-08-08'
    if crash_date in flash_crash_returns.index:
        # 在指定日期，人為地製造一個 -20% 的單日暴跌
        flash_crash_returns.loc[crash_date] = -0.20
        
        # 重新運行波動率引擎，處理這個衝擊
        engine_crash = ProfessionalBacktester(base_config)
        engine_crash.price_data = engine.price_data # 重複使用已載入的數據，避免重新下載
        vol_targeted_crash_returns = engine_crash.run(external_returns=flash_crash_returns)
        crash_metrics = engine_crash.get_performance_metrics(vol_targeted_crash_returns)
    else:
        # 如果找不到日期，則跳過此情境
        vol_targeted_crash_returns = pd.Series(dtype=float)
        crash_metrics = {"Sharpe Ratio": np.nan, "Max Drawdown": np.nan, "Annualized Return": np.nan}

    # --- 劇本二：動量崩潰 (失落十年) ---
    print("\n--- Scenario 2: Momentum Crash (Lost Decade) Simulation ---")
    momentum_crash_returns = baseline_returns.copy()
    # 我們將 2009-2019 牛市這段時間的策略報酬率「反轉」，模擬因子失效
    crash_period = (momentum_crash_returns.index >= '2009-01-01') & (momentum_crash_returns.index <= '2019-12-31')
    momentum_crash_returns[crash_period] *= -1
    
    engine_momocrash = ProfessionalBacktester(base_config)
    engine_momocrash.price_data = engine.price_data # 重複使用數據
    vol_targeted_momocrash_returns = engine_momocrash.run(external_returns=momentum_crash_returns)
    momocrash_metrics = engine_momocrash.get_performance_metrics(vol_targeted_momocrash_returns)

    # --- 彙總並輸出結果 ---
    print("\n--- Scenario Analysis Results ---")
    results = {
        "Baseline (Normal World)": baseline_metrics,
        "Flash Crash Scenario": crash_metrics,
        "Momentum Crash Scenario": momocrash_metrics
    }
    results_df = pd.DataFrame(results).T
    print(results_df.to_string(formatters={
        'Sharpe Ratio': '{:,.2f}'.format,
        'Max Drawdown': '{:,.2%}'.format,
        'Annualized Return': '{:,.2%}'.format
    }))

    # --- 繪製圖表 ---
    print("\nGenerating scenario analysis charts...")
    plt.style.use('seaborn-v0_8-darkgrid')

    # *** 核心修正一：為閃電崩盤建立獨立、清晰的圖表 ***
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    baseline_equity = (1 + baseline_returns).cumprod()
    crash_equity = (1 + vol_targeted_crash_returns).cumprod() if not vol_targeted_crash_returns.empty else pd.Series()
    
    baseline_equity.plot(ax=ax1, label='Baseline (Normal World)', color='blue', linewidth=2, zorder=2)
    if not crash_equity.empty:
        crash_equity.plot(ax=ax1, label='Flash Crash Scenario', color='orange', linewidth=2, zorder=3)
    
    ax1.set_title('Scenario 1: Flash Crash Survival Test', fontsize=18)
    ax1.set_xlabel('Date'); ax1.set_ylabel('Equity Curve (Log Scale)'); ax1.set_yscale('log')
    ax1.legend(title='Scenario')
    ax1.grid(True, which="both", ls="--", linewidth=0.5)
    
    # 建立一個更清晰的放大鏡
    if not crash_equity.empty:
        zoom_start, zoom_end = '2017-07-01', '2017-10-31'
        ax_inset = ax1.inset_axes([0.05, 0.45, 0.4, 0.45])
        baseline_equity.plot(ax=ax_inset, color='blue')
        crash_equity.plot(ax=ax_inset, color='orange')
        ax_inset.set_xlim(zoom_start, zoom_end)
        y_min = min(baseline_equity.loc[zoom_start:zoom_end].min(), crash_equity.loc[zoom_start:zoom_end].min())
        y_max = max(baseline_equity.loc[zoom_start:zoom_end].max(), crash_equity.loc[zoom_start:zoom_end].max())
        ax_inset.set_ylim(y_min * 0.95, y_max * 1.05)
        ax_inset.set_title('Flash Crash Detail')
        ax_inset.axvline(pd.to_datetime(crash_date), color='red', linestyle='--', linewidth=1.5, label='Crash Event')
        ax_inset.legend()
        ax1.indicate_inset_zoom(ax_inset, edgecolor="black")
    plt.show()

    # *** 核心修正二：為動量崩潰建立獨立的、帶有相對強度的故事圖表 ***
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    momocrash_equity = (1 + vol_targeted_momocrash_returns).cumprod()
    
    # 上層圖：權益曲線
    baseline_equity.plot(ax=ax2, label='Baseline (Normal World)', color='blue', linewidth=2)
    momocrash_equity.plot(ax=ax2, label='Momentum Crash Scenario', color='green', linewidth=2)
    ax2.set_title('Scenario 2: Momentum Crash (Lost Decade) Test', fontsize=18)
    ax2.set_ylabel('Equity Curve (Log Scale)')
    ax2.set_yscale('log')
    ax2.legend(title='Scenario')
    ax2.grid(True, which="both", ls="--", linewidth=0.5)

    # 下層圖：相對強度
    relative_strength = momocrash_equity / baseline_equity
    relative_strength.plot(ax=ax3, label='Relative Strength (Scenario / Baseline)', color='purple')
    ax3.axhline(1, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Date'); ax3.set_ylabel('Relative Strength')
    ax3.grid(True, which="both", ls="--", linewidth=0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.show()

# --- 主執行流程 ---
if __name__ == '__main__':
    run_scenario_analysis(config)
