# 匯入必要的函式庫
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# --- 1. 定義策略與環境 ---

# a. 資產代碼
TICKERS = ['QQQ', 'GLD', '^VIX']
ATTACK_ASSET = 'QQQ'
DEFENSE_ASSET = 'GLD'
VIX_TICKER = '^VIX'

# b. *** 核心：定義時間切點與優化後的參數 ***
REGIME_SHIFT_DATE = '2022-01-01'
OPTIMIZED_HOLDING_PERIOD = 50 # 根據 Level 9 的結論
VIX_THRESHOLD = 25
FED_HIKE_DATES = [
    '2015-12-17', '2016-12-15', '2017-03-16', '2017-06-15', '2017-12-14',
    '2018-03-22', '2018-06-14', '2018-09-27', '2018-12-20', '2022-03-17',
    '2022-05-05', '2022-06-16', '2022-07-28', '2022-09-22', '2022-11-03',
    '2022-12-15', '2023-02-02', '2023-03-23', '2023-05-04', '2023-07-27'
]
hike_dates = pd.to_datetime(FED_HIKE_DATES)

# c. 設定中文字體
try:
    font_path = 'C:/Windows/Fonts/msjh.ttc'
    my_font = FontProperties(fname=font_path, size=12)
    title_font = FontProperties(fname=font_path, size=18)
except FileNotFoundError:
    my_font = FontProperties(size=12)
    title_font = FontProperties(size=18)

# --- 2. 下載完整數據 ---

print("Downloading full historical data...")
all_data = yf.download(TICKERS, start='2015-01-01', end='2024-12-31')
prices = all_data[['Open', 'Close']]
print("Data download complete.")

# --- 3. 將回測與績效計算打包成一個函式 ---

def run_strategy_on_period(price_data, hike_dates_list, vix_threshold, holding_period):
    """在給定的數據上，執行完整的 Level 4 策略回測"""
    # 準備訊號
    signals = pd.DataFrame(index=price_data.index)
    signals['vix'] = price_data['Close'][VIX_TICKER]
    signals['is_hike_day'] = signals.index.isin(hike_dates_list)
    signals['is_high_vix'] = signals['vix'] > vix_threshold
    signals['defense_trigger'] = (signals['is_hike_day']) & (signals['is_high_vix'])

    # 建立部位
    signals['position_qqq'] = 1
    signals['position_gld'] = 0
    defense_indices = signals[signals['defense_trigger']].index
    for trigger_date in defense_indices:
        if trigger_date not in signals.index: continue
        trigger_loc = signals.index.get_loc(trigger_date)
        start_pos = trigger_loc + 1
        end_pos = start_pos + holding_period
        if end_pos < len(signals):
            signals.iloc[start_pos:end_pos, signals.columns.get_loc('position_qqq')] = 0
            signals.iloc[start_pos:end_pos, signals.columns.get_loc('position_gld')] = 1
    
    # 計算報酬率
    qqq_returns = price_data['Open'][ATTACK_ASSET].pct_change()
    gld_returns = price_data['Open'][DEFENSE_ASSET].pct_change()
    strategy_returns = (qqq_returns * signals['position_qqq'].shift(1) + gld_returns * signals['position_gld'].shift(1)).fillna(0)
    benchmark_returns = qqq_returns.fillna(0)
    
    # 計算績效指標
    def calculate_metrics(returns):
        if returns.abs().sum() == 0: return {"Sharpe Ratio": 0, "Max Drawdown": 0, "Total Return": 0}
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        equity = (1 + returns).cumprod()
        max_dd = (equity / equity.cummax() - 1).min()
        return {"Sharpe Ratio": sharpe, "Max Drawdown": max_dd, "Total Return": total_return}

    return calculate_metrics(strategy_returns), calculate_metrics(benchmark_returns)

# --- 4. 執行兩階段實驗 ---

print("\n--- Running In-Sample Test (Pre-2022) ---")
in_sample_prices = prices[prices.index < REGIME_SHIFT_DATE]
in_sample_hikes = [d for d in hike_dates if d < pd.to_datetime(REGIME_SHIFT_DATE)]
strategy_metrics_in, benchmark_metrics_in = run_strategy_on_period(in_sample_prices, in_sample_hikes, VIX_THRESHOLD, OPTIMIZED_HOLDING_PERIOD)

print("\n[Strategy Performance - Pre-2022]")
for key, value in strategy_metrics_in.items(): print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")
print("\n[Benchmark Performance - Pre-2022]")
for key, value in benchmark_metrics_in.items(): print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")

print("\n" + "="*40)

print("\n--- Running Out-of-Sample Test (Post-2022) ---")
out_sample_prices = prices[prices.index >= REGIME_SHIFT_DATE]
out_sample_hikes = [d for d in hike_dates if d >= pd.to_datetime(REGIME_SHIFT_DATE)]
strategy_metrics_out, benchmark_metrics_out = run_strategy_on_period(out_sample_prices, out_sample_hikes, VIX_THRESHOLD, OPTIMIZED_HOLDING_PERIOD)

print("\n[Strategy Performance - Post-2022]")
for key, value in strategy_metrics_out.items(): print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")
print("\n[Benchmark Performance - Post-2022]")
for key, value in benchmark_metrics_out.items(): print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")
print("-" * 40)

# --- 5. 結論 ---
print("\n--- Diagnosis Conclusion ---")
if strategy_metrics_in["Sharpe Ratio"] > benchmark_metrics_in["Sharpe Ratio"] and \
   strategy_metrics_out["Sharpe Ratio"] < benchmark_metrics_out["Sharpe Ratio"]:
    print("Hypothesis Confirmed: A significant regime shift occurred around 2022.")
    print("The strategy, which was effective in the old low-inflation environment, failed in the new macro regime.")
    print("This justifies the need for a new defensive asset model.")
else:
    print("Hypothesis Not Confirmed: The strategy's failure is likely not due to a regime shift.")
    print("We may need to re-examine other core assumptions.")

