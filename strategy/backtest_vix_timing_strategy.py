# 匯入必要的函式庫
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# --- 1. 定義策略參數與環境設定 ---

# a. 策略標的與指數
TICKERS = ['QQQ', '^VIX']
STRATEGY_ASSET = 'QQQ'
VIX_TICKER = '^VIX'

# b. 數據下載區間
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

# c. 交易規則參數
FED_HIKE_DATES = [
    '2015-12-17', '2016-12-15', '2017-03-16', '2017-06-15', '2017-12-14',
    '2018-03-22', '2018-06-14', '2018-09-27', '2018-12-20', '2022-03-17',
    '2022-05-05', '2022-06-16', '2022-07-28', '2022-09-22', '2022-11-03',
    '2022-12-15', '2023-02-02', '2023-03-23', '2023-05-04', '2023-07-27'
]
hike_dates = pd.to_datetime(FED_HIKE_DATES)
VIX_THRESHOLD = 25
HOLDING_PERIOD = 30 # 持有30個交易日

# d. 設定中文字體
try:
    font_path = 'C:/Windows/Fonts/msjh.ttc'
    my_font = FontProperties(fname=font_path, size=12)
    title_font = FontProperties(fname=font_path, size=20)
except FileNotFoundError:
    my_font = FontProperties(size=12)
    title_font = FontProperties(size=20)

# --- 2. 數據準備 ---

print("Downloading historical data...")
all_data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
prices = all_data[['Open', 'Close']]
print("Data download complete.")

# --- 3. 生成交易訊號 ---

print("Generating trading signals...")
signals = pd.DataFrame(index=prices.index)
signals['vix'] = prices['Close'][VIX_TICKER]
signals['is_hike_day'] = signals.index.isin(hike_dates)
signals['is_low_vix'] = signals['vix'] < VIX_THRESHOLD
signals['entry_signal'] = (signals['is_hike_day']) & (signals['is_low_vix'])

# --- 4. 建立部位狀態 ---

print("Constructing position states...")
# *** 修正點：初始化 position 欄位為整數型別，以避免 FutureWarning ***
signals['position'] = 0
entry_indices = signals[signals['entry_signal']].index

# 找出所有滿足進場條件的日期，並標記其後的持倉週期
for entry_date in entry_indices:
    entry_loc = signals.index.get_loc(entry_date)
    # 我們在訊號出現的 "隔天" 開盤進場
    start_pos = entry_loc + 1
    end_pos = start_pos + HOLDING_PERIOD
    if end_pos < len(signals):
        # 檢查此持倉週期是否與現有持倉重疊
        if signals.iloc[start_pos:end_pos]['position'].sum() == 0:
            signals.iloc[start_pos:end_pos, signals.columns.get_loc('position')] = 1

# --- 5. 執行回測與計算績效 ---

print("Running backtest...")
qqq_returns = prices['Open'][STRATEGY_ASSET].pct_change()
# 使用 .shift(1) 來確保我們用昨天的部位決定今天的報酬，避免未來函數
signals['strategy_returns'] = qqq_returns * signals['position'].shift(1).fillna(0)
signals['benchmark_returns'] = qqq_returns.fillna(0)

initial_capital = 1.0
signals['strategy_equity'] = initial_capital * (1 + signals['strategy_returns']).cumprod()
signals['benchmark_equity'] = initial_capital * (1 + signals['benchmark_returns']).cumprod()

# --- 6. 定義績效指標計算函式 ---

def calculate_performance_metrics(returns_series):
    """計算指定報酬率序列的績效指標"""
    risk_free_rate = 0.0
    trading_days = 252
    
    if len(returns_series) == 0 or returns_series.sum() == 0:
        return {
            "Total Return": 0.0, "Annualized Return": 0.0, "Annualized Volatility": 0.0,
            "Sharpe Ratio": 0.0, "Max Drawdown": 0.0
        }
        
    total_return = (1 + returns_series).prod() - 1
    annualized_return = (1 + total_return) ** (trading_days / len(returns_series)) - 1
    annualized_volatility = returns_series.std() * np.sqrt(trading_days)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
    
    equity_curve = (1 + returns_series).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # *** 修正點：將字典的鍵改為英文 ***
    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }

# --- 7. 計算並印出結果 ---

strategy_metrics = calculate_performance_metrics(signals['strategy_returns'])
benchmark_metrics = calculate_performance_metrics(signals['benchmark_returns'])

print("\n--- Backtest Results ---")
print("\n[VIX Timing Strategy]")
for key, value in strategy_metrics.items():
    print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")

print("\n[Benchmark: Buy and Hold QQQ]")
for key, value in benchmark_metrics.items():
    print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")
print("-" * 30)

# --- 8. 可視化 ---

print("Generating equity curve chart...")
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(16, 9))

signals['strategy_equity'].plot(ax=ax, label='VIX 時機策略', color='royalblue')
signals['benchmark_equity'].plot(ax=ax, label='基準策略 (無腦持有)', color='grey', linestyle='--')

ax.set_title('策略績效 vs. 基準', fontproperties=title_font)
ax.set_xlabel('日期', fontproperties=my_font)
ax.set_ylabel('權益曲線 (初始資金=1)', fontproperties=my_font)
legend = ax.legend(prop=my_font)
plt.setp(legend.get_texts(), fontproperties=my_font)
plt.show()
