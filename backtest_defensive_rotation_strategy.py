# 匯入必要的函式庫
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# --- 1. 定義策略參數與環境設定 ---

# a. 資產代碼
TICKERS = ['QQQ', 'GLD', '^VIX']
ATTACK_ASSET = 'QQQ'
DEFENSE_ASSET = 'GLD'
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
HOLDING_PERIOD = 30 

# d. 設定中文字體
try:
    font_path = 'C:/Windows/Fonts/msjh.ttc'
    my_font = FontProperties(fname=font_path, size=12)
    title_font = FontProperties(fname=font_path, size=18)
except FileNotFoundError:
    my_font = FontProperties(size=12)
    title_font = FontProperties(size=18)

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
signals['is_high_vix'] = signals['vix'] > VIX_THRESHOLD
signals['defense_trigger'] = (signals['is_hike_day']) & (signals['is_high_vix'])

# --- 4. 建立多資產部位狀態 ---

print("Constructing position states...")
signals['position_qqq'] = 1
signals['position_gld'] = 0
defense_indices = signals[signals['defense_trigger']].index

for trigger_date in defense_indices:
    trigger_loc = signals.index.get_loc(trigger_date)
    start_pos = trigger_loc + 1
    end_pos = start_pos + HOLDING_PERIOD
    
    if end_pos < len(signals):
        signals.iloc[start_pos:end_pos, signals.columns.get_loc('position_qqq')] = 0
        signals.iloc[start_pos:end_pos, signals.columns.get_loc('position_gld')] = 1

# --- 5. 執行回測與計算績效 ---

print("Running backtest...")
qqq_returns = prices['Open'][ATTACK_ASSET].pct_change()
gld_returns = prices['Open'][DEFENSE_ASSET].pct_change()

signals['strategy_returns'] = (
    qqq_returns * signals['position_qqq'].shift(1) +
    gld_returns * signals['position_gld'].shift(1)
).fillna(0)

signals['benchmark_returns'] = qqq_returns.fillna(0)
initial_capital = 1.0
signals['strategy_equity'] = initial_capital * (1 + signals['strategy_returns']).cumprod()
signals['benchmark_equity'] = initial_capital * (1 + signals['benchmark_returns']).cumprod()

# --- 6. 定義績效指標計算函式 ---

def calculate_performance_metrics(returns_series):
    # ... (此函式與上一版完全相同，為節省篇幅省略)
    risk_free_rate = 0.0
    trading_days = 252
    if len(returns_series) == 0 or returns_series.abs().sum() == 0: return {"Total Return": 0.0, "Annualized Return": 0.0, "Annualized Volatility": 0.0, "Sharpe Ratio": 0.0, "Max Drawdown": 0.0}
    total_return = (1 + returns_series).prod() - 1
    annualized_return = (1 + total_return) ** (trading_days / len(returns_series)) - 1
    annualized_volatility = returns_series.std() * np.sqrt(trading_days)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
    equity_curve = (1 + returns_series).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    return {"Total Return": total_return, "Annualized Return": annualized_return, "Annualized Volatility": annualized_volatility, "Sharpe Ratio": sharpe_ratio, "Max Drawdown": max_drawdown}

# --- 7. 計算並印出結果 ---

strategy_metrics = calculate_performance_metrics(signals['strategy_returns'])
benchmark_metrics = calculate_performance_metrics(signals['benchmark_returns'])

print("\n--- Backtest Results ---")
print("\n[Defensive Rotation Strategy]")
for key, value in strategy_metrics.items(): print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")
print("\n[Benchmark: Buy and Hold QQQ]")
for key, value in benchmark_metrics.items(): print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")
print("-" * 30)

# --- 8. *** 全新可視化 *** ---

print("Generating enhanced performance charts...")
plt.style.use('seaborn-v0_8-darkgrid')
# 建立包含兩個子圖的圖表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# --- 上圖：權益曲線 ---
signals['strategy_equity'].plot(ax=ax1, label='防禦性輪動策略', color='green')
signals['benchmark_equity'].plot(ax=ax1, label='基準策略 (無腦持有)', color='grey', linestyle='--')
ax1.set_title('策略績效 vs. 基準', fontproperties=title_font)
ax1.set_ylabel('權益曲線 (初始資金=1)', fontproperties=my_font)

# --- 下圖：相對強度 ---
relative_strength = signals['strategy_equity'] / signals['benchmark_equity']
relative_strength.plot(ax=ax2, label='相對強度 (策略/基準)', color='purple')
ax2.axhline(1, color='black', linestyle='--', linewidth=1)
ax2.set_ylabel('相對強度', fontproperties=my_font)
ax2.set_xlabel('日期', fontproperties=my_font)

# --- 在兩張圖上都標示防禦區間 ---
if signals['position_gld'].sum() > 0:
    is_start = (signals['position_gld'].diff() == 1)
    is_end = (signals['position_gld'].diff() == -1)
    start_dates = signals.index[is_start]
    end_dates = signals.index[is_end]
    if len(start_dates) > len(end_dates): end_dates = end_dates.append(pd.Index([signals.index[-1]]))
    
    for s, e in zip(start_dates, end_dates):
        ax1.axvspan(s, e, color='red', alpha=0.15)
        ax2.axvspan(s, e, color='red', alpha=0.15)

# 整理圖例
from matplotlib.patches import Patch
handles, labels = ax1.get_legend_handles_labels()
handles.append(Patch(color='red', alpha=0.15))
labels.append('防禦模式 (持有黃金)')
legend = ax1.legend(handles=handles, labels=labels, prop=my_font)
plt.setp(legend.get_texts(), fontproperties=my_font)

ax2.legend(prop=my_font)
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.show()
