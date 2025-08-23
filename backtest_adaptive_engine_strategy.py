# 匯入必要的函式庫
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# --- 1. 定義策略參數與環境設定 ---

# a. 資產組合
TICKERS = ['QQQ', 'GLD', 'SHY', '^VIX']
ATTACK_ASSET = 'QQQ'
DEFENSE_ASSETS = ['GLD', 'SHY']
VIX_TICKER = '^VIX'

# b. 數據下載區間
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

# c. *** 最終版交易規則 ***
FED_HIKE_DATES = [
    '2015-12-17', '2016-12-15', '2017-03-16', '2017-06-15', '2017-12-14',
    '2018-03-22', '2018-06-14', '2018-09-27', '2018-12-20', '2022-03-17',
    '2022-05-05', '2022-06-16', '2022-07-28', '2022-09-22', '2022-11-03',
    '2022-12-15', '2023-02-02', '2023-03-23', '2023-05-04', '2023-07-27'
]
hike_dates = pd.to_datetime(FED_HIKE_DATES)

# 進場規則 (不變)
VIX_ENTER_THRESHOLD = 25

# *** 終極動態出場規則 ***
VIX_EXIT_THRESHOLD = 20  # 情緒解除條件：VIX 必須低於 20
MA_WINDOW = 50         # 趨勢解除條件：QQQ 價格必須站上更穩健的 50 日均線

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
prices_open = all_data['Open']
prices_close = all_data['Close']
print("Data download complete.")

# --- 3. 計算技術指標 ---
print("Calculating technical indicators...")
qqq_ma = prices_close[ATTACK_ASSET].rolling(window=MA_WINDOW).mean()

# --- 4. 核心：狀態機回測循環 ---

print("Running state machine backtest...")
positions = pd.DataFrame(index=prices_open.index)
positions[ATTACK_ASSET] = 0.0
for asset in DEFENSE_ASSETS:
    positions[asset] = 0.0

current_state = 'ATTACK'
positions.iloc[0, positions.columns.get_loc(ATTACK_ASSET)] = 1.0

for i in range(1, len(prices_open)):
    yesterday = prices_open.index[i-1]
    today = prices_open.index[i]
    
    positions.iloc[i] = positions.iloc[i-1]

    is_hike_day = yesterday in hike_dates
    vix_value = prices_close[VIX_TICKER].get(yesterday)
    
    # --- 狀態轉換邏輯 ---
    if current_state == 'ATTACK':
        # 檢查是否觸發防禦
        if is_hike_day and vix_value is not None and vix_value > VIX_ENTER_THRESHOLD:
            current_state = 'DEFENSE'
            positions.loc[today, ATTACK_ASSET] = 0.0
            for asset in DEFENSE_ASSETS:
                positions.loc[today, asset] = 1 / len(DEFENSE_ASSETS)
    
    elif current_state == 'DEFENSE':
        # *** 終極大腦：雙重「威脅解除」確認機制 ***
        vix_yesterday = prices_close[VIX_TICKER].get(yesterday)
        qqq_close_yesterday = prices_close[ATTACK_ASSET].get(yesterday)
        qqq_ma_yesterday = qqq_ma.get(yesterday)
        
        if all(v is not None for v in [vix_yesterday, qqq_close_yesterday, qqq_ma_yesterday]):
            # 條件一：市場情緒是否已解除警報？
            is_sentiment_ok = vix_yesterday < VIX_EXIT_THRESHOLD
            # 條件二：中期下跌趨勢是否已扭轉？
            is_trend_ok = qqq_close_yesterday > qqq_ma_yesterday
            
            # 只有兩個威脅「同時」解除，才回歸進攻
            if is_sentiment_ok and is_trend_ok:
                current_state = 'ATTACK'
                positions.loc[today, ATTACK_ASSET] = 1.0
                for asset in DEFENSE_ASSETS:
                    positions.loc[today, asset] = 0.0

# --- 5. 計算績效 ---

print("Calculating performance...")
asset_returns = prices_open.pct_change()
strategy_returns = (positions.shift(1) * asset_returns).sum(axis=1).fillna(0)
benchmark_returns = asset_returns[ATTACK_ASSET].fillna(0)

initial_capital = 1.0
strategy_equity = initial_capital * (1 + strategy_returns).cumprod()
benchmark_equity = initial_capital * (1 + benchmark_returns).cumprod()

# --- 6. 績效指標計算與輸出 ---

def calculate_performance_metrics(returns_series):
    risk_free_rate = 0.0; trading_days = 252
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

strategy_metrics = calculate_performance_metrics(strategy_returns)
benchmark_metrics = calculate_performance_metrics(benchmark_returns)

print("\n--- Backtest Results ---")
print("\n[Adaptive Engine Strategy]")
for key, value in strategy_metrics.items(): print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")
print("\n[Benchmark: Buy and Hold QQQ]")
for key, value in benchmark_metrics.items(): print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")
print("-" * 30)

# --- 7. 可視化 ---

print("Generating enhanced performance charts...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
plt.style.use('seaborn-v0_8-darkgrid')

# 上圖：權益曲線
strategy_equity.plot(ax=ax1, label='動態適應策略', color='darkorange')
benchmark_equity.plot(ax=ax1, label='基準策略 (無腦持有)', color='grey', linestyle='--')
ax1.set_title('策略績效 vs. 基準', fontproperties=title_font)
ax1.set_ylabel('權益曲線 (初始資金=1)', fontproperties=my_font)

# 下圖：相對強度
relative_strength = strategy_equity / benchmark_equity
relative_strength.plot(ax=ax2, label='相對強度 (策略/基準)', color='purple')
ax2.axhline(1, color='black', linestyle='--', linewidth=1)
ax2.set_ylabel('相對強度', fontproperties=my_font)
ax2.set_xlabel('日期', fontproperties=my_font)

# 標示防禦區間
is_defense = positions[DEFENSE_ASSETS[0]] > 0
if is_defense.sum() > 0:
    is_start = (is_defense.diff() > 0)
    is_end = (is_defense.diff() < 0)
    start_dates = positions.index[is_start]
    end_dates = positions.index[is_end]
    if len(start_dates) > len(end_dates): end_dates = end_dates.append(pd.Index([positions.index[-1]]))
    
    for s, e in zip(start_dates, end_dates):
        ax1.axvspan(s, e, color='orange', alpha=0.2)
        ax2.axvspan(s, e, color='orange', alpha=0.2)

from matplotlib.patches import Patch
handles, labels = ax1.get_legend_handles_labels()
handles.append(Patch(color='orange', alpha=0.2))
labels.append('防禦模式 (持有GLD+SHY)')
legend = ax1.legend(handles=handles, labels=labels, prop=my_font)
plt.setp(legend.get_texts(), fontproperties=my_font)
ax2.legend(prop=my_font)

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.show()
