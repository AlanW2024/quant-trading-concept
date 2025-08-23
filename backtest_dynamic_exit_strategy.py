# 匯入必要的函式庫
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# --- 1. 定義策略參數與環境設定 ---

# a. *** 升級資產組合 ***
TICKERS = ['QQQ', 'GLD', 'SHY', '^VIX']
ATTACK_ASSET = 'QQQ'
DEFENSE_ASSETS = ['GLD', 'SHY'] # 防禦組合
VIX_TICKER = '^VIX'

# b. 數據下載區間
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

# c. *** 升級交易規則參數 ***
FED_HIKE_DATES = [
    '2015-12-17', '2016-12-15', '2017-03-16', '2017-06-15', '2017-12-14',
    '2018-03-22', '2018-06-14', '2018-09-27', '2018-12-20', '2022-03-17',
    '2022-05-05', '2022-06-16', '2022-07-28', '2022-09-22', '2022-11-03',
    '2022-12-15', '2023-02-02', '2023-03-23', '2023-05-04', '2023-07-27'
]
hike_dates = pd.to_datetime(FED_HIKE_DATES)

# 進場規則
VIX_ENTER_THRESHOLD = 25 # VIX > 25 觸發防禦

# 動態出場規則
VIX_EXIT_THRESHOLD = 20  # VIX < 20 視為安全
VIX_EXIT_DAYS = 3        # VIX 需要連續 3 天低於安全閾值才解除防禦

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
prices = all_data['Open'] # 我們只需要開盤價來計算報酬
vix_close = all_data['Close'][VIX_TICKER]
print("Data download complete.")

# --- 3. *** 核心升級：狀態機回測循環 (State Machine Backtest Loop) *** ---

print("Running state machine backtest...")
# 建立一個 DataFrame 來記錄每日的部位狀態
positions = pd.DataFrame(index=prices.index)
positions[ATTACK_ASSET] = 0
for asset in DEFENSE_ASSETS:
    positions[asset] = 0

# 狀態變數: 'ATTACK' 或 'DEFENSE'
current_state = 'ATTACK'
# 初始化：第一天開始就持有攻擊型資產
positions.iloc[0, positions.columns.get_loc(ATTACK_ASSET)] = 1

# 這個循環是整個策略的大腦。它會一天一天地遍歷歷史數據。
# 這種寫法比之前的 "向量化" 寫法更慢，但對於處理複雜的狀態轉換邏輯來說，更清晰、更可靠。
for i in range(1, len(prices)):
    # 獲取昨天和今天的日期
    yesterday = prices.index[i-1]
    today = prices.index[i]
    
    # 預設繼承昨天的部位
    positions.iloc[i] = positions.iloc[i-1]

    # 檢查是否為升息日
    is_hike_day = yesterday in hike_dates
    # 獲取昨天的 VIX 收盤價
    vix_value = vix_close.get(yesterday)

    # --- 狀態轉換邏輯 ---
    if current_state == 'ATTACK':
        # 檢查是否觸發防禦
        if is_hike_day and vix_value is not None and vix_value > VIX_ENTER_THRESHOLD:
            # 觸發！從今天開始，切換到防禦模式
            current_state = 'DEFENSE'
            positions.loc[today, ATTACK_ASSET] = 0
            # 將資金平均分配到防禦資產上
            for asset in DEFENSE_ASSETS:
                positions.loc[today, asset] = 1 / len(DEFENSE_ASSETS)
    
    elif current_state == 'DEFENSE':
        # 檢查是否解除防禦
        # 條件：VIX 需要連續 VIX_EXIT_DAYS 天都低於 VIX_EXIT_THRESHOLD
        if i >= VIX_EXIT_DAYS:
            # 截取過去幾天的 VIX 數據
            recent_vix = vix_close.iloc[i - VIX_EXIT_DAYS : i]
            if (recent_vix < VIX_EXIT_THRESHOLD).all():
                # 解除！從今天開始，切換回攻擊模式
                current_state = 'ATTACK'
                positions.loc[today, ATTACK_ASSET] = 1
                for asset in DEFENSE_ASSETS:
                    positions.loc[today, asset] = 0

# --- 4. 計算績效 ---

print("Calculating performance...")
# 計算各資產的每日報酬率
asset_returns = prices.pct_change()

# 計算策略的每日報酬率
# 報酬率 = 各資產的部位權重 * 各資產的報酬率，然後加總
# .shift(1) 是關鍵，確保我們用昨天的部位來計算今天的報酬
strategy_returns = (positions.shift(1) * asset_returns).sum(axis=1).fillna(0)

# 計算基準策略 (無腦持有QQQ)
benchmark_returns = asset_returns[ATTACK_ASSET].fillna(0)

# 計算權益曲線
initial_capital = 1.0
strategy_equity = initial_capital * (1 + strategy_returns).cumprod()
benchmark_equity = initial_capital * (1 + benchmark_returns).cumprod()

# --- 5. 績效指標計算與輸出 (與上一版相同) ---

def calculate_performance_metrics(returns_series):
    # ... (此函式與上一版完全相同，為節省篇幅省略)
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
print("\n[Dynamic Exit Strategy]")
for key, value in strategy_metrics.items(): print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")
print("\n[Benchmark: Buy and Hold QQQ]")
for key, value in benchmark_metrics.items(): print(f"{key}: {value:.2%}" if key != "Sharpe Ratio" else f"{key}: {value:.2f}")
print("-" * 30)

# --- 6. 可視化 (與上一版類似，但標示邏輯更簡單) ---

print("Generating enhanced performance charts...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
plt.style.use('seaborn-v0_8-darkgrid')

# 上圖：權益曲線
strategy_equity.plot(ax=ax1, label='動態出場策略', color='red')
benchmark_equity.plot(ax=ax1, label='基準策略 (無腦持有)', color='grey', linestyle='--')
ax1.set_title('策略績效 vs. 基準', fontproperties=title_font)
ax1.set_ylabel('權益曲線 (初始資金=1)', fontproperties=my_font)

# 下圖：相對強度
relative_strength = strategy_equity / benchmark_equity
relative_strength.plot(ax=ax2, label='相對強度 (策略/基準)', color='purple')
ax2.axhline(1, color='black', linestyle='--', linewidth=1)
ax2.set_ylabel('相對強度', fontproperties=my_font)
ax2.set_xlabel('日期', fontproperties=my_font)

# 在兩張圖上都標示防禦區間
# 防禦區間 = 持有 GLD 的部位 > 0
defense_periods = positions[positions[DEFENSE_ASSETS[0]] > 0]
if not defense_periods.empty:
    is_start = (positions[DEFENSE_ASSETS[0]].diff() > 0)
    is_end = (positions[DEFENSE_ASSETS[0]].diff() < 0)
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
