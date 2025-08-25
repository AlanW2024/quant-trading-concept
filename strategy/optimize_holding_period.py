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

# d. 參數優化範圍
HOLDING_PERIOD_RANGE = range(10, 91, 5)

# e. 設定中文字體
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

# --- 3. 將回測邏輯打包成函式 ---

def run_backtest(holding_period, price_data, hike_dates_list, vix_threshold):
    signals = pd.DataFrame(index=price_data.index)
    signals['vix'] = price_data['Close'][VIX_TICKER]
    signals['is_hike_day'] = signals.index.isin(hike_dates_list)
    signals['is_high_vix'] = signals['vix'] > vix_threshold
    signals['defense_trigger'] = (signals['is_hike_day']) & (signals['is_high_vix'])

    signals['position_qqq'] = 1
    signals['position_gld'] = 0
    defense_indices = signals[signals['defense_trigger']].index

    for trigger_date in defense_indices:
        trigger_loc = signals.index.get_loc(trigger_date)
        start_pos = trigger_loc + 1
        end_pos = start_pos + holding_period
        if end_pos < len(signals):
            signals.iloc[start_pos:end_pos, signals.columns.get_loc('position_qqq')] = 0
            signals.iloc[start_pos:end_pos, signals.columns.get_loc('position_gld')] = 1
    
    qqq_returns = price_data['Open'][ATTACK_ASSET].pct_change()
    gld_returns = price_data['Open'][DEFENSE_ASSET].pct_change()
    strategy_returns = (
        qqq_returns * signals['position_qqq'].shift(1) +
        gld_returns * signals['position_gld'].shift(1)
    ).fillna(0)

    def calculate_metrics(returns_series):
        risk_free_rate = 0.0; trading_days = 252
        if len(returns_series) == 0 or returns_series.abs().sum() == 0: return {"Sharpe Ratio": 0.0, "Max Drawdown": 0.0, "Total Return": 0.0}
        total_return = (1 + returns_series).prod() - 1
        annualized_volatility = returns_series.std() * np.sqrt(trading_days)
        annualized_return = (1 + total_return) ** (trading_days / len(returns_series)) - 1
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
        equity_curve = (1 + returns_series).cumprod()
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return {"Sharpe Ratio": sharpe_ratio, "Max Drawdown": max_drawdown, "Total Return": total_return}

    return calculate_metrics(strategy_returns)

# --- 4. 執行優化循環 ---

print("Running parameter optimization loop...")
optimization_results = []
for period in HOLDING_PERIOD_RANGE:
    print(f"Testing holding period: {period} days...")
    metrics = run_backtest(period, prices, hike_dates, VIX_THRESHOLD)
    result_row = {
        "Holding Period": period,
        "Sharpe Ratio": metrics["Sharpe Ratio"],
        "Max Drawdown": metrics["Max Drawdown"],
        "Total Return": metrics["Total Return"]
    }
    optimization_results.append(result_row)
results_df = pd.DataFrame(optimization_results)
print("Optimization complete.")

# --- 5. 輸出優化結果報告 ---

print("\n--- Optimization Results ---")
# 為了清晰，我們重新格式化 results_df
printable_df = results_df.copy()
printable_df['Max Drawdown'] = printable_df['Max Drawdown'].apply(lambda x: f"{x:.2%}")
printable_df['Total Return'] = printable_df['Total Return'].apply(lambda x: f"{x:.2%}")
printable_df['Sharpe Ratio'] = printable_df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
print(printable_df.to_string(index=False))
print("-" * 30)

# *** 修正點：使用格式化的英文輸出，避免亂碼 ***
best_sharpe_row = results_df.loc[results_df['Sharpe Ratio'].idxmax()]
print("\nParameters for Best Sharpe Ratio:")
print(f"Holding Period: {best_sharpe_row['Holding Period']:.0f} days")
print(f"Sharpe Ratio:   {best_sharpe_row['Sharpe Ratio']:.3f}")
print(f"Max Drawdown:   {best_sharpe_row['Max Drawdown']:.2%}")
print(f"Total Return:   {best_sharpe_row['Total Return']:.2%}")


# --- 6. 可視化優化曲線 ---

print("Generating optimization curve chart...")
plot_df = pd.DataFrame(optimization_results)
fig, ax1 = plt.subplots(figsize=(16, 9))
plt.style.use('seaborn-v0_8-darkgrid')

color = 'tab:blue'
ax1.set_xlabel('防禦模式持有天數 (天)', fontproperties=my_font)
ax1.set_ylabel('夏普比率 (Sharpe Ratio)', color=color, fontproperties=my_font)
ax1.plot(plot_df['Holding Period'], plot_df['Sharpe Ratio'], color=color, marker='o', label='夏普比率')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('最大回撤 (Max Drawdown)', color=color, fontproperties=my_font)
ax2.plot(plot_df['Holding Period'], plot_df['Max Drawdown'], color=color, marker='s', linestyle='--', label='最大回撤')
ax2.tick_params(axis='y', labelcolor=color)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

fig.suptitle('策略參數優化曲線：尋找最佳持有期', fontproperties=title_font)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best', prop=my_font)

plt.show()
