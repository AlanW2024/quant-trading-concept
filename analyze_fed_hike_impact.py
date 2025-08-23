# 匯入必要的函式庫
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
# *** 新增匯入 ***
from matplotlib.font_manager import FontProperties

# --- 1. 定義研究參數 ---

# a. 定義分析的股票代碼
TICKERS = ['SPY', 'QQQ']

# b. 定義歷史數據的下載區間
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

# c. 定義聯準會升息日期
FED_HIKE_DATES = [
    '2015-12-17', '2016-12-15', '2017-03-16', '2017-06-15', '2017-12-14',
    '2018-03-22', '2018-06-14', '2018-09-27', '2018-12-20', '2022-03-17',
    '2022-05-05', '2022-06-16', '2022-07-28', '2022-09-22', '2022-11-03',
    '2022-12-15', '2023-02-02', '2023-03-23', '2023-05-04', '2023-07-27'
]
hike_dates = [datetime.strptime(date, '%Y-%m-%d') for date in FED_HIKE_DATES]

# d. 定義事件研究的窗口大小
WINDOW_BEFORE = 30
WINDOW_AFTER = 30

# --- 2. 下載並準備數據 ---

print("Downloading historical price data...")
try:
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)['Close']
    log_returns = np.log(data / data.shift(1)).dropna()
    print("Data download and processing complete.")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# --- 3. 執行事件研究 (Event Study) ---

all_event_windows = []

print("Performing event study analysis...")
for event_date in hike_dates:
    if event_date < log_returns.index.min() or event_date > log_returns.index.max():
        continue

    actual_event_index = log_returns.index.get_indexer([event_date], method='nearest')[0]
    start_index = actual_event_index - WINDOW_BEFORE
    end_index = actual_event_index + WINDOW_AFTER

    if start_index < 0 or end_index >= len(log_returns):
        continue

    event_window_returns = log_returns.iloc[start_index:end_index + 1]

    # --- 4. 計算指標 ---
    cumulative_returns = np.exp(event_window_returns.cumsum()) - 1
    cumulative_returns.index = range(-WINDOW_BEFORE, WINDOW_AFTER + 1)
    volatility = event_window_returns.std() * np.sqrt(252)

    event_df = cumulative_returns.stack().reset_index()
    event_df.columns = ['Day', 'Ticker', 'Cumulative_Return']
    event_df['Event_Date'] = event_date.strftime('%Y-%m-%d')
    event_df['Volatility_SPY'] = volatility['SPY']
    event_df['Volatility_QQQ'] = volatility['QQQ']
    all_event_windows.append(event_df)

results_df = pd.concat(all_event_windows, ignore_index=True)
print("Event study analysis complete.")

# --- 5. 數據匯總與可視化 ---

print("Generating chart and statistical summary...")

# *** 修正點：直接指定字體檔案路徑，這是最可靠的方法 ***
try:
    # Windows 系統內建的微軟正黑體
    font_path = 'C:/Windows/Fonts/msjh.ttc'
    my_font = FontProperties(fname=font_path, size=12)
    title_font = FontProperties(fname=font_path, size=20)
    print(f"Successfully loaded font: {font_path}")
except FileNotFoundError:
    print(f"WARNING: Font file not found at {font_path}. Chinese characters may not display correctly.")
    # 如果找不到字體，就使用預設字體，避免程式崩潰
    my_font = FontProperties(size=12)
    title_font = FontProperties(size=20)

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(14, 8))

sns.lineplot(data=results_df, x='Day', y='Cumulative_Return', hue='Ticker', ax=ax, palette=['#1f77b4', '#ff7f0e'])

# *** 修正點：在每個需要中文的地方，明確傳入字體屬性 ***
ax.set_title('升息事件前後 SPY vs. QQQ 平均累計報酬率', fontproperties=title_font, pad=20)
ax.set_xlabel('相對於升息事件的交易日', fontproperties=my_font)
ax.set_ylabel('平均累計報酬率 (%)', fontproperties=my_font)
ax.axvline(x=0, color='r', linestyle='--', linewidth=1.5, label='升息事件日 (Day 0)')
ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

# 設定圖例的字體
legend = ax.legend(title='指數', prop=my_font)
plt.setp(legend.get_title(), fontproperties=my_font)

plt.tight_layout()
plt.show()

# --- 統計摘要 (英文輸出) ---
avg_volatility = results_df.groupby('Event_Date')[['Volatility_SPY', 'Volatility_QQQ']].first().mean()

print("\n--- Statistical Summary ---")
print("Average Annualized Volatility during the event window (+/- 30 trading days):")
print(f"SPY (Broad Market): {avg_volatility['Volatility_SPY']:.2%}")
print(f"QQQ (Tech Stocks):  {avg_volatility['Volatility_QQQ']:.2%}")
print("---------------------------")

if avg_volatility['Volatility_QQQ'] > avg_volatility['Volatility_SPY']:
    print("\nConclusion: The hypothesis is supported by the data.")
    print("During Fed rate hike periods, QQQ exhibited significantly higher volatility than SPY.")
else:
    print("\nConclusion: The hypothesis is not supported by the data.")
    print("During Fed rate hike periods, QQQ did not exhibit significantly higher volatility than SPY.")
