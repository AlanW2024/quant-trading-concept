# 匯入必要的函式庫
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from matplotlib.font_manager import FontProperties

# --- 1. 定義研究參數 ---

# a. *** 新增 ^VIX 代碼 ***
# SPY: S&P 500 (大盤)
# QQQ: Nasdaq 100 (科技股)
# ^VIX: CBOE 波動率指數 (恐慌指數)
TICKERS = ['SPY', 'QQQ', '^VIX']

# b. 數據下載區間
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

# c. 聯準會升息日期
FED_HIKE_DATES = [
    '2015-12-17', '2016-12-15', '2017-03-16', '2017-06-15', '2017-12-14',
    '2018-03-22', '2018-06-14', '2018-09-27', '2018-12-20', '2022-03-17',
    '2022-05-05', '2022-06-16', '2022-07-28', '2022-09-22', '2022-11-03',
    '2022-12-15', '2023-02-02', '2023-03-23', '2023-05-04', '2023-07-27'
]
hike_dates = [datetime.strptime(date, '%Y-%m-%d') for date in FED_HIKE_DATES]

# d. 事件研究窗口大小
WINDOW_BEFORE = 30
WINDOW_AFTER = 30

# e. *** 新增 VIX 恐慌閾值定義 ***
# 我們定義 VIX 指數高於 25 為 "高恐慌" 狀態
VIX_THRESHOLD = 25

# --- 2. 下載並準備數據 ---

print("Downloading historical price data...")
try:
    # 下載所有代碼的收盤價
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)['Close']
    # 分別提取 VIX 數據和股價數據
    vix_data = data['^VIX'].dropna()
    price_data = data[['SPY', 'QQQ']]
    # 計算股價的對數報酬率
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    print("Data download and processing complete.")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# --- 3. *** 新增：根據 VIX 將事件分組 *** ---

high_vix_hikes = []
low_vix_hikes = []

print(f"Grouping hike events by VIX threshold ({VIX_THRESHOLD})...")
for date in hike_dates:
    try:
        # 找到最接近升息日的 VIX 數值
        vix_on_event_day = vix_data.asof(date)
        if vix_on_event_day > VIX_THRESHOLD:
            high_vix_hikes.append(date)
        else:
            low_vix_hikes.append(date)
    except KeyError:
        # 如果某個日期找不到對應數據，就跳過
        continue

print(f"Found {len(low_vix_hikes)} events in Low VIX regime.")
print(f"Found {len(high_vix_hikes)} events in High VIX regime.")


# --- 4. *** 重構：將核心分析邏輯打包成函式 *** ---
def perform_event_analysis(event_dates, returns_data):
    """
    對給定的事件日期列表，執行事件研究分析。
    Args:
        event_dates (list): datetime 物件的列表。
        returns_data (pd.DataFrame): 包含對數報酬率的 DataFrame。
    Returns:
        pd.DataFrame: 包含所有事件窗口分析結果的 DataFrame。
    """
    all_windows = []
    for event_date in event_dates:
        if event_date < returns_data.index.min() or event_date > returns_data.index.max():
            continue

        actual_event_index = returns_data.index.get_indexer([event_date], method='nearest')[0]
        start_index = actual_event_index - WINDOW_BEFORE
        end_index = actual_event_index + WINDOW_AFTER

        if start_index < 0 or end_index >= len(returns_data):
            continue

        event_window_returns = returns_data.iloc[start_index:end_index + 1]
        
        cumulative_returns = np.exp(event_window_returns.cumsum()) - 1
        cumulative_returns.index = range(-WINDOW_BEFORE, WINDOW_AFTER + 1)
        
        volatility = event_window_returns.std() * np.sqrt(252)
        
        event_df = cumulative_returns.stack().reset_index()
        event_df.columns = ['Day', 'Ticker', 'Cumulative_Return']
        event_df['Event_Date'] = event_date.strftime('%Y-%m-%d')
        event_df['Volatility_SPY'] = volatility['SPY']
        event_df['Volatility_QQQ'] = volatility['QQQ']
        all_windows.append(event_df)
    
    if not all_windows:
        return pd.DataFrame() # 如果沒有任何事件，返回空的 DataFrame
        
    return pd.concat(all_windows, ignore_index=True)

# --- 5. 分別對兩組事件進行分析 ---

print("Analyzing Low VIX events...")
low_vix_results = perform_event_analysis(low_vix_hikes, log_returns)

print("Analyzing High VIX events...")
high_vix_results = perform_event_analysis(high_vix_hikes, log_returns)

# --- 6. 數據匯總與可視化 ---

print("Generating charts and statistical summary...")

# 設定中文字體
try:
    font_path = 'C:/Windows/Fonts/msjh.ttc'
    my_font = FontProperties(fname=font_path, size=12)
    title_font = FontProperties(fname=font_path, size=16)
except FileNotFoundError:
    my_font = FontProperties(size=12)
    title_font = FontProperties(size=16)

# *** 重構：建立 1x2 的子圖，並排比較 ***
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True) # sharey=True 讓兩張圖的Y軸刻度一致
plt.style.use('seaborn-v0_8-darkgrid')

# 繪製左圖：低 VIX
if not low_vix_results.empty:
    sns.lineplot(data=low_vix_results, x='Day', y='Cumulative_Return', hue='Ticker', ax=axes[0], palette=['#1f77b4', '#ff7f0e'])
    axes[0].set_title(f'低 VIX (VIX <= {VIX_THRESHOLD}) 環境下的升息反應', fontproperties=title_font)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=1.5)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes[0].set_xlabel('相對於升息事件的交易日', fontproperties=my_font)
    axes[0].set_ylabel('平均累計報酬率 (%)', fontproperties=my_font)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    legend_low = axes[0].legend(title='指數', prop=my_font)
    plt.setp(legend_low.get_title(), fontproperties=my_font)

# 繪製右圖：高 VIX
if not high_vix_results.empty:
    sns.lineplot(data=high_vix_results, x='Day', y='Cumulative_Return', hue='Ticker', ax=axes[1], palette=['#1f77b4', '#ff7f0e'])
    axes[1].set_title(f'高 VIX (VIX > {VIX_THRESHOLD}) 環境下的升息反應', fontproperties=title_font)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=1.5)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes[1].set_xlabel('相對於升息事件的交易日', fontproperties=my_font)
    axes[1].set_ylabel('') # 左圖已有，無需重複
    legend_high = axes[1].legend(title='指數', prop=my_font)
    plt.setp(legend_high.get_title(), fontproperties=my_font)

plt.tight_layout()
plt.show()

# --- 7. 統計摘要 ---
def print_summary(results_df, regime_name):
    if results_df.empty:
        print(f"\nNo data for {regime_name} regime.")
        return
    avg_volatility = results_df.groupby('Event_Date')[['Volatility_SPY', 'Volatility_QQQ']].first().mean()
    print(f"\n--- Summary for {regime_name} Regime ---")
    print("Average Annualized Volatility:")
    print(f"SPY: {avg_volatility['Volatility_SPY']:.2%}")
    print(f"QQQ: {avg_volatility['Volatility_QQQ']:.2%}")
    print("-" * 30)

print_summary(low_vix_results, f"Low VIX (VIX <= {VIX_THRESHOLD})")
print_summary(high_vix_results, f"High VIX (VIX > {VIX_THRESHOLD})")
