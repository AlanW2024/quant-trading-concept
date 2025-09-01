# 匯入必要的函式庫
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import warnings
import os

# 忽略 pandas 在未來版本中的一些警告，讓輸出更乾淨
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. 策略參數與環境設定 ---

# a. 股票池與回測區間
UNIVERSE = 'SP500' 
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'
REBALANCE_FREQ = 'M' 

# b. 動量因子定義
LOOKBACK_PERIOD = 12 
SKIP_PERIOD = 1      

# c. 分層回測參數
NUM_QUANTILES = 5 

# d. *** 新增：數據快取檔案名稱 ***
CACHE_FILE = 'sp500_prices_cache.csv'

# e. 設定中文字體
try:
    font_path = 'C:/Windows/Fonts/msjh.ttc'
    my_font = FontProperties(fname=font_path, size=12)
    title_font = FontProperties(fname=font_path, size=18)
except FileNotFoundError:
    my_font = FontProperties(size=12)
    title_font = FontProperties(size=18)

# --- 2. 數據獲取 (升級版，帶快取功能) ---

def get_sp500_tickers():
    """從維基百科抓取 S&P 500 的成分股列表"""
    print("Fetching S&P 500 tickers from Wikipedia...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)
        tickers = table[0]['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"Successfully fetched {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        print(f"Failed to fetch tickers: {e}")
        return ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']

def download_prices(tickers, start, end, cache_file):
    """
    下載指定股票列表的每日收盤價。
    如果快取檔案存在，則從快取讀取。
    """
    if os.path.exists(cache_file):
        print(f"Loading prices from cache file: {cache_file}...")
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print("Price data loaded from cache.")
        return prices
    else:
        print(f"Downloading historical prices for {len(tickers)} tickers...")
        prices = yf.download(tickers, start=start, end=end)['Close']
        print("Price data download complete.")
        print(f"Saving data to cache file: {cache_file}...")
        prices.to_csv(cache_file)
        return prices

# --- 3. 因子分層回測引擎 ---

def run_factor_quintile_backtest(prices, lookback, skip):
    """執行因子分層回測的核心邏輯"""
    print("Calculating monthly returns...")
    monthly_returns = prices.resample(REBALANCE_FREQ).last().pct_change()
    portfolio_returns = pd.DataFrame()
    rebalance_dates = monthly_returns.index

    print("Starting monthly rebalancing backtest...")
    # *** 修正點：修正迴圈的起始點，確保有足夠的歷史數據 ***
    for i in range(lookback + skip, len(rebalance_dates)):
        current_date = rebalance_dates[i]
        
        # --- a. 計算因子值 ---
        price_lookback_start = rebalance_dates[i - lookback - skip]
        price_lookback_end = rebalance_dates[i - skip]
        
        lookback_prices = prices.loc[price_lookback_start:price_lookback_end]
        
        # 確保 lookback_prices 不是空的
        if lookback_prices.empty:
            continue
            
        momentum_factor = lookback_prices.iloc[-1] / lookback_prices.iloc[0] - 1
        momentum_factor = momentum_factor.dropna()
        
        if momentum_factor.empty:
            continue

        # --- b. 排序與分組 ---
        labels = [f'Q{j+1}' for j in range(NUM_QUANTILES)]
        try:
            factor_quantiles = pd.qcut(momentum_factor, NUM_QUANTILES, labels=labels, duplicates='drop')
        except ValueError:
            continue
        
        # --- c. 計算下個月各投資組合的報酬 ---
        next_month_returns = monthly_returns.loc[current_date]
        for q in factor_quantiles.unique():
            stocks_in_quantile = factor_quantiles[factor_quantiles == q].index
            portfolio_returns.loc[current_date, q] = next_month_returns[stocks_in_quantile].mean()

    # --- d. 計算多空組合 ---
    if 'Q1' in portfolio_returns.columns and f'Q{NUM_QUANTILES}' in portfolio_returns.columns:
        portfolio_returns['Long-Short'] = portfolio_returns['Q1'] - portfolio_returns[f'Q{NUM_QUANTILES}']
    else:
        portfolio_returns['Long-Short'] = 0
    
    print("Backtest complete.")
    return portfolio_returns.fillna(0)

# --- 4. 主執行流程 ---

sp500_tickers = get_sp500_tickers()
price_data = download_prices(sp500_tickers, START_DATE, END_DATE, CACHE_FILE)
factor_returns = run_factor_quintile_backtest(price_data, LOOKBACK_PERIOD, SKIP_PERIOD)

# --- 5. 績效分析與可視化 ---

equity_curves = (1 + factor_returns).cumprod()

def calculate_metrics(returns):
    if returns.abs().sum() == 0: return {"Annualized Return": 0, "Annualized Volatility": 0, "Sharpe Ratio": 0}
    annualized_return = (1 + returns).prod() ** (12 / len(returns)) - 1
    annualized_volatility = returns.std() * np.sqrt(12)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    return {"Annualized Return": annualized_return, "Annualized Volatility": annualized_volatility, "Sharpe Ratio": sharpe_ratio}

long_short_metrics = calculate_metrics(factor_returns['Long-Short'])

print("\n--- Factor Performance (Long Winners / Short Losers) ---")
for key, value in long_short_metrics.items():
    print(f"{key}: {value:.2%}" if "Return" in key or "Volatility" in key else f"{key}: {value:.2f}")
print("-" * 50)

print("Generating performance chart...")
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(16, 9))

# 繪製 Q1 (贏家) 和 Q5 (輸家)
if 'Q1' in equity_curves.columns:
    ax.plot(equity_curves.index, equity_curves['Q1'], label='Q1 (Winners)')
if f'Q{NUM_QUANTILES}' in equity_curves.columns:
    ax.plot(equity_curves.index, equity_curves[f'Q{NUM_QUANTILES}'], label=f'Q{NUM_QUANTILES} (Losers)')

# 繪製多空組合
ax.plot(equity_curves.index, equity_curves['Long-Short'], label=f'Long-Short (Q1 - Q{NUM_QUANTILES})', color='black', linewidth=3, linestyle='--')

ax.set_title('Momentum Factor Quintile Backtest (S&P 500)', fontproperties=title_font)
ax.set_xlabel('Date', fontproperties=my_font)
ax.set_ylabel('Equity Curve (Log Scale, Initial Capital=1)', fontproperties=my_font)
ax.set_yscale('log')
legend = ax.legend(prop=my_font)
plt.setp(legend.get_texts(), fontproperties=my_font)
plt.show()
