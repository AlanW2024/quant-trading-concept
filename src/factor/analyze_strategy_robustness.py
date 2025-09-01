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
# 專業的做法是將所有可變的參數都集中在這個區域進行管理。
# 這就像是飛機的儀表板，你可以在這裡調整策略的所有設定，而不需要修改核心的引擎程式碼。
config = {
    # --- 數據設定 ---
    "start_date": "1995-01-01",
    "end_date": "2024-12-31",
    "rebalance_freq": "M", # *** 核心修正：將遺失的關鍵參數加回來 ***
    "data": {
        "tickers": ['QQQ', 'GLD', '^VIX'], # 策略需要用到的核心資產代碼
        "cache_file_prefix": "price_cache" # 快取檔案的前綴，引擎會根據日期和股票數量自動生成完整檔名
    },
    # --- 交易成本設定 ---
    "costs": {
        "transaction_pct": 0.001,  # 0.1%，模擬每次買賣的手續費和稅
        "slippage_pct": 0.0005     # 0.05%，模擬因大額交易導致的成交價偏差
    },
    # --- 核心策略參數 (我們將在下面對這些參數進行壓力測試) ---
    "risk_management": {
        "volatility_target": 0.15, 
        "volatility_lookback": 60,
        "max_leverage": 1.5
    },
    "macro_timing": {
        "vix_enter_threshold": 25,
        "defense_asset": "GLD",
        "defense_holding_period": 50,
        "fed_hike_dates": [ # 包含了從1995年以來的主要升息日期
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
# 這是一個「類別 (Class)」，你可以把它想像成一個藍圖，用來製造「回測機器人」。
# 我們把所有複雜的邏輯都封裝在這個藍圖裡，讓之後的呼叫變得非常簡單。
class ProfessionalBacktester:
    def __init__(self, config):
        # 每個機器人被製造出來時，都會拿到一份策略設定 (config)
        self.config = config
        self.sp500_tickers = []
        self.price_data = None
        self.strategy_returns_net = None # 我們只關心淨報酬

    def _get_sp500_tickers(self):
        # 這個函式負責獲取 S&P 500 的成分股列表
        print("Fetching S&P 500 tickers...")
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)
            self.sp500_tickers = [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
        except Exception as e:
            print(f"Failed to fetch tickers: {e}. Using a small backup list.")
            self.sp500_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']

    def _get_price_data(self):
        # 這個函式負責下載或從快取讀取價格數據，是我們專業工作流程的核心。
        if self.price_data is not None: # 如果數據已經存在，就不重複載入
            return

        self._get_sp500_tickers()
        all_tickers = sorted(list(set(self.sp500_tickers + self.config['data']['tickers'])))
        
        start_str = self.config['start_date']
        end_str = self.config['end_date']
        cache_file = f"price_cache_{start_str}_to_{end_str}_{len(all_tickers)}_tickers.csv"

        if os.path.exists(cache_file):
            print(f"Loading prices from cache: {cache_file}...")
            self.price_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            missing = [t for t in all_tickers if t not in self.price_data.columns]
            if not missing:
                print("Price data loaded successfully.")
                return
            else:
                print(f"Cache is incomplete. Forcing fresh download...")
        
        print(f"Downloading prices for {len(all_tickers)} tickers...")
        downloaded_data = yf.download(all_tickers, start=start_str, end=end_str)
        self.price_data = downloaded_data['Close'].dropna(axis=1, how='all')
        print(f"Saving data to cache: {cache_file}...")
        self.price_data.to_csv(cache_file)

    def run(self):
        # 這是執行回測的主要函式
        self._get_price_data()
        
        daily_returns = self.price_data.pct_change()
        rebalance_dates = self.price_data.resample(self.config['rebalance_freq']).last().index

        hike_dates = pd.to_datetime(self.config['macro_timing']['fed_hike_dates'])
        macro_signals = pd.DataFrame(index=self.price_data.index)
        macro_signals['vix'] = self.price_data['^VIX']
        macro_signals['is_hike_day'] = macro_signals.index.isin(hike_dates)
        macro_signals['is_high_vix'] = macro_signals['vix'] > self.config['macro_timing']['vix_enter_threshold']
        macro_signals['defense_trigger'] = (macro_signals['is_hike_day']) & (macro_signals['is_high_vix'])

        positions = pd.DataFrame(0.0, index=self.price_data.index, columns=self.price_data.columns)
        current_winners = []
        in_defense_mode = False
        defense_end_date = pd.NaT

        # print("Generating daily target positions...")
        for i in range(1, len(self.price_data)):
            today = self.price_data.index[i]
            yesterday = self.price_data.index[i-1]
            
            if in_defense_mode and today > defense_end_date: in_defense_mode = False
            if yesterday in macro_signals.index and macro_signals.loc[yesterday, 'defense_trigger']:
                if not in_defense_mode:
                    in_defense_mode = True
                    period = self.config['macro_timing']['defense_holding_period']
                    end_idx = i + period
                    defense_end_date = self.price_data.index[end_idx] if end_idx < len(self.price_data) else self.price_data.index[-1]

            if yesterday in rebalance_dates:
                lookback = self.config['factor_selection']['lookback_period']
                skip = self.config['factor_selection']['skip_period']
                try:
                    start_loc = rebalance_dates.get_loc(yesterday) - lookback - skip
                    end_loc = rebalance_dates.get_loc(yesterday) - skip
                    if start_loc >= 0:
                        prices = self.price_data.loc[rebalance_dates[start_loc]:rebalance_dates[end_loc]]
                        if len(prices) > 1:
                            factor = prices.iloc[-1] / prices.iloc[0] - 1
                            valid_factor = factor.dropna()
                            valid_factor = valid_factor[valid_factor.index.isin(self.sp500_tickers)]
                            if not valid_factor.empty:
                                quantiles = pd.qcut(valid_factor, 5, labels=False, duplicates='drop')
                                current_winners = valid_factor[quantiles == 4].index.tolist()
                except (KeyError, ValueError): pass

            positions.loc[today] = 0.0
            if in_defense_mode:
                positions.loc[today, self.config['macro_timing']['defense_asset']] = 1.0
            else:
                if current_winners:
                    valid_winners = [t for t in current_winners if t in self.price_data.columns]
                    if valid_winners:
                        weight = 1.0 / len(valid_winners)
                        positions.loc[today, valid_winners] = weight
        
        # print("Simulating costs and applying volatility target...")
        turnover = positions.diff().abs().sum(axis=1)
        total_cost_rate = self.config['costs']['transaction_pct'] + self.config['costs']['slippage_pct']
        daily_costs = turnover * total_cost_rate
        
        base_strategy_returns = (positions.shift(1) * daily_returns).sum(axis=1).fillna(0)
        base_strategy_returns_net = base_strategy_returns - daily_costs

        vol_target = self.config['risk_management']['volatility_target']
        vol_lookback = self.config['risk_management']['volatility_lookback']
        max_leverage = self.config['risk_management']['max_leverage']
        
        realized_vol = base_strategy_returns.rolling(window=vol_lookback).std() * np.sqrt(252)
        leverage = vol_target / realized_vol
        leverage = leverage.fillna(1.0).clip(upper=max_leverage) 

        self.strategy_returns_net = (base_strategy_returns_net * leverage.shift(1)).fillna(0)

    def get_performance_metrics(self):
        # 這個函式只負責計算績效，並返回結果，讓主流程更乾淨。
        if self.strategy_returns_net is None:
            raise ValueError("You must run the backtest before getting metrics.")
        
        returns = self.strategy_returns_net
        if returns.abs().sum() == 0: return {"Sharpe Ratio": 0, "Max Drawdown": 0}
        
        annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        equity = (1 + returns).cumprod()
        max_dd = (equity / equity.cummax() - 1).min()
        
        return {"Sharpe Ratio": sharpe_ratio, "Max Drawdown": max_dd}

# ==============================================================================
# 敏感性分析實驗室 (The Sensitivity Analysis Lab)
# ==============================================================================
# 這是我們的主流程，它會像一個科學家一樣，控制變因，執行多次實驗。
def run_sensitivity_analysis(base_config):
    print("\n" + "="*50)
    print("Starting Strategy Robustness Check: Sensitivity Analysis")
    print("="*50)
    
    # 建立一個共用的回測引擎實例，避免重複下載數據
    engine = ProfessionalBacktester(base_config)
    engine._get_price_data() # 預先載入數據
    
    # --- 實驗一：VIX 閾值敏感性 ---
    print("\n--- Test 1: VIX Enter Threshold Sensitivity ---")
    vix_results = []
    vix_range = range(20, 31, 2) # 測試 VIX 從 20 到 30，間隔為 2
    for vix_thresh in vix_range:
        print(f"Testing VIX Threshold: {vix_thresh}...")
        test_config = base_config.copy()
        test_config['macro_timing'] = base_config['macro_timing'].copy()
        test_config['macro_timing']['vix_enter_threshold'] = vix_thresh
        
        # 傳入已有的引擎，只修改設定
        engine.config = test_config
        engine.run()
        metrics = engine.get_performance_metrics()
        metrics['vix_threshold'] = vix_thresh
        vix_results.append(metrics)
    
    # --- 實驗二：波動率目標敏感性 ---
    print("\n--- Test 2: Volatility Target Sensitivity ---")
    vol_results = []
    vol_range = np.arange(0.12, 0.19, 0.01) # 測試波動率目標從 12% 到 18%
    for vol_target in vol_range:
        print(f"Testing Volatility Target: {vol_target:.2f}...")
        test_config = base_config.copy()
        test_config['risk_management'] = base_config['risk_management'].copy()
        test_config['risk_management']['volatility_target'] = vol_target
        
        engine.config = test_config
        engine.run()
        metrics = engine.get_performance_metrics()
        metrics['vol_target'] = vol_target
        vol_results.append(metrics)

    # --- 實驗三：防禦期敏感性 ---
    print("\n--- Test 3: Defense Holding Period Sensitivity ---")
    period_results = []
    period_range = range(40, 61, 5) # 測試防禦期從 40 天到 60 天
    for period in period_range:
        print(f"Testing Holding Period: {period} days...")
        test_config = base_config.copy()
        test_config['macro_timing'] = base_config['macro_timing'].copy()
        test_config['macro_timing']['defense_holding_period'] = period
        
        engine.config = test_config
        engine.run()
        metrics = engine.get_performance_metrics()
        metrics['holding_period'] = period
        period_results.append(metrics)
        
    return pd.DataFrame(vix_results), pd.DataFrame(vol_results), pd.DataFrame(period_results)

def plot_sensitivity_curves(vix_df, vol_df, period_df):
    # 這個函式負責將我們的實驗結果視覺化
    print("\nGenerating sensitivity analysis charts...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 圖一：VIX 閾值
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(vix_df['vix_threshold'], vix_df['Sharpe Ratio'], 'o-', color='blue', label='Sharpe Ratio')
    ax1_twin.plot(vix_df['vix_threshold'], vix_df['Max Drawdown'], 's--', color='red', label='Max Drawdown')
    ax1.set_xlabel("VIX Enter Threshold")
    ax1.set_ylabel("Sharpe Ratio", color='blue')
    ax1_twin.set_ylabel("Max Drawdown", color='red')
    ax1.set_title("Sensitivity to VIX Threshold")
    
    # 圖二：波動率目標
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.plot(vol_df['vol_target'], vol_df['Sharpe Ratio'], 'o-', color='blue', label='Sharpe Ratio')
    ax2_twin.plot(vol_df['vol_target'], vol_df['Max Drawdown'], 's--', color='red', label='Max Drawdown')
    ax2.set_xlabel("Volatility Target")
    ax2.set_ylabel("Sharpe Ratio", color='blue')
    ax2_twin.set_ylabel("Max Drawdown", color='red')
    ax2.set_title("Sensitivity to Volatility Target")

    # 圖三：防禦期
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    ax3.plot(period_df['holding_period'], period_df['Sharpe Ratio'], 'o-', color='blue', label='Sharpe Ratio')
    ax3_twin.plot(period_df['holding_period'], period_df['Max Drawdown'], 's--', color='red', label='Max Drawdown')
    ax3.set_xlabel("Defense Holding Period (Days)")
    ax3.set_ylabel("Sharpe Ratio", color='blue')
    ax3_twin.set_ylabel("Max Drawdown", color='red')
    ax3.set_title("Sensitivity to Defense Holding Period")
    
    # 統一圖例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    fig.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()

# --- 主執行流程 ---
if __name__ == '__main__':
    # 執行敏感性分析
    vix_results_df, vol_results_df, period_results_df = run_sensitivity_analysis(config)
    
    # 繪製結果圖表
    plot_sensitivity_curves(vix_results_df, vol_results_df, period_results_df)

