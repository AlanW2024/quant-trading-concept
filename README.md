# 量化交易策略的進化、失敗與頓悟 (2015-2024)

## 專案總結 (STAR Principle)

- **情境 (Situation):**
  在觀察到 2022 年後「高通膨、高利率」的宏觀典範轉移對傳統投資組合造成巨大衝擊後，我啟動了這個專案，旨在驗證一個核心問題：能否將專家的宏觀洞見（聯準會升息、市場恐慌情緒）轉化為一個系統化的交易策略，以應對並戰勝這個充滿不確定性的新市場環境。

- **任務 (Task):**
  我的目標是設計、回測並迭代一個量化輪動策略，使其在過去十年的完整市場週期中，不僅在總報酬率上，更要在**風險調整後的回報（夏普比率）與最大回撤**這兩個關鍵指標上，超越無腦持有科技股 (QQQ) 的基準策略。

- **行動 (Action):**
  我運用 Python（Pandas, yfinance, Matplotlib）執行了一個**史詩級的 10 層策略進化**。從最初的事件研究（Level 1-2）驗證假說，到建立多個回測引擎，系統性地測試了：被動避險（Level 3）、以黃金作為主動防禦資產（Level 4）、多種動態出場機制（Level 6-8），並最終對表現最佳的策略進行了參數優化與穩健性測試（Level 9）。此外，我還開發了**多因子投資引擎**，結合動量與質量價值因子，進行專業級的組合建構與波動率目標管理。

- **成果 (Result):**
  我成功開發出一個**穩健的防禦性輪動策略 (Level 4/9)**，在回測中展現了**卓越的風險管理能力**：相較於基準策略，它的**夏普比率更高 (0.94 vs 0.85)**，**最大回撤更小 (-32.54% vs -36.69%)**，同時總回報略微勝出。多因子策略也顯示出強勁的風險調整後收益。更重要的成果是，專案最終的失效分析（Level 10）揭示了**宏觀典範轉移**對所有策略的致命影響，證明了**「沒有策略能永恆」**這一核心交易哲學，並為未來策略的開發奠定了堅實的基礎。

---

## 研究日誌：一個策略的 10 個進化層級

我們的研究遵循了一條嚴謹的、從觀察到回測的科學路徑。每一個層級的策略，都是對前一個層級失敗的修正與進化。

### Level 1: 現象觀察 (Observation)

- **假說：** 聯準會升息，是否會對科技股 (QQQ) 造成比大盤 (SPY) 更大的衝擊？
- **方法：** 執行事件研究，分析升息前後 30 天的平均報酬率與波動率。
- **腳本：** `strategy/analyze_fed_hike_impact.py`
- **結論：** 假說成立。數據證實，升息期間 QQQ 的波動性顯著高於 SPY。

### Level 2: 引入背景 (Context)

- **假說：** 市場的背景情緒，是否會改變市場對升息的反應？
- **方法：** 引入 VIX 指數，將升息事件分為「高恐慌 (VIX > 25)」與「低恐慌」兩組，進行對比分析。
- **腳本：** `strategy/analyze_fed_hike_by_vix.py`
- **結論：** 假說成立。升息本身不是利空，而是市場情緒的放大器。在低 VIX 時市場反應平淡，在高 VIX 時則引發劇烈拋售。

### Level 3: 首次回測 (Passive Hedging)

- **假說：** 一個懂得在「高 VIX + 升息」的危險時刻退回現金的策略，能否超越無腦持有？
- **方法：** 建立第一個回測引擎，策略在觸發風險時清倉，持有現金 30 天。
- **腳本：** `strategy/backtest_vix_timing_strategy.py`
- **結論：** **策略慘敗。** 雖然成功避開了風險，但因在場外時間過長，完美錯過了過去十年的牛市。**教訓：機會成本是真實的。**

### Level 4: 主動防禦 (Defensive Rotation)

- **假說：** 與其退回現金，不如在危險時刻將資金從 QQQ **輪動到**傳統避險資產（黃金 GLD），能否改善績效？
- **方法：** 修改回測引擎，在觸發風險時賣出 QQQ，買入 GLD 並持有 30 天。
- **腳本：** `strategy/backtest_defensive_rotation_strategy.py`
- **結論：** **首次成功！** 策略在總回報上略微超越基準，同時顯著降低了波動率與最大回撤，獲得了更高的夏普比率。**我們發現了真正的 Alpha。**

### Level 5-8: 對出場機制的痛苦探索 (The Exit Problem)

- **假說：** 我們能否設計出一個比「固定持有 30 天」更聰明的出場機制？
- **方法：** 分別測試了基於 VIX 指數、移動平均線、以及兩者結合的「雙重確認」動態出場機制。
- **腳本：** `strategy/backtest_dynamic_exit_strategy.py`, `strategy/backtest_momentum_exit_strategy.py`, `strategy/backtest_confirmation_exit_strategy.py`
- **結論：** **全部失敗。** 所有試圖變得「更聰明」的複雜規則，都因為反應過慢或過於敏感，導致了災難性的「兩面挨耳光 (Whipsaw)」，績效遠遜於 Level 4 的簡單規則。**教訓：過度優化是 Alpha 的殺手。**

### Level 9: 參數優化 (Optimization)

- **假說：** Level 4 策略的成功，是否只是因為幸運地猜對了「極致化」？它的 Alpha 是否穩健？
- **方法：** 對 Level 4 策略的參數進行系統性的掃描測試。
- **腳本：** `strategy/optimize_holding_period.py`
- **結論：** **策略極度穩健。** 績效並非一個尖銳的山峰，而是一個表現優異的「平坦高原」。數據證明 Alpha 是真實存在的。

### Level 10 & 畢業考: 典範轉移的診斷 (The Regime Shift)

- **假說：** 我們的最佳策略為何在 2022 年之後的歷史回測中表現掙扎？是否因為市場的宏觀典範發生了轉移？
- **方法：** 將數據以 2022 年為切點，進行樣本內與樣本外測試，並最終測試了「現金為王」的終極避險策略。
- **腳本：** `strategy/diagnose_regime_shift.py`, `strategy/backtest_cash_is_king_strategy.py`
- **結論：** **典範轉移被證實。** 策略在 2022 年前的「低通膨、低利率」時代表現優異，但在 2022 年後的「高通膨、高利率」時代，其核心避險資產（黃金、債券）的避險屬性暫時失效，導致策略失靈。

---

## 多因子投資引擎 (Multi-Factor Investing Engine)

在宏觀策略的基礎上，我進一步開發了專業級的多因子投資引擎，專注於系統化的因子投資方法。

### 核心特性

- **雙因子組合：** 動量因子 (12M-1M) + 質量價值因子 (Quality + Value)
- **波動率目標管理：** 動態調整槓桿以維持目標年化波動率 (10%)
- **月度再平衡：** 定期調整投資組合權重
- **專業績效報告：** 完整的風險調整後收益分析

### 主要腳本

- `factor/professional_multifactor_engine.py` - 專業多因子回測引擎 (自動化數據管道)
- `factor/backtest_multifactor_portfolio.py` - 離線多因子回測 (支援命令行參數)
- `factor/backtest_momentum_factor.py` - 純動量因子測試
- `factor/backtest_value_factor.py` - 純價值因子測試
- `factor/backtest_vol_targeted_factor_strategy.py` - 波動率目標因子策略
- `factor/analyze_performance_attribution.py` - 績效歸因分析
- `factor/analyze_strategy_robustness.py` - 策略穩健性測試
- `factor/analyze_scenario_survival.py` - 場景生存分析

### 數據緩存

專案包含預下載的數據緩存文件，加速回測過程：

- `price_cache_*.csv` - S&P 500 成分股價格數據
- `fundamentals_cache_*.json` - 基本面數據 (PB, ROE, 毛利率等)

---

## 最終頓悟 (The Ultimate Conclusion)

這個專案最珍貴的產出，不是一個能賺錢的策略，而是下面這條血的教訓：

**所有基於歷史數據的 Alpha，都隱含著對「宏觀典範 (Macro Regime)」持續不變的假設。當典範轉移時，昨日的聖杯，可能就是明日的毒藥。**

一個成功的量化交易者，其核心能力不是找到一個完美的策略，而是：

1.  深刻理解自己策略的**適用邊界**與**內在弱點**。
2.  建立一套能**監測宏觀典範**是否正在轉移的系統。
3.  擁有在必要時，**果斷拋棄舊策略**的紀律與勇氣。

---

## 如何使用這個儲存庫

### 安裝依賴

```bash
pip install yfinance pandas numpy matplotlib seaborn argparse
```

### 執行策略回測 (按進化層級)

1.  按照 `Level 1` 到 `Level 10` 的順序，依次執行 `strategy/*.py` 腳本。
2.  仔細閱讀每個腳本中的註解，並對比回測結果，親身體驗策略的進化與失敗。

### 執行多因子回測

```bash
# 執行專業多因子引擎
python factor/professional_multifactor_engine.py

# 或使用命令行版本的離線回測
python factor/backtest_multifactor_portfolio.py --prices "price_cache_1995-01-01_to_2024-12-31_504_tickers.csv" --fundamentals "fundamentals_cache_503_tickers.json" --out "output"
```

### 數據緩存

- 專案已包含預下載的數據緩存，可直接使用
- 若要更新數據，刪除緩存文件後重新運行腳本即可自動下載

---

## 免責聲明

本儲存庫所有內容僅供學術研究與教育目的使用，不構成任何形式的投資建議。所有回測結果均基於歷史數據，歷史績效不代表未來回報。金融市場充滿風險，請自行承擔交易決策的後果。
