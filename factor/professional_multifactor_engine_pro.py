# -*- coding: utf-8 -*-
"""
機構味多因子回測（預設 Momentum-only；避免未來資訊）
重點：月末交易日調倉、T+1、不偷看、波動率目標、交易成本、No-trade band、風險表、
      歸因、敏感度、（選）walk-forward、run_meta 紀錄。全程 ASCII log。
"""

import argparse                      # 解析命令列參數
import json                          # 寫 run_meta / 各種 JSON 檔
import math                          # sqrt(252) 等數學運算
from dataclasses import dataclass    # 設定用 dataclass
from pathlib import Path             # 跨平台路徑處理
from typing import Dict, List, Tuple, Optional  # 型別標註，幫閱讀/IDE

import numpy as np                   # 數值運算
import pandas as pd                  # 時間序/表格運算
import matplotlib.pyplot as plt      # 繪圖輸出
from datetime import datetime        # 產生 UTC 時戳寫入 meta


# ------------------------
# 工具函式（只輸出 ASCII，避免亂碼）
# ------------------------
def log(msg: str):
    print(msg, flush=True)           # 立即 flush，避免訊息卡住


def guess_existing_file(preferred: str, patterns: List[str], roots: List[Path]) -> Optional[str]:
    """找不到使用者指定檔時：到常見資料夾用萬用字元抓『體積最大』的候選檔"""
    if preferred:
        p = Path(preferred)          # 先看使用者指定路徑有沒有
        if p.exists():
            return str(p)
    cands: List[Path] = []
    for r in roots:                  # 到多個根資料夾下去找
        for pat in patterns:         # 用多組 pattern（萬用字元）嘗試匹配
            cands.extend(r.glob(pat))
    cands = [c for c in cands if c.is_file()]
    if not cands:
        return None
    best = max(cands, key=lambda x: x.stat().st_size)  # 體積最大視為最完整
    log(f"[WARN] Preferred path not found. Using discovered file: {best}")
    return str(best)


def load_prices(price_csv: str, start_date: str, end_date: str) -> pd.DataFrame:
    """讀價格 CSV → Date 當索引 → 依時間切片 → 刪空欄"""
    log(f"Loading prices from: {price_csv}")
    df = pd.read_csv(price_csv)                                  # 讀進記憶體
    date_cols = [c for c in df.columns if c.lower() in ("date", "timestamp")]  # 找日期欄
    if not date_cols:
        date_cols = [df.columns[0]]                              # 沒標準名，假設第一欄是日期
    df.rename(columns={date_cols[0]: "Date"}, inplace=True)      # 統一欄名
    df["Date"] = pd.to_datetime(df["Date"])                      # 轉 Timestamp
    df.set_index("Date", inplace=True)                           # 設為索引
    df = df.sort_index()                                         # 按日期排序
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]          # 起始切片
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]            # 結束切片
    df = df.dropna(axis=1, how="all")                            # 整欄皆 NaN 的股票刪掉
    log(f"[INFO] Price data shape: {df.shape} ({df.index.min().date()} ~ {df.index.max().date()})")
    return df


def load_fundamentals(fund_json: Optional[str]) -> Dict[str, dict]:
    """讀靜態基本面快取（非 point-in-time，預設不用以免偷看未來）"""
    if not fund_json or not Path(fund_json).exists():            # 沒提供或找不到就跳過
        log("[INFO] Fundamentals file not provided or not found. Skipping fundamentals.")
        return {}
    with open(fund_json, "r") as f:
        data = json.load(f)                                      # 讀 JSON 字典
    data = {k: (v if isinstance(v, dict) else {}) for k, v in data.items()}  # 保證每檔是 dict
    log(f"[INFO] Fundamentals loaded for {len(data)} tickers (WARNING: static cache not point-in-time).")
    return data


def align_to_trading_pos(trading_index: pd.DatetimeIndex, date_like) -> Optional[int]:
    """把任何日曆日對齊到『<=該日』最近的交易日位置（避免 1999-01-31 星期日的 KeyError）"""
    try:
        return trading_index.get_loc(pd.Timestamp(date_like))     # 完全對得上就回傳位置
    except KeyError:
        pos = trading_index.get_indexer([pd.Timestamp(date_like)], method="ffill")[0]  # 貼齊到之前最近交易日
        if pos == -1:
            return None
        return pos


def month_end_trading_days(trading_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """取得每月最後一個『交易日』：用交易日索引 resample('ME') 取每組最大值"""
    tmp = pd.Series(trading_index, index=trading_index).resample("ME").max().dropna()
    return pd.DatetimeIndex(tmp.values)


# ---- 評估指標（年化報酬/波動/Sharpe/Sortino/回撤 等） ----
def annualize_return(daily_returns: pd.Series) -> float:
    return (1 + daily_returns.mean()) ** 252 - 1                  # 252 交易日年化


def annualized_vol(daily_returns: pd.Series) -> float:
    return daily_returns.std(ddof=0) * math.sqrt(252)             # 日標準差 × sqrt(252)


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    s = daily_returns.std(ddof=0)
    if s < 1e-12:
        return 0.0                                                # 避免 0 除
    excess = daily_returns - rf / 252.0                           # 近似扣無風險日利
    return (excess.mean() / (excess.std(ddof=0) + 1e-12)) * math.sqrt(252)


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    excess = daily_returns - rf / 252.0                           # 超額日報酬
    downside = excess.copy(); downside[downside > 0] = 0.0        # 只保留下跌部分
    dd = np.sqrt((downside**2).mean()) * np.sqrt(252)             # 下行波動年化
    if dd < 1e-12:
        return 0.0
    return (excess.mean() * 252) / dd


def max_drawdown(equity_curve: pd.Series) -> float:
    return (equity_curve / equity_curve.cummax() - 1).min()       # 相對歷史高點的最深跌幅


def max_recovery_days(equity_curve: pd.Series) -> int:
    highs = equity_curve.cummax()                                  # 過去高點
    dd = equity_curve / highs                                      # 當前淨值 / 高點
    cur = 0; worst = 0
    for v in dd.values:
        if v < 1.0:                                               # 低於高點，持續計數
            cur += 1; worst = max(worst, cur)
        else:
            cur = 0                                               # 回到新高，歸零
    return int(worst)


def hit_ratio(daily_returns: pd.Series) -> float:
    n = (daily_returns > 0).sum()                                  # 正報酬天數
    d = (daily_returns != 0).sum()                                 # 有效天數
    return float(n) / float(d) if d > 0 else 0.0


# ------------------------
# 設定（所有可調參數集中這裡）
# ------------------------
@dataclass
class EngineConfig:
    # 路徑
    price_cache_path: str
    fundamentals_cache_path: Optional[str]
    out_dir: str

    # 市場 & 時間
    benchmark: str = "QQQ"                                         # 基準（用來畫線 & 風險表）
    start_date: str = "1999-01-01"
    end_date: str = "2024-12-31"
    min_history_days: int = 252                                    # 最少歷史，避免新股亂跳
    ipo_seasoning_days: int = 60                                   # 上市冷卻天數

    # 因子（預設只開 Momentum，避免 QV 污染）
    enable_qv: bool = False
    factor_weights: Tuple[float, float] = (1.0, 0.0)               # (Momentum, QV) 權重

    # 動量參數
    top_n: int = 50                                                # 每月選前 N 檔
    mom_lookback_days: int = 252                                   # 回看期間（約 12M）
    mom_skip_days: int = 20                                        # 略過最近 1M（12M-1M）

    # 組合控制
    max_weight: float = 0.05                                       # 單檔上限 5%
    target_vol: float = 0.10                                       # 年化波動目標 10%
    vol_window: int = 20                                           # 用近 20 日估波動
    leverage_cap: float = 3.0                                      # 槓桿上限（波控不超過 3x）
    rebalance_band: float = 0.005                                  # No-trade band（單檔變動<=0.5%不交易）

    # 成本
    tc_bps: float = 15.0                                           # 單邊成本 bps

    # （選）流動性過濾
    adv_csv: Optional[str] = None
    min_adv_usd: Optional[float] = None

    # Walk-forward（選）
    walk_forward: bool = False
    wf_split: Optional[str] = None


# ------------------------
# 因子實作：Momentum &（可選）QV
# ------------------------
def momentum_picks(prices: pd.DataFrame, universe: List[str], date: pd.Timestamp,
                   lookback: int, skip: int, top_n: int) -> List[str]:
    """12M-1M 動量：先把 date 貼到交易日，再計期間報酬，挑前 N 名"""
    pos = align_to_trading_pos(prices.index, date)                 # 對齊到交易日位置
    if pos is None:
        return []
    start_idx = pos - lookback - skip                              # 視窗起點（含 skip）
    end_idx = pos - skip                                           # 視窗終點
    if start_idx < 0 or end_idx <= start_idx:
        return []
    window = prices.iloc[start_idx:end_idx + 1]                    # 取視窗價
    if len(window) < 2:
        return []
    perf = (window.iloc[-1] / window.iloc[0] - 1.0).dropna()       # 視窗頭尾報酬
    perf = perf[perf.index.isin(universe)]                         # 只在可投資清單內排名
    if perf.empty:
        return []
    if top_n and top_n > 0:
        return perf.sort_values(ascending=False).head(top_n).index.tolist()  # 取前 N
    # （備用）用分位數挑最上層
    ranks = perf.rank(method="first") / len(perf)
    qcut = pd.qcut(ranks, 5, labels=False, duplicates="drop")
    return perf[qcut == 4].index.tolist()


def qvalue_picks(fund_map: Dict[str, dict], universe: List[str], top_n: int) -> List[str]:
    """示範用靜態 Q+V 排名（非 point-in-time，預設不建議啟用）"""
    rows = []
    for t in universe:
        f = fund_map.get(t, {})
        if not isinstance(f, dict):
            continue
        rows.append({
            "ticker": t,
            "roe": f.get("returnOnEquity"),
            "gross": f.get("grossMargins"),
            "pm": f.get("profitMargins"),
            "pb": f.get("priceToBook"),
            "ps": f.get("priceToSalesTrailing12Months") or f.get("priceToSales"),
            "evebitda": f.get("enterpriseToEbitda") or f.get("enterpriseToEbitdaratio"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    def _num(x): return pd.to_numeric(x, errors="coerce")         # 轉數字，非法變 NaN
    df = df.set_index("ticker")
    for c in ["roe","gross","pm","pb","ps","evebitda"]:
        df[c] = _num(df[c])
    for c in ["pb","ps","evebitda"]:
        df.loc[(df[c]<=0) | (df[c].isna()), c] = np.nan           # 非法值改 NaN
    def _z(s): return (s - s.mean()) / (s.std(ddof=0) + 1e-12)    # Z 分數
    q = _z(df["roe"]).fillna(0) + _z(df["gross"]).fillna(0) + _z(df["pm"]).fillna(0)
    v = -_z(np.log(df["pb"])).fillna(0) + -_z(np.log(df["ps"])).fillna(0) + -_z(np.log(df["evebitda"])).fillna(0)
    score = (q + v).replace([np.inf, -np.inf], np.nan).fillna(0)  # 合成分數
    if top_n and top_n > 0:
        return score.sort_values(ascending=False).head(top_n).index.tolist()
    ranks = score.rank(method="first") / len(score)
    qcut = pd.qcut(ranks, 5, labels=False, duplicates="drop")
    return score[qcut == 4].index.tolist()


# ------------------------
# 核心組合邏輯
# ------------------------
def build_universe(prices: pd.DataFrame,
                   fundamentals: Dict[str, dict],
                   cfg: EngineConfig) -> List[str]:
    """可投資池：踢掉基準、要求最少歷史與上市冷卻"""
    cols = list(prices.columns)
    if cfg.benchmark in cols:
        cols.remove(cfg.benchmark)                                 # 不把基準當成可投資標的
    universe = []
    for t in cols:
        s = prices[t].dropna()
        if s.shape[0] < cfg.min_history_days:                      # 歷史太短就丟
            continue
        if (s.index.max() - s.index.min()).days < cfg.ipo_seasoning_days:  # 上市太近就丟
            continue
        universe.append(t)
    log(f"[INFO] Universe size after min history & seasoning: {len(universe)}")
    return sorted(universe)


def apply_adv_filter(universe: List[str], adv_df: Optional[pd.DataFrame], min_adv_usd: Optional[float]) -> List[str]:
    """（選）用成交額 ADV 過濾低流動性股票"""
    if adv_df is None or min_adv_usd is None:
        return universe
    last = adv_df.dropna(how="all").iloc[-1]                       # 取最後一行（最近日）的 ADV
    keep = [t for t in universe if t in last.index and pd.notna(last[t]) and last[t] >= min_adv_usd]
    log(f"[INFO] ADV filter applied: {len(keep)} names kept (min_adv_usd={min_adv_usd:,.0f})")
    return keep


def cap_and_normalize_weights(w: Dict[str, float], max_weight: float) -> Dict[str, float]:
    """套單檔上限後重正規化（總和=1）"""
    if max_weight is None or max_weight <= 0:
        s = sum(w.values());  return {k: v / s for k, v in w.items()} if s > 0 else {}
    capped = {k: min(v, max_weight) for k, v in w.items()}         # 逐檔上限
    s = sum(capped.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in capped.items()}                   # 歸一化


def compute_weights_time_series(prices: pd.DataFrame,
                                rebalance_dates: pd.DatetimeIndex,
                                target_portfolios: pd.DataFrame,
                                cfg: EngineConfig) -> pd.DataFrame:
    """
    生成每日權重：
    - 只在調倉日把權重往目標移動，期間持有不動
    - No-trade band：若 |目標-前一日| <= band，就不交易該檔
    - 先鎖住 band 內不動的那部分，再把剩餘權重按目標比例分給需要調整的標的
    """
    cols = prices.columns
    weights = pd.DataFrame(0.0, index=prices.index, columns=cols)  # 初始化全 0 權重

    def _pos(d):                                                   # 幫手：把任意日期貼到最近交易日的位置
        return prices.index.get_indexer([pd.Timestamp(d)], method="ffill")[0]

    prev_pos = None                                                # 前一次調倉位置（第一次為 None）
    prev_w = pd.Series(0.0, index=cols)                            # 前一日權重（初始全 0）

    for i, d in enumerate(rebalance_dates):                        # 逐個月末交易日
        cur_pos = _pos(d)                                          # 這次調倉所在位置
        next_pos = (_pos(rebalance_dates[i+1]) if i < len(rebalance_dates)-1 else len(prices.index)-1)  # 下次之前的最後一日

        # --- 目標等權重（先依名單計、含因子權重占比） ---
        row = target_portfolios.loc[d]
        mom_list = row.get("Momentum", []) or []                   # 當月動量名單
        w_target = {}
        if mom_list:
            wm = cfg.factor_weights[0] / len(mom_list)             # 動量部分平均分配
            for t in mom_list:
                if t in cols:
                    w_target[t] = w_target.get(t, 0.0) + wm
        if cfg.enable_qv:                                          # 若啟用 QV，加入 QV 權重（預設關閉）
            qv_list = row.get("QualityValue", []) or []
            if qv_list:
                wq = cfg.factor_weights[1] / len(qv_list)
                for t in qv_list:
                    if t in cols:
                        w_target[t] = w_target.get(t, 0.0) + wq

        w_target = pd.Series(w_target, index=cols).fillna(0.0)     # 補齊所有欄（沒選到=0）
        w_target = w_target.clip(upper=cfg.max_weight)             # 單檔上限

        # --- No-trade band（以『前一日』作比較基準） ---
        band = float(cfg.rebalance_band)
        if prev_pos is None:                                       # 第一次沒有前一日，直接用目標
            w_new = w_target.copy()
            if w_new.sum() > 0:
                w_new = w_new / w_new.sum()                        # 歸一化
        else:
            dev = (w_target - prev_w).abs()                        # 權重變動幅度
            fixed_mask = dev <= band                               # 小於帶寬者：不調
            sum_fixed = prev_w[fixed_mask].sum()                   # 被鎖定的總權重

            change_mask = ~fixed_mask                              # 需要調的集合
            w_change = w_target[change_mask].clip(upper=cfg.max_weight)
            if w_change.sum() > 0 and (1 - sum_fixed) > 1e-12:
                w_change = w_change * (1 - sum_fixed) / w_change.sum()  # 把剩餘權重按比例分給要調的
            else:
                w_change[:] = 0.0

            w_new = prev_w.copy()                                  # 從舊權重出發
            w_new[fixed_mask] = prev_w[fixed_mask]                 # 帶寬內保持不動
            w_new[change_mask] = w_change                          # 其他才調整

            if w_new.sum() > 0:
                w_new = w_new / w_new.sum()                        # 最後再歸一化（保險）

        # --- 把這次調倉到下次調倉前的每天權重都設為 w_new ---
        weights.iloc[cur_pos:next_pos+1, :] = w_new.values
        prev_pos = cur_pos                                         # 更新前一次調倉位置
        prev_w = w_new.copy()                                      # 更新前一日權重

    return weights


def risk_report(daily: pd.Series) -> Dict[str, float]:
    """產生風險表（年化、Sharpe、Sortino、回撤、最差月等）"""
    eq = (1 + daily).cumprod()                                     # 權益曲線
    monthly = (1 + daily).resample("ME").prod() - 1                # 月報酬（'ME' 月末）
    rep = {
        "Total Return %": (eq.iloc[-1] - 1) * 100,
        "Ann Return %": annualize_return(daily) * 100,
        "Ann Vol %": annualized_vol(daily) * 100,
        "Sharpe": sharpe_ratio(daily),
        "Sortino": sortino_ratio(daily),
        "Max Drawdown %": max_drawdown(eq) * 100,
        "Calmar": (annualize_return(daily) / abs(max_drawdown(eq))) if abs(max_drawdown(eq)) > 1e-12 else 0.0,
        "Max Recovery Days": max_recovery_days(eq),
        "Worst Day %": daily.min() * 100,
        "Worst Month %": monthly.min() * 100,
        "Hit Ratio %": hit_ratio(daily) * 100,
    }
    # 轉成乾淨的 float，避免出現 numpy 型別在 CSV 裡亂碼
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in rep.items()}


def attribution_series(raw_ret: pd.Series, vt_ret: pd.Series, tc: pd.Series) -> pd.DataFrame:
    """把累積報酬拆成三塊：選股、波控、成本；另算淨值"""
    df = pd.DataFrame({
        "Selection (raw)": (1 + raw_ret).cumprod() - 1,           # 只有選股（沒波控、沒成本）
        "VolTargeting adj": (1 + vt_ret).cumprod() - 1,           # 加上波動率目標
        "Transaction Cost drag": -tc.cumsum(),                    # 成本累積拖累（負數）
    })
    df["Net"] = df["VolTargeting adj"] - df["Transaction Cost drag"]  # 淨值 = 波控後 − 成本
    return df


def build_target_portfolios(prices: pd.DataFrame,
                            universe: List[str],
                            fundamentals: Dict[str, dict],
                            cfg: EngineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """每月（月末交易日）產生『目標名單』，並輸出覆盤用 CSV"""
    rebalance_dates = month_end_trading_days(prices.index)         # 每月最後交易日
    target = pd.DataFrame(index=rebalance_dates, columns=["Momentum","QualityValue"], dtype=object)

    log("Calculating monthly target portfolios...")
    for d in rebalance_dates:
        mom = momentum_picks(prices, universe, d, cfg.mom_lookback_days, cfg.mom_skip_days, cfg.top_n)  # 動量名單
        target.loc[d, "Momentum"] = mom
        if cfg.enable_qv:
            qv = qvalue_picks(fundamentals, universe, cfg.top_n)  #（選）QV 名單
        else:
            qv = []
        target.loc[d, "QualityValue"] = qv

    # 把名單攤平成字串，方便人工覆盤
    picks = pd.DataFrame({
        "date": target.index,
        "momentum_tickers": target["Momentum"].apply(lambda x: ";".join(x or [])),
        "qv_tickers": target["QualityValue"].apply(lambda x: ";".join(x or [])),
    })
    return target, picks


def run_once(prices: pd.DataFrame,
             fundamentals: Dict[str, dict],
             cfg: EngineConfig,
             out_dir: Path,
             tag: str = "main") -> Dict[str, object]:
    """以目前參數跑一次：建投資池→選股→生成權重→T+1→波控→扣成本→輸出檔案"""
    universe = build_universe(prices, fundamentals, cfg)           # 可投資清單

    # （選）ADV 流動性過濾
    adv_df = None
    if cfg.adv_csv and Path(cfg.adv_csv).exists():
        try:
            adv_df = pd.read_csv(cfg.adv_csv)                      # 讀 ADV（若有）
            date_cols = [c for c in adv_df.columns if c.lower() in ("date", "timestamp")]
            if date_cols:
                adv_df.rename(columns={date_cols[0]:"Date"}, inplace=True)
                adv_df["Date"] = pd.to_datetime(adv_df["Date"])
                adv_df.set_index("Date", inplace=True)
        except Exception as e:
            log(f"[WARN] Failed to load ADV CSV: {e}")
            adv_df = None
    universe = apply_adv_filter(universe, adv_df, cfg.min_adv_usd) # 真的需要才啟用

    target, picks_df = build_target_portfolios(prices, universe, fundamentals, cfg)  # 每月名單
    picks_df.to_csv(out_dir / f"monthly_picks_{tag}.csv", index=False)               # 存檔：覆盤用

    weights = compute_weights_time_series(prices, target.index, target, cfg)         # 每日權重
    rets = prices.pct_change().fillna(0.0)                                           # 每日簡單報酬

    raw_ret = (weights.shift(1) * rets).sum(axis=1)                                  # T+1：昨天權重 × 今天報酬
    roll_vol = raw_ret.rolling(cfg.vol_window).std(ddof=0) * math.sqrt(252)         # 近 20 日年化波動
    scaler = (cfg.target_vol / (roll_vol + 1e-8)).clip(upper=cfg.leverage_cap).shift(1).fillna(0.0)  # 倍數（前移1日）
    vt_ret = raw_ret * scaler                                                        # 波控後日報酬

    turnover = (weights.diff().abs().sum(axis=1) / 2).fillna(0.0)                    # 單邊換手（權重變化/2）
    tc = (cfg.tc_bps / 1e4) * turnover                                              # 成本 = bps × 換手
    net_ret = vt_ret - tc                                                            # 淨報酬（含成本）
    bench_ret = rets[cfg.benchmark].copy() if cfg.benchmark in rets.columns else pd.Series(index=rets.index, dtype=float)

    risk_strat = risk_report(net_ret)                                                # 策略風險表
    risk_bench = risk_report(bench_ret) if not bench_ret.dropna().empty else {}      # 基準風險表（若有）
    perf_table = pd.DataFrame([risk_strat, risk_bench], index=[f"Strategy({tag})", f"Benchmark({cfg.benchmark})"])  # 合併
    attrib = attribution_series(raw_ret, vt_ret, tc)                                 # 歸因

    # --- 各種輸出檔（CSV/圖） ---
    (out_dir / f"strategy_net_daily_returns_{tag}.csv").write_text(net_ret.to_csv(header=["ret"]))
    (out_dir / f"benchmark_daily_returns_{tag}.csv").write_text(bench_ret.to_csv(header=["ret"]))
    (out_dir / f"turnover_daily_{tag}.csv").write_text(turnover.to_csv(header=["turnover"]))
    (out_dir / f"risk_report_{tag}.csv").write_text(perf_table.to_csv())
    attrib.to_csv(out_dir / f"attribution_{tag}.csv")

    # 權益曲線圖（對數）
    eq_net = (1 + net_ret).cumprod()
    eq_b = (1 + bench_ret).cumprod() if not bench_ret.dropna().empty else None
    plt.figure(figsize=(12,6))
    plt.plot(eq_net, label=f"Strategy (net, {tag})")
    if eq_b is not None:
        plt.plot(eq_b, label=f"Benchmark ({cfg.benchmark})")
    plt.yscale("log"); plt.xlabel("Date"); plt.ylabel("Equity (log)")
    plt.title(f"Strategy vs Benchmark ({tag})"); plt.legend(); plt.tight_layout()
    fig_path = out_dir / f"equity_{tag}.png"
    plt.savefig(fig_path, dpi=200); plt.close()
    log(f"[INFO] Saved: {fig_path}")

    # --- 月度成本/換手/名單變化（方便找成本高峰與原因） ---
    cost_df = pd.DataFrame({"turnover": turnover, "tc": tc})       # 合成成本表
    monthly_cost = cost_df.resample("ME").agg({"turnover":"sum","tc":"sum"})  # 月度彙總
    monthly_cost.rename(columns={"turnover":"Turnover_sum","tc":"Cost_sum"}, inplace=True)

    rows = []
    prev_set = set()
    for d in target.index:                                         # 每月名單變化：新增/刪除/留存
        cur_set = set(target.loc[d, "Momentum"] or [])
        adds = len(cur_set - prev_set)
        drops = len(prev_set - cur_set)
        kept = len(cur_set & prev_set)
        rows.append({"date": d, "adds": adds, "drops": drops, "kept": kept})
        prev_set = cur_set
    chg = pd.DataFrame(rows).set_index("date")
    chg_month = chg.resample("ME").sum()                           # 按月累計新增/刪除數

    breakdown = monthly_cost.join(chg_month, how="left").fillna(0) # 合併成本與名單變化
    breakdown.to_csv(out_dir / f"turnover_breakdown_{tag}.csv")    # 輸出

    return {                                                        # 回傳給上層（敏感度等會用到）
        "net_ret": net_ret,
        "bench_ret": bench_ret,
        "turnover": turnover,
        "perf_table": perf_table,
        "attrib": attrib,
        "picks_df": picks_df,
    }


def sensitivity_grid(prices: pd.DataFrame,
                     fundamentals: Dict[str, dict],
                     cfg: EngineConfig,
                     out_dir: Path) -> pd.DataFrame:
    """小型參數網格（top_n / target_vol / lookback）評估穩定性"""
    grid_topn = [max(10, cfg.top_n//2), cfg.top_n, min(200, cfg.top_n*2)]          # 前後各拉一檔
    grid_tv = [max(0.04, cfg.target_vol-0.02), cfg.target_vol, min(0.30, cfg.target_vol+0.02)]
    grid_lb = [max(120, cfg.mom_lookback_days-63), cfg.mom_lookback_days, min(378, cfg.mom_lookback_days+63)]

    rows = []
    for tn in grid_topn:
        for tv in grid_tv:
            for lb in grid_lb:
                cfg2 = EngineConfig(**{**cfg.__dict__})                             # 複製設定
                cfg2.top_n = int(tn); cfg2.target_vol = float(tv); cfg2.mom_lookback_days = int(lb)
                res = run_once(prices, fundamentals, cfg2, out_dir, tag=f"sens_tn{tn}_tv{tv:.2f}_lb{lb}")
                net = res["net_ret"]; eq = (1 + net).cumprod()
                rows.append({                                                       # 收集指標
                    "top_n": tn,
                    "target_vol": tv,
                    "mom_lookback_days": lb,
                    "AnnRet%": annualize_return(net) * 100,
                    "AnnVol%": annualized_vol(net) * 100,
                    "Sharpe": sharpe_ratio(net),
                    "Sortino": sortino_ratio(net),
                    "MDD%": max_drawdown(eq) * 100,
                    "HitRate%": hit_ratio(net) * 100,
                    "AvgTurnover%/day": res["turnover"].mean() * 100,
                    "MedTurnover%/day": res["turnover"].median() * 100,
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sensitivity.csv", index=False)                                # 匯出敏感度總表
    log("[INFO] Saved sensitivity.csv")
    return df


def walk_forward(prices: pd.DataFrame,
                 fundamentals: Dict[str, dict],
                 cfg: EngineConfig,
                 out_dir: Path) -> None:
    """簡版 Walk-forward：split 前當訓練（挑參數），之後當測試（報表）"""
    if not cfg.wf_split:
        log("[WARN] walk_forward requested but wf_split not provided. Skipping.")
        return
    split = pd.to_datetime(cfg.wf_split)                                             # 切分日期
    train = prices[prices.index <= split]                                            # 訓練段
    test = prices[prices.index > split]                                              # 測試段
    if train.empty or test.empty:
        log("[WARN] walk_forward split produced empty train/test. Skipping.")
        return

    # 簡單的參數網格（可擴）
    grid = [
        (cfg.top_n, cfg.target_vol, cfg.mom_lookback_days),
        (cfg.top_n, max(0.06, cfg.target_vol-0.02), cfg.mom_lookback_days),
        (cfg.top_n, min(0.20, cfg.target_vol+0.02), cfg.mom_lookback_days),
        (max(10, cfg.top_n//2), cfg.target_vol, max(120, cfg.mom_lookback_days-63)),
        (min(200, cfg.top_n*2), cfg.target_vol, min(378, cfg.mom_lookback_days+63)),
    ]
    best = None; best_sharpe = -1e9
    for tn, tv, lb in grid:
        cfg2 = EngineConfig(**{**cfg.__dict__})
        cfg2.top_n = int(tn); cfg2.target_vol = float(tv); cfg2.mom_lookback_days = int(lb)
        res_tr = run_once(train, fundamentals, cfg2, out_dir, tag=f"wf_train_tn{tn}_tv{tv:.2f}_lb{lb}")
        sh = sharpe_ratio(res_tr["net_ret"])
        if sh > best_sharpe:                                       # 挑 Sharpe 最高者
            best_sharpe = sh; best = (tn, tv, lb)
    if best is None:
        log("[WARN] No best params found in train. Skipping test.")
        return

    tn, tv, lb = best
    cfg_best = EngineConfig(**{**cfg.__dict__})                    # 用最佳參數跑測試段
    cfg_best.top_n = int(tn); cfg_best.target_vol = float(tv); cfg_best.mom_lookback_days = int(lb)
    res_te = run_once(test, fundamentals, cfg_best, out_dir, tag=f"wf_test_best")

    summary = {                                                    # 寫入總結 JSON
        "wf_split": cfg.wf_split,
        "best_params": {"top_n": tn, "target_vol": tv, "mom_lookback_days": lb, "train_sharpe": best_sharpe},
        "test_risk": res_te["perf_table"].loc[f"Strategy(wf_test_best)"].to_dict()
                     if f"Strategy(wf_test_best)" in res_te["perf_table"].index else {}
    }
    (out_dir / "walkforward_summary.json").write_text(json.dumps(summary, indent=2))
    log("[INFO] Saved walkforward_summary.json")


# ------------------------
# 入口（CLI）
# ------------------------
def parse_args() -> EngineConfig:
    base = Path(__file__).resolve().parent                           # 檔案所在資料夾
    cwd = Path.cwd()                                                # 目前工作目錄
    roots = [base, cwd, base.parent, base/"data", cwd/"data"]       # 搜尋根路徑

    default_prices = str(base / "price_cache_1995-01-01_to_2024-12-31_504_tickers.csv")  # 預設價格檔名
    default_funds = str(base / "fundamentals_cache_503_tickers.json")                    # 預設基本面檔名
    default_out = str(base / "out")                                                       # 預設輸出資料夾

    ap = argparse.ArgumentParser(description="Professional Multi-Factor Backtest (ASCII-only logs)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prices", default=default_prices, help="Price CSV path")           # 價格檔路徑
    ap.add_argument("--fundamentals", default=default_funds, help="Fundamentals JSON path (static cache; DISABLED by default)")
    ap.add_argument("--out", default=default_out, help="Output directory")               # 輸出資料夾
    ap.add_argument("--benchmark", default="QQQ")                                        # 基準
    ap.add_argument("--start", default="1999-01-01")                                     # 起始日
    ap.add_argument("--end", default="2024-12-31")                                       # 結束日

    ap.add_argument("--top-n", type=int, default=50)                                     # 動量：前 N 檔
    ap.add_argument("--lookback", type=int, default=252)                                 # 動量視窗
    ap.add_argument("--skip-days", type=int, default=20)                                 # 動量略過天數

    ap.add_argument("--target-vol", type=float, default=0.10)                            # 波控目標
    ap.add_argument("--vol-window", type=int, default=20)                                # 估波窗口
    ap.add_argument("--leverage-cap", type=float, default=3.0)                           # 槓桿上限
    ap.add_argument("--max-weight", type=float, default=0.05)                            # 單檔上限
    ap.add_argument("--rebalance-band", type=float, default=0.005, help="No-trade band per name (abs delta weight <= band -> no trade)")

    ap.add_argument("--tc-bps", type=float, default=15.0)                                # 單邊成本 bps

    ap.add_argument("--min-history-days", type=int, default=252)                         # 最少歷史
    ap.add_argument("--ipo-seasoning-days", type=int, default=60)                        # 上市冷卻

    ap.add_argument("--adv-csv", default=None, help="Optional ADV csv (Date index, tickers columns)")  #（選）ADV 檔
    ap.add_argument("--min-adv-usd", type=float, default=None, help="Optional minimum ADV in USD")     #（選）最低 ADV 門檻

    ap.add_argument("--enable-qv", action="store_true", help="Enable Quality+Value (WARNING: requires point-in-time fundamentals with lag)")  #（選）開 QV

    ap.add_argument("--walk-forward", action="store_true")                                #（選）啟用 walk-forward
    ap.add_argument("--wf-split", default=None, help="YYYY-MM-DD split date for simple hold-out")      # 切分日

    args = ap.parse_args()

    # 自動搜尋檔案（使用者沒給正確路徑時）
    price_path = guess_existing_file(args.prices, [
        "price_cache*1995*504*tickers*.csv",
        "price_cache*1995*507*tickers*.csv",
        "*price*cache*.csv",
    ], roots) or args.prices

    fund_path = guess_existing_file(args.fundamentals, [
        "fundamentals_cache*503*tickers*.json",
        "fundamentals_cache*.json",
        "*fundamentals*.json",
    ], roots) or args.fundamentals

    # 把所有參數塞進 dataclass
    return EngineConfig(
        price_cache_path=price_path,
        fundamentals_cache_path=fund_path,
        out_dir=args.out,
        benchmark=args.benchmark,
        start_date=args.start,
        end_date=args.end,
        top_n=int(args.top_n),
        mom_lookback_days=int(args.lookback),
        mom_skip_days=int(args.skip_days),
        target_vol=float(args.target_vol),
        vol_window=int(args.vol_window),
        leverage_cap=float(args.leverage_cap),
        max_weight=float(args.max_weight),
        rebalance_band=float(args.rebalance_band),
        tc_bps=float(args.tc_bps),
        min_history_days=int(args.min_history_days),
        ipo_seasoning_days=int(args.ipo_seasoning_days),
        adv_csv=args.adv_csv,
        min_adv_usd=args.min_adv_usd,
        enable_qv=bool(args.enable_qv),
        walk_forward=bool(args.walk_forward),
        wf_split=args.wf_split,
    )


def save_run_meta(cfg: EngineConfig, prices: pd.DataFrame, out_dir: Path):
    """把本次跑的參數、資料範圍、套件版本寫 run_meta.json（方便重現）"""
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",         # UTC 時戳
        "params": {k: (v if isinstance(v, (int,float,str,bool)) or v is None else str(v)) for k,v in cfg.__dict__.items()},
        "price_shape": [int(prices.shape[0]), int(prices.shape[1])],
        "date_range": [str(prices.index.min()), str(prices.index.max())],
        "libs": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": plt.matplotlib.__version__}
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log("[INFO] Saved run_meta.json")


def main():
    """主流程：讀參數→載資料→跑主回測→敏感度→（選）walk-forward→寫 meta"""
    cfg = parse_args()                                             # 讀 CLI 參數
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)  # 確保輸出資料夾存在

    prices = load_prices(cfg.price_cache_path, cfg.start_date, cfg.end_date)  # 讀價格
    if cfg.benchmark not in prices.columns:
        raise ValueError(f"Benchmark {cfg.benchmark} not found in price columns.")  # 沒找到基準就報錯
    fundamentals = load_fundamentals(cfg.fundamentals_cache_path)  #（可能為空）基本面 map

    _ = run_once(prices, fundamentals, cfg, out_dir, tag="main")   # 主回測（輸出風險表/圖/CSV）
    _ = sensitivity_grid(prices, fundamentals, cfg, out_dir)       # 參數敏感度（小網格）
    if cfg.walk_forward:                                           #（選）walk-forward
        walk_forward(prices, fundamentals, cfg, out_dir)

    save_run_meta(cfg, prices, out_dir)                            # 紀錄本次執行的環境
    log("[DONE] All outputs written.")                             # 結束訊息


if __name__ == "__main__":
    main()                                                         # 入口
