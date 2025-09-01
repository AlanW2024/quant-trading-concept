#!/usr/bin/env python3
# src/tools/build_macro_csv.py
from __future__ import annotations

import argparse
from datetime import datetime
from typing import List, Tuple

import pandas as pd
from pandas_datareader import data as pdr


def parse_series_arg(arg: str) -> Tuple[str, str]:
    """Parse 'ID[:name]' into (ID, name)."""
    if ":" in arg:
        sid, name = arg.split(":", 1)
        return sid.strip(), name.strip()
    return arg.strip(), arg.strip()


def fetch_fred_series(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch a single FRED series as a pandas Series with name set to series_id."""
    df = pdr.DataReader(series_id, "fred", start, end)
    # DataReader returns a DataFrame with one column = series_id
    if isinstance(df, pd.DataFrame):
        if df.shape[1] == 1:
            s = df.iloc[:, 0]
        else:
            s = df[series_id]
    else:
        s = df  # already a Series
    s = s.loc[(s.index >= start) & (s.index <= end)]
    s.name = series_id  # ✅ 正確地設定名稱，避免把字串當函式呼叫
    return s


def build_dataframe(series: List[Tuple[str, str]], start: str, end: str, freq: str, fill: str, strict: bool) -> pd.DataFrame:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    cols = []
    errors = []
    for sid, name in series:
        try:
            s = fetch_fred_series(sid, start_ts, end_ts)
            s.name = name  # 允許自訂欄名
            cols.append(s)
        except Exception as e:
            errors.append((sid, str(e)))
            print(f"[WARN] failed to fetch {sid}: {e}")

    if strict and errors:
        raise RuntimeError(f"Some series failed: {errors}")

    if not cols:
        raise RuntimeError("No series fetched; nothing to write.")

    df = pd.concat(cols, axis=1)

    if freq:
        # 將頻率對齊到指定頻率
        df = df.asfreq(freq)

    if fill == "ffill":
        df = df.ffill()
    elif fill == "bfill":
        df = df.bfill()
    elif fill == "none":
        pass
    else:
        raise ValueError(f"Unknown fill method: {fill}")

    return df


def main() -> int:
    p = argparse.ArgumentParser(description="Build a macro CSV from FRED series.")
    p.add_argument("--start", required=True, help="start date, e.g. 2020-01-01")
    p.add_argument("--end", required=True, help="end date, e.g. 2024-12-31")
    p.add_argument("--out", required=True, help="output CSV path, e.g. data/macro.csv")
    p.add_argument(
        "--series",
        required=True,
        nargs="+",
        help="One or more FRED series like CPIAUCSL[:cpi] DGS10[:us10y] VIXCLS[:vix]",
    )
    p.add_argument("--freq", default="D", help="Target frequency: D/W/M/Q (default: D)")
    p.add_argument("--fill", default="ffill", choices=["ffill", "bfill", "none"], help="Missing value fill (default: ffill)")
    p.add_argument("--strict", action="store_true", help="Fail if any single series fails (default: False)")

    args = p.parse_args()
    pairs = [parse_series_arg(x) for x in args.series]

    df = build_dataframe(pairs, args.start, args.end, args.freq, args.fill, args.strict)
    out_path = args.out
    # 確保輸出資料夾存在
    pd.Path = None
    from pathlib import Path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index_label="date")
    print(f"[OK] wrote {out_path} with shape {df.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
