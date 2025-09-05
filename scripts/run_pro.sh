#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# 可選：載入 .env（若存在）
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

OUT_DIR="${OUT:-out}"
mkdir -p "$OUT_DIR"

# 不添加任何假資料或強制參數；使用者可在 .env 或命令列自行提供
exec python run_engine.py --out "$OUT_DIR" "$@"
