#!/usr/bin/env python3
# run_engine.py — force PRO engine only
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent

    # 只接受 PRO，兩個可能路徑其一存在即可
    candidates = [
        root / "src" / "factor" / "professional_multifactor_engine_pro.py",
        root / "factor" / "professional_multifactor_engine_pro.py",
    ]

    for script in candidates:
        if script.exists():
            if os.environ.get("RUN_ENGINE_VERBOSE") == "1":
                print(f"[run_engine] using {script.relative_to(root)}", file=sys.stderr)
            cmd = [sys.executable, str(script), *sys.argv[1:]]
            return subprocess.call(cmd)

    # 找不到就直接報錯，退出碼 2
    sys.stderr.write(
        "[FATAL] professional_multifactor_engine_pro.py is required but was not found.\n"
        "Searched:\n"
        "  - src/factor/professional_multifactor_engine_pro.py\n"
        "  - factor/professional_multifactor_engine_pro.py\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
