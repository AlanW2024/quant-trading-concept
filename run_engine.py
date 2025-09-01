#!/usr/bin/env python3
# run_engine.py — thin launcher
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path

def main() -> int:
    root = Path(__file__).resolve().parent

    candidates = [
        root / "src" / "factor" / "professional_multifactor_engine_pro.py",
        root / "factor" / "professional_multifactor_engine_pro.py",
        root / "src" / "factor" / "professional_multifactor_engine.py",
        root / "factor" / "professional_multifactor_engine.py",
    ]

    for script in candidates:
        if script.exists():
            if os.environ.get("RUN_ENGINE_VERBOSE") == "1":
                print(f"[run_engine] using {script.relative_to(root)}", file=sys.stderr)
            cmd = [sys.executable, str(script), *sys.argv[1:]]
            return subprocess.call(cmd)

    sys.stderr.write(
        "[FATAL] engine not found. Looked for:\n"
        "  - src/factor/professional_multifactor_engine_pro.py\n"
        "  - factor/professional_multifactor_engine_pro.py\n"
        "  - src/factor/professional_multifactor_engine.py\n"
        "  - factor/professional_multifactor_engine.py\n"
    )
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
