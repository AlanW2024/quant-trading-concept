# tests/test_entrypoint.py
import pathlib
import subprocess
import sys

import pytest


def _pro_exists() -> bool:
    root = pathlib.Path(__file__).resolve().parents[1]
    return (root / "src/factor/professional_multifactor_engine_pro.py").exists() or (
        root / "factor/professional_multifactor_engine_pro.py"
    ).exists()


def _deps_ok() -> bool:
    try:
        import numpy  # noqa: F401
        # 如需更嚴謹可加：import pandas  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not (_pro_exists() and _deps_ok()), reason="pro engine or deps not present")
def test_run_engine_help():
    root = pathlib.Path(__file__).resolve().parents[1]
    ret = subprocess.call([sys.executable, str(root / "run_engine.py"), "--help"])
    assert ret == 0