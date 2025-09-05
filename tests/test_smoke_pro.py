import subprocess, sys, pathlib, pytest

def _pro_exists() -> bool:
    root = pathlib.Path(__file__).resolve().parents[1]
    return (root / "src/factor/professional_multifactor_engine_pro.py").exists() or \
           (root / "factor/professional_multifactor_engine_pro.py").exists()

@pytest.mark.smoke
@pytest.mark.skipif(not _pro_exists(), reason="pro engine not present")
def test_run_engine_help_exit_zero():
    root = pathlib.Path(__file__).resolve().parents[1]
    ret = subprocess.call([sys.executable, str(root / "run_engine.py"), "--help"])
    assert ret == 0