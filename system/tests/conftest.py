#conftest -- loaded by pytest before any test in system/tests/; puts the four code folders on sys.path.
#Run: pytest system/tests -q      (from the repo root)
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]          ##system/tests/conftest.py -> system/ -> PhasicNet/
_CODE_DIRS = (_ROOT / "system", _ROOT / "system_modules", _ROOT / "experiment_dynamics", _ROOT / "experiments")
sys.path[:0] = [str(d) for d in _CODE_DIRS if str(d) not in sys.path]


##every test gets its own empty split store (beside tmp_path, not in it), so the suite never touches datasets/train_test/
import pytest

@pytest.fixture(autouse=True)
def _own_split_store(monkeypatch, tmp_path_factory):
    import datahandler
    monkeypatch.setattr(datahandler, "TRAIN_TEST_DIR", tmp_path_factory.mktemp("train_test"))
