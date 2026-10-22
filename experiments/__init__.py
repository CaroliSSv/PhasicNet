#experiments/ -- the ONE entry door for the scripts. Importing experiments (or anything in it) puts the FOUR code folders on sys.path, found from THIS file's location.

##NOTE
# #Run from the repo root (conda activate PhasicNet; cd <PhasicNet>):   python -c "import experiments, dc_reproduce; dc_reproduce.main(tag='t1')"
# Experiment files import each other FLAT (from dc_protocol import ...); importing one also as experiments.X loads it twice (ghost cfg classes).
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]          ##experiments/__init__.py -> PhasicNet/
_CODE_DIRS = (_ROOT / "system", _ROOT / "system_modules", _ROOT / "experiment_dynamics", _ROOT / "experiments")
sys.path[:0] = [str(d) for d in _CODE_DIRS if str(d) not in sys.path]   ##system/ FIRST: a framework name must never be shadowed by an experiment file
