#paths - the ONE place that knows the directory layout. Library of PATHS only; no IO logic.
#
#LAYOUT (level 0 = the repo root):
#   PhasicNet/                      ROOT_DIR
#     system/                       every framework file          <- this file lives here      SYSTEM_DIR
#       tests/                                                                                 TESTS_DIR
#       runlog.yml                  the logging config                                          LOG_CFG_PATH
#     system_modules/               capture, operations, phases, protocols, scorer              MODULES_DIR
#     experiment_dynamics/          the model modules (deckers2022, hlsp2026, ... original_dynamics)   DYNAMICS_DIR
#     experiments/                  entry points: build/run scripts, readouts, preprocessing     EXPERIMENTS_DIR
#     datasets/
#       raw/                        raw datasets land here by default
#       train_test/                 the ONE-TIME test/train split (or the spec to recreate it)
#     runs/                         everything a RUN emits: history, trained models, rasters
#     networks/                     realized networks + netspecs, shared ACROSS runs
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

##system/paths.py -> system/ -> PhasicNet/
SYSTEM_DIR = Path(__file__).resolve().parent
ROOT_DIR = SYSTEM_DIR.parent

TESTS_DIR = SYSTEM_DIR / "tests"
MODULES_DIR = ROOT_DIR / "system_modules"         ##capture / operations / phases / protocols / scorer
DYNAMICS_DIR = ROOT_DIR / "experiment_dynamics"   ##the model modules
EXPERIMENTS_DIR = ROOT_DIR / "experiments"        ##the build/run scripts


DATASET_DIR = ROOT_DIR / "datasets"
RAW_DIR = DATASET_DIR / "raw"  ##dataset imports
TRAIN_TEST_DIR = DATASET_DIR / "train_test"   ##Spec + indices for ONE-time-splits -> SAME test-set is held out across every hyperparameter run 

##RUNS. Files are named TYPE_<run_id> (ex. History_<run_id>, TrainedModel_<run_id>, Raster_<run_id>).
RUNS_DIR = ROOT_DIR / "runs"

##NETWORKS. A network reused across ten runs is stored ONCE and referenced by network_id from RunMeta
NETWORKS_DIR = ROOT_DIR / "networks"

##EXPORT_DIR default for save() calls without destination
EXPORT_DIR = RUNS_DIR

for _d in (DATASET_DIR, RAW_DIR, TRAIN_TEST_DIR, RUNS_DIR, NETWORKS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_PATH = RAW_DIR / "mnist.npz"
FMNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"      ##Fashion-MNIST as 4 idx files (Keras' mirror)
FMNIST_PATH = RAW_DIR / "fmnist.npz"

##the logging config runlog.setup_logging() reads
LOG_CFG_PATH = SYSTEM_DIR / "runlog.yml"

##NOTE the sys.path entry doors: experiments/__init__.py (scripts) and system/tests/conftest.py (pytest) put the four code folders on sys.path.
## Python must find paths.py BEFORE paths can run, so that step cannot live here.