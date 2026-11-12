"""感知数据与模型资源的项目路径。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = PROJECT_ROOT.parent / "storage" / "ckpts"
DEFAULT_TEST_DIR = "data/perception_test"
