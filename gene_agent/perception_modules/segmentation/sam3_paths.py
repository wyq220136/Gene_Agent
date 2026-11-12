"""SAM3 third_party 代码和资源的路径配置。"""

from __future__ import annotations

from gene_agent.perception_modules.paths import PROJECT_ROOT

import sys
from dataclasses import dataclass
from pathlib import Path


_SEGMENTATION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = PROJECT_ROOT


@dataclass(frozen=True)
class Sam3Paths:
    """SAM3 源码目录及其 BPE 资源路径。"""

    repo_root: Path
    sam3_root: Path
    bpe_path: Path


def get_sam3_paths() -> Sam3Paths:
    """返回仓库内 third_party/sam3 的源码和资源路径。"""
    sam3_root = _REPO_ROOT / "third_party" / "sam3"
    return Sam3Paths(
        repo_root=_REPO_ROOT,
        sam3_root=sam3_root,
        bpe_path=sam3_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    )


def configure_sam3_import() -> Sam3Paths:
    """将 third_party/sam3 放到 sam3 包导入路径的最前面。"""
    paths = get_sam3_paths()
    if str(paths.sam3_root) not in sys.path:
        sys.path.insert(0, str(paths.sam3_root))
    return paths
