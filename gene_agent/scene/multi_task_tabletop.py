"""旧 import 兼容；唯一实现见 benchmarks.tabletop。

仓库内调用已迁移；外部调用迁移后可移除此别名。
"""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("benchmarks.tabletop")
