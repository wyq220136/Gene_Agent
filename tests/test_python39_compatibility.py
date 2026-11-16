"""阻止真机 Python 3.9 的导入/语法回归，保留严格迭代的运行行为。"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys

import pytest

from gene_agent._compat import strict_zip


@pytest.mark.parametrize("values, expected", [
    ((), []), (([], []), []), (([1, 2],), [(1,), (2,)]),
    (([1, 2], [3, 4]), [(1, 3), (2, 4)]),
])
def test_strict_zip_equal_lengths(values, expected):
    assert list(strict_zip(*values)) == expected


@pytest.mark.parametrize("values", [([1], []), ([], [1]), ([1], [2, 3]),
                                       ([1, 2], [3]), ([], [], [1]), ([1], [2], [])])
def test_strict_zip_rejects_length_mismatch(values):
    with pytest.raises(ValueError):
        list(strict_zip(*values))


def test_strict_zip_does_not_consume_ahead_or_hide_iterator_errors():
    first, second = iter([1, 2]), iter([3, 4])
    rows = strict_zip(first, second)
    assert next(rows) == (1, 3)
    assert next(first) == 2
    assert next(second) == 4
    with pytest.raises(TypeError):
        strict_zip(None)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="内建 strict 参数只用于较新解释器的对照")
def test_strict_zip_matches_builtin_errors_and_consumption():
    import itertools
    for lengths in itertools.product(range(3), repeat=3):
        outcomes = []
        for implementation in (strict_zip, lambda *values: zip(*values, strict=True)):
            inputs = [iter(range(length)) for length in lengths]
            rows, error = [], None
            try:
                for row in implementation(*inputs):
                    rows.append(row)
            except ValueError as exc:
                error = str(exc)
            outcomes.append((rows, error, [list(value) for value in inputs]))
        assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("name", [
    "gene_agent.knowledge.geometry.loader",
    "gene_agent.knowledge.operation.rotation",
    "gene_agent.knowledge.operation.door_push",
    "gene_agent.realworld.scene_drift",
    "gene_agent.realworld.microwave.configuration",
    "gene_agent.realworld.microwave.live_perception",
    "gene_agent.perception_modules.pose_est.pose_server",
    "gene_agent.perception_modules.pose_est.visualization.rgbd_points",
    "gene_agent.scene.objects",
    "gene_agent.simulation.mujoco_scene",
    "gene_agent.tasks.checks",
    "gene_agent.tasks.policies.sweep_long_task.simulation",
    "tools.run.run_microwave_task1_realworld",
    "tools.run.prepare_microwave_realworld",
])
def test_realworld_imports_on_current_interpreter(name):
    importlib.import_module(name)


def test_owned_sources_parse_on_python39_and_avoid_new_runtime_features():
    root = Path(__file__).resolve().parents[1]
    problems = []
    for folder in ("gene_agent", "tools", "benchmarks", "experiments"):
        for path in (root / folder).rglob("*.py"):
            if "FoundationPose" in path.parts:
                continue  # 第三方源码由原 FoundationPose 环境管理。
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=9)
            future_annotations = any(
                isinstance(node, ast.ImportFrom) and node.module == "__future__"
                and any(alias.name == "annotations" for alias in node.names)
                for node in tree.body
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                    keywords = {kw.arg for kw in node.keywords}
                    if (name == "dataclass" and keywords & {"slots", "kw_only", "weakref_slot"}
                            or name == "zip" and "strict" in keywords):
                        problems.append(f"{path.relative_to(root)}:{node.lineno}: {name}")
                if not future_annotations:
                    annotation = getattr(node, "annotation", None) or getattr(node, "returns", None)
                    if annotation and any(isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr)
                                          for child in ast.walk(annotation)):
                        problems.append(f"{path.relative_to(root)}:{node.lineno}: eager union")
    assert not problems, "\n".join(problems)
