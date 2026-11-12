"""Viser 抓取调参 demo 的纯数据层测试；测试过程不会启动 Web server。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gene_agent.knowledge.geometry import CylinderConcept, RingConcept
from gene_agent.visualization.grasp_tuner.session import EditableGraspSession, build_session_state


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("profile_name", "concept_type", "visual_kind"),
    [
        ("task_1_bowl_0", RingConcept, "mesh"),
        ("task_1_knob_0", CylinderConcept, "cylinder"),
    ],
)
def test_two_task_profiles_build_renderable_session_state(
    profile_name: str,
    concept_type: type,
    visual_kind: str,
) -> None:
    """两个首批 profile 都应由同一入口形成可渲染、可切换的候选状态。"""
    state = build_session_state(ROOT, profile_name)

    assert state.profile_name == profile_name
    assert isinstance(state.concept, concept_type)
    assert state.object_visual.kind == visual_kind
    np.testing.assert_allclose(state.object_pose_world, np.eye(4))
    assert state.candidates
    assert state.selected_index == 0
    assert state.selected_candidate is state.candidates[0]
    assert all(candidate.closing_width_m >= 0.0 for candidate in state.candidates)


def test_bowl_visual_uses_repository_mesh() -> None:
    state = build_session_state(ROOT, "task_1_bowl_0")

    assert state.object_visual.mesh_path == (ROOT / "assets/objects/bowl_visual.obj").resolve()


def test_default_save_keeps_runtime_input_unchanged(tmp_path) -> None:
    """干净检出也能调参；默认保存只写开发目录，不修改运行输入。"""
    for relative in ("data/task_1/knowledge/grasp_profiles.json", "assets/objects/bowl_concept.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    source = tmp_path / "data/task_1/knowledge/grasp_profiles.json"
    before = source.read_bytes()
    editor = EditableGraspSession(tmp_path, "task_1_bowl_0")

    assert editor.output_path == tmp_path / "archive/task_1/configs/grasp_profiles.tuned.json"
    profile = editor.profile_json
    profile["postprocess"] = []
    editor.replace_profile(profile)
    assert editor.save().is_file()
    assert source.read_bytes() == before
    assert not (source.parent / "grasp_profiles.tuned.json").exists()


def test_replace_profile_restores_full_validated_candidate_without_saving(tmp_path) -> None:
    editor = EditableGraspSession(ROOT, "task_1_bowl_0", output_path=tmp_path / "tuned.json")
    before = editor.state.candidates[0].tcp_pose_world.copy()
    profile = editor.profile_json
    profile["postprocess"] = [{"type": "translate", "frame": "object", "xyz_m": [0., -.015, 0.]}]
    restored = editor.replace_profile(profile)

    assert editor.profile_json == profile
    assert not editor.output_path.exists()
    assert not np.allclose(restored.candidates[0].tcp_pose_world, before)
    with pytest.raises(ValueError):
        editor.replace_profile({**profile, "parameters": {"unknown_parameter": {"values": [0]}}})
    assert editor.profile_json == profile
