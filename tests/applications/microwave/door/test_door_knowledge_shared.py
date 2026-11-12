"""门 Knowledge 参数装配迁移后，原 Viewer 的完整阶段目标保持不变。"""

import hashlib
import json
from pathlib import Path

import numpy as np

from gene_agent.tasks.policies.microwave_task.planning.door_session import DoorDebugSession
from gene_agent.tasks.policies.microwave_task.operations.door import build_door_knowledge
from gene_agent.tasks.policies.microwave_task.planning.bowl import prepare_pick_bowl_scene
from gene_agent.realworld.microwave.door_planning import build_observed_door_skill_targets
from gene_agent.realworld.microwave.scene import build_observed_collision_scene
from gene_agent.tasks.policies.microwave_task.planning.door_scene import door_stage_collision_options, export_observed_door_stage_scene
from gene_agent.tasks.policies.microwave_task.operations.door import approximate_ik_budget_for, local_rotation_search_for
from gene_agent.realworld.perception import ObservedScene
from gene_agent.visualization.grasp_tuner.microwave_scene import _body_pose


ROOT = Path(__file__).resolve().parents[4]
EXPECTED = {
    "handle_pull": (9, "74f62d2f751837fe37d3c34197707bca005ceb74f979b230b82ea439c680c428"),
    "push_open": (6, "e358b849185105c7a9cb0aaeb39d5caed5c8862a6ef35455e83fdc161422dc98"),
    "push_close": (9, "75a93567dc13c9bf4ec410922b6d108687e6aacf5363533a1d4c5068cff8841e"),
}


def test_stage_planning_policy_matches_viewer_for_every_phase():
    """配置提取前后，门 Skill 和 Task 过渡阶段的策略保持相同。"""
    viewer = DoorDebugSession()
    for phase in viewer.phases:
        assert approximate_ik_budget_for(viewer.config, phase) == viewer.budget_for(phase.phase_id)
        assert local_rotation_search_for(viewer.config, phase) == viewer.local_rotation_search_for(
            phase.phase_id
        )
    pre = next(p for p in viewer.phases if p.name == "pre_contact")
    overridden = {**viewer.config,
                  "approximate_ik_by_stage": {pre.phase_id: None},
                  "local_rotation_search_by_stage": {pre.phase_id: None}}
    assert approximate_ik_budget_for(overridden, pre) is None
    assert local_rotation_search_for(overridden, pre) is None


def _signature(phases):
    rows = [
        {
            "id": phase.phase_id,
            "pose": [round(float(value), 9)
                     for value in phase.tcp_pose_world.reshape(-1)],
            "angle": round(phase.door_angle_rad, 9),
            "width": round(phase.gripper_width_m, 9),
            "motion": phase.motion_kind,
            "contact": phase.contact_policy,
        }
        for phase in phases
    ]
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return len(rows), hashlib.sha256(payload.encode()).hexdigest()


def test_shared_door_knowledge_preserves_viewer_targets(tmp_path):
    case_dir = tmp_path / "case"
    prepare_pick_bowl_scene(case_dir, seed=0)
    viewer = DoorDebugSession(run_dir=case_dir)
    angles = {
        "handle_pull": (0., float(viewer.config["pull_angle_deg"])),
        "push_open": (float(viewer.config.get("push_start_angle_deg",
                                              viewer.config["pull_angle_deg"])),
                      float(viewer.config["push_angle_deg"])),
        "push_close": (float(viewer.config.get("close_start_angle_deg",
                                               viewer.config["push_angle_deg"])), 0.),
    }
    for kind, (start, target) in angles.items():
        assert _signature([phase for phase in viewer.phases
                           if phase.skill_id == kind]) == EXPECTED[kind]
        shared = build_door_knowledge(
            ROOT, viewer.case, viewer.config, viewer.knowledge_state,
            kind=kind, start_deg=start, target_deg=target,
        )
        displayed = viewer.knowledge[kind]
        assert shared.handle_pull_labels == displayed.handle_pull_labels
        assert shared.door_push_labels == displayed.door_push_labels
        assert shared.door_selected_candidate_ids == displayed.door_selected_candidate_ids
        for left, right in zip(shared.handle_pull_path.waypoints,
                               displayed.handle_pull_path.waypoints, strict=True):
            np.testing.assert_allclose(left.tcp_pose_world, right.tcp_pose_world, atol=1e-10)
        for left, right in zip(shared.door_path.waypoints,
                               displayed.door_path.waypoints, strict=True):
            np.testing.assert_allclose(left.tcp_pose_world, right.tcp_pose_world, atol=1e-10)


def test_observed_door_targets_use_zero_angle_geometry_once(tmp_path):
    """现场门角只用于 Skill 起点和规划碰撞，不再旋转一次门板几何。"""
    case_dir = tmp_path / "case"
    prepare_pick_bowl_scene(case_dir, seed=0)
    viewer = DoorDebugSession(run_dir=case_dir)
    config = json.loads((ROOT / "config/scenes/task_1.json").read_text())
    table = np.eye(4)
    table[:3, 3] = [*config["table"]["center_xy_m"],
                    config["table"]["top_z_m"] - config["table"]["thickness_m"] / 2]
    poses = {
        "Microwave/Microwave_0": _body_pose(viewer.case.data, "microwave"),
        "Bowl/Bowl_0": _body_pose(viewer.case.data, "bowl"),
    }
    observed = ObservedScene(
        "right", "0", 12, 345.0, np.eye(4), poses,
        {"camera": "synthetic", **{name: "synthetic" for name in poses}}, True,
    )
    groups = {
        group: [float(viewer.case.data.qpos[viewer.case.model.joint(name).qposadr[0]])
                for name in names]
        for group, names in viewer.case.metadata["robot"]["joint_groups"].items()
    }
    support = config["microwave_support"]
    for skill_id, start_deg in (("handle_pull", 0.), ("push_open", 25.),
                                ("push_close", 90.)):
        shadow = build_observed_collision_scene(
            observed, T_world_sdk=np.eye(4),
            T_perception_object_scene_asset={name: np.eye(4) for name in poses},
            table_pose_world=table, table_size_xy_m=config["table"]["size_xy_m"],
            accepted_support_size_xyz_m=(*support["size_xy_m"], support["height_m"]),
            accepted_support_center_xy_local_m=support["center_xy_local_m"],
            accepted_table_thickness_m=config["table"]["thickness_m"],
            joint_positions_by_group=groups,
            gripper_widths_m={"left": .08, "right": .08},
            door_angle_rad=np.deg2rad(start_deg), knob_angle_rad=0.,
            turntable_height_offset_m=0., microwave_floor_height_offset_m=0.,
            active_group="right", output_dir=tmp_path / skill_id,
        )
        result = build_observed_door_skill_targets(shadow, skill_id=skill_id)
        old = [phase for phase in viewer.phases if phase.skill_id == skill_id]
        assert _signature(result["phases"]) == EXPECTED[skill_id]
        assert len(result["phases"]) == len(old)
        for actual, expected in zip(result["phases"], old, strict=True):
            np.testing.assert_allclose(actual.tcp_pose_world,
                                       expected.tcp_pose_world, atol=1e-9)
        assert result["observed_door_angle_rad"] == np.deg2rad(start_deg)
        assert result["knowledge_geometry_door_angle_rad"] == 0.
        joint = viewer.case.model.joint("microwave_door_joint")
        assert shadow["snapshot"]["qpos_snapshot"][joint.qposadr[0]] == np.deg2rad(start_deg)

        if skill_id == "handle_pull":
            phase = next(p for p in result["phases"] if p.name == "pregrasp")
            expected = {"disabled_body_names": ("microwave", "microwave_support"),
                        "moving_body_names": ()}
        else:
            phase = next(p for p in result["phases"] if p.name == "contact")
            expected = {"disabled_body_names": (),
                        "moving_body_names": ("microwave_door",)}
        assert door_stage_collision_options(phase) == expected
        stage = export_observed_door_stage_scene(
            shadow, phase, output_dir=tmp_path / f"{skill_id}_stage",
        )
        assert stage["snapshot"]["disabled_body_names"] == list(expected["disabled_body_names"])
        assert stage["snapshot"]["moving_body_names"] == list(expected["moving_body_names"])
        assert stage["provenance"]["parent_collision_snapshot_sha256"] == (
            shadow["snapshot"]["snapshot_sha256"]
        )
        if skill_id == "handle_pull":
            assert not any(row["body_name"] == "microwave" for row in stage["snapshot"]["records"])
        else:
            assert any(row["body_name"] == "microwave" for row in stage["snapshot"]["records"])
            assert not any(row["body_name"] == "microwave_door"
                           for row in stage["snapshot"]["records"])
