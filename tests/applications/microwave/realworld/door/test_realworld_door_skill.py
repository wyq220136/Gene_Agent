"""门 Skill 现场编排：只推进实测终态，失败保留已执行前缀。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gene_agent.realworld.microwave import door as door_skill
from gene_agent.skills.library.door.sequence import DoorPhase


def _scene(name, *, q=(0., 0.), width=.06, door=0.):
    frame = {"initial": 10, "after-gripper": 11, "after-pull": 12,
             "after-failed-command": 11, "after": 11}[name]
    return {
        "snapshot": {"snapshot_sha256": name, "active_group": "right",
                     "world_frame": "right_sdk_base"},
        "provenance": {
            "physics_stepped": False, "pose_frame_verified": True,
            "collision_snapshot_sha256": name,
            "aligned_object_poses_world": {
                "Microwave/Microwave_0": np.eye(4).tolist(),
            },
            "scene_config_sha256": "asset-v1",
            "joint_positions_by_group_rad": {"right": list(q)},
            "gripper_widths_m": {"right": width},
            "door_angle_rad": door,
            "source_frame_number": frame, "source_timestamp_ms": float(frame),
        },
        "robot_metadata": {"joint_groups": {"right": ["joint_1", "joint_2"]}},
    }


def _targets():
    pose = np.eye(4)
    phases = (
        DoorPhase("handle_pull/00_open", "handle_pull", "open",
                  pose.copy(), 0., .08, "gripper", "avoid_door"),
        DoorPhase("handle_pull/01_pull_01", "handle_pull", "pull_01",
                  pose.copy(), .5, .08, "linear_tcp", "handle_contact"),
    )
    return {"skill_id": "handle_pull", "phases": phases,
            "collision_snapshot_sha256": "initial",
            "target_anchor": {
                "microwave_pose_world": pose.tolist(),
                "scene_config_sha256": "asset-v1", "active_group": "right",
                "world_frame": "right_sdk_base",
            }}


def _options(tmp_path, observe, make_runtime, *, execute):
    return dict(
        skill_id="handle_pull", observe_collision_scene=observe,
        make_runtime=make_runtime, planner_pool=object(),
        output_dir=tmp_path / "run", config_path=Path("unused.json"),
        execute=execute, max_microwave_translation_m=.005,
        max_microwave_rotation_rad=.02, max_start_error_rad=.02,
        max_gripper_width_error_m=.005, max_door_angle_error_rad=.04,
    )


def test_plan_only_stops_before_using_unexecuted_gripper_state(tmp_path, monkeypatch):
    observations = [_scene("initial")]
    monkeypatch.setattr(door_skill, "build_observed_door_skill_targets",
                        lambda *_args, **_kwargs: _targets())
    monkeypatch.setattr(door_skill, "plan_observed_door_stage",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("不能从未执行的夹爪开口继续规划")))

    class Runtime:
        def run_gripper_stage(self, *_args, **kwargs):
            assert kwargs["execute"] is False
            return {"source_format": "realworld_gripper_stage",
                    "status": "validated", "success": True}

    report = door_skill.run_observed_door_skill(
        **_options(tmp_path, lambda: observations.pop(0),
                   lambda _hash: Runtime(), execute=False)
    )
    assert report["success"] is False
    assert report["status"] == "validated_prefix"
    assert report["first_failure"]["code"] == "not_executed"
    assert len(report["stages"]) == 1
    assert (tmp_path / "run/skill_result.json").is_file()


@pytest.mark.parametrize("last_microwave_shift_m, expected_success", [
    (0.0, True),
    (0.02, False),
])
def test_executed_skill_rebases_each_stage_on_measured_scene(
    tmp_path, monkeypatch, last_microwave_shift_m, expected_success,
):
    observations = [
        _scene("initial"),
        _scene("after-gripper", width=.08),
        _scene("after-pull", q=(.1, .2), width=.08, door=.5),
    ]
    # 最后一段执行后，门角即使达到目标，柜体若被机械臂推走也不能
    # 把单个门 Skill 报为成功；后面已经没有下一段做起点复核。
    observations[-1]["provenance"]["aligned_object_poses_world"][
        "Microwave/Microwave_0"
    ][0][3] = last_microwave_shift_m
    seen = []
    monkeypatch.setattr(door_skill, "build_observed_door_skill_targets",
                        lambda *_args, **_kwargs: _targets())

    def plan(scene, targets, **kwargs):
        seen.append((scene["snapshot"]["snapshot_sha256"],
                     targets["collision_snapshot_sha256"], kwargs["phase_id"]))
        return {"success": True, "stage_collision_snapshot_sha256": "stage-after-gripper",
                "skill_stage_ref": "synthetic-stage.json"}

    monkeypatch.setattr(door_skill, "plan_observed_door_stage", plan)
    monkeypatch.setattr(door_skill, "load_planned_door_skill_stage",
                        lambda _path: SimpleNamespace(name="handle_pull/01_pull_01"))

    class Runtime:
        def __init__(self, snapshot_hash):
            seen.append(("runtime", snapshot_hash))

        def run_gripper_stage(self, *_args, **kwargs):
            assert kwargs["execute"] is True
            return {"source_format": "realworld_gripper_stage",
                    "status": "executed", "success": True,
                    "measured_end_width_m": .08}

        def run_motion_stage(self, _stage, *, execute):
            assert execute is True
            return {"source_format": "realworld_motion_stage",
                    "status": "executed", "success": True,
                    "measured_end": {"joint_names": ["joint_1", "joint_2"],
                                     "positions_rad": [.1, .2]}}

    report = door_skill.run_observed_door_skill(
        **_options(tmp_path, lambda: observations.pop(0), Runtime, execute=True)
    )
    assert report["success"] is expected_success
    assert report["status"] == ("executed" if expected_success else "failed")
    if expected_success:
        assert report["postcondition"]["status"] == "pass"
    else:
        assert report["first_failure"]["code"] == "target_anchor_changed"
    assert seen == [("runtime", "initial"),
                    ("after-gripper", "after-gripper", "handle_pull/01_pull_01"),
                    ("runtime", "stage-after-gripper")]
    assert not observations


def test_failed_gripper_stage_stops_without_planning_next_phase(tmp_path, monkeypatch):
    observations = [_scene("initial"), _scene("after-failed-command", width=.07)]
    monkeypatch.setattr(door_skill, "build_observed_door_skill_targets",
                        lambda *_args, **_kwargs: _targets())
    monkeypatch.setattr(door_skill, "plan_observed_door_stage",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("失败后不能规划下一段")))

    class Runtime:
        def run_gripper_stage(self, *_args, **_kwargs):
            return {"source_format": "realworld_gripper_stage", "status": "failed",
                    "success": False, "failure": {"code": "gripper_start_mismatch",
                                                  "message": "夹爪起点不匹配"}}

    report = door_skill.run_observed_door_skill(
        **_options(tmp_path, lambda: observations.pop(0),
                   lambda _hash: Runtime(), execute=True)
    )
    assert report["status"] == "failed"
    assert report["first_failure"]["code"] == "gripper_start_mismatch"
    assert len(report["stages"]) == 1
    assert report["stages"][0]["observed_after_failure"]["gripper_width_m"] == .07


@pytest.mark.parametrize("skill,name,mode,force", [
    ("handle_pull", "close", "grasp", 1.),
    ("handle_pull", "release", "release", None),
    ("push_close", "close", "position", None),
])
def test_door_gripper_action_preserves_legacy_sdk_semantics(
    tmp_path, monkeypatch, skill, name, mode, force,
):
    """把手力控抓取/松开与推门前位置闭指是不同的硬件动作。"""
    pose = np.eye(4)
    first_width = .04 if name == "release" else .08
    target_width = .08 if name == "release" else 0.
    after_width = .08 if name == "release" else .04
    phase = DoorPhase(f"{skill}/00_{name}", skill, name, pose, 0.,
                      target_width, "gripper", "avoid_door")
    targets = {"skill_id": skill, "phases": (phase,),
               "collision_snapshot_sha256": "initial",
               "target_anchor": {
                   "microwave_pose_world": pose.tolist(),
                   "scene_config_sha256": "asset-v1", "active_group": "right",
                   "world_frame": "right_sdk_base",
               }}
    monkeypatch.setattr(door_skill, "build_observed_door_skill_targets",
                        lambda *_args, **_kwargs: targets)
    scenes = [_scene("initial", width=first_width),
              _scene("after", width=after_width)]
    seen = []

    class Runtime:
        def run_gripper_stage(self, _stage, **kwargs):
            seen.append(kwargs)
            return {"source_format": "realworld_gripper_stage", "status": "executed",
                    "success": True, "measured_end_width_m": after_width}

    options = _options(tmp_path, lambda: scenes.pop(0),
                       lambda _hash: Runtime(), execute=True)
    options["skill_id"] = skill
    door_skill.run_observed_door_skill(**options)
    assert seen[0].get("command_mode", "position") == mode
    if force is not None:
        assert seen[0]["grasp_force"] == force
