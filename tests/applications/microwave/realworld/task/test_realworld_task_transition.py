"""开门后的默认 TCP 撤离属于 Task 过渡，沿用门阶段规划语义。"""

import json
from types import SimpleNamespace

import numpy as np

from gene_agent.realworld.microwave import task as module
from gene_agent.tasks.policies.microwave_task.planning.door_scene import door_stage_collision_options, export_observed_door_stage_scene
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.simulation import build_scene
from gene_agent.skills.library.door.sequence import DoorPhase


def _scene(frame, *, q=(.1, .2), width=.06, door=1.57):
    return {
        "snapshot": {"snapshot_sha256": f"frame-{frame}", "active_group": "right"},
        "provenance": {
            "source_frame_number": frame, "source_timestamp_ms": frame * 100.,
            "joint_positions_by_group_rad": {"right": list(q)},
            "gripper_widths_m": {"right": width}, "door_angle_rad": door,
            "aligned_object_poses_world": {"Microwave/Microwave_0": np.eye(4).tolist()},
        },
        "robot_metadata": {"joint_groups": {"right": ["j1", "j2"]}},
    }


def _config(tmp_path):
    path = tmp_path / "door_config.json"
    path.write_text(json.dumps({"schema_version": 1,
                                "return_default_tcp_non_linear_fallback": True}))
    return path


def _options(tmp_path, observe, runtime, *, execute):
    return dict(
        observe_collision_scene=observe, make_runtime=runtime,
        planner_pool=object(), output_dir=tmp_path / "transition",
        config_path=_config(tmp_path), execute=execute,
        max_start_error_rad=.02, max_gripper_width_error_m=.003,
        max_door_angle_error_rad=.04,
    )


def test_transition_uses_home_tcp_linear_and_door_contact(tmp_path, monkeypatch):
    pose = np.eye(4)
    pose[0, 3] = .3
    monkeypatch.setattr(module, "_home_tcp_pose", lambda scene: pose)
    plans = []

    def plan(scene, targets, *, phase_id, **kwargs):
        phase = targets["phases"][0]
        plans.append((scene, targets, phase_id))
        assert phase.skill_id == "task_transition"
        assert phase.motion_kind == "linear_tcp"
        assert phase.contact_policy == "door_contact"
        assert np.allclose(phase.tcp_pose_world, pose)
        assert door_stage_collision_options(phase)["moving_body_names"] == (
            "microwave_door",)
        return {"success": True, "stage_collision_snapshot_sha256": "stage-collision",
                "skill_stage_ref": "planned-stage.json", "operation": "linear_tcp"}

    monkeypatch.setattr(module, "plan_observed_door_stage", plan)
    monkeypatch.setattr(module, "load_planned_door_skill_stage",
                        lambda path: SimpleNamespace(name="return_default_tcp"))
    calls = []

    class Runtime:
        def __init__(self, scene_hash):
            calls.append(scene_hash)

        def run_motion_stage(self, stage, *, execute):
            calls.append((stage.name, execute))
            return {"source_format": "realworld_motion_stage", "success": True,
                    "status": "validated", "measured_end": None}

    scene = _scene(1)
    result = module.run_observed_door_return_default_tcp(
        **_options(tmp_path, lambda: scene, Runtime, execute=False))
    assert result["status"] == "validated_prefix"
    assert result["success"] is False
    assert calls == ["stage-collision", ("return_default_tcp", False)]
    assert len(plans) == 1


def test_transition_executes_then_checks_new_observed_state(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_home_tcp_pose", lambda scene: np.eye(4))
    monkeypatch.setattr(module, "plan_observed_door_stage", lambda *args, **kwargs: {
        "success": True, "stage_collision_snapshot_sha256": "stage-collision",
        "skill_stage_ref": "planned-stage.json", "operation": "linear_tcp"})
    monkeypatch.setattr(module, "load_planned_door_skill_stage",
                        lambda path: SimpleNamespace(name="return_default_tcp"))

    class Runtime:
        def __init__(self, scene_hash):
            assert scene_hash == "stage-collision"

        def run_motion_stage(self, stage, *, execute):
            assert execute is True
            return {"source_format": "realworld_motion_stage", "success": True,
                    "status": "executed", "measured_end": {
                        "joint_names": ["j1", "j2"], "positions_rad": [.3, .4]}}

    scenes = iter((_scene(1), _scene(2, q=(.3, .4))))
    result = module.run_observed_door_return_default_tcp(
        **_options(tmp_path, lambda: next(scenes), Runtime, execute=True))
    assert result["success"] is True
    assert result["postcondition"]["status"] == "pass"
    assert result["postcondition"]["source"] == "observed_robot_and_door"
    assert (tmp_path / "transition/transition_result.json").is_file()


def test_transition_stops_when_door_moves_during_retreat(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_home_tcp_pose", lambda scene: np.eye(4))
    monkeypatch.setattr(module, "plan_observed_door_stage", lambda *args, **kwargs: {
        "success": True, "stage_collision_snapshot_sha256": "stage-collision",
        "skill_stage_ref": "planned-stage.json"})
    monkeypatch.setattr(module, "load_planned_door_skill_stage",
                        lambda path: SimpleNamespace(name="return_default_tcp"))

    class Runtime:
        def __init__(self, scene_hash):
            pass

        def run_motion_stage(self, stage, *, execute):
            return {"source_format": "realworld_motion_stage", "success": True,
                    "status": "executed", "measured_end": {
                        "joint_names": ["j1", "j2"], "positions_rad": [.3, .4]}}

    scenes = iter((_scene(1), _scene(2, q=(.3, .4), door=1.4)))
    result = module.run_observed_door_return_default_tcp(
        **_options(tmp_path, lambda: next(scenes), Runtime, execute=True))
    assert result["status"] == "failed"
    assert result["first_failure"]["code"] == "door_changed_during_transition"


def test_home_tcp_target_comes_from_robot_home_fk_not_current_arm_pose(tmp_path):
    import mujoco
    from gene_agent.simulation import build_scene

    model, data, metadata = build_scene(
        tmp_path / "asset_scene", seed=0, perturb=False,
        robot_profile="realman_dual_standard",
    )
    tcp_name = metadata["robot"]["tcp_sites"]["right"]
    site = data.site(tcp_name)
    expected = np.eye(4)
    expected[:3, :3] = np.asarray(site.xmat).reshape(3, 3)
    expected[:3, 3] = site.xpos
    joint = model.joint(metadata["robot"]["joint_groups"]["right"][0])
    data.qpos[joint.qposadr[0]] += .2
    mujoco.mj_forward(model, data)
    scene = {
        "scene_xml_path": metadata["xml_path"],
        "snapshot": {"qpos_snapshot": data.qpos.copy(), "active_group": "right"},
        "robot_metadata": metadata["robot"],
    }
    actual = module._home_tcp_pose(scene)
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_before_reopen_return_uses_full_collision_and_regular_plan(tmp_path, monkeypatch):
    """旧完整 Task 在第二次开门前回默认 TCP，必须保留所有场景碰撞。"""
    monkeypatch.setattr(module, "_home_tcp_pose", lambda scene: np.eye(4))
    phases = []

    def plan(_scene, targets, *, phase_id, **_kwargs):
        phase = targets["phases"][0]
        phases.append(phase)
        assert phase_id == "task/before_reopen/return_default_tcp"
        assert phase.motion_kind == "plan"
        assert phase.contact_policy == "avoid_door"
        assert door_stage_collision_options(phase)["moving_body_names"] == ()
        assert door_stage_collision_options(phase)["disabled_body_names"] == ()
        return {"success": True, "stage_collision_snapshot_sha256": "full-scene",
                "skill_stage_ref": "planned-stage.json", "operation": "plan"}

    monkeypatch.setattr(module, "plan_observed_door_stage", plan)
    monkeypatch.setattr(module, "load_planned_door_skill_stage",
                        lambda path: SimpleNamespace(name="return_default_tcp"))

    class Runtime:
        def __init__(self, scene_hash):
            assert scene_hash == "full-scene"

        def run_motion_stage(self, stage, *, execute):
            assert execute is False
            return {"source_format": "realworld_motion_stage", "success": True,
                    "status": "validated", "measured_end": None}

    result = module.run_observed_door_return_default_tcp(
        **_options(tmp_path, lambda: _scene(1, door=0.), Runtime, execute=False),
        phase_id="task/before_reopen/return_default_tcp",
        motion_kind="plan", contact_policy="avoid_door",
    )
    assert result["status"] == "validated_prefix"
    assert result["stage_id"] == "task/before_reopen/return_default_tcp"
    assert len(phases) == 1


def test_before_reopen_waits_after_verified_execution(tmp_path, monkeypatch):
    """旧 Task1 的五秒停顿只能发生在本段实际完成之后。"""
    monkeypatch.setattr(module, "_home_tcp_pose", lambda scene: np.eye(4))
    monkeypatch.setattr(module, "plan_observed_door_stage", lambda *args, **kwargs: {
        "success": True, "stage_collision_snapshot_sha256": "full-scene",
        "skill_stage_ref": "planned-stage.json", "operation": "plan"})
    monkeypatch.setattr(module, "load_planned_door_skill_stage",
                        lambda path: SimpleNamespace(name="return_default_tcp"))
    waits = []
    monkeypatch.setattr(module.time, "sleep", waits.append)

    class Runtime:
        def __init__(self, scene_hash):
            assert scene_hash == "full-scene"

        def run_motion_stage(self, stage, *, execute):
            assert execute is True
            return {"source_format": "realworld_motion_stage", "success": True,
                    "status": "executed", "measured_end": {
                        "joint_names": ["j1", "j2"], "positions_rad": [.3, .4]}}

    scenes = iter((_scene(1, door=0.), _scene(2, q=(.3, .4), door=0.)))
    result = module.run_observed_door_return_default_tcp(
        **_options(tmp_path, lambda: next(scenes), Runtime, execute=True),
        phase_id="task/before_reopen/return_default_tcp",
        motion_kind="plan", contact_policy="avoid_door",
    )
    assert result["success"] is True
    assert result["settle_wait_s"] == 5.0
    assert waits == [5.0]


def test_before_reopen_collision_snapshot_keeps_door_body_and_table(tmp_path):
    """检查真正导出的碰撞资产，不能只相信阶段策略中的空排除名单。"""
    model, data, metadata = build_scene(
        tmp_path / "scene", seed=0, perturb=False,
        robot_profile="realman_dual_standard",
    )
    base = export_collision_scene(
        model, data, metadata["robot"], "right", tmp_path / "base_collision",
    )
    scene = {
        "scene_xml_path": metadata["xml_path"],
        "robot_metadata": metadata["robot"],
        "snapshot": {**base, "qpos_snapshot": data.qpos.copy(), "active_group": "right"},
        "provenance": {
            "physics_stepped": False, "pose_frame_verified": True,
            "collision_snapshot_sha256": base["snapshot_sha256"],
        },
    }
    phase = DoorPhase(
        "task/before_reopen/return_default_tcp", "task_transition",
        "return_default_tcp", np.eye(4), 0., .08, "plan", "avoid_door",
    )
    exported = export_observed_door_stage_scene(
        scene, phase, output_dir=tmp_path / "before_reopen",
    )["snapshot"]
    bodies = {row["body_name"] for row in exported["records"]}
    table_geoms = {row["geom_name"] for row in exported["records"]
                   if row["body_name"] == "world"}
    assert {"microwave", "microwave_door", "microwave_support"} <= bodies
    assert "scene_table" in table_geoms
    assert exported["moving_body_names"] == []
    assert exported["disabled_body_names"] == []
