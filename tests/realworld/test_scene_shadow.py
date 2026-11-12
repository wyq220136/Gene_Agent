"""用实测状态驱动现有碰撞代理，不运行 MuJoCo 动力学。"""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from gene_agent.realworld.perception import ObservedScene
from gene_agent.realworld.planning import prepare_plan_only_context
from gene_agent.realworld.microwave.scene import build_observed_collision_scene
from gene_agent.motion.curobo.session import CuroboPlannerPool
from gene_agent.skills.runtime.planning import SkillContext


ROOT = Path(__file__).resolve().parents[2]


def _fixture():
    config = json.loads((ROOT / "config/scenes/task_1.json").read_text())
    microwave = np.asarray(config["objects"]["microwave"]["simulation_nominal_pose"], dtype=float)
    bowl = np.asarray(config["objects"]["bowl"]["simulation_nominal_pose"], dtype=float).copy()
    bowl[0, 3] += 0.07
    table = np.eye(4)
    table[:3, 3] = [*config["table"]["center_xy_m"],
                    config["table"]["top_z_m"] - config["table"]["thickness_m"] / 2]
    scene = ObservedScene(
        side="right", camera_id="0", frame_number=123, timestamp_ms=456.0,
        T_sdk_camera=np.eye(4),
        object_poses_sdk={"Bowl/Bowl_0": bowl, "Microwave/Microwave_0": microwave},
        sources={"camera": "synthetic-frame-123", "Bowl/Bowl_0": "synthetic-bowl",
                 "Microwave/Microwave_0": "synthetic-microwave"},
        pose_frame_verified=True,
    )
    joints = {}
    for group in ("left", "right"):
        payload = json.loads((ROOT / f"config/robots/realman_{group}_default_pose.json").read_text())
        joints[group] = payload["joints_rad"]
    support = config["microwave_support"]
    return config, scene, table, joints, (
        *support["size_xy_m"], support["height_m"],
    )


def test_observed_collision_scene_uses_measured_poses_and_joint_angles(tmp_path):
    """若又使用仿真名义碗位姿、遗漏另一臂或给门瞬移错误角度，此测试失败。"""
    config, observed, table, joints, support = _fixture()
    result = build_observed_collision_scene(
        observed, T_world_sdk=np.eye(4),
        T_perception_object_scene_asset={name: np.eye(4) for name in observed.object_poses_sdk},
        table_pose_world=table, table_size_xy_m=tuple(config["table"]["size_xy_m"]),
        accepted_support_size_xyz_m=support,
        accepted_support_center_xy_local_m=config["microwave_support"]["center_xy_local_m"],
        accepted_table_thickness_m=config["table"]["thickness_m"],
        joint_positions_by_group=joints,
        gripper_widths_m={"left": 0.08, "right": 0.04},
        door_angle_rad=0.3, knob_angle_rad=0.1,
        turntable_height_offset_m=0.0, microwave_floor_height_offset_m=0.0,
        active_group="right", output_dir=tmp_path / "observed",
    )

    snapshot = result["snapshot"]
    assert snapshot["active_group"] == "right"
    assert snapshot["scene_model"]["cuboid"] or snapshot["scene_model"]["mesh"]
    assert any(row["body_name"] == "bowl" for row in snapshot["records"])
    assert any(row["body_name"] == "microwave_support" for row in snapshot["records"])
    assert any(row["geom_name"] == "scene_table" for row in snapshot["records"])
    model = mujoco.MjModel.from_xml_path(str(result["scene_xml_path"]))
    bowl_bottom = next(row for row in snapshot["records"]
                       if row["geom_name"] == "bowl_bottom")
    expected_bottom = observed.object_poses_sdk["Bowl/Bowl_0"] @ np.array(
        [0, config["bowl_proxy"]["bottom_y_m"], 0, 1]
    )
    assert np.asarray(bowl_bottom["T_world_geom"])[:3, 3] == pytest.approx(expected_bottom[:3])
    table_geom = next(row for row in snapshot["records"]
                      if row["geom_name"] == "scene_table")
    assert np.asarray(table_geom["T_world_geom"])[:3, 3] == pytest.approx(table[:3, 3])
    door_adr = model.joint("microwave_door_joint").qposadr[0]
    knob_adr = model.joint("knob_0_joint").qposadr[0]
    assert snapshot["qpos_snapshot"][door_adr] == pytest.approx(0.3)
    assert snapshot["qpos_snapshot"][knob_adr] == pytest.approx(0.1)
    for group in ("left", "right"):
        for index, value in enumerate(joints[group], start=1):
            address = model.joint(f"{group}_joint_{index}").qposadr[0]
            assert snapshot["qpos_snapshot"][address] == pytest.approx(value)
        width = 0.08 if group == "left" else 0.04
        for side in ("positive", "negative"):
            address = model.joint(f"{group}_gripper_{side}_slide").qposadr[0]
            assert snapshot["qpos_snapshot"][address] == pytest.approx(width / 2)
    assert result["provenance"]["physics_stepped"] is False
    assert result["provenance"]["source_frame_number"] == 123
    assert result["provenance"]["aligned_object_poses_world"]["Bowl/Bowl_0"][0][3] == pytest.approx(
        observed.object_poses_sdk["Bowl/Bowl_0"][0, 3]
    )


def test_unverified_observation_cannot_create_planning_collision_snapshot(tmp_path):
    """历史 pose 缺少同帧来源时，不能假装是当前真机障碍。"""
    config, observed, table, joints, support = _fixture()
    unverified = ObservedScene(
        observed.side, observed.camera_id, observed.frame_number,
        observed.timestamp_ms, observed.T_sdk_camera, observed.object_poses_sdk,
        observed.sources, False,
    )
    output = tmp_path / "unverified"
    with pytest.raises(ValueError, match="同帧"):
        build_observed_collision_scene(
            unverified, T_world_sdk=np.eye(4),
            T_perception_object_scene_asset={name: np.eye(4) for name in unverified.object_poses_sdk},
            table_pose_world=table, table_size_xy_m=tuple(config["table"]["size_xy_m"]),
            accepted_support_size_xyz_m=support,
            accepted_support_center_xy_local_m=config["microwave_support"]["center_xy_local_m"],
            accepted_table_thickness_m=config["table"]["thickness_m"],
            joint_positions_by_group=joints,
            gripper_widths_m={"left": 0.08, "right": 0.04},
            door_angle_rad=0.3, knob_angle_rad=0.1,
            turntable_height_offset_m=0.0, microwave_floor_height_offset_m=0.0,
            active_group="right", output_dir=output,
        )
    assert not output.exists()


def test_missing_other_arm_and_unapproved_support_are_rejected_before_export(tmp_path):
    """另一臂不能沿用 home；支撑尺寸不能未经核对而使用仿真默认值。"""
    config, observed, table, joints, support = _fixture()
    shared = dict(
        T_world_sdk=np.eye(4),
        T_perception_object_scene_asset={name: np.eye(4)
                                         for name in observed.object_poses_sdk},
        table_pose_world=table,
        table_size_xy_m=config["table"]["size_xy_m"],
        accepted_support_size_xyz_m=support,
        accepted_support_center_xy_local_m=config["microwave_support"]["center_xy_local_m"],
        accepted_table_thickness_m=config["table"]["thickness_m"],
        joint_positions_by_group=joints,
        gripper_widths_m={"left": 0.08, "right": 0.04},
        door_angle_rad=0.3, knob_angle_rad=0.1,
        turntable_height_offset_m=0.0, microwave_floor_height_offset_m=0.0,
        active_group="right",
    )
    missing_arm = tmp_path / "missing_arm"
    with pytest.raises(ValueError, match="另一臂"):
        build_observed_collision_scene(
            observed, output_dir=missing_arm,
            **(shared | {"joint_positions_by_group": {"right": joints["right"]}}),
        )
    assert not missing_arm.exists()

    wrong_support = tmp_path / "wrong_support"
    with pytest.raises(ValueError, match="支撑尺寸"):
        build_observed_collision_scene(
            observed, output_dir=wrong_support,
            **(shared | {"accepted_support_size_xyz_m": (0.9, 0.9, 0.16)}),
        )
    assert not wrong_support.exists()


def test_door_angle_outside_asset_joint_limit_is_rejected(tmp_path):
    config, observed, table, joints, support = _fixture()
    with pytest.raises(ValueError, match="microwave_door_joint.*超出模型限位"):
        build_observed_collision_scene(
            observed, T_world_sdk=np.eye(4),
            T_perception_object_scene_asset={name: np.eye(4)
                                             for name in observed.object_poses_sdk},
            table_pose_world=table, table_size_xy_m=config["table"]["size_xy_m"],
            accepted_support_size_xyz_m=support,
            accepted_support_center_xy_local_m=config["microwave_support"]["center_xy_local_m"],
            accepted_table_thickness_m=config["table"]["thickness_m"],
            joint_positions_by_group=joints,
            gripper_widths_m={"left": 0.08, "right": 0.04},
            door_angle_rad=3.0, knob_angle_rad=0.1,
            turntable_height_offset_m=0.0, microwave_floor_height_offset_m=0.0,
            active_group="right", output_dir=tmp_path / "invalid_door",
        )


def test_observed_snapshot_connects_to_existing_curobo_settings_without_gpu(tmp_path):
    """真实场景导出后的 gripper 球配置能进入现有 planner 创建入口。"""
    config, observed, table, joints, support = _fixture()
    scene = build_observed_collision_scene(
        observed, T_world_sdk=np.eye(4),
        T_perception_object_scene_asset={name: np.eye(4) for name in observed.object_poses_sdk},
        table_pose_world=table, table_size_xy_m=config["table"]["size_xy_m"],
        accepted_support_size_xyz_m=support,
        accepted_support_center_xy_local_m=config["microwave_support"]["center_xy_local_m"],
        accepted_table_thickness_m=config["table"]["thickness_m"],
        joint_positions_by_group=joints,
        gripper_widths_m={"left": 0.08, "right": 0.04},
        door_angle_rad=0.3, knob_angle_rad=0.1,
        turntable_height_offset_m=0.0, microwave_floor_height_offset_m=0.0,
        active_group="right", output_dir=tmp_path / "observed",
    )
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(
            planner_mode=kwargs["planner_mode"], random_seed=kwargs["random_seed"],
            scene_scope=kwargs["scene_scope"],
            scene_snapshot_sha256=kwargs["scene_snapshot_sha256"],
            joint_names=kwargs["joint_names"],
        )

    pool = CuroboPlannerPool(planner_factory=factory)
    context = prepare_plan_only_context(
        scene, group="right", planner_pool=pool, scene_revision=1,
    )
    assert len(created) == 1
    assert context["start"].positions_rad == pytest.approx(joints["right"])
    assert context["scene"].snapshot_sha256 == scene["snapshot"]["snapshot_sha256"]
    assert created[0]["gripper_snapshot"]["snapshot_id"] == (
        scene["snapshot"]["gripper_snapshot"]["snapshot_id"]
    )
    assert created[0]["scene_model"] is scene["snapshot"]["scene_model"]
    skill_context = SkillContext(
        group="right", start=context["start"], scene=context["scene"],
        object_id="Bowl/Bowl_0",
    )
    assert skill_context.scene.snapshot_sha256 == scene["snapshot"]["snapshot_sha256"]


def test_observed_bowl_reuses_visual_grasp_collision_policy(tmp_path):
    """现场开爪严格/目标接触两层与旧抓碗共用同一个导出函数。"""
    from gene_agent.realworld.microwave.pick_planning import build_observed_bowl_grasp_stage_snapshots

    config, observed, table, joints, support = _fixture()
    scene = build_observed_collision_scene(
        observed, T_world_sdk=np.eye(4),
        T_perception_object_scene_asset={name: np.eye(4)
                                         for name in observed.object_poses_sdk},
        table_pose_world=table, table_size_xy_m=config["table"]["size_xy_m"],
        accepted_support_size_xyz_m=support,
        accepted_support_center_xy_local_m=config["microwave_support"]["center_xy_local_m"],
        accepted_table_thickness_m=config["table"]["thickness_m"],
        joint_positions_by_group=joints,
        gripper_widths_m={"left": 0.08, "right": 0.04},
        door_angle_rad=0.3, knob_angle_rad=0.1,
        turntable_height_offset_m=0.0, microwave_floor_height_offset_m=0.0,
        active_group="right", output_dir=tmp_path / "observed",
    )
    original_qpos = tuple(scene["snapshot"]["qpos_snapshot"])
    strict, contact = build_observed_bowl_grasp_stage_snapshots(
        scene, open_width_m=0.041, output_dir=tmp_path / "bowl_grasp",
    )
    assert any(row["body_name"] == "bowl" for row in strict["records"])
    assert not any(row["body_name"] == "bowl" for row in contact["records"])
    assert tuple(scene["snapshot"]["qpos_snapshot"]) == original_qpos
    assert strict["gripper_snapshot"]["group"] == "right"
    model = mujoco.MjModel.from_xml_path(str(scene["scene_xml_path"]))
    for name in scene["robot_metadata"]["grippers"]["right"]["joint_names"]:
        address = model.joint(name).qposadr[0]
        assert strict["qpos_snapshot"][address] == pytest.approx(0.041 / 2)
        assert contact["qpos_snapshot"][address] == pytest.approx(0.041 / 2)


def test_observed_bowl_plan_only_calls_original_pick_skill_selection(tmp_path, monkeypatch):
    """现场入口复用已验证的 PickSkill 规划，保留同帧姿态和旧 profile。"""
    from gene_agent.realworld.microwave import pick_planning as realworld_bowl

    config, observed, table, joints, support = _fixture()
    scene = build_observed_collision_scene(
        observed, T_world_sdk=np.eye(4),
        T_perception_object_scene_asset={name: np.eye(4)
                                         for name in observed.object_poses_sdk},
        table_pose_world=table, table_size_xy_m=config["table"]["size_xy_m"],
        accepted_support_size_xyz_m=support,
        accepted_support_center_xy_local_m=config["microwave_support"]["center_xy_local_m"],
        accepted_table_thickness_m=config["table"]["thickness_m"],
        joint_positions_by_group=joints,
        gripper_widths_m={"left": 0.08, "right": 0.04},
        door_angle_rad=0.3, knob_angle_rad=0.1,
        turntable_height_offset_m=0.0, microwave_floor_height_offset_m=0.0,
        active_group="right", output_dir=tmp_path / "observed",
    )
    seen = {}

    class StubPlan:
        selected_candidate_id = "bowl_ring_02_00"

        def stage(self, name):
            return SimpleNamespace(status="planned" if name in {
                "open", "pregrasp", "approach", "close",
            } else "not_executed")

        def to_dict(self):
            return {"skill": "pick", "selected_candidate_id": self.selected_candidate_id,
                    "stages": [{"name": name, "status": self.stage(name).status}
                               for name in ("open", "pregrasp", "approach", "close", "lift", "hold")]}

    def old_selection(case, settings, *, seed):
        seen["seed"] = seed
        seen["settings"] = settings
        seen["case"] = case
        assert len(case.candidates) == 40
        assert case.candidates[0]["candidate_id"] == "bowl_ring_00_00"
        assert case.snapshot["snapshot_sha256"] == scene["snapshot"]["snapshot_sha256"]
        np.testing.assert_allclose(case.data.qpos, scene["snapshot"]["qpos_snapshot"])
        np.testing.assert_allclose(case.bowl_pose, observed.object_poses_sdk["Bowl/Bowl_0"])
        return SimpleNamespace(
            plan=StubPlan(),
            strict_snapshot={"snapshot_sha256": "strict"},
            contact_snapshot={"snapshot_sha256": "contact"},
            proposals=(),
        )

    monkeypatch.setattr(realworld_bowl, "plan_bowl_grasp_stages", old_selection)
    report = realworld_bowl.plan_observed_bowl_grasp_stages(
        scene, output_dir=tmp_path / "plan", seed=321,
    )
    assert seen["seed"] == 321
    assert seen["settings"].approach_distance_m == pytest.approx(0.05)
    assert seen["settings"].candidate_selection == "legacy_ik"
    assert report["grasp_prefix_planned"] is True
    assert report["physical_pick_success"] is False
    assert report["stage_collision_snapshot_sha256"] == {
        "pregrasp": "strict", "approach": "contact",
    }
    assert report["source_frame_number"] == observed.frame_number
    assert report["active_group"] == "right"
    assert report["world_frame"] == scene["snapshot"]["world_frame"]
    assert report["scene_config_sha256"] == scene["provenance"]["scene_config_sha256"]
    np.testing.assert_allclose(report["bowl_pose_world"], observed.object_poses_sdk["Bowl/Bowl_0"])
    assert json.loads((tmp_path / "plan" / "selection.json").read_text()) == report


def test_observed_bowl_stage_real_export_and_motion_adapter_without_gpu(tmp_path):
    """真实碰撞导出经原 cuRobo 适配器进入单段规划；后端仅用 CPU 替身。"""
    import hashlib
    from gene_agent.realworld.microwave.pick_planning import plan_observed_bowl_motion_stage
    from gene_agent.motion.model import (
        CollisionAllowance, CollisionPolicy, JointState, MotionOperation,
        MotionRequest, PlanningSceneRef, PoseTarget,
    )

    config, observed, table, joints, support = _fixture()
    scene = build_observed_collision_scene(
        observed, T_world_sdk=np.eye(4),
        T_perception_object_scene_asset={name: np.eye(4)
                                         for name in observed.object_poses_sdk},
        table_pose_world=table, table_size_xy_m=config["table"]["size_xy_m"],
        accepted_support_size_xyz_m=support,
        accepted_support_center_xy_local_m=config["microwave_support"]["center_xy_local_m"],
        accepted_table_thickness_m=config["table"]["thickness_m"],
        joint_positions_by_group=joints,
        gripper_widths_m={"left": 0.08, "right": 0.04},
        door_angle_rad=0.3, knob_angle_rad=0.1,
        turntable_height_offset_m=0.0, microwave_floor_height_offset_m=0.0,
        active_group="right", output_dir=tmp_path / "observed",
    )
    names = tuple(scene["robot_metadata"]["joint_groups"]["right"])
    original = MotionRequest(
        "bowl_ring_02_00:approach", MotionOperation.LINEAR_TCP, "right",
        JointState(names, tuple(joints["right"])), PoseTarget("world_tcp", np.eye(4)),
        PlanningSceneRef(0, "old-contact", "old"),
        CollisionPolicy("finger_object_contact", (
            CollisionAllowance("gripper_fingers", "bowl", "预期指垫接触"),
        )), seed=123,
    )
    selection = {
        "grasp_prefix_planned": True, "selected_candidate_id": "bowl_ring_02_00",
        "bowl_pose_world": observed.object_poses_sdk["Bowl/Bowl_0"].tolist(),
        "scene_config_sha256": scene["provenance"]["scene_config_sha256"],
        "active_group": "right", "world_frame": scene["snapshot"]["world_frame"],
        "skill_profile_sha256": hashlib.sha256(
            (ROOT / "data/task_1/skills/skill_profiles.json").read_bytes()).hexdigest(),
        "plan": {"stages": [
            {"name": "open", "status": "planned", "details": {"width_m": .041}},
            {"name": "approach", "status": "planned", "request": original.to_dict()},
        ]},
    }
    calls = []

    def factory(**kwargs):
        class FakeRawPlanner:
            planner_mode = kwargs["planner_mode"]
            random_seed = kwargs["random_seed"]
            scene_scope = kwargs["scene_scope"]
            scene_snapshot_sha256 = kwargs["scene_snapshot_sha256"]
            joint_names = tuple(kwargs["joint_names"])

            def plan_linear_tcp(self, request_names, q, target, *, target_frame):
                calls.append((tuple(request_names), tuple(q), target_frame))
                return {"success": True, "status": "success", "duration_s": .01,
                        "positions_rad": [list(q), list(np.asarray(q) + .001)],
                        "interpolation_dt_s": .1}

        return FakeRawPlanner()

    result = plan_observed_bowl_motion_stage(
        scene, selection, stage_name="approach",
        planner_pool=CuroboPlannerPool(planner_factory=factory),
        output_dir=tmp_path / "stage", max_bowl_translation_m=.005,
        max_bowl_rotation_rad=.02, max_gripper_width_error_m=.002,
    )
    assert result["success"] is True
    assert result["stage"].request.start.positions_rad == pytest.approx(joints["right"])
    assert result["stage"].request.operation is MotionOperation.LINEAR_TCP
    assert result["stage"].request.collision_policy == original.collision_policy
    assert calls == [(names, tuple(joints["right"]), "world_tcp")]
    snapshot = json.loads((tmp_path / "stage/collision_grasp_open/target_contact/collision_snapshot.json").read_text())
    assert snapshot["snapshot_sha256"] == result["stage_collision_snapshot_sha256"]
    assert not any(row["body_name"] == "bowl" for row in snapshot["records"])


def test_confirmed_bowl_grasp_transfers_world_obstacle_to_tool_spheres(tmp_path):
    """只有现场确认夹持后，才把 Bowl 从 world 移到末端球代理。"""
    from gene_agent.realworld.microwave.attachment import prepare_observed_bowl_attachment

    config, observed, table, joints, support = _fixture()
    scene = build_observed_collision_scene(
        observed, T_world_sdk=np.eye(4),
        T_perception_object_scene_asset={name: np.eye(4)
                                         for name in observed.object_poses_sdk},
        table_pose_world=table, table_size_xy_m=config["table"]["size_xy_m"],
        accepted_support_size_xyz_m=support,
        accepted_support_center_xy_local_m=config["microwave_support"]["center_xy_local_m"],
        accepted_table_thickness_m=config["table"]["thickness_m"],
        joint_positions_by_group=joints,
        gripper_widths_m={"left": 0.08, "right": 0.007},
        door_angle_rad=0.3, knob_angle_rad=0.1,
        turntable_height_offset_m=0.0, microwave_floor_height_offset_m=0.0,
        active_group="right", output_dir=tmp_path / "observed",
    )
    original_hash = scene["snapshot"]["snapshot_sha256"]
    grasp = {"execute": True, "status": "executed", "grasp_success": True,
             "selected_candidate_id": "bowl_ring_02_00",
             "physical_grasp_check": {"status": "pass", "source": "manual_terminal",
                                      "evidence": {"answer": "y"}, "reason": "已夹住"}}
    result = prepare_observed_bowl_attachment(
        scene, grasp, output_dir=tmp_path / "attached",
    )
    snapshot = result["snapshot"]
    assert scene["snapshot"]["snapshot_sha256"] == original_hash
    assert snapshot["snapshot_sha256"] != original_hash
    assert not any(row["body_name"] == "bowl" for row in snapshot["records"])
    assert "bowl" in snapshot["transferred_objects"]
    world = np.asarray(result["spheres_world_m"])
    tool = np.asarray(result["spheres_tool_m"])
    assert world.shape == tool.shape == (5, 4)
    T_world_tool = np.asarray(result["T_world_tool"])
    np.testing.assert_allclose(
        (T_world_tool[:3, :3] @ tool[:, :3].T).T + T_world_tool[:3, 3],
        world[:, :3], atol=1e-8,
    )
    assert result["physical_hold_evidence"]["source"] == "manual_terminal"
    assert result["provenance"]["physics_stepped"] is False
    fresh = {"status": "pass", "source": "manual_terminal",
             "evidence": {"answer": "still-held"}, "reason": "放碗前复核"}
    refreshed = prepare_observed_bowl_attachment(
        scene, grasp, output_dir=tmp_path / "attached_fresh", hold_evidence=fresh,
    )
    assert refreshed["physical_hold_evidence"] == fresh
    from gene_agent.realworld.microwave.attachment import make_manual_held_bowl_report
    from gene_agent.tasks.checks import CheckResult
    manual = make_manual_held_bowl_report(
        scene, CheckResult("pass", "manual_terminal", {"answer": "prepared"}),
    )
    manual_attachment = prepare_observed_bowl_attachment(
        scene, manual, output_dir=tmp_path / "attached_manual",
    )
    assert manual_attachment["hold_origin"] == "manual_prepared_held_bowl"
    assert manual_attachment["selected_candidate_id"] is None
    assert manual_attachment["physical_hold_evidence"] == manual["postcondition"]
    with pytest.raises(ValueError, match="仿真"):
        prepare_observed_bowl_attachment(
            scene, grasp, output_dir=tmp_path / "rejected_weld",
            hold_evidence={"status": "pass", "source": "simulation_privileged_weld"},
        )

    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(
            planner_mode=kwargs["planner_mode"], random_seed=kwargs["random_seed"],
            scene_scope=kwargs["scene_scope"],
            scene_snapshot_sha256=kwargs["scene_snapshot_sha256"],
            joint_names=kwargs["joint_names"],
        )

    context = prepare_plan_only_context(
        result, group="right", planner_pool=CuroboPlannerPool(planner_factory=factory),
        attachment_spheres_tool=result["spheres_tool_m"],
    )
    assert context["scene"].snapshot_sha256 == snapshot["snapshot_sha256"]
    assert created[0]["attachment_spheres_tool"] == result["spheres_tool_m"]
    assert context["start"].positions_rad == pytest.approx(joints["right"])


def test_unconfirmed_bowl_grasp_cannot_create_attachment(tmp_path):
    from gene_agent.realworld.microwave.attachment import prepare_observed_bowl_attachment
    scene = {"snapshot": {"snapshot_sha256": "scene"},
             "provenance": {"collision_snapshot_sha256": "scene"}}
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="夹持"):
        prepare_observed_bowl_attachment(
            scene, {"execute": True, "status": "unknown", "grasp_success": False,
                    "physical_grasp_check": {"status": "unknown", "source": "none"}},
            output_dir=output,
        )
    assert not output.exists()


def test_extract_attachment_keeps_legacy_32_surface_spheres(tmp_path):
    """取碗撤离必须覆盖旧 mesh 表面；默认抓碗的 5 球配置仍由上例守护。"""
    from gene_agent.realworld.microwave.attachment import prepare_observed_bowl_attachment

    config, observed, table, joints, support = _fixture()
    scene = build_observed_collision_scene(
        observed, T_world_sdk=np.eye(4),
        T_perception_object_scene_asset={name: np.eye(4)
                                         for name in observed.object_poses_sdk},
        table_pose_world=table, table_size_xy_m=config["table"]["size_xy_m"],
        accepted_support_size_xyz_m=support,
        accepted_support_center_xy_local_m=config["microwave_support"]["center_xy_local_m"],
        accepted_table_thickness_m=config["table"]["thickness_m"],
        joint_positions_by_group=joints,
        gripper_widths_m={"left": .08, "right": .007},
        door_angle_rad=1.5, knob_angle_rad=0.,
        turntable_height_offset_m=0., microwave_floor_height_offset_m=0.,
        active_group="right", output_dir=tmp_path / "observed",
    )
    grasp = {"execute": True, "status": "executed", "grasp_success": True,
             "selected_candidate_id": "fixed_keypose_0",
             "physical_grasp_check": {
                 "status": "pass", "source": "manual_terminal",
                 "evidence": {"answer": "held"},
             }}
    result = prepare_observed_bowl_attachment(
        scene, grasp, output_dir=tmp_path / "attached",
        sphere_source="legacy_extract_surface_32",
    )
    world = np.asarray(result["spheres_world_m"])
    tool = np.asarray(result["spheres_tool_m"])
    assert world.shape == tool.shape == (32, 4)
    np.testing.assert_allclose(world[:, 3], .002)
    assert result["coverage"] == "legacy_extract_surface_32"
    assert not any(row["body_name"] == "bowl" for row in result["snapshot"]["records"])
    np.testing.assert_allclose(
        (np.asarray(result["T_world_tool"])[:3, :3] @ tool[:, :3].T).T
        + np.asarray(result["T_world_tool"])[:3, 3],
        world[:, :3], atol=1e-8,
    )
