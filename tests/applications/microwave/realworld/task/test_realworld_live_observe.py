"""现场动态读数必须进入同一份规划碰撞快照。"""

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from gene_agent.motion.model import JointState
from gene_agent.realworld.perception import ObservedScene
from gene_agent.realworld.scene_drift import SceneDriftLimits, compare_scene_drift
from gene_agent.realworld.microwave.scene import build_live_microwave_collision_scene


ROOT = Path(__file__).resolve().parents[5]


def _inputs(tmp_path):
    config = json.loads((ROOT / "config/scenes/task_1.json").read_text())
    poses = {
        object_id: np.asarray(config["objects"][name]["simulation_nominal_pose"], dtype=float)
        for object_id, name in (("Bowl/Bowl_0", "bowl"),
                                ("Microwave/Microwave_0", "microwave"))
    }
    observed = ObservedScene(
        "right", "0", 73, 999500., np.eye(4), poses,
        {"camera": "live-frame-73", **{name: "tracking-frame-73" for name in poses}},
        True,
    )
    table = np.eye(4)
    table[:3, 3] = [*config["table"]["center_xy_m"],
                    config["table"]["top_z_m"] - config["table"]["thickness_m"] / 2]
    scene_parameters = {
        "T_world_sdk": np.eye(4),
        "T_perception_object_scene_asset": {name: np.eye(4) for name in poses},
        "table_pose_world": table,
        "table_size_xy_m": config["table"]["size_xy_m"],
        "accepted_support_size_xyz_m": [*config["microwave_support"]["size_xy_m"],
                                         config["microwave_support"]["height_m"]],
        "accepted_support_center_xy_local_m": config["microwave_support"]["center_xy_local_m"],
        "accepted_table_thickness_m": config["table"]["thickness_m"],
        "active_group": "right", "output_dir": tmp_path / "live",
    }
    robots = {}
    grippers = {}
    for group in ("left", "right"):
        joints = json.loads((ROOT / f"config/robots/realman_{group}_default_pose.json").read_text())["joints_rad"]
        state = JointState(tuple(f"{group}_joint_{i}" for i in range(1, 8)), tuple(joints))
        robots[group] = SimpleNamespace(read_joint_state=lambda state=state: state)
        width = .04 if group == "right" else .08
        grippers[group] = SimpleNamespace(read_width_m=lambda width=width: width)
    angles = {
        "door": {"angle_rad": .3, "observed_at_unix_ms": 999800., "source": "operator-checked-door"},
        "knob": {"angle_rad": .1, "observed_at_unix_ms": 999900., "source": "operator-checked-knob"},
    }
    return observed, robots, grippers, angles, scene_parameters


def test_live_scene_uses_robot_gripper_and_articulated_readings(tmp_path, monkeypatch):
    """如果退回仿真默认关节或把门角写死，qpos 和来源证据会变错。"""
    from gene_agent.realworld.microwave import scene as module
    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    observed, robots, grippers, angles, static = _inputs(tmp_path)
    result = build_live_microwave_collision_scene(
        observed=observed, robots=robots, grippers=grippers,
        angles=angles, scene_parameters=static, max_age_s=2.,
        microwave_pose_mesh_path="microwave/meshes/obj_mesh.obj",
    )
    model = mujoco.MjModel.from_xml_path(str(result["scene_xml_path"]))
    qpos = result["snapshot"]["qpos_snapshot"]
    assert qpos[model.joint("microwave_door_joint").qposadr[0]] == pytest.approx(.3)
    assert qpos[model.joint("knob_0_joint").qposadr[0]] == pytest.approx(.1)
    assert qpos[model.joint("right_gripper_positive_slide").qposadr[0]] == pytest.approx(.02)
    assert result["provenance"]["live_state"]["door"]["source"] == "operator-checked-door"
    assert result["provenance"]["live_state"]["robot_sampled_at_unix_ms"] == 1000000.


def test_live_scene_rejects_stale_door_reading_before_scene_export(tmp_path, monkeypatch):
    """上一轮门角不能流入新一轮碰撞场景。"""
    from gene_agent.realworld.microwave import scene as module
    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    observed, robots, grippers, angles, static = _inputs(tmp_path)
    angles["door"]["observed_at_unix_ms"] = 996000.
    with pytest.raises(ValueError, match="门角.*过期"):
        build_live_microwave_collision_scene(
            observed=observed, robots=robots, grippers=grippers,
            angles=angles, scene_parameters=static, max_age_s=2.,
            microwave_pose_mesh_path="microwave/meshes/obj_mesh.obj",
        )
    assert not static["output_dir"].exists()


def test_live_scene_rejects_missing_other_arm(tmp_path, monkeypatch):
    """单臂活动时另一臂仍是碰撞物，不能默用 home。"""
    from gene_agent.realworld.microwave import scene as module
    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    observed, robots, grippers, angles, static = _inputs(tmp_path)
    del robots["left"]
    with pytest.raises(ValueError, match="两臂"):
        build_live_microwave_collision_scene(
            observed=observed, robots=robots, grippers=grippers,
            angles=angles, scene_parameters=static, max_age_s=2.,
            microwave_pose_mesh_path="microwave/meshes/obj_mesh.obj",
        )
    assert not static["output_dir"].exists()


def test_open_door_rejects_old_handle_mesh_pose_as_microwave_root(tmp_path, monkeypatch):
    """旧 Task1 开门后重新跟踪把手；把该 pose 当机身会移动全部碰撞。"""
    from gene_agent.realworld.microwave import scene as module
    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    observed, robots, grippers, angles, static = _inputs(tmp_path)
    with pytest.raises(ValueError, match="把手 mesh.*微波炉本体"):
        build_live_microwave_collision_scene(
            observed=observed, robots=robots, grippers=grippers,
            angles=angles, scene_parameters=static, max_age_s=2.,
            microwave_pose_mesh_path="microwave/meshes/handle_mesh.obj",
        )
    assert not static["output_dir"].exists()


def test_open_door_uses_verified_stationary_body_anchor_not_handle_pose(tmp_path, monkeypatch):
    """旧把手随门运动时，碰撞中的静止柜体仍须在关门时的真实位置。"""
    from gene_agent.realworld.microwave import scene as module
    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    observed, robots, grippers, angles, static = _inputs(tmp_path)
    root = observed.object_poses_sdk["Microwave/Microwave_0"].copy()
    handle = root.copy()
    handle[0, 3] += .25
    observed.object_poses_sdk["Microwave/Microwave_0"] = handle
    anchor = {"T_world_scene_asset": root.tolist(),
              "verified_at_unix_ms": 999900.,
              "source": "operator-confirmed-cabinet-stationary"}
    result = build_live_microwave_collision_scene(
        observed=observed, robots=robots, grippers=grippers,
        angles=angles, scene_parameters=static, max_age_s=2.,
        microwave_pose_mesh_path="microwave/meshes/handle_mesh.obj",
        microwave_root_anchor=anchor,
    )
    provenance = result["provenance"]
    np.testing.assert_allclose(
        provenance["aligned_object_poses_world"]["Microwave/Microwave_0"], root,
    )
    assert provenance["pose_source_modes"]["Microwave/Microwave_0"] == "verified_stationary_anchor"
    assert provenance["microwave_root_anchor"]["source"] == anchor["source"]
    assert provenance["source_frame_number"] == 73


def test_stale_stationary_anchor_rejected_before_scene_export(tmp_path, monkeypatch):
    """机身锚点可以来自旧帧，但现场未重新确认静止就不能沿用。"""
    from gene_agent.realworld.microwave import scene as module
    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    observed, robots, grippers, angles, static = _inputs(tmp_path)
    anchor = {"T_world_scene_asset": observed.object_poses_sdk["Microwave/Microwave_0"].tolist(),
              "verified_at_unix_ms": 995000., "source": "old-operator-check"}
    with pytest.raises(ValueError, match="本体锚点.*过期"):
        build_live_microwave_collision_scene(
            observed=observed, robots=robots, grippers=grippers,
            angles=angles, scene_parameters=static, max_age_s=2.,
            microwave_pose_mesh_path="microwave/meshes/handle_mesh.obj",
            microwave_root_anchor=anchor,
        )
    assert not static["output_dir"].exists()


def test_measured_and_snapped_door_angles_remain_distinct_in_scene_report(tmp_path, monkeypatch):
    """旧 Task1 用五度规划角，但不能丢失实测的 37° 原始证据。"""
    from gene_agent.realworld.microwave import scene as module
    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    observed, robots, grippers, angles, static = _inputs(tmp_path)
    angles["door"].update(angle_rad=math.radians(35),
                          measured_angle_rad=math.radians(37))
    result = build_live_microwave_collision_scene(
        observed=observed, robots=robots, grippers=grippers,
        angles=angles, scene_parameters=static, max_age_s=2.,
        microwave_pose_mesh_path="microwave/meshes/obj_mesh.obj",
    )
    assert result["provenance"]["door_angle_rad"] == pytest.approx(math.radians(35))
    assert result["provenance"]["live_state"]["door"]["measured_angle_rad"] == pytest.approx(
        math.radians(37))


def test_live_knob_reading_above_180_degrees_uses_asset_limit(tmp_path, monkeypatch):
    """现场 knob_0 转过半圈后仍可按真实关节限位导出碰撞。"""
    from gene_agent.realworld.microwave import scene as module
    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    observed, robots, grippers, angles, static = _inputs(tmp_path)
    angles["knob"].update(angle_rad=3.5, measured_angle_rad=3.5)

    result = build_live_microwave_collision_scene(
        observed=observed, robots=robots, grippers=grippers,
        angles=angles, scene_parameters=static, max_age_s=2.,
        microwave_pose_mesh_path="microwave/meshes/obj_mesh.obj",
    )

    assert result["provenance"]["knob_angle_rad"] == pytest.approx(3.5)


def test_scene_observer_refreshes_frame_and_articulated_state_on_every_call(tmp_path, monkeypatch):
    """独立/组合子任务反复调用时，不能复用上段门角、物体 pose 或目录。"""
    from gene_agent.realworld.microwave import scene as module
    from gene_agent.realworld.microwave.scene import make_live_microwave_observer
    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    observed, robots, grippers, angles, static = _inputs(tmp_path)
    del static["output_dir"]
    captures = []

    def capture_frame(*, output_dir, **_kwargs):
        path = Path(output_dir)
        path.mkdir(parents=True)
        captures.append(path)
        return path

    def track_frame(*, output_dir, **_kwargs):
        frame = observed.frame_number + len(captures) - 1
        bowl = observed.object_poses_sdk["Bowl/Bowl_0"].copy()
        bowl[0, 3] += .02 * (len(captures) - 1)
        return ObservedScene(
            "right", "0", frame, 999500., np.eye(4),
            {**observed.object_poses_sdk, "Bowl/Bowl_0": bowl},
            observed.sources, True,
        )

    monkeypatch.setattr(module, "capture_frozen_scene_frame", capture_frame)
    monkeypatch.setattr(module, "track_observed_scene_from_capture", track_frame)
    observer = make_live_microwave_observer(
        output_dir=tmp_path / "observations",
        camera_endpoint="camera", expected_serial="camera-1",
        pose_client=object(), sessions={"Bowl/Bowl_0": "bowl",
                                        "Microwave/Microwave_0": "microwave"},
        side="right", camera_id="0", T_sdk_camera=np.eye(4),
        robots=robots, grippers=grippers,
        angle_reader=lambda _scene: {
            **angles, "door": {**angles["door"],
                               "angle_rad": .3 + .1 * (len(captures) - 1)}},
        scene_parameters=static, max_age_s=2.,
        microwave_pose_mesh_path="microwave/meshes/obj_mesh.obj",
    )
    first = observer()
    second = observer()
    assert first["provenance"]["source_frame_number"] == 73
    assert second["provenance"]["source_frame_number"] == 74
    assert second["provenance"]["door_angle_rad"] == pytest.approx(.4)
    assert second["provenance"]["aligned_object_poses_world"]["Bowl/Bowl_0"][0][3] == pytest.approx(
        first["provenance"]["aligned_object_poses_world"]["Bowl/Bowl_0"][0][3] + .02)
    assert captures == [tmp_path / "observations/0000/capture",
                        tmp_path / "observations/0001/capture"]


def test_exported_live_scenes_can_be_compared_before_robot_command(tmp_path, monkeypatch):
    """用真实碰撞导出复核两次现场帧，避免只在合成字典上通过。"""
    from gene_agent.realworld.microwave import scene as module

    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    observed, robots, grippers, angles, static = _inputs(tmp_path)
    limits = SceneDriftLimits(.01, .05, .02, .005, .03)

    def export(frame, bowl_shift):
        poses = {name: pose.copy() for name, pose in observed.object_poses_sdk.items()}
        poses["Bowl/Bowl_0"][0, 3] += bowl_shift
        current = ObservedScene(
            "right", "0", frame, 999500. + frame - 73,
            np.eye(4), poses, observed.sources, True,
        )
        return build_live_microwave_collision_scene(
            observed=current, robots=robots, grippers=grippers,
            angles=angles,
            scene_parameters={**static, "output_dir": tmp_path / f"scene_{frame}"},
            max_age_s=2., microwave_pose_mesh_path="microwave/meshes/obj_mesh.obj",
        )

    planned, noisy, moved = export(73, 0.), export(74, .003), export(75, .04)
    assert compare_scene_drift(planned, noisy, limits=limits).status == "pass"
    result = compare_scene_drift(planned, moved, limits=limits)
    assert result.status == "fail"
    assert "Bowl/Bowl_0" in result.reason


def test_changed_camera_calibration_rejects_same_aligned_poses(tmp_path, monkeypatch):
    """配置文件不变、物体 world pose 恰好相同时仍要发现标定被更换。"""
    from gene_agent.realworld.microwave import scene as module

    monkeypatch.setattr(module.time, "time", lambda: 1000.)
    observed, robots, grippers, angles, static = _inputs(tmp_path)
    updated_camera = np.eye(4)
    updated_camera[0, 3] = .02
    changed = replace(observed, frame_number=74, timestamp_ms=999600.,
                      T_sdk_camera=updated_camera)

    def export(scene, name):
        return build_live_microwave_collision_scene(
            observed=scene, robots=robots, grippers=grippers,
            angles=angles,
            scene_parameters={**static, "output_dir": tmp_path / name},
            max_age_s=2., microwave_pose_mesh_path="microwave/meshes/obj_mesh.obj",
        )

    planned = export(observed, "planned")
    current = export(changed, "current")
    assert planned["provenance"]["scene_config_sha256"] == current["provenance"]["scene_config_sha256"]
    result = compare_scene_drift(
        planned, current, limits=SceneDriftLimits(.01, .05, .02, .005, .03),
    )
    assert result.status == "fail"
    assert "标定" in result.reason
