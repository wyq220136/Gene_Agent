"""旋钮阶段几何与 Skill 回放格式的行为边界。"""

import json
from pathlib import Path

import numpy as np
import pytest

from gene_agent.knowledge.operation import AxisSpec
from gene_agent.motion.model import MotionOperation
from gene_agent.skills.library.knob.planning import KnobSkillSettings, RotateKnobSkill
from gene_agent.visualization.replay import ReplayTimeline


def test_knob_stages_preserve_axis_and_use_linear_approach():
    grasp = np.eye(4)
    grasp[:3, 3] = [.4, .2, .1]
    axis = AxisSpec([.37, .2, .1], [1, 0, 0], "world", "explicit_joint")
    # 旧 Viser 仅在 grasp 的局部 X 倾角非零时才走逐段 TCP 直线旋转；
    # 默认 0° profile 会一次旋转末关节。
    settings = KnobSkillSettings(.03, np.pi / 4, np.pi / 36, .08, .055, .08,
                                 tip_rotations_x_rad=(np.deg2rad(5.),))

    stages = RotateKnobSkill().targets(grasp, axis, start_angle_rad=0., settings=settings)

    assert [stage.name for stage in stages[:4]] == ["open", "pregrasp", "approach", "close"]
    assert stages[2].operation is MotionOperation.LINEAR_TCP
    assert stages[-1].name == "retreat"
    assert stages[-1].operation is MotionOperation.LINEAR_TCP
    np.testing.assert_allclose(stages[1].tcp_pose_world[:3, 3],
                               grasp[:3, 3] - .03 * grasp[:3, 2])
    rotations = [stage for stage in stages if stage.name.startswith("rotate_")]
    assert len(rotations) == 9
    assert rotations[-1].end_angle_rad == pytest.approx(np.pi / 4)
    assert all(stage.start_angle_rad < stage.end_angle_rad for stage in rotations)
    # 这个案例的 TCP 位于轴线上，旋钮转动时只能改变姿态，不能绕世界原点漂移。
    for stage in rotations:
        np.testing.assert_allclose(stage.tcp_pose_world[:3, 3], grasp[:3, 3], atol=1e-10)
    # 旧版仅在“转末关节”分支保留旋转后的 TCP 姿态；逐点直线分支
    # 从原 pregrasp 姿态计算撤离方向。
    np.testing.assert_allclose(stages[-1].tcp_pose_world[:3, 3],
                               grasp[:3, 3] - .03 * stages[1].tcp_pose_world[:3, 2])


def test_knob_run_uses_complete_qpos_and_stage_collision_snapshot(tmp_path: Path):
    run = tmp_path
    (run / "source_scene").mkdir()
    (run / "source_scene/scene.xml").write_text("<mujoco/>", encoding="utf-8")
    (run / "collision/rotate_05").mkdir(parents=True)
    (run / "collision/rotate_05/collision_snapshot.json").write_text(
        json.dumps({"snapshot_sha256": "snapshot-one"}), encoding="utf-8")
    row = {
        "name": "rotate_05", "status": "planned", "operation": "plan",
        "collision_snapshot_ref": "collision/rotate_05/collision_snapshot.json",
        "request": {"operation": "plan", "scene": {"snapshot_sha256": "snapshot-one"}},
        "result": {"operation": "plan", "success": True},
        "planned_replay_samples": [
            {"time_s": 0., "qpos": [0., 0.], "robot_group": "right",
             "joint_names": ["joint"], "joint_positions_rad": [0.],
             "gripper_width_m": .06, "object_poses_world": {"knob_0": np.eye(4).tolist()},
             "tcp_pose_world": np.eye(4).tolist()},
            {"time_s": .2, "qpos": [.1, .087], "robot_group": "right",
             "joint_names": ["joint"], "joint_positions_rad": [.1],
             "gripper_width_m": .06, "object_poses_world": {"knob_0": np.eye(4).tolist()},
             "tcp_pose_world": np.eye(4).tolist()},
        ],
    }
    (run / "skill_rotate_knob_0_result.json").write_text(json.dumps({
        "schema_version": 1, "skill": "rotate_knob_0", "instance_id": "knob_0",
        "selected_candidate_id": "calibrated", "model_ref": "source_scene/scene.xml",
        "stages": [row],
    }), encoding="utf-8")

    timeline = ReplayTimeline.load(run)
    frame = timeline.seek(.2)
    assert timeline.view.skill == "rotate_knob_0"
    assert frame.sample_quality == "planned_kinematic_preview"
    assert frame.qpos == (.1, .087)
    assert frame.collision_snapshot_ref == row["collision_snapshot_ref"]

    corrupted = json.loads((run / "skill_rotate_knob_0_result.json").read_text(encoding="utf-8"))
    corrupted["stages"][0]["request"]["scene"]["snapshot_sha256"] = "other-scene"
    (run / "skill_rotate_knob_0_result.json").write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="碰撞快照不一致"):
        ReplayTimeline.load(run)
