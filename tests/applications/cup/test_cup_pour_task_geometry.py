"""杯子倾倒与回位目标的几何约束。"""

import numpy as np
import mujoco
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

from gene_agent.tasks.policies.cup_pour_task.simulation import (
    CupPourTaskRun, target_tcp_for_cup_pose, tilt_cup_about_lip)
from gene_agent.motion.mujoco_ik import MujocoPosePlanner


def test_cup_target_keeps_measured_tcp_to_cup_transform():
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    tcp[:3, 3] = [.35, -.08, .20]
    cup = np.eye(4)
    cup[:3, 3] = [.35, -.16, .08]
    desired = np.eye(4)
    desired[:3, 3] = [.34, .065, .115]
    target = target_tcp_for_cup_pose(tcp, cup, desired)
    assert np.allclose(target @ np.linalg.inv(tcp) @ cup, desired, atol=1e-9)


def test_tilt_preserves_world_lip_position_and_rotates_cup_only_15_degrees():
    cup = np.eye(4)
    cup[:3, 3] = [.34, .065, .115]
    pivot = np.array([0., .035, .095])
    goal = tilt_cup_about_lip(cup, pivot, -15.)
    before = (cup @ np.r_[pivot, 1.])[:3]
    after = (goal @ np.r_[pivot, 1.])[:3]
    assert np.allclose(before, after, atol=1e-9)
    assert np.isclose(np.linalg.norm(Rotation.from_matrix(
        goal[:3, :3] @ cup[:3, :3].T).as_rotvec()), np.deg2rad(15))


def test_pour_stops_at_fifty_degrees_after_five_balls_reach_bowl(tmp_path):
    """少量球进入目标碗即可，不为了倒空杯子继续转动。"""
    names = {f"ball_{index}" for index in range(100)}
    stages = []
    holds = []
    cup = SimpleNamespace(xmat=np.eye(3).reshape(-1), xpos=np.zeros(3))
    bowl = SimpleNamespace(xpos=np.array([.3, .04, 0.]))
    runner = SimpleNamespace(
        data=SimpleNamespace(body=lambda name: bowl if name == "bowl_target" else cup),
        target_name="bowl_target", object_name="cup_green",
        initial_cup_pose=np.eye(4), initial_cargo=names,
        scene={"nominal_object_poses": {"cup_green": {"T_world_object": np.eye(4).tolist()}}},
        output_dir=tmp_path, report_task={"skills": {}, "stages": {}},
    )
    runner._task_motion = lambda stage, goal, **kwargs: (
        stages.append(stage) or {"success": True})
    runner._hold = lambda stage, seconds: (holds.append((stage, seconds)) or
                                           {"success": True})
    runner._ball_membership = lambda: (
        {"in_cup": set(sorted(names)[6:]), "in_target": set(sorted(names)[:5]),
         "outside": {sorted(names)[5]}}
        if any(stage == "hold_50_after_lower" for stage, _ in holds) else
        {"in_cup": names, "in_target": set(), "outside": set()})

    assert CupPourTaskRun._run_pour(runner)
    assert ("hold_50_after_lower", 4.) in holds
    assert "tilt_60" not in stages
    assert "tilt_90" not in stages
    assert runner.report_task["skills"]["pour"]["final_tilt_deg"] == 50


def test_attached_ball_uses_current_cup_frame_during_path_check():
    """携杯路径复核不能把杯内球留在起点世界坐标。"""
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body name="robot">
          <joint name="slider" type="slide" axis="1 0 0" range="-0.1 0.4"/>
          <geom name="finger" type="sphere" size="0.03" mass="1"/>
          <site name="tcp"/>
        </body>
        <body name="cup" pos="0.3 0 0">
          <freejoint name="cup_free"/>
          <geom name="cup_geom" type="sphere" size="0.01" mass="1"/>
        </body>
        <body name="ball" pos="0.3 0 0">
          <freejoint name="ball_free"/>
          <geom name="ball_geom" type="sphere" size="0.015" mass="0.01"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    metadata = {"joint_groups": {"arm": ["slider"]},
                "tcp_sites": {"arm": "tcp"},
                "group_root_bodies": {"arm": "robot"}}
    checker = MujocoPosePlanner(model, data, metadata, "arm")
    offset = np.eye(4)
    offset[0, 3] = .3
    path = np.array([[0.], [.1], [.2], [.3]])
    allowed = [("cup", "ball"), ("robot", "cup")]

    stale = checker._check_path(
        path, allowed_contacts=allowed, attached_object="cup", T_tcp_object=offset)
    moving = checker._check_path(
        path, allowed_contacts=allowed, attached_object="cup", T_tcp_object=offset,
        moving_body_names=("ball",))

    assert any("ball" in row["body_names"] for row in stale)
    assert moving == []
