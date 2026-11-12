"""用旧真机采集法兰位姿审查新 MuJoCo 的 TCP 坐标链。"""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.realworld.devices.arms.realman.adapter import load_standard_gripper_flange_tcp
from gene_agent.simulation import build_scene, set_joint_positions

ROOT = Path(__file__).resolve().parents[2]


def test_recorded_sdk_flange_matches_mujoco_tcp_for_both_arms(tmp_path):
    """只检历史默认关节样本；当前真机安装仍需现场多姿态核对。"""
    model, data, metadata = build_scene(
        tmp_path / "scene", seed=0, perturb=False,
        robot_profile="realman_dual_standard",
    )
    robot = metadata["robot"]
    installed = json.loads((ROOT / "assets/robots/realman/installation.json").read_text(
        encoding="utf-8"))["mounted"]
    world_from_right_sdk = np.linalg.inv(np.asarray(
        installed["body_from_sdk"]["right"], dtype=float,
    ))
    flange_tcp = load_standard_gripper_flange_tcp()
    for group in ("right", "left"):
        saved = json.loads((ROOT / f"config/robots/realman_{group}_default_pose.json").read_text(
            encoding="utf-8"))
        names = robot["joint_groups"][group]
        assert tuple(names) == tuple(f"{group}_joint_{i}" for i in range(1, 8))
        set_joint_positions(model, data, dict(zip(names, saved["joints_rad"])))
        mujoco.mj_forward(model, data)
        site = data.site(robot["tcp_sites"][group])
        actual = np.eye(4)
        actual[:3, :3] = site.xmat.reshape(3, 3)
        actual[:3, 3] = site.xpos
        world_from_sdk = world_from_right_sdk @ np.asarray(
            installed["body_from_sdk"][group], dtype=float,
        )
        expected = world_from_sdk @ np.asarray(saved["flange_pose_sdk"]) @ flange_tcp
        translation = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
        rotation = float(np.linalg.norm(Rotation.from_matrix(
            expected[:3, :3].T @ actual[:3, :3],
        ).as_rotvec()))
        assert translation < 3e-5, group
        assert rotation < 3e-4, group
