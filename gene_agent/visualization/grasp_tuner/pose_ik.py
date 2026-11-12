"""仅用于调参反馈的 MuJoCo 位姿 IK；不使用碰撞或路径规划。"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseIKResult:
    reachable: bool
    position_error_m: float
    rotation_error_deg: float
    joint_positions: tuple[float, ...] | None
    backend: str = "mujoco"
    constraint_feasible: bool | None = None
    failure_category: str | None = None
    collision_check: dict | None = None
    diagnostic: dict | None = None


class MuJoCoPoseIK:
    """对指定机器人关节组做阻尼最小二乘 IK。

    只调用运动学正解与 TCP 雅可比，完全不查询接触、碰撞代理或轨迹。
    为避免把某个姿态的单次局部极小值误判为不可达，会从当前姿态和
    若干确定性扰动初值求解；所有初值都受关节限位约束。
    """

    def __init__(self, model, source_data, joint_names: tuple[str, ...], site_name: str):
        self.model = model
        self.source_data = source_data
        self.joint_names = joint_names
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"TCP site 不存在: {site_name}")
        self.qpos_ids = np.array([model.joint(name).qposadr[0] for name in joint_names], dtype=int)
        self.dof_ids = np.array([model.joint(name).dofadr[0] for name in joint_names], dtype=int)
        self.low = np.array([model.joint(name).range[0] for name in joint_names], dtype=float)
        self.high = np.array([model.joint(name).range[1] for name in joint_names], dtype=float)
        self.home = source_data.qpos[self.qpos_ids].copy()

    def solve(self, goal_pose: np.ndarray) -> PoseIKResult:
        model = self.model
        data = mujoco.MjData(model)
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        rng = np.random.default_rng(2718)
        starts = [self.home, *[np.clip(self.home + rng.normal(0, .55, len(self.home)), self.low, self.high) for _ in range(3)]]
        best = PoseIKResult(False, float("inf"), float("inf"), None)
        for start in starts:
            mujoco.mj_copyData(data, model, self.source_data)
            data.qpos[self.qpos_ids] = start
            for _ in range(90):
                mujoco.mj_forward(model, data)
                site = data.site(self.site_id)
                translation = goal_pose[:3, 3] - site.xpos
                rotation = Rotation.from_matrix(goal_pose[:3, :3] @ site.xmat.reshape(3, 3).T).as_rotvec()
                pos_error = float(np.linalg.norm(translation))
                rot_error = float(np.degrees(np.linalg.norm(rotation)))
                score = pos_error + .002 * rot_error
                if score < best.position_error_m + .002 * best.rotation_error_deg:
                    best = PoseIKResult(pos_error < .003 and rot_error < 4., pos_error, rot_error,
                                        tuple(float(x) for x in data.qpos[self.qpos_ids]))
                if best.reachable:
                    return best
                mujoco.mj_jacSite(model, data, jac_pos, jac_rot, self.site_id)
                jacobian = np.vstack((jac_pos[:, self.dof_ids], jac_rot[:, self.dof_ids]))
                error = np.r_[translation, rotation]
                delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .0025 * np.eye(6), error)
                delta = np.clip(delta, -.18, .18)
                data.qpos[self.qpos_ids] = np.clip(data.qpos[self.qpos_ids] + delta, self.low, self.high)
        return best
