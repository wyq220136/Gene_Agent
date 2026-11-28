"""固定在 TCP 上的轴向路径约束；求解代价与任务硬后验分开。"""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np
from scipy.spatial.transform import Rotation

from .core import SynchronousPlanner, add_tcp_tool_frame

FRAME = "gene_path_orientation"


def normalize_orientation_constraint(value: dict) -> dict:
    """配置采用米、world 单位向量；局部 +Z 是要保持朝上的轴。"""
    transform = np.asarray(value["T_tcp_frame"], dtype=float)
    up = np.asarray(value["up_world"], dtype=float)
    weight = float(value.get("non_terminal_weight", 1.))
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or np.linalg.det(transform[:3, :3]) < 0
            or up.shape != (3,) or not np.isfinite(up).all()
            or not np.isclose(np.linalg.norm(up), 1.)
            or not np.isfinite(weight) or weight <= 0):
        raise ValueError("路径轴向约束需要合法 T_tcp_frame、world 单位上向量和正权重")
    return {"T_tcp_frame": transform.tolist(), "up_world": up.tolist(),
            "non_terminal_weight": weight}


def upright_reference(target_world_frame, up_world):
    """保留目标航向；不要求轴对称物体在搬运途中保持偏航角。"""
    target = np.asarray(target_world_frame, dtype=float)
    up = np.asarray(up_world, dtype=float)
    x = target[:3, 0] - up * np.dot(up, target[:3, 0])
    if np.linalg.norm(x) <= 1e-8:
        # 目标 X 接近上向量时选与 up 最不平行的参考轴，避免零除。
        basis = np.eye(3)[int(np.argmin(np.abs(up)))]
        x = basis - up * np.dot(up, basis)
    x /= np.linalg.norm(x)
    result = target.copy()
    result[:3, :3] = np.column_stack((x, np.cross(up, x), up))
    return result


class OrientationConstrainedPlanner(SynchronousPlanner):
    """辅助 frame 参与模型身份；代价只在显式 plan_with_orientation 内启用。"""

    def __init__(self, *, path_orientation_constraint, **kwargs):
        self.path_orientation_constraint = normalize_orientation_constraint(
            path_orientation_constraint)
        self._orientation_active = False
        self._orientation_target = None
        super().__init__(**kwargs)

    def _configure_extra_frames(self, config, tool_frame, link_from_tcp):
        return add_tcp_tool_frame(
            config, tool_frame,
            np.asarray(link_from_tcp) @ np.asarray(
                self.path_orientation_constraint["T_tcp_frame"]),
            tcp_frame=FRAME,
        )

    def _initialize_motion_planner(self):
        new = self._planner is None
        super()._initialize_motion_planner()
        if new:
            from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
            self._planner.update_tool_pose_criteria({FRAME: ToolPoseCriteria.disabled()})

    def _append_goal(self, goal, target, *, constrained):
        from curobo.types import GoalToolPose, Pose
        frame = np.asarray(target) @ np.asarray(
            self.path_orientation_constraint["T_tcp_frame"])
        if constrained:
            frame = upright_reference(
                frame, self.path_orientation_constraint["up_world"])
        base = np.linalg.inv(self.world_from_base) @ frame
        quaternion = Rotation.from_matrix(base[:3, :3]).as_quat()[[3, 0, 1, 2]]
        poses = goal.to_dict()
        poses[FRAME] = Pose(
            position=self.torch.tensor(base[None, :3, 3], device="cuda",
                                       dtype=self.torch.float32),
            quaternion=self.torch.tensor(quaternion[None], device="cuda",
                                         dtype=self.torch.float32),
        )
        return GoalToolPose.from_poses(
            poses, ordered_tool_frames=[self.tool_frame, self.tcp_frame, FRAME],
            num_goalset=1,
        )

    def _goal(self, target, target_frame):
        return self._append_goal(
            super()._goal(target, target_frame), target,
            constrained=self._orientation_active)

    def _linear_goal(self, target, target_frame, start_tcp_world, *,
                     stationary_rotation=False):
        # 直线辅助 TCP 的旋转用于定义平移轴；固定路径 frame 的真实目标仍须齐全。
        return self._append_goal(
            super()._linear_goal(target, target_frame, start_tcp_world,
                                 stationary_rotation=stationary_rotation),
            target, constrained=False)

    def _physical_goal(self, target, target_frame):
        return self._append_goal(
            super()._goal(target, target_frame), target, constrained=False)

    def _followup_ik_goal(self, operation, target, target_frame, solver_goal):
        # IK 只验证真实终点；不能把非终点的直立参考当成矛盾的第三个末端目标。
        if self._orientation_active:
            return self._physical_goal(target, target_frame)
        return super()._followup_ik_goal(operation, target, target_frame, solver_goal)

    def _diagnose_world_obstruction(self, start, goal):
        if self._orientation_active:
            goal = self._physical_goal(self._orientation_target, "world_tcp")
        return super()._diagnose_world_obstruction(start, goal)

    @contextmanager
    def _linear_criteria(self, non_terminal_scale=1.):
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
        self._planner.update_tool_pose_criteria({FRAME: ToolPoseCriteria.disabled()})
        try:
            with super()._linear_criteria(non_terminal_scale):
                yield
        finally:
            self._planner.update_tool_pose_criteria({FRAME: ToolPoseCriteria.disabled()})

    def plan_with_orientation(self, joint_names, positions, target, *,
                              target_frame="world_tcp", max_attempts=1,
                              diagnostic_label=None):
        if target_frame != "world_tcp":
            raise ValueError("路径轴向约束只接受明确的 world_tcp 目标")
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
        self._initialize_motion_planner()
        scale = self.path_orientation_constraint["non_terminal_weight"]
        self._orientation_active = True
        self._orientation_target = np.asarray(target, dtype=float).copy()
        try:
            self._planner.update_tool_pose_criteria({
                self.tool_frame: ToolPoseCriteria(),
                self.tcp_frame: ToolPoseCriteria(),
                FRAME: ToolPoseCriteria(
                    terminal_pose_axes_weight_factor=[0.] * 6,
                    non_terminal_pose_axes_weight_factor=[0., 0., 0., scale, scale, 0.],
                    project_distance_to_goal=True,
                ),
            })
            result = self.plan(joint_names, positions, target,
                               target_frame=target_frame, max_attempts=max_attempts,
                               diagnostic_label=diagnostic_label)
            return {**result, "path_orientation_constraint": {
                **self.path_orientation_constraint,
                "frame": FRAME, "yaw_free": True,
                "source": "nonterminal_pose_cost_with_unchanged_tcp_endpoint",
                "hard_constraint_check": "caller_postcheck_required",
            }}
        finally:
            self._orientation_active = False
            self._orientation_target = None
            self._planner.update_tool_pose_criteria({
                self.tool_frame: ToolPoseCriteria(),
                self.tcp_frame: ToolPoseCriteria(),
                FRAME: ToolPoseCriteria.disabled(),
            })
