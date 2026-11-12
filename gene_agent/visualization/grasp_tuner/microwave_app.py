"""在完整场景中检查门把手抓取/拉门、门板推门和旋钮 Knowledge。"""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.tasks.policies.microwave_task.planning.bowl import load_pick_bowl_case
from gene_agent.knowledge.reachability import assess_pose_ik
from gene_agent.visualization.pick_bowl.scene import PickBowlScene
from .feedback import write_live_state
from .gripper_preview import GripperMeshPreview
from .microwave_scene import _body_pose, build_microwave_knowledge, set_preview_microwave_pose


MODES = ("handle_grasp", "handle_pull", "door_push", "knob_grasp", "knob_turn")
IK_BACKENDS = ("mujoco", "curobo")
MODE_LABELS = {
    "handle_grasp": "抓门把手",
    "handle_pull": "把手预抓取到拉门",
    "door_push": "门板接触推门",
    "knob_grasp": "抓 knob_0",
    "knob_turn": "旋转 knob_0",
}


def explain_curobo_failure(ik) -> str:
    """把失败种子与整个目标的证据分开陈述，避免把一次搜索失败说成不可达。"""
    diagnostic = ik.diagnostic or {}
    preview_source = diagnostic.get("preview_source")
    if preview_source == "followup_all_seeds":
        preview_note = ("画面显示补充搜索中通过关节/自身约束的近似候选"
                        if ik.constraint_feasible else
                        "画面显示补充搜索中误差较小的候选，但它未通过关节/自身约束")
        preview_note += (f"；TCP 位置误差 {ik.position_error_m*1000:.1f} mm，"
                         f"姿态误差 {ik.rotation_error_deg:.1f}°；仍非严格 IK 解")
    else:
        preview_note = "画面显示原求解返回的失败候选；它未必是最接近目标的姿态"
    check = diagnostic.get("failed_seed_check") or {}
    if check.get("joint_limit_violation"):
        seed_note = "失败候选触及关节限位"
    elif check.get("self_collision"):
        seed_note = "失败候选发生机器人自身碰撞"
    elif diagnostic.get("failed_seed_constraint_feasible") is False:
        seed_note = "失败候选未通过约束，但具体类型未确认"
    elif diagnostic.get("solver_reason") == "pose_not_converged":
        seed_note = "候选满足已检查约束，但位姿误差仍超过 cuRobo 阈值"
    else:
        seed_note = "未找到满足 cuRobo 阈值和约束的候选"
    comparison = diagnostic.get("mujoco_candidate_comparison") or {}
    conclusion = comparison.get("conclusion")
    if conclusion == "solver_search_missed_known_valid_candidate":
        position = comparison["curobo_fk_position_error_m"] * 1000
        rotation = comparison["curobo_fk_rotation_error_deg"]
        comparison_note = (f"MuJoCo 解经 cuRobo 复核：{position:.1f} mm、{rotation:.1f}°，"
                           "关节限位和自身碰撞均通过；本次搜索未找到，不能据此判定目标无解")
    elif conclusion == "mujoco_candidate_joint_limit":
        comparison_note = "MuJoCo 解违反 cuRobo 关节限位；两边的约束结论不同"
    elif conclusion == "mujoco_candidate_self_collision":
        comparison_note = "MuJoCo 解在 cuRobo 机器人碰撞球下发生自身碰撞"
    elif conclusion == "different_pose_tolerance":
        comparison_note = "MuJoCo 解满足其位姿阈值，但未满足 cuRobo 的严格阈值"
    elif conclusion == "mujoco_also_not_strict":
        check = comparison.get("candidate_check") or {}
        constraint_note = ("；该候选经 cuRobo 复核存在自身碰撞" if check.get("self_collision") else
                           "；该候选经 cuRobo 复核违反关节限位" if check.get("joint_limit_violation") else
                           "；该候选通过 cuRobo 关节/自身约束复核" if check.get("self_collision") is False
                           and check.get("joint_limit_violation") is False else "")
        comparison_note = ("相同目标的 MuJoCo 近似候选误差为 "
                           f"{comparison['mujoco_position_error_m']*1000:.1f} mm、"
                           f"{comparison['mujoco_rotation_error_deg']:.1f}°；也未达到严格 IK 阈值"
                           f"{constraint_note}")
    elif conclusion == "mujoco_candidate_missing":
        comparison_note = "MuJoCo 未返回可用于约束复核的候选姿态"
    elif conclusion == "comparison_failed":
        comparison_note = f"MuJoCo 解交叉复核失败：{comparison.get('error', '原因未记录')}"
    else:
        comparison_note = "尚无足够证据区分搜索失败与约束不满足"
    return (f"**画面中的机械臂：** {preview_note}。  \n"
            f"**原始 cuRobo 尝试：** {seed_note}；只描述原返回候选。  \n"
            f"**跨求解器复核：** {comparison_note}。  \n")


class MicrowaveKnowledgeApp:
    """只改预览数据，不步进动力学，也不修改冻结的 MuJoCo run。"""

    def __init__(self, server: Any, root: str | Path, run_dir: str | Path, *, mode: str,
                 state_path: str | Path | None = None, initial_waypoint: int = 0,
                 restored: dict | None = None, ik_backend: str | None = None):
        if mode not in MODES:
            raise ValueError(f"未知展示模式：{mode}")
        self.server = server
        self.root = Path(root).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.state_path = Path(state_path).resolve() if state_path else None
        self.case = load_pick_bowl_case(self.run_dir)
        restored = restored or {}
        backend = ik_backend or restored.get("ik_backend") or "mujoco"
        if backend not in IK_BACKENDS:
            raise ValueError(f"未知 IK 后端：{backend}")
        if restored and restored.get("scene_snapshot_sha256") != self.case.snapshot.get("snapshot_sha256"):
            raise ValueError("恢复状态所属的场景快照已变化，不能直接套用旧 TCP 与相机")
        # 数值旋转控件始终相对冻结场景的名义姿态；从状态恢复时也使用
        # 同一个基准，不依赖上次拖拽经过的中间位置。
        self.placement_origin = _body_pose(self.case.data, "microwave")
        preview_settings = json.loads((self.root / "data/task_1/knowledge/microwave_knowledge_view.json").read_text(encoding="utf-8"))
        if restored.get("microwave_pose_world") is not None:
            set_preview_microwave_pose(self.case, np.asarray(restored["microwave_pose_world"], dtype=float))
        elif not restored and "preview_microwave_xy_m" in preview_settings:
            preview_pose = self.placement_origin.copy()
            preview_pose[:2, 3] = preview_settings["preview_microwave_xy_m"]
            set_preview_microwave_pose(self.case, preview_pose)
        tune = restored.get("tuner_values", {})
        handle_tune = tune.get("handle", {})
        pull_selection_tune = tune.get("handle_pull_selection", {})
        push_tune = tune.get("door_push", {})
        push_transition_tune = tune.get("door_push_transition", {})
        push_selection_tune = tune.get("door_push_selection", {})
        knob_tune = tune.get("knob", {})
        self.knowledge = build_microwave_knowledge(self.root, self.case)
        self.preview_data = mujoco.MjData(self.case.model)
        mujoco.mj_copyData(self.preview_data, self.case.model, self.case.data)
        self._syncing = True
        # Viser 的 GUI 回调可能由不同连接线程触发；场景节点增删与状态
        # 写入必须作为一个整体执行，避免同名路径刚建立就被另一回调删除。
        self._update_lock = RLock()
        self._last_valid_inputs: dict | None = None
        self.last_rejected_error: str | None = None
        self.current_error: str | None = None

        server.scene.set_up_direction("+z")
        server.scene.set_background_image(np.full((8, 8, 3), 238, dtype=np.uint8), format="png")
        server.gui.configure_theme(dark_mode=False, control_width="large", show_logo=False,
                                   show_share_button=False, brand_color=(53, 112, 169))
        self.scene = PickBowlScene(server, self.case.model, self.case.data, self.case.metadata, self.case.snapshot)
        self.scene.set_collision_visible(False)
        microwave_pose = _body_pose(self.case.data, "microwave")
        from gene_agent.visualization.pick_bowl.scene import _pose
        microwave_wxyz, microwave_position = _pose(microwave_pose)
        self.microwave_control = server.scene.add_transform_controls(
            "/microwave_knowledge/edit_microwave", scale=.17,
            position=microwave_position, wxyz=microwave_wxyz,
        )
        # 机械臂上原本的夹爪必须始终跟随 FK。另一个金色夹爪只是期望
        # 位姿的可选半透明预览；IK 失败时绝不能冒充机械臂末端。
        self.gripper = GripperMeshPreview(server, self.case,
                                          prefix="/microwave_knowledge/target_gripper",
                                          color_rgba=(245, 190, 55, 90))
        if not self.gripper.available:
            raise ValueError("当前场景没有可复用的真实夹爪 visual mesh")
        self.gripper.set_palm_visible(False)
        self.path_handle = None
        self.tcp_gap_handle = None
        self.push_candidate_cloud = None
        self.push_candidate_directions = None
        self.push_normal_handle = None
        self.push_selected_contact = None
        self.push_precontact_point = None
        self.tcp_frame = server.scene.add_frame("/microwave_knowledge/selected_tcp", axes_length=.035,
                                               axes_radius=.0015, origin_radius=.003)
        self.actual_tcp_frame = server.scene.add_frame("/microwave_knowledge/robot_tcp", axes_length=.03,
                                                       axes_radius=.0015, origin_radius=.003)

        initial_overview = bool(restored.get("robot_overview", mode != "handle_pull"))
        self.camera = restored.get("camera") or self._camera_for(mode, overview=initial_overview)
        server.initial_camera.position = self.camera["position"]
        server.initial_camera.look_at = self.camera["look_at"]
        server.initial_camera.up = self.camera["up"]
        server.initial_camera.fov = self.camera["fov"]

        @server.on_client_connect
        def _on_client(client: Any) -> None:
            self._set_client_camera(client)
            client.camera.on_update(self._on_camera_update)
            client.flush()
            client.get_render(height=8, width=8, transport_format="png", timeout=8.)
            print(f"[microwave-knowledge] scene-ready: {self.mode.value}", flush=True)

        gui = server.gui
        gui.add_markdown("### 微波炉 Knowledge 调参\n把手抓取、把手拉门、门板推门和旋钮参数可独立调整；灰蓝路径尚未检查整段 IK。")
        self.status = gui.add_markdown("")
        self.ik_backend = gui.add_dropdown("IK 求解器", options=IK_BACKENDS, initial_value=backend)
        self.ik_backend.on_update(lambda event: self._regenerate())
        self.mode = gui.add_dropdown("检查内容", options=MODES, initial_value=mode)
        self.mode.on_update(lambda event: self._regenerate())
        self.candidate = gui.add_slider("旋钮抓取候选", min=0, max=len(self.knowledge.knob_grasps)-1,
                                        step=1, initial_value=int(restored.get("candidate_index", 0)))
        self.candidate.on_update(lambda event: self._regenerate())
        # 恢复清单里的路径可能比默认目标角的路径更长；先容纳旧序号，
        # 待参数重算后由 _show 收紧到实际 waypoint 数量。
        waypoint_max = max(len(self.knowledge.handle_pull_path.waypoints),
                           len(self.knowledge.door_path.waypoints),
                           len(self.knowledge.knob_path.waypoints)) - 1
        waypoint_max = max(waypoint_max, int(restored.get("waypoint_index", 0)))
        self.waypoint = gui.add_slider("路径 waypoint", min=0,
                                       max=waypoint_max,
                                       step=1, initial_value=int(restored.get("waypoint_index", (
                                           len(self.knowledge.handle_pull_path.waypoints) - 1 if mode == "handle_pull" else
                                           len(self.knowledge.door_path.waypoints) - 1 if mode == "door_push" else
                                           len(self.knowledge.knob_path.waypoints) - 1) if initial_waypoint == -1 else initial_waypoint)))
        self.waypoint.on_update(lambda event: self._regenerate())
        self.palm = gui.add_checkbox("显示夹爪掌部", initial_value=bool(restored.get("gripper_palm_visible", False)))
        self.palm.on_update(lambda event: self.gripper.set_palm_visible(bool(self.palm.value)))
        self.gripper.set_palm_visible(bool(self.palm.value))
        self.target_ghost = gui.add_checkbox("显示目标夹爪预览（未达位姿）",
                                             initial_value=bool(restored.get("target_gripper_ghost_visible", False)))
        self.target_ghost.on_update(lambda event: self._show())
        gui.add_markdown("机械臂上的橙色夹爪始终是实际关节姿态；金色半透明夹爪仅表示 IK 未求得时的目标位姿。")
        approx_tune = tune.get("approximate_ik", {})
        with gui.add_folder("近似 IK 候选（仅 Knowledge 预览）", expand_by_default=False):
            gui.add_markdown("仅标记误差预算内的位姿候选；Knowledge 不检查场景碰撞或连续轨迹，不等于可执行。Skill 需单独启用同等阶段预算。")
            self.approximate_ik_enabled = gui.add_checkbox("启用近似候选标记", initial_value=bool(approx_tune.get("enabled", False)))
            self.approximate_position_mm = gui.add_slider("平移误差上限 (mm)", min=1, max=100, step=1,
                initial_value=float(approx_tune.get("max_position_error_m", .02))*1000)
            self.approximate_rotation_deg = gui.add_slider("姿态误差上限 (°)", min=1, max=90, step=1,
                initial_value=float(approx_tune.get("max_rotation_error_deg", 10.)))
        for control in (self.approximate_ik_enabled, self.approximate_position_mm, self.approximate_rotation_deg):
            control.on_update(lambda event: self._show())
        self.overview = gui.add_checkbox("全景（显示机械臂）", initial_value=initial_overview)
        self.overview.on_update(lambda event: self._focus())
        with gui.add_folder("微波炉摆位（仅当前预览）", expand_by_default=mode != "door_push"):
            gui.add_markdown("可拖拽场景中的坐标轴，或用滑块调整。微波炉、门、旋钮及垫台同步移动；每次改动重算当前目标和纯 IK。")
            self.microwave_xyz = {}
            for index, axis in enumerate("xyz"):
                value = float(microwave_pose[index, 3])
                half_range = .12 if axis == "z" else .35
                self.microwave_xyz[axis] = gui.add_slider(
                    f"微波炉 {axis.upper()} (m)",
                    min=min(float(self.placement_origin[index, 3]) - half_range, value),
                    max=max(float(self.placement_origin[index, 3]) + half_range, value),
                    step=.005, initial_value=value,
                )
            relative = microwave_pose[:3, :3] @ self.placement_origin[:3, :3].T
            angles = Rotation.from_matrix(relative).as_euler("xyz", degrees=True)
            self.microwave_rpy = {}
            for axis, value in zip(("roll", "pitch", "yaw"), angles):
                self.microwave_rpy[axis] = gui.add_slider(
                    f"相对名义位姿 {axis} (°)", min=min(-30., float(value)),
                    max=max(30., float(value)), step=1., initial_value=float(value),
                )
            self.microwave_reset = gui.add_button("恢复冻结场景摆位")
            self.microwave_reset.on_click(lambda event: self._apply_microwave_pose(self.placement_origin))
        for control in (*self.microwave_xyz.values(), *self.microwave_rpy.values()):
            control.on_update(lambda event: self._on_microwave_numbers())
        self.microwave_control.on_update(lambda event: self._on_microwave_transform(event))
        self.handle_parameters_folder = gui.add_folder(
            "把手抓取参数（初值为旧 profile）", expand_by_default=mode == "handle_grasp")
        with self.handle_parameters_folder:
            self.handle_vertical = gui.add_slider("沿把手上移 (mm)", min=-65, max=65, step=1, initial_value=handle_tune.get("vertical_offset_m", .04)*1000)
            self.handle_tip_x = gui.add_slider("局部 X 倾角 (°)", min=-90, max=90, step=1, initial_value=handle_tune.get("tip_x_deg", -60))
            self.handle_tip_y = gui.add_slider("局部 Y 倾角 (°)", min=-30, max=30, step=1, initial_value=handle_tune.get("tip_y_deg", 0))
            self.handle_depth = gui.add_slider("抓取深度 (mm)", min=0, max=30, step=1, initial_value=handle_tune.get("depth_m", .013)*1000)
        pull_tune = tune.get("handle_pull", {})
        self.handle_pull_folder = gui.add_folder(
            "把手拉门路径（纯几何预览）", expand_by_default=mode == "handle_pull")
        with self.handle_pull_folder:
            gui.add_markdown("预抓取、到达把手、闭爪、逐角拉门、松爪、撤离。Task1 候选只绕 grasp TCP 改变局部 X/Y 姿态；页面可切换名义路线，不自动做整路 IK 筛选。")
            initial_pull_candidate = min(
                max(int(pull_selection_tune.get("candidate_rank", 0)), 0),
                len(self.knowledge.handle_pull_candidates) - 1,
            )
            self.pull_candidate = gui.add_slider(
                "把手路线候选序号",
                min=0,
                max=len(self.knowledge.handle_pull_candidates) - 1,
                step=1,
                initial_value=initial_pull_candidate,
            )
            self.pull_target = gui.add_slider("拉门目标角 (°)", min=1, max=45, step=1,
                                               initial_value=pull_tune.get("target_angle_deg", 25.))
            self.pull_pregrasp = gui.add_slider("预抓取后退距离 (mm)", min=5, max=100, step=1,
                                                 initial_value=pull_tune.get("pregrasp_distance_m", .03)*1000)
            self.pull_retreat = gui.add_slider("松爪后局部 -Z 撤离 (mm)", min=0, max=100, step=1,
                                               initial_value=pull_tune.get("retreat_distance_m", .015)*1000)
            self.pull_retreat_up = gui.add_slider("松爪后世界 +Z 抬高 (mm)", min=0, max=80, step=1,
                                                  initial_value=pull_tune.get("retreat_up_m", 0.)*1000)
        self.door_push_folder = gui.add_folder(
            "Task1 逐角推门候选（零偏移即旧数值）", expand_by_default=mode == "door_push")
        with self.door_push_folder:
            gui.add_markdown("右臂开门默认从 30° 推到 85°：先抓把手拉开，再从门内侧推。每个角度保留 Task1 的完整有序候选集合；页面可切换起始 contact candidate。")
            gui.add_markdown("画面图例：紫色是当前门角前 50 个候选，红色球是已选 contact，绿色球和线段是 pre-contact 及门面朝外法向。")
            self.push_mode = gui.add_dropdown("推门模式", options=("right_open", "right_close"),
                                              initial_value=tune.get("door_push_mode", self.knowledge.settings["door_push_mode"]))
            self.push_fraction = gui.add_slider("门宽比例偏移", min=-.35, max=.35, step=.01,
                                                 initial_value=push_tune.get("hinge_fraction_offset", 0.))
            self.push_top = gui.add_slider("距顶部偏移 (mm)", min=-25, max=25, step=1,
                                            initial_value=push_tune.get("top_down_distance_offset_m", 0.) * 1000)
            self.push_standoff = gui.add_slider("离门距离偏移 (mm)", min=-30, max=30, step=1,
                                                 initial_value=push_tune.get("standoff_offset_m", 0.) * 1000)
            self.push_tip_x = gui.add_slider("局部 X 倾角偏移 (°)", min=-40, max=40, step=1,
                                              initial_value=push_tune.get("tip_x_offset_deg", 0.))
            self.push_tip_y = gui.add_slider("局部 Y 倾角偏移 (°)", min=-30, max=30, step=1,
                                              initial_value=push_tune.get("tip_y_offset_deg", 0.))
            self.push_surface = gui.add_dropdown("接触表面覆盖", options=("auto", "inside", "outside"),
                                                  initial_value=push_tune.get("surface", "auto"))
            initial_push_candidate = int(push_selection_tune.get("start_candidate_rank", 0))
            initial_push_candidate = min(
                max(initial_push_candidate, 0),
                len(self.knowledge.door_candidate_sets[0].candidates) - 1,
            )
            self.push_candidate = gui.add_slider(
                "起始门角接触候选序号",
                min=0,
                max=len(self.knowledge.door_candidate_sets[0].candidates) - 1,
                step=1,
                initial_value=initial_push_candidate,
            )
        self.door_push_transition_folder = gui.add_folder(
            "推门 pre-contact（Task1 同源）", expand_by_default=mode == "door_push")
        with self.door_push_transition_folder:
            gui.add_markdown("先选择起始门角的 contact candidate，再沿该门面的朝外法向生成 pre-contact；姿态保持不变。Task1 推完后的默认 TCP 撤离由 Skill/Task 提供。")
            self.push_pre_contact = gui.add_slider(
                "沿门面朝外法向退让 (mm)",
                min=10,
                max=120,
                step=1,
                initial_value=push_transition_tune.get("pre_contact_distance_m", .05) * 1000,
            )
        self.knob_parameters_folder = gui.add_folder(
            "旋钮抓取参数", expand_by_default=mode == "knob_grasp")
        with self.knob_parameters_folder:
            self.knob_azimuth = gui.add_slider("全部方位角偏移 (°)", min=-90, max=90, step=1, initial_value=knob_tune.get("azimuth_shift_deg", 0))
            self.knob_depth = gui.add_slider("抓取深度 (mm)", min=0, max=20, step=1, initial_value=knob_tune.get("depth_m", .005)*1000)
            self.knob_tip_x = gui.add_slider("局部 X 倾角 (°)", min=-30, max=30, step=1, initial_value=knob_tune.get("tip_x_deg", 0))
        self.operation_target_folder = gui.add_folder(
            "操作目标", expand_by_default=mode in {"door_push", "knob_turn"})
        with self.operation_target_folder:
            self.door_start = gui.add_slider("推门起始角 (°)", min=0, max=110, step=1,
                                             initial_value=tune.get("door_start_angle_deg", float(np.degrees(self.knowledge.door_path.start_angle_rad))))
            self.door_target = gui.add_slider("推门目标角 (°)", min=0, max=110, step=1,
                                              initial_value=tune.get("door_target_angle_deg", float(np.degrees(self.knowledge.door_path.target_angle_rad))))
            self.knob_target = gui.add_slider("旋钮目标角 (°)", min=0, max=270, step=1,
                                              initial_value=tune.get("knob_target_angle_deg", float(np.degrees(self.knowledge.knob_path.target_angle_rad))))
        for control in (self.handle_vertical, self.handle_tip_x, self.handle_tip_y, self.handle_depth,
                        self.pull_candidate, self.pull_target, self.pull_pregrasp,
                        self.pull_retreat, self.pull_retreat_up,
                        self.push_fraction, self.push_top, self.push_standoff, self.push_tip_x,
                        self.push_tip_y, self.push_surface, self.push_candidate,
                        self.push_pre_contact,
                        self.knob_azimuth, self.knob_depth,
                        self.knob_tip_x, self.door_start, self.door_target, self.knob_target):
            control.on_update(lambda event: self._regenerate())
        self.push_mode.on_update(lambda event: self._on_push_mode())
        self.focus = gui.add_button("对焦当前部件")
        self.focus.on_click(lambda event: self._focus())
        self.check_all = gui.add_button("检查全部候选与路径 IK", color="blue")
        self.check_all.on_click(lambda event: self._regenerate(check_all=True))
        self._update_parameter_visibility(mode)
        self._last_mode = mode
        self._syncing = False
        if restored or backend == "curobo":
            self._regenerate()
        else:
            self._show()

    def _apply_microwave_pose(self, pose: np.ndarray) -> None:
        """统一应用滑块、拖拽和重置位姿，并同步全部输入与 Knowledge。"""
        with self._update_lock:
            self._apply_microwave_pose_locked(pose)

    def _apply_microwave_pose_locked(self, pose: np.ndarray) -> None:
        try:
            set_preview_microwave_pose(self.case, pose)
        except ValueError as error:
            self.status.content = f"❌ 微波炉摆位无效：{error}"
            return
        from gene_agent.visualization.pick_bowl.scene import _pose
        actual = _body_pose(self.case.data, "microwave")
        relative = actual[:3, :3] @ self.placement_origin[:3, :3].T
        angles = Rotation.from_matrix(relative).as_euler("xyz", degrees=True)
        self._syncing = True
        try:
            for index, axis in enumerate("xyz"):
                value = float(actual[index, 3])
                control = self.microwave_xyz[axis]
                control.min = min(float(control.min), value)
                control.max = max(float(control.max), value)
                control.value = value
            for axis, value in zip(("roll", "pitch", "yaw"), angles):
                control = self.microwave_rpy[axis]
                control.min = min(float(control.min), float(value))
                control.max = max(float(control.max), float(value))
                control.value = float(value)
            self.microwave_control.wxyz, self.microwave_control.position = _pose(actual)
        finally:
            self._syncing = False
        self._regenerate()

    def _on_microwave_numbers(self) -> None:
        if self._syncing:
            return
        pose = np.eye(4)
        pose[:3, 3] = [float(self.microwave_xyz[axis].value) for axis in "xyz"]
        angles = [float(self.microwave_rpy[axis].value) for axis in ("roll", "pitch", "yaw")]
        pose[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix() @ self.placement_origin[:3, :3]
        self._apply_microwave_pose(pose)

    def _on_microwave_transform(self, event: Any) -> None:
        if self._syncing or getattr(event, "phase", "update") != "update":
            return
        position = np.asarray(self.microwave_control.position, dtype=float)
        wxyz = np.asarray(self.microwave_control.wxyz, dtype=float)
        if (position.shape != (3,) or wxyz.shape != (4,) or not np.isfinite(position).all()
                or not np.isfinite(wxyz).all() or not np.isclose(np.linalg.norm(wxyz), 1, atol=1e-3)):
            self.status.content = "❌ 拖拽产生无效位姿；场景未修改。"
            return
        pose = np.eye(4)
        pose[:3, 3] = position
        pose[:3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
        self._apply_microwave_pose(pose)

    def _on_push_mode(self) -> None:
        """右臂开／关门切换时，同步旧任务的阶段起止角。"""
        if self._syncing:
            return
        self._syncing = True
        try:
            self.door_start.value = 30. if self.push_mode.value == "right_open" else 85.
            self.door_target.value = 85. if self.push_mode.value == "right_open" else 0.
            self.push_candidate.value = 0
        finally:
            self._syncing = False
        self._regenerate()

    def _update_parameter_visibility(self, mode: str) -> None:
        """只展示当前 Knowledge 自己的参数，公共检查控件保持可见。"""
        self.handle_parameters_folder.visible = mode == "handle_grasp"
        self.handle_pull_folder.visible = mode == "handle_pull"
        self.door_push_folder.visible = mode == "door_push"
        self.door_push_transition_folder.visible = mode == "door_push"
        self.knob_parameters_folder.visible = mode == "knob_grasp"
        self.operation_target_folder.visible = mode in {"door_push", "knob_turn"}
        self.door_start.visible = mode == "door_push"
        self.door_target.visible = mode == "door_push"
        self.knob_target.visible = mode == "knob_turn"

    def _tuner_values(self) -> dict:
        """保留滑块原单位以便 JSON 截图直接复现 GUI 状态。"""
        return {
            "handle": {"vertical_offset_m": float(self.handle_vertical.value) / 1000,
                       "tip_x_deg": float(self.handle_tip_x.value),
                       "tip_y_deg": float(self.handle_tip_y.value),
                       "depth_m": float(self.handle_depth.value) / 1000},
            "handle_pull": {"target_angle_deg": float(self.pull_target.value),
                            "pregrasp_distance_m": float(self.pull_pregrasp.value) / 1000,
                            "retreat_distance_m": float(self.pull_retreat.value) / 1000,
                            "retreat_up_m": float(self.pull_retreat_up.value) / 1000},
            "handle_pull_selection": {"candidate_rank": int(self.pull_candidate.value)},
            "approximate_ik": {"enabled": bool(self.approximate_ik_enabled.value),
                               "max_position_error_m": float(self.approximate_position_mm.value) / 1000,
                               "max_rotation_error_deg": float(self.approximate_rotation_deg.value)},
            "door_push": {"hinge_fraction_offset": float(self.push_fraction.value),
                          "top_down_distance_offset_m": float(self.push_top.value) / 1000,
                          "standoff_offset_m": float(self.push_standoff.value) / 1000,
                          "tip_x_offset_deg": float(self.push_tip_x.value),
                          "tip_y_offset_deg": float(self.push_tip_y.value),
                          "surface": str(self.push_surface.value)},
            "door_push_transition": {"pre_contact_distance_m": float(self.push_pre_contact.value) / 1000,
                                     },
            "door_push_selection": {"start_candidate_rank": int(self.push_candidate.value)},
            "knob": {"azimuth_shift_deg": float(self.knob_azimuth.value),
                     "depth_m": float(self.knob_depth.value) / 1000,
                     "tip_x_deg": float(self.knob_tip_x.value)},
            "door_push_mode": str(self.push_mode.value),
            "door_start_angle_deg": float(self.door_start.value),
            "door_target_angle_deg": float(self.door_target.value),
            "knob_target_angle_deg": float(self.knob_target.value),
        }

    def _regenerate(self, *, check_all: bool = False) -> None:
        """滑块只检查当前位姿；整段 IK 由按钮触发，避免拖动事件排队。"""
        with self._update_lock:
            self._regenerate_locked(check_all=check_all)

    def _restore_last_valid_inputs(self) -> None:
        """无效滑块回退，保留最后一次可展示的 Knowledge 与场景。"""
        saved = self._last_valid_inputs
        if saved is None:
            return
        values = saved["tuner_values"]
        self._syncing = True
        try:
            controls = (
                (self.handle_vertical, values["handle"]["vertical_offset_m"] * 1000),
                (self.handle_tip_x, values["handle"]["tip_x_deg"]),
                (self.handle_tip_y, values["handle"]["tip_y_deg"]),
                (self.handle_depth, values["handle"]["depth_m"] * 1000),
                (self.pull_candidate, values["handle_pull_selection"]["candidate_rank"]),
                (self.pull_target, values["handle_pull"]["target_angle_deg"]),
                (self.pull_pregrasp, values["handle_pull"]["pregrasp_distance_m"] * 1000),
                (self.pull_retreat, values["handle_pull"]["retreat_distance_m"] * 1000),
                (self.pull_retreat_up, values["handle_pull"]["retreat_up_m"] * 1000),
                (self.approximate_ik_enabled, values["approximate_ik"]["enabled"]),
                (self.approximate_position_mm, values["approximate_ik"]["max_position_error_m"] * 1000),
                (self.approximate_rotation_deg, values["approximate_ik"]["max_rotation_error_deg"]),
                (self.push_fraction, values["door_push"]["hinge_fraction_offset"]),
                (self.push_top, values["door_push"]["top_down_distance_offset_m"] * 1000),
                (self.push_standoff, values["door_push"]["standoff_offset_m"] * 1000),
                (self.push_tip_x, values["door_push"]["tip_x_offset_deg"]),
                (self.push_tip_y, values["door_push"]["tip_y_offset_deg"]),
                (self.push_surface, values["door_push"]["surface"]),
                (self.push_candidate, values["door_push_selection"]["start_candidate_rank"]),
                (self.push_pre_contact, values["door_push_transition"]["pre_contact_distance_m"] * 1000),
                (self.knob_azimuth, values["knob"]["azimuth_shift_deg"]),
                (self.knob_depth, values["knob"]["depth_m"] * 1000),
                (self.knob_tip_x, values["knob"]["tip_x_deg"]),
                (self.push_mode, values["door_push_mode"]),
                (self.door_start, values["door_start_angle_deg"]),
                (self.door_target, values["door_target_angle_deg"]),
                (self.knob_target, values["knob_target_angle_deg"]),
                (self.mode, saved["mode"]),
                (self.ik_backend, saved["ik_backend"]),
                (self.candidate, saved["candidate_index"]),
                (self.waypoint, saved["waypoint_index"]),
            )
            for control, value in controls:
                control.value = value
            previous_pose = saved["microwave_pose_world"]
            if not np.allclose(_body_pose(self.case.data, "microwave"), previous_pose):
                set_preview_microwave_pose(self.case, previous_pose)
            from gene_agent.visualization.pick_bowl.scene import _pose
            self.microwave_control.wxyz, self.microwave_control.position = _pose(previous_pose)
            relative = previous_pose[:3, :3] @ self.placement_origin[:3, :3].T
            angles = Rotation.from_matrix(relative).as_euler("xyz", degrees=True)
            for index, axis in enumerate("xyz"):
                self.microwave_xyz[axis].value = float(previous_pose[index, 3])
            for axis, value in zip(("roll", "pitch", "yaw"), angles):
                self.microwave_rpy[axis].value = float(value)
        finally:
            self._syncing = False

    def _regenerate_locked(self, *, check_all: bool = False) -> None:
        if self._syncing:
            return
        values = self._tuner_values()
        try:
            refreshed = build_microwave_knowledge(
                self.root, self.case, handle_parameters=values["handle"],
                handle_pull_target_angle_rad=np.deg2rad(values["handle_pull"]["target_angle_deg"]),
                handle_pull_candidate_rank=values["handle_pull_selection"]["candidate_rank"],
                handle_pull_pregrasp_distance_m=values["handle_pull"]["pregrasp_distance_m"],
                handle_pull_retreat_distance_m=values["handle_pull"]["retreat_distance_m"],
                handle_pull_retreat_up_m=values["handle_pull"]["retreat_up_m"],
                door_push_pre_contact_distance_m=values["door_push_transition"]["pre_contact_distance_m"],
                door_push_start_candidate_rank=values["door_push_selection"]["start_candidate_rank"],
                knob_parameters=values["knob"], door_push_parameters=values["door_push"],
                door_push_mode=values["door_push_mode"],
                door_start_angle_rad=np.deg2rad(values["door_start_angle_deg"]),
                door_target_angle_rad=np.deg2rad(values["door_target_angle_deg"]),
                knob_target_angle_rad=np.deg2rad(values["knob_target_angle_deg"]),
                selected_knob_index=int(self.candidate.value),
                ik_scope="all" if check_all else "selected",
                selected_mode=str(self.mode.value), selected_waypoint=int(self.waypoint.value),
                ik_backend=str(self.ik_backend.value),
            )
        except Exception as error:
            self.last_rejected_error = f"{type(error).__name__}: {error}"
            if self._last_valid_inputs is None:
                self.current_error = self.last_rejected_error
                self.status.content = f"❌ 初始参数生成失败：{self.current_error}"
                self._write_state(error=self.current_error)
                return
            self._restore_last_valid_inputs()
            self.current_error = None
            self._show()
            self.status.content += f"  \n⚠️ 无效输入已退回上次有效状态：{self.last_rejected_error}"
            self._write_state()
            return
        self.knowledge = refreshed
        self.current_error = None
        self.last_rejected_error = None
        self.tcp_frame.visible = True
        self._show()

    def _camera_for(self, mode: str, *, overview: bool = False) -> dict:
        source = self.knowledge.handle_pose_world if mode.startswith("handle") or mode == "door_push" else self.knowledge.knob_pose_world
        # part +Z 是朝向用户的法向；略从侧上方看，夹爪与目标间隙更清楚。
        outward = source[:3, 2]
        outward = outward / np.linalg.norm(outward)
        focus = (self.case.data.geom(self.knowledge.settings["handle_visual_geom"]).xpos
                 if mode in {"handle_grasp", "handle_pull", "door_push"} else self.knowledge.knob_pose_world[:3, 3])
        if mode in {"handle_pull", "door_push"}:
            path = self.knowledge.handle_pull_path if mode == "handle_pull" else self.knowledge.door_path
            start = path.waypoints[0].tcp_pose_world[:3, 3]
            end = path.waypoints[-1].tcp_pose_world[:3, 3]
            focus = .5 * (start + end)
        if overview:
            # 把活动臂基座和操作部件同时纳入画面；近景仍沿用下面的部件相机。
            base = self.case.data.body(f"{self.case.snapshot['active_group']}_base_link").xpos
            center = .42 * np.asarray(focus) + .58 * base
            # 微波炉正面朝世界 -X；相机必须从 -X 一侧看，否则只见背板。
            position = center + np.array([-.85, .65, .62])
            return {"position": position.tolist(), "look_at": center.tolist(),
                    "up": [0., 0., 1.], "fov": .95}
        lateral = np.cross([0., 0., 1.], outward)
        distance = .42 if mode == "handle_pull" else .52 if mode == "door_push" else .38
        position = focus + distance * outward + .12 * lateral + np.array([0., 0., .09])
        return {"position": position.tolist(), "look_at": np.asarray(focus).tolist(),
                "up": [0., 0., 1.], "fov": .75 if mode == "handle_pull" else .9 if mode == "door_push" else .75}

    def _set_client_camera(self, client: Any) -> None:
        client.camera.position = self.camera["position"]
        client.camera.look_at = self.camera["look_at"]
        client.camera.up_direction = self.camera["up"]
        client.camera.fov = self.camera["fov"]
        client.camera.near = .01
        client.camera.far = 10.

    def _on_camera_update(self, camera: Any) -> None:
        self.camera = {"position": np.asarray(camera.position).tolist(),
                       "look_at": np.asarray(camera.look_at).tolist(),
                       "up": np.asarray(camera.up_direction).tolist(), "fov": float(camera.fov)}
        self._write_state()

    def _focus(self) -> None:
        self.camera = self._camera_for(str(self.mode.value), overview=bool(self.overview.value))
        for client in self.server.get_clients().values():
            self._set_client_camera(client)
        self._write_state()

    def _operation_path(self, mode: str):
        if mode == "handle_pull":
            return self.knowledge.handle_pull_path, self.knowledge.handle_pull_ik
        if mode == "door_push":
            return self.knowledge.door_path, self.knowledge.door_ik
        if mode == "knob_turn":
            return self.knowledge.knob_path, self.knowledge.knob_ik
        return None, None

    def _ik_quality(self, ik) -> dict | None:
        if ik is None or not bool(self.approximate_ik_enabled.value):
            return None
        return assess_pose_ik(
            ik,
            max_position_error_m=float(self.approximate_position_mm.value) / 1000,
            max_rotation_error_deg=float(self.approximate_rotation_deg.value),
        )

    def _clear_push_candidate_visuals(self) -> None:
        """移除上一帧门面候选和法向，避免切角后保留过期几何。"""
        for name in (
            "push_candidate_cloud",
            "push_candidate_directions",
            "push_normal_handle",
            "push_selected_contact",
            "push_precontact_point",
        ):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.remove()
                setattr(self, name, None)

    def _show(self) -> None:
        with self._update_lock:
            self._show_locked()

    def show_waypoint(self, index: int) -> None:
        """将已检查的操作路径定位到一帧，供界面和确定性视频共用。"""
        with self._update_lock:
            mode = str(self.mode.value)
            if mode not in {"handle_pull", "door_push", "knob_turn"}:
                raise ValueError("当前模式没有可回放的操作路径")
            path, results = self._operation_path(mode)
            if not 0 <= index < len(path.waypoints):
                raise IndexError(f"waypoint {index} 超出 0..{len(path.waypoints)-1}")
            if results[index] is None:
                raise ValueError("该 waypoint 尚未完成纯 IK 检查；请先检查全部路径")
            self._syncing = True
            try:
                self.waypoint.value = index
            finally:
                self._syncing = False
            self._show_locked()

    def _show_locked(self) -> None:
        if self._syncing or self.current_error is not None:
            return
        mode = str(self.mode.value)
        self._update_parameter_visibility(mode)
        mode_changed = mode != self._last_mode
        if mode_changed:
            self._last_mode = mode
        path, checked = self._operation_path(mode)
        self.candidate.visible = mode == "knob_grasp"
        self.waypoint.visible = path is not None
        self.pull_candidate.visible = mode == "handle_pull"
        self.push_candidate.visible = mode == "door_push"
        if mode == "handle_pull":
            candidate_max = len(self.knowledge.handle_pull_candidates) - 1
            self.pull_candidate.max = candidate_max
            if int(self.pull_candidate.value) > candidate_max:
                self._syncing = True
                self.pull_candidate.value = candidate_max
                self._syncing = False
        if mode == "door_push":
            candidate_max = len(self.knowledge.door_candidate_sets[0].candidates) - 1
            self.push_candidate.max = candidate_max
            if int(self.push_candidate.value) > candidate_max:
                self._syncing = True
                self.push_candidate.value = candidate_max
                self._syncing = False
        if path is not None:
            self.waypoint.max = len(path.waypoints)-1
            index = min(int(self.waypoint.value), len(path.waypoints)-1)
            if int(self.waypoint.value) != index:
                self._syncing = True
                self.waypoint.value = index
                self._syncing = False
            pose = path.waypoints[index].tcp_pose_world
            ik = checked[index]
            width = (self.knowledge.handle_pull_widths_m[index] if mode == "handle_pull" else
                     .06 if mode == "door_push" else self.knowledge.knob_grasp.closing_width_m)
            angle = path.waypoints[index].angle_rad
        elif mode == "handle_grasp":
            pose = self.knowledge.handle_grasp.tcp_pose_world
            ik = self.knowledge.handle_grasp_ik
            width = self.knowledge.handle_grasp.closing_width_m
            index, angle = 0, 0.
        else:
            index = int(self.candidate.value)
            grasp = self.knowledge.knob_grasps[index]
            pose, width = grasp.tcp_pose_world, grasp.closing_width_m
            ik = self.knowledge.knob_grasp_ik[index]
            angle = 0.

        mujoco.mj_copyData(self.preview_data, self.case.model, self.case.data)
        if path is not None:
            joint_name = (self.knowledge.settings["door_joint"] if mode in {"handle_pull", "door_push"}
                          else self.knowledge.settings["knob_joint"])
            joint = self.case.model.joint(joint_name)
            self.preview_data.qpos[joint.qposadr[0]] = angle
        # 纯 IK 的关节解用于运动学预览。不可达时也是最近尝试解，不能
        # 当成成功姿态；状态栏同时显示残差，红线连接实际与目标 TCP。
        if ik.joint_positions is not None:
            group = self.case.snapshot["active_group"]
            names = self.case.metadata["robot"]["joint_groups"][group]
            for name, value in zip(names, ik.joint_positions):
                self.preview_data.qpos[self.case.model.joint(name).qposadr[0]] = value
        group = self.case.snapshot["active_group"]
        gripper = self.case.metadata["robot"]["grippers"][group]
        if gripper.get("mode") == "parallel_slide_proxy":
            opening = float(np.clip(width, *gripper["width_range_m"]))
            for name in gripper["joint_names"]:
                self.preview_data.qpos[self.case.model.joint(name).qposadr[0]] = opening / 2
        mujoco.mj_forward(self.case.model, self.preview_data)
        self.scene.update_visuals(self.preview_data)
        self.gripper.select(pose, opening_width_m=width)
        self.gripper.set_visible(bool(self.target_ghost.value) and not ik.reachable)
        from gene_agent.visualization.pick_bowl.scene import _pose
        self.tcp_frame.wxyz, self.tcp_frame.position = _pose(pose)
        group = self.case.snapshot["active_group"]
        site = self.preview_data.site(self.case.metadata["robot"]["tcp_sites"][group])
        actual_pose = np.eye(4)
        actual_pose[:3, :3] = site.xmat.reshape(3, 3)
        actual_pose[:3, 3] = site.xpos
        self.actual_tcp_frame.wxyz, self.actual_tcp_frame.position = _pose(actual_pose)
        self.actual_tcp_frame.visible = not ik.reachable
        if self.tcp_gap_handle is not None:
            self.tcp_gap_handle.remove()
            self.tcp_gap_handle = None
        if not ik.reachable:
            gap = np.asarray([[actual_pose[:3, 3], pose[:3, 3]]], dtype=np.float32)
            self.tcp_gap_handle = self.server.scene.add_line_segments("/microwave_knowledge/tcp_gap", gap,
                                                                       (220, 62, 56), thickness=.005)
        if self.path_handle is not None:
            self.path_handle.remove()
            self.path_handle = None
        if path is not None:
            positions = np.asarray([item.tcp_pose_world[:3, 3] for item in path.waypoints], dtype=np.float32)
            segments = np.stack((positions[:-1], positions[1:]), axis=1)
            segments = segments[np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) > 1e-6]
            path_color = ((110, 145, 184) if any(item is None for item in checked) else
                          (34, 160, 88) if all(item.reachable for item in checked) else (230, 132, 39))
            self.path_handle = self.server.scene.add_line_segments("/microwave_knowledge/path", segments,
                                                                    path_color, thickness=.004)
        self._clear_push_candidate_visuals()
        if mode == "door_push":
            candidate_set_index = min(
                max(index - 1, 0),
                len(self.knowledge.door_candidate_sets) - 1,
            )
            candidate_set = self.knowledge.door_candidate_sets[candidate_set_index]
            selected_id = self.knowledge.door_selected_candidate_ids[candidate_set_index]
            selected_candidate = next(
                item for item in candidate_set.candidates
                if item.candidate_id == selected_id
            )
            # 全集合保留在 Knowledge 中；画面只取前 50 个姿态方向，避免
            # 数百条线遮住门板。重复位置仍可通过方向线看出 tip 搜索范围。
            preview_candidates = candidate_set.candidates[:50]
            points = np.asarray(
                [item.tcp_pose_world[:3, 3] for item in preview_candidates],
                dtype=np.float32,
            )
            directions = np.asarray(
                [item.tcp_pose_world[:3, 2] for item in preview_candidates],
                dtype=np.float32,
            )
            self.push_candidate_cloud = self.server.scene.add_point_cloud(
                "/microwave_knowledge/door_candidates/positions",
                points,
                (132, 92, 190),
                point_size=.004,
                point_shape="circle",
            )
            self.push_candidate_directions = self.server.scene.add_line_segments(
                "/microwave_knowledge/door_candidates/tcp_z",
                np.stack((points, points + .018 * directions), axis=1),
                (132, 92, 190),
                thickness=.0015,
            )
            normal_start = selected_candidate.tcp_pose_world[:3, 3]
            normal_end = normal_start + (
                float(self.push_pre_contact.value) / 1000.
            ) * selected_candidate.surface_normal_world
            self.push_normal_handle = self.server.scene.add_line_segments(
                "/microwave_knowledge/door_candidates/surface_normal",
                np.asarray([[normal_start, normal_end]], dtype=np.float32),
                (25, 145, 92),
                thickness=.006,
            )
            self.push_selected_contact = self.server.scene.add_icosphere(
                "/microwave_knowledge/door_candidates/selected_contact",
                radius=.010,
                color=(220, 76, 62),
                position=normal_start,
            )
            if candidate_set_index == 0:
                # Task1 只有第一个 contact 前存在 pre-contact；用独立端点
                # 标出它，避免绿色法向与整段路径重合后难以辨认。
                precontact_position = self.knowledge.door_path.waypoints[0].tcp_pose_world[:3, 3]
                self.push_precontact_point = self.server.scene.add_icosphere(
                    "/microwave_knowledge/door_candidates/precontact",
                    radius=.010,
                    color=(25, 178, 96),
                    position=precontact_position,
                )
        quality = self._ik_quality(ik)
        backend_name = "cuRobo IK" if ik.backend == "curobo" else "MuJoCo 位姿 IK"
        status = f"{backend_name} 可解" if ik.reachable else f"{backend_name} 未求得"
        if quality is not None and quality["approximate"]:
            status += ("；关节/自身约束可行的近似候选（未验场景碰撞和轨迹）" if ik.backend == "curobo"
                       else "；误差预算内的近似候选（未验碰撞）")
        failure_explanation = (explain_curobo_failure(ik)
                               if ik.backend == "curobo" and not ik.reachable else "")
        path_note = ("；整段 IK 尚未检查（灰蓝色路径）" if any(item is None for item in checked) else
                     f"；整段 {sum(item.reachable for item in checked)}/{len(checked)} waypoint 可解") if checked is not None else ""
        push = self.knowledge.door_waypoint_parameters[index] if mode == "door_push" else None
        phase_name = (self.knowledge.handle_pull_labels[index] if mode == "handle_pull" else
                      self.knowledge.door_push_labels[index] if mode == "door_push" else "")
        phase_label = ({"pregrasp": "预抓取", "grasp": "到达把手", "close": "闭爪",
                        "release": "松开把手", "retreat": "撤离把手"}.get(
            phase_name, f"拉门至 {np.degrees(angle):.0f}°"
        ) if mode == "handle_pull" else "")
        pull_tune = self._tuner_values()["handle_pull"] if mode == "handle_pull" else {}
        selected_handle_candidate = (
            next(
                item for item in self.knowledge.handle_pull_candidates
                if item.candidate_id == self.knowledge.handle_pull_selected_candidate_id
            )
            if mode == "handle_pull" else None
        )
        pull_candidate_detail = (
            f"；路线候选 `{selected_handle_candidate.candidate_id}`"
            f"（偏好序号 {selected_handle_candidate.preference_rank}，"
            f"局部 X {selected_handle_candidate.parameters['local_x_rotation_deg']:.0f}°，"
            f"局部 Y {selected_handle_candidate.parameters['local_y_rotation_deg']:.0f}°）"
            if selected_handle_candidate is not None else ""
        )
        pull_detail = (f"；局部 -Z {pull_tune['retreat_distance_m']*1000:.0f} mm，世界 +Z {pull_tune['retreat_up_m']*1000:.0f} mm"
                       if phase_name == "retreat" else "")
        push_phase_label = ({"pre_contact": "预接触", "contact": "接触门板"}.get(
            phase_name, f"推门至 {np.degrees(angle):.0f}°") if mode == "door_push" else "")
        push_transition = self._tuner_values()["door_push_transition"] if mode == "door_push" else {}
        if phase_name == "pre_contact":
            push_detail = (
                f"由候选 `{push['candidate_id']}` 沿门面朝外法向退 "
                f"{push_transition['pre_contact_distance_m']*1000:.0f} mm；"
                f"法向 {np.round(push['surface_normal_world'], 3).tolist()}，姿态不变。"
            )
        else:
            push_detail = (f"候选 `{push['candidate_id']}`（偏好序号 {push['preference_rank']}），"
                           f"{push['surface']} 面，门宽比例 {push['hinge_fraction']:.3f}，"
                           f"距顶部 {push['top_down_distance_m']*1000:.0f} mm，退让 {push['standoff_m']*1000:.0f} mm，"
                           f"局部 X 倾角 {push['tip_x_deg']:.1f}°，末段门板倾角 {push['panel_tilt_degrees']:.1f}°，"
                           f"门面朝外法向 {np.round(push['surface_normal_world'], 3).tolist()}。") if push else ""
        profile_note = ("**把手抓取：** 由左侧滑块参数生成；初值沿用旧 profile。  \n" if mode == "handle_grasp" else
                        f"**阶段 {index + 1}/{len(path.waypoints)}：{phase_label}**；"
                        f"夹爪开口 {width*1000:.0f} mm{pull_candidate_detail}{pull_detail}。  \n" if mode == "handle_pull" else
                        f"**Task1 {self.knowledge.settings['door_push_mode']} 候选 · 阶段 {index + 1}/{len(path.waypoints)}：{push_phase_label}：** {push_detail}  \n"
                        "**阶段边界：** pre-contact 前保持门为障碍；contact 后的允许接触与推完撤离由 Skill/Task 提供。  \n"
                        if mode == "door_push" else "")
        self.status.content = (f"**{MODE_LABELS[mode]}** · {status}{path_note}  \n"
                               f"{profile_note}"
                               f"**当前角度：** {np.degrees(angle):.1f}°；**TCP：** {np.round(pose[:3, 3], 3).tolist()} m  \n"
                               f"**{'IK 解误差' if ik.reachable else '画面 TCP 与目标误差'}：** {ik.position_error_m*1000:.1f} mm，{ik.rotation_error_deg:.1f}°  \n"
                               f"{failure_explanation}"
                               f"**机械臂：** {'当前 waypoint 的 IK 解' if ik.reachable else '诊断候选姿态；红线连接实际 TCP 与目标 TCP' if ik.joint_positions is not None else '未返回候选关节姿态；显示当前关节状态'}。  \n"
                               +
                               ("**边界：** cuRobo IK 不检查场景碰撞；保留关节限位和自身碰撞，未规划连续运动或验证接触力。"
                                if ik.backend == "curobo" else
                                "**边界：** 只检查位姿 IK；未检查碰撞、连续运动或接触力。"))
        if mode_changed:
            self._focus()
        self._last_valid_inputs = {
            "tuner_values": self._tuner_values(), "mode": mode,
            "candidate_index": int(self.candidate.value),
            "waypoint_index": int(self.waypoint.value),
            "ik_backend": str(self.ik_backend.value),
            "microwave_pose_world": _body_pose(self.case.data, "microwave").copy(),
        }
        self._write_state()

    def _write_state(self, *, error: str | None = None) -> None:
        if self.state_path is None or not hasattr(self, "mode") or not hasattr(self, "status"):
            return
        error = error or self.current_error
        if error is not None:
            write_live_state(self.state_path, {
                "schema_version": 1, "tool": "microwave_knowledge", "mode": str(self.mode.value),
                "run": str(self.run_dir), "scene_revision": self.case.revision,
                "ik_backend": str(self.ik_backend.value),
                "tuner_values": self._tuner_values(), "error": error, "input_valid": False,
                "microwave_pose_world": _body_pose(self.case.data, "microwave").tolist(),
                "stale_candidate_hidden": True, "camera": self.camera,
            })
            return
        mode = str(self.mode.value)
        path, path_results = self._operation_path(mode)
        index = min(int(self.waypoint.value), len(path.waypoints)-1) if path is not None else 0
        pose = (path.waypoints[index].tcp_pose_world if path is not None else
                self.knowledge.handle_grasp.tcp_pose_world if mode == "handle_grasp" else
                self.knowledge.knob_grasps[int(self.candidate.value)].tcp_pose_world)
        results = (path_results if path is not None else
                   (self.knowledge.handle_grasp_ik,) if mode == "handle_grasp" else
                   self.knowledge.knob_grasp_ik)
        selected_ik = results[index] if path is not None or mode == "handle_grasp" else results[int(self.candidate.value)]
        door_candidate_set = None
        selected_door_candidate = None
        selected_handle_candidate = None
        if mode == "handle_pull":
            selected_handle_candidate = next(
                item for item in self.knowledge.handle_pull_candidates
                if item.candidate_id == self.knowledge.handle_pull_selected_candidate_id
            )
        if mode == "door_push":
            candidate_set_index = min(
                max(index - 1, 0),
                len(self.knowledge.door_candidate_sets) - 1,
            )
            door_candidate_set = self.knowledge.door_candidate_sets[candidate_set_index]
            selected_id = self.knowledge.door_selected_candidate_ids[candidate_set_index]
            selected_door_candidate = next(
                item for item in door_candidate_set.candidates
                if item.candidate_id == selected_id
            )
        write_live_state(self.state_path, {
            "schema_version": 1, "tool": "microwave_knowledge", "mode": mode,
            "ik_backend": str(self.ik_backend.value),
            "input_valid": True,
            "last_rejected_error": self.last_rejected_error,
            "run": str(self.run_dir), "scene_revision": self.case.revision,
            "scene_snapshot_sha256": self.case.snapshot.get("snapshot_sha256"),
            "calibration": {"handle_pose_world": self.knowledge.handle_pose_world.tolist(),
                            "knob_pose_world": self.knowledge.knob_pose_world.tolist(),
                            "source": "current_calibrated_urdf_visual_and_metadata"},
            "profile_overrides": self.knowledge.settings["profile_overrides"],
            "selected_grasp_profile": (self.knowledge.settings["handle_profile"]
                                       if mode in {"handle_grasp", "handle_pull"} else
                                       self.knowledge.settings["knob_profile"] if mode != "door_push" else None),
            "selected_grasp_parameters": dict(
                self.knowledge.door_waypoint_parameters[index] if mode == "door_push" else
                {**self.knowledge.handle_grasp.parameters,
                 **selected_handle_candidate.parameters,
                 "candidate_id": selected_handle_candidate.candidate_id,
                 "preference_rank": selected_handle_candidate.preference_rank,
                 "candidate_source": selected_handle_candidate.source}
                if mode == "handle_pull" else
                self.knowledge.handle_grasp.parameters if mode == "handle_grasp" else
                self.knowledge.knob_grasps[int(self.candidate.value)].parameters
            ),
            "selected_phase": (self.knowledge.handle_pull_labels[index] if mode == "handle_pull" else
                               self.knowledge.door_push_labels[index] if mode == "door_push" else None),
            "path_phase_names": (list(self.knowledge.handle_pull_labels) if mode == "handle_pull" else
                                 list(self.knowledge.door_push_labels) if mode == "door_push" else None),
            "path_gripper_widths_m": list(self.knowledge.handle_pull_widths_m) if mode == "handle_pull" else None,
            "handle_pull_candidate_count": (len(self.knowledge.handle_pull_candidates)
                                             if mode == "handle_pull" else None),
            "selected_handle_pull_candidate_id": (
                self.knowledge.handle_pull_selected_candidate_id
                if mode == "handle_pull" else None
            ),
            "selected_handle_pull_candidate_rank": (
                selected_handle_candidate.preference_rank
                if selected_handle_candidate is not None else None
            ),
            "handle_pull_candidate_ids": (
                [item.candidate_id for item in self.knowledge.handle_pull_candidates]
                if mode == "handle_pull" else None
            ),
            "ik_quality": self._ik_quality(selected_ik),
            "door_selection_style": "task1_candidate_set" if mode == "door_push" else None,
            "door_candidate_count": (len(door_candidate_set.candidates)
                                     if door_candidate_set is not None else None),
            "selected_door_candidate_id": (selected_door_candidate.candidate_id
                                            if selected_door_candidate is not None else None),
            "selected_door_candidate_rank": (selected_door_candidate.preference_rank
                                              if selected_door_candidate is not None else None),
            "selected_door_surface_normal_world": (
                selected_door_candidate.surface_normal_world.tolist()
                if selected_door_candidate is not None else None
            ),
            "previewed_door_candidate_ids": (
                [item.candidate_id for item in door_candidate_set.candidates[:50]]
                if door_candidate_set is not None else None
            ),
            "door_waypoint_parameters": list(self.knowledge.door_waypoint_parameters),
            "tuner_values": self._tuner_values(),
            "microwave_pose_world": _body_pose(self.case.data, "microwave").tolist(),
            "candidate_index": int(self.candidate.value), "waypoint_index": index,
            "selected_tcp_pose_world": pose.tolist(),
            "ik_scope": "curobo_pose_self_collision_no_scene_no_path" if self.ik_backend.value == "curobo" else "pose_only_no_collision",
            "ik_constraint_feasible": [item.constraint_feasible if item is not None else None for item in results],
            "ik_failure_category": [item.failure_category if item is not None else None for item in results],
            "ik_collision_check": selected_ik.collision_check if selected_ik is not None else None,
            "ik_diagnostic": selected_ik.diagnostic if selected_ik is not None else None,
            "ik_reachable": [item.reachable if item is not None else None for item in results],
            "ik_position_error_m": [item.position_error_m if item is not None else None for item in results],
            "ik_rotation_error_deg": [item.rotation_error_deg if item is not None else None for item in results],
            "path": None if path is None else path.to_dict(),
            "camera": self.camera, "robot_visual_visible": True,
            "robot_preview_kind": "ik_solution" if results[index if path is not None else int(self.candidate.value) if mode == "knob_grasp" else 0].reachable else "best_ik_attempt",
            "robot_overview": bool(self.overview.value),
            "gripper_palm_visible": bool(self.palm.value),
            "target_gripper_ghost_visible": bool(self.target_ghost.value),
        })


__all__ = ["MicrowaveKnowledgeApp", "MODES"]
