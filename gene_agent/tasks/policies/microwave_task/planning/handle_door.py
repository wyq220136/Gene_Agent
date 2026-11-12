"""微波炉门的逐阶段规划与运动学回放；开门、关门分别生成独立 run。"""

from __future__ import annotations
from gene_agent.motion.curobo.scene import planner_scene_options

import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import load_concept
from gene_agent.knowledge.grasp import generate_grasps, load_grasp_profiles
from gene_agent.knowledge.operation import AxisSpec
from gene_agent.motion import (
    CuroboPlannerAdapter, JointState, MotionRequest, PlannerMode, PlanningSceneRef,
    PoseTarget, SynchronousPlanner, load_realman_motion_config,
)
from gene_agent.motion.curobo.scene import _descendant, export_collision_scene
from gene_agent.simulation import build_scene
from gene_agent.skills.library.door.planning import DoorSkill, DoorSkillSettings


ROOT = Path(__file__).resolve().parents[5]


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _pose(position, rotation) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation).reshape(3, 3)
    result[:3, 3] = position
    return result


def _angle(model, data, joint_name: str) -> float:
    return float(data.qpos[model.joint(joint_name).qposadr[0]])


def _width(model, data, descriptor: dict, width_m: float) -> None:
    lower, upper = descriptor["width_range_m"]
    if not lower <= width_m <= upper:
        raise ValueError(f"夹爪开口 {width_m} m 超出 [{lower}, {upper}] m")
    for name in descriptor["joint_names"]:
        data.qpos[model.joint(name).qposadr[0]] = width_m / 2


def _sample(model, data, robot: dict, group: str, names: tuple[str, ...],
            joint_name: str, time_s: float) -> dict:
    mujoco.mj_forward(model, data)
    tcp = data.site(robot["tcp_sites"][group])
    door = data.body("microwave_door")
    return {
        "time_s": float(time_s), "qpos": data.qpos.tolist(),
        "robot_group": group, "joint_names": list(names),
        "joint_positions_rad": [_angle(model, data, name) for name in names],
        "gripper_width_m": sum(_angle(model, data, name)
                               for name in robot["grippers"][group]["joint_names"]),
        "articulation_joint": joint_name,
        "articulation_angle_rad": _angle(model, data, joint_name),
        "tcp_pose_world": _pose(tcp.xpos, tcp.xmat).tolist(),
        "object_poses_world": {"microwave_door": _pose(door.xpos, door.xmat).tolist()},
    }


def _arc_metrics(model, data, names, positions, axis: AxisSpec,
                 tcp_site: str, expected_delta: float) -> tuple[dict, list[float]]:
    """用真实 FK 逐帧检查绕轴半径、轴向位移和旋转方向。"""
    points, matrices = [], []
    for row in positions:
        for name, value in zip(names, row):
            data.qpos[model.joint(name).qposadr[0]] = float(value)
        mujoco.mj_forward(model, data)
        site = data.site(tcp_site)
        points.append(np.asarray(site.xpos).copy())
        matrices.append(np.asarray(site.xmat).reshape(3, 3).copy())
    relative = np.asarray(points) - axis.point
    axial = relative @ axis.direction
    radii = np.linalg.norm(relative - np.outer(axial, axis.direction), axis=1)
    signed, off_axis = [], []
    for matrix in matrices:
        vector = Rotation.from_matrix(matrix @ matrices[0].T).as_rotvec()
        angle = float(vector @ axis.direction)
        signed.append(angle)
        off_axis.append(float(np.linalg.norm(vector - angle * axis.direction)))
    sign = 1 if expected_delta >= 0 else -1
    metrics = {
        "max_axis_radius_change_m": float(max(abs(radii - radii[0]))),
        "max_axial_drift_m": float(max(abs(axial - axial[0]))),
        "max_rotation_off_axis_rad": float(max(off_axis)),
        "max_angle_regression_rad": float(max(0., -min(np.diff(signed) * sign))),
        "tcp_rotation_rad": float(signed[-1]),
        "expected_rotation_rad": float(expected_delta),
    }
    return metrics, signed


def _door_contacts(model, data, robot: dict, group: str) -> dict:
    """逐帧检查被静态规划排除的门；只允许指垫与把手发生接触。

    这里检查 MuJoCo 当前 qpos 的离散接触，不能替代连续扫掠验证或抓取力验证。
    """
    arm_root = model.body(robot["group_root_bodies"][group]).id
    door_root = model.body("microwave_door").id
    pad_names = set(robot["grippers"][group]["pad_geom_names"])
    allowed, forbidden = 0, []
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        first_body, second_body = int(model.geom_bodyid[first]), int(model.geom_bodyid[second])
        if _descendant(model, first_body, arm_root) and _descendant(model, second_body, door_root):
            arm_geom, door_geom = first, second
        elif _descendant(model, second_body, arm_root) and _descendant(model, first_body, door_root):
            arm_geom, door_geom = second, first
        else:
            continue
        arm_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, arm_geom)
        door_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, door_geom)
        if arm_name in pad_names and door_name == "microwave_handle_grip_collision":
            allowed += 1
        else:
            forbidden.append(f"{arm_name} ↔ {door_name}")
    return {"allowed_pad_handle_contacts": allowed,
            "forbidden_contacts": sorted(set(forbidden))}


def run_door_planning(output_dir: str | Path, *, action: str,
                      task_path: str | Path = ROOT / "data/task_1/skills/door_task.json") -> dict:
    """按显式门轴规划；关节角仅运动学跟随，不把预览声明为接触物理。"""
    if action not in {"open", "close"}:
        raise ValueError("门 Skill action 只支持 open 或 close")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    task_source = Path(task_path).resolve()
    task = json.loads(task_source.read_text(encoding="utf-8"))
    if task.get("schema_version") != 1 or task.get("instance_id") != "microwave_door":
        raise ValueError("门任务配置格式或实例 ID 不正确")
    group, seed = task["robot_group"], int(task["seed"])
    joint_name = task["joint_name"]
    microwave_pose = np.asarray(task["microwave_pose_world"], dtype=float)
    scene_poses = {name: np.asarray(pose, dtype=float)
                   for name, pose in task.get("object_poses_world", {}).items()}
    if "microwave" in scene_poses and not np.allclose(
            scene_poses["microwave"], microwave_pose, atol=1e-9):
        raise ValueError("门配置中的微波炉位姿与场景物体位姿不一致")
    scene_poses["microwave"] = microwave_pose
    model, data, metadata = build_scene(
        output / "source_scene", seed=seed,
        pose_overrides=scene_poses,
        turntable_height_offset_m=float(task.get("turntable_height_offset_m", 0.)),
        microwave_floor_height_offset_m=float(
            task.get("microwave_floor_height_offset_m", 0.)),
        simulation_attachment_pairs=task.get("simulation_attachment_pairs"),
    )
    robot = metadata["robot"]
    names = tuple(robot["joint_groups"][group])
    descriptor = robot["grippers"][group]
    initial_joints = task.get("start_robot_joint_positions_rad")
    if initial_joints is not None:
        if not isinstance(initial_joints, list) or len(initial_joints) != len(names) or not np.isfinite(initial_joints).all():
            raise ValueError("指定的门 Skill 初始机器人关节值必须与活动关节组一致且有限")
        for name, value in zip(names, initial_joints):
            data.qpos[model.joint(name).qposadr[0]] = float(value)
    start_angle = 0. if action == "open" else float(task["close_start_angle_rad"])
    target_angle = float(task["open_target_angle_rad"]) if action == "open" else 0.
    joint = model.joint(joint_name)
    low, high = map(float, joint.range)
    if not low <= start_angle <= high or not low <= target_angle <= high:
        raise ValueError("门初角或目标角超出编译后关节限位")
    data.qpos[joint.qposadr[0]] = start_angle
    _width(model, data, descriptor, float(task["open_width_m"]))
    mujoco.mj_forward(model, data)
    axis_input = task["door_axis_in_microwave"]
    axis = AxisSpec(
        microwave_pose[:3, :3] @ axis_input["point_m"] + microwave_pose[:3, 3],
        microwave_pose[:3, :3] @ axis_input["direction"], "world", "task_config",
    )
    door = data.body("microwave_door")
    compiled_axis = door.xmat.reshape(3, 3) @ model.jnt_axis[joint.id]
    point_error = float(np.linalg.norm(axis.point - door.xpos))
    axis_error = float(np.linalg.norm(axis.direction - compiled_axis))
    if point_error > .001 or axis_error > .01:
        raise ValueError(f"任务门轴与编译关节失配：{point_error:.4f} m / {axis_error:.4f}")
    concept_path = ROOT / task["concept_path"]
    profile_path = ROOT / task["grasp_profile_path"]
    concept = load_concept(concept_path, part_name="microwave_handle")
    profile = load_grasp_profiles(profile_path)[task["grasp_profile_name"]]
    candidates = generate_grasps(concept.concept, microwave_pose, profile).proposals
    if not candidates:
        raise ValueError("门把手 profile 未生成 grasp 候选")
    candidate_index = int(task.get("candidate_index", 0))
    if not 0 <= candidate_index < len(candidates):
        raise ValueError("门把手 candidate_index 超出候选范围")
    candidate = candidates[candidate_index]
    handle_closed = microwave_pose @ concept.concept.pose_in_object @ concept.concept.grip_segment.pose_in_trifold
    site = data.site(task["handle_site_name"])
    rotate_start = Rotation.from_rotvec(axis.direction * start_angle).as_matrix()
    expected_center = axis.point + rotate_start @ (handle_closed[:3, 3] - axis.point)
    handle_error = float(np.linalg.norm(site.xpos - expected_center))
    if handle_error > .001:
        raise ValueError(f"门把手 Concept 与 MuJoCo site 失配：{handle_error:.4f} m")
    settings = DoorSkillSettings(
        float(task["approach_distance_m"]), target_angle,
        float(task["rotation_step_rad"]), float(task["open_width_m"]),
        float(task.get("close_width_m", candidate.closing_width_m)),
        float(task["release_width_m"]),
    )
    base_grasp = candidate.tcp_pose_world
    grasp = np.array(base_grasp, copy=True)
    grasp[:3, :3] = rotate_start @ base_grasp[:3, :3]
    grasp[:3, 3] = axis.point + rotate_start @ (base_grasp[:3, 3] - axis.point)
    stages = DoorSkill().targets(grasp, axis, start_angle_rad=start_angle, settings=settings)
    report_name = f"skill_{action}_door_result.json"
    report = {
        "schema_version": 1, "skill": f"{action}_door", "instance_id": "microwave_door",
        "source_format": "kinematic_plan", "success": False,
        "claim": "机器人为 cuRobo 规划轨迹；门关节为 MuJoCo 运动学跟随，未验证接触物理",
        "scene_revision": 0, "seed": seed, "robot_group": group,
        "model_ref": "source_scene/scene.xml",
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in (("task", task_source), ("concept", concept_path),
                                            ("grasp_profile", profile_path))},
        "selected_candidate_id": candidate.candidate_id,
        "candidate_count": len(candidates), "task_config_ref": "task_config.json",
        "axis_point_world": axis.point.tolist(),
        "axis_direction_world": axis.direction.tolist(),
        "axis_joint_error_m": point_error, "axis_direction_error": axis_error,
        "concept_site_error_m": handle_error,
        "joint_name": joint_name, "joint_start_angle_rad": start_angle,
        "joint_target_angle_rad": target_angle,
        "stage_contract": [{"name": stage.name,
                            "operation": stage.operation.value if stage.operation else "gripper_command"}
                           for stage in stages],
        "stages": [],
    }
    shutil.copyfile(task_source, output / "task_config.json")
    _write(output / "grasp_candidates.json", {
        "source_profile": str(profile_path), "resolved_profile": profile.to_dict(),
        "selected_candidate_id": candidate.candidate_id,
        "candidates": [{"candidate_id": item.candidate_id,
                        "tcp_pose_world": item.tcp_pose_world.tolist(),
                        "closing_width_m": item.closing_width_m}
                       for item in candidates],
    })
    current_width = settings.open_width_m
    failed = False
    for index, stage in enumerate(stages):
        moving = stage.name.startswith("swing_")
        snapshot = export_collision_scene(
            model, data, robot, group, output / "collision" / stage.name,
            moving_body_names=("microwave_door",) if moving else (),
        )
        scene_ref = PlanningSceneRef(index, snapshot["snapshot_sha256"], snapshot["scope"])
        row = {
            "name": stage.name, "status": "not_executed" if failed else "planned",
            "operation": stage.operation.value if stage.operation else "gripper_command",
            "collision_snapshot_ref": f"collision/{stage.name}/collision_snapshot.json",
            "start_angle_rad": stage.start_angle_rad, "end_angle_rad": stage.end_angle_rad,
            "gripper_width_m": stage.gripper_width_m,
            "target": (PoseTarget("world_tcp", stage.tcp_pose_world).to_dict()
                       if stage.tcp_pose_world is not None else None),
            "request": None, "result": None, "planned_replay_samples": [],
            "moving_body_names": snapshot["moving_body_names"],
        }
        if failed:
            report["stages"].append(row)
            continue
        start_qpos = data.qpos.copy()
        if stage.operation is None:
            steps = 11 if stage.name in {"close", "release"} else 1
            for i in range(steps):
                fraction = i / max(steps - 1, 1)
                _width(model, data, descriptor,
                       current_width + fraction * (stage.gripper_width_m - current_width))
                row["planned_replay_samples"].append(
                    _sample(model, data, robot, group, names, joint_name, .4 * fraction))
            current_width = stage.gripper_width_m
        else:
            config = load_realman_motion_config(
                group, robot, gripper_snapshot=snapshot.get("gripper_snapshot"))
            planner = SynchronousPlanner(
                **config, **planner_scene_options(snapshot),
                planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK, random_seed=seed,
            )
            adapter = CuroboPlannerAdapter(planner, group=group, scene_revision=index)
            start = JointState(names, tuple(_angle(model, data, name) for name in names))
            request = MotionRequest(
                f"{candidate.candidate_id}:{stage.name}", stage.operation,
                group, start, PoseTarget("world_tcp", stage.tcp_pose_world),
                scene_ref, seed=seed,
            )
            result = adapter.plan(request)
            row["request"], row["result"] = request.to_dict(), result.to_dict()
            if not result.success or result.trajectory is None:
                row["status"] = "failed"
                failed = True
            else:
                trajectory = result.trajectory
                turns = None
                if stage.name.startswith("swing_"):
                    metrics, turns = _arc_metrics(
                        model, data, trajectory.joint_names,
                        trajectory.positions_rad, axis, robot["tcp_sites"][group],
                        stage.end_angle_rad - stage.start_angle_rad,
                    )
                    row["axis_path_validation"] = metrics
                    if (metrics["max_axis_radius_change_m"] > .005 or
                            metrics["max_axial_drift_m"] > .005 or
                            metrics["max_rotation_off_axis_rad"] > .05 or
                            metrics["max_angle_regression_rad"] > .03 or
                            abs(metrics["tcp_rotation_rad"] -
                                metrics["expected_rotation_rad"]) > .05):
                        row["status"] = "failed"
                        row["axis_path_failure"] = "规划轨迹偏离门轴或旋转方向"
                        failed = True
                if not failed:
                    forbidden_contacts = set()
                    allowed_pad_handle_contacts = 0
                    for sample_index, (time_s, positions) in enumerate(zip(
                            trajectory.times_s, trajectory.positions_rad)):
                        for name, value in zip(trajectory.joint_names, positions):
                            data.qpos[model.joint(name).qposadr[0]] = float(value)
                        if turns is not None:
                            fraction = float(np.clip(
                                turns[sample_index] /
                                (stage.end_angle_rad - stage.start_angle_rad), 0., 1.))
                            data.qpos[joint.qposadr[0]] = (
                                stage.start_angle_rad + fraction *
                                (stage.end_angle_rad - stage.start_angle_rad))
                            if sample_index == len(trajectory.positions_rad) - 1:
                                data.qpos[joint.qposadr[0]] = stage.end_angle_rad
                        row["planned_replay_samples"].append(
                            _sample(model, data, robot, group, names, joint_name, time_s))
                        if moving:
                            contacts = _door_contacts(model, data, robot, group)
                            forbidden_contacts.update(contacts["forbidden_contacts"])
                            allowed_pad_handle_contacts += contacts["allowed_pad_handle_contacts"]
                    if moving:
                        row["dynamic_door_contact_check"] = {
                            "sample_count": len(row["planned_replay_samples"]),
                            "allowed_pad_handle_contacts": allowed_pad_handle_contacts,
                            "forbidden_contacts": sorted(forbidden_contacts),
                            "scope": "MuJoCo 离散 qpos 接触；未验证连续扫掠和实际抓取力",
                        }
                        if forbidden_contacts:
                            row["status"] = "failed"
                            row["dynamic_door_contact_failure"] = "活动臂与运动门存在非指垫把手接触"
                            row["planned_replay_samples"] = []
                            failed = True
        if failed:
            data.qpos[:] = start_qpos
            mujoco.mj_forward(model, data)
        report["stages"].append(row)
        _write(output / report_name, report)
    report["final_joint_angle_rad"] = _angle(model, data, joint_name)
    report["success"] = not failed and abs(report["final_joint_angle_rad"] - target_angle) < 1e-8
    report["completion_basis"] = "planned_kinematic_target_only"
    _write(output / report_name, report)
    return report


__all__ = ["run_door_planning"]
