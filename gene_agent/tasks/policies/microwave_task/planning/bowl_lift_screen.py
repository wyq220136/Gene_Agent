"""候选筛选时预测闭爪后的 lift 可行性；执行仍消费实测闭爪状态。"""
from pathlib import Path
import json

import mujoco
import numpy as np

from gene_agent.motion import CuroboPlannerAdapter, load_realman_motion_config
from gene_agent.motion.model import (
    CollisionPolicy, MotionOperation, MotionRequest, PlannerMode,
    PlanningSceneRef, PoseTarget,
)
from gene_agent.motion.curobo.scene import export_collision_scene
from .bowl_lift import make_bowl_lift_candidates


def screen_bowl_lift(case, proposal, approach_result, *, fk, distance_m,
                     base_position_world, attachment_profile, pool, seed, output):
    """用预测 approach 终点检查下一阶段合同，不声明物理夹持成功。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    end = approach_result.goal_state
    tcp = np.asarray(fk(end), dtype=float)
    frozen = mujoco.MjData(case.model)
    mujoco.mj_copyData(frozen, case.model, case.data)
    for name, value in zip(end.joint_names, end.positions_rad):
        frozen.qpos[case.model.joint(name).qposadr[0]] = value
    robot = case.metadata["robot"]
    for name in robot["grippers"]["right"]["joint_names"]:
        frozen.qpos[case.model.joint(name).qposadr[0]] = proposal.closing_width_m / 2.
    mujoco.mj_forward(case.model, frozen)
    bowl = frozen.body("bowl")
    object_pose = np.eye(4)
    object_pose[:3, :3] = bowl.xmat.reshape(3, 3)
    object_pose[:3, 3] = bowl.xpos
    local = np.asarray(attachment_profile["sphere_centers_object_m"], dtype=float)
    centers = (object_pose[:3, :3] @ local.T).T + object_pose[:3, 3]
    spheres = np.column_stack((
        centers, np.full(len(centers), attachment_profile["sphere_radius_m"]),
    ))
    snapshot = export_collision_scene(
        case.model, frozen, robot, "right", output / "collision",
        transferred_objects={"bowl": spheres.tolist()},
    )
    settings = load_realman_motion_config(
        "right", robot, gripper_snapshot=snapshot["gripper_snapshot"],
    )
    world_link = tcp @ np.linalg.inv(np.asarray(settings["link_from_tcp"]))
    link_world = np.linalg.inv(world_link)
    centers_link = (link_world[:3, :3] @ centers.T).T + link_world[:3, 3]
    settings["attachment_spheres_tool"] = np.column_stack(
        (centers_link, spheres[:, 3]),
    ).tolist()
    lease = pool.acquire(
        settings, snapshot, planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK,
        random_seed=seed,
    )
    adapter = CuroboPlannerAdapter(
        lease.planner, group="right", scene_revision=case.revision,
    )
    scene = PlanningSceneRef(
        case.revision, snapshot["snapshot_sha256"], snapshot["scope"],
    )
    rows = []
    selected = None
    for index, target in enumerate(make_bowl_lift_candidates(
            tcp, distance_m, base_position_world)):
        request = MotionRequest(
            f"{proposal.candidate_id}:predicted_lift_{index:02d}",
            MotionOperation.LINEAR_TCP, "right", end,
            PoseTarget("world_tcp", target), scene, CollisionPolicy(), seed=seed,
        )
        result = adapter.plan(request)
        rows.append({"index": index, "request": request.to_dict(),
                     "result": result.to_dict()})
        if result.success and result.trajectory is not None:
            selected = index
            break
    evidence = {
        "success": selected is not None,
        "source": "predicted_approach_endpoint_and_current_object_pose",
        "claim": "候选前瞻；未验证闭爪接触、实际抬升或后续完整任务",
        "execution_must_reobserve_and_replan": True,
        "selected_index": selected,
        "attempts": rows,
        "planner_pool": lease.to_dict(),
        "T_tcp_bowl": (np.linalg.inv(tcp) @ object_pose).tolist(),
        "attachment_coverage": attachment_profile.get("coverage"),
    }
    path = output / "result.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf8")
    return {
        "success": evidence["success"], "evidence_ref": str(path),
        "selected_index": selected, "candidates_checked": len(rows),
        "source": evidence["source"], "claim": evidence["claim"],
        "execution_must_reobserve_and_replan": True,
    }
