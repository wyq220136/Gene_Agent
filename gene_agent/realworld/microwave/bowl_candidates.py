"""AffordToolAct 真机 Bowl 输入；复用共享 Ring 几何，不依赖仿真接触支点。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from gene_agent.knowledge import bowl_grasp_candidates
from gene_agent.knowledge.geometry import validate_pose
from gene_agent.tasks.policies.microwave_task.planning.grasp import legacy_bowl_proposals

PROFILE_PATH = Path(__file__).resolve().parents[3] / "data/task_1/realworld/pick_bowl.json"


def observed_bowl_candidates(collision_scene, bowl_pose_world, *, pose_source):
    """先在 SDK frame 生成、排序和限高，再统一变换到规划 world。

    AffordToolAct 仅偏好 SDK -X、其次 -Y，不删除背侧/近竖直候选。
    限高发生在派生 pregrasp 之前，保留原始目标和修改量供现场审核。
    """
    raw = PROFILE_PATH.read_bytes()
    profile = json.loads(raw)
    if profile.get("schema_version") != 1:
        raise ValueError("不支持的真机 Bowl 候选配置版本")
    provenance = collision_scene["provenance"]
    transform = provenance.get("T_world_sdk")
    if transform is None:
        if collision_scene["snapshot"].get("world_frame") != "right_sdk_base":
            raise ValueError("真机 Bowl 候选缺少 T_world_sdk，不能猜测 SDK/world 变换")
        transform = np.eye(4)
    world_sdk = validate_pose(transform, field_name="T_world_sdk")
    sdk_world = np.linalg.inv(world_sdk)
    world_bowl = validate_pose(bowl_pose_world, field_name="T_world_bowl")
    sdk_bowl = sdk_world @ world_bowl
    if "table_pose_world" not in provenance or "accepted_table_thickness_m" not in provenance:
        raise ValueError("真机 Bowl 候选需要已确认的桌面位姿和厚度，不能使用仿真名义高度")
    sdk_table = sdk_world @ validate_pose(provenance["table_pose_world"], field_name="T_world_table")
    thickness = float(provenance["accepted_table_thickness_m"])
    clearance = np.asarray(profile["table_clearance_m"], dtype=float)
    if (not np.isfinite(thickness) or thickness <= 0 or clearance.shape != (2,)
            or not np.isfinite(clearance).all() or not 0 <= clearance[0] <= clearance[1]):
        raise ValueError("真机 Bowl 桌面厚度/净空配置无效")
    from .bowl_pose_check import check_bowl_pose_table, BowlPoseTableError
    try:
        pose_consistency = check_bowl_pose_table(sdk_bowl, sdk_table, thickness)
    except BowlPoseTableError as error:
        error.diagnostics.update(source_frame_number=provenance.get("source_frame_number"),
            source_timestamp_ms=provenance.get("source_timestamp_ms"),
            collision_snapshot_sha256=collision_scene["snapshot"].get("snapshot_sha256"),
            source_paths=provenance.get("source_paths", {}))
        raise
    from .geometry_alignment import table_top_z_at_xy
    table_height = table_top_z_at_xy(sdk_table, thickness, sdk_table[:2, 3])
    depths = np.asarray(profile["depths_m"], dtype=float)
    if depths.ndim != 1 or not len(depths) or not np.isfinite(depths).all():
        raise ValueError("真机 Bowl 抓取深度列表必须非空且有限")
    parameters = {
        "azimuths_rad": np.deg2rad(profile["azimuths_deg"]).tolist(),
        "tip_y_rad": np.deg2rad(profile["tip_y_deg"]).tolist(),
        "tcp_world_offset_m": profile["tcp_sdk_offset_m"],
        "pregrasp_opening_width_m": profile["pregrasp_opening_width_m"],
    }
    for key in ("azimuths_rad", "tip_y_rad"):
        values = np.asarray(parameters[key], dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"真机 Bowl {key} 必须是一维非空有限列表")
    generated = []
    for index, depth in enumerate(depths):
        for candidate in bowl_grasp_candidates(
            sdk_bowl, parameters={**parameters, "depth_m": float(depth)}, pose_source=pose_source,
        ):
            generated.append({**candidate, "candidate_id": f"real_d{index:02d}_{candidate['candidate_id']}"})
    # 共享旧排序直接用 TCP 位置，和 AffordToolAct 相同；不传仿真 tilt_pivot override。
    ordered = legacy_bowl_proposals(generated, object_pose_world=sdk_bowl)
    by_id = {item["candidate_id"]: item for item in generated}
    candidates, adjustments = [], []
    bowl_world = np.linalg.inv(world_bowl)
    for proposal in ordered:
        candidate = by_id[proposal.candidate_id]
        original = np.asarray(candidate["T_world_tcp"], dtype=float)
        target = original.copy()
        local_table_height = table_top_z_at_xy(sdk_table, thickness, target[:2, 3])
        target[2, 3] = np.clip(target[2, 3], local_table_height + clearance[0], local_table_height + clearance[1])
        world_tcp = world_sdk @ target
        object_tcp = bowl_world @ world_tcp
        # 同步更新每层 pose；T_primitive_tcp_raw 保留未偏移、未限高的原始局部候选。
        sdk_primitive = sdk_bowl @ np.asarray(candidate["T_object_tcp"]) @ np.linalg.inv(
            np.asarray(candidate["T_primitive_tcp"])
        )
        primitive_tcp = np.linalg.inv(sdk_primitive) @ target
        candidates.append({**candidate, "T_world_tcp": world_tcp.tolist(),
                           "T_object_tcp": object_tcp.tolist(),
                           "T_primitive_tcp": primitive_tcp.tolist()})
        adjustments.append({"candidate_id": proposal.candidate_id,
                            "T_sdk_tcp_before_table_clamp": original.tolist(),
                            "T_world_tcp": world_tcp.tolist(),
                            "table_height_at_candidate_sdk_m": local_table_height,
                            "table_clamp_delta_sdk_z_m": float(target[2, 3] - original[2, 3])})
    return tuple(candidates), {
        "pose_table_consistency": pose_consistency,
        "source": profile["source"], "profile_path": str(PROFILE_PATH),
        "profile_sha256": hashlib.sha256(raw).hexdigest(),
        "depths_m": depths.tolist(), **parameters, "parameter_frame": "right_sdk_base",
        "T_world_sdk": world_sdk.tolist(), "table_height_sdk_m": table_height,
        "table_clearance_m": clearance.tolist(),
        "table_clearance_source": profile["table_clearance_source"],
        "candidate_order_policy": "affordtoolact_sdk_minus_x_then_minus_y_no_geometry_pruning",
        "candidate_targets": adjustments,
    }
