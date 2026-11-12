"""读取已有 FoundationPose 输出，构造带来源的真机场景快照。

pose.json 保存的是相机坐标系中的物体 pose；相机参数里的
``<side>_extrinsic_matrix`` 是 ``T_sdk_camera``。这里只做坐标组合和
有效期检查，不重新运行分割或姿态估计，也不假定历史文件是实时感知。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from gene_agent.knowledge.geometry import validate_pose


@dataclass(frozen=True)
class ObservedScene:
    """同一采集帧下的具名物体位姿，坐标系为对应机械臂 SDK base。"""

    side: str
    camera_id: str
    frame_number: int
    timestamp_ms: float
    T_sdk_camera: np.ndarray
    object_poses_sdk: Mapping[str, np.ndarray]
    sources: Mapping[str, str]
    pose_frame_verified: bool


def _object_pose_path(task_dir: Path, object_id: str) -> Path:
    parts = object_id.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("object_id 必须为 Category/Instance，不能包含相对路径")
    if any("\\" in part for part in parts):
        raise ValueError("object_id 不能包含路径分隔符")
    return task_dir / "objs" / parts[0] / parts[1] / "pose.json"


def load_observed_scene(
    task_dir: str | Path, *, side: str, object_ids: Sequence[str],
    camera_id: str = "0", max_age_s: float | None = None,
    now_ms: float | None = None, require_matched_frame: bool = False,
) -> ObservedScene:
    """只读现有感知产物，校验帧有效期后返回 ``T_sdk_object``。

    这里的 ``sdk`` 是 ``side`` 对应的机械臂 SDK base，不自动等于
    MuJoCo 世界坐标。双臂协作时须另外使用已标定的左右 base 转换。
    """
    if side not in {"left", "right"}:
        raise ValueError("side 必须是 left 或 right")
    if not str(camera_id).isdigit():
        raise ValueError("camera_id 必须是非负整数编号")
    ids = tuple(str(value) for value in object_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("object_ids 必须非空且不能重复")
    task_path = Path(task_dir).expanduser().resolve()
    camera_path = task_path / "camera" / f"camera_parameters_{camera_id}.json"
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    transform_name = f"{side}_extrinsic_matrix"
    if transform_name not in camera:
        raise ValueError(f"{camera_path} 缺少 {transform_name}")
    T_sdk_camera = validate_pose(camera[transform_name], field_name=transform_name)
    frame_number = int(camera["frame_number"])
    timestamp_ms = float(camera["timestamp_ms"])
    if frame_number < 0 or not math.isfinite(timestamp_ms) or timestamp_ms < 0:
        raise ValueError("相机帧号或时间戳无效")
    if max_age_s is not None:
        age_limit = float(max_age_s)
        if not math.isfinite(age_limit) or age_limit < 0:
            raise ValueError("max_age_s 必须为非负有限数")
        current_ms = time.time() * 1_000.0 if now_ms is None else float(now_ms)
        if not math.isfinite(current_ms):
            raise ValueError("now_ms 必须为有限数")
        age_ms = current_ms - timestamp_ms
        if age_ms < -1_000.0 or age_ms > age_limit * 1_000.0:
            raise ValueError(
                f"相机帧过期或时间异常：帧龄 {age_ms / 1_000.0:.3f} s，"
                f"允许 {age_limit:.3f} s"
            )

    poses: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {"camera": str(camera_path)}
    all_frames_verified = True
    pose_key = f"camera_parameters_{camera_id}"
    for object_id in ids:
        pose_path = _object_pose_path(task_path, object_id)
        payload = json.loads(pose_path.read_text(encoding="utf-8"))
        provenance = payload.get("_capture")
        if provenance is None:
            all_frames_verified = False
            if require_matched_frame:
                raise ValueError(f"{pose_path} 缺少与相机同帧来源 _capture")
        elif (not isinstance(provenance, dict)
              or str(provenance.get("camera_id")) != str(camera_id)
              or provenance.get("frame_number") != frame_number
              or not math.isclose(float(provenance.get("timestamp_ms", -1.0)),
                                  timestamp_ms, rel_tol=0.0, abs_tol=1.0)):
            raise ValueError(f"{pose_path} 的同帧来源与相机参数不一致")
        candidates = payload.get(pose_key)
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError(f"{pose_path} 的 {pose_key} 必须恰好一个 pose")
        T_camera_object = validate_pose(candidates[0], field_name=str(pose_path))
        poses[object_id] = validate_pose(
            T_sdk_camera @ T_camera_object, field_name=f"T_sdk_{object_id}"
        )
        sources[object_id] = str(pose_path)
    return ObservedScene(
        side, str(camera_id), frame_number, timestamp_ms,
        T_sdk_camera, poses, sources, all_frames_verified,
    )
