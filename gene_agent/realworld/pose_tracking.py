"""把多个 FoundationPose session 的同一冻结帧结果交给真机现场接口。

现有 Pose server 的 ``track()`` 会为每个物体各拍一次，不能证明
多物体结果同帧。本适配只使用 ``track_frame()``，给每个 session 传
同一组 RGB-D 文件，并在前后核对文件哈希。这里不创建或验证标定。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Mapping

import numpy as np

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.perception_modules.camera.camera_client import CameraClient
from gene_agent.perception_modules.interactive_realworld_perception import save_capture

from .perception import _object_pose_path, load_observed_scene, ObservedScene


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_frozen_scene_frame(
    *, output_dir: str | Path, endpoint: str, expected_serial: str,
    now_ms: float | None = None,
) -> Path:
    """从 camera server 等到一帧新图，只保存它并记录主机采集时刻。

    RealSense 的原始 ``timestamp_ms`` 可能使用设备时钟，不能直接与
    ``time.time()`` 相减。这里另存主机 Unix 毫秒供现场帧龄检查，
    原始设备时间继续留在 camera.json 中供跨传感器诊断。
    """
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("采集目录已有内容；新现场帧必须使用新目录")
    client = CameraClient(endpoint)
    try:
        if client.info().get("serial") != expected_serial:
            raise ValueError("camera server 设备序列号与指定相机不一致")
        # 新 client 的首请求可能立刻返回缓存帧。先取一帧作哨兵，
        # 第二次请求由 CameraClient 等待帧号变化后才会返回。
        previous = client.get_frame()
        packet = client.get_frame()
        if packet.frame_number == previous.frame_number:
            raise ValueError("camera server 没有返回更新的采集帧")
        captured_at = time.time() * 1000. if now_ms is None else float(now_ms)
    finally:
        client.close()
    if packet.serial != expected_serial:
        raise ValueError("采集帧的相机序列号与指定相机不一致")
    if not math.isfinite(captured_at) or captured_at < 0:
        raise ValueError("主机采集时间必须是有效 Unix 毫秒")
    saved = save_capture(packet, output)
    camera = json.loads(saved.camera.read_text(encoding="utf-8"))
    camera["captured_at_unix_ms"] = captured_at
    saved.camera.write_text(json.dumps(
        camera, ensure_ascii=False, indent=2, allow_nan=False,
    ) + "\n", encoding="utf-8")
    return output


def track_observed_scene_from_capture(
    *, capture_dir: str | Path, pose_client,
    sessions: Mapping[str, str], side: str, camera_id: str,
    T_sdk_camera, expected_serial: str, max_age_s: float,
    output_dir: str | Path, now_ms: float | None = None,
) -> ObservedScene:
    """同步跟踪同一 RGB-D 帧，并保存兼容现有 ``pose.json`` 的证据。

    调用方先从 camera server 只采集一帧，再为每个物体提供已经注册的
    FoundationPose session。本函数不会调用机器人 SDK，也不会把
    tracking 成功当作位置精度或 mesh→碰撞资产 frame 标定通过。
    """
    if side not in {"left", "right"}:
        raise ValueError("side 必须是 left 或 right")
    if not str(camera_id).isdigit():
        raise ValueError("camera_id 必须是非负整数编号")
    if not isinstance(sessions, Mapping) or not sessions:
        raise ValueError("同帧跟踪需要至少一个物体 session")
    if any(not isinstance(object_id, str) or not isinstance(session_id, str)
           for object_id, session_id in sessions.items()):
        raise ValueError("物体 ID 和 FoundationPose session 必须是字符串")
    if len(set(sessions.values())) != len(sessions):
        raise ValueError("每个物体必须有独立的 FoundationPose session")
    T_camera = validate_pose(T_sdk_camera, field_name="T_sdk_camera")
    capture = Path(capture_dir).expanduser().resolve()
    paths = {name: capture / name for name in ("rgb.png", "depth.png", "camera.json")}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("同帧跟踪缺少 rgb.png、depth.png 或 camera.json")
    camera = json.loads(paths["camera.json"].read_text(encoding="utf-8"))
    if camera.get("serial") != expected_serial:
        raise ValueError("采集相机序列号与已核对设备不一致")
    frame = int(camera["frame_number"])
    sensor_timestamp = float(camera["timestamp_ms"])
    if "captured_at_unix_ms" not in camera:
        raise ValueError("采集文件缺少主机采集时间，不能把 RealSense 设备时间当作 Unix 时间")
    timestamp = float(camera["captured_at_unix_ms"])
    age_limit = float(max_age_s)
    current = time.time() * 1000. if now_ms is None else float(now_ms)
    if (frame < 0 or not math.isfinite(timestamp) or timestamp < 0
            or not math.isfinite(sensor_timestamp) or sensor_timestamp < 0
            or not math.isfinite(age_limit) or age_limit < 0
            or not math.isfinite(current)):
        raise ValueError("相机帧号、时间或有效期无效")
    age = current - timestamp
    if age < -1000. or age > age_limit * 1000.:
        raise ValueError(f"相机帧过期或时间异常：帧龄 {age / 1000.:.3f} s")
    # 先核对 ID，再请求远端模型；非法路径不能进入输出目录。
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录已有内容；请为新观测使用新目录")
    for object_id, session_id in sessions.items():
        _object_pose_path(output, str(object_id))
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(f"{object_id} 缺少有效 FoundationPose session")
    before = {name: _sha256(path) for name, path in paths.items()}
    responses: dict[str, np.ndarray] = {}
    for object_id, session_id in sessions.items():
        response = pose_client.track_frame(
            session_id, rgb_path=paths["rgb.png"],
            depth_path=paths["depth.png"], camera_path=paths["camera.json"],
        )
        if (not isinstance(response, dict) or response.get("ok") is not True
                or response.get("session_id") != session_id
                or str(response.get("camera_id")) != str(camera_id)):
            raise ValueError(f"{object_id} tracking session 或相机来源不一致")
        if "frame_number" in response and int(response["frame_number"]) != frame:
            raise ValueError(f"{object_id} tracking 帧号与冻结帧不一致")
        if ("timestamp_ms" in response
                and not math.isclose(float(response["timestamp_ms"]), sensor_timestamp,
                                     rel_tol=0., abs_tol=1.)):
            raise ValueError(f"{object_id} tracking 时间与冻结帧不一致")
        responses[str(object_id)] = validate_pose(
            response["T_camera_object"], field_name=f"T_camera_{object_id}",
        )
    after = {name: _sha256(path) for name, path in paths.items()}
    if before != after:
        raise ValueError("多物体 tracking 期间采集文件已变化，不能声称同帧")
    verified_now = current if now_ms is not None else time.time() * 1000.
    if verified_now - timestamp > age_limit * 1000.:
        raise ValueError("多物体 tracking 完成时相机帧已过期")
    output.mkdir(parents=True, exist_ok=True)
    camera_output = output / "camera" / f"camera_parameters_{camera_id}.json"
    camera_output.parent.mkdir(parents=True, exist_ok=True)
    camera_output.write_text(json.dumps({
        **camera, "sensor_timestamp_ms": sensor_timestamp,
        "timestamp_ms": timestamp,
        f"{side}_extrinsic_matrix": T_camera.tolist(),
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for object_id, pose in responses.items():
        path = _object_pose_path(output, object_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "object_id": object_id,
            f"camera_parameters_{camera_id}": [pose.tolist()],
            "_capture": {"camera_id": str(camera_id),
                         "frame_number": frame, "timestamp_ms": timestamp},
            "tracking_session_id": sessions[object_id],
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "capture_provenance.json").write_text(json.dumps({
        "source_files": {name: str(path) for name, path in paths.items()},
        "source_sha256": before, "camera_serial": expected_serial,
        "frame_number": frame, "captured_at_unix_ms": timestamp,
        "sensor_timestamp_ms": sensor_timestamp,
        "object_ids": list(responses),
        "claim": "多个 tracking session 使用同一冻结 RGB-D；不证明位姿精度或资产标定",
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return load_observed_scene(
        output, side=side, object_ids=tuple(responses), camera_id=str(camera_id),
        max_age_s=age_limit, now_ms=verified_now, require_matched_frame=True,
    )


__all__ = ["capture_frozen_scene_frame", "track_observed_scene_from_capture"]
