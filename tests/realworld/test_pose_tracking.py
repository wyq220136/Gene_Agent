"""多物体 FoundationPose 必须来自同一冻结 RGB-D 帧。"""

import json
from pathlib import Path

import numpy as np
import pytest

from gene_agent.realworld.pose_tracking import (
    capture_frozen_scene_frame, track_observed_scene_from_capture,
)


def _capture(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "rgb.png").write_bytes(b"frozen-rgb")
    (capture / "depth.png").write_bytes(b"frozen-depth")
    (capture / "camera.json").write_text(json.dumps({
        "serial": "camera-1", "frame_number": 42,
        "timestamp_ms": 42000., "captured_at_unix_ms": 1000000.,
        "intrinsic_matrix": np.eye(3).tolist(),
    }), encoding="utf-8")
    return capture


def _pose(x):
    transform = np.eye(4)
    transform[0, 3] = x
    return transform.tolist()


def _settings(tmp_path, client):
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    extrinsic[:3, 3] = [.3, .1, .5]
    return dict(
        capture_dir=_capture(tmp_path), pose_client=client,
        sessions={"Bowl/Bowl_0": "bowl-session",
                  "Microwave/Microwave_0": "microwave-session"},
        side="right", camera_id="0", T_sdk_camera=extrinsic,
        expected_serial="camera-1", max_age_s=1., now_ms=1000500.,
        output_dir=tmp_path / "tracked",
    )


def test_multiobject_tracking_reads_one_frozen_frame_and_composes_frames(tmp_path):
    called = []

    class PoseClient:
        def track_frame(self, session_id, *, rgb_path, depth_path, camera_path):
            called.append((session_id, rgb_path, depth_path, camera_path))
            return {"ok": True, "session_id": session_id, "camera_id": "0",
                    "T_camera_object": _pose(.2 if session_id == "bowl-session" else .4)}

    settings = _settings(tmp_path, PoseClient())
    scene = track_observed_scene_from_capture(**settings)
    assert [row[0] for row in called] == ["bowl-session", "microwave-session"]
    assert called[0][1:] == called[1][1:]
    assert scene.pose_frame_verified is True
    assert scene.frame_number == 42 and scene.timestamp_ms == 1000000.
    np.testing.assert_allclose(scene.object_poses_sdk["Bowl/Bowl_0"][:3, 3],
                               [.3, .3, .5])
    np.testing.assert_allclose(scene.object_poses_sdk["Microwave/Microwave_0"][:3, 3],
                               [.3, .5, .5])
    for object_id in settings["sessions"]:
        source = scene.sources[object_id]
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        assert payload["_capture"]["frame_number"] == 42
        assert payload["object_id"] == object_id
    camera = json.loads((settings["output_dir"] / "camera/camera_parameters_0.json")
                        .read_text(encoding="utf-8"))
    assert camera["sensor_timestamp_ms"] == 42000.
    assert camera["timestamp_ms"] == 1000000.


def test_multiobject_tracking_rejects_stale_capture_before_pose_requests(tmp_path):
    class PoseClient:
        def track_frame(self, *_args, **_kwargs):
            raise AssertionError("过期帧不应发送给 Pose server")

    settings = _settings(tmp_path, PoseClient())
    settings["now_ms"] = 1002000.
    with pytest.raises(ValueError, match="过期"):
        track_observed_scene_from_capture(**settings)


def test_multiobject_tracking_rejects_files_modified_between_objects(tmp_path):
    class PoseClient:
        def track_frame(self, session_id, *, rgb_path, **_kwargs):
            if session_id == "microwave-session":
                rgb_path.write_bytes(b"another-camera-frame")
            return {"ok": True, "session_id": session_id, "camera_id": "0",
                    "T_camera_object": _pose(.2)}

    with pytest.raises(ValueError, match="采集文件已变化"):
        track_observed_scene_from_capture(**_settings(tmp_path, PoseClient()))


def test_multiobject_tracking_rejects_wrong_camera_response(tmp_path):
    class PoseClient:
        def track_frame(self, session_id, **_kwargs):
            return {"ok": True, "session_id": session_id, "camera_id": "1",
                    "T_camera_object": _pose(.2)}

    with pytest.raises(ValueError, match="相机"):
        track_observed_scene_from_capture(**_settings(tmp_path, PoseClient()))


def test_multiobject_tracking_rechecks_age_after_slow_pose_requests(tmp_path, monkeypatch):
    from gene_agent.realworld import pose_tracking as module

    class PoseClient:
        def track_frame(self, session_id, **_kwargs):
            return {"ok": True, "session_id": session_id, "camera_id": "0",
                    "T_camera_object": _pose(.2)}

    settings = _settings(tmp_path, PoseClient())
    settings.pop("now_ms")
    clock = iter((1000.5, 1002.0))
    monkeypatch.setattr(module.time, "time", lambda: next(clock))
    with pytest.raises(ValueError, match="完成时.*过期"):
        track_observed_scene_from_capture(**settings)
    assert not settings["output_dir"].exists()


def test_tracking_rejects_capture_without_host_time(tmp_path):
    class PoseClient:
        def track_frame(self, *_args, **_kwargs):
            raise AssertionError("设备时间不能用于帧龄检查")

    settings = _settings(tmp_path, PoseClient())
    camera_path = settings["capture_dir"] / "camera.json"
    payload = json.loads(camera_path.read_text(encoding="utf-8"))
    del payload["captured_at_unix_ms"]
    camera_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="主机采集时间"):
        track_observed_scene_from_capture(**settings)


def test_capture_once_preserves_device_time_and_records_wall_time(tmp_path, monkeypatch):
    from gene_agent.realworld import pose_tracking as module
    from gene_agent.perception_modules.camera.live_rgbd_protocol import FramePacket

    packet = FramePacket(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.float32),
        frame_number=42, timestamp_ms=42000., serial="camera-1",
        intrinsic_matrix=np.eye(3),
    )
    closed = []
    reads = []

    class Camera:
        def __init__(self, endpoint):
            assert endpoint == "test-camera"

        def info(self):
            return {"serial": "camera-1"}

        def get_frame(self):
            reads.append(True)
            if len(reads) == 1:
                return FramePacket(
                    rgb=packet.rgb, depth=packet.depth,
                    frame_number=41, timestamp_ms=41900.,
                    serial=packet.serial, intrinsic_matrix=packet.intrinsic_matrix,
                )
            return packet

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module, "CameraClient", Camera)
    capture = capture_frozen_scene_frame(
        output_dir=tmp_path / "capture", endpoint="test-camera",
        expected_serial="camera-1", now_ms=1000000.,
    )
    saved = json.loads((capture / "camera.json").read_text(encoding="utf-8"))
    assert saved["timestamp_ms"] == 42000.
    assert saved["captured_at_unix_ms"] == 1000000.
    assert reads == [True, True]
    assert closed == [True]


def test_capture_rejects_repeated_cached_frame(tmp_path, monkeypatch):
    from gene_agent.realworld import pose_tracking as module
    from gene_agent.perception_modules.camera.live_rgbd_protocol import FramePacket

    packet = FramePacket(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.float32), frame_number=41,
        timestamp_ms=41900., serial="camera-1", intrinsic_matrix=np.eye(3),
    )

    class Camera:
        def __init__(self, _endpoint):
            pass

        def info(self):
            return {"serial": "camera-1"}

        def get_frame(self):
            return packet

        def close(self):
            pass

    monkeypatch.setattr(module, "CameraClient", Camera)
    with pytest.raises(ValueError, match="更新的采集帧"):
        capture_frozen_scene_frame(
            output_dir=tmp_path / "capture", endpoint="test-camera",
            expected_serial="camera-1", now_ms=1000000.,
        )
    assert not (tmp_path / "capture").exists()


def test_tracked_scene_enters_existing_collision_export_without_simulation_step(tmp_path):
    from gene_agent.realworld.microwave.scene import build_observed_collision_scene
    from gene_agent.realworld import pose_tracking as module

    root = Path(module.__file__).resolve().parents[2]
    config = json.loads((root / "config/scenes/task_1.json").read_text(encoding="utf-8"))
    poses = {
        "Bowl/Bowl_0": config["objects"]["bowl"]["simulation_nominal_pose"],
        "Microwave/Microwave_0": config["objects"]["microwave"]["simulation_nominal_pose"],
    }

    class PoseClient:
        def track_frame(self, session_id, **_kwargs):
            object_id = next(name for name in poses if name.startswith(session_id))
            return {"ok": True, "session_id": session_id, "camera_id": "0",
                    "T_camera_object": poses[object_id]}

    settings = _settings(tmp_path, PoseClient())
    settings["sessions"] = {name: name.split("/")[0] for name in poses}
    settings["T_sdk_camera"] = np.eye(4)
    scene = track_observed_scene_from_capture(**settings)
    table = np.eye(4)
    table[:3, 3] = [*config["table"]["center_xy_m"],
                    config["table"]["top_z_m"] - config["table"]["thickness_m"] / 2]
    joints = {group: json.loads(
        (root / f"config/robots/realman_{group}_default_pose.json").read_text(
            encoding="utf-8"))["joints_rad"] for group in ("left", "right")}
    support = config["microwave_support"]
    collision = build_observed_collision_scene(
        scene, T_world_sdk=np.eye(4),
        T_perception_object_scene_asset={name: np.eye(4) for name in poses},
        table_pose_world=table, table_size_xy_m=config["table"]["size_xy_m"],
        accepted_support_size_xyz_m=(*support["size_xy_m"], support["height_m"]),
        accepted_support_center_xy_local_m=support["center_xy_local_m"],
        accepted_table_thickness_m=config["table"]["thickness_m"],
        joint_positions_by_group=joints,
        gripper_widths_m={"left": .08, "right": .04},
        door_angle_rad=.3, knob_angle_rad=.1,
        turntable_height_offset_m=0., microwave_floor_height_offset_m=0.,
        active_group="right", output_dir=tmp_path / "collision",
    )
    assert collision["provenance"]["source_frame_number"] == 42
    assert collision["provenance"]["physics_stepped"] is False
    assert collision["snapshot"]["active_group"] == "right"
