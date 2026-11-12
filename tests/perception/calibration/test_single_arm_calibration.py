"""单臂 SDK base 标定的结果与拒绝条件。"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from gene_agent.perception_modules.camera.calibration.single_arm_calibration import (
    calibrate_single_arm, calibrate_single_arm_from_positions,
)
from gene_agent.perception_modules.camera.calibration import guided_single_arm_calibration as guided
from gene_agent.realworld.devices.arms.realman.connection import read_realman_tcp_pose


def test_right_arm_uses_sdk_frame_without_extra_rotation(tmp_path):
    tcp = np.eye(4)
    tcp[:3, :3] = np.diag([1, -1, -1])
    tcp[:3, 3] = [0.5, 0.2, 0.3]
    camera_board = np.eye(4)
    camera_board[:3, 3] = [0.2, 0, 0.1]
    (tmp_path / "camera_to_board.json").write_text(json.dumps({
        "camera_to_board": camera_board.tolist(),
        "origin_confirmed": True,
        "captured_at_ns": 123,
    }))
    (tmp_path / "cam_param.json").write_text(json.dumps({"intrinsic_matrix": np.eye(3).tolist()}))
    task_camera = tmp_path / "camera_parameters_0.json"
    task_camera.write_text(json.dumps({"frame_number": 8, "timestamp_ms": 1234}))
    result = calibrate_single_arm(
        tmp_path, "right", tcp, [0.1, 0.05, 0.15],
        board_unchanged_confirmed=True, task_camera_parameters=task_camera,
    )
    expected = np.eye(4)
    expected[:3, 3] = [0.6, 0.15, 0.25]
    assert np.allclose(result, expected)
    camera = json.loads((tmp_path / "cam_param.json").read_text())
    assert np.allclose(camera["right_extrinsic_matrix"], expected)
    assert "left_extrinsic_matrix" not in camera
    assert np.allclose(json.loads(task_camera.read_text())["right_extrinsic_matrix"], expected)
    assert json.loads(task_camera.read_text())["frame_number"] == 8


def test_rejects_unconfirmed_board_and_wrong_tcp_orientation(tmp_path):
    tcp = np.eye(4)
    with pytest.raises(ValueError, match="标定板"):
        calibrate_single_arm(tmp_path, "right", tcp, [0, 0, 0.15],
                             board_unchanged_confirmed=False)
    assert not (tmp_path / "board_to_robot.json").exists()
    with pytest.raises(ValueError, match="TCP 姿态"):
        calibrate_single_arm(tmp_path, "right", tcp, [0, 0, 0.15],
                             board_unchanged_confirmed=True)
    assert not (tmp_path / "board_to_robot.json").exists()


def test_position_only_calibration_does_not_invent_tcp_orientation(tmp_path):
    camera_board = np.eye(4)
    camera_board[:3, 3] = [0.2, 0, 0.1]
    (tmp_path / "camera_to_board.json").write_text(json.dumps({
        "camera_to_board": camera_board.tolist(), "origin_confirmed": True,
        "captured_at_ns": 123,
    }))
    (tmp_path / "cam_param.json").write_text("{}")
    result = calibrate_single_arm_from_positions(
        tmp_path, "right", [0.5, 0.2, 0.3], [0.1, 0.05, 0.15],
        board_unchanged_confirmed=True,
    )
    assert np.allclose(result[:3, 3], [0.6, 0.15, 0.25])
    board = json.loads((tmp_path / "board_to_robot.json").read_text())
    assert board["board_axes_aligned_to_sdk"] is True
    assert "robot_tcp_pose" not in board


def test_guided_pipeline_exports_only_after_open3d_acceptance(tmp_path, monkeypatch):
    tcp = np.eye(4)
    tcp[:3, :3] = np.diag([1, -1, -1])
    tcp[:3, 3] = [0.5, 0.2, 0.3]
    tcp_path = tmp_path / "tcp.json"
    tcp_path.write_text(json.dumps(tcp.tolist()))
    task_camera = tmp_path / "camera_parameters_0.json"
    task_camera.write_text(json.dumps({"frame_number": 8}))

    def capture(**kwargs):
        directory = tmp_path
        (directory / "camera_to_board.json").write_text(json.dumps({
            "camera_to_board": np.eye(4).tolist(),
            "origin_confirmed": True,
            "captured_at_ns": 123,
        }))
        (directory / "cam_param.json").write_text(json.dumps({"intrinsic_matrix": np.eye(3).tolist()}))
        return {"camera_to_board": np.eye(4)}

    monkeypatch.setattr(guided, "calibrate_camera_to_board", capture)
    monkeypatch.setattr(guided, "visualize_extrinsic", lambda *args, **kwargs: None)
    answers = iter(["BOARD_ALIGNED", "CAMERA_OK", "0.1 0.05 0.15",
                    "BOARD_UNCHANGED", "TCP_VISIBLE", "REJECT"])
    result = guided.run_guided_calibration(
        tmp_path, "right", tcp_path, task_camera_parameters=task_camera,
        input_fn=lambda _: next(answers),
    )
    assert result is None
    assert "right_extrinsic_matrix" not in json.loads(task_camera.read_text())
    assert "right_extrinsic_matrix" in json.loads((tmp_path / "cam_param.json").read_text())

    answers = iter(["BOARD_ALIGNED", "CAMERA_OK", "0.1 0.05 0.15",
                    "BOARD_UNCHANGED", "TCP_VISIBLE", "ACCEPT"])
    result = guided.run_guided_calibration(
        tmp_path, "right", tcp_path, task_camera_parameters=task_camera,
        input_fn=lambda _: next(answers),
    )
    assert np.allclose(result[:3, 3], [0.4, 0.15, 0.15])
    task = json.loads(task_camera.read_text())
    assert np.allclose(task["right_extrinsic_matrix"], result)
    assert task["frame_number"] == 8


def test_guided_pipeline_accepts_tcp_position_without_pose_file(tmp_path, monkeypatch):
    def capture(**kwargs):
        (tmp_path / "camera_to_board.json").write_text(json.dumps({
            "camera_to_board": np.eye(4).tolist(),
            "origin_confirmed": True, "captured_at_ns": 123,
        }))
        (tmp_path / "cam_param.json").write_text("{}")
        return {"camera_to_board": np.eye(4)}

    monkeypatch.setattr(guided, "calibrate_camera_to_board", capture)
    monkeypatch.setattr(guided, "visualize_extrinsic", lambda *args, **kwargs: None)
    answers = iter(["BOARD_ALIGNED", "CAMERA_OK",
                    "0.1 0.05 0.15", "BOARD_UNCHANGED", "TCP_VISIBLE", "ACCEPT"])
    result = guided.run_guided_calibration(
        tmp_path, "right", sdk_tcp_position=[0.5, 0.2, 0.3],
        input_fn=lambda _: next(answers),
    )
    assert np.allclose(result[:3, 3], [0.4, 0.15, 0.15])


def test_read_realman_tcp_pose_uses_sdk_joints_fk_and_closes_connection():
    class ReadOnlySdk:
        def __init__(self):
            self.closed = False
            self.fk_joints = None

        def rm_create_robot_arm(self, ip, port, level):
            assert (ip, port, level) == ("169.254.128.19", 8080, 3)
            return SimpleNamespace(id=1)

        def rm_get_joint_degree(self):
            return 0, [0, 10, 20, 30, 40, 50, 60]

        def rm_algo_forward_kinematics(self, joints, flag):
            self.fk_joints = joints
            assert flag == 1
            return [0.5, 0.2, 0.3, 0, 0, 0]

        def rm_algo_pos2matrix(self, pose):
            matrix = np.eye(4)
            matrix[:3, 3] = pose[:3]
            return SimpleNamespace(data=matrix.flatten().tolist())

        def rm_delete_robot_arm(self):
            self.closed = True
            return 0

    sdk = ReadOnlySdk()
    pose, joints = read_realman_tcp_pose(
        "169.254.128.19", group="right", sdk=sdk,
        T_flange_tcp=np.eye(4),
    )
    assert np.allclose(pose[:3, 3], [0.5, 0.2, 0.3])
    assert np.allclose(joints, np.deg2rad([0, 10, 20, 30, 40, 50, 60]))
    assert np.allclose(sdk.fk_joints, [0, 10, 20, 30, 40, 50, 60])
    assert sdk.closed


def test_guided_pipeline_reads_tcp_from_robot_api_by_default(tmp_path, monkeypatch):
    pose = np.eye(4)
    pose[:3, 3] = [0.5, 0.2, 0.3]

    def read_live(ip, *, group, port):
        assert (ip, group, port) == ("169.254.128.19", "right", 8080)
        return pose.copy(), np.zeros(7)

    def capture(**kwargs):
        (tmp_path / "camera_to_board.json").write_text(json.dumps({
            "camera_to_board": np.eye(4).tolist(),
            "origin_confirmed": True, "captured_at_ns": 123,
        }))
        (tmp_path / "cam_param.json").write_text("{}")
        return {"camera_to_board": np.eye(4)}

    monkeypatch.setattr(guided, "read_realman_tcp_pose", read_live)
    monkeypatch.setattr(guided, "calibrate_camera_to_board", capture)
    monkeypatch.setattr(guided, "visualize_extrinsic", lambda *args, **kwargs: None)
    answers = iter(["BOARD_ALIGNED", "CAMERA_OK", "0.1 0.05 0.15",
                    "BOARD_UNCHANGED", "TCP_VISIBLE", "ACCEPT"])
    result = guided.run_guided_calibration(
        tmp_path, "right", robot_ip="169.254.128.19",
        input_fn=lambda _: next(answers),
    )
    assert np.allclose(result[:3, 3], [0.4, 0.15, 0.15])
    evidence = json.loads((tmp_path / "sdk_tcp_reading.json").read_text())
    assert evidence["source"] == "realman_sdk_fk"
    assert np.allclose(evidence["tcp_pose_sdk"], pose)
