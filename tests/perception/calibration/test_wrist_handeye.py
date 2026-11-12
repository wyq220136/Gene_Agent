"""手腕相机手眼标定的离线几何回归；无需真机。"""

import unittest
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.perception_modules.camera.calibration.wrist_handeye import (
    detect_board_pose,
    camera_pose_from_flange,
    check_stationary,
    solve_handeye,
)
from gene_agent.realworld.devices.arms.realman.connection import connect_realman_read_only, normalize_realman_ip


def pose(rotvec, xyz):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    result[:3, 3] = xyz
    return result


class WristHandeyeTest(unittest.TestCase):
    def test_live_preview_refreshes_and_ignores_space_without_board(self):
        import cv2
        from gene_agent.perception_modules.camera.calibration import guided_wrist_handeye as guided

        class Camera:
            def __init__(self):
                self.calls = 0

            def get_frame(self):
                self.calls += 1
                return SimpleNamespace(
                    frame_number=self.calls, serial="wrist",
                    rgb=np.zeros((40, 60, 3), np.uint8),
                    intrinsic_matrix=np.eye(3),
                )

        camera = Camera()
        args = SimpleNamespace(serial="wrist", pattern_size=(5, 4),
                               square_size=0.02, max_reprojection_px=2.0)
        detected = {"corners": np.tile([20.0, 20.0], (20, 1)),
                    "reprojection_rmse_px": 0.3}
        with patch.object(guided, "detect_board_pose", side_effect=[None, detected]), \
             patch.object(cv2, "imshow") as imshow, \
             patch.object(cv2, "waitKey", side_effect=[32, 32]):
            action = guided._wait_for_live_capture(camera, args, index=1, total=20,
                                                   first_serial=None, first_K=None)
        self.assertEqual(action, "capture")
        self.assertEqual(camera.calls, 2)
        self.assertEqual(imshow.call_count, 2)

    def test_gui_candidate_confirmation_uses_window_keys(self):
        import cv2
        from gene_agent.perception_modules.camera.calibration import guided_wrist_handeye as guided

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            cv2.imwrite(str(path), np.zeros((40, 60, 3), np.uint8))
            with patch.object(cv2, "imshow"), patch.object(cv2, "waitKey", return_value=ord("y")):
                self.assertEqual(guided._confirm_candidate_gui(path), "y")

    def test_second_space_confirms_only_after_pedal_release(self):
        import cv2
        from gene_agent.perception_modules.camera.calibration import guided_wrist_handeye as guided

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            cv2.imwrite(str(path), np.zeros((40, 60, 3), np.uint8))
            with patch.object(cv2, "imshow"), \
                 patch.object(cv2, "waitKey", side_effect=[32, -1, 32]) as wait:
                self.assertEqual(guided._confirm_candidate_gui(path), "y")
            self.assertEqual(wait.call_count, 3)

    def test_terminal_pipeline_opens_live_window_by_default(self):
        import cv2
        from gene_agent.perception_modules.camera.calibration import guided_wrist_handeye as guided

        class FakeCamera:
            def health(self):
                return {"ok": True}

            def close(self):
                pass

        class FakeRobot:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.dict(os.environ, {"DISPLAY": ":1"}), \
                 patch.object(guided, "CameraClient", return_value=FakeCamera()), \
                 patch.object(guided, "connect_realman_read_only", return_value=FakeRobot()), \
                 patch.object(guided, "_wait_for_live_capture", return_value="quit") as preview, \
                 patch.object(cv2, "namedWindow") as named, \
                 patch.object(cv2, "resizeWindow"), \
                 patch.object(cv2, "destroyAllWindows") as closed:
                status = guided.main(["--ip", "192.0.2.1", "--poses", "10",
                                      "--data-dir", str(output)])
            self.assertEqual(status, 1)
            self.assertEqual(preview.call_count, 1)
            named.assert_called_once()
            closed.assert_called_once()
            self.assertEqual(json.loads((output / "samples.json").read_text())["samples"], [])

    def test_normalizes_browser_style_robot_url_before_sdk(self):
        self.assertEqual(normalize_realman_ip("http://192.168.1.18/"), "192.168.1.18")
        self.assertEqual(normalize_realman_ip("192.168.1.18"), "192.168.1.18")
        with self.assertRaisesRegex(ValueError, "IP|地址"):
            normalize_realman_ip("http://192.168.1.18/other")

    def test_read_only_connection_recreates_sdk_after_failed_handle(self):
        calls = []

        class SDK:
            def __init__(self, attempt):
                self.attempt = attempt
                calls.append(("new", attempt))

            def rm_create_robot_arm(self, ip, port, level):
                calls.append(("connect", self.attempt, ip, port, level))
                return SimpleNamespace(id=-1 if self.attempt == 1 else 2)

            def rm_delete_robot_arm(self):
                calls.append(("close", self.attempt))
                return 0

        attempts = iter([1, 2])
        with patch("gene_agent.realworld.devices.arms.realman.connection.time.sleep") as sleep:
            with connect_realman_read_only(
                "http://192.168.1.18/", group="right",
                sdk_factory=lambda: SDK(next(attempts)),
                connect_retries=2, connect_retry_interval_s=0.5,
            ):
                pass
        self.assertEqual(calls, [
            ("new", 1), ("connect", 1, "192.168.1.18", 8080, 3), ("close", 1),
            ("new", 2), ("connect", 2, "192.168.1.18", 8080, 3), ("close", 2),
        ])
        sleep.assert_called_once_with(0.5)

    def test_read_only_connection_exhausts_bounded_retries(self):
        calls = []

        class FailedSDK:
            def __init__(self, index):
                self.index = index

            def rm_create_robot_arm(self, ip, port, level):
                calls.append(("connect", self.index, ip, port))
                return SimpleNamespace(id=-1)

            def rm_delete_robot_arm(self):
                calls.append(("close", self.index))
                return 0

        counter = iter(range(1, 4))
        with patch("gene_agent.realworld.devices.arms.realman.connection.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "尝试 3 次"):
                connect_realman_read_only(
                    "192.168.1.18", group="right",
                    sdk_factory=lambda: FailedSDK(next(counter)),
                    connect_retries=3, connect_retry_interval_s=0.5,
                )
        self.assertEqual([call[0] for call in calls],
                         ["connect", "close", "connect", "close", "connect", "close"])
        self.assertEqual(sleep.call_count, 2)

    def test_connection_failure_keeps_new_run_directory_absent(self):
        from gene_agent.perception_modules.camera.calibration import guided_wrist_handeye as guided

        class FakeCamera:
            def health(self):
                return {"ok": True}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new_run"
            with patch.object(guided, "CameraClient", return_value=FakeCamera()), \
                 patch.object(guided, "connect_realman_read_only", side_effect=RuntimeError("offline")) as connect:
                with self.assertRaisesRegex(RuntimeError, "offline"):
                    guided.main(["--ip", "http://192.168.1.18/", "--data-dir", str(output)])
            self.assertFalse(output.exists())
            self.assertEqual(connect.call_args.args[0], "192.168.1.18")

    def test_empty_legacy_run_is_reusable_without_resume_flag(self):
        from gene_agent.perception_modules.camera.calibration import guided_wrist_handeye as guided

        class FakeCamera:
            def health(self):
                return {"ok": True}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            manifest = {
                "schema_version": 1, "calibration_type": "eye_in_hand_fixed_board",
                "robot_group": "right", "robot_sdk_ip": "http://192.168.1.18/",
                "camera_endpoint": "tcp://127.0.0.1:18082",
                "pattern_size_inner_corners": [11, 8], "square_size_m": 0.015,
                "samples": [],
            }
            (output / "samples.json").write_text(json.dumps(manifest))
            with patch.object(guided, "CameraClient", return_value=FakeCamera()), \
                 patch.object(guided, "connect_realman_read_only", side_effect=RuntimeError("offline")):
                with self.assertRaisesRegex(RuntimeError, "offline"):
                    guided.main(["--ip", "192.168.1.18", "--data-dir", str(output)])
            self.assertEqual(json.loads((output / "samples.json").read_text()), manifest)

    def test_board_pnp_returns_board_to_camera(self):
        import cv2

        cols, rows, size = 5, 4, 0.02
        points = np.array([[x * size, y * size, 0] for y in range(rows)
                           for x in range(cols)], dtype=np.float64)
        T_camera_board = pose([0.11, -0.08, 0.03], [0.02, 0.01, 0.5])
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=float)
        corners = cv2.projectPoints(
            points, np.array([0.11, -0.08, 0.03]), T_camera_board[:3, 3],
            K, np.zeros(5),
        )[0].astype(np.float32)
        with patch.object(cv2, "findChessboardCorners", return_value=(True, corners)), \
             patch.object(cv2, "cornerSubPix", return_value=corners):
            result = detect_board_pose(np.zeros((480, 640, 3), np.uint8), K,
                                       (cols, rows), size)
        self.assertLess(np.max(np.abs(result["T_camera_board"] - T_camera_board)), 1e-5)
        self.assertLess(result["reprojection_rmse_px"], 1e-3)

    def test_synthetic_eye_in_hand_direction_and_holdout(self):
        rng = np.random.default_rng(41)
        T_flange_camera = pose([0.16, -0.12, 0.09], [0.04, -0.02, 0.11])
        T_sdk_board = pose([0.02, 0.01, -0.08], [0.55, 0.10, 0.04])
        samples = []
        for index in range(20):
            T_sdk_flange = pose(
                rng.uniform(-0.5, 0.5, 3),
                [0.3, 0.05, 0.35] + rng.uniform(-0.1, 0.1, 3),
            )
            T_camera_board = (
                np.linalg.inv(T_sdk_flange @ T_flange_camera) @ T_sdk_board
            )
            samples.append({"T_sdk_flange": T_sdk_flange,
                            "T_camera_board": T_camera_board})
        result = solve_handeye(samples)
        self.assertLess(np.max(np.abs(result["T_flange_camera"] - T_flange_camera)), 1e-6)
        self.assertEqual(result["holdout_count"], 4)
        self.assertLess(result["holdout_translation_max_m"], 1e-6)
        self.assertLess(result["holdout_rotation_max_deg"], 1e-6)
        self.assertTrue(np.allclose(
            camera_pose_from_flange(samples[0]["T_sdk_flange"], result["T_flange_camera"]),
            samples[0]["T_sdk_flange"] @ T_flange_camera,
        ))

    def test_rejects_no_rotation_diversity(self):
        sample = {"T_sdk_flange": np.eye(4), "T_camera_board": np.eye(4)}
        with self.assertRaisesRegex(ValueError, "旋转|姿态"):
            solve_handeye([sample] * 10)

    def test_rejects_motion_during_capture(self):
        with self.assertRaisesRegex(ValueError, "移动|变化"):
            check_stationary(np.zeros(7), np.array([0.0, 0.0, 0.01, 0, 0, 0, 0]))
        check_stationary(np.zeros(7), np.full(7, np.deg2rad(0.03)))

    def test_read_only_sdk_flange_fk_does_not_use_gripper_or_tcp(self):
        class FakeSDK:
            def __init__(self):
                self.calls = []

            def rm_create_robot_arm(self, ip, port, level):
                self.calls.append(("connect", ip, port, level))
                return SimpleNamespace(id=1)

            def rm_get_joint_degree(self):
                self.calls.append(("joints",))
                return 0, [10, 20, 30, 40, 50, 60, 70]

            def rm_algo_forward_kinematics(self, joints, flag):
                self.calls.append(("fk", joints, flag))
                return [0.1, 0.2, 0.3, 0, 0, 0]

            def rm_algo_pos2matrix(self, pose):
                matrix = np.eye(4)
                matrix[:3, 3] = pose[:3]
                return SimpleNamespace(data=matrix.reshape(-1))

            def rm_delete_robot_arm(self):
                self.calls.append(("close",))
                return 0

        sdk = FakeSDK()
        with connect_realman_read_only("192.0.2.1", group="right", sdk=sdk) as robot:
            T_sdk_flange, joints = robot.read_flange_pose()
        self.assertTrue(np.allclose(T_sdk_flange[:3, 3], [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(np.rad2deg(joints), [10, 20, 30, 40, 50, 60, 70]))
        self.assertEqual([call[0] for call in sdk.calls], ["connect", "joints", "fk", "close"])

    def test_capture_rejects_robot_motion_while_reading_new_frame(self):
        from gene_agent.perception_modules.camera.calibration.guided_wrist_handeye import _capture

        class Camera:
            def __init__(self):
                self.calls = 0

            def get_frame(self):
                self.calls += 1
                return SimpleNamespace(frame_number=self.calls, serial="test",
                                       rgb=np.zeros((20, 20, 3), np.uint8),
                                       intrinsic_matrix=np.eye(3), depth=np.zeros((20, 20), np.float32))

        class Robot:
            def __init__(self):
                self.calls = 0

            def read_flange_pose(self):
                self.calls += 1
                joints = np.zeros(7)
                joints[0] = self.calls * 0.01
                return np.eye(4), joints

        camera = Camera()
        with patch("gene_agent.perception_modules.camera.calibration.guided_wrist_handeye.detect_board_pose") as detector:
            with self.assertRaisesRegex(ValueError, "移动"):
                _capture(camera, Robot(), SimpleNamespace(max_joint_change_deg=0.2, serial=None),
                         None)
        self.assertEqual(camera.calls, 2)
        detector.assert_not_called()

    def test_terminal_pipeline_saves_result_without_robot_motion_calls(self):
        import cv2
        from gene_agent.perception_modules.camera.calibration import guided_wrist_handeye as guided

        rng = np.random.default_rng(5)
        T_flange_camera = pose([0.1, -0.2, 0.05], [0.03, 0.01, 0.08])
        T_sdk_board = pose([0.02, 0.03, 0.04], [0.5, 0.1, 0.0])
        candidates = []
        for index in range(10):
            flange = pose(rng.uniform(-0.5, 0.5, 3),
                          [0.35, 0.02, 0.3] + rng.uniform(-0.1, 0.1, 3))
            candidates.append({
                "T_sdk_flange": flange,
                "T_camera_board": np.linalg.inv(flange @ T_flange_camera) @ T_sdk_board,
                "joints_rad": np.zeros(7), "joint_change_deg": 0.0,
                "reprojection_rmse_px": 0.1, "frame_number": index,
                "sensor_timestamp_ms": float(index), "host_request_before_ms": float(index),
                "host_frame_received_ms": float(index) + 1,
                "serial": "wrist", "K": np.eye(3),
                "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
                "depth_m": np.ones((10, 10), dtype=np.float32),
            })

        class FakeCamera:
            def health(self):
                return {"ok": True}

            def close(self):
                pass

        class FakeRobot:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        def capture(_camera, _robot, _args, candidate_path):
            cv2.imwrite(str(candidate_path), np.zeros((10, 10, 3), dtype=np.uint8))
            return candidates.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.object(guided, "CameraClient", return_value=FakeCamera()), \
                 patch.object(guided, "connect_realman_read_only", return_value=FakeRobot()), \
                 patch.object(guided, "_capture", side_effect=capture), \
                 patch("builtins.input", side_effect=["", "y"] * 10):
                status = guided.main(["--ip", "192.0.2.1", "--serial", "wrist",
                                      "--poses", "10", "--no-preview",
                                      "--data-dir", str(output)])
            self.assertEqual(status, 0)
            result = json.loads((output / "handeye.json").read_text())
            self.assertLess(np.max(np.abs(np.asarray(result["T_flange_camera"]) - T_flange_camera)), 1e-6)
            self.assertEqual(len(json.loads((output / "samples.json").read_text())["samples"]), 10)


if __name__ == "__main__":
    unittest.main()
