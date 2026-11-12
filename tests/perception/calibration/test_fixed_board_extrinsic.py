"""固定板完整外参与关节回放的离线回归；不连接硬件。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation


def pose(xyz=(0, 0, 0), rotation=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    result[:3, 3] = xyz
    return result


class ReplayTest(unittest.TestCase):
    class SDK:
        def rm_get_current_tool_frame(self):
            return 0, {"name": "tool", "pose": [0, 0, 0.12, 0, 0, 0]}

        def rm_get_current_work_frame(self):
            return 0, {"name": "work", "pose": [0.2, -0.1, 0, 0, 0, np.pi / 2]}

        def rm_algo_pos2matrix(self, values):
            t = pose(values[:3])
            t[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix()
            return SimpleNamespace(data=t.reshape(-1).tolist())

        def rm_algo_matrix2pos(self, matrix, flag=1):
            t = np.array(matrix.data).reshape(4, 4)
            return list(t[:3, 3]) + list(Rotation.from_matrix(t[:3, :3]).as_euler("xyz"))

        def rm_movel(self, goal, v, r, c, block):
            return 0

    def robot(self, target, reached=True):
        states = iter([np.zeros(7), np.ones(7), np.ones(7), np.ones(7)])
        return SimpleNamespace(
            read_joint_state=lambda: SimpleNamespace(positions_rad=next(states) if reached else np.zeros(7)),
            flange_pose_from_sdk_fk=lambda q: target.copy() if q[0] else np.eye(4), stop=lambda: None)

    def test_movel_converts_tool_work_frames_and_checks_pose_not_old_joints(self):
        from gene_agent.realworld.devices.arms.realman.calibration_replay import RealmanPoseReplayConnection

        target = pose([0.4, 0.2, 0.5])
        sdk = self.SDK()
        connection = RealmanPoseReplayConnection(sdk, self.robot(target))
        with patch.object(sdk, "rm_movel", wraps=sdk.rm_movel) as move, \
             patch("gene_agent.realworld.devices.arms.realman.calibration_replay.time.sleep"):
            result = connection.move_to_flange(target, speed_percent=5)
        np.testing.assert_allclose(move.call_args.args[0], [0.3, -0.2, 0.62, 0, 0, -np.pi / 2], atol=1e-12)
        self.assertEqual(move.call_args.args[1:4], (5, 0, 0))
        self.assertEqual(move.call_args.kwargs, {"block": 0})
        self.assertEqual(result["backend"], "realman_movel")
        self.assertLess(result["max_position_error_mm"], 0.001)
        self.assertFalse(result["collision_checked"])

    def test_movel_failure_and_timeout_stop_without_fallback_or_resend(self):
        from gene_agent.realworld.devices.arms.realman.calibration_replay import RealmanPoseReplayConnection

        for code in (0, 1):
            with self.subTest(code=code):
                sdk = self.SDK()
                robot = self.robot(pose([0.4, 0.2, 0.5]), reached=False)
                connection = RealmanPoseReplayConnection(sdk, robot)
                with patch.object(sdk, "rm_movel", return_value=code) as sent, \
                     patch.object(robot, "stop") as stop, \
                     patch("gene_agent.realworld.devices.arms.realman.calibration_replay.time.monotonic", side_effect=[0.0, 2.0]):
                    with self.assertRaises(RuntimeError):
                        connection.move_to_flange(pose([0.4, 0.2, 0.5]), timeout_s=1)
                self.assertEqual(sent.call_count, 1)
                stop.assert_called_once()

    def test_movel_uncertain_acknowledgement_can_succeed_by_measured_endpoint(self):
        from gene_agent.realworld.devices.arms.realman.calibration_replay import RealmanPoseReplayConnection

        for code in (-1, -2, -3):
            with self.subTest(code=code):
                sdk = self.SDK()
                robot = self.robot(pose([0.4, 0.2, 0.5]))
                with patch.object(sdk, "rm_movel", return_value=code) as sent, \
                     patch.object(robot, "stop") as stop, \
                     patch("gene_agent.realworld.devices.arms.realman.calibration_replay.time.sleep"):
                    result = RealmanPoseReplayConnection(sdk, robot).move_to_flange(pose([0.4, 0.2, 0.5]))
                sent.assert_called_once()
                stop.assert_not_called()
                self.assertEqual(result["sdk_command_return_code"], code)
                self.assertTrue(result["acknowledgement_uncertain"])
                self.assertTrue(result["endpoint_verified"])

    def test_movel_uncertain_ack_without_reaching_target_times_out_without_resend(self):
        from gene_agent.realworld.devices.arms.realman.calibration_replay import RealmanPoseReplayConnection

        sdk = self.SDK()
        robot = self.robot(pose([0.4, 0.2, 0.5]), reached=False)
        with patch.object(sdk, "rm_movel", return_value=-2) as sent, patch.object(robot, "stop") as stop, \
             patch("gene_agent.realworld.devices.arms.realman.calibration_replay.time.monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(RuntimeError, "超时"):
                RealmanPoseReplayConnection(sdk, robot).move_to_flange(pose([0.4, 0.2, 0.5]), timeout_s=1)
        sent.assert_called_once()
        stop.assert_called_once()

    def test_movel_rejects_invalid_pose_or_unreadable_controller_frames(self):
        from gene_agent.realworld.devices.arms.realman.calibration_replay import RealmanPoseReplayConnection

        sdk = self.SDK()
        connection = RealmanPoseReplayConnection(sdk, self.robot(np.eye(4)))
        for invalid in (np.zeros(7), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                connection.move_to_flange(invalid)
        with patch.object(sdk, "rm_get_current_tool_frame", return_value=(-1, {})), \
             patch.object(sdk, "rm_movel") as move:
            with self.assertRaises(RuntimeError):
                connection.move_to_flange(np.eye(4))
            move.assert_not_called()

    def test_movel_refuses_changed_start_and_stops_on_keyboard_interrupt(self):
        from gene_agent.realworld.devices.arms.realman.calibration_replay import RealmanPoseReplayConnection

        sdk = self.SDK()
        robot = self.robot(np.eye(4), reached=False)
        connection = RealmanPoseReplayConnection(sdk, robot)
        with patch.object(sdk, "rm_movel") as sent:
            with self.assertRaisesRegex(ValueError, "起点已变化"):
                connection.move_to_flange(np.eye(4), expected_start_joints_rad=np.ones(7))
            sent.assert_not_called()
        with patch.object(sdk, "rm_movel", side_effect=KeyboardInterrupt), patch.object(robot, "stop") as stop:
            with self.assertRaises(KeyboardInterrupt):
                connection.move_to_flange(np.eye(4))
            stop.assert_called_once()

    def test_frame_timeout_retries_reads_then_sends_movel_only_once(self):
        from gene_agent.realworld.devices.arms.realman.calibration_replay import RealmanPoseReplayConnection

        sdk = self.SDK()
        valid = sdk.rm_get_current_work_frame()
        connection = RealmanPoseReplayConnection(sdk, self.robot(pose([0.4, 0.2, 0.5])))
        with patch.object(sdk, "rm_get_current_work_frame", side_effect=[(-2, {}), (-2, {}), valid, valid]) as reads, \
             patch.object(sdk, "rm_movel", wraps=sdk.rm_movel) as move, \
             patch("gene_agent.realworld.devices.arms.realman.calibration_replay.time.sleep"):
            result = connection.move_to_flange(pose([0.4, 0.2, 0.5]))
        self.assertEqual(reads.call_count, 4)
        self.assertEqual(move.call_count, 1)
        self.assertEqual(result["backend"], "realman_movel")

    def test_persistent_frame_timeout_or_rejection_never_sends_motion(self):
        from gene_agent.realworld.devices.arms.realman.calibration_replay import RealmanPoseReplayConnection, ControllerFrameReadError

        for code, count in ((-2, 3), (1, 1)):
            with self.subTest(code=code):
                sdk = self.SDK()
                connection = RealmanPoseReplayConnection(sdk, self.robot(np.eye(4)))
                with patch.object(sdk, "rm_get_current_work_frame", return_value=(code, {})) as read, \
                     patch.object(sdk, "rm_movel") as move, \
                     patch("gene_agent.realworld.devices.arms.realman.calibration_replay.time.sleep"):
                    with self.assertRaises(ControllerFrameReadError) as caught:
                        connection.move_to_flange(np.eye(4))
                    self.assertFalse(caught.exception.motion_sent)
                    self.assertEqual(read.call_count, count)
                    move.assert_not_called()

    def test_post_move_frame_read_failure_stops_and_cannot_retry_motion(self):
        from gene_agent.realworld.devices.arms.realman.calibration_replay import RealmanPoseReplayConnection, ControllerFrameReadError

        sdk = self.SDK()
        valid = sdk.rm_get_current_work_frame()
        robot = self.robot(pose([0.4, 0.2, 0.5]))
        connection = RealmanPoseReplayConnection(sdk, robot)
        with patch.object(sdk, "rm_get_current_work_frame", side_effect=[valid, (-2, {}), (-2, {}), (-2, {})]), \
             patch.object(sdk, "rm_movel", wraps=sdk.rm_movel) as move, \
             patch.object(robot, "stop") as stop, \
             patch("gene_agent.realworld.devices.arms.realman.calibration_replay.time.sleep"):
            with self.assertRaises(ControllerFrameReadError) as caught:
                connection.move_to_flange(pose([0.4, 0.2, 0.5]))
            self.assertTrue(caught.exception.motion_sent)
            move.assert_called_once()
            stop.assert_called_once()

    def state_robot(self, sdk, replies):
        from gene_agent.realworld.devices.arms.realman.adapter import RealmanAdapter

        sdk.rm_get_joint_degree = lambda: next(replies)
        sdk.rm_algo_forward_kinematics = lambda q, flag=1: [0.4, 0.2, 0.5, 0, 0, 0]
        return RealmanAdapter(sdk, group="right", joint_names=tuple(f"right_joint_{i}" for i in range(1, 8)),
                              T_flange_tcp=np.eye(4))

    def test_joint_read_timeout_during_movel_recovers_without_resending(self):
        from gene_agent.realworld.devices.arms.realman.calibration_replay import RealmanPoseReplayConnection

        sdk = self.SDK()
        robot = self.state_robot(sdk, iter([(0, [0] * 7), (-2, []), *[(0, [1] * 7)] * 3]))
        with patch.object(sdk, "rm_get_joint_degree", wraps=sdk.rm_get_joint_degree) as reads, \
             patch.object(sdk, "rm_movel", wraps=sdk.rm_movel) as move, patch.object(robot, "stop") as stop, \
             patch("gene_agent.realworld.devices.arms.realman.adapter.time.sleep"):
            result = RealmanPoseReplayConnection(sdk, robot).move_to_flange(pose([0.4, 0.2, 0.5]))
        self.assertEqual(reads.call_count, 5)
        move.assert_called_once()
        stop.assert_not_called()
        self.assertLess(result["max_position_error_mm"], 0.001)

    def test_persistent_joint_read_timeout_stops_after_one_movel(self):
        from gene_agent.realworld.devices.arms.realman.calibration_replay import RealmanPoseReplayConnection, ControllerReadError

        sdk = self.SDK()
        robot = self.state_robot(sdk, iter([(0, [0] * 7), *[(-2, [])] * 3]))
        with patch.object(sdk, "rm_get_joint_degree", wraps=sdk.rm_get_joint_degree) as reads, \
             patch.object(sdk, "rm_movel", wraps=sdk.rm_movel) as move, patch.object(robot, "stop") as stop, \
             patch("gene_agent.realworld.devices.arms.realman.adapter.time.sleep"):
            with self.assertRaises(ControllerReadError) as caught:
                RealmanPoseReplayConnection(sdk, robot).move_to_flange(pose([0.4, 0.2, 0.5]))
        self.assertTrue(caught.exception.motion_sent)
        self.assertEqual(reads.call_count, 4)
        move.assert_called_once()
        stop.assert_called_once()

    def test_joint_read_failure_before_movel_is_marked_not_sent(self):
        from gene_agent.realworld.devices.arms.realman.calibration_replay import RealmanPoseReplayConnection, ControllerReadError

        sdk = self.SDK()
        robot = self.state_robot(sdk, iter([(-2, [])] * 3))
        with patch.object(sdk, "rm_movel") as move, patch.object(robot, "stop") as stop, \
             patch("gene_agent.realworld.devices.arms.realman.adapter.time.sleep"):
            with self.assertRaises(ControllerReadError) as caught:
                RealmanPoseReplayConnection(sdk, robot).move_to_flange(np.eye(4))
        self.assertFalse(caught.exception.motion_sent)
        move.assert_not_called()
        stop.assert_not_called()

    def test_joint_read_retries_only_communication_errors_and_rejects_invalid_data(self):
        for reply, attempts, error in (((1, []), 1, RuntimeError), ((0, [0] * 6), 1, ValueError)):
            with self.subTest(reply=reply):
                sdk = self.SDK()
                robot = self.state_robot(sdk, iter([reply]))
                with patch.object(sdk, "rm_get_joint_degree", wraps=sdk.rm_get_joint_degree) as reads:
                    with self.assertRaises(error):
                        robot.read_joint_state()
                self.assertEqual(reads.call_count, attempts)


class FixedBoardExtrinsicTest(unittest.TestCase):
    def setUp(self):
        self.board = pose([0.4, -0.1, 0.02], [0.1, 0.2, 0.3])
        self.handeye = pose([-0.04, 0.01, 0.06], [0.1, 0.2, 1.4])
        self.third = pose([0.7, 0.3, 0.8], [0.3, -0.2, 0.5])

    def sample(self, flange, board=None):
        current = self.board if board is None else board
        return {"T_sdk_flange": flange.tolist(),
                "wrist": {"T_camera_board": (np.linalg.inv(flange @ self.handeye) @ current).tolist()},
                "third": {"T_camera_board": (np.linalg.inv(self.third) @ current).tolist()}}

    def test_first_sample_solves_and_later_samples_independently_validate(self):
        from gene_agent.perception_modules.camera.calibration.guided_fixed_board_extrinsic import evaluate_samples

        samples = [self.sample(pose([0.2, 0, 0.4])),
                   self.sample(pose([0.3, 0.1, 0.5], [0.3, -0.1, 0]))]
        result = evaluate_samples(samples, self.handeye)
        np.testing.assert_allclose(result["T_sdk_third"], self.third, atol=1e-12)
        np.testing.assert_allclose(result["T_sdk_board"], self.board, atol=1e-12)
        self.assertTrue(result["quality_passed"])
        self.assertEqual(result["validation_count"], 1)
        samples[1]["wrist"]["T_camera_board"][0][3] += 0.05
        bad = evaluate_samples(samples, self.handeye)
        self.assertFalse(bad["quality_passed"])
        np.testing.assert_allclose(bad["T_sdk_third"], self.third, atol=1e-12)

    def test_moved_board_rejected_even_when_camera_chains_agree(self):
        from gene_agent.perception_modules.camera.calibration.guided_fixed_board_extrinsic import evaluate_samples

        samples = [self.sample(np.eye(4)), self.sample(np.eye(4), self.board @ pose([0.07, 0, 0]))]
        result = evaluate_samples(samples, self.handeye)
        self.assertFalse(result["quality_passed"])
        self.assertGreater(result["third_board_spread_mm"], 60)
        self.assertLess(result["chain_translation_max_mm"], 1e-8)

    def test_same_actual_pose_or_single_pair_cannot_publish_as_validated(self):
        from gene_agent.perception_modules.camera.calibration.guided_fixed_board_extrinsic import evaluate_samples

        one = self.sample(np.eye(4))
        for samples in ([one], [one, one, one]):
            with self.subTest(count=len(samples)):
                report = evaluate_samples(samples, self.handeye)
                self.assertFalse(report["quality_passed"])
                self.assertEqual(report["validation_count"], 0)
                self.assertTrue(report["validation_error"])

    def test_saved_pose_selection_validates_source_identity_and_seven_joints(self):
        from gene_agent.perception_modules.camera.calibration.guided_fixed_board_extrinsic import load_replay_targets

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            record = {"robot_sdk_ip": "http://192.168.1.18/", "robot_group": "right",
                      "samples": [{"index": 1, "joints_rad": [0.1] * 7, "T_sdk_flange": np.eye(4).tolist()}]}
            path.write_text(json.dumps(record))
            targets = load_replay_targets(path, [1], ip="192.168.1.18", group="right")
            np.testing.assert_allclose(targets[0]["joints_rad"], [0.1] * 7)
            with self.assertRaises(ValueError):
                load_replay_targets(path, [1], ip="192.168.1.19", group="right")
            with self.assertRaises(ValueError):
                load_replay_targets(path, [1, 1], ip="192.168.1.18", group="right")
            with self.assertRaises(ValueError):
                load_replay_targets(path, [2], ip="192.168.1.18", group="right")

    def pipeline_fixture(self, root):
        source = root / "samples.json"
        source.write_text(json.dumps({"robot_group": "right", "robot_sdk_ip": "192.168.1.18",
                                      "samples": [{"index": i, "joints_rad": [i / 10] * 7, "T_sdk_flange": pose([i / 10, 0, 0.4]).tolist()} for i in (1, 2)]}))
        handeye = root / "handeye.json"
        handeye.write_text(json.dumps({"calibration_type": "eye_in_hand_fixed_board", "robot_group": "right",
                                       "T_flange_camera": self.handeye.tolist(), "camera_serial": "wrist",
                                       "pattern_size_inner_corners": [5, 4], "square_size_m": 0.02,
                                       "intrinsic_matrix": np.eye(3).tolist(),
                                       "T_sdk_board_estimate": pose([9, 9, 9]).tolist()}))
        pairs = []
        for number, flange in enumerate((pose([0.2, 0, 0.4]), pose([0.3, 0.1, 0.5], [0.3, -0.1, 0])), 1):
            pair = self.sample(flange)
            pair["T_sdk_flange"] = np.asarray(pair["T_sdk_flange"])
            pair.update(joints_rad=np.full(7, number / 10), joint_change_deg=0.001,
                        host_capture_start_ns=1, host_capture_end_ns=2, host_frame_gap_ms=10)
            for key in ("wrist", "third"):
                pair[key].update(T_camera_board=np.array(pair[key]["T_camera_board"]),
                                 serial=key, corners=np.zeros((20, 2)), reprojection_rmse_px=0.1,
                                 frame_number=number, timestamp_ms=float(number), K=np.eye(3),
                                 rgb=np.zeros((32, 40, 3), np.uint8), depth_m=np.ones((32, 40), np.float32))
            pairs.append(pair)
        args = ["--handeye", str(handeye), "--pose-indices", "1", "2", "--no-preview",
                "--third-serial", "third", "--data-dir", str(root / "run")]
        return args, pairs

    @staticmethod
    def camera_factory(*_):
        return SimpleNamespace(health=lambda: {"ok": True}, close=lambda: None)

    class Robot:
        def __init__(self):
            self.moves = []
            self.communication_info = {"state_source": "udp", "port": 8888, "sdk_timeout_ms": 2000}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read_flange_pose(self):
            return np.eye(4), np.zeros(7)

        def move_to_flange(self, target, **kwargs):
            self.moves.append(np.asarray(target))
            return {"measured_joints_rad": list(target), "max_target_error_deg": 0.0}

    def test_complete_replay_pipeline_uses_new_observations_and_exports_after_accept(self):
        from gene_agent.perception_modules.camera.calibration import guided_fixed_board_extrinsic as guided

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, pairs = self.pipeline_fixture(root)
            robot = self.Robot()
            with patch.object(guided, "CameraClient", side_effect=self.camera_factory), \
                 patch.object(guided, "connect_realman_pose_replay", return_value=robot), \
                 patch.object(guided, "connect_realman_read_only", side_effect=AssertionError("应为显式回放连接")), \
                 patch.object(guided, "_prompt_pair", side_effect=pairs), \
                 patch("builtins.input", side_effect=["", "", "ACCEPT"]):
                status = guided.main(args + ["--execute"])
            self.assertEqual(status, 0)
            self.assertEqual(len(robot.moves), 2)
            np.testing.assert_allclose(robot.moves[0], pose([0.1, 0, 0.4]))
            np.testing.assert_allclose(robot.moves[1], pose([0.2, 0, 0.4]))
            saved = json.loads((root / "run/third_view_extrinsic.json").read_text())
            np.testing.assert_allclose(saved["extrinsic_matrix"], self.third, atol=1e-12)
            np.testing.assert_allclose(saved["T_sdk_board"], self.board, atol=1e-12)
            self.assertEqual(saved["validation_count"], 1)
            self.assertTrue((root / "run/board_to_robot.json").exists())
            self.assertTrue((root / "run/camera_to_board.json").exists())

    def test_read_only_resume_never_moves_or_overwrites_accepted_sample(self):
        from gene_agent.perception_modules.camera.calibration import guided_fixed_board_extrinsic as guided

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, pairs = self.pipeline_fixture(root)
            robot = self.Robot()
            with patch.object(guided, "CameraClient", side_effect=self.camera_factory), \
                 patch.object(guided, "connect_realman_read_only", return_value=robot), \
                 patch.object(guided, "connect_realman_pose_replay", side_effect=AssertionError("未授权运动")), \
                 patch.object(guided, "_prompt_pair", side_effect=[pairs[0]]), \
                 patch("builtins.input", side_effect=["", "Q"]):
                self.assertEqual(guided.main(args), 1)
            before = (root / "run/pose_001_wrist.npz").read_bytes()
            with patch.object(guided, "CameraClient", side_effect=self.camera_factory), \
                 patch.object(guided, "connect_realman_read_only", return_value=robot), \
                 patch.object(guided, "_prompt_pair", side_effect=[pairs[1]]), \
                 patch("builtins.input", side_effect=["c", "CANCEL"]):
                self.assertEqual(guided.main(args + ["--resume"]), 1)
            self.assertEqual(robot.moves, [])
            self.assertEqual((root / "run/pose_001_wrist.npz").read_bytes(), before)
            self.assertFalse((root / "run/third_view_extrinsic.json").exists())
            report = json.loads((root / "run/quality_report.json").read_text())
            self.assertTrue(report["quality_passed"])

    def test_failed_validation_keeps_candidates_without_export(self):
        from gene_agent.perception_modules.camera.calibration import guided_fixed_board_extrinsic as guided

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, pairs = self.pipeline_fixture(root)
            pairs[1]["wrist"]["T_camera_board"][0, 3] += 0.05
            with patch.object(guided, "CameraClient", side_effect=self.camera_factory), \
                 patch.object(guided, "connect_realman_read_only", return_value=self.Robot()), \
                 patch.object(guided, "_prompt_pair", side_effect=pairs), \
                 patch("builtins.input", side_effect=["CURRENT", "CURRENT"]):
                self.assertEqual(guided.main(args), 2)
            self.assertFalse((root / "run/third_view_extrinsic.json").exists())
            self.assertTrue((root / "run/quality_report.json").exists())

    def test_list_targets_connects_no_devices(self):
        from gene_agent.perception_modules.camera.calibration import guided_fixed_board_extrinsic as guided

        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.pipeline_fixture(Path(directory))
            with patch.object(guided, "CameraClient", side_effect=AssertionError("离线列出不能连接相机")), \
                 patch.object(guided, "connect_realman_pose_replay", side_effect=AssertionError("离线列出不能连接机器人")):
                self.assertEqual(guided.main(args + ["--execute", "--list-poses"]), 0)

    def test_read_failure_before_move_returns_to_confirmation_and_allows_current(self):
        from gene_agent.perception_modules.camera.calibration import guided_fixed_board_extrinsic as guided
        from gene_agent.realworld.devices.arms.realman.calibration_replay import ControllerFrameReadError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, pairs = self.pipeline_fixture(root)
            robot = self.Robot()
            with patch.object(guided, "CameraClient", side_effect=self.camera_factory), \
                 patch.object(guided, "connect_realman_pose_replay", return_value=robot), \
                 patch.object(robot, "move_to_flange", side_effect=ControllerFrameReadError("工作系读取失败", motion_sent=False)), \
                 patch.object(guided, "_prompt_pair", side_effect=pairs), \
                 patch("builtins.input", side_effect=["MOVE", "CURRENT", "CURRENT", "ACCEPT"]):
                self.assertEqual(guided.main(args + ["--execute"]), 0)
            session = json.loads((root / "run/session.json").read_text())
            self.assertEqual(session["motion_events"][0]["status"], "not_sent")
            self.assertEqual(len(session["samples"]), 2)

    def test_confirmation_normalizes_fullwidth_and_edge_paste_characters(self):
        from gene_agent.perception_modules.camera.calibration.guided_fixed_board_extrinsic import normalize_command

        self.assertEqual(normalize_command("\ufeffＭＯＶＥ\u200b "), "MOVE")
        self.assertEqual(normalize_command("current\n"), "CURRENT")
        self.assertNotEqual(normalize_command("M\u200bOVE"), "MOVE")
        self.assertNotEqual(normalize_command("MOVE something"), "MOVE")

    def test_failed_run_directory_cannot_resume_or_connect_devices(self):
        from gene_agent.perception_modules.camera.calibration import guided_fixed_board_extrinsic as guided

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "RUN_FAILED.json").write_text(json.dumps({"status": "failed"}))
            with patch.object(guided, "CameraClient", side_effect=AssertionError("失败目录不能连接设备")):
                with self.assertRaisesRegex(ValueError, "失败"):
                    guided.main(["--data-dir", directory, "--resume"])

    def test_window_only_exists_during_capture_not_terminal_motion_prompt(self):
        import cv2
        from gene_agent.perception_modules.camera.calibration import guided_fixed_board_extrinsic as guided

        opened = []

        def answer(prompt):
            self.assertFalse(opened, "等待终端输入时不能留下未刷新的 OpenCV 窗口")
            return "ACCEPT" if "ACCEPT" in prompt else "CURRENT"

        with tempfile.TemporaryDirectory() as directory:
            args, pairs = self.pipeline_fixture(Path(directory))
            args.remove("--no-preview")
            iterator = iter(pairs)

            def capture(*a, **kw):
                self.assertTrue(opened, "实际采样时须打开实时窗口")
                return next(iterator)

            with patch.dict("os.environ", {"DISPLAY": ":0"}), \
                 patch.object(guided, "CameraClient", side_effect=self.camera_factory), \
                 patch.object(guided, "connect_realman_read_only", return_value=self.Robot()), \
                 patch.object(guided, "_prompt_pair", side_effect=capture), \
                 patch("builtins.input", side_effect=answer), \
                 patch.object(cv2, "namedWindow", side_effect=lambda *a: opened.append(True)), \
                 patch.object(cv2, "resizeWindow"), patch.object(cv2, "imshow"), \
                 patch.object(cv2, "waitKey", return_value=-1), \
                 patch.object(cv2, "destroyWindow", side_effect=lambda *a: opened.clear()), \
                 patch.object(cv2, "destroyAllWindows", side_effect=lambda: opened.clear()):
                self.assertEqual(guided.main(args), 0)
            self.assertFalse(opened)


if __name__ == "__main__":
    unittest.main()
