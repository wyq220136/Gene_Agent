"""固定机器人移板标定：几何与流程回归，不连接设备。"""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from gene_agent.perception_modules.camera.calibration.moving_board_extrinsic import (
    evaluate_moving_board, check_fixed_robot, check_new_board,
)


def pose(t, r):
    a = np.eye(4)
    a[:3, :3] = Rotation.from_rotvec(r).as_matrix()
    a[:3, 3] = t
    return a


class MovingBoardTest(unittest.TestCase):
    def setUp(self):
        self.flange = pose([.3, 0, .4], [.2, -.3, 0])
        self.wrist = pose([.01, .02, .04], [0, .1, .2])
        self.third = pose([.6, .2, .8], [.4, .2, -.5])
        self.samples = []
        for i in range(5):
            board = pose([.4 + .04*i, -.1, .1], [.05*i, -.1*i, 0])
            self.samples.append({'T_sdk_flange': self.flange.tolist(), 'joints_rad': [0.]*7,
                'wrist': {'T_camera_board': (np.linalg.inv(self.flange@self.wrist)@board).tolist()},
                'third': {'T_camera_board': (np.linalg.inv(self.third)@board).tolist()}})

    def evaluate(self, samples=None, wrist=None):
        return evaluate_moving_board(self.samples if samples is None else samples,
            self.flange, self.wrist if wrist is None else wrist, validation_poses=2)

    def test_recovers_base_extrinsic_and_each_distinct_board(self):
        report = self.evaluate()
        self.assertTrue(report['quality_passed'])
        np.testing.assert_allclose(report['T_sdk_third'], self.third, atol=1e-10)
        self.assertEqual(report['train_count'], 3)
        self.assertEqual(report['validation_count'], 2)
        self.assertFalse(np.allclose(report['boards'][0]['T_sdk_board'], report['boards'][1]['T_sdk_board']))

    def test_holdout_cannot_change_fit_and_failure_is_reported(self):
        expected = self.evaluate()['T_sdk_third']
        self.samples[-1]['third']['T_camera_board'][0][3] += .05
        report = self.evaluate()
        np.testing.assert_allclose(report['T_sdk_third'], expected, atol=1e-10)
        self.assertFalse(report['quality_passed'])
        self.assertGreater(report['validation_translation_max_mm'], 10)

    def test_train_outlier_not_silently_removed(self):
        self.samples[0]['third']['T_camera_board'][0][3] += .1
        self.assertFalse(self.evaluate()['quality_passed'])

    def test_handeye_bias_is_not_detectable_by_fixed_camera_consistency(self):
        wrong = self.wrist.copy()
        wrong[0, 3] += .05
        report = self.evaluate(wrist=wrong)
        self.assertTrue(report['quality_passed'])
        self.assertGreater(np.linalg.norm(np.array(report['T_sdk_third'])[:3,3]-self.third[:3,3]), .04)

    def test_reject_robot_motion_and_repeated_board(self):
        check_fixed_robot(self.samples[0], self.flange, np.zeros(7))
        self.samples[1]['joints_rad'][0] = np.deg2rad(.3)
        with self.assertRaisesRegex(ValueError, '机械臂'):
            check_fixed_robot(self.samples[1], self.flange, np.zeros(7))
        with self.assertRaisesRegex(ValueError, '板位'):
            check_new_board(self.samples[0], self.samples[:1])
        check_new_board(self.samples[1], self.samples[:1])
        moved = self.flange.copy()
        moved[0,3] += .005
        with self.assertRaises(ValueError):
            check_fixed_robot(self.samples[0], moved, np.zeros(7))

    def test_reject_insufficient_samples_and_invalid_limits(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.samples[:3])
        with self.assertRaises(ValueError):
            evaluate_moving_board(self.samples, self.flange, self.wrist, validation_poses=0)
        with self.assertRaises(ValueError):
            evaluate_moving_board(self.samples, self.flange, self.wrist, validation_poses=2, max_mm=float('nan'))

    def test_guided_resume_save_and_offline_preview_without_motion_interface(self):
        from gene_agent.perception_modules.camera.calibration import guided_moving_board_extrinsic as gui

        class Robot:
            communication_info = {'state_source': 'fake'}

            def __enter__(robot):
                return robot

            def __exit__(robot, *args):
                pass

            def read_flange_pose(robot):
                return self.flange.copy(), np.zeros(7)

        class Camera:
            def __init__(self, endpoint):
                pass

            def close(self):
                pass

        pairs = []
        for sample in self.samples:
            pair = {**sample, 'joint_change_deg': 0., 'host_capture_start_ns': 0,
                    'host_capture_end_ns': 1, 'host_frame_gap_ms': 1.}
            pair['T_sdk_flange'] = np.asarray(pair['T_sdk_flange'])
            pair['joints_rad'] = np.asarray(pair['joints_rad'])
            for camera in ('wrist', 'third'):
                pair[camera] = {**sample[camera], 'serial': camera, 'frame_number': 1,
                    'timestamp_ms': 1., 'K': np.eye(3), 'rgb': np.zeros((32,40,3), np.uint8),
                    'depth_m': np.ones((32,40), np.float32), 'corners': np.zeros((88,2)),
                    'reprojection_rmse_px': .1}
                pair[camera]['T_camera_board'] = np.asarray(pair[camera]['T_camera_board'])
            pairs.append(pair)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handeye = root / 'handeye.json'
            handeye.write_text(json.dumps({'calibration_type': 'eye_in_hand_fixed_board',
                'robot_group': 'right', 'camera_serial': 'wrist', 'T_flange_camera': self.wrist.tolist(),
                'robot_sdk_ip': '192.168.1.18',
                'pattern_size_inner_corners': [11,8], 'square_size_m': .015,
                'intrinsic_matrix': np.eye(3).tolist()}))
            output = root / 'run'
            args = ['--handeye', str(handeye), '--data-dir', str(output), '--poses', '5', '--no-preview']
            with patch.object(gui, 'CameraClient', Camera), patch.object(gui, '_check_camera_health'), \
                 patch.object(gui, 'connect_realman_read_only', return_value=Robot()), \
                 patch.object(gui, '_prompt_pair', side_effect=pairs[:2]+[None]):
                self.assertEqual(gui.main(args), 1)
            self.assertFalse((output/'third_view_extrinsic.json').exists())
            with patch.object(gui, 'CameraClient', Camera), patch.object(gui, '_check_camera_health'), \
                 patch.object(gui, 'connect_realman_read_only', return_value=Robot()), \
                 patch.object(gui, '_prompt_pair', side_effect=pairs[2:]), \
                 patch('builtins.input', return_value='ACCEPT'):
                self.assertEqual(gui.main(args+['--resume']), 0)
            saved = json.loads((output/'third_view_extrinsic.json').read_text())
            np.testing.assert_allclose(saved['extrinsic_matrix'], self.third, atol=1e-10)
            self.assertEqual(len(json.loads((output/'board_poses.json').read_text())['boards']), 5)
            with patch.object(gui, 'CameraClient', side_effect=AssertionError('不能连接相机')), \
                 patch.object(gui, 'connect_realman_read_only', side_effect=AssertionError('不能连接机器人')), \
                 patch.object(gui, 'preview') as preview:
                self.assertEqual(gui.main(['--preview-only','--data-dir',str(output)]), 0)
                preview.assert_called_once()

    def test_wrong_robot_provenance_rejected_before_connecting(self):
        from gene_agent.perception_modules.camera.calibration import guided_moving_board_extrinsic as gui
        with tempfile.TemporaryDirectory() as directory:
            handeye = Path(directory) / 'handeye.json'
            handeye.write_text(json.dumps({'calibration_type': 'eye_in_hand_fixed_board',
                'robot_group': 'right', 'T_flange_camera': self.wrist.tolist(),
                'robot_sdk_ip': '192.168.1.19'}))
            with patch.object(gui, 'CameraClient') as camera:
                with self.assertRaisesRegex(ValueError, '来源机器人'):
                    gui.main(['--handeye',str(handeye),'--no-preview'])
                camera.assert_not_called()

    def test_failed_quality_cannot_publish(self):
        from gene_agent.perception_modules.camera.calibration.guided_moving_board_extrinsic import save_results
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_results(Path(directory), {}, {'quality_passed': False}, visual_confirmed=True)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_preview_filters_far_points_in_both_cameras_without_opening_window(self):
        import open3d as o3d
        from gene_agent.perception_modules.camera.calibration import guided_moving_board_extrinsic as gui
        from gene_agent.perception_modules.camera.calibration import guided_fixed_board_extrinsic as fixed
        item={'depth_m':np.array([[.5,.8],[1.1,2.0]]),
              'rgb':np.zeros((2,2,3),np.uint8),'K':np.eye(3)}
        pair={'T_sdk_flange':self.flange,'wrist':item,'third':item}
        session={'samples':self.samples,'T_flange_wrist':self.wrist.tolist()}
        for depth, expected in ((1.2,3),(.9,2)):
            with patch.object(fixed,'_load_pair',return_value=pair), \
                 patch.object(o3d.visualization,'draw_geometries') as draw:
                gui.preview(Path('.'),session,self.evaluate(),max_depth=depth)
            clouds=[x for x in draw.call_args.args[0] if isinstance(x,o3d.geometry.PointCloud)]
            self.assertEqual([len(x.points) for x in clouds],[expected,expected])


if __name__ == '__main__':
    unittest.main()
