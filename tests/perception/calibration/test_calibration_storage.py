"""标定数据归档与统一入口的离线回归。"""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from gene_agent.perception_modules.camera.calibration.storage import (
    default_run_dir, resolve_calibration_path, same_recorded_config,
    publish_result,
    standard_camera_parameters,
)


class StorageTest(unittest.TestCase):
    def test_default_outputs_are_grouped(self):
        path = default_run_dir('moving_board')
        self.assertEqual(path.parts[:4], ('data','calibration','runs','moving_board'))
        self.assertTrue(path.name.startswith('calibration_moving_board_'))
        with self.assertRaises(ValueError):
            default_run_dir('unrelated')

    def test_legacy_path_resolves_without_changing_original_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root/'data/calibration_wrist_right_001/handeye.json'
            new = root/'data/calibration/runs/handeye/calibration_wrist_right_001/handeye.json'
            new.parent.mkdir(parents=True)
            new.write_text('{}')
            self.assertEqual(resolve_calibration_path(old, root=root),new)
            original={'handeye_file':str(old),'poses':8}
            with patch('gene_agent.perception_modules.camera.calibration.storage.ROOT',root):
                self.assertTrue(same_recorded_config(original,{'handeye_file':str(new),'poses':8},('handeye_file',)))
                self.assertFalse(same_recorded_config(original,{'handeye_file':str(new),'poses':9},('handeye_file',)))
            self.assertEqual(original['handeye_file'],str(old))
            unrelated=root/'other/calibration_wrist_right_001/handeye.json'
            self.assertEqual(resolve_calibration_path(unrelated,root=root),unrelated)

    def test_unified_entry_dispatches_only_requested_workflow(self):
        from gene_agent.perception_modules.camera.calibration.__main__ import main
        with patch('gene_agent.perception_modules.camera.calibration.guided_moving_board_extrinsic.main',return_value=0) as moving:
            self.assertEqual(main(['moving-board','--poses','8']),0)
            moving.assert_called_once_with(['--poses','8'])
        with patch('gene_agent.perception_modules.camera.calibration.guided_wrist_handeye.main',return_value=0) as handeye:
            self.assertEqual(main(['handeye','--help']),0)
            handeye.assert_called_once_with(['--help'])

    def test_latest_updates_only_for_saved_valid_result_and_keeps_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root/'data/calibration/runs/moving_board/first'
            second = first.parent/'second'
            for run in (first,second):
                run.mkdir(parents=True)
                (run/'third_view_extrinsic.json').write_text(json.dumps({'extrinsic_matrix':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],'quality_passed':True}))
            alias = publish_result(first,'third_camera',root=root)
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.resolve(),first)
            publish_result(second,'third_camera',root=root)
            self.assertEqual(alias.resolve(),second)
            from gene_agent.perception_modules.camera.calibration.storage import protect_latest_result
            with patch('gene_agent.perception_modules.camera.calibration.storage.ROOT',root):
                with self.assertRaises(FileExistsError):
                    protect_latest_result(second)
            self.assertTrue((first/'third_view_extrinsic.json').is_file())
            (first/'third_view_extrinsic.json').write_text('{"quality_passed":false}')
            with self.assertRaises(ValueError):
                publish_result(first,'third_camera',root=root)
            self.assertEqual(alias.resolve(),second)

    def test_standard_schema_uses_actual_camera_intrinsics_and_renames_equivalent_keys(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            K=np.array([[600.,0,320],[0,601.,240],[0,0,1]])
            np.savez(root/'frame.npz',rgb=np.zeros((480,640,3),np.uint8),K=K)
            sample={'rgbd_file':'frame.npz','K':K.tolist(),'serial':'new-camera'}
            result=standard_camera_parameters(root,sample,{'T_sdk_third':np.eye(4).tolist(),
                'camera_serial':'new-camera','robot_group':'right'})
            self.assertEqual(set(result),{'width','height','intrinsic_matrix','serial','extrinsic_matrix','robot_group'})
            self.assertEqual((result['width'],result['height']),(640,480))
            self.assertEqual(result['serial'],'new-camera')
            np.testing.assert_array_equal(result['intrinsic_matrix'],K)
            np.testing.assert_array_equal(result['extrinsic_matrix'],np.eye(4))
            with self.assertRaises(ValueError):
                standard_camera_parameters(root,sample,{'T_sdk_third':np.eye(4).tolist(),'camera_serial':'wrong-camera'})


if __name__ == '__main__':
    unittest.main()
