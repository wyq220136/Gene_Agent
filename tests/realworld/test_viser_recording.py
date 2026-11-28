"""首帧便携点云与成功规划落盘；只使用冻结数据和假设备。"""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from gene_agent.feedback.realworld_diagnostics import RealworldRunDiagnostics
from gene_agent.realworld.scene_archive import SceneArchive, verify_scene_archive
from gene_agent.visualization.realworld_scene import load_observation
from gene_agent.visualization.realworld_alignment import load_recorded_initial_rgbd
from gene_agent.visualization.realworld_pick_debug import _load_trajectories
from test_scene_archive import toy_scene
from test_runtime import FakeRobot, _stage


def archive_capture(tmp_path, *, raw_frame=7):
    scene = toy_scene(tmp_path)
    folder = tmp_path / 'source/observations/0000'
    capture = folder / 'capture'; capture.mkdir(parents=True)
    record = dict(frame_number=raw_frame, captured_at_unix_ms=100., depth_unit='millimeter',
                  serial='cam', intrinsic_matrix=[[100, 0, 0], [0, 100, 0], [0, 0, 1]])
    (capture/'camera.json').write_text(json.dumps(record))
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(capture/'rgb.png')
    Image.fromarray(np.full((12, 12), 1000, dtype=np.uint16)).save(capture/'depth.png')
    tracked = folder/'tracked/camera/camera_parameters_0.json'; tracked.parent.mkdir(parents=True)
    pose = np.eye(4); pose[0, 3] = .5
    tracked.write_text(json.dumps({**record, 'frame_number': 7, 'right_extrinsic_matrix': pose.tolist()}))
    (folder/'tracked/capture_provenance.json').write_text(json.dumps({
        'source_files': {n: str(capture/n) for n in ('rgb.png', 'depth.png', 'camera.json')}}))
    scene['provenance']['source_paths'] = {'camera': str(tracked)}
    archive = SceneArchive(tmp_path/'bundle')
    archive.add_scene(scene); archive.add_perception(scene, [])
    return archive, scene


def test_initial_cloud_survives_missing_setup_bowl_and_original_host_files(tmp_path):
    archive, _ = archive_capture(tmp_path)
    (tmp_path/'source').rename(tmp_path/'source_unavailable')
    restored, _, _, rgbd, warning = load_observation(archive.root)
    assert rgbd is not None and '独立加载' in warning
    assert rgbd['source']['frame_number'] == 7
    assert rgbd['mesh_file'] is None and rgbd['T_camera_bowl'] is None
    np.testing.assert_allclose(rgbd['cloud'][0, 0], [0, 0, 1])
    np.testing.assert_allclose(rgbd['T_sdk_camera'][:3, 3], [.5, 0, 0])
    _, manifest = verify_scene_archive(archive.root)
    assert manifest['perception_status'] == 'captured'
    assert all(manifest['initial_frame_files'])


def test_initial_cloud_does_not_replace_mismatched_capture_with_another_frame(tmp_path):
    archive, scene = archive_capture(tmp_path, raw_frame=8)
    with pytest.raises(ValueError, match='不是同帧'):
        load_recorded_initial_rgbd(archive.root, scene)
    _, manifest = verify_scene_archive(archive.root)
    assert manifest['perception_status'] == 'frame_mismatch'


def test_successful_trajectory_is_immediately_saved_with_failure_only_logging(tmp_path):
    stage = _stage()
    with RealworldRunDiagnostics(tmp_path, failure_only=True) as recorder:
        recorder.curobo_logs.record(SimpleNamespace(), [stage.request], [stage.result])
        paths = list((tmp_path/'planned_trajectories').glob('*.json'))
        assert len(paths) == 1  # finish 前即可供 --follow 加载。
        value = json.loads(paths[0].read_text())
        assert value['source_format'] == 'planned_motion_trajectory'
        assert value['trajectory'] == stage.result.trajectory.to_dict()
        assert not (tmp_path/'motion_results').exists()
        recorder.finish({'success': True})
    rows = _load_trajectories(tmp_path, tmp_path/'none', {})
    assert len(rows) == 1
    np.testing.assert_allclose(rows[0]['positions_rad'], stage.result.trajectory.positions_rad)


def test_selected_cached_stage_is_saved_before_any_motion_and_is_not_duplicated(tmp_path):
    from gene_agent.realworld.runtime import RealWorldRuntime
    robot = FakeRobot(); stage = _stage()
    runtime = RealWorldRuntime(robot, scene_snapshot_sha256='current', max_start_error_rad=.02,
                               tcp_height_fn=lambda _: .2, min_tcp_z_m=.08)
    with RealworldRunDiagnostics(tmp_path, failure_only=True):
        assert runtime.run_motion_stage(stage)['status'] == 'validated'
        assert runtime.run_motion_stage(stage)['status'] == 'validated'
        assert len(list((tmp_path/'planned_trajectories').glob('*.json'))) == 1
    assert robot.calls == 0


def test_old_door_skill_stage_and_motion_result_are_discovered_without_selection(tmp_path):
    stage = _stage().to_dict()
    path = tmp_path/'door/pre_contact/skill_stage.json'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps(stage))
    result = tmp_path/'motion_results/old.json'; result.parent.mkdir()
    result.write_text(json.dumps(stage['result']))
    rows = _load_trajectories(tmp_path, tmp_path/'none', {})
    assert len(rows) == 1  # 相同轨迹只播放一份。


def test_no_motion_trajectory_is_fabricated_for_ik_only_or_failed_plans(tmp_path):
    from dataclasses import replace
    from gene_agent.motion.model import MotionOperation
    stage = _stage()
    with RealworldRunDiagnostics(tmp_path, failure_only=True) as recorder:
        recorder.record_trajectory(replace(stage.request, operation=MotionOperation.IK), stage.result)
        recorder.record_trajectory(stage.request, _stage(success=False).result)
    assert not (tmp_path/'planned_trajectories').exists()
