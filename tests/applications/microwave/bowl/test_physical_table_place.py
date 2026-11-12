"""桌面放置只接收同一模型、同一实际终态且仍附着的 Bowl。"""

from types import SimpleNamespace

import numpy as np
import pytest

from gene_agent.tasks.policies.microwave_task.simulation_bowl.place import (
    align_table_tcp_to_surface, run_assisted_table_place,
)
from gene_agent.motion.model import PoseTarget


def test_surface_height_alignment_keeps_old_xy_and_orientation():
    original = np.eye(4)
    original[:3, 3] = [.074589741, .289288992, -.141397623]
    target = align_table_tcp_to_surface(
        PoseTarget("world_tcp", original),
        current_top_z_m=-.120, reference_top_z_m=-.20601678639650346,
        height_bias_m=-.004,
    )
    assert target.matrix[:2, 3] == pytest.approx(original[:2, 3])
    assert target.matrix[:3, :3] == pytest.approx(original[:3, :3])
    assert target.matrix[2, 3] == pytest.approx(-.05938083660349655)


def test_place_rejects_unverified_extract_before_creating_output(tmp_path):
    output = tmp_path / "table_place"
    with pytest.raises(ValueError, match="辅助取碗成功终态"):
        run_assisted_table_place(
            None, output, {"group": "right", "object_id": "Bowl_0",
                           "assisted_extract_success": False},
        )
    assert not output.exists()


def test_place_rejects_stale_extract_state_before_planning(tmp_path):
    runtime = SimpleNamespace(query=lambda **_kwargs: {"state_id": "new"})
    output = tmp_path / "table_place"
    with pytest.raises(ValueError, match="终态与当前 MuJoCo 状态不一致"):
        run_assisted_table_place(
            runtime, output,
            {"group": "right", "object_id": "Bowl_0",
             "assisted_extract_success": True,
             "stages": [{"name": "post_extract_2", "end_state_id": "old"}]},
        )
    assert not output.exists()


def test_place_rejects_other_model_even_with_same_state(tmp_path, monkeypatch):
    from gene_agent.tasks.policies.microwave_task.simulation_bowl import place as physical_table_place

    runtime = SimpleNamespace(
        model=object(), query=lambda **_kwargs: {"state_id": "same"},
    )
    monkeypatch.setattr(physical_table_place, "model_fingerprint",
                        lambda _model: "actual_model")
    output = tmp_path / "table_place"
    with pytest.raises(ValueError, match="模型指纹不一致"):
        run_assisted_table_place(
            runtime, output,
            {"group": "right", "object_id": "Bowl_0",
             "assisted_extract_success": True,
             "model_fingerprint": "old_model",
             "stages": [{"name": "post_extract_2", "end_state_id": "same"}]},
        )
    assert not output.exists()


def test_place_rejects_missing_contact_backed_attachment(tmp_path, monkeypatch):
    from gene_agent.tasks.policies.microwave_task.simulation_bowl import place as physical_table_place

    runtime = SimpleNamespace(
        model=object(), query=lambda **_kwargs: {"state_id": "same"},
        simulation_attachments=lambda: [
            {"object_instance_id": "bowl", "activation_contact": None}],
    )
    monkeypatch.setattr(physical_table_place, "model_fingerprint",
                        lambda _model: "actual_model")
    output = tmp_path / "table_place"
    with pytest.raises(ValueError, match="真实双侧接触"):
        run_assisted_table_place(
            runtime, output,
            {"group": "right", "object_id": "Bowl_0",
             "assisted_extract_success": True,
             "model_fingerprint": "actual_model",
             "stages": [{"name": "post_extract_2", "end_state_id": "same"}]},
        )
    assert not output.exists()


@pytest.mark.parametrize('condition,success,expected_tail', [
    ('normal', True, 'final_check'),
    ('debug_preplace', None, 'preplace'),
    ('debug_release', None, 'release_check'),
    ('unsupported', False, 'release_check'),
    ('lost_hold', False, 'hold_check'),
    ('place_execution_failed', False, 'place'),
    ('place_planning_failed', False, 'place'),
])
def test_shared_table_flow_preserves_hold_release_without_critical_fallback(
    tmp_path, monkeypatch, condition, success, expected_tail,
):
    from gene_agent.tasks.policies.microwave_task.simulation_bowl import place as module
    calls = []
    bowl = np.eye(4)
    bowl[:3, :3] = [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]
    class Runtime:
        model = object()
        data = object()
        metadata = {}
        scene_revision = 1
        robot_metadata = {'grippers': {'right': {'width_range_m': [0., .08]}}}
        index = 0
        attached = True
        def query(self, **kwargs):
            return {'state_id': f'actual-{self.index}',
                    'objects': {'bowl': {'T_world_object': bowl.tolist()}},
                    'robots': {'right': {'T_world_tcp': np.eye(4).tolist()}}}
        def simulation_attachments(self):
            if not self.attached:
                return []
            return [{'object_instance_id': 'bowl',
                     'activation_contact': {'bilateral_pad_contact': True}}]
        def copy_data(self):
            return None
        def execute_gripper_command(self, **kwargs):
            calls.append('release')
            self.index += 1
            return {'success': True}
        def release_simulation_object(self, group, object_name):
            calls.append('detach')
            self.attached = False
            self.index += 1
        def execute_hold(self, **kwargs):
            assert self.attached is False
            calls.append('settle')
            self.index += 1
            return {'success': True}
    runtime = Runtime()
    class Bridge:
        def __init__(self, runtime, *args, **kwargs):
            pass
        def __call__(self, name, target, state_id):
            assert state_id == f'actual-{runtime.index}'
            calls.append(name)
            if name == 'preplace' and condition == 'lost_hold':
                runtime.attached = False
            if name == 'place' and condition.startswith('place_'):
                return {'success': False, 'plan': {
                    'success': condition == 'place_execution_failed'}}
            runtime.index += 1
            return {'success': True, 'plan': {'success': True}}
    monkeypatch.setattr(module, 'CuroboRuntimeBridge', Bridge)
    monkeypatch.setattr(module, 'model_fingerprint', lambda model: 'model')
    monkeypatch.setattr(module, '_world_legacy_frame', lambda case: (np.eye(4), {}))
    monkeypatch.setattr(module, 'load_object_operation_profile', lambda *args: {})
    monkeypatch.setattr(module, 'extract_profile_targets',
                        lambda *args: (None, PoseTarget('world_tcp', np.eye(4))))
    monkeypatch.setattr(module, '_physical_table_target', lambda runtime, target: (target, {}))
    monkeypatch.setattr(module, '_object_relative_table_target',
                        lambda *args: (PoseTarget('world_tcp', np.eye(4)), bowl, {}))
    monkeypatch.setattr(module.trimesh, 'load_mesh', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'build_attachment_surface_spheres_from_mesh',
                        lambda *args, **kwargs: (np.zeros((1, 3)), np.array([.002]), {}))
    monkeypatch.setattr(module, 'save_state', lambda *args: None)
    monkeypatch.setattr(module, '_table_support', lambda runtime: {
        'supported': condition != 'unsupported', 'source': 'test_contact_observation'})
    report = run_assisted_table_place(runtime, tmp_path / 'place', {
        'group': 'right', 'object_id': 'Bowl_0', 'assisted_extract_success': True,
        'model_fingerprint': 'model',
        'stages': [{'name': 'post_extract_2', 'end_state_id': 'actual-0'}],
    }, stop_after=condition.removeprefix('debug_') if condition.startswith('debug_') else None)
    assert report['physical_table_place_success'] is success
    assert report['shared_flow']['stages'][-1] == expected_tail
    if condition.startswith('debug_'):
        assert report['operation_success']
        assert 'retreat' not in calls
        assert runtime.attached is (condition == 'debug_preplace')
    if success:
        assert calls[-4:] == ['release', 'detach', 'settle', 'retreat']
        assert report['shared_flow']['end_state']['state_id'] == f'actual-{runtime.index}'
    if condition == 'unsupported':
        assert 'retreat' not in calls
    if condition in {'lost_hold', 'place_execution_failed', 'place_planning_failed'}:
        assert 'release' not in calls
    assert 'place_pose_fallback' not in calls
