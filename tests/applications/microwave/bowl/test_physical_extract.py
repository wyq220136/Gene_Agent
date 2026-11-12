"""取碗物理验收不能被轨迹或单侧接触替代。"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gene_agent.tasks.policies.microwave_task.simulation_bowl.extract import (
    build_bowl_motion_trace,
    evaluate_extract_grasp,
)


def _bowl_pose(*, tilt_deg: float = 0., displacement_m: float = 0.) -> np.ndarray:
    pose = np.eye(4)
    # Bowl_0 局部 +Y 朝碗口；先将其对齐世界 +Z，再施加指定倾角。
    pose[:3, :3] = Rotation.from_euler(
        "x", 90. + tilt_deg, degrees=True,
    ).as_matrix()
    pose[0, 3] = displacement_m
    return pose


def test_grasp_contact_does_not_hide_tipped_bowl():
    contact = {"success": True, "bilateral_pad_contact": True,
               "undesired_gripper_contacts": []}
    result = evaluate_extract_grasp(
        _bowl_pose(), _bowl_pose(tilt_deg=35.), contact,
        close_execution_success=True,
    )
    assert not result["success"]
    assert result["valid_pad_contact"]
    assert result["reason_codes"] == ["bowl_tilt_exceeded"]


def test_grasp_requires_contact_and_small_displacement():
    contact = {"success": False, "bilateral_pad_contact": False,
               "undesired_gripper_contacts": []}
    result = evaluate_extract_grasp(
        _bowl_pose(), _bowl_pose(displacement_m=.025), contact,
        close_execution_success=True,
    )
    assert not result["success"]
    assert result["reason_codes"] == [
        "invalid_pad_contact", "bowl_displacement_exceeded",
    ]


def test_grasp_accepts_contact_upright_and_stable_bowl():
    contact = {"success": True, "bilateral_pad_contact": True,
               "undesired_gripper_contacts": []}
    result = evaluate_extract_grasp(
        _bowl_pose(), _bowl_pose(tilt_deg=8., displacement_m=.003), contact,
        close_execution_success=True,
    )
    assert result["success"]
    assert result["reason_codes"] == []


def test_grasp_cannot_accept_a_contact_flag_without_both_pads():
    result = evaluate_extract_grasp(
        _bowl_pose(), _bowl_pose(),
        {"success": True, "bilateral_pad_contact": False,
         "undesired_gripper_contacts": []},
        close_execution_success=True,
    )
    assert not result["success"]
    assert result["reason_codes"] == ["invalid_pad_contact"]


def test_grasp_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="有效"):
        evaluate_extract_grasp(
            _bowl_pose(), _bowl_pose(), {"success": True},
            close_execution_success=True, max_tilt_deg=0.,
        )


def test_bowl_motion_trace_locates_first_slide_and_tilt():
    initial = _bowl_pose()
    samples = []
    for time_s, displacement, tilt in (
        (10., 0., 0.), (10.1, .006, 0.),
        (10.2, .012, 6.), (10.3, .013, 18.),
    ):
        samples.append({
            "sim_time_s": time_s,
            "objects": {"bowl": {"T_world_object": _bowl_pose(
                tilt_deg=tilt, displacement_m=displacement,
            ).tolist()}},
            "robots": {"right": {
                "T_world_tcp": _bowl_pose(displacement_m=.4).tolist(),
                "gripper": {"width_m": .041},
            }},
        })
    result = build_bowl_motion_trace(samples, initial, group="right")
    assert result["sample_count"] == 4
    assert result["first_displacement_over_5mm"]["time_from_stage_start_s"] == .1
    assert result["first_tilt_over_5deg"]["time_from_stage_start_s"] == .2
    assert result["first_tilt_over_15deg"]["time_from_stage_start_s"] == .3
    assert result["max_tilt_deg"] == 18.
    assert result["max_displacement_mm"] == 13.


def test_bowl_motion_trace_reports_missing_replay_samples():
    result = build_bowl_motion_trace([], _bowl_pose(), group="right")
    assert result["sample_count"] == 0
    assert result["first_tilt_over_15deg"] is None


@pytest.mark.parametrize('contact_success', [True, False])
@pytest.mark.parametrize('raise_at', [None, 'pre_extract_1', 'close'])
def test_physical_prefix_uses_shared_grasp_gate(tmp_path, monkeypatch, contact_success, raise_at):
    from types import SimpleNamespace
    from gene_agent.tasks.policies.microwave_task.simulation_bowl import extract as module

    actions = []
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    def query(**kwargs):
        return {'state_id': str(len(actions)),
                'objects': {'bowl': {'T_world_object': pose.tolist()}},
                'robots': {'right': {'T_world_tcp': np.eye(4).tolist()}}}
    def gripper(**kwargs):
        actions.append(('gripper', kwargs['width_m'], kwargs['duration_s']))
        if raise_at == 'close' and kwargs['width_m'] == .007:
            raise RuntimeError('partial extraction close')
        return {'success': True}
    runtime = SimpleNamespace(query=query, model=SimpleNamespace(geom=lambda name: SimpleNamespace(id=0)), data=object(), metadata={},
                              scene_revision=1, robot_metadata={'grippers': {'right': {'pad_geom_names': ['pad']}}},
                              execute_gripper_command=gripper, copy_data=lambda: SimpleNamespace(
                                  geom_xpos=np.zeros((1, 3)),
                                  body=lambda name: SimpleNamespace(xmat=np.eye(3).ravel())))
    monkeypatch.setattr(module, 'PickBowlCase', lambda *args: None)
    monkeypatch.setattr(module, '_world_legacy_frame', lambda case: (np.eye(4), {}))
    monkeypatch.setattr(module, 'load_object_operation_profile', lambda *args: {
        'grasp_pregrasp_opening_width_m': .041, 'grasp_opening_width_m': .007})
    targets = SimpleNamespace(**{name: SimpleNamespace(matrix=np.eye(4))
                                for name in ('pre_extract_1', 'pre_extract_2', 'grasp')})
    monkeypatch.setattr(module, 'extract_profile_targets', lambda *args: (targets, None))
    monkeypatch.setattr(module, 'actual_bowl_extract_targets', lambda *args: (
        {name: np.eye(4) for name in ('pre_extract_1', 'pre_extract_2', 'grasp')}, {}))
    monkeypatch.setattr(module, 'close_bowl_on_contact', lambda runtime, **kwargs:
                        gripper(group='right', width_m=.007, duration_s=1.5))
    monkeypatch.setattr(module, 'model_fingerprint', lambda model: 'model')
    monkeypatch.setattr(module, 'save_state', lambda *args: None)
    monkeypatch.setattr(module, 'query_diagnostics', lambda *args, **kw: {'contacts': []})
    monkeypatch.setattr(module, 'classify_grasp_contacts', lambda *args, **kw: {
        'success': contact_success, 'bilateral_pad_contact': contact_success,
        'pad_sides': [], 'contact_count': 2 if contact_success else 0,
        'undesired_gripper_contacts': []})
    def bridge(*args, **kwargs):
        assert kwargs['linear_tcp_stages'] == ('pre_extract_2', 'grasp')
        assert not kwargs.get('self_collision_only_stages')
        assert kwargs['disabled_body_names_for_stage']('grasp', {}) == ('bowl',)
        assert kwargs['disabled_body_names_for_stage']('pre_extract_2', {}) == ()
        assert callable(kwargs['plan_postcheck_for_stage'])
        def move(name, pose, start):
            assert start == str(len(actions))
            actions.append(name)
            if name == raise_at:
                raise RuntimeError('partial extraction motion')
            return {'success': True}
        move.close = lambda: None
        return move
    monkeypatch.setattr(module, 'CuroboRuntimeBridge', bridge)
    result = module.run_physical_extract_grasp(runtime, tmp_path / 'prefix')
    if raise_at:
        assert not result['physical_grasp_prefix_success']
        assert result['shared_flow']['failure_stage'] == raise_at
        assert result['shared_flow']['end_state']['state_id'] == str(len(actions))
        assert 'RuntimeError' in result['stages'][-1]['traceback']
        assert (tmp_path / 'prefix/result.json').is_file()
        return
    assert result['physical_grasp_prefix_success'] is contact_success
    assert result['shared_flow']['success'] is contact_success
    assert result['shared_flow']['stages'][-1] == 'grasp_check'
    assert actions == [('gripper', .041, .6), 'pre_extract_1', 'pre_extract_2',
                       'grasp', ('gripper', .007, 1.5)]
