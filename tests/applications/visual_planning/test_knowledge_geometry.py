"""纯视觉操作使用源 URDF 的轴、闭门参考和共享 Knowledge。"""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from gene_agent.tasks.planning.knowledge import proposals


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def inputs():
    path = ROOT / 'assets/objects/microwave_calibrated/microwave_calibrated.urdf'
    urdf = URDF.load(str(path), load_meshes=False)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler('z', .63).as_matrix()
    world[:3, 3] = [.5, .3, -.1]
    joints = {'door_hinge': .27, 'power_knob_rotation': .1}
    urdf.update_cfg(joints)
    return SimpleNamespace(
        objects={'microwave': {'T_world_object': world}},
        object_urdfs={'microwave': (urdf, path)},
        object_joint_positions={'microwave': joints},
        robots={'right': SimpleNamespace(world_from_base=np.eye(4)),
                'left': SimpleNamespace(world_from_base=np.eye(4))},
        legacy_microwave_transform=np.eye(4),
    )


def test_push_uses_closed_reference_once_and_normal_standoff(inputs):
    rows = proposals(inputs, 'door_push', parameters={'start_angle_rad': .4,
                     'target_angle_rad': .6, 'step_rad': .2})
    assert [r['waypoint_label'] for r in rows] == ['pre_contact', 'contact', 'push_01']
    joint = inputs.object_urdfs['microwave'][0].joint_map['door_hinge']
    world = inputs.objects['microwave']['T_world_object']
    axis = world[:3, :3] @ joint.axis
    hinge = (world @ joint.origin)[:3, 3]
    np.testing.assert_allclose(rows[1]['axis_direction_world'], axis)
    np.testing.assert_allclose(rows[1]['axis_point_world'], hinge)
    normal = Rotation.from_rotvec(axis * .4).apply(world[:3, 1])
    # right_open 默认推门板内侧；pre-contact 沿面外法向退开 5 cm。
    np.testing.assert_allclose(rows[1]['surface_normal_world'], normal)
    np.testing.assert_allclose(rows[0]['pose'][:3, 3] - rows[1]['pose'][:3, 3], .05 * normal)
    contact = np.asarray(rows[1]['contact_points_world'])[0]
    np.testing.assert_allclose(rows[1]['pose'][:3, 3] - contact,
                               rows[1]['parameters']['standoff_m'] * normal)
    assert rows[-1]['object_joint_positions']['microwave']['door_hinge'] == pytest.approx(.6)
    assert inputs.object_joint_positions['microwave']['door_hinge'] == .27
    for row in rows:
        json.dumps({key: value for key, value in row.items() if key != 'pose'}, allow_nan=False)


def test_handle_pull_uses_current_angle_and_gripper_stages(inputs):
    rows = proposals(inputs, 'handle_pull', parameters={'target_angle_rad': .6})
    assert [r['waypoint_label'] for r in rows[:3]] == ['pregrasp', 'grasp', 'close']
    assert [r['waypoint_label'] for r in rows[-2:]] == ['release', 'retreat']
    np.testing.assert_allclose(rows[0]['pose'][:3, 3],
        rows[1]['pose'][:3, 3] - .03 * rows[1]['pose'][:3, 2])
    assert rows[1]['width_m'] == .08 and rows[2]['width_m'] == .015
    assert rows[0]['object_joint_positions']['microwave']['door_hinge'] == .27
    assert rows[-1]['object_joint_positions']['microwave']['door_hinge'] == .6
    before = np.asarray(rows[2]['contact_points_world'])[0]
    after = np.asarray(rows[-2]['contact_points_world'])[0]
    axis = np.asarray(rows[0]['axis_direction_world'])
    hinge = np.asarray(rows[0]['axis_point_world'])
    np.testing.assert_allclose(after, hinge + Rotation.from_rotvec(axis * .33).apply(before - hinge))


def test_knob_axis_matches_urdf_and_current_angle_is_not_applied_twice(inputs):
    rows = proposals(inputs, 'knob_turn', parameters={'target_angle_rad': .3, 'step_rad': .1})
    world = inputs.objects['microwave']['T_world_object']
    joint = inputs.object_urdfs['microwave'][0].joint_map['power_knob_rotation']
    axis = world[:3, :3] @ joint.axis
    np.testing.assert_allclose(rows[0]['axis_direction_world'], axis, atol=1e-12)
    point = np.asarray(rows[0]['axis_point_world'])
    expected = Rotation.from_rotvec(axis * .2)
    np.testing.assert_allclose(rows[-1]['pose'][:3, :3], expected.as_matrix() @ rows[0]['pose'][:3, :3])
    np.testing.assert_allclose(rows[-1]['pose'][:3, 3], point + expected.apply(rows[0]['pose'][:3, 3] - point))
    assert rows[0]['object_joint_positions']['microwave']['power_knob_rotation'] == .1
    assert rows[-1]['object_joint_positions']['microwave']['power_knob_rotation'] == .3


def test_knowledge_import_and_generation_do_not_load_mujoco():
    result = subprocess.run([sys.executable, '-c', '''
import importlib.abc
import sys
class RejectPhysics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'mujoco' or fullname.startswith('mujoco.'):
            raise RuntimeError('Pure geometry attempted to import MuJoCo')
sys.meta_path.insert(0, RejectPhysics())
import runpy
namespace = runpy.run_path('tests/applications/visual_planning/test_knowledge_geometry.py')
inputs = namespace['inputs'].__wrapped__()
for kind in ('door_push', 'handle_pull', 'knob_turn'):
    assert namespace['proposals'](inputs, kind)
assert not any(name == 'mujoco' or name.startswith('mujoco.') for name in sys.modules)
'''], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_operation_rejects_joint_limit_and_unknown_parameter(inputs):
    with pytest.raises(ValueError, match='限位'):
        proposals(inputs, 'door_push', parameters={'target_angle_rad': 4.})
    with pytest.raises(ValueError, match='未知'):
        proposals(inputs, 'knob_turn', parameters={'typo': 1})
