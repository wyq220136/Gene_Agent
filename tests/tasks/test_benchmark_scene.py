"""Benchmark 可脱离 policy 应用完整初始姿态，seed 和旋转语义保持。"""
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from benchmarks.sweep_cube_task.scene import OBJECTS, place_initial_objects


def test_full_pose_override_keeps_roll_pitch_and_height():
    bodies = "".join(f'<body name="{name}" pos="{i * .2} 0 .1">'
                     f'<freejoint name="{name}_free"/><geom type="sphere" size=".01" mass=".1"/>'
                     '</body>' for i, name in enumerate(OBJECTS))
    model = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>{bodies}</worldbody></mujoco>')
    data = mujoco.MjData(model)
    poses = {}
    for i, name in enumerate(OBJECTS):
        pose = np.eye(4)
        pose[:3, 3] = [i * .2, 0., .1]
        poses[name] = {"T_world_object": pose.tolist(), "extents_xyz_m": [.02, .02, .02]}
    override = np.array(poses["brush"]["T_world_object"])
    override[:3, :3] = Rotation.from_euler("xyz", [20, 10, 30], degrees=True).as_matrix()
    override[2, 3] = .16
    poses["brush"]["T_world_object"] = override.tolist()
    scene = {"objects": poses, "initial_pose_overrides": {"brush": override.tolist()},
             "randomization": {"ranges_xyyaw_rad": {n: [.001, .001, .01] for n in OBJECTS},
                               "table_bounds_xy_m": [-2, 2, -2, 2],
                               "min_clearance_m": .01, "max_attempts": 100}}
    environment = SimpleNamespace(model=model, data=data, scene=scene)
    first = place_initial_objects(environment, randomize=False, seed=17)
    np.testing.assert_allclose(data.body("brush").xmat.reshape(3, 3), override[:3, :3], atol=1e-12)
    assert data.body("brush").xpos[2] == .16
    a = place_initial_objects(environment, randomize=True, seed=17)
    a_qpos = data.qpos.copy()
    b = place_initial_objects(environment, randomize=True, seed=17)
    np.testing.assert_array_equal(a_qpos, data.qpos)
    assert a["sample"] == b["sample"]
    assert first["mode"] == "nominal"


def test_microwave_scene_can_load_without_planner_or_task_policy():
    import subprocess, sys
    script = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('gene_agent.motion.curobo', 'gene_agent.tasks.policies')):
        raise RuntimeError('scene must not load solver/policy: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from benchmarks.microwave_task.scene import build_initial_scene, resolve_settings
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_microwave_benchmark_input_excludes_policy_tuning():
    import json
    from pathlib import Path
    from benchmarks.microwave_task.scene import environment_settings
    from gene_agent.tasks.policies.microwave_task.policy import operation_parameters
    raw = json.loads(Path('data/task_1/simulation/mujoco_task1_physical.json').read_text())
    environment = environment_settings(raw)
    parameters = operation_parameters(raw)
    assert environment['bowl_pose_world'] == raw['bowl_pose_world']
    assert environment['simulation_attachment_pairs'] == raw['simulation_attachment_pairs']
    assert parameters['pick_depth_m'] == raw['pick_depth_m']
    assert parameters['knob_azimuth_deg'] == raw['knob_azimuth_deg']
    assert not (environment.keys() & parameters.keys())
    assert 'pick_depth_m' not in environment
    assert 'microwave_pose_world' not in parameters


def test_long_brush_benchmark_does_not_choose_a_policy(tmp_path):
    import json
    from pathlib import Path
    from benchmarks.sweep_long_task.scene import prepare_long_brush_scene
    source = Path('data/realworld/my_scene_0/scenes/sweep_long')
    scene = prepare_long_brush_scene(source, tmp_path / 'benchmark')
    raw = json.loads((scene / 'scene.json').read_text(encoding='utf-8'))
    assert not {'grasp_mode', 'grasp_yaw_deg', 'target_order_policy',
                'alignment_lateral_trigger_m', 'goal_inset_extra_m'} & raw.keys()
    assert raw['simulation_attachment_pairs'] == {'left': 'brush'}
