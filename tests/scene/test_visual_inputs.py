"""纯视觉输入的依赖边界及独立 FK 对照。"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_custom_table_visual_matches_physics(tmp_path):
    import json
    from gene_agent.scene.visual_inputs import load_visual_inputs
    config=json.loads((ROOT/'data/task_1/simulation/mujoco_task1_physical.json').read_text())
    config['table']={'center_xy_m':[.25,.3],'size_xy_m':[1.4,1.2],'top_z_m':-.12,'thickness_m':.04}
    source=tmp_path/'task.json';source.write_text(json.dumps(config))
    scene=load_visual_inputs('microwave',source,tmp_path/'visual')
    table=next(row for row in scene.static_visuals if row['name']=='table')
    assert np.allclose(table['size_xyz_m'],[1.4,1.2,.04])
    assert np.allclose(table['T_world_visual'][:3,3],[.25,.3,-.14])
    assert scene.table_top_z_m==-.12


def test_visual_inputs_do_not_import_mujoco(tmp_path):
    code = '''
import importlib.abc, sys
class RejectMujoco(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mujoco' or fullname.startswith('mujoco.'):
            raise AssertionError('纯视觉路径导入了 MuJoCo: ' + fullname)
sys.meta_path.insert(0, RejectMujoco())
from gene_agent.scene.visual_inputs import load_visual_inputs
for task, source in [('sweep','data/realworld/brush_cube_dustpan_0'),
                     ('microwave','data/task_1/simulation/mujoco_task1_physical.json')]:
    scene = load_visual_inputs(task, source, sys.argv[1]+'/'+task)
    for group, robot in scene.robots.items():
        robot.fk(robot.start_positions_rad)
        settings, snapshot = scene.prepare_planner(group, robot.home_width_m)
        assert snapshot['collision_mode'] == 'robot_only'
        assert not snapshot['environment_geometry_exported']
        assert settings['gripper_snapshot']['width_m'] > 0
assert 'mujoco' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('task,source', [
    ('sweep', 'data/realworld/brush_cube_dustpan_0'),
    ('microwave', 'data/task_1/simulation/mujoco_task1_physical.json'),
])
def test_visual_fk_matches_independent_mujoco(task, source, tmp_path):
    import mujoco
    from gene_agent.scene.visual_inputs import load_visual_inputs
    scene = load_visual_inputs(task, ROOT/source, tmp_path)
    rng = np.random.default_rng(29)
    for robot in scene.robots.values():
        model = mujoco.MjModel.from_xml_path(str(robot.urdf_path))
        data = mujoco.MjData(model)
        for q in (robot.start_positions_rad, np.zeros(7), rng.uniform(-.6, .6, 7)):
            for name, value in zip(robot.urdf_joint_names, q):
                data.qpos[model.joint(name).qposadr[0]] = value
            mujoco.mj_forward(model, data)
            link = data.body('link_7')
            transform = np.eye(4)
            transform[:3, :3] = link.xmat.reshape(3, 3)
            transform[:3, 3] = link.xpos
            expected = robot.world_from_base @ transform @ robot.link_from_tcp
            np.testing.assert_allclose(robot.fk(q), expected, atol=2e-7)


def test_robot_only_retains_body_and_other_arm(tmp_path):
    from gene_agent.scene.visual_inputs import load_visual_inputs
    scene = load_visual_inputs('microwave', ROOT/'data/task_1/simulation/mujoco_task1_physical.json', tmp_path)
    _, snapshot = scene.prepare_planner('right', .06)
    assert snapshot['frozen_groups'] == ['left']
    assert any(r['body_name'] == 'robot_body' for r in snapshot['records'])
    assert any(r['body_name'].startswith('left_') for r in snapshot['records'])
    assert not any(r['body_name'] in {'microwave', 'bowl', 'table'} for r in snapshot['records'])
    assert snapshot['mounting_contact_policy']['robot_links'] == ['base_link', 'link_1']
    scene.robots['left'].set_state(np.zeros(7),.04)
    _,updated=scene.prepare_planner('right',.06)
    assert snapshot['scene_model']!=updated['scene_model']


def test_standard_gripper_geometry_matches_physics_export(tmp_path):
    import mujoco
    from gene_agent.robots.realman import build_robot
    from gene_agent.motion.curobo.gripper import export_gripper_snapshot
    from gene_agent.scene.visual_inputs import load_visual_inputs
    scene = load_visual_inputs('microwave', ROOT/'data/task_1/simulation/mujoco_task1_physical.json', tmp_path/'visual')
    metadata = build_robot(tmp_path/'physics')
    model=mujoco.MjModel.from_xml_path(metadata['xml_path']);data=mujoco.MjData(model)
    for group, robot in scene.robots.items():
        for width in (.02,.06):
            for name in metadata['grippers'][group]['joint_names']:
                data.qpos[model.joint(name).qposadr[0]]=width/2
            mujoco.mj_forward(model,data)
            physical=export_gripper_snapshot(model,data,metadata,group)
            visual=robot.gripper_snapshot(width)
            # 球覆盖与 FK 双侧独立构造；浮点绕回差异允许 1e-9 m。
            for link,balls in physical['added_spheres_by_link'].items():
                np.testing.assert_allclose([b['center'] for b in visual['added_spheres_by_link'][link]],
                                           [b['center'] for b in balls],atol=1e-9)
                np.testing.assert_allclose([b['radius'] for b in visual['added_spheres_by_link'][link]],
                                           [b['radius'] for b in balls],atol=1e-9)
            expected=np.asarray(metadata['transforms'][group]['world_from_urdf_base'])
            np.testing.assert_allclose(robot.world_from_base,expected,atol=1e-9)


def test_legacy_frame_matches_compiled_visual(tmp_path):
    import mujoco
    from gene_agent.scene.objects import build_objects
    from gene_agent.scene.microwave_frame import estimate_legacy_microwave_frame
    from gene_agent.scene.visual_inputs import load_visual_inputs
    scene=load_visual_inputs('microwave',ROOT/'data/task_1/simulation/mujoco_task1_physical.json',tmp_path/'visual')
    objects=build_objects(tmp_path/'physics')
    model=mujoco.MjModel.from_xml_path(objects['xml_path']);data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    expected=estimate_legacy_microwave_frame(model,data,ROOT/'data/task_1')
    np.testing.assert_allclose(scene.legacy_microwave_transform,expected.T_calibrated_legacy,atol=1e-7)
    assert abs(scene.legacy_frame_alignment.handle_center_residual_m-expected.handle_center_residual_m)<1e-7


def test_frozen_inputs_keep_effective_pose_and_config(tmp_path):
    import json
    from gene_agent.scene.visual_inputs import load_visual_inputs
    config=json.loads((ROOT/'data/task_1/simulation/mujoco_task1_physical.json').read_text())
    source=tmp_path/'task.json';source.write_text(json.dumps(config))
    scene=load_visual_inputs('microwave',source,tmp_path/'first')
    config['bowl_pose_world'][0][3]+=.2
    config['turntable_height_offset_m']=0.
    source.write_text(json.dumps(config))
    reloaded=load_visual_inputs('microwave',tmp_path/'first/inputs.json',tmp_path/'second',randomize=True)
    np.testing.assert_array_equal(reloaded.objects['bowl']['T_world_object'],scene.objects['bowl']['T_world_object'])
    assert reloaded.config==scene.config


def test_sweep_randomization_freezes_sample(tmp_path):
    from gene_agent.scene.visual_inputs import load_visual_inputs
    scene=load_visual_inputs('sweep',ROOT/'data/realworld/brush_cube_dustpan_0',tmp_path/'first',seed=36,randomize=True)
    again=load_visual_inputs('sweep',tmp_path/'first/inputs.json',tmp_path/'again',seed=999,randomize=True)
    assert scene.placement['valid']
    assert not scene.placement['physics_settling_checked']
    assert scene.placement==again.placement
    for name,row in scene.objects.items():
        np.testing.assert_array_equal(row['T_world_object'],again.objects[name]['T_world_object'])


def test_full_environment_export_has_no_physics_dependency(tmp_path):
    from gene_agent.scene.visual_inputs import load_visual_inputs
    scene=load_visual_inputs('sweep',ROOT/'data/realworld/brush_cube_dustpan_0',tmp_path/'inputs')
    r=scene.robots['left']
    _,robot_only=scene.prepare_planner('left',r.home_width_m,collision_mode='robot_only')
    _,full=scene.prepare_planner('left',r.home_width_m,collision_mode='full')
    assert not robot_only['environment_geometry_exported']
    assert full['environment_geometry_exported']
    assert full['collision_mode']=='full'
    assert any(n.startswith('object_brush_') for n in full['scene_model']['mesh'])
    assert any(n.startswith('environment_') for n in full['scene_model']['cuboid'])
    assert not robot_only['scene_model']['mesh']
    assert full['snapshot_sha256']!=robot_only['snapshot_sha256']
