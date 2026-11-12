"""独立场景的初态、扰动和凹腔物理验证，不调用机器人或 IK。"""
import tempfile
import unittest
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from gene_agent.scene import build_objects
from gene_agent.simulation import build_scene
from gene_agent.motion.curobo.scene import export_collision_scene


def test_scene_rebuild_works_under_ascii_process_locale(tmp_path):
    env = os.environ.copy()
    env.update({"LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONNOUSERSITE": "1"})
    result = subprocess.run(
        [sys.executable, "-c",
         "from gene_agent.simulation import build_scene; "
         "from pathlib import Path; "
         f"build_scene(Path({str(tmp_path / 'ascii_revision')!r}))"],
        cwd=Path(__file__).resolve().parents[2], env=env,
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr

class SceneObjectsTest(unittest.TestCase):
    def test_table_override_moves_and_rotates_contact_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            original = build_objects(Path(d) / 'original')
            nominal = np.array(original['sample_poses']['table'])
            np.testing.assert_allclose(
                nominal[:3, 3],
                [*original['table']['center_xy_m'],
                 original['table']['top_z_m'] - original['table']['thickness_m'] / 2],
            )
            edited_pose = nominal.copy()
            edited_pose[:3, :3] = Rotation.from_euler('xyz', [.08, -.06, .12]).as_matrix()
            edited_pose[:3, 3] += [.03, -.02, .01]
            edited = build_objects(Path(d) / 'edited', pose_overrides={'table': edited_pose})
            self.assertEqual(edited['pose_overrides'], ['table'])
            np.testing.assert_allclose(edited['sample_poses']['table'], edited_pose)
            np.testing.assert_allclose(edited['table']['T_world_table'], edited_pose)
            model = mujoco.MjModel.from_xml_path(edited['xml_path'])
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            geom_id = model.geom('scene_table').id
            np.testing.assert_allclose(data.geom_xpos[geom_id], edited_pose[:3, 3])
            np.testing.assert_allclose(data.geom_xmat[geom_id].reshape(3, 3), edited_pose[:3, :3])
            self.assertNotEqual(model.geom_contype[geom_id], 0)

    def test_table_override_changes_exported_collision_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            model, data, meta = build_scene(Path(d) / 'nominal')
            original = export_collision_scene(model, data, meta['robot'], 'right', Path(d) / 'nominal_collision')
            table_pose = np.array(meta['objects']['sample_poses']['table'])
            table_pose[:3, :3] = Rotation.from_euler('xyz', [.04, .02, .1]).as_matrix()
            table_pose[:3, 3] += [.02, 0, .015]
            moved_model, moved_data, moved_meta = build_scene(
                Path(d) / 'moved', pose_overrides={'table': table_pose}
            )
            edited = export_collision_scene(
                moved_model, moved_data, moved_meta['robot'], 'right', Path(d) / 'moved_collision'
            )
            def table_record(snapshot):
                return next(row for row in snapshot['records'] if row['geom_name'] == 'scene_table')
            np.testing.assert_allclose(table_record(edited)['T_world_geom'], table_pose)
            self.assertNotEqual(original['snapshot_sha256'], edited['snapshot_sha256'])
            self.assertEqual(table_record(edited)['export_kind'], 'cuboid')

    def test_explicit_pose_override_moves_microwave_and_support_together(self):
        with tempfile.TemporaryDirectory() as d:
            original = build_objects(Path(d) / 'original')
            microwave = np.array(original['sample_poses']['microwave'])
            microwave[0, 3] -= .08
            edited = build_objects(Path(d) / 'edited', pose_overrides={'microwave': microwave})
            np.testing.assert_allclose(edited['sample_poses']['microwave'], microwave)
            support = np.array(edited['microwave_support']['T_world_support'])
            old_support = np.array(original['microwave_support']['T_world_support'])
            self.assertAlmostEqual(support[0, 3] - old_support[0, 3], -.08)
            self.assertAlmostEqual(support[2, 3], old_support[2, 3])

    def test_invalid_pose_override_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            pose = np.eye(4)
            pose[0, 0] = 2
            with self.assertRaisesRegex(ValueError, '刚体'):
                build_objects(Path(d), pose_overrides={'bowl': pose})

    def test_microwave_support_matches_cabinet_footprint_and_raises_asset(self):
        with tempfile.TemporaryDirectory() as d:
            meta = build_objects(Path(d))
            support = meta['microwave_support']
            self.assertGreaterEqual(support['height_m'], .15)
            np.testing.assert_allclose(
                support['size_xy_m'], [.4479437482, .2682940816], atol=1e-10
            )
            np.testing.assert_allclose(
                support['center_xy_local_m'], [0, .004065061842], atol=1e-12
            )
            self.assertAlmostEqual(
                meta['sample_poses']['microwave'][2][3],
                meta['table']['top_z_m'] + support['height_m'],
            )
            model = mujoco.MjModel.from_xml_path(str(meta['xml_path']))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            geom = model.geom('microwave_support_collision')
            np.testing.assert_allclose(
                model.geom_size[geom.id],
                [support['size_xy_m'][0] / 2, support['size_xy_m'][1] / 2, support['height_m'] / 2],
            )
            top_z = data.geom_xpos[geom.id, 2] + model.geom_size[geom.id, 2]
            self.assertAlmostEqual(top_z, meta['sample_poses']['microwave'][2][3])
            self.assertFalse(any(contact.dist < -1e-6 for contact in data.contact))

    def test_settles_and_cavity_is_open(self):
        with tempfile.TemporaryDirectory() as d:
            meta = build_objects(Path(d))
            m = mujoco.MjModel.from_xml_path(str(meta['xml_path']))
            data = mujoco.MjData(m)
            for _ in range(1500):
                mujoco.mj_step(m, data)
            self.assertLess(np.linalg.norm(data.qvel[:6]), .01)
            body = m.body('bowl').id
            self.assertLess(np.linalg.norm(data.xpos[body] - np.array(meta['sample_poses']['bowl'])[:3,3]), .025)
            # 射线从碗内部向局部底部，应命中底而非被凸包封住的碗口。
            origin = data.xpos[body] + data.xmat[body].reshape(3,3) @ np.array([0,.09,0])
            direction = data.xmat[body].reshape(3,3) @ np.array([0,-1,0])
            # mj_ray 忽略 alpha=0；测试时仅显现接触代理，不改变接触几何。
            m.geom_rgba[m.geom_group == 3,3] = 1
            geom = np.array([-1],dtype=np.int32)
            distance = mujoco.mj_ray(m,data,origin,direction,np.array([0,0,0,1,0,0],dtype=np.uint8),1,-1,geom)
            self.assertGreater(distance,.10)
            self.assertLess(distance,.15)
            self.assertEqual(m.geom(geom[0]).name,'bowl_bottom')

    def test_reproducible_bounded_perturbation(self):
        with tempfile.TemporaryDirectory() as d:
            a=build_objects(Path(d)/'a',seed=17,perturb=True)
            b=build_objects(Path(d)/'b',seed=17,perturb=True)
            self.assertEqual(a['sample_poses'],b['sample_poses'])
            for name in ['bowl','microwave']:
                delta=a['perturbations'][name]
                self.assertLessEqual(abs(delta['dx_m']),.015)
                self.assertLessEqual(abs(delta['dy_m']),.015)
                self.assertLessEqual(abs(delta['yaw_rad']),.087267)
                self.assertEqual(a['sample_poses'][name][2][3],a['nominal_poses'][name][2][3])

    def test_probe_falls_into_bowl(self):
        with tempfile.TemporaryDirectory() as d:
            meta=build_objects(Path(d))
            root=ET.parse(meta['xml_path']).getroot()
            T=np.array(meta['sample_poses']['bowl'])
            pos=T[:3,3]+T[:3,:3]@np.array([0,.12,0])
            probe=ET.SubElement(root.find('worldbody'),'body',name='probe',pos=' '.join(map(str,pos)))
            ET.SubElement(probe,'freejoint',name='probe_free')
            ET.SubElement(probe,'geom',name='probe_geom',type='sphere',size='.008',mass='.002')
            m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
            data=mujoco.MjData(m)
            touched_bottom=False
            for _ in range(1500):
                mujoco.mj_step(m,data)
                for contact in data.contact:
                    names={m.geom(contact.geom1).name,m.geom(contact.geom2).name}
                    touched_bottom |= names=={'probe_geom','bowl_bottom'}
            bowl=m.body('bowl').id
            local=data.xmat[bowl].reshape(3,3).T@(data.xpos[m.body('probe').id]-data.xpos[bowl])
            self.assertTrue(touched_bottom)
            self.assertLess(local[1],0)
            self.assertLess(np.linalg.norm(local[[0,2]]),.035)

    def test_collision_layers_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=build_objects(Path(d))['resolved_config']['bowl_proxy']
            dr=(cfg['rim_radius_m']-cfg['bottom_radius_m'])/cfg['layers']
            # 第一层须与底盘重叠；各层完整厚度须覆盖层间半径步长。
            self.assertGreater(cfg['wall_half_thickness_m'],dr/2)
            self.assertGreater(2*cfg['wall_half_thickness_m'],dr)

    def test_calibrated_asset_initial_state_and_joint_axes(self):
        with tempfile.TemporaryDirectory() as d:
            meta=build_objects(Path(d));m=mujoco.MjModel.from_xml_path(meta['xml_path']);v=mujoco.MjData(m);mujoco.mj_forward(m,v)
            self.assertFalse(any(c.dist < -1e-6 for c in v.contact))
            self.assertIn('captured_poses',meta)
            self.assertFalse(np.allclose(meta['captured_poses']['microwave'],meta['sample_poses']['microwave']))
            axis=v.xmat[m.body('knob_0').id].reshape(3,3)@m.jnt_axis[m.joint('knob_0_joint').id]
            np.testing.assert_allclose(axis,np.array(meta['T_world_knob_0'])[:3,2],atol=1e-10)
            for angle in np.linspace(0,1.91986,25):
                v.qpos[m.jnt_qposadr[m.joint('microwave_door_joint').id]]=angle;mujoco.mj_forward(m,v)
                self.assertFalse(any(c.dist < -1e-5 for c in v.contact),[(m.geom(c.geom1).name,m.geom(c.geom2).name,c.dist) for c in v.contact])

    def test_calibrated_cavity_probe_and_door_blockage(self):
        with tempfile.TemporaryDirectory() as d:
            meta=build_objects(Path(d));root=ET.parse(meta['xml_path']).getroot();T=np.array(meta['sample_poses']['microwave'])
            probe=ET.SubElement(root.find('worldbody'),'body',name='microwave_probe',pos=' '.join(map(str,T[:3,:3]@[-.04,0,.15]+T[:3,3])))
            ET.SubElement(probe,'freejoint',name='microwave_probe_free');ET.SubElement(probe,'geom',name='microwave_probe_geom',type='sphere',size='.008',mass='.002')
            m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));v=mujoco.MjData(m)
            touched=False
            for _ in range(1500):
                mujoco.mj_step(m,v)
                touched|=any({m.geom(c.geom1).name,m.geom(c.geom2).name}=={'microwave_probe_geom','microwave_turntable_collision'} for c in v.contact)
            self.assertTrue(touched)
            # 同一条入口路径，门闭合必须阻挡、门打开必须净空；不靠关闭碰撞通过。
            counts=[];q=m.jnt_qposadr[m.joint('microwave_probe_free').id]
            for door_angle in [0,np.deg2rad(100)]:
                blocked=0
                for y in np.linspace(-.20,0,50):
                    v.qpos[q:q+3]=T[:3,:3]@[-.04,y,.12]+T[:3,3]
                    v.qpos[m.jnt_qposadr[m.joint('microwave_door_joint').id]]=door_angle;mujoco.mj_forward(m,v)
                    blocked+=any('microwave_probe_geom' in {m.geom(c.geom1).name,m.geom(c.geom2).name} and c.dist<0 for c in v.contact)
                counts.append(blocked)
            self.assertGreater(counts[0],0);self.assertEqual(counts[1],0)

    def test_native_mjcf_joint_parity_and_perturbed_clearance(self):
        from scipy.spatial.transform import Rotation
        native=Path(__file__).parents[2]/'assets/objects/microwave_calibrated/source_mjcf.xml'
        nr=ET.parse(native).getroot();nr.find('compiler').set('meshdir',str(native.parent/'meshes'))
        nm=mujoco.MjModel.from_xml_string(ET.tostring(nr,encoding='unicode'));nd=mujoco.MjData(nm)
        with tempfile.TemporaryDirectory() as d:
            for seed in range(10):
                meta=build_objects(Path(d),seed=seed,perturb=True);m=mujoco.MjModel.from_xml_path(meta['xml_path']);v=mujoco.MjData(m);mujoco.mj_forward(m,v)
                self.assertFalse(any(c.dist< -1e-6 for c in v.contact))
                T=np.array(meta['sample_poses']['microwave'])
                for values in [[0,0,0],[.6,1.2,2],[1.7,4,4.5]]:
                    for old,new,value in zip(['door_hinge','power_knob_rotation','timer_knob_rotation'],['microwave_door_joint','knob_0_joint','knob_1_joint'],values):
                        nd.qpos[nm.jnt_qposadr[nm.joint(old).id]]=value;v.qpos[m.jnt_qposadr[m.joint(new).id]]=value
                    mujoco.mj_forward(nm,nd);mujoco.mj_forward(m,v)
                    for old,new in [('door','microwave_door'),('power_knob','knob_0'),('timer_knob','knob_1')]:
                        nb=nm.body(old).id;b=m.body(new).id
                        np.testing.assert_allclose(v.xpos[b],T[:3,:3]@nd.xpos[nb]+T[:3,3],atol=1e-9)
                        np.testing.assert_allclose(v.xmat[b].reshape(3,3),T[:3,:3]@nd.xmat[nb].reshape(3,3),atol=1e-9)
                        np.testing.assert_allclose(m.body_inertia[b],nm.body_inertia[nb],atol=1e-10)
