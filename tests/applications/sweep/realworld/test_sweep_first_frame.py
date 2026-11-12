"""首帧输入契约，不需要相机、SDK 或模型服务。"""
import json
import numpy as np
import pytest
from gene_agent.realworld.sweep.first_frame import load_calibration, check_capture, estimate_table


def test_final_extrinsic_used_once(tmp_path):
    t=np.eye(4);t[0,3]=.8
    p=tmp_path/'cam.json'
    p.write_text(json.dumps({'serial':'a','intrinsic_matrix':[[100,0,2],[0,100,2],[0,0,1]],
        'extrinsic_matrix':t.tolist(),'board_to_robot':(t@t).tolist(),'width':4,'height':4}))
    c=load_calibration(p)
    np.testing.assert_array_equal(c['T_sdk_camera'],t)
    check_capture(c,{'serial':'a','intrinsic_matrix':c['intrinsic_matrix']},(4,4))
    with pytest.raises(ValueError,match='序列号'):
        check_capture(c,{'serial':'b','intrinsic_matrix':c['intrinsic_matrix']},(4,4))


def test_table_rejects_vertical_background_and_excludes_objects():
    # 深度相机朝下，桌面在 SDK z=0，物体区域不可进入桌面拟合。
    t=np.diag([1.,-1.,-1.,1.]);t[2,3]=1
    depth=np.ones((60,80));depth[:10]=.8
    excluded=np.zeros_like(depth,dtype=bool);excluded[:10]=True
    camera={'intrinsic_matrix':[[100,0,40],[0,100,30],[0,0,1]],'T_sdk_camera':t}
    pose,evidence=estimate_table(depth,excluded,camera,[.96,1,.04])
    assert pose[2,3]==pytest.approx(-.02,abs=1e-6)
    assert evidence['inlier_ratio']>.99
    camera['T_sdk_camera']=np.eye(4)
    # 固定 z 平面允许朝上相机；斜面/竖直背景不能误认为水平桌面。
    bad=np.linspace(.2,1.2,60)[:,None]*np.ones((1,80))
    with pytest.raises(ValueError,match='桌面'):
        estimate_table(bad,excluded,camera,[.96,1,.04])


def test_first_frame_cannot_enter_held_scene_api():
    import time
    from gene_agent.realworld.sweep.observation import validate_snapshot
    s={'schema_version':1,'source':'first_frame_preview','execution_enabled':False,
       'initial_brush_state':'on_table','robot_state_source':'assumed_template_standby',
       'joint_names':[f'joint_{i}' for i in range(1,8)],'joint_positions_rad':[0]*7,
       'gripper_width_m':.04,'pose_frame_verified':True,'T_sdk_flange':np.eye(4).tolist(),
       'T_sdk_object':{r:np.eye(4).tolist() for r in ['brush','cube','dustpan']},
       'observed_at_unix_ms':time.time()*1000,'hold':{'confirmed':False},
       'camera_serial':'fixture','camera_frame_number':1}
    with pytest.raises(ValueError,match='显式'):
        validate_snapshot(s,max_age_s=120)
    validate_snapshot(s,max_age_s=120,allow_first_frame_preview=True)
    s['hold']['confirmed']=True
    with pytest.raises(ValueError,match='不能声称'):
        validate_snapshot(s,max_age_s=120,allow_first_frame_preview=True)


def test_visual_mesh_export_uses_body_frame_once():
    import mujoco
    from gene_agent.scene.rigid_geometry import visual_mesh_object
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="object" pos="1 2 3" quat=".70710678 0 0 .70710678">
      <geom type="box" size=".01 .02 .03" pos=".04 0 0" contype="0" conaffinity="0"/>
      <geom type="box" size="1 1 1" group="3"/>
      </body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    mesh=visual_mesh_object(m,d,'object')
    np.testing.assert_allclose(mesh.bounds,[[.03,-.02,-.03],[.05,.02,.03]],atol=1e-8)


def test_pregrasp_prediction_does_not_attach_object():
    import mujoco
    from gene_agent.realworld.sweep.planning import _set_prediction
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="arm"><joint name="q" type="slide"/><geom size=".01" mass="1"/>
      <site name="pika_tcp"/></body><body name="brush" pos="1 2 3"><freejoint/>
      <geom size=".01" mass="1"/></body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    _set_prediction(m,d,{'joint_groups':{'left':['q']}},[.2],None)
    np.testing.assert_allclose(d.body('brush').xpos,[1,2,3])
    _set_prediction(m,d,{'joint_groups':{'left':['q']}},[.3],np.eye(4))
    np.testing.assert_allclose(d.body('brush').xpos,d.site('pika_tcp').xpos)


def test_failed_sweep_keeps_validated_prefix(tmp_path,monkeypatch):
    import mujoco
    from types import SimpleNamespace
    from gene_agent.realworld.sweep.scene import read_site,ROOT
    from gene_agent.realworld.sweep import planning,preview_motion
    from gene_agent.robots.pika_frames import pose_from_mujoco
    site=read_site(ROOT/'config/realworld/sweep_old.example.json')
    m=mujoco.MjModel.from_xml_path(site['scene_template']);d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
    (tmp_path/'scene/concepts').mkdir(parents=True)
    for role,row in site['objects'].items():
        (tmp_path/f'scene/concepts/{role}.json').write_bytes(__import__('pathlib').Path(row['concept_path']).read_bytes())
    robot={'joint_groups':{'left':[f'joint_{i}' for i in range(1,8)]},'world_frame':'test'}
    grip=np.linalg.inv(pose_from_mujoco(d.site('pika_tcp')))@pose_from_mujoco(d.body('brush'))
    monkeypatch.setattr(planning,'CuroboPlannerPool',lambda **kw:SimpleNamespace(clear=lambda:None))
    monkeypatch.setattr(planning,'check_world_contacts',lambda *a,**kw:{'success':True})
    def plan(*args,**kwargs):
        stage=args[5]
        success=stage!='sweep'
        return {'name':stage,'success':success},([{'stage':stage,'qpos':d.qpos.tolist(),'predicted':True}] if success else [])
    monkeypatch.setattr(preview_motion,'plan_stage',plan)
    site.update(snapshot_source='first_frame_preview',max_candidates=1,planning_start_source='predicted_after_pick')
    report=planning.plan_sweep(m,d,robot,grip,site,tmp_path)
    frames=json.loads((tmp_path/'preview_frames.json').read_text())
    assert not report['planning_complete']
    assert [f['stage'] for f in frames]==['predicted_after_pick','move_behind_cube','lower_to_contact']
    assert report['candidates'][0]['stages'][0]['start_source']=='predicted_after_pick'


def test_live_robot_is_default_and_standby_requires_explicit_choice():
    from gene_agent.realworld.sweep.observation import resolve_preview_robot_ip
    assert resolve_preview_robot_ip({'devices':{'robot_ip':'FILL_ROBOT_IP'}},None)=='169.254.128.19'
    assert resolve_preview_robot_ip({'devices':{'robot_ip':'192.168.1.9'}},None)=='192.168.1.9'
    assert resolve_preview_robot_ip({},'192.168.1.8')=='192.168.1.8'
    calibration={'communication':{'robot_ip':'192.168.1.18'}}
    assert resolve_preview_robot_ip({},None,calibration=calibration)=='192.168.1.18'
    assert resolve_preview_robot_ip({},None,assume_standby=True) is None
    with pytest.raises(ValueError,match='离线'):
        resolve_preview_robot_ip({},None,saved_capture=True)


def test_failure_summary_distinguishes_initial_collision_from_ik_failure():
    from gene_agent.realworld.sweep.diagnostics import failure_summary
    r={'planning_complete':False,'pick':{'stages':[{'name':'open_assumed','success':False,
       'postcheck':{'forbidden_contacts':[{'roles':['table','brush'],'penetration_m':.0164}]}}]}}
    text=failure_summary(r)
    assert 'open_assumed' in text and '16.4 mm' in text and 'table' in text and 'brush' in text


def test_support_projection_is_bounded_atomic_and_keeps_xy_rotation():
    import mujoco
    from gene_agent.scene.tabletop_support import correct_tabletop_support
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="table" type="box" pos="0 0 -.02" size="1 1 .02"/>
      <body name="brush" pos=".3 .2 -.01"><freejoint/><geom type="box" size=".05 .02 .01" mass="1"/></body>
      <body name="cube" pos=".4 .2 .02"><freejoint/><geom type="box" size=".01 .01 .01" mass="1"/></body>
      </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d);before=d.qpos.copy()
    with pytest.raises(ValueError,match='上限'):
        correct_tabletop_support(m,d,['brush','cube'],'table',max_adjustment_m=.01)
    np.testing.assert_array_equal(d.qpos,before)
    report=correct_tabletop_support(m,d,['brush','cube'],'table',max_adjustment_m=.03)
    assert report['objects']['brush']['shift_world_z_m']==pytest.approx(.021)
    np.testing.assert_array_equal(d.qpos[:2],before[:2])
    np.testing.assert_array_equal(d.qpos[3:7],before[3:7])
    assert d.body('brush').xpos[2]==pytest.approx(.011)
    assert report['objects']['cube']['shift_world_z_m']==0


def test_support_correction_does_not_exempt_robot_table_collision():
    import mujoco
    from gene_agent.scene.tabletop_support import correct_tabletop_support
    from gene_agent.realworld.sweep.planning import check_world_contacts
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="scene_table" type="box" pos="0 0 -.02" size="1 1 .02"/>
      <body name="link_1" pos="0 0 0"><joint name="q"/><geom type="sphere" size=".03" mass="1"/></body>
      <body name="brush" pos=".4 0 -.01"><freejoint/><geom type="box" size=".04 .02 .01" mass="1"/></body>
      </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    correct_tabletop_support(m,d,['brush'],'scene_table')
    report=check_world_contacts(m,d,{'joint_groups':{'left':['q']}},[[0.]],None,(('brush','table'),))
    assert not report['success']
    assert any('link_1' in c['roles'] for c in report['forbidden_contacts'])
