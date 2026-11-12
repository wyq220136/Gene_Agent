import json,time
import numpy as np
import pytest


def test_pika_uses_millimeters_and_rejects_stale_state():
    from gene_agent.realworld.devices.grippers.pika.adapter import PikaGripperAdapter
    class SDK:
        is_connected=True
        last_motor_update=time.monotonic()
        def get_gripper_distance(self):return 40.
        def set_gripper_distance(self,target):self.target=target;return True
        def disconnect(self):self.is_connected=False
    sdk=SDK();g=PikaGripperAdapter(sdk)
    assert g.read_width_m()==pytest.approx(.04)
    g.command_width_m(.03,blocking=False)
    assert sdk.target==pytest.approx(30.)
    sdk.last_motor_update-=10
    with pytest.raises(RuntimeError,match='过期'):g.read_width_m()


def test_execution_uses_live_input_and_freshness_without_manual_tcp_flags():
    from gene_agent.realworld.sweep.execution import check_execution_input
    snapshot={'source':'first_frame_preview','input_mode':'new_camera_frame',
        'robot_state_source':'sdk_read_only','observed_at_unix_ms':time.time()*1000}
    site={'frame_profile':'config/robots/pika_frames.rm75_6fb.json'}
    config={'tcp_verified':False,'tcp_verification_source':'','max_observation_age_s':180}
    # 旧配置的false/空来源不再是手动放行门槛。
    check_execution_input(snapshot,site,config)
    snapshot['observed_at_unix_ms']-=200000
    with pytest.raises(ValueError,match='过期'):check_execution_input(snapshot,site,config)


def test_default_cli_does_not_create_sdk_bridge(monkeypatch,tmp_path):
    from tools.run.execute_sweep_realworld import main
    from gene_agent.realworld.sweep.sdk_bridge import SDKBridge
    def forbidden(*args,**kwargs):raise AssertionError('default must not touch hardware')
    monkeypatch.setattr(SDKBridge,'__enter__',forbidden)
    run=tmp_path/'preview';run.mkdir()
    (run/'result.json').write_text(json.dumps({'task':'sweep_realworld_plan_only','planning_complete':True,'brush_variant':'old'}))
    (run/'site_input.json').write_text('{}');(run/'observation.json').write_text('{}')
    assert main(['--run',str(run),'--output',str(tmp_path/'out')])==0


def execution_fixture(tmp_path,monkeypatch):
    import mujoco
    from types import SimpleNamespace
    from pathlib import Path
    from gene_agent.realworld.sweep import execution as module
    from gene_agent.robots.pika_frames import pose_from_mujoco
    root=Path(__file__).resolve().parents[4]
    directory=root/'data/realworld/brush_cube_dustpan_0'
    m=mujoco.MjModel.from_xml_path(str(directory/'scene.xml'));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
    robot=json.loads((directory/'robot/robot_metadata.json').read_text())
    q=[float(d.qpos[m.joint(n).qposadr[0]]) for n in robot['joint_groups']['left']]
    class Backend:
        def __init__(self):self.q=list(q);self.commands=[];self.width=.04
        def call(self,cmd,payload=None):
            self.commands.append(cmd)
            if cmd=='state':return {'joint_positions_rad':list(self.q),'gripper_width_m':self.width}
            if cmd=='motion':
                self.q[0]+=.001
                return {'success':True,'status':'executed'}
            if cmd=='gripper':return {'success':True,'status':'executed'}
    backend=Backend();starts=[]
    def plan(model,data,robot,pool,path,stage,target,*args,**kwargs):
        starts.append(float(data.qpos[model.joint('joint_1').qposadr[0]]))
        data.qpos[model.joint('joint_1').qposadr[0]]=1.5 # 不可拿该预测值交接
        return {'name':stage,'success':True},[]
    monkeypatch.setattr(module,'check_world_contacts',lambda *a,**kw:{'success':True})
    monkeypatch.setattr(module,'pose_error',lambda *a,**kw:{'translation_m':0.,'rotation_deg':0.})
    import gene_agent.motion.curobo.scene as scene
    monkeypatch.setattr(scene,'export_collision_scene',lambda *a,**kw:{'snapshot_sha256':'fixture'})
    site={'execution_scene_xml':str(directory/'scene.xml'),'execution_concept_dir':str(directory/'concepts'),
          'execution_robot_ip':'fixture'}
    config={'tcp_verified':True,'tcp_verification_source':'unit_fixture','max_observation_age_s':180,'max_start_error_rad':.02,'max_width_error_m':.005,'max_endpoint_translation_m':.005,'max_endpoint_rotation_deg':3}
    kwargs={'stage_planner':plan,'pool_factory':lambda **kw:SimpleNamespace(clear=lambda:None)}
    return module,m,d,robot,site,{'joint_positions_rad':q,'source':'first_frame_preview','input_mode':'new_camera_frame','robot_state_source':'sdk_read_only','observed_at_unix_ms':time.time()*1000},config,backend,starts,kwargs


def test_unconfirmed_grasp_does_not_lift_and_uses_measured_state(tmp_path,monkeypatch):
    module,m,d,robot,site,snapshot,config,backend,starts,kwargs=execution_fixture(tmp_path,monkeypatch)
    r=module.execute_observed_sweep(m,d,robot,site,snapshot,config,backend,tmp_path/'out',
        stop_after='task',ask=lambda q:'n',**kwargs)
    assert r['failure_stage']=='grasp_not_confirmed'
    assert [s['name'] for s in r['stages']]==['open','pregrasp','approach','close']
    assert starts[1]==pytest.approx(starts[0]+.001) # actual end, not predicted1.5


def test_motion_failure_stops_before_gripper_close(tmp_path,monkeypatch):
    module,m,d,robot,site,snapshot,config,backend,starts,kwargs=execution_fixture(tmp_path,monkeypatch)
    original=backend.call
    def fail(cmd,payload=None):
        if cmd=='motion':return {'success':False,'status':'failed'}
        return original(cmd,payload)
    backend.call=fail
    r=module.execute_observed_sweep(m,d,robot,site,snapshot,config,backend,tmp_path/'out',stop_after='task',**kwargs)
    assert r['failure_stage']=='pregrasp'
    assert [s['name'] for s in r['stages']]==['open','pregrasp']


def test_confirmation_wait_cannot_bypass_frame_age(tmp_path,monkeypatch):
    from pathlib import Path
    from tools.run.execute_sweep_realworld import main
    from gene_agent.realworld.sweep.sdk_bridge import SDKBridge
    root=Path(__file__).resolve().parents[4]
    config=json.loads((root/'config/realworld/sweep_execution.example.json').read_text())
    config.update(tcp_verified=True,tcp_verification_source='unit fixture')
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config))
    run=tmp_path/'run';run.mkdir()
    now=time.time()
    (run/'result.json').write_text(json.dumps({'task':'sweep_realworld_plan_only','planning_complete':True,'brush_variant':'old'}))
    (run/'site_input.json').write_text('{}')
    (run/'observation.json').write_text(json.dumps({'source':'first_frame_preview','input_mode':'new_camera_frame',
        'robot_state_source':'sdk_read_only','observed_at_unix_ms':now*1000}))
    def answer(_):
        monkeypatch.setattr(time,'time',lambda:now+400)
        return 'y'
    monkeypatch.setattr('builtins.input',answer)
    monkeypatch.setattr(SDKBridge,'__enter__',lambda *a:pytest.fail('must reject before enabling hardware'))
    with pytest.raises(ValueError,match='过期'):
        main(['--run',str(run),'--config',str(cfg),'--execute','--output',str(tmp_path/'out')])


def test_width_change_after_grasp_stops_attachment_chain(tmp_path,monkeypatch):
    module,m,d,robot,site,snapshot,config,backend,starts,kwargs=execution_fixture(tmp_path,monkeypatch)
    original=backend.call
    def change(cmd,payload=None):
        result=original(cmd,payload)
        if cmd=='motion' and len(starts)==3:backend.width+=.01
        return result
    backend.call=change
    with pytest.raises(ValueError,match='夹爪开口'):
        module.execute_observed_sweep(m,d,robot,site,snapshot,config,backend,tmp_path/'out',
            stop_after='task',ask=lambda q:'y',**kwargs)
    assert len(starts)==3


def test_pregrasp_is_not_blocked_by_later_sweep_failure():
    from gene_agent.realworld.sweep.execution import preview_ready_for_stop
    r={'planning_complete':False,'pick':{'stages':[
        {'name':'open_assumed','success':True},{'name':'pregrasp','success':True},
        {'name':'approach','success':True},{'name':'close_assumed','success':True},
        {'name':'lift','success':True}]}}
    assert preview_ready_for_stop(r,'pregrasp')
    assert preview_ready_for_stop(r,'approach')
    assert preview_ready_for_stop(r,'pick')
    assert not preview_ready_for_stop(r,'task')
    r['pick']['stages'][1]['success']=False
    assert not preview_ready_for_stop(r,'pregrasp')


@pytest.mark.parametrize('stop_after,expected',[('pregrasp',['pregrasp']),
    ('approach',['pregrasp','approach']),('pick',['pregrasp','approach','lift'])])
def test_table_preview_stops_at_requested_prefix(tmp_path,monkeypatch,stop_after,expected):
    import mujoco,shutil
    from pathlib import Path
    from types import SimpleNamespace
    from gene_agent.realworld.sweep import table_preview as module
    root=Path(__file__).resolve().parents[4];source=root/'data/realworld/brush_cube_dustpan_0'
    m=mujoco.MjModel.from_xml_path(str(source/'scene.xml'));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
    robot=json.loads((source/'robot/robot_metadata.json').read_text())
    shutil.copytree(source/'concepts',tmp_path/'scene/concepts')
    calls=[]
    def plan(*args,**kwargs):
        stage=args[5];calls.append(stage)
        return {'name':stage,'success':True},[{'stage':stage,'qpos':d.qpos.tolist(),'predicted':True}]
    monkeypatch.setattr(module,'plan_stage',plan)
    monkeypatch.setattr(module,'CuroboPlannerPool',lambda **kw:SimpleNamespace(clear=lambda:None))
    monkeypatch.setattr(module,'check_world_contacts',lambda *a,**kw:{'success':True})
    monkeypatch.setattr(module,'plan_sweep',lambda *a,**kw:pytest.fail('must not plan beyond requested prefix'))
    gap_calls=[]
    def predict_gap(*args,**kw):
        gap_calls.append((args,kw))
        return {'predicted_width_m':.025,'physical_hold_verified':False}
    monkeypatch.setattr(module,'pika_predict_contact_width',predict_gap)
    r=module.preview_table_task(m,d,robot,{'snapshot_source':'fixture','robot_state_source':'fixture'},
                                tmp_path,stop_after=stop_after)
    assert calls==expected
    assert r['requested_scope_complete']
    assert not r['planning_complete'] # 不冒称完整任务已规划成功
    assert len(gap_calls)==(1 if stop_after=='pick' else 0)
    if stop_after=='pick':
        close=next(s for s in r['pick']['stages'] if s['name']=='close_assumed')
        assert close['width_m']==.025
        assert not r['pick']['grasp_geometry']['physical_hold_verified']


def test_camera_execution_entry_forwards_pregrasp_scope(tmp_path,monkeypatch):
    from pathlib import Path
    from tools.run.execute_sweep_realworld import main
    import tools.run.preview_sweep_from_camera as camera
    def preview(argv):
        assert argv[argv.index('--stop-after')+1]=='pregrasp'
        out=Path(argv[argv.index('--output')+1]);out.mkdir()
        (out/'result.json').write_text(json.dumps({'task':'sweep_realworld_plan_only','brush_variant':'old',
            'planning_complete':False,'pick':{'stages':[{'name':'open_assumed','success':True},
                {'name':'pregrasp','success':True}]}}))
        (out/'observation.json').write_text('{}');(out/'site_input.json').write_text('{}')
        return 0
    monkeypatch.setattr(camera,'main',preview)
    assert main(['--camera-param',str(tmp_path/'camera.json'),'--stop-after','pregrasp',
                 '--output',str(tmp_path/'out')])==0
