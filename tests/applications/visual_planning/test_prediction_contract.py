"""纯预测的状态/证据契约；规划器均为替身，不属于 GPU 成功验收。"""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import gene_agent.tasks.planning.runner as runner
from gene_agent.tasks.planning.knowledge import proposals

ROOT=Path(__file__).resolve().parents[3]
SOURCES={'sweep':ROOT/'data/realworld/brush_cube_dustpan_0',
         'microwave':ROOT/'data/task_1/simulation/mujoco_task1_physical.json'}


def make_run(tmp_path,task='microwave',**kwargs):
    return runner.VisualPlanningRun(task,SOURCES[task],tmp_path/'run',**kwargs)


def test_runner_and_viewer_import_without_physics():
    code='''
import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='mujoco' or fullname.startswith('mujoco.'):
            raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
import gene_agent.tasks.planning.runner
import gene_agent.visualization.pure_planning.app
assert 'mujoco' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_state_snapshot_does_not_change_with_later_joints(tmp_path):
    run=make_run(tmp_path)
    previous=run.state();saved=copy.deepcopy(previous)
    run.inputs.set_object_joints('microwave',{'door_hinge':.7,'power_knob_rotation':.3})
    run.q['left'][0]+=.1
    run.width['right']=.04
    pose=run.inputs.objects['bowl']['T_world_object'].copy();pose[0,3]+=.02
    run.inputs.objects['bowl']['T_world_object']=pose
    assert previous==saved


def _lease():
    return SimpleNamespace(planner=object(),cache_hit=True,setup_duration_s=.001,rebuild_reason='cache_hit')


def test_failed_plan_keeps_joint_and_attachment_state(tmp_path,monkeypatch):
    run=make_run(tmp_path)
    run.event('assumed_hold',group='right',attach='bowl')
    before=run.state()
    monkeypatch.setattr(run,'planner',lambda group:_lease())
    monkeypatch.setattr(runner,'plan_with_approximate_ik',lambda *a,**k:
                        {'success':False,'failure_category':'goal_ik_no_solution','raw_test_evidence':'mock'})
    outcome=run.move('approach',np.eye(4),'linear_tcp',group='right')
    assert not outcome['success']
    assert run.state()==before
    row=run.report['stages'][-1]
    assert row['start_state']==row['end_state']
    assert run.failed=='approach'
    assert json.loads((run.output/row['plan_file']).read_text())['raw_test_evidence']=='mock'


def test_approximate_result_preserves_requested_and_fk(tmp_path,monkeypatch):
    run=make_run(tmp_path)
    start=run.q['right'].copy();end=start.copy();end[1]+=.001
    actual=run.inputs.robots['right'].fk(end)
    requested=actual.copy();requested[0,3]+=.008
    approximation={'approximate':True,'strict_ik_success':False,'constraint_feasible':True,
        'requested_tcp_pose_world':requested.tolist(),'actual_endpoint_tcp_pose_world':actual.tolist(),
        'budget':{'max_position_error_m':.025,'max_rotation_error_rad':.065}}
    raw={'success':True,'positions_rad':[start.tolist(),end.tolist()],
         'interpolation_dt_s':.02,'approximate_ik':approximation,'raw_test_evidence':'mock'}
    received={}
    def plan(*args,**kwargs):
        received.update(kwargs)
        np.testing.assert_array_equal(args[3],requested)
        return copy.deepcopy(raw)
    monkeypatch.setattr(run,'planner',lambda group:_lease())
    monkeypatch.setattr(runner,'plan_with_approximate_ik',plan)
    assert run.move('pregrasp',requested,group='right')['success']
    row=run.report['stages'][-1]
    np.testing.assert_array_equal(row['target_tcp_world'],requested)
    np.testing.assert_allclose(row['actual_tcp_world'],actual)
    assert row['position_error_m']==pytest.approx(.008)
    assert row['approximation']==approximation
    assert received['budget'].max_position_error_m==.025
    assert json.loads((run.output/row['plan_file']).read_text())['approximate_ik']==approximation


def test_continuous_extract_does_not_reset_prior_bowl_pose(tmp_path,monkeypatch):
    run=make_run(tmp_path)
    carried=run.inputs.objects['bowl']['T_world_object'].copy();carried[0,3]+=.013
    run.inputs.objects['bowl']['T_world_object']=carried.copy()
    observed=[]
    def first_move(*args,**kwargs):
        observed.append(run.inputs.objects['bowl']['T_world_object'].copy())
        return {'success':False}
    monkeypatch.setattr(run,'move',first_move)
    assert not run.transfer(extract=True)
    np.testing.assert_array_equal(observed[0],carried)
    np.testing.assert_array_equal(run.inputs.objects['bowl']['T_world_object'],carried)
    assert not run.report['initial_assumptions']


def test_complete_pick_to_sweep_uses_table_contact_height(tmp_path,monkeypatch):
    """理想位姿执行替身用于隔离阶段语义，不声称目标 IK 可达。"""
    run=make_run(tmp_path,'sweep')
    robot=run.inputs.robots['left']
    tcp=[robot.fk(run.q['left'])]
    monkeypatch.setattr(robot,'fk',lambda q:tcp[0].copy())
    monkeypatch.setattr(run,'choose',lambda kind,**kw:proposals(run.inputs,kind,tilt=run.tilt)[0])
    monkeypatch.setattr(run,'planner',lambda group:SimpleNamespace(planner=SimpleNamespace(
        solve_ik=lambda *a,**k:{'success':True})))
    contacts=[]
    def ideal_move(name,target,kind='joint',*,group,**kwargs):
        tcp[0]=np.asarray(target).copy()
        run.event(name,group=group)
        if name=='lower_to_contact':contacts.append(run.inputs.objects['brush']['T_world_object'].copy())
        return {'success':True}
    monkeypatch.setattr(run,'move',ideal_move)
    initial=run.inputs.objects['brush']['T_world_object'].copy()
    cubes={n:r['T_world_object'].copy() for n,r in run.inputs.objects.items() if n.startswith('cube')}
    assert run.pick('brush')
    assert run.inputs.objects['brush']['T_world_object'][2,3]>initial[2,3]+.03
    assert run.sweep()
    assert contacts
    # 当前资产名义接触高度是 z=0；不能携带抬升后的 45 mm 高度继续清扫。
    assert all(abs(p[2,3]-initial[2,3])<.02 for p in contacts)
    for name,pose in cubes.items():np.testing.assert_array_equal(run.inputs.objects[name]['T_world_object'],pose)


def test_handle_candidates_follow_door_frame_once(tmp_path):
    from gene_agent.knowledge.geometry import load_concept
    from gene_agent.knowledge.grasp import load_grasp_profiles,generate_grasps
    run=make_run(tmp_path)
    closed=proposals(run.inputs,'handle_grasp')
    profile=next(iter(load_grasp_profiles(ROOT/'data/task_1/knowledge/grasp_profiles.microwave_door.json').values()))
    part=load_concept(ROOT/'assets/objects/microwave_calibrated/handle_concept.json',part_name=profile.part_name).concept
    expected_closed=generate_grasps(part,run.inputs.objects['microwave']['T_world_object'],profile).proposals
    for row,expected in zip(closed,expected_closed):
        np.testing.assert_allclose(row['pose'],expected.tcp_pose_world,atol=1e-9)
    urdf=run.inputs.object_urdfs['microwave'][0]
    joint=urdf.joint_map['door_hinge']
    world_joint=run.inputs.objects['microwave']['T_world_object']@joint.origin
    axis=world_joint[:3,:3]@np.asarray(joint.axis);point=world_joint[:3,3]
    angle=.7;run.inputs.set_object_joints('microwave',{'door_hinge':angle})
    opened=proposals(run.inputs,'handle_grasp')
    rotation=Rotation.from_rotvec(axis*angle).as_matrix()
    for first,last in zip(closed,opened):
        expected=np.asarray(first['pose']).copy()
        expected[:3,:3]=rotation@expected[:3,:3]
        expected[:3,3]=point+rotation@(expected[:3,3]-point)
        np.testing.assert_allclose(last['pose'],expected,atol=1e-9)


def test_knob_targets_rotate_about_declared_urdf_axis(tmp_path,monkeypatch):
    run=make_run(tmp_path)
    chosen=proposals(run.inputs,'knob_grasp')[0]
    monkeypatch.setattr(run,'choose',lambda *a,**kw:chosen)
    moves={}
    def record(name,target,kind='joint',*,group,**kwargs):
        moves[name]=np.asarray(target).copy()
        run.event(name,group=group)
        return {'success':True}
    monkeypatch.setattr(run,'move',record)
    assert run.knob()
    joint=run.inputs.object_urdfs['microwave'][0].joint_map['power_knob_rotation']
    origin=run.inputs.objects['microwave']['T_world_object']@joint.origin
    axis=origin[:3,:3]@np.asarray(joint.axis)
    rotation=Rotation.from_rotvec(axis*np.deg2rad(45.)).as_matrix()
    expected=chosen['pose'].copy()
    expected[:3,:3]=rotation@expected[:3,:3]
    expected[:3,3]=origin[:3,3]+rotation@(expected[:3,3]-origin[:3,3])
    np.testing.assert_allclose(moves['rotate_45'],expected,atol=1e-9)
    assert run.inputs.object_joint_positions['microwave']['power_knob_rotation']==pytest.approx(np.deg2rad(45.))


class _Control(SimpleNamespace):
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def remove(self): pass
    def on_update(self, callback): self.callback=callback
    def on_click(self, callback): self.callback=callback


class _Gui:
    def configure_theme(self,**kwargs): pass
    def add_markdown(self,content): return _Control(content=content)
    def add_folder(self,*a,**kw):return _Control()
    def add_checkbox(self,name,initial_value):return _Control(value=initial_value)
    def add_dropdown(self,name,options,**kwargs): return _Control(value=kwargs.get('initial_value',options[0]),options=options)
    def add_slider(self,name,**kwargs): return _Control(value=kwargs['initial_value'])
    def add_button(self,name): return _Control()


class _Scene:
    def set_up_direction(self,*args):pass
    def add_grid(self,*args,**kwargs): pass
    def add_frame(self,*args,**kwargs): return _Control(**kwargs)


class _Server:
    def __init__(self): self.gui=_Gui();self.scene=_Scene();self.initial_camera=_Control()
    def get_clients(self):return {}
    def on_client_connect(self,callback): self.callback=callback


def test_viewer_uses_frozen_input_and_recorded_end_rotation(tmp_path,monkeypatch):
    from gene_agent.visualization.pure_planning import app
    from gene_agent.scene import placement,tabletop_randomization
    run=make_run(tmp_path,'sweep',seed=36,randomize=True)
    before=run.state();after=copy.deepcopy(before)
    rotated=np.asarray(after['objects']['brush'])
    rotated[:3,:3]=Rotation.from_euler('z',.4).as_matrix()@rotated[:3,:3]
    rotated[0,3]+=.03;after['objects']['brush']=rotated.tolist()
    run.report.update(planning_success=True,stages=[dict(name='explicit_initial_state',path_kind='prediction',
        group='left',success=True,start_state=before,end_state=after)])
    runner.write(run.output/'result.json',run.report)
    def cannot_sample(*args,**kwargs):raise AssertionError('回放不应重新采样')
    monkeypatch.setattr(placement,'sample_placement',cannot_sample)
    monkeypatch.setattr(tabletop_randomization,'sample_tabletop_layout',cannot_sample)
    monkeypatch.setattr(app,'ViserUrdf',lambda *a,**kw:SimpleNamespace(update_cfg=lambda cfg:None))
    monkeypatch.setattr(app.PurePlanningApp,'add_geometry',lambda *a,**kw:_Control())
    viewer=app.PurePlanningApp(_Server(),run=run.output,output=tmp_path/'viewer')
    assert viewer.inputs.placement==run.inputs.placement
    np.testing.assert_array_equal(viewer.inputs.objects['brush']['T_world_object'],before['objects']['brush'])
    viewer.position.value=1.;viewer.update()
    np.testing.assert_allclose(viewer.inputs.objects['brush']['T_world_object'],rotated,atol=1e-12)
    # 回放不得移动任何未在记录中变化的方块。
    for name in ('cube','cube_red','cube_blue'):
        np.testing.assert_allclose(viewer.inputs.objects[name]['T_world_object'],before['objects'][name],atol=1e-12)


@pytest.mark.parametrize('selection,joint,target',[('knob_turn','power_knob_rotation',.45),('handle_pull','door_hinge',.43633)])
def test_knowledge_rebuild_uses_original_joints_after_viewing_endpoint(tmp_path,selection,joint,target):
    from threading import RLock
    from gene_agent.visualization.pure_planning.app import PurePlanningApp
    from gene_agent.scene.visual_inputs import load_visual_inputs
    scene=load_visual_inputs('microwave',SOURCES['microwave'],tmp_path/'inputs')
    initial=copy.deepcopy(scene.object_joint_positions)
    scene.set_object_joints('microwave',{joint:target})
    viewer=PurePlanningApp.__new__(PurePlanningApp)
    viewer._lock=RLock();viewer.inputs=scene;viewer.knowledge_selection=selection
    viewer.knowledge_initial_joints=initial;viewer.tune={'tilt':SimpleNamespace(value=-20.),'target_angle_rad':SimpleNamespace(value=target)}
    viewer.overlay=[];viewer.choice=SimpleNamespace();viewer.show_path=SimpleNamespace(value=True)
    viewer.server=SimpleNamespace(scene=SimpleNamespace(add_point_cloud=lambda *a,**kw:_Control(),add_line_segments=lambda *a,**kw:_Control()))
    viewer.output=tmp_path/'view';viewer.select_knowledge=lambda:None
    viewer.rebuild_knowledge()
    assert len(viewer.rows)>2
    assert viewer.rows[-1]['angle_rad']==pytest.approx(target)
    assert scene.object_joint_positions['microwave'][joint]==pytest.approx(target)
