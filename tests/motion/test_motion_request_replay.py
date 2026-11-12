"""冻结规划使用同一adapter，不执行仿真；缺现场或资产变更必须拒绝。"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pytest
from gene_agent.motion import JointState,PoseTarget
from gene_agent.motion.model import MotionRequest,MotionOperation,PlanningSceneRef
from gene_agent.motion.curobo.diagnostics import digest


def bundle(tmp_path):
    import curobo
    from gene_agent.motion.model import PlannerMode
    scene={'cuboid':{}}
    request=MotionRequest('native/test',MotionOperation.PLAN,'right',JointState(('q',),(0.,)),PoseTarget('world_tcp',np.eye(4)),PlanningSceneRef(0,'s','test'),seed=123)
    replay=dict(joint_names=['q'],planner_joint_names=['q'],robot_config={},world_from_base=np.eye(4).tolist(),link_from_tcp=np.eye(4).tolist(),tool_frame='tool',scene_model=scene,scene_scope='test',scene_snapshot_sha256='s',planner_mode='cuda_graph_benchmark',random_seed=123)
    snapshot=dict(scene_model=scene,scope='test',snapshot_sha256='s')
    row=dict(replay_config=replay,replay_config_sha256=digest(replay),request=request.to_dict(),request_sha256=digest(request.to_dict()),snapshot=snapshot,snapshot_payload_sha256=digest(snapshot),environment=dict(curobo_module=curobo.__file__,python=sys.version,executable=sys.executable),asset_hashes={curobo.__file__:hashlib.sha256(Path(curobo.__file__).read_bytes()).hexdigest()})
    path=tmp_path/'request.json';path.write_text(json.dumps(row));return path,row


def test_replanning_uses_formal_adapter_and_retains_cold_warm_results(tmp_path,monkeypatch):
    from gene_agent.motion.curobo.diagnostics import replan_bundle
    from gene_agent.motion.curobo import core
    from gene_agent.motion.model import PlannerMode
    path,row=bundle(tmp_path);calls=[]
    class Planner:
        def __init__(self,**config):
            for k,v in config.items():setattr(self,k,v)
            self.planner_mode=PlannerMode(self.planner_mode)
            self.diagnostic_options=None
        def plan(self,names,q,target,**kwargs):
            calls.append(np.array(target));return dict(success=True,duration_s=.1,positions_rad=[[.2]],interpolation_dt_s=.025)
        def reset_seed(self):pass
        def close(self):calls.append('closed')
    monkeypatch.setattr(core,'SynchronousPlanner',Planner)
    out=replan_bundle(path,tmp_path/'out',warm_repeats=1)
    report=json.loads(out.read_text())
    assert report['mode']=='offline_motion_replan_no_execution'
    assert [r['phase'] for r in report['attempts']]==['cold','warm']
    assert all(r['results'][0]['success'] for r in report['attempts'])
    assert len(calls)==3 and calls[-1]=='closed'
    with pytest.raises(FileExistsError):replan_bundle(path,tmp_path/'out',warm_repeats=0)


def test_replanning_refuses_changed_asset_before_creating_output(tmp_path):
    from gene_agent.motion.curobo.diagnostics import replan_bundle
    path,row=bundle(tmp_path);asset=tmp_path/'mesh.obj';asset.write_text('old')
    row['asset_hashes'][str(asset)]=hashlib.sha256(asset.read_bytes()).hexdigest();path.write_text(json.dumps(row));asset.write_text('changed')
    with pytest.raises(ValueError,match='hash'):replan_bundle(path,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_code_change_requires_explicit_flag_and_is_reported(tmp_path):
    from gene_agent.feedback.motion_diagnostics import validate_diagnostic_bundle
    path,row=bundle(tmp_path);code=tmp_path/'adapter.py';code.write_text('old')
    row['code_hashes']={str(code):hashlib.sha256(code.read_bytes()).hexdigest()};code.write_text('new')
    with pytest.raises(ValueError,match='代码'):validate_diagnostic_bundle(row)
    changes=validate_diagnostic_bundle(row,allow_code_change=True)
    assert changes and changes[0]['path']==str(code)


def test_original_batch_missing_targets_cannot_be_replanned_as_single_ik(tmp_path):
    from gene_agent.motion.curobo.diagnostics import replan_bundle
    path,row=bundle(tmp_path);row['source_batch_size']=4;path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='batch'):replan_bundle(path,tmp_path/'out')


def test_robot_only_capture_retains_protection_scene_and_scope(tmp_path):
    from types import SimpleNamespace
    from gene_agent.motion.configuration import CollisionMode
    from gene_agent.motion.model import PlannerMode
    from gene_agent.motion.curobo.diagnostics import capture_request_context
    path,row=bundle(tmp_path)
    protection={'cuboid':{'other_arm':{'dims':[.1,.1,.1],'pose':[0,0,0,1,0,0,0]}}}
    config=row['replay_config']
    planner=SimpleNamespace(**config,collision_mode=CollisionMode.ROBOT_ONLY,_source_robot_config={},
        _gripper_snapshot=None,held_object=None,attachment_spheres_tool=None,mounting_contact_policy={})
    planner.robot_config={};planner.planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK
    planner.world_from_base=planner.link_from_tcp=np.eye(4)
    planner.scene_model=protection
    request=__import__('gene_agent.motion.curobo.diagnostics',fromlist=['request_from_dict']).request_from_dict(row['request'])
    captured=capture_request_context(planner,request,dict(row['snapshot'],scene_model=protection))
    assert captured['replay_config']['robot_protection_scene']==protection
    assert captured['replay_config']['scene_scope']==request.scene.scope


def test_dependency_implementation_change_rejected_even_with_code_flag(tmp_path):
    from gene_agent.feedback.motion_diagnostics import validate_diagnostic_bundle
    from gene_agent.motion.curobo.diagnostics import dependency_signature
    path,row=bundle(tmp_path)
    dep=tmp_path/'dependency';dep.mkdir();source=dep/'motion_planner.py';source.write_text('old')
    row.update(dependency_root=str(dep),dependency_hashes=dependency_signature(str(dep)))
    source.write_text('new')
    with pytest.raises(ValueError,match='依赖'):validate_diagnostic_bundle(row,allow_code_change=True)


def test_one_target_batch_replay_retains_batch_api(tmp_path,monkeypatch):
    from gene_agent.motion.curobo.diagnostics import replan_bundle
    from gene_agent.motion.curobo import core
    from gene_agent.motion.model import PlannerMode
    path,row=bundle(tmp_path);row['request']['operation']='ik';row['request_sha256']=digest(row['request'])
    row.update(requests=[row['request']],requests_sha256=digest([row['request']]),source_batch_size=1,invocation='evaluate_candidates')
    path.write_text(json.dumps(row));calls=[]
    class Planner:
        def __init__(self,**config):
            for k,v in config.items():setattr(self,k,v)
            self.planner_mode=PlannerMode(self.planner_mode)
        def solve_ik_batch(self,names,q,targets,**kwargs):
            calls.append(len(targets));return [dict(success=True,duration_s=.1,positions_rad=[[.2]])]
        def reset_seed(self):pass
        def close(self):pass
    monkeypatch.setattr(core,'SynchronousPlanner',Planner)
    replan_bundle(path,tmp_path/'out',warm_repeats=0)
    assert calls==[1]
