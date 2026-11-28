"""固定输入下比对迁移前后的策略；只运行几何与mock，不推进物理或调用GPU。"""
import ast
import os
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from gene_agent.motion.model import PoseTarget
from gene_agent.realworld.microwave import strategy_planning as planning
from gene_agent.tasks.policies.microwave_task.operations import transfer_scene as geometry
from gene_agent.tasks.policies.microwave_task.operations.transfer_scene import actual_bowl_extract_targets
from gene_agent.realworld.microwave.scene import build_observed_collision_scene

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(os.environ.get("GENE_AGENT_NEW_REFERENCE", str(ROOT.parent/"Gene_Agent_new")))

@pytest.fixture
def scene(tmp_path):
    from test_scene_shadow import _fixture
    config, observed, table, joints, support = _fixture()
    return build_observed_collision_scene(
        observed, T_world_sdk=np.eye(4),
        T_perception_object_scene_asset={n:np.eye(4) for n in observed.object_poses_sdk},
        table_pose_world=table, table_size_xy_m=config["table"]["size_xy_m"],
        accepted_support_size_xyz_m=support,
        accepted_support_center_xy_local_m=config["microwave_support"]["center_xy_local_m"],
        accepted_table_thickness_m=config["table"]["thickness_m"],
        joint_positions_by_group=joints, gripper_widths_m={"left":.08,"right":.04},
        door_angle_rad=np.pi/2, knob_angle_rad=.1,
        turntable_height_offset_m=0., microwave_floor_height_offset_m=0.,
        active_group="right", output_dir=tmp_path/"initial")

def reference_function(file, name, namespace):
    if not REFERENCE.exists():
        pytest.skip("设置GENE_AGENT_NEW_REFERENCE以运行独立参考比对")
    text = (REFERENCE/file).read_text()
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name==name)
    code = "\n".join(text.splitlines()[node.lineno-1:node.end_lineno])
    exec(compile(code, str(REFERENCE/file), "exec"), namespace)
    return namespace[name]

def compare(left, right):
    if isinstance(left, dict):
        assert left.keys()==right.keys()
        for key in left:
            compare(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left)==len(right)
        for x,y in zip(left,right):
            compare(x,y)
    elif isinstance(left, np.ndarray):
        np.testing.assert_allclose(left,right,atol=1e-12,rtol=0)
    elif isinstance(left, (float, np.floating)):
        assert left==pytest.approx(right,abs=1e-12)
    else:
        assert left==right

@pytest.mark.parametrize("yaw", [-120., 0., 77.])
def test_extract_targets_equal_reference_and_follow_object_translation(yaw):
    import gene_agent.knowledge as knowledge
    bowl=np.eye(4)
    bowl[:3,:3]=Rotation.from_euler("zx", [yaw,90],degrees=True).as_matrix()
    bowl[:3,3]=[.42,.35,.1]
    tcp=np.eye(4); tcp[:3,3]=[.3,.2,.2]
    pad=tcp[:3,3]+[.01,0,.05]
    namespace={"np":np,"bowl_grasp_candidates":knowledge.bowl_grasp_candidates,
               "bowl_azimuth_for_world_approach":knowledge.bowl_azimuth_for_world_approach}
    original=reference_function(
        "gene_agent/tasks/policies/microwave_task/operations/extract.py",
        "actual_bowl_extract_targets",namespace)
    actual=actual_bowl_extract_targets(bowl,tcp,pad,[1,0,0])
    compare(actual,original(bowl,tcp,pad,[1,0,0]))
    shifted=bowl.copy(); shifted[:3,3]+=[.01,-.02,.005]
    moved=actual_bowl_extract_targets(shifted,tcp,pad,[1,0,0])[0]
    for name in moved:
        np.testing.assert_allclose(moved[name][:3,3]-actual[0][name][:3,3],
                                  [.01,-.02,.005],atol=1e-12)

def test_insert_candidate_order_and_targets_equal_reference(scene):
    if not REFERENCE.exists():
        pytest.skip("缺少独立参考目录")
    frozen=planning.frozen_geometry(scene)
    from gene_agent.realworld.scene import observed_tcp_pose
    bowl=np.asarray(scene["provenance"]["aligned_object_poses_world"]["Bowl/Bowl_0"])
    tcp_bowl=np.linalg.inv(observed_tcp_pose(scene))@bowl
    path=REFERENCE/"gene_agent/tasks/policies/microwave_task/simulation_bowl/insert.py"
    source=path.read_text().splitlines()
    original_code=("\ndef original(runtime, world_bowl, tcp_bowl, *, "
        "insert_depth_offset_m=.02, insert_tcp_table_angle_max_deg=70., "
        "insert_max_bowl_tilt_deg=14., retreat_clearance_margin_m=.005,"
        "retreat_rotation_local_deg=(0.,0.,0.)):\n"
        +"\n".join(source[831:975])+"\n    return candidate_families\n")
    namespace=dict(vars(geometry))
    exec(compile(original_code,str(path),"exec"),namespace)
    original=namespace["original"](frozen,bowl,tcp_bowl)
    actual=geometry.insert_candidate_families(frozen,bowl,tcp_bowl)
    compare(actual,original)
    assert len(actual)>=3
    assert all("held_turn" not in f for f in actual[:len(actual)//3])

def test_table_target_equal_reference(scene):
    if not REFERENCE.exists():
        pytest.skip("缺少独立参考目录")
    frozen=planning.frozen_geometry(scene)
    from gene_agent.realworld.scene import observed_tcp_pose
    bowl=np.asarray(scene["provenance"]["aligned_object_poses_world"]["Bowl/Bowl_0"])
    tcp=observed_tcp_pose(scene)
    config=REFERENCE/"data/task_1/simulation/physical_table_place.json"
    options=json.loads(config.read_text())["object_goal"]
    namespace={"np":np,"json":json,"mujoco":__import__("mujoco"),
        "PoseTarget":PoseTarget,"PHYSICAL_PLACE_CONFIG":config}
    original=reference_function(
        "gene_agent/tasks/policies/microwave_task/simulation_bowl/place.py",
        "_object_relative_table_target",namespace)
    profile={"microwave_from_placed_object":np.eye(4)}
    old=SimpleNamespace(grasp=PoseTarget("world_tcp",np.eye(4)))
    expected=original(frozen,PoseTarget("world_tcp",np.eye(4)),old,np.eye(4),profile,bowl,tcp)
    actual=geometry.object_relative_table_target(frozen,options,bowl,tcp)
    np.testing.assert_allclose(actual[0],expected[0].matrix,atol=1e-12)
    np.testing.assert_allclose(actual[1],expected[1],atol=1e-12)
    for key in actual[2]:
        compare(actual[2][key],expected[2][key])
    goal=actual[0]@actual[1]
    np.testing.assert_allclose(goal[:3,1],[0,0,1],atol=1e-12)

def test_geometry_adapter_has_no_execution_or_step_interface(scene):
    frozen=planning.frozen_geometry(scene)
    for name in ("step","execute","attach_simulation_object"):
        assert not hasattr(frozen,name)
    original=frozen.data.qpos.copy()
    copy=frozen.copy_data();copy.qpos[:]=0.
    np.testing.assert_array_equal(frozen.data.qpos,original)
    assert scene["provenance"]["physics_stepped"] is False

def test_full_insert_screen_skips_release_rejected_and_stops_at_first_complete(scene,tmp_path,monkeypatch):
    calls=[];families=[{"targets":{n:np.eye(4) for n in ("preplace","insert","retreat_1","retreat_2")},
                      "object_goal":np.eye(4)} for _ in range(3)]
    trajectory=SimpleNamespace(to_dict=lambda:{"joint_names":["j"],"positions_rad":[[0.],[.1]]})
    stage=SimpleNamespace(result=SimpleNamespace(trajectory=trajectory))
    def sequence(current,sequence,**kwargs):
        names=[n for n,_ in sequence]
        calls.append(tuple(names))
        return {"success":True,"rows":[],"stages":{n:stage for n in names}}
    counter=[]
    def release(*args):
        counter.append(1)
        if len(counter)==1:
            return None,{"success":False,"reason_code":"release_envelope_forbidden_contact"}
        return SimpleNamespace(model=None,data=None,robot_metadata={}),{"success":True}
    monkeypatch.setattr(planning,"plan_sequence",sequence)
    monkeypatch.setattr(planning,"predict_released_insert_state",release)
    monkeypatch.setattr(planning,"predicted_scene",lambda *args: {
        **scene,"snapshot":dict(scene["snapshot"]),"provenance":dict(scene["provenance"])})
    monkeypatch.setattr(planning,"released_ceiling_clearance_snapshot",lambda x,*args:x)
    monkeypatch.setattr(planning,"raise_retreat_for_released_pad_clearance",
        lambda *args,**kw:({**families[0]["targets"],"retreat_lift":np.eye(4)},{"outward_fraction":kw["outward_fraction"]}))
    attachment={**scene,"spheres_tool_m":[[0.,0.,0.,.01]]}
    result=planning.screen_insert(scene,attachment,families,planner_pool=object(),output=tmp_path/"screen")
    assert result["success"] and result["index"]==1
    assert len(counter)==2
    assert calls==[("preplace","insert"),("preplace","insert"),
                   ("retreat_lift","retreat_1","retreat_2")]


def test_sdk_tcp_uses_same_sampled_joints_once_without_model_fk():
    from gene_agent.realworld.state_sampling import RobotStateSampler
    from gene_agent.motion.model import JointState
    calls=[]
    class Arm:
        def read_joint_state(self):
            calls.append("joints")
            return JointState(tuple("j"+str(i) for i in range(7)),tuple(.1 for _ in range(7)))
        def tcp_pose_from_sdk_fk(self,q):
            calls.append(("vendor_fk",tuple(q)))
            pose=np.eye(4);pose[:3,3]=[.1,.2,.3];return pose
    class Gripper:
        def read_width_m(self):return .04
    sampler=RobotStateSampler({"left":Arm(),"right":Arm()},
        {"left":Gripper(),"right":Gripper()})
    try:
        with sampler.sample() as pending:
            samples=pending.result()
        assert calls.count("joints")==2
        assert sum(isinstance(c,tuple) and c[0]=="vendor_fk" for c in calls)==2
        np.testing.assert_allclose(samples["right"]["tcp_pose_sdk"],[
            [1,0,0,.1],[0,1,0,.2],[0,0,1,.3],[0,0,0,1]])
    finally:sampler.close()

def test_live_tcp_cannot_fall_back_to_model_fk(monkeypatch):
    from gene_agent.realworld.scene import observed_tcp_pose
    measured=np.eye(4);measured[:3,3]=[.3,.2,.1]
    value={"snapshot":{"active_group":"right"},"provenance":{
        "live_state":{},"tcp_poses_world":{"right":measured.tolist()}}}
    np.testing.assert_allclose(observed_tcp_pose(value),measured)
    value["provenance"]["tcp_poses_world"].clear()
    with pytest.raises(ValueError,match="SDK TCP"):
        observed_tcp_pose(value)

def test_real_insert_executes_clearance_lift_from_actual_feedback(tmp_path,monkeypatch):
    from gene_agent.realworld.microwave import insert as module,attachment as attachments
    from gene_agent.tasks.checks import CheckResult
    from gene_agent.skills.runtime.planning import SkillStage
    import gene_agent.realworld.microwave.strategy_planning as planner
    action=[];reads=[]
    def stage(name):return SkillStage(name,"planned")
    poses={n:np.eye(4).tolist() for n in ("preplace","insert","retreat_1","retreat_2")}
    targets={"source_frame_number":1,"T_world_profile_microwave":np.eye(4).tolist(),
        "T_tcp_bowl_from_actual_hold":np.eye(4).tolist(),
        "keyposes_world_tcp":poses,"held_turn_world_tcp":None,
        "screened_held_trajectories":{}}
    def prefix(*args,**kw):
        return {"success":True,"targets":targets,"preplace_stage":stage("preplace"),
                "attached_collision_snapshot_sha256":"attached"}
    def current_plan(scene,name,*args,**kw):
        return {"success":True,"stage":stage(name),"stage_collision_snapshot_sha256":"now"}
    def linear(scene,*args,**kw):
        return current_plan(scene,"insert")
    def retreat(scene,*args,stage_name,**kw):
        assert reads[-1]>=5
        return current_plan(scene,stage_name)
    monkeypatch.setattr(module,"plan_observed_bowl_insert_prefix",prefix)
    monkeypatch.setattr(module,"plan_observed_bowl_insert_linear",linear)
    monkeypatch.setattr(module,"plan_observed_bowl_detached_retreat_stage",retreat)
    monkeypatch.setattr(planner,"plan_stage",current_plan)
    monkeypatch.setattr(attachments,"prepare_observed_bowl_attachment",
        lambda scene,*args,**kw:{"snapshot":{"snapshot_sha256":"attached"},
                                "spheres_tool_m":[[0,0,0,.01]]})
    monkeypatch.setattr(module.time,"sleep",lambda s:None)
    def observe():
        frame=len(reads)+1;reads.append(frame)
        return {"snapshot":{"snapshot_sha256":str(frame),"active_group":"right"},
            "provenance":{"source_frame_number":frame,"source_timestamp_ms":frame*1000.,
                "physics_stepped":False,"pose_frame_verified":True,
                "collision_snapshot_sha256":str(frame),
                "joint_positions_by_group_rad":{"right":[.1,.2]},
                "gripper_widths_m":{"right":.0792 if "release" in action else .007},
                "aligned_object_poses_world":{"Bowl/Bowl_0":np.eye(4).tolist(),
                    "Microwave/Microwave_0":np.eye(4).tolist()}},
            "robot_metadata":{"joint_groups":{"right":["j1","j2"]},
                "grippers":{"right":{"width_range_m":[0.,.08]}}}}
    class Runtime:
        def run_motion_stage(self,stage,*,execute):
            action.append(stage.name)
            return {"source_format":"realworld_motion_stage","status":"executed","success":True,
                "measured_end":{"joint_names":["j1","j2"],"positions_rad":[.1,.2]}}
        def run_gripper_stage(self,name,**kwargs):
            assert kwargs["target_width_m"]==pytest.approx(.0792)
            action.append(name)
            return {"source_format":"realworld_gripper_stage","status":"executed","success":True,
                "measured_end_width_m":.0792,"measured_joint_start":{
                    "joint_names":["j1","j2"],"positions_rad":[.1,.2]}}
    check=lambda *args:CheckResult("pass","manual_terminal",{"answer":"y"})
    pick={"execute":True,"status":"executed","pick_success":True,
        "post_lift_scene_snapshot_sha256":"1","post_lift_source_frame_number":1,
        "post_lift_source_timestamp_ms":1000.,"postcondition":check().to_dict(),
        "grasp":{"selected_candidate_id":"test"}}
    report=module.run_observed_bowl_insert(pick_report=pick,observe_collision_scene=observe,
        make_runtime=lambda key:Runtime(),planner_pool=object(),microwave_frame_alignment={},
        output_dir=tmp_path/"insert",execute=True,max_microwave_translation_m=.01,
        max_microwave_rotation_rad=.1,max_hold_translation_m=.01,max_hold_rotation_rad=.1,
        max_start_error_rad=.03,max_gripper_width_error_m=.002,
        check_hold=check,check_insert_path=check,check_release=check,check_place=check)
    assert report["insert_success"],report.get("first_failure")
    assert action==["preplace","insert","release","retreat_lift","retreat_1","retreat_2"]
    assert len(reads)==8

def test_default_knob_grasp_matches_simulation_parameters(scene):
    from gene_agent.realworld.microwave.knob_planning import build_observed_knob_targets
    result=build_observed_knob_targets(scene)
    assert len(result["proposals"])==1
    params=result["proposals"][0].parameters
    assert float(params["tip_x_rad"])==pytest.approx(np.deg2rad(10.))
    assert float(params["azimuth_rad"])==pytest.approx(np.deg2rad(210.))

def test_object_refresh_uses_current_frame_segmentation_registration(tmp_path):
    from test_pose_tracking import _settings
    from gene_agent.realworld.pose_tracking import track_observed_scene_from_capture
    calls=[]
    class Client:
        def stop(self,session_id):calls.append(("stop",session_id))
        def track_frame(self,*args,**kwargs):
            raise AssertionError("独立感知不得退回旧session单步track")
    client=Client()
    settings=_settings(tmp_path,client)
    settings["sessions"]={"Bowl/Bowl_0":"bowl-session"}
    def estimate(object_id,session_id,*,capture_dir,pose_client,camera_id):
        calls.append(("segment_register",object_id,session_id,str(capture_dir)))
        return {"ok":True,"session_id":session_id,"camera_id":camera_id,
                "T_camera_object":np.eye(4).tolist()}
    observed=track_observed_scene_from_capture(**settings,
        recover_session=estimate,fresh_estimation=True)
    assert calls[0]==("stop","bowl-session")
    assert calls[1][:3]==("segment_register","Bowl/Bowl_0","bowl-session")
    assert set(observed.object_poses_sdk)=={"Bowl/Bowl_0"}
    provenance=json.loads((settings["output_dir"]/"capture_provenance.json").read_text())
    assert provenance["pose_update_method"]=="current_frame_segmentation_then_registration"

def test_object_refresh_missing_estimator_does_not_destroy_session(tmp_path):
    from test_pose_tracking import _settings
    from gene_agent.realworld.pose_tracking import track_observed_scene_from_capture
    class Client:
        def stop(self,*args):raise AssertionError("参数不完整不应清除session")
    with pytest.raises(ValueError,match="感知参数"):
        track_observed_scene_from_capture(**_settings(tmp_path,Client()),fresh_estimation=True)

def test_predicted_release_snapshot_keeps_real_state_contract(scene,tmp_path):
    from gene_agent.realworld.planning import prepare_plan_only_context
    from gene_agent.realworld.microwave.prediction import is_prediction
    frozen=planning.frozen_geometry(scene)
    geometry._set_frozen_gripper_width(frozen.model,frozen.data,frozen.robot_metadata,"right",.0792)
    import mujoco
    mujoco.mj_forward(frozen.model,frozen.data)
    predicted=planning.predicted_scene(scene,frozen,tmp_path/"predicted")
    assert is_prediction(predicted)
    assert predicted["provenance"]["gripper_widths_m"]["right"]==pytest.approx(.0792)
    assert predicted["snapshot"]["other_arm_frozen"]
    assert predicted["snapshot"]["frozen_groups"]==["left"] or set(predicted["snapshot"]["frozen_groups"])=={"left"}
    class Pool:
        def acquire(self,settings,snapshot,**kwargs):
            self.settings=settings
            return SimpleNamespace(planner=SimpleNamespace(),cache_hit=True,
                setup_duration_s=0.,rebuild_reason=None)
    pool=Pool()
    context=prepare_plan_only_context(predicted,group="right",planner_pool=pool)
    assert tuple(context["start"].positions_rad)==tuple(
        predicted["provenance"]["joint_positions_by_group_rad"]["right"])
