"""完整预测规划的状态交接、只读边界和回放契约；不模拟 cuRobo 成功率。"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gene_agent.realworld.microwave.scene import build_observed_collision_scene
from gene_agent.realworld.microwave.plan_only_world import PlanOnlyWorld
from gene_agent.realworld.microwave.plan_only import TaskPreview, plan_microwave_task1
from gene_agent.realworld.microwave.prediction import (
    assumption_check, predicted_report, report_basis)
from gene_agent.realworld.checks import resolve_realworld_check
from gene_agent.motion.model import (
    JointState, JointTrajectory, MotionOperation, MotionRequest, MotionResult,
    PlanningSceneRef, PoseTarget)
from gene_agent.skills.runtime.planning import SkillStage
from gene_agent.tasks.policies.microwave_task.policy import (
    microwave_task_steps, MicrowaveTaskPolicy)
from gene_agent.visualization.replay import ReplayTimeline


@pytest.fixture
def observed_scene(tmp_path):
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
        door_angle_rad=0., knob_angle_rad=.1,
        turntable_height_offset_m=0., microwave_floor_height_offset_m=0.,
        active_group="right", output_dir=tmp_path/"initial")


def stage_for(world, *, offset=.001, start_offset=0., backend="curobo"):
    # 这是状态接线测试返回值，不作为求解器或可达性证据。
    scene = world.scene()
    q = np.array([world.joint(n) for n in world.names])
    end = q.copy()
    end[0] += offset
    start = JointState(world.names, tuple(q))
    request = MotionRequest("test", MotionOperation.PLAN, world.group, start,
        PoseTarget("world_tcp", world.tcp()),
        PlanningSceneRef(world.revision, scene["snapshot"]["snapshot_sha256"], scene["snapshot"]["scope"]))
    result = MotionResult("test", MotionOperation.PLAN, True, backend=backend,
        planner_mode="cuda_graph_benchmark", duration_s=0.,
        goal_state=JointState(world.names, tuple(end)),
        trajectory=JointTrajectory(world.names, (0.,.1),
                                   (tuple(q+start_offset),tuple(end))))
    return SkillStage("test", "planned", request.target, request, result)


def test_prediction_does_not_retimestamp_or_move_other_arm(observed_scene, tmp_path):
    original = deepcopy(observed_scene)
    w = PlanOnlyWorld(observed_scene, tmp_path/"plan", max_start_error_rad=.03)
    w.event("open", width=.08)
    w.event("attach", attach=True, assumption="测试预测附着")
    hold = w.attachment.copy()
    w.motion("move", stage_for(w, offset=.02))
    scene = w.scene()
    for key in ("source_frame_number", "source_timestamp_ms"):
        assert scene["provenance"][key] == original["provenance"][key]
    assert scene["provenance"]["joint_positions_by_group_rad"]["left"] == original["provenance"]["joint_positions_by_group_rad"]["left"]
    np.testing.assert_allclose(w.pose("bowl"), w.tcp()@hold, atol=1e-10)
    assert observed_scene == original
    assert scene["provenance"]["prediction_step"] == 3
    assert scene["provenance"]["physics_stepped"] is False
    assert scene["snapshot"]["snapshot_sha256"] != original["snapshot"]["snapshot_sha256"]


@pytest.mark.parametrize("bad", ["start", "backend", "limit"])
def test_invalid_motion_cannot_advance_shadow(observed_scene, tmp_path, bad):
    w = PlanOnlyWorld(observed_scene, tmp_path/"plan", max_start_error_rad=.03)
    before = w.data.qpos.copy()
    stage = stage_for(w, offset=100. if bad=="limit" else .001,
                      start_offset=.1 if bad=="start" else 0.,
                      backend="diagnostic" if bad=="backend" else "curobo")
    with pytest.raises(ValueError):
        w.motion("bad", stage)
    np.testing.assert_array_equal(w.data.qpos, before)
    assert not w.stages


def test_planning_assumption_is_rejected_by_real_checks(observed_scene, tmp_path):
    w = PlanOnlyWorld(observed_scene, tmp_path/"plan", max_start_error_rad=.03)
    report = w.hold_report("candidate")
    assert report["execute"] is False and report["grasp_success"] is False
    assert predicted_report(w.scene(), report, "hold")
    with pytest.raises(ValueError, match="同一"):
        predicted_report(observed_scene, report, "hold")
    with pytest.raises(ValueError, match="不能作为真机"):
        resolve_realworld_check(assumption_check(w.scene()))


@pytest.mark.parametrize("knob", [False, True])
def test_full_task_uses_shared_order_and_previous_predicted_end(
        observed_scene, tmp_path, monkeypatch, knob):
    seen = []
    def planned(self, spec, folder):
        seen.append((spec.name, self.world.state()))
        self.world.motion(spec.name+"/move", stage_for(self.world))
    for kind in ("door", "transition", "pick", "insert", "knob", "extract", "table"):
        monkeypatch.setattr(TaskPreview, "plan_"+kind, planned)
    out = tmp_path/"preview"
    report = plan_microwave_task1(observed_scene, planner_pool=object(), output_dir=out,
        subtask_options={"door":{"max_start_error_rad":.03}}, include_knob=knob)
    expected = [s.name for s in microwave_task_steps(MicrowaveTaskPolicy(include_knob=knob))]
    assert [n for n,_ in seen] == expected
    assert report["planning_success"] is True
    assert report["status"] == "planned" and report["task_success"] is None
    assert report["execute"] is False and not report["operation_success"]
    for left, right in zip(report["steps"], report["steps"][1:]):
        assert left["predicted_end"] == right["predicted_start"]
    timeline = ReplayTimeline.load(out).select(mode="planned")
    assert timeline.duration_s > 0
    end = timeline.seek(timeline.duration_s)
    np.testing.assert_allclose(end.qpos, report["predicted_final"]["qpos"])
    with pytest.raises(ValueError, match="实际执行"):
        timeline.select(mode="actual")


def test_failure_keeps_prefix_and_does_not_enter_later_steps(
        observed_scene, tmp_path, monkeypatch):
    seen=[]
    def planned(self, spec, folder):
        seen.append(spec.name)
        if spec.name=="pick_bowl":
            raise RuntimeError("injected planning failure")
        self.world.motion(spec.name+"/move", stage_for(self.world))
    for kind in ("door", "transition", "pick"):
        monkeypatch.setattr(TaskPreview, "plan_"+kind, planned)
    report=plan_microwave_task1(observed_scene, planner_pool=object(),
        output_dir=tmp_path/"preview", subtask_options={"door":{"max_start_error_rad":.03}})
    assert seen==["open_door_first","after_open_transition","pick_bowl"]
    assert report["status"]=="planning_failed" and not report["planning_success"]
    assert report["completed_steps"]==2
    assert report["first_failure"]["step_id"]=="pick_bowl"
    timeline=ReplayTimeline.load(tmp_path/"preview").select(mode="planned")
    np.testing.assert_allclose(timeline.seek(timeline.duration_s).qpos,
                               report["predicted_final"]["qpos"])


def test_read_only_connection_does_not_configure_gripper():
    from gene_agent.realworld.devices.arms.realman.connection import connect_realman
    class SDK:
        def rm_create_robot_arm(self, *args):
            return SimpleNamespace(id=1)
        def rm_delete_robot_arm(self):
            return 0
        def rm_set_gripper_route(self, *args, **kwargs):
            raise AssertionError("plan_only must not change gripper configuration")
    with connect_realman("127.0.0.1", group="right", sdk=SDK(),
                         configure_gripper=False):
        pass


def test_cli_rejects_execute_and_plan_only():
    from tools.run.run_microwave_task1_realworld import main
    with pytest.raises(SystemExit) as error:
        main(["--perception","unused","--left-ip","127.0.0.1","--right-ip","127.0.0.2",
              "--subtask","all","--plan-only","--execute","--yes"])
    assert error.value.code == 2


def test_live_entry_reads_once_and_never_constructs_runtime(observed_scene, tmp_path, monkeypatch):
    from contextlib import contextmanager
    from test_live_cli import _full_site_config, load_site_config
    from tools.run import run_microwave_task1_realworld as cli
    calls = []
    path, _ = _full_site_config(tmp_path)
    config = load_site_config(path, subtask="all", execute=False, no_knob=True)
    @contextmanager
    def connect(ip, **kwargs):
        assert kwargs["configure_gripper"] is False
        calls.append(("connect", kwargs["group"]))
        try:
            yield SimpleNamespace(robot=object(), gripper=object())
        finally:
            calls.append(("close", kwargs["group"]))
    def observe():
        calls.append("observe")
        return observed_scene
    def plan(scene, **kwargs):
        from gene_agent.motion.curobo.optimization import current_optimization
        assert current_optimization() is not None
        assert scene is observed_scene
        calls.append("plan")
        return {"status":"planned","success":True,"task_success":None}
    monkeypatch.setattr("gene_agent.realworld.devices.arms.realman.connection.connect_realman", connect)
    monkeypatch.setattr("gene_agent.realworld.microwave.scene.make_live_microwave_observer",
                        lambda **kwargs:observe)
    monkeypatch.setattr("gene_agent.realworld.microwave.plan_only.plan_microwave_task1", plan)
    monkeypatch.setattr("gene_agent.realworld.runtime.RealWorldRuntime",
                        lambda *a,**k:pytest.fail("plan_only cannot construct hardware runtime"))
    monkeypatch.setattr(cli,"run_observed_microwave_task1",
                        lambda **k:pytest.fail("plan_only cannot enter hardware task"))
    result=cli.run_site(config, subtask="all", output_dir=tmp_path/"live_preview",
                        execute=False, plan_only=True)
    assert result["status"]=="planned"
    assert calls.count("observe")==1 and calls.count("plan")==1
    assert ("close","left") in calls and ("close","right") in calls
    optimization = json.loads((tmp_path / 'live_preview/planning_optimization.json').read_text())
    assert optimization['mode'] == 'single_reuse' and optimization['closed']


def test_shared_hold_geometry_accepts_prediction_without_physical_success(observed_scene, tmp_path):
    from gene_agent.realworld.microwave.attachment import prepare_observed_bowl_attachment
    from gene_agent.realworld.microwave.insert_planning import build_observed_bowl_insert_targets
    from gene_agent.realworld.microwave.place_planning import build_observed_table_place_targets
    w=PlanOnlyWorld(observed_scene,tmp_path/"plan",max_start_error_rad=.03)
    w.event("attach",attach=True)
    scene=w.scene()
    hold=w.hold_report("candidate")
    attachment=prepare_observed_bowl_attachment(scene,hold,output_dir=tmp_path/"attachment")
    assert attachment["hold_origin"]=="planned_grasp_assumption"
    assert attachment["physical_hold_evidence"]["evidence"]["predicted"]
    assert len(attachment["spheres_tool_m"])==5
    targets=build_observed_bowl_insert_targets(scene,hold,
        microwave_frame_alignment={"accepted_for_planning":True,"source":"test",
                                   "T_scene_asset_profile":np.eye(4).tolist()})
    assert set(targets["keyposes_world_tcp"])=={"preplace","insert","retreat_1","retreat_2"}
    table=build_observed_table_place_targets(scene,
        table_height_alignment={"accepted_for_planning":True,"source":"test","height_bias_m":0.},
        current_hold_check=assumption_check(scene))
    assert set(table["keyposes_world_tcp"])=={"preplace","place","retreat"}
    assert hold["grasp_success"] is False and hold["execute"] is False
    with pytest.raises(ValueError,match="同一"):
        prepare_observed_bowl_attachment(observed_scene,hold,output_dir=tmp_path/"bad")


@pytest.mark.parametrize("operation", [
    "insert_prefix", "insert_linear", "insert_retreat",
    "extract_motion", "extract_retreat",
    "table_prefix", "table_descent", "table_retreat", "lift"])
def test_prediction_reaches_existing_planner_with_unchanged_capture_frame(
        observed_scene, tmp_path, monkeypatch, operation):
    # 只替换求解器边界；实际运行目标、附着、frame/锚点校验。
    import hashlib
    from dataclasses import replace
    from gene_agent.realworld.microwave import (
        insert_planning as ins, extract_planning as ext,
        place_planning as plc, pick_planning as pk)
    from gene_agent.tasks.policies.microwave_task.operations.bowl import load_object_operation_profile
    class AtPlanner(Exception):
        pass
    def context(*a,**k):
        raise AtPlanner()
    for module in (ins,ext,plc,pk):
        monkeypatch.setattr(module,"prepare_plan_only_context",context)
    root=Path(__file__).resolve().parents[2]
    w=PlanOnlyWorld(observed_scene,tmp_path/"plan",max_start_error_rad=.03)
    w.set_joint("microwave_door_joint",np.pi/2)
    profile=load_object_operation_profile(root/"data/task_1","Bowl_0")
    w.event("open",width=profile["grasp_pregrasp_opening_width_m"])
    w.event("attach",attach=True)
    scene=w.scene()
    hold=w.hold_report("candidate")
    hold["grasp"]=dict(hold)
    check=assumption_check(scene)
    alignment={"accepted_for_planning":True,"source":"test",
               "T_scene_asset_profile":np.eye(4).tolist()}
    height={"accepted_for_planning":True,"source":"test","height_bias_m":0.}
    pose_limits=dict(max_microwave_translation_m=.005,max_microwave_rotation_rad=.15,
                     max_hold_translation_m=.005,max_hold_rotation_rad=.15)
    object_limits=dict(max_bowl_translation_m=.005,max_bowl_rotation_rad=.15)
    common=dict(planner_pool=object(),output_dir=tmp_path/"planned")
    initial=ins.build_observed_bowl_insert_targets(scene,hold,microwave_frame_alignment=alignment)
    table=plc.build_observed_table_place_targets(scene,table_height_alignment=height,current_hold_check=check)
    targets=ext.build_observed_bowl_extract_targets(scene,microwave_frame_alignment=alignment)
    release=w.release_report()
    with pytest.raises(AtPlanner):
        if operation=="insert_prefix":
            ins.plan_observed_bowl_insert_prefix(scene,hold,**common,
                microwave_frame_alignment=alignment,current_hold_check=check.to_dict())
        elif operation=="insert_linear":
            ins.plan_observed_bowl_insert_linear(scene,initial,hold,**common,**pose_limits,
                microwave_frame_alignment=alignment,hold_check=check.to_dict())
        elif operation=="insert_retreat":
            ins.plan_observed_bowl_detached_retreat_stage(scene,initial,release,
                **common,**object_limits,stage_name="retreat_1",
                microwave_frame_alignment=alignment,
                max_microwave_translation_m=.005,max_microwave_rotation_rad=.15)
        elif operation=="extract_motion":
            ext.plan_observed_bowl_extract_motion_stage(scene,targets,**common,**object_limits,
                stage_name="pre_extract_2",microwave_frame_alignment=alignment,
                max_microwave_translation_m=.005,max_microwave_rotation_rad=.15)
        elif operation=="extract_retreat":
            ext.plan_observed_bowl_extract_retreat_stage(scene,targets,hold,**common,**pose_limits,
                stage_name="post_extract_1",current_hold_check=check,
                T_tcp_bowl_after_close=w.attachment,microwave_frame_alignment=alignment)
        elif operation=="table_prefix":
            plc.plan_observed_table_place_prefix(scene,hold,**common,
                current_hold_check=check,table_height_alignment=height)
        elif operation=="table_descent":
            plc.plan_observed_table_descent(scene,table,hold,**common,
                current_hold_check=check,table_height_alignment=height,
                max_table_height_delta_m=.005,max_hold_translation_m=.005,max_hold_rotation_rad=.15)
        elif operation=="table_retreat":
            plc.plan_observed_table_detached_retreat(scene,table,release,**common,**object_limits,
                max_table_translation_m=.005,max_table_rotation_rad=.15)
        elif operation=="lift":
            from gene_agent.realworld.microwave.attachment import prepare_observed_bowl_attachment
            approach=stage_for(w)
            request=replace(approach.request,operation=MotionOperation.LINEAR_TCP)
            result=replace(approach.result,operation=MotionOperation.LINEAR_TCP)
            approach=replace(approach,name="approach",request=request,result=result)
            hold["stages"]=[{"stage":"approach","status":"planned","plan":{"stage":approach.to_dict()}}]
            attachment=prepare_observed_bowl_attachment(scene,hold,output_dir=tmp_path/"attachment")
            selection={"selected_candidate_id":"candidate",
                "skill_profile_sha256":hashlib.sha256(pk._PROFILE_PATH.read_bytes()).hexdigest(),
                "skill_profile":"pick_bowl_legacy_visual",
                "plan":{"stages":[approach.to_dict()]}}
            pk.plan_observed_bowl_lift(scene,selection,hold,attachment,**common)


def test_playback_uses_the_same_collision_layer_as_the_planner(observed_scene,tmp_path):
    from dataclasses import replace
    from gene_agent.motion.curobo.scene import export_self_collision_only_snapshot
    w=PlanOnlyWorld(observed_scene,tmp_path/"plan",max_start_error_rad=.03)
    base=w.scene()
    layer=export_self_collision_only_snapshot(base["snapshot"],w.output/"contact_self_only")
    stage=stage_for(w)
    request=replace(stage.request,scene=PlanningSceneRef(
        w.revision,layer["snapshot_sha256"],layer["scope"]))
    stage=replace(stage,request=request)
    w.motion("specific_collision_layer",stage)
    row=w.stages[-1]
    collision=json.loads(Path(row["collision_snapshot_ref"]).read_text())
    assert collision["snapshot_sha256"]==request.scene.snapshot_sha256
    assert collision["snapshot_sha256"]!=base["snapshot"]["snapshot_sha256"]


def test_saved_prediction_is_rejected_before_hardware_access(observed_scene,tmp_path):
    from dataclasses import replace
    from gene_agent.realworld.runtime import RealWorldRuntime
    w=PlanOnlyWorld(observed_scene,tmp_path/"plan",max_start_error_rad=.03)
    stage=replace(stage_for(w),details={"execution_forbidden":True})
    runtime=object.__new__(RealWorldRuntime)
    # runtime 故意没有 robot/gripper，防护必须先于任何 SDK 读取或命令。
    result=runtime.run_motion_stage(stage,execute=True)
    assert result["success"] is False
    assert result["failure"]["code"]=="prediction_execution_forbidden"
