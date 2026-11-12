"""操作 Pipeline 从真实 observation 规划，并只执行对应版本的轨迹。"""
import numpy as np
import pytest

from gene_agent.motion.curobo.runtime_bridge import CuroboRuntimeBridge


class Runtime:
    def __init__(self):
        self.current_id = "state-1"
        self.robot_metadata = {"joint_groups": {"right": ["j1"]}}
        self.model = object()
        self.data = object()
        self.executed = None

    def query(self, **kwargs):
        assert kwargs == {"robot_groups": ("right",)}
        return {"state_id": self.current_id, "robots": {"right": {
            "joint_names": ["j1"], "joint_positions_rad": [0.2],
        }}}

    def execute_joint_trajectory(self, **kwargs):
        self.executed = kwargs
        return {"success": True, "end_state_id": "state-2"}


def test_bridge_rejects_stale_state_before_constructing_planner(tmp_path):
    runtime = Runtime()
    bridge = CuroboRuntimeBridge(runtime, tmp_path)
    with pytest.raises(ValueError, match="过期"):
        bridge("final", np.eye(4), "state-0")
    assert runtime.executed is None


def test_bridge_uses_explicit_pivot_contract_and_keeps_rejection(tmp_path, monkeypatch):
    from gene_agent.motion.curobo import runtime_bridge as bridge_module
    runtime = Runtime()
    calls = []
    monkeypatch.setattr(bridge_module, "export_collision_scene", lambda *args: {
        "scene_model": {}, "scope": "test", "snapshot_sha256": "hash",
        "mounting_contact_policy": {}, "gripper_snapshot": {},
    })
    monkeypatch.setattr(bridge_module, "load_realman_motion_config",
                        lambda *args, **kwargs: {})

    class Planner:
        def __init__(self, **kwargs):
            pass
        def plan_pivot_tcp(self, names, q, target, **kwargs):
            calls.append(kwargs)
            return {"success": False, "status": "pivot_path_rejected",
                    "failure_category": "constraint_violation"}

    monkeypatch.setattr(bridge_module, "SynchronousPlanner", Planner)
    bridge = CuroboRuntimeBridge(runtime, tmp_path, pivot_tcp_stages={
        "tilt": {"max_tcp_excursion_m": .025, "max_tcp_path_length_m": .05}})
    result = bridge("tilt", np.eye(4), "state-1")
    assert result["success"] is False
    assert calls == [{"target_frame": "world_tcp", "diagnostic_label": "tilt", "max_tcp_excursion_m": .025,
                      "max_tcp_path_length_m": .05}]
    assert runtime.executed is None


@pytest.mark.parametrize("execute", [True, False])
@pytest.mark.parametrize("stage_policy", [True, False])
def test_bridge_executes_plan_from_latest_named_robot_state(tmp_path, monkeypatch, execute, stage_policy):
    from gene_agent.motion.curobo import runtime_bridge as bridge_module
    runtime = Runtime()
    captured = {}

    def export(model, data, metadata, group, output):
        captured["snapshot_output"] = output
        return {"scene_model": {}, "scope": "test", "snapshot_sha256": "hash",
                "mounting_contact_policy": {}, "gripper_snapshot": {}}

    class Planner:
        def __init__(self, **kwargs):
            captured["planner"] = kwargs

        def plan(self, names, q, target, *, target_frame, diagnostic_label):
            assert diagnostic_label == 'final'
            captured["input"] = (names, q, target_frame)
            return {"success": True, "joint_names": ["j1"],
                    "positions_rad": [[0.2], [0.3]], "interpolation_dt_s": .025}

    monkeypatch.setattr(bridge_module, "export_collision_scene", export)
    monkeypatch.setattr(bridge_module, "load_realman_motion_config",
                        lambda *args, **kwargs: {"joint_names": ["j1"]})
    monkeypatch.setattr(bridge_module, "SynchronousPlanner", Planner)
    checks = []
    def check(*args):
        checks.append(True)
        return {"success": True}
    bridge = CuroboRuntimeBridge(runtime, tmp_path, plan_postcheck_for_stage=check,
        execution_position_kp_scale_by_stage={'final': 1.5} if stage_policy else None,
        optimizer_collision_activation_distance_by_stage={'final': .001} if stage_policy else None)
    method = bridge if execute else bridge.plan_only
    result = method("final", np.eye(4), "state-1")

    assert result["success"] is True
    assert captured["input"] == (["j1"], [0.2], "world_tcp")
    assert checks == [True]
    if stage_policy:
        assert captured['planner']['optimizer_collision_activation_distance'] == .001
    else:
        assert 'optimizer_collision_activation_distance' not in captured['planner']
    if execute:
        assert runtime.executed["planned_state_id"] == "state-1"
        assert runtime.executed["positions_rad"] == [[0.2], [0.3]]
        if stage_policy:
            assert runtime.executed['position_kp_scale'] == 1.5
        else:
            assert 'position_kp_scale' not in runtime.executed
    else:
        assert runtime.executed is None
        assert result["execution"] is None
        assert result["evidence_mode"] == "planning_only"


def test_bridge_attaches_live_recorder_to_execution_not_planning(tmp_path, monkeypatch):
    from gene_agent.motion.curobo import runtime_bridge as bridge_module
    runtime = Runtime()
    events = []
    class Recorder:
        def begin_stage(self, stage, model, data):
            events.append(("begin", stage))
        def observer(self, stage):
            events.append(("observer", stage))
            return lambda *args: None
        def end_stage(self, stage, model, data):
            events.append(("end", stage))
    monkeypatch.setattr(bridge_module, "export_collision_scene", lambda *args: {
        "scene_model": {}, "scope": "test", "snapshot_sha256": "hash",
        "mounting_contact_policy": {}, "gripper_snapshot": {},
    })
    monkeypatch.setattr(bridge_module, "load_realman_motion_config", lambda *args, **kwargs: {})
    class Planner:
        def __init__(self, **kwargs):
            pass
        def plan(self, *args, **kwargs):
            return {"success": True, "joint_names": ["j1"],
                    "positions_rad": [[.2], [.3]], "interpolation_dt_s": .025}
    monkeypatch.setattr(bridge_module, "SynchronousPlanner", Planner)
    result = CuroboRuntimeBridge(runtime, tmp_path, visual_recorder=Recorder())(
        "pregrasp", np.eye(4), "state-1",
    )
    assert result["success"] is True
    assert events == [("begin", "pregrasp"), ("observer", "pregrasp"), ("end", "pregrasp")]
    assert callable(runtime.executed["step_observer"])


def test_bridge_triggers_failure_feedback_without_executing_failed_plan(tmp_path, monkeypatch):
    from gene_agent.motion.curobo import runtime_bridge as bridge_module
    runtime = Runtime()
    failures = []
    monkeypatch.setattr(bridge_module, "export_collision_scene", lambda *args: {
        "scene_model": {}, "scope": "test", "snapshot_sha256": "hash",
        "mounting_contact_policy": {}, "gripper_snapshot": {},
    })
    monkeypatch.setattr(bridge_module, "load_realman_motion_config",
                        lambda *args, **kwargs: {})

    class Planner:
        def __init__(self, **kwargs):
            pass

        def plan(self, *args, **kwargs):
            return {"success": False, "status": "goal_ik_no_solution",
                    "message": "no seed"}

    monkeypatch.setattr(bridge_module, "SynchronousPlanner", Planner)
    bridge = CuroboRuntimeBridge(
        runtime, tmp_path,
        on_failure=lambda stage, kind, details: failures.append((stage, kind, details)),
    )
    result = bridge("final", np.eye(4), "state-1")
    assert result["success"] is False
    assert runtime.executed is None
    assert failures == [("final", "planner_failed", result["plan"])]


def test_bridge_executes_task_constraint_rejected_prefix_for_video(
        tmp_path, monkeypatch):
    from gene_agent.motion.curobo import runtime_bridge as bridge_module

    runtime = Runtime()
    monkeypatch.setattr(bridge_module, "export_collision_scene", lambda *args: {
        "scene_model": {}, "scope": "test", "snapshot_sha256": "hash",
        "mounting_contact_policy": {}, "gripper_snapshot": {},
    })
    monkeypatch.setattr(bridge_module, "load_realman_motion_config",
                        lambda *args, **kwargs: {})

    class Planner:
        def __init__(self, **kwargs):
            pass
        def plan(self, *args, **kwargs):
            return {"success": True, "joint_names": ["j1"],
                    "positions_rad": [[.2], [.25], [.3], [.35]],
                    "interpolation_dt_s": .025}
        def export_pre_execution_failure(self, *args, **kwargs):
            return {"kind": "task_constraint"}

    monkeypatch.setattr(bridge_module, "SynchronousPlanner", Planner)
    bridge = CuroboRuntimeBridge(
        runtime, tmp_path, execute_rejected_plan_prefixes=True,
        plan_postcheck_for_stage=lambda *_: {
            "success": False, "waypoint_index": 2,
            "status": "pre_execution_task_constraint_rejected",
            "failure_category": "task_constraint_postcheck_rejected",
            "reason_code": "bowl_tilt_limit_exceeded",
        },
    )
    result = bridge("preplace", np.eye(4), "state-1")

    assert result["success"] is False
    assert result["plan"]["failure_category"] == "task_constraint_postcheck_rejected"
    assert result["execution"]["diagnostic_only"] is True
    assert result["execution"]["executed_prefix_end_index"] == 2
    assert runtime.executed["positions_rad"] == [[.2], [.25], [.3]]


def test_bridge_screens_complete_stage_sequence_and_preserves_trajectories(
        tmp_path, monkeypatch):
    from gene_agent.motion.curobo import runtime_bridge as bridge_module

    runtime = Runtime()
    monkeypatch.setattr(bridge_module, "export_collision_scene", lambda *args: {
        "scene_model": {}, "scope": "test", "snapshot_sha256": "hash",
        "mounting_contact_policy": {}, "gripper_snapshot": {},
    })
    monkeypatch.setattr(bridge_module, "load_realman_motion_config",
                        lambda *args, **kwargs: {})
    calls = []

    class Planner:
        def __init__(self, **kwargs):
            pass
        def plan(self, names, q, target, **kwargs):
            x = float(target[0, 3])
            calls.append(("preplace", x, tuple(q)))
            return {"success": True, "joint_names": ["j1"],
                    "positions_rad": [list(q), [x]], "interpolation_dt_s": .1}
        def plan_linear_tcp(self, names, q, target, **kwargs):
            x = float(target[0, 3])
            calls.append(("insert", x, tuple(q)))
            return {"success": x > .3, "status": "success" if x > .3
                    else "linear_path_rejected", "joint_names": ["j1"],
                    "positions_rad": [list(q), [x]], "interpolation_dt_s": .1}

    monkeypatch.setattr(bridge_module, "SynchronousPlanner", Planner)
    poses = []
    for x in (.1, .2, .3, .4):
        pose = np.eye(4)
        pose[0, 3] = x
        poses.append(pose)
    bridge = CuroboRuntimeBridge(
        runtime, tmp_path, linear_tcp_stages=("insert",),
        plan_postcheck_for_stage=lambda *_: {"success": True},
    )
    result = bridge.plan_only_candidate_sequences(
        "insert_sequence_screen",
        ((('preplace', poses[0]), ('insert', poses[1])),
         (('preplace', poses[2]), ('insert', poses[3]))),
        "state-1",
    )

    assert result["success"] is True
    assert result["plan"]["selected_candidate_index"] == 1
    assert set(result["plan"]["selected_stage_trajectories"]) == {"preplace", "insert"}
    assert calls == [
        ("preplace", .1, (.2,)), ("insert", .2, (.1,)),
        ("preplace", .3, (.2,)), ("insert", .4, (.3,)),
    ]
    assert runtime.executed is None


def test_bridge_rejects_large_joint_reversal_before_execution(tmp_path, monkeypatch):
    from gene_agent.motion.curobo import runtime_bridge as bridge_module
    runtime = Runtime()
    failures = []
    monkeypatch.setattr(bridge_module, "export_collision_scene", lambda *args: {
        "scene_model": {}, "scope": "test", "snapshot_sha256": "hash",
        "mounting_contact_policy": {}, "gripper_snapshot": {},
    })
    monkeypatch.setattr(
        bridge_module, "load_realman_motion_config", lambda *args, **kwargs: {},
    )

    class Planner:
        def __init__(self, **kwargs):
            pass

        def plan(self, *args, **kwargs):
            return {
                "success": True,
                "joint_names": ["j1"],
                "positions_rad": [[.2], [2.0]],
                "interpolation_dt_s": .025,
            }

    monkeypatch.setattr(bridge_module, "SynchronousPlanner", Planner)
    bridge = CuroboRuntimeBridge(
        runtime, tmp_path,
        max_joint_travel_rad_by_stage={"approach": 1.0},
        on_failure=lambda stage, kind, details: failures.append((stage, kind, details)),
    )
    result = bridge("approach", np.eye(4), "state-1")

    assert result["success"] is False
    assert result["execution"] is None
    assert runtime.executed is None
    assert result["plan"]["status"] == "joint_travel_rejected"
    assert result["plan"]["joint_travel_check"]["maximum_joint_travel_rad"] == 1.8
    assert failures[0][:2] == ("approach", "joint_travel_postcheck_rejected")


def test_bridge_reuses_selected_candidate_trajectory_after_live_revalidation(
        tmp_path, monkeypatch):
    from gene_agent.motion.curobo import runtime_bridge as bridge_module
    from gene_agent.motion.model import JointTrajectory

    class PoseRuntime(Runtime):
        def query(self, **kwargs):
            row = super().query(**kwargs)
            row["robots"]["right"]["T_world_tcp"] = np.eye(4).tolist()
            return row

    runtime = PoseRuntime()
    target = np.eye(4)
    target[0, 3] = .05
    monkeypatch.setattr(bridge_module, "export_collision_scene", lambda *args: {
        "scene_model": {}, "scope": "test", "snapshot_sha256": "hash",
        "mounting_contact_policy": {}, "gripper_snapshot": {},
    })
    monkeypatch.setattr(bridge_module, "load_realman_motion_config",
                        lambda *args, **kwargs: {})

    class Planner:
        def __init__(self, **kwargs):
            pass
        def plan_linear_tcp(self, *args, **kwargs):
            raise AssertionError("已有候选轨迹时不应重新求解另一个 IK 分支")
        def fk_batch(self, names, positions):
            poses = np.repeat(np.eye(4)[None], len(positions), axis=0)
            poses[:, 0, 3] = np.linspace(0., .05, len(positions))
            return poses
        def validate_joint_trajectory(self, names, positions):
            return {"valid": True, "checked_waypoints": len(positions)}

    monkeypatch.setattr(bridge_module, "SynchronousPlanner", Planner)
    preferred = JointTrajectory(("j1",), (0., .025, .05),
                                ((.2,), (.25,), (.3,)))
    result = CuroboRuntimeBridge(
        runtime, tmp_path, linear_tcp_stages=("approach",),
        preferred_trajectory_by_stage={"approach": preferred},
        endpoint_tolerance_rad=.03,
    )("approach", target, "state-1")

    assert result["success"] is True
    assert result["plan"]["trajectory_source"] == "selected_candidate_stage_branch"
    assert runtime.executed["positions_rad"] == [[.2], [.25], [.3]]


def test_bridge_enables_approximate_ik_only_for_configured_stage(tmp_path, monkeypatch):
    from gene_agent.motion.curobo import runtime_bridge as bridge_module
    runtime = Runtime()
    calls = []
    monkeypatch.setattr(bridge_module, "export_collision_scene", lambda *args: {
        "scene_model": {}, "scope": "test", "snapshot_sha256": "hash",
        "mounting_contact_policy": {}, "gripper_snapshot": {},
    })
    monkeypatch.setattr(bridge_module, "load_realman_motion_config", lambda *args, **kwargs: {})
    class Planner:
        def __init__(self, **kwargs):
            pass
        def plan(self, *args, **kwargs):
            calls.append(np.array(args[2], copy=True))
            return {"success": True, "joint_names": ["j1"],
                    "positions_rad": [[.2], [.3]], "interpolation_dt_s": .025}
    monkeypatch.setattr(bridge_module, "SynchronousPlanner", Planner)
    observed = {}
    def wrapped(planner, names, q, target, **kwargs):
        observed.update(kwargs)
        return planner.plan(names, q, target, target_frame="world_tcp")
    monkeypatch.setattr(bridge_module, "plan_with_approximate_ik", wrapped)
    bridge = CuroboRuntimeBridge(runtime, tmp_path, approximate_ik_by_stage={
        "pregrasp": {"max_position_error_m": .02, "max_rotation_error_rad": .1},
    })
    result = bridge("pregrasp", np.eye(4), "state-1")
    assert result["success"] and observed["budget"].max_position_error_m == .02
    assert len(calls) == 1


def test_bridge_requires_actual_tcp_within_approximate_budget(tmp_path, monkeypatch):
    from gene_agent.motion.curobo import runtime_bridge as bridge_module
    class ReportingRuntime(Runtime):
        def query(self, **kwargs):
            row = super().query(**kwargs)
            pose = np.eye(4); pose[0, 3] = .04
            row["robots"]["right"]["T_world_tcp"] = pose.tolist()
            return row
    runtime = ReportingRuntime()
    monkeypatch.setattr(bridge_module, "export_collision_scene", lambda *args: {
        "scene_model": {}, "scope": "test", "snapshot_sha256": "hash",
        "mounting_contact_policy": {}, "gripper_snapshot": {},
    })
    monkeypatch.setattr(bridge_module, "load_realman_motion_config", lambda *args, **kwargs: {})
    class Planner:
        def __init__(self, **kwargs): pass
        def plan(self, *args, **kwargs):
            return {"success": True, "joint_names": ["j1"],
                    "positions_rad": [[.2], [.3]], "interpolation_dt_s": .025}
    monkeypatch.setattr(bridge_module, "SynchronousPlanner", Planner)
    def wrapped(planner, names, q, target, **kwargs):
        result = planner.plan(names, q, target)
        result["approximate_ik"] = {"approximate": True, "budget": kwargs["budget"].to_dict(),
                                     "requested_tcp_pose_world": np.eye(4).tolist()}
        return result
    monkeypatch.setattr(bridge_module, "plan_with_approximate_ik", wrapped)
    result = CuroboRuntimeBridge(runtime, tmp_path, approximate_ik_by_stage={
        "pregrasp": {"max_position_error_m": .02, "max_rotation_error_rad": .1},
    })("pregrasp", np.eye(4), "state-1")
    assert not result["success"]
    assert result["execution"]["success"]
    assert result["approximate_ik_postcheck"]["position_error_m"] == .04


@pytest.mark.parametrize("change_policy", [False, True])
def test_bridge_defaults_to_graph_and_reuses_solver_between_stages(tmp_path, monkeypatch, change_policy):
    from gene_agent.motion.curobo import runtime_bridge as module
    from gene_agent.motion.model import PlannerMode
    created = []
    class Planner:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.closed = False
            created.append(self)
        def plan(self, *args, **kwargs):
            return {"success": True, "joint_names": ["j1"],
                    "positions_rad": [[.2], [.3]], "interpolation_dt_s": .025}
        def close(self):
            self.closed = True
    monkeypatch.setattr(module, "SynchronousPlanner", Planner)
    monkeypatch.setattr(module, "load_realman_motion_config", lambda *a, **k: {})
    monkeypatch.setattr(module, "export_collision_scene", lambda *a, **k: {
        "scene_model": {}, "scope": "test", "snapshot_sha256": "same",
        "mounting_contact_policy": {}, "gripper_snapshot": {}})
    runtime = Runtime()
    bridge = CuroboRuntimeBridge(runtime, tmp_path, contact_policy_for_stage=lambda stage, snapshot:
        {"robot_links": ["base"]} if change_policy and stage == "retreat" else {})
    assert bridge.plan_only("pregrasp", np.eye(4), "state-1")["success"]
    assert bridge.plan_only("retreat", np.eye(4), "state-1")["success"]
    assert len(created) == (2 if change_policy else 1)
    assert created[0].options["planner_mode"] is PlannerMode.CUDA_GRAPH_BENCHMARK
    assert runtime.executed is None
    bridge.close()
    assert all(planner.closed for planner in created)
