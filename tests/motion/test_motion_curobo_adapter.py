"""cuRobo 适配器测试使用同步假后端，不启动 GPU。"""

import numpy as np
import pytest

from gene_agent.motion import (
    CollisionPolicy,
    JointState,
    MotionFailureKind,
    MotionOperation,
    MotionRequest,
    PlannerMode,
    PlanningSceneRef,
    PoseTarget,
)
from gene_agent.motion.curobo import CuroboPlannerAdapter, SynchronousPlanner, describe_start_failure, solver_runtime_options, solver_failure_diagnostic
from tools.diagnostics.check_motion import measured_calls


class FakeLegacyPlanner:
    def __init__(self, raw):
        self.raw = raw
        self.joint_names = ["joint_a", "joint_b"]
        self.scene_snapshot_sha256 = "scene-hash"
        self.scene_scope = "table+bowl"
        self.planner_mode = PlannerMode.DEVELOPMENT
        self.random_seed = 17
        self.calls = []

    def fk(self, names, positions):
        self.calls.append(("fk", tuple(names), tuple(positions)))
        return np.eye(4)

    def solve_ik(self, names, positions, target, *, target_frame):
        self.calls.append(("ik", tuple(names), tuple(positions), target_frame))
        return dict(self.raw)

    def plan(self, names, positions, target, *, target_frame):
        self.calls.append(("plan", tuple(names), tuple(positions), target_frame))
        return dict(self.raw)


def _request(operation=MotionOperation.IK, *, scene_hash="scene-hash"):
    pose = np.eye(4)
    pose[:3, 3] = [0.3, 0.1, 0.2]
    return MotionRequest(
        "adapter-request",
        operation,
        "right",
        JointState(("joint_a", "joint_b"), (0.0, 0.1)),
        PoseTarget("world_tcp", pose),
        PlanningSceneRef(4, scene_hash, "table+bowl"),
        CollisionPolicy(),
        seed=17,
        planner_mode=PlannerMode.DEVELOPMENT,
    )


def test_adapter_routes_pivot_without_losing_path_evidence():
    from dataclasses import replace
    from gene_agent.motion import PivotLimits
    backend = FakeLegacyPlanner({"success": True, "positions_rad": [[0., .1], [.1, .2]],
                                 "interpolation_dt_s": .025, "duration_s": .03,
                                 "pivot_path_metrics": {"max_tcp_excursion_m": .002}})
    def pivot(names, q, target, **kwargs):
        backend.calls.append(kwargs)
        return backend.raw
    backend.plan_pivot_tcp = pivot
    request = replace(_request(MotionOperation.PLAN), operation=MotionOperation.PIVOT_TCP,
                      pivot_limits=PivotLimits(.01, .02))
    result = CuroboPlannerAdapter(backend, group="right", scene_revision=4).plan(request)
    assert result.success and result.trajectory is not None
    assert backend.calls == [{"target_frame": "world_tcp", "max_tcp_excursion_m": .01,
                              "max_tcp_path_length_m": .02}]
    assert result.metrics["pivot_path_metrics"]["max_tcp_excursion_m"] == .002


def test_adapter_maps_legacy_success_to_typed_ik_and_plan_results() -> None:
    ik_backend = FakeLegacyPlanner(
        {
            "success": True,
            "status": "success",
            "positions_rad": [[0.2, 0.3]],
            "duration_s": 0.012,
        }
    )
    ik_result = CuroboPlannerAdapter(ik_backend, group="right", scene_revision=4).solve_ik(
        _request()
    )
    assert ik_result.success
    assert ik_result.goal_state.positions_rad == (0.2, 0.3)
    assert ik_result.trajectory is None

    plan_backend = FakeLegacyPlanner(
        {
            "success": True,
            "status": "success",
            "positions_rad": [[0.0, 0.1], [0.2, 0.3], [0.25, 0.35]],
            "interpolation_dt_s": 0.025,
            "duration_s": 0.032,
        }
    )
    plan_result = CuroboPlannerAdapter(plan_backend, group="right", scene_revision=4).plan(
        _request(MotionOperation.PLAN)
    )
    assert plan_result.success
    assert plan_result.goal_state.positions_rad == (0.25, 0.35)
    assert plan_result.trajectory.times_s == (0.0, 0.025, 0.05)


def test_adapter_preserves_structured_failure_detail() -> None:
    raw = {
        "success": False,
        "status": "solver_failed",
        "failure_category": "path_not_found",
        "message": "候选路径违反关节限位（首个失败 waypoint=12）",
        "failure_detail": {
            "stage": "trajectory",
            "reason_code": "path_joint_limit_violation",
            "human_summary": "候选路径违反关节限位（首个失败 waypoint=12）",
            "first_invalid_waypoint": 12,
        },
        "duration_s": 0.02,
    }
    result = CuroboPlannerAdapter(
        FakeLegacyPlanner(raw), group="right", scene_revision=4,
    ).plan(_request(MotionOperation.PLAN))
    assert not result.success
    assert result.failure.kind is MotionFailureKind.PATH_NOT_FOUND
    assert result.failure.details["failure_detail"]["reason_code"] == (
        "path_joint_limit_violation"
    )


def test_plan_no_seed_diagnostic_survives_adapter_boundary() -> None:
    backend = FakeLegacyPlanner({
        "success": False, "status": "goal_ik_no_solution",
        "failure_category": "goal_ik_no_solution", "duration_s": .1,
        "ik_diagnostic": {"selected_seed_reason": "pose_not_converged",
                          "world_obstruction_probe": {"reason": "unresolved_without_world"}},
    })
    result = CuroboPlannerAdapter(backend, group="right", scene_revision=4).plan(
        _request(MotionOperation.PLAN)
    )
    assert result.failure.details["ik_diagnostic"]["selected_seed_reason"] == "pose_not_converged"


def test_adapter_routes_explicit_approximate_budget_and_preserves_evidence(monkeypatch):
    from dataclasses import replace
    from gene_agent.motion.curobo import core as core_module
    backend = FakeLegacyPlanner({"success": True})
    observed = {}
    def wrapped(planner, names, q, target, **kwargs):
        observed.update(kwargs)
        return {"success": True, "status": "success", "positions_rad": [[0., .1], [.2, .3]],
                "interpolation_dt_s": .025, "duration_s": .1,
                "approximate_ik": {"approximate": True, "position_error_m": .01,
                                   "strict_ik_success": False}}
    monkeypatch.setattr(core_module, "plan_with_approximate_ik", wrapped)
    request = replace(_request(MotionOperation.PLAN), approximate_ik_budget={
        "max_position_error_m": .02, "max_rotation_error_rad": .1,
    })
    result = CuroboPlannerAdapter(backend, group="right", scene_revision=4).plan(request)
    assert result.success
    assert observed["budget"].max_position_error_m == .02
    assert result.metrics["approximate_ik"]["strict_ik_success"] is False


def test_adapter_rejects_stale_scene_before_calling_backend() -> None:
    backend = FakeLegacyPlanner({"success": True})
    result = CuroboPlannerAdapter(backend, group="right", scene_revision=4).solve_ik(
        _request(scene_hash="stale-hash")
    )

    assert not result.success
    assert result.failure.kind is MotionFailureKind.INVALID_REQUEST
    assert "scene snapshot" in result.failure.summary
    assert backend.calls == []


@pytest.mark.parametrize(
    "operation, category, expected",
    [
        (MotionOperation.IK, "start_collision", MotionFailureKind.START_COLLISION),
        (MotionOperation.IK, "solver_rejected", MotionFailureKind.CONSTRAINT_VIOLATION),
        (MotionOperation.PLAN, "solver_rejected", MotionFailureKind.CONSTRAINT_VIOLATION),
        (MotionOperation.IK, "collision_postcheck_rejected", MotionFailureKind.GOAL_COLLISION),
        (MotionOperation.PLAN, "collision_postcheck_rejected", MotionFailureKind.PATH_COLLISION),
        (MotionOperation.PLAN, "input_validation_error", MotionFailureKind.INVALID_REQUEST),
        (MotionOperation.PLAN, "backend_exception", MotionFailureKind.BACKEND_ERROR),
    ],
)
def test_adapter_normalizes_legacy_failure_categories(operation, category, expected) -> None:
    backend = FakeLegacyPlanner(
        {
            "success": False,
            "status": "legacy_status",
            "failure_category": category,
            "message": "legacy failure",
            "raw_error": "full traceback",
            "duration_s": 0.02,
        }
    )
    adapter = CuroboPlannerAdapter(backend, group="right", scene_revision=4)

    result = adapter.solve_ik(_request(operation)) if operation is MotionOperation.IK else adapter.plan(_request(operation))

    assert not result.success
    assert result.failure.kind is expected
    assert result.failure.backend_status == "legacy_status"
    assert result.failure.details["raw_error"] == "full traceback"


def test_solver_modes_have_explicit_cuda_graph_and_seed_options() -> None:
    assert solver_runtime_options(PlannerMode.DEVELOPMENT, 17) == {
        "use_cuda_graph": False,
        "random_seed": 17,
    }


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeSolver:
    class config:
        position_tolerance = 0.005
        orientation_tolerance = 0.05


def test_ik_diagnostic_distinguishes_pose_error_from_feasible_constraint() -> None:
    raw = type("Result", (), {
        "position_error": _FakeTensor([[0.0078]]),
        "rotation_error": _FakeTensor([[0.017]]),
        "feasible": _FakeTensor([[True]]),
    })()
    diagnostic = solver_failure_diagnostic(raw, "ik", _FakeSolver())
    assert diagnostic["reason"] == "pose_not_converged"
    assert diagnostic["position_error_m"] == pytest.approx(0.0078)
    assert diagnostic["position_tolerance_m"] == 0.005
    assert diagnostic["constraint_feasible"] is True


def test_ik_diagnostic_records_constraint_failure_without_asserting_collision() -> None:
    raw = type("Result", (), {
        "position_error": _FakeTensor([[0.001]]),
        "rotation_error": _FakeTensor([[0.01]]),
        "feasible": _FakeTensor([[False]]),
    })()
    diagnostic = solver_failure_diagnostic(raw, "ik", _FakeSolver())
    assert diagnostic["reason"] == "constraint_rejected"
    assert diagnostic["collision_confirmed"] is False


def test_adapter_maps_diagnosed_ik_and_path_failures_to_distinct_kinds() -> None:
    for operation, category, expected in [
        (MotionOperation.IK, "ik_pose_not_converged", MotionFailureKind.GOAL_IK_NO_SOLUTION),
        (MotionOperation.PLAN, "path_not_found", MotionFailureKind.PATH_NOT_FOUND),
    ]:
        backend = FakeLegacyPlanner({
            "success": False, "status": "solver_failed",
            "failure_category": category,
            "solver_diagnostic": {"reason": "pose_not_converged"},
            "duration_s": 0.01,
        })
        adapter = CuroboPlannerAdapter(backend, group="right", scene_revision=4)
        result = adapter.solve_ik(_request(operation)) if operation is MotionOperation.IK else adapter.plan(_request(operation))
        assert result.failure.kind is expected
        assert result.failure.details["solver_diagnostic"]["reason"] == "pose_not_converged"


def test_start_check_rejects_external_collision_even_if_optimizer_mask_passes() -> None:
    class StrictValidator:
        calls = 0

        def check(self, positions, names):
            self.calls += 1
            assert names == ["joint_a", "joint_b"]
            return {"valid": False, "max_collision_costs": {"body": 0.0, "external": 0.02, "self": 0.0}}

    planner = SynchronousPlanner.__new__(SynchronousPlanner)
    planner._validator = StrictValidator()
    planner.planner_joint_names = ["joint_a", "joint_b"]
    planner._start_validation_cache = {}
    planner._configuration_validity = lambda positions: [True]
    planner._collision_diagnostic = lambda positions, **kwargs: {
        "scene_collision": True, "self_collision": False,
        "colliding_obstacles": ["bowl_collision"],
    }
    first = planner._start_configuration_check([0.0, 0.1])
    second = planner._start_configuration_check([0.0, 0.1])
    assert first["valid"] is False
    assert first["collision_diagnostic"]["colliding_obstacles"] == ["bowl_collision"]
    assert first == second
    assert planner._validator.calls == 1


def test_start_check_keeps_explicit_mounting_contact_valid() -> None:
    class StrictValidator:
        def check(self, positions, names):
            return {"valid": True, "max_collision_costs": {"body": 0.0, "external": 0.0, "self": 0.0}}

    planner = SynchronousPlanner.__new__(SynchronousPlanner)
    planner._validator = StrictValidator()
    planner.planner_joint_names = ["joint_a", "joint_b"]
    planner._start_validation_cache = {}
    planner._configuration_validity = lambda positions: [True]
    assert planner._start_configuration_check([0.0, 0.1])["valid"] is True


def test_start_failure_names_only_forbidden_contact() -> None:
    status, category, message = describe_start_failure({"collision_diagnostic": {
        "joint_limit_violation": False, "self_collision": False,
        "scene_collision": True,
        "colliding_obstacles": ["bowl_collision"],
        "allowed_mounting_contacts": ["robot_body_attachment"],
    }})
    assert (status, category) == ("start_collision", "start_collision")
    assert "bowl_collision" in message
    assert "robot_body_attachment" not in message
    assert solver_runtime_options(PlannerMode.CUDA_GRAPH_BENCHMARK, 23) == {
        "use_cuda_graph": True,
        "random_seed": 23,
    }


@pytest.mark.parametrize(
    "raw",
    [
        {"success": True, "status": "success", "duration_s": 0.01},
        {"success": True, "status": "success", "positions_rad": [[0.2, 0.3]], "duration_s": float("nan")},
    ],
)
def test_adapter_classifies_malformed_backend_success_as_backend_error(raw) -> None:
    result = CuroboPlannerAdapter(
        FakeLegacyPlanner(raw), group="right", scene_revision=4
    ).solve_ik(_request())

    assert not result.success
    assert result.failure.kind is MotionFailureKind.BACKEND_ERROR
    assert "backend" in result.failure.summary.lower()


def test_benchmark_separates_initialization_warmup_and_measurement() -> None:
    calls = []

    def call():
        calls.append(len(calls) + 1)
        return {"duration_s": calls[-1] / 1000.0, "success": True}

    final, timing = measured_calls(call, warmup_runs=2, measure_runs=3)
    assert calls == [1, 2, 3, 4, 5, 6]
    assert timing["initialization_duration_s"] == 0.001
    assert timing["warmup_duration_s"] == [0.002, 0.003]
    assert timing["measured_duration_s"] == [0.004, 0.005, 0.006]
    assert final["duration_s"] == 0.006


def test_typed_result_preserves_collision_scope_and_failed_approximate_attempts():
    raw={'success':False,'status':'no_path','failure_category':'path_not_found','duration_s':.3,
         'collision_mode':'robot_only','environment_collision_enabled':False,
         'planning_attachment_enabled':False,'self_collision_check':True,
         'collision_verification':{'valid':True},'start_state_check':{'valid':True},
         'approximate_ik_attempts':[{'seed_index':2,'success':False}],
         'approximate_ik_candidate_screen':[{'seed_index':1,'reason':'fk_error_above_budget'}]}
    result=CuroboPlannerAdapter(FakeLegacyPlanner(raw),group='right',scene_revision=4).plan(_request(MotionOperation.PLAN))
    for key in ('collision_mode','environment_collision_enabled','planning_attachment_enabled','self_collision_check','start_state_check','collision_verification'):
        assert result.metrics[key]==raw[key]
    for key in ('approximate_ik_attempts','approximate_ik_candidate_screen'):
        assert result.failure.details[key]==raw[key]


def test_local_contact_request_requires_matching_validated_scene_policy():
    from dataclasses import replace
    from gene_agent.motion import CollisionAllowance
    policy=CollisionPolicy('lift_support',(CollisionAllowance('gene_held_tool','table','lift'),))
    backend=FakeLegacyPlanner({'success':True,'positions_rad':[[.2,.3]],'duration_s':.01})
    backend.mounting_contact_policy={'robot_links':['gene_held_tool'],'obstacle_names':['table_shape'],'reason':policy.to_dict()}
    backend._validator=object()
    adapter=CuroboPlannerAdapter(backend,group='right',scene_revision=4)
    result=adapter.solve_ik(replace(_request(),collision_policy=policy))
    assert result.success
    assert not adapter.solve_ik(_request()).success
    backend._validator=None
    assert not adapter.solve_ik(replace(_request(),collision_policy=policy)).success


def test_bad_diagnostic_payload_preserves_original_failure(monkeypatch):
    from gene_agent.motion.curobo.diagnostics import DiagnosticOptions
    backend=FakeLegacyPlanner({'success':False,'status':'goal_ik_no_solution','failure_category':'goal_ik_no_solution'})
    backend.diagnostic_options=DiagnosticOptions(mode="on_failure")
    monkeypatch.setattr('gene_agent.motion.curobo.diagnostics.collect_failure_diagnostics',lambda *args: {'bad':object()})
    result=CuroboPlannerAdapter(backend,group="right",scene_revision=4).plan(_request(MotionOperation.PLAN))
    assert result.failure.kind is MotionFailureKind.GOAL_IK_NO_SOLUTION
    assert 'diagnostic_error' in result.failure.details


def test_optional_request_capture_saves_actual_success_without_changing_plan(tmp_path):
    from gene_agent.motion.configuration import CollisionMode
    import json
    backend=FakeLegacyPlanner({'success':True,'positions_rad':[[.2,.3]],'duration_s':.01})
    backend._source_robot_config=backend.robot_config={}
    backend.planner_joint_names=backend.joint_names
    backend.world_from_base=backend.link_from_tcp=np.eye(4)
    backend.tool_frame='tool';backend.scene_model={'cuboid':{}}
    backend.mounting_contact_policy={};backend._gripper_snapshot=None
    backend.held_object=None;backend.attachment_spheres_tool=None;backend.collision_mode=CollisionMode.FULL
    backend.diagnostic_snapshot=dict(scene_model=backend.scene_model,snapshot_sha256=backend.scene_snapshot_sha256,scope=backend.scene_scope)
    adapter=CuroboPlannerAdapter(backend,group='right',scene_revision=4,request_output_dir=tmp_path/'requests')
    result=adapter.solve_ik(_request())
    assert result.success and result.goal_state.positions_rad==(.2,.3)
    saved=json.loads(__import__('pathlib').Path(result.metrics['replay_request_ref']).read_text())
    assert saved['request']==_request().to_dict()
    assert saved['replay_config']['collision_mode']=='full'
    assert saved['results'][0]['success']
