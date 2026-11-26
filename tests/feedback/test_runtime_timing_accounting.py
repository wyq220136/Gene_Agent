"""可控单调时钟的端到端核算；不依赖真实等待、GPU 或机器人。"""
import json
import pytest

from gene_agent.feedback.runtime_timing import (
    RuntimeTimingRecorder, current_runtime_timing, measure_runtime,
    record_runtime_duration, runtime_timing_scope, timed_runtime_call,
)


class Clock:
    def __init__(self, seconds=0):
        self.ns = round(seconds * 1e9)
    def __call__(self):
        return self.ns
    def advance(self, seconds):
        self.ns += round(seconds * 1e9)


def prepare(output):
    output.mkdir()
    (output / "runtime_input.json").write_text("{}")


def assert_reconciled(value, expected):
    assert value["total_duration_s"] == expected
    accounting = value["accounting"]
    assert sum(accounting["exclusive_components_s"].values()) == pytest.approx(expected, abs=1e-9)
    assert sum(accounting["exclusive_stages_s"].values()) == pytest.approx(expected, abs=1e-9)
    assert accounting["reconciliation_error_s"] == 0


def test_200_seconds_includes_startup_wait_execution_cleanup_and_reporting(tmp_path):
    clock = Clock(100)
    output = tmp_path / "run"
    with runtime_timing_scope(started_ns=0, clock_ns=clock) as recorder:
        recorder.record("startup", "imports", 100, started_offset_s=0)
        with runtime_timing_scope(output) as nested:
            assert nested is recorder
            prepare(output)
            with measure_runtime("stage", "pick_bowl"):
                with measure_runtime("curobo_planning", "candidate_batch"):
                    clock.advance(20)
                with measure_runtime("perception", "observe"):
                    clock.advance(30)
                with measure_runtime("robot_execution", "pregrasp"):
                    clock.advance(10)
                with measure_runtime("manual_wait", "confirm"):
                    clock.advance(10)
                clock.advance(10)
            with measure_runtime("cleanup", "close_resources"):
                clock.advance(10)
        # 退出旧的内层 scope 后，总计时必须继续到最终汇总完成。
        with measure_runtime("reporting", "summary"):
            clock.advance(10)
        record_runtime_duration("curobo_solver", "backend_report_only", 2000)
    value = json.loads((output / "runtime_timing.json").read_text())
    assert_reconciled(value, 200)
    assert value["comparison"]["curobo_planning_wall_s"] == 20
    assert value["components"]["curobo_solver"]["reported_duration_s"] == 2000
    assert value["components"]["curobo_solver"]["exclusive_duration_s"] == 0
    assert value["stages"][0]["inclusive_duration_s"] == 80
    assert value["accounting"]["unattributed_s"] == 0


def test_nested_same_component_and_solver_are_never_added_twice():
    clock = Clock()
    with runtime_timing_scope(clock_ns=clock) as recorder:
        with measure_runtime("curobo_planning", "outer"):
            clock.advance(10)
            with measure_runtime("curobo_planning", "adapter"):
                clock.advance(5)
                with measure_runtime("curobo_solver", "cuda_synchronized_solve"):
                    clock.advance(10)
                clock.advance(5)
            clock.advance(50)
    value = recorder.to_dict()
    assert_reconciled(value, 80)
    assert value["components"]["curobo_planning"]["total_duration_s"] == 80
    assert value["components"]["curobo_planning"]["inclusive_sum_s"] == 100
    assert value["accounting"]["exclusive_components_s"] == {"curobo_planning": 70, "curobo_solver": 10}


def test_nested_stages_have_additive_exclusive_values():
    clock = Clock()
    with runtime_timing_scope(clock_ns=clock) as recorder:
        with measure_runtime("stage", "grasp"):
            clock.advance(5)
            with measure_runtime("stage", "pregrasp"):
                clock.advance(20)
            with measure_runtime("stage", "approach"):
                clock.advance(30)
    value = recorder.to_dict()
    assert_reconciled(value, 55)
    stages = {stage["operation"]: stage for stage in value["stages"]}
    assert stages["grasp"]["inclusive_duration_s"] == 55
    assert stages["grasp"]["exclusive_duration_s"] == 5
    assert stages["pregrasp"]["exclusive_duration_s"] == 20
    assert stages["approach"]["exclusive_duration_s"] == 30


def test_gaps_and_reported_backend_values_remain_visible_without_fake_allocation():
    clock = Clock()
    with runtime_timing_scope(clock_ns=clock) as recorder:
        clock.advance(90)
        with measure_runtime("curobo_solver", "solve"):
            clock.advance(20)
        clock.advance(90)
        record_runtime_duration("mesh_file_read", "backend_estimate", 9999)
    value = recorder.to_dict()
    assert_reconciled(value, 200)
    assert value["accounting"]["unattributed_s"] == 180
    assert value["comparison"]["curobo_solver_s"] == 20
    assert value["comparison"]["mesh_file_read_s"] == 0


def test_failure_and_finally_cleanup_preserve_error_and_total(tmp_path):
    output = tmp_path / "run"
    prepare(output)
    clock = Clock()
    error = RuntimeError("original")
    with pytest.raises(RuntimeError) as caught:
        with runtime_timing_scope(output, clock_ns=clock):
            try:
                with measure_runtime("stage", "approach"):
                    clock.advance(180)
                    raise error
            finally:
                with measure_runtime("cleanup", "close"):
                    clock.advance(20)
    assert caught.value is error
    value = json.loads((output / "runtime_timing.json").read_text())
    assert_reconciled(value, 200)
    assert value["status"] == "failed" and value["stages"][0]["status"] == "failed"


def test_full_stage_report_keeps_backend_duration_separate():
    clock = Clock()
    @timed_runtime_call("robot_execution", name_parameter="stage", result_duration_key="stage_wall_duration_s")
    def execute(stage):
        clock.advance(80)
        return {"execution_duration_s": 20, "success": True}
    with runtime_timing_scope(clock_ns=clock) as recorder:
        result = execute("pregrasp")
    assert result["stage_wall_duration_s"] == 80
    assert result["execution_duration_s"] == 20
    assert_reconciled(recorder.to_dict(), 80)


def test_positioned_solver_duration_is_attributed_inside_adapter():
    clock = Clock()
    with runtime_timing_scope(clock_ns=clock) as recorder:
        with measure_runtime("curobo_planning", "adapter"):
            clock.advance(10)
            started = clock() / 1e9
            clock.advance(20)
            record_runtime_duration("curobo_solver", "solve", 20, started_at_s=started)
            clock.advance(10)
    value = recorder.to_dict()
    assert_reconciled(value, 40)
    assert value["accounting"]["exclusive_components_s"] == {"curobo_planning": 20, "curobo_solver": 20}


def test_cli_wrapper_keeps_root_active_after_old_inner_scope(tmp_path, monkeypatch):
    from tools.run import run_microwave_task1_realworld as cli
    from gene_agent.feedback import runtime_timing as module
    clock = Clock()
    monkeypatch.setattr(module.time, "perf_counter_ns", clock)
    output = tmp_path / "run"
    def fake_main(argv):
        clock.advance(30)  # 配置/准备在旧 scope 之前。
        with runtime_timing_scope(output):
            prepare(output)
            clock.advance(150)
        assert current_runtime_timing() is not None
        clock.advance(20)  # 旧 scope 之后的控制台解析/输出。
        return 2
    monkeypatch.setattr(cli, "_main", fake_main)
    assert cli.main([]) == 2
    value = json.loads((output / "runtime_timing.json").read_text())
    assert_reconciled(value, 200)


def test_delayed_cli_binding_cannot_overwrite_an_existing_run(tmp_path):
    output = tmp_path / "run"
    prepare(output)
    previous = output / "runtime_timing.json"
    previous.write_text("historical report")
    with pytest.raises(ValueError):
        with runtime_timing_scope():
            with runtime_timing_scope(output):
                raise ValueError("output already exists")
    assert previous.read_text() == "historical report"


def test_live_snapshot_includes_open_spans_without_double_recording_them():
    clock = Clock()
    with runtime_timing_scope(clock_ns=clock) as recorder:
        with measure_runtime("stage", "pick"):
            clock.advance(75)
            with measure_runtime("curobo_solver", "solve"):
                clock.advance(5)
            live = recorder.to_dict()
            assert_reconciled(live, 80)
            assert live["accounting"]["exclusive_stages_s"] == {"pick": 80}
            assert live["components"]["stage"]["wall_duration_s"] == 80
            assert live["accounting"]["unattributed_s"] == 0
            clock.advance(20)
    final = recorder.to_dict()
    assert_reconciled(final, 100)
    assert len(final["events"]) == 2 and final["components"]["stage"]["call_count"] == 1


def test_direct_adapter_wall_time_includes_work_missing_from_backend_duration():
    import numpy as np
    from gene_agent.motion.curobo.adapter import CuroboPlannerAdapter
    from gene_agent.motion.model import (JointState, MotionOperation, MotionRequest,
                                        PlannerMode, PlanningSceneRef, PoseTarget)
    clock = Clock()
    class Planner:
        joint_names = ("q",)
        scene_snapshot_sha256 = "hash"
        scene_scope = "scene"
        planner_mode = PlannerMode.CUDA_GRAPH_BENCHMARK
        random_seed = 123
        def plan(self, *args, **kwargs):
            clock.advance(80)
            return {"success": True, "positions_rad": [[0.], [.1]],
                    "duration_s": 20., "interpolation_dt_s": .025}
    request = MotionRequest("pregrasp", MotionOperation.PLAN, "right", JointState(("q",), (0.,)),
        PoseTarget("world_tcp", np.eye(4)), PlanningSceneRef(0, "hash", "scene"), seed=123)
    with runtime_timing_scope(clock_ns=clock) as recorder:
        result = CuroboPlannerAdapter(Planner(), group="right", scene_revision=0).plan(request)
    value = recorder.to_dict()
    assert_reconciled(value, 80)
    assert value["comparison"]["curobo_planning_wall_s"] == 80
    assert result.duration_s == 20
    assert result.metrics["runtime_timing_event_id"] == value["events"][0]["event_id"]
    assert "backend_method_only" in result.metrics["duration_scope"]


def test_mesh_and_initialization_comparison_uses_union_not_nested_sum():
    clock = Clock()
    with runtime_timing_scope(clock_ns=clock) as recorder:
        with measure_runtime("mesh_preparation", "mesh"):
            clock.advance(5)
            with measure_runtime("curobo_initialization", "init"):
                clock.advance(5)
    value = recorder.to_dict()
    assert_reconciled(value, 10)
    assert value["comparison"]["scene_mesh_and_curobo_initialization_s"] == 10
