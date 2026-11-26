import json
import time

import pytest

from gene_agent.feedback.runtime_timing import (
    current_runtime_timing,
    measure_runtime,
    record_runtime_duration,
    runtime_timing_scope,
    timed_runtime,
)


def _prepare_output(path):
    path.mkdir()
    (path / "runtime_input.json").write_text("{}\n", encoding="utf-8")


def test_runtime_timing_writes_component_comparison_and_total(tmp_path):
    output = tmp_path / "run"
    _prepare_output(output)

    with runtime_timing_scope(output):
        with measure_runtime("mesh_preparation", "mesh"):
            time.sleep(.001)
        record_runtime_duration("curobo_planning", "plan", .25)
        record_runtime_duration("curobo_initialization", "solver_initialization_s", .08)
        record_runtime_duration("curobo_solver", "solver_call_including_graph_capture_s", .15)

    payload = json.loads((output / "runtime_timing.json").read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["started_at"] and payload["finished_at"]
    assert payload["total_duration_s"] > 0
    assert payload["comparison"]["curobo_planning_wall_s"] == .25
    assert payload["comparison"]["curobo_initialization_s"] == .08
    assert payload["comparison"]["dominant_mesh_or_curobo_solver"] == "curobo_solver"
    assert payload["comparison"]["dominant_recorded_component"] == "curobo_planning"
    assert payload["bottleneck"]["component"] == "curobo_planning"
    assert payload["components"]["mesh_preparation"]["call_count"] == 1


def test_nested_scope_reuses_one_recorder_and_decorator_keeps_return(tmp_path):
    output = tmp_path / "run"
    _prepare_output(output)
    seen = []

    @timed_runtime("output_dir")
    def operation(*, output_dir):
        seen.append(current_runtime_timing())
        with runtime_timing_scope(output_dir) as nested:
            assert nested is seen[0]
            record_runtime_duration("curobo_setup", "acquire", .1)
        return "unchanged"

    assert operation(output_dir=output) == "unchanged"
    assert seen[0] is not None
    payload = json.loads((output / "runtime_timing.json").read_text(encoding="utf-8"))
    assert payload["components"]["curobo_setup"]["call_count"] == 1


def test_runtime_timing_records_failure_without_swallowing_it(tmp_path):
    output = tmp_path / "run"
    _prepare_output(output)

    with pytest.raises(RuntimeError, match="boom"):
        with runtime_timing_scope(output):
            with measure_runtime("perception", "observe"):
                raise RuntimeError("boom")

    payload = json.loads((output / "runtime_timing.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"] == "RuntimeError: boom"
    assert payload["events"][0]["status"] == "failed"
