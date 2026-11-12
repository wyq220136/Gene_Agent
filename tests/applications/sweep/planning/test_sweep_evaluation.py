"""测评不能忽略无效初态或失败 seed。"""

from pathlib import Path

from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun
from benchmarks.sweep_cube_task.summary import summarize_evaluation


SCENE = Path(__file__).resolve().parents[4] / "data/realworld/brush_cube_dustpan_0"


def test_sweep_runner_defaults_to_curobo_without_planning_on_construction(tmp_path, monkeypatch):
    monkeypatch.setattr(SweepCubeRun, "_render", lambda *_args: None)
    output = tmp_path / "run"
    run = SweepCubeRun(SCENE, output)
    assert run.motion_backend == "curobo"
    assert run.report["execution_backend"] == "curobo_planning_mujoco_physical_execution"
    assert output.is_dir()


def test_summarize_evaluation_counts_all_requested_seeds():
    rows = [
        {"seed": 1, "success": True, "failure_stage": None},
        {"seed": 2, "success": False, "failure_stage": "cube_red__sweep"},
        {"seed": 3, "success": False, "failure_stage": "invalid_initial_state"},
    ]
    report = summarize_evaluation(rows)
    assert report["episodes"] == 3
    assert report["successes"] == 1
    assert report["success_rate"] == 1 / 3
    assert report["failure_counts"] == {
        "cube_red__sweep": 1, "invalid_initial_state": 1}
