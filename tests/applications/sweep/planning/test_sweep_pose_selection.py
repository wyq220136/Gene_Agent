"""整组候选必须走完后续规划，且不得改变实际仿真状态。"""
from pathlib import Path
import numpy as np
from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun
from gene_agent.tasks.policies.sweep_cube_task.execution.selection import select_sweep_sequence

SCENE = Path(__file__).resolve().parents[4] / "data/realworld/brush_cube_dustpan_0"


def test_sequence_rejects_late_failure_and_never_mutates_live_state(tmp_path, monkeypatch):
    monkeypatch.setattr(SweepCubeRun, "_render", lambda *args: None)
    run = SweepCubeRun(SCENE, tmp_path / "run")
    live_q = run.data.qpos.copy()
    live_id = run.observe()["state_id"]
    calls = []

    def plan(preview, name, target, kind, duration, *, allowed_contacts, execute):
        assert execute is False
        assert preview.data is not run.data
        index = int(preview.output_dir.name)
        robot = preview.observe()["robots"][preview.group]
        q = np.asarray(robot["joint_positions_rad"])
        calls.append((index, name))
        ok = not (index == 0 and name.endswith("__sweep"))
        return {"success": ok, "execution": None, "plan": {
            "success": ok, "failure_category": None if ok else "goal_ik_no_solution",
            "joint_names": robot["joint_names"], "positions_rad": [q.tolist(), (q + .001).tolist()]}}

    monkeypatch.setattr(SweepCubeRun, "_move_curobo", plan)
    monkeypatch.setattr("gene_agent.tasks.policies.sweep_cube_task.execution.curobo.check_cartesian_brush_pan_walls", lambda *args: [])
    monkeypatch.setattr("gene_agent.tasks.policies.sweep_cube_task.execution.curobo.inspect_push_contact", lambda *args: {"success": True})
    selected, report = select_sweep_sequence(run, "cube", "cube")
    assert selected is not None
    assert report["selected_index"] == 1
    assert not report["candidates"][0]["success"]
    assert report["candidates"][1]["stages"]["retreat"]["success"]
    np.testing.assert_array_equal(run.data.qpos, live_q)
    assert run.observe()["state_id"] == live_id
