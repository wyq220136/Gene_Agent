"""检查点必须保存实际积分状态，并拒绝错误模型或 Skill。"""

import json

import pytest

from gene_agent.tasks.policies.microwave_task.planning.door_session import DEFAULT_CONFIG
from gene_agent.tasks.policies.microwave_task.simulation_door.session import ExperimentSession


def _config(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "fresh_scene")
    path = tmp_path / "door_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_checkpoint_restores_actual_state_and_next_stage_in_new_run(tmp_path):
    config = _config(tmp_path)
    source = ExperimentSession("handle_pull", "physics", config, 0,
                               output_dir=tmp_path / "source")
    first = source.step()
    assert first["execution_success"]
    source_trace = source._trace_for("handle_pull")
    assert source_trace.samples
    checkpoint = source.save_checkpoint(tmp_path / "after_open")

    resumed = ExperimentSession("handle_pull", "physics", config, 0,
                                output_dir=tmp_path / "resumed")
    resumed.restore_checkpoint(checkpoint)
    assert resumed._observation()["state_id"] == first["end_state_id"]
    assert resumed.planning.phase_index == 1
    assert resumed.planning.phases[1].phase_id == "handle_pull/01_pregrasp"
    assert resumed.planning.case.data.qpos.tolist() == pytest.approx(
        resumed.runtime.data.qpos.tolist()
    )
    assert resumed.rows == []
    assert resumed.prefix_stages == ["handle_pull/00_open"]
    restored_trace = resumed._trace_for("handle_pull")
    assert restored_trace.samples == source_trace.samples
    assert restored_trace.contact_pairs == source_trace.contact_pairs


def test_checkpoint_rejects_changed_physics_without_partial_restore(tmp_path):
    config = _config(tmp_path)
    source = ExperimentSession("handle_pull", "physics", config, 0,
                               output_dir=tmp_path / "source")
    checkpoint = source.save_checkpoint(tmp_path / "home")
    resumed = ExperimentSession("handle_pull", "physics", config, 0,
                                output_dir=tmp_path / "resumed")
    model = resumed.runtime.model
    model.dof_damping[0] += .01
    before = resumed._observation()["state_id"]
    with pytest.raises(ValueError, match="模型指纹"):
        resumed.restore_checkpoint(checkpoint)
    assert resumed._observation()["state_id"] == before


def test_checkpoint_rejects_other_skill_selection(tmp_path):
    config = _config(tmp_path)
    source = ExperimentSession("handle_pull", "physics", config, 0,
                               output_dir=tmp_path / "source")
    checkpoint = source.save_checkpoint(tmp_path / "home")
    resumed = ExperimentSession("push_open", "physics", config, 0,
                                output_dir=tmp_path / "resumed")
    with pytest.raises(ValueError, match="selection"):
        resumed.restore_checkpoint(checkpoint)


def test_checkpoint_restores_runtime_changed_weld_parameters(tmp_path):
    config = _config(tmp_path)
    source = ExperimentSession("handle_pull", "physics", config, 0,
                               output_dir=tmp_path / "source")
    source.runtime.model.eq_data[0, 3] += .01
    expected = source.runtime.model.eq_data.copy()
    checkpoint = source.save_checkpoint(tmp_path / "changed_weld")

    resumed = ExperimentSession("handle_pull", "physics", config, 0,
                                output_dir=tmp_path / "resumed")
    resumed.restore_checkpoint(checkpoint)
    assert (resumed.runtime.model.eq_data == expected).all()
