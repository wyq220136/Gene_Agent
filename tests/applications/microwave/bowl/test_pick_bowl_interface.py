"""临时抓碗探针验证接口编排，不复制 Knowledge 或 planner。"""

import numpy as np
import pytest


def test_pick_contact_approach_uses_load_capable_stage_gain():
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.pick import (
        PICK_CLOSE_POSITION_KP_SCALE,
        PICK_EXECUTION_POSITION_KP_SCALE_BY_STAGE,
    )

    assert PICK_EXECUTION_POSITION_KP_SCALE_BY_STAGE == {
        "pregrasp": 1.0,
        "approach": 1.0,
        "lift": 1.0,
    }
    assert PICK_CLOSE_POSITION_KP_SCALE == pytest.approx(1.0)
import json

from gene_agent.tasks.policies.microwave_task.simulation_bowl.pick import make_pregrasp_pose, run_probe
from gene_agent.feedback.grasp_contact import classify_grasp_contacts
from gene_agent.simulation.pick_backend import run_simulation_pick as run_pick_skill
from gene_agent.skills.library.pick.model import GraspSettings, PICK_STAGE_ORDER


class FakeRuntime:
    def __init__(self):
        self.index = 0
        self.gripper_calls = []

    @property
    def current_id(self):
        return f"state-{self.index}"

    def query(self, **kwargs):
        return {
            "state_id": self.current_id,
            "objects": {"bowl": {"T_world_object": np.eye(4).tolist()}},
            "robots": {"right": {"T_world_tcp": np.eye(4).tolist()}},
        }

    def execute_gripper_command(self, **kwargs):
        assert kwargs["planned_state_id"] == self.current_id
        self.gripper_calls.append(kwargs)
        self.index += 1
        return {"success": True, "end_state_id": self.current_id}


def test_bowl_closing_stops_after_bilateral_contact_and_keeps_actual_state(monkeypatch):
    import pytest
    from gene_agent.tasks.policies.microwave_task.simulation_bowl import pick as module
    descriptor = {'pad_geom_names': ['p', 'n'], 'collision_geom_names': ['p', 'n'],
                  'fingers': {'positive': {'pad_geom_name': 'p'},
                              'negative': {'pad_geom_name': 'n'}}}
    class Runtime(FakeRuntime):
        robot_metadata = {'grippers': {'right': descriptor}}
        def query(self, **kwargs):
            return {'state_id': self.current_id, 'robots': {'right': {
                'gripper': {'width_m': .041 - .001 * self.index}}}}
        def execute_gripper_command(self, **kwargs):
            result = super().execute_gripper_command(**kwargs)
            return {**result, 'sim_duration_s': .15, 'samples': [{
                'time_s': .15, 'reference_width_m': kwargs['width_m'],
                'actual_width_m': kwargs['width_m']}], 'replay_samples': []}
    runtime = Runtime()
    def diagnostics(*args, **kwargs):
        return {'contacts': [{'geom_names': [pad, 'wall'], 'body_names': ['finger', 'bowl']}
                             for pad in (['p', 'n'] if runtime.index >= 3 else ['p'])]}
    monkeypatch.setattr(module, 'query_diagnostics', diagnostics)
    result = module.close_bowl_on_contact(runtime, group='right', width_m=.007,
                                          planned_state_id='state-0', duration_s=1.)
    assert len(runtime.gripper_calls) == 3
    assert runtime.gripper_calls[-1]['width_m'] == pytest.approx(.038)
    assert result['start_state_id'] == 'state-0'
    assert result['end_state_id'] == 'state-3'
    assert result['sim_duration_s'] == pytest.approx(.45)
    assert result['samples'][-1]['time_s'] == pytest.approx(.45)
    assert result['close_attempts'][-1]['bilateral_pad_contact']
    with pytest.raises(ValueError, match='过期'):
        module.close_bowl_on_contact(runtime, group='right', width_m=.007,
                                    planned_state_id='state-0', duration_s=1.)
    assert len(runtime.gripper_calls) == 3


def test_pregrasp_moves_opposite_tcp_approach_axis():
    final = np.eye(4)
    final[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    final[:3, 3] = [.3, .2, .1]

    pregrasp = make_pregrasp_pose(final, retreat_m=.08)

    np.testing.assert_allclose(pregrasp[:3, :3], final[:3, :3])
    np.testing.assert_allclose(pregrasp[:3, 3], [.22, .2, .1])


def test_executor_consumes_shared_skill_settings_and_stage_contract(tmp_path):
    runtime = FakeRuntime()
    targets = {}
    final = np.eye(4)
    final[2, 3] = .2

    def move(stage, target, planned_state_id):
        assert planned_state_id == runtime.current_id
        targets[stage] = np.asarray(target)
        runtime.index += 1
        return {"success": True, "end_state_id": runtime.current_id}

    result = run_pick_skill(
        runtime, tmp_path,
        candidate={"candidate_id": "c", "T_world_tcp": final.tolist()},
        motion_stage=move, object_id="bowl", group="right",
        settings=GraspSettings(approach_distance_m=.03, lift_distance_m=.08,
                               open_width_m=.06, close_width_m=.01),
        diagnostics=lambda: {"contacts": []}, evidence_writer=lambda *args: None,
    )

    assert [row["name"] for row in result["stage_contract"]] == list(PICK_STAGE_ORDER)
    assert list(result["stages"]) == ["open", "pregrasp", "approach", "close"]
    assert targets["pregrasp"][2, 3] == .17
    assert runtime.gripper_calls[0]["width_m"] == .06
    assert runtime.gripper_calls[1]["width_m"] == .01


def test_probe_consumes_latest_state_for_every_stage(tmp_path):
    runtime = FakeRuntime()
    calls = []
    failures = []

    def move(stage, target, planned_state_id):
        assert planned_state_id == runtime.current_id
        calls.append((stage, planned_state_id, np.asarray(target)))
        runtime.index += 1
        return {"success": True, "end_state_id": runtime.current_id}

    result = run_probe(
        runtime,
        tmp_path,
        candidate={"candidate_id": "candidate", "T_world_tcp": np.eye(4).tolist()},
        motion_stage=move,
        diagnostics=lambda: {"contacts": []},
        evidence_writer=lambda *args, **kwargs: None,
        failure_handler=lambda stage, kind, details: failures.append((stage, kind, details)),
    )

    assert [item[0] for item in calls] == ["pregrasp", "approach"]
    assert [item[1] for item in calls] == ["state-1", "state-2"]
    assert runtime.gripper_calls[0]["planned_state_id"] == "state-0"
    assert runtime.gripper_calls[0]["width_m"] == 0.08
    assert runtime.gripper_calls[1]["planned_state_id"] == "state-3"
    assert result["stage_state_ids"] == {
        "initial": "state-0", "opened": "state-1", "pregrasp": "state-2",
        "approach": "state-3", "closed": "state-4",
    }
    assert failures[0][:2] == ("close_contact", "postcondition_failed")
    assert failures[0][2]["bilateral_pad_contact"] is False
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["failure_stage"] == "close_contact"


def test_probe_forwards_hold_position_gain(tmp_path, monkeypatch):
    forwarded = {}

    def fake_run_pick_skill(*args, **kwargs):
        forwarded.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(
        "gene_agent.tasks.policies.microwave_task.simulation_bowl.pick.run_pick_skill",
        fake_run_pick_skill,
    )
    run_probe(
        FakeRuntime(), tmp_path,
        candidate={"candidate_id": "candidate", "T_world_tcp": np.eye(4).tolist()},
        motion_stage=lambda *args: {"success": True},
        diagnostics=lambda: {"contacts": []},
        evidence_writer=lambda *args, **kwargs: None,
        hold_position_kp_scale=2.5,
    )
    assert forwarded["hold_position_kp_scale"] == pytest.approx(2.5)


def test_probe_records_open_and_close_during_execution(tmp_path):
    runtime = FakeRuntime()
    runtime.model = object()
    runtime.data = object()
    events = []
    class Recorder:
        def begin_stage(self, stage, model, data):
            events.append(("begin", stage))
        def observer(self, stage):
            events.append(("observer", stage))
            return lambda *args: None
        def end_stage(self, stage, model, data):
            events.append(("end", stage))
    def move(stage, target, planned_state_id):
        runtime.index += 1
        return {"success": True, "end_state_id": runtime.current_id}
    run_probe(
        runtime, tmp_path, candidate={"candidate_id": "c", "T_world_tcp": np.eye(4).tolist()},
        motion_stage=move, diagnostics=lambda: {"contacts": []},
        evidence_writer=lambda *args: None, visual_recorder=Recorder(),
    )
    assert events == [("begin", "open"), ("observer", "open"), ("end", "open"),
                      ("begin", "close"), ("observer", "close"), ("end", "close")]
    assert all(callable(call["step_observer"]) for call in runtime.gripper_calls)


def test_contact_classification_distinguishes_bilateral_pads_and_structure():
    descriptor = {
        "pad_geom_names": ["positive_pad", "negative_pad"],
        "collision_geom_names": ["positive_pad", "negative_pad", "positive_shank"],
        "fingers": {
            "positive": {"pad_geom_name": "positive_pad"},
            "negative": {"pad_geom_name": "negative_pad"},
        },
    }
    contacts = [
        {"geom_names": ["positive_pad", "bowl_wall_0"], "body_names": ["finger", "bowl"]},
        {"geom_names": ["bowl_wall_1", "negative_pad"], "body_names": ["bowl", "finger"]},
        {"geom_names": ["positive_shank", "bowl_wall_2"], "body_names": ["finger", "bowl"]},
        {"geom_names": ["positive_pad", "table"], "body_names": ["finger", "world"]},
    ]

    result = classify_grasp_contacts(contacts, descriptor, object_body="bowl")

    assert result["bilateral_pad_contact"] is True
    assert result["pad_sides"] == {"positive": True, "negative": True}
    assert [item["geom_names"] for item in result["undesired_gripper_contacts"]] == [
        ["positive_shank", "bowl_wall_2"]
    ]
    assert result["success"] is False


def test_experiment_uses_verified_candidate_and_explicit_scene_perturbation(tmp_path, monkeypatch):
    from gene_agent.tasks.policies.microwave_task.simulation_bowl import pick as probe
    captured = {}

    def fake_run(runtime, output, **kwargs):
        captured["candidate_id"] = kwargs["candidate"]["candidate_id"]
        captured["retreat_m"] = kwargs["retreat_m"]
        return {"success": True}

    monkeypatch.setattr(probe, "run_probe", fake_run)
    probe.run_experiment(tmp_path / "run", seed=17, perturb=True)
    metadata = json.loads((tmp_path / "run/model/objects/scene_metadata.json").read_text())
    assert captured == {"candidate_id": "bowl_ring_00_00", "retreat_m": .03}
    assert metadata["seed"] == 17
    assert metadata["perturb"] is True
    assert metadata["perturbations"]["bowl"]["dx_m"] != 0


def test_experiment_record_video_attaches_recorder_and_saves_manifest(tmp_path, monkeypatch):
    from gene_agent.tasks.policies.microwave_task.simulation_bowl import pick as probe
    captured = {}
    class Recorder:
        def __init__(self, output, **kwargs):
            captured["output"] = output
            captured["options"] = kwargs
        def finalize(self):
            return {"combined_video": {"path": "skill_execution.mp4"}}
    def fake_run(runtime, output, **kwargs):
        captured["recorder"] = kwargs["visual_recorder"]
        captured["bridge_recorder"] = kwargs["motion_stage"].visual_recorder
        return {"success": True, "failure_stage": None}
    monkeypatch.setattr(probe, "run_probe", fake_run)
    monkeypatch.setattr("gene_agent.visualization.trajectory.TrajectoryRecorder", Recorder)
    result = probe.run_experiment(tmp_path / "run", record_video=True)
    assert captured["recorder"] is captured["bridge_recorder"]
    assert captured["options"]["combine_mp4"] is True
    assert result["visual_evidence"]["combined_video"]["path"] == "skill_execution.mp4"
    assert json.loads((tmp_path / "run/result.json").read_text(encoding="utf-8")) == result
