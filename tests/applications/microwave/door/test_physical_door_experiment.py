"""独立关门 Skill 必须从显式开门初态开始，并以真实闭门角验收。"""

import json
from pathlib import Path

import numpy as np
import pytest
import mujoco

import gene_agent.tasks.policies.microwave_task.simulation_door.session as experiment
from gene_agent.tasks.policies.microwave_task.simulation_door.session import ExperimentSession, _door_contact_policy_failure
from gene_agent.tasks.policies.microwave_task.simulation_door.execution import classify_door_contact
from gene_agent.skills.library.door.sequence import DoorPhase
from tools.viewers.recording.record_physical_door_video import _apply_run_scene_pose
from gene_agent.feedback.state import model_fingerprint
from gene_agent.simulation.execution import temporary_position_actuator_kp_scale


@pytest.mark.parametrize("skill", ["push_open", "push_close"])
@pytest.mark.parametrize("part", ["negative_carriage", "positive_shank", "negative_shank"])
def test_user_approved_gripper_parts_can_push_panel_but_not_touch_in_precontact(skill, part):
    """用户允许夹爪指部推门，不等于允许接近时提前碰门或允许机械臂撞门。"""
    config_path = Path(__file__).resolve().parents[4] / "data/task_1/simulation/door_physical_open.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    panel = "microwave_door_panel_collision"
    geom = f"right_gripper_{part}"
    allowed = set(config["physics_push_allowed_gripper_contact_geoms"])
    classification = classify_door_contact(geom, panel, set(), (panel,), allowed)
    assert classification == "gripper_target"
    contacts = {f"{geom} ↔ {panel}": {"classification": classification, "step_count": 1}}
    push = DoorPhase(f"{skill}/push", skill, "push", np.eye(4),
                     .5, 0., "linear_tcp", "door_contact")
    approach = DoorPhase(f"{skill}/pre_contact", skill, "pre_contact", np.eye(4),
                         .5, 0., "plan", "avoid_door")
    assert _door_contact_policy_failure(push, contacts) is None
    assert _door_contact_policy_failure(approach, contacts) == "unexpected_precontact_door_contact"
    assert classify_door_contact("right_forearm", panel, set(), (panel,), allowed) == "forbidden"
    assert classify_door_contact(geom, "microwave_handle_grip_collision", set(),
                                 (panel,), allowed) == "forbidden"


def test_default_physical_door_cycle_uses_verified_close_parameters(tmp_path):
    """默认入口须读取已通过物理关门的 Knowledge 与阶段控制参数。"""
    config = Path(__file__).resolve().parents[4] / "data/task_1/simulation/door_physical_open.json"
    session = ExperimentSession(
        "microwave_door_cycle", "physics", config, 0,
        output_dir=tmp_path / "physical_door_cycle",
    )

    close_30_degrees = session.planning.knowledge["push_close"].door_candidate_sets[3]
    assert close_30_degrees.candidates[0].parameters["hinge_fraction"] == \
        pytest.approx(0.55)
    assert session.arm_position_kp_scale_by_phase["push_close/01_pre_contact"] == \
        pytest.approx(4.0)
    assert "right_r_inspire_jaw_base_link_collision_2" in \
        session._trace_for("push_close").allowed_gripper_geom_names


def test_standalone_push_close_uses_explicit_open_door_reset(tmp_path):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    config["push_angle_deg"] = 80.
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    session = ExperimentSession("push_close", "physics", path, 0,
                                output_dir=tmp_path / "run")

    assert session.planning.phases[session.planning.phase_index].phase_id == \
        "push_close/00_close"
    assert session.initialization["kind"] == "explicit_reset_scenario"
    assert session._observation()["door_angle_rad"] == pytest.approx(np.deg2rad(80.))
    row = session.step()
    assert row["stage_id"] == "push_close/00_close"
    assert row["execution_success"] is True
    assert row["actual_end"]["robots"]["right"]["gripper"]["width_m"] < .01


def test_push_close_final_check_targets_closed_door(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession("push_close", "physics", path, 0,
                                output_dir=tmp_path / "run")
    captured = []
    monkeypatch.setattr(experiment, "evaluate_checks", lambda specs, _samples:
                        captured.extend(specs) or [{"status": "fail", "code": "test"}])

    session._finish_door_skill("push_close")

    assert captured[0]["kind"] == "door_angle"
    assert captured[0]["target_rad"] == pytest.approx(0.)


def test_handle_pull_requires_actual_handle_contact_for_goal(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession("handle_pull", "physics", path, 0,
                                output_dir=tmp_path / "run")
    captured = []
    monkeypatch.setattr(experiment, "evaluate_checks", lambda specs, _samples:
                        captured.extend(specs) or [{"status": "fail", "code": "test"}])

    session._finish_door_skill("handle_pull")

    assert captured[0]["required_contact_key"] == "pad_handle_contact_count"


def test_stage_contact_policy_rejects_precontact_and_nonpad_contacts():
    pose = np.eye(4)
    def phase(name, policy):
        return DoorPhase(f"push_close/{name}", "push_close", name, pose,
                         np.pi / 2, 0., "linear_tcp", policy)

    pad = {"right_pad ↔ door": {"classification": "pad_target", "step_count": 2}}
    base = {"right_base ↔ door": {"classification": "forbidden", "step_count": 100}}
    assert _door_contact_policy_failure(phase("pre_contact", "avoid_door"), pad) == \
        "unexpected_precontact_door_contact"
    assert _door_contact_policy_failure(phase("pre_contact", "avoid_door"), base) == \
        "unexpected_precontact_door_contact"
    assert _door_contact_policy_failure(phase("contact", "door_contact"), base) == \
        "forbidden_robot_door_contact"
    assert _door_contact_policy_failure(phase("contact", "door_contact"), pad) is None


def test_push_contact_accepts_configured_finger_root_but_not_other_robot_parts():
    phase = DoorPhase("push_open/02_push_01", "push_open", "push_01",
                      np.eye(4), np.deg2rad(40.), 0., "linear_tcp", "door_contact")
    jaw = {"finger_root ↔ door": {"classification": "gripper_target", "step_count": 3}}
    arm = {"forearm ↔ door": {"classification": "forbidden", "step_count": 1}}

    assert _door_contact_policy_failure(phase, jaw) is None
    assert _door_contact_policy_failure(phase, {**jaw, **arm}) == \
        "forbidden_robot_door_contact"


def test_target_contact_classification_distinguishes_pad_finger_root_and_arm():
    panel = "microwave_door_panel_collision"
    allowed = (panel,)
    pads = {"right_positive_pad", "right_negative_pad"}
    extra = {"right_jaw_base_1"}
    assert classify_door_contact("right_positive_pad", panel, pads, allowed, extra) == \
        "pad_target"
    assert classify_door_contact("right_jaw_base_1", panel, pads, allowed, extra) == \
        "gripper_target"
    assert classify_door_contact("right_forearm", panel, pads, allowed, extra) == \
        "forbidden"
    assert classify_door_contact("right_jaw_base_1", "microwave_handle_grip_collision",
                                 pads, allowed, extra) == "forbidden"


def test_physical_push_start_keeps_actual_door_angle(tmp_path):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession("microwave_door_open", "physics", path, 0,
                                output_dir=tmp_path / "run")
    address = session._door_joint.qposadr[0]
    actual = np.deg2rad(23.4)
    session.runtime.data.qpos[address] = actual
    mujoco.mj_forward(session.runtime.model, session.runtime.data)

    session._refresh_push_start_from_actual()

    assert session.runtime.data.qpos[address] == pytest.approx(actual)
    assert session.planning.phases[session.planning.phase_index].door_angle_rad == \
        pytest.approx(actual)


def test_push_contact_config_applies_only_to_push_and_goal_check(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    config["physics_push_allowed_gripper_contact_geoms"] = [
        "right_r_inspire_jaw_base_link_collision_1",
    ]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession("microwave_door_open", "physics", path, 0,
                                output_dir=tmp_path / "run")

    assert session._trace_for("handle_pull").allowed_gripper_geom_names == frozenset()
    assert session._trace_for("push_open").allowed_gripper_geom_names == frozenset(
        {"right_r_inspire_jaw_base_link_collision_1"}
    )
    captured = []
    monkeypatch.setattr(experiment, "evaluate_checks", lambda specs, _samples:
                        captured.extend(specs) or [{"status": "fail", "code": "test"}])
    session._finish_door_skill("push_open")
    assert captured[0]["required_contact_key"] == "allowed_target_contact_count"


def test_stage_boundary_rejects_unexplained_door_angle_jump(tmp_path):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession("microwave_door_open", "physics", path, 0,
                                output_dir=tmp_path / "run")
    session.rows.append({"actual_end": {"door_angle_rad": 0.}})
    address = session._door_joint.qposadr[0]
    session.runtime.data.qpos[address] = np.deg2rad(25.)
    mujoco.mj_forward(session.runtime.model, session.runtime.data)

    row = session.step()

    assert row["failure_category"] == "door_angle_discontinuity"
    assert session.stopped is True


def test_task_transition_only_accepts_initial_light_pad_departure_contact():
    phase = DoorPhase("task/open_to_close/return_default_tcp", "task_transition",
                      "return_default_tcp", np.eye(4), np.pi / 2, 0.,
                      "linear_tcp", "door_contact")
    pad = {"right_pad ↔ door": {
        "classification": "pad_target", "step_count": 2,
        "normal_impulse_ns": .05, "last_contact_time_s": 10.12,
    }}
    assert _door_contact_policy_failure(
        phase, pad, stage_start_time_s=10., door_motion_rad=np.deg2rad(.5),
    ) is None
    late = {"right_pad ↔ door": {**pad["right_pad ↔ door"],
                                  "last_contact_time_s": 10.3}}
    assert _door_contact_policy_failure(
        phase, late, stage_start_time_s=10., door_motion_rad=np.deg2rad(.5),
    ) == "forbidden_task_transition_contact"
    base = {"right_base ↔ door": {**pad["right_pad ↔ door"],
                                   "classification": "forbidden"}}
    assert _door_contact_policy_failure(
        phase, base, stage_start_time_s=10., door_motion_rad=np.deg2rad(.5),
    ) == "forbidden_task_transition_contact"
    assert _door_contact_policy_failure(
        phase, pad, stage_start_time_s=10., door_motion_rad=np.deg2rad(3.),
    ) == "forbidden_task_transition_contact"


def test_handle_contact_keeps_existing_contact_rule_during_physical_validation():
    handle = DoorPhase("handle_pull/03_close", "handle_pull", "close", np.eye(4),
                       0., 0., "gripper", "handle_contact")
    panel_brush = {
        "right_gripper_negative_pad ↔ microwave_door_panel_collision": {
            "classification": "forbidden", "step_count": 53,
            "max_penetration_depth_m": 0.,
        },
    }

    # 门板接触仍写进逐阶段记录；是否完成拉把手由真实把手接触和门角决定。
    assert _door_contact_policy_failure(handle, panel_brush) is None
    closing = DoorPhase("handle_pull/03_close", "handle_pull", "close", np.eye(4),
                        0., 0., "gripper", "avoid_door")
    assert _door_contact_policy_failure(closing, panel_brush) is None
    releasing = DoorPhase("handle_pull/07_release", "handle_pull", "release",
                          np.eye(4), 0., 0., "gripper", "avoid_door")
    assert _door_contact_policy_failure(releasing, panel_brush) is None


def test_physical_video_restores_edited_microwave_and_support_pose(tmp_path):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    config["microwave_pose_world"][0][3] += .005
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession("handle_pull", "physics", path, 0,
                                output_dir=tmp_path / "run")

    replay_model = mujoco.MjModel.from_xml_path(
        str(session.planning.case.metadata["xml_path"])
    )
    _apply_run_scene_pose(replay_model, session.output_dir)
    replay_data = mujoco.MjData(replay_model)
    mujoco.mj_forward(replay_model, replay_data)
    for body_name in ("microwave", "microwave_support"):
        assert np.allclose(replay_data.body(body_name).xpos,
                           session.runtime.data.body(body_name).xpos)
        assert np.allclose(replay_data.body(body_name).xmat,
                           session.runtime.data.body(body_name).xmat)


def test_physical_arm_gain_scale_does_not_change_gripper_or_other_arm(tmp_path):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    config["physics_arm_position_kp_scale"] = 2.0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    session = ExperimentSession("handle_pull", "physics", path, 0,
                                output_dir=tmp_path / "run")
    model = session.runtime.model
    gain = lambda name: float(model.actuator_gainprm[model.actuator(name).id, 0])

    assert gain("right_joint_1_position") == pytest.approx(300.)
    assert gain("right_joint_7_position") == pytest.approx(300.)
    assert gain("left_joint_1_position") == pytest.approx(150.)
    assert gain("right_gripper_positive_slide_position") == pytest.approx(3000.)
    assert model.actuator_biasprm[model.actuator("right_joint_1_position").id, 1] \
        == pytest.approx(-300.)
    assert model.actuator_forcerange[model.actuator("right_joint_1_position").id] \
        == pytest.approx([-60., 60.])


@pytest.mark.parametrize("scale", [0., -1., float("nan")])
def test_invalid_physical_arm_gain_scale_is_rejected(tmp_path, scale):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    config["physics_arm_position_kp_scale"] = scale
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="kp 比例"):
        ExperimentSession("handle_pull", "physics", path, 0,
                          output_dir=tmp_path / "run")


@pytest.mark.parametrize("selection", ["microwave_door_cycle", "microwave_door_close"])
def test_full_door_selection_keeps_close_after_actual_open_angle_refresh(
        tmp_path, selection):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession(selection, "physics", path, 0,
                                output_dir=tmp_path / "run")
    assert session.planning.phases[session.planning.stop_index - 1].skill_id == \
        "push_close"

    address = session._door_joint.qposadr[0]
    session.runtime.data.qpos[address] = np.deg2rad(22.5)
    mujoco.mj_forward(session.runtime.model, session.runtime.data)
    session._refresh_push_start_from_actual()

    assert session.planning.phases[session.planning.phase_index].phase_id == \
        "push_open/00_pre_contact"
    assert session.planning.phases[session.planning.stop_index - 1].skill_id == \
        "push_close"
    assert session.runtime.data.qpos[address] == pytest.approx(np.deg2rad(22.5))


def test_close_knowledge_starts_from_actual_door_angle_without_moving_physics(tmp_path):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession("microwave_door_cycle", "physics", path, 0,
                                output_dir=tmp_path / "run")
    address = session._door_joint.qposadr[0]
    session.runtime.data.qpos[address] = np.deg2rad(88.7)
    mujoco.mj_forward(session.runtime.model, session.runtime.data)

    session._refresh_close_start_from_actual()

    phase = session.planning.phases[session.planning.phase_index]
    assert phase.skill_id == "push_close"
    assert phase.door_angle_rad == pytest.approx(np.deg2rad(88.7))
    assert session.runtime.data.qpos[address] == pytest.approx(np.deg2rad(88.7))


def test_task_transition_rejects_all_door_contact():
    phase = DoorPhase("task/open_to_close/return_default_tcp", "task_transition",
                      "return_default_tcp", np.eye(4), np.pi / 2, 0.,
                      "linear_tcp", "door_contact")
    pad = {"right_pad ↔ door": {"classification": "pad_target", "step_count": 1}}

    assert _door_contact_policy_failure(phase, pad) == \
        "forbidden_task_transition_contact"


def test_close_knowledge_can_be_tuned_before_physical_cycle(tmp_path):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession("microwave_door_cycle", "physics", path, 0,
                                output_dir=tmp_path / "run")
    before = next(phase.tcp_pose_world.copy() for phase in session.planning.phases
                  if phase.phase_id == "push_close/01_pre_contact")

    applied = session.apply_parameters({
        "knowledge_state": {"push_close": {"door_push": {"tip_x_offset_deg": -10.}}}
    })

    after = next(phase.tcp_pose_world for phase in session.planning.phases
                 if phase.phase_id == "push_close/01_pre_contact")
    assert applied["changed_skills"] == ["push_close"]
    assert not np.allclose(before[:3, :3], after[:3, :3])
    assert session.planning.phases[session.planning.phase_index].skill_id == \
        "handle_pull"


def test_physical_close_uses_same_progressive_candidate_selection_as_viser(
        tmp_path, monkeypatch):
    import gene_agent.visualization.microwave.catalog as visual_catalog

    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession("push_close", "physics", path, 0,
                                output_dir=tmp_path / "run")
    assert session.step()["stage_id"] == "push_close/00_close"

    selected = []
    def select(planning):
        selected.append(planning.phases[planning.phase_index].phase_id)
        return {"success": True, "candidate_id": "right_close:test_candidate",
                "preference_rank": 15}

    monkeypatch.setattr(visual_catalog, "_select_task1_push_phase", select)
    monkeypatch.setattr(session.planning, "plan_next", lambda: {
        "success": False, "operation": "plan", "failure_category": "test_stop",
    })

    row = session.step()

    assert selected == ["push_close/01_pre_contact"]
    assert row["candidate_selection"]["candidate_id"] == \
        "right_close:test_candidate"


def test_temporary_arm_gain_restores_exact_model_after_exception(tmp_path):
    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession("push_close", "physics", path, 0,
                                output_dir=tmp_path / "run")
    model = session.runtime.model
    names = session.planning.case.metadata["robot"]["joint_groups"]["right"]
    original = model_fingerprint(model)

    with pytest.raises(RuntimeError, match="probe"):
        with temporary_position_actuator_kp_scale(model, names, 2.0):
            assert model_fingerprint(model) != original
            assert model.actuator_gainprm[model.actuator("right_joint_4_position").id, 0] \
                == pytest.approx(600.)
            raise RuntimeError("probe")

    assert model_fingerprint(model) == original


def test_physical_kp_patch_applies_only_to_future_motion_stage(tmp_path, monkeypatch):
    import gene_agent.visualization.microwave.catalog as visual_catalog

    source = Path(__file__).resolve().parents[4] / (
        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "scene")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession("push_close", "physics", path, 0,
                                output_dir=tmp_path / "run")
    with pytest.raises(ValueError, match="无效的阶段 kp"):
        session.apply_parameters({"physics_arm_position_kp_scale_by_phase": {
            "push_close/00_close": 4.0,
        }})
    patch = {"physics_arm_position_kp_scale_by_phase": {
        "push_close/01_pre_contact": 4.0,
    }}
    session.apply_parameters(patch)
    assert session.step()["stage_id"] == "push_close/00_close"
    with pytest.raises(ValueError, match="已执行"):
        session.apply_parameters({"physics_arm_position_kp_scale_by_phase": {
            "push_close/00_close": 4.0,
        }})

    monkeypatch.setattr(visual_catalog, "_select_task1_push_phase", lambda _p: {
        "success": True, "candidate_id": "probe", "preference_rank": 0,
    })
    names = session.planning.case.metadata["robot"]["joint_groups"]["right"]
    start = np.asarray([session.runtime.data.qpos[session.runtime.model.joint(n).qposadr[0]]
                        for n in names])
    def plan():
        session.planning.planned_q = np.asarray([start, start])
        session.planning.planned_joint_names = names
        return {"success": True, "operation": "plan", "interpolation_dt_s": .025}
    monkeypatch.setattr(session.planning, "plan_next", plan)
    monkeypatch.setattr(session, "_door_path_probes", lambda *_a: {})
    seen = []
    def execute(**kwargs):
        seen.append(kwargs["position_kp_scale"])
        return {"success": True, "status": "completed"}
    monkeypatch.setattr(session.runtime, "execute_joint_trajectory", execute)

    row = session.step()

    assert row["stage_id"] == "push_close/01_pre_contact"
    assert row["physical_arm_position_kp_scale"] == pytest.approx(4.0)
    assert seen == [2.0]
