"""在线实验必须从实际 MuJoCo 状态开始，并显式区分物理与规划结果。"""

import json

import mujoco
import numpy as np
import pytest

from gene_agent.tasks.policies.microwave_task.planning.door_session import DEFAULT_CONFIG
from gene_agent.tasks.policies.microwave_task.planning.door_session import DoorDebugSession
from gene_agent.tasks.policies.microwave_task.simulation_door.session import ExperimentSession


def _config(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "fresh_scene")
    path = tmp_path / "door_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _push_profile(tmp_path):
    """合成逐角输入只检验 40° 换点契约，不依赖本机历史调参产物。"""
    profile = {
        "schema_version": 1, "robot_group": "right", "surface": "inside",
        "precontact_offset_m": .02,
        "push_waypoints": [
            {"angle_deg": angle, "hinge_fraction": 1., "top_down_distance_m": .03,
             "standoff_m": 0., "tip_x_deg": -30. if angle == 20 else -60.,
             "tip_y_deg": -75. if angle == 40 else -90., "panel_tilt_degrees": 0.}
            for angle in (20, 30, 40, 50, 60, 70, 80, 85, 90)
        ],
        "reposition_after_40": {"retreat_world_z_m": .05, "precontact_standoff_m": .03},
    }
    path = tmp_path / "push_profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return str(path)


def test_door_visual_session_can_use_verified_push_profile(tmp_path):
    """旧物理推点先进入同一 Knowledge，Viser 和 MuJoCo 才能检查相同目标。"""
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "fresh_scene")
    config["pull_angle_deg"] = 20.
    config["push_open_profile"] = _push_profile(tmp_path)
    path = tmp_path / "door_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    session = DoorDebugSession(config_path=path)
    candidate_sets = session.knowledge["push_open"].door_candidate_sets

    assert len(candidate_sets) == 9
    assert candidate_sets[0].candidates[0].parameters["tip_y_deg"] == -90.
    assert candidate_sets[1].candidates[0].parameters["tip_x_deg"] == -60.
    assert [round(np.rad2deg(item.door_angle_rad)) for item in candidate_sets[:3]] == [20, 30, 40]
    open_phases = [phase for phase in session.phases if phase.skill_id == "push_open"]
    names = [phase.name for phase in open_phases]
    after_40 = names.index("push_02")
    assert names[after_40 + 1:after_40 + 4] == [
        "retract_up", "new_precontact", "new_contact",
    ]
    assert [phase.motion_kind for phase in open_phases[after_40 + 1:after_40 + 4]] == [
        "linear_tcp", "plan", "linear_tcp",
    ]


def test_recontact_uses_actual_door_and_tcp_after_push_40(tmp_path):
    """物理换点必须从真实门角、真实夹爪位置重算，不能回放名义 40°。"""
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["scene_run"] = str(tmp_path / "fresh_scene")
    config["pull_angle_deg"] = 20.
    config["push_open_profile"] = _push_profile(tmp_path)
    path = tmp_path / "door_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = ExperimentSession("microwave_door_open", "physics", path, 0,
                                output_dir=tmp_path / "experiment")
    joint = session.runtime.model.joint("microwave_door_joint")
    actual_angle = np.deg2rad(37.3)
    session.runtime.data.qpos[joint.qposadr[0]] = actual_angle
    mujoco.mj_forward(session.runtime.model, session.runtime.data)
    session._sync_planning_from_actual()

    retract_index = next(i for i, phase in enumerate(session.planning.phases)
                         if phase.skill_id == "push_open" and phase.name == "retract_up")
    session.planning.phase_index = retract_index
    actual_tcp = session.runtime.data.site(
        session.runtime.robot_metadata["tcp_sites"][session.planning.group]
    ).xpos.copy()
    session._refresh_profile_reposition_from_actual()
    retract = session.planning.phases[retract_index]
    np.testing.assert_allclose(retract.tcp_pose_world[:3, 3],
                               actual_tcp + [0., 0., .05])

    session.planning.phase_index = retract_index + 1
    session._refresh_profile_reposition_from_actual()
    precontact = session.planning.phases[retract_index + 1]
    expected_precontact, _ = session.planning.push_open_recontact_poses(actual_angle)
    assert precontact.door_angle_rad == pytest.approx(actual_angle)
    np.testing.assert_allclose(precontact.tcp_pose_world, expected_precontact)


def test_first_handle_stage_executes_gripper_in_physics_and_saves_actual_state(tmp_path):
    session = ExperimentSession("handle_pull", "physics", _config(tmp_path), 0,
                                output_dir=tmp_path / "experiment")

    row = session.step()

    assert row["stage_id"] == "handle_pull/00_open"
    assert row["planning_success"] is True
    assert row["execution_success"] is True
    assert row["source_format"] == "mujoco_physical"
    assert row["start_state_id"] != row["end_state_id"]
    assert row["actual_end"]["robots"]["right"]["gripper"]["width_m"] == pytest.approx(
        session.planning.phases[0].gripper_width_m, abs=.01
    )
    assert (tmp_path / "experiment/stages/handle_pull_00_open.json").is_file()


def test_unsupported_physical_selection_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="物理执行尚未接入"):
        ExperimentSession("rotate_knob", "physics", _config(tmp_path), 0,
                          output_dir=tmp_path / "experiment")


def test_physical_speed_override_does_not_change_knowledge_targets(tmp_path):
    config = _config(tmp_path)
    baseline = ExperimentSession(
        "microwave_door_open", "physics", config, 0,
        output_dir=tmp_path / "baseline",
    )
    session = ExperimentSession(
        "microwave_door_open", "physics", config, 0,
        output_dir=tmp_path / "experiment",
        speed_scale_by_phase={"push_open/00_pre_contact": .25},
    )
    assert session.speed_scale_by_phase["push_open/00_pre_contact"] == .25
    assert [phase.phase_id for phase in session.planning.phases] == [
        phase.phase_id for phase in baseline.planning.phases
    ]
    assert (session.planning.phase("push_open/00_pre_contact").tcp_pose_world ==
            baseline.planning.phase("push_open/00_pre_contact").tcp_pose_world).all()

    with pytest.raises(ValueError, match="无效的物理阶段速度比例"):
        ExperimentSession(
            "microwave_door_open", "physics", config, 0,
            output_dir=tmp_path / "other_experiment",
            speed_scale_by_phase={"push_open/00_pre_contact": 1.5},
        )


def test_next_plan_uses_actual_door_state_after_execution(tmp_path, monkeypatch):
    session = ExperimentSession("handle_pull", "physics", _config(tmp_path), 0,
                                output_dir=tmp_path / "experiment")
    assert session.step()["execution_success"]
    joint = session.runtime.model.joint("microwave_door_joint")
    session.runtime.data.qpos[joint.qposadr[0]] = 0.12
    mujoco.mj_forward(session.runtime.model, session.runtime.data)

    def inspect_actual_start():
        actual_in_planner = session.planning.case.data.qpos[joint.qposadr[0]]
        assert actual_in_planner == pytest.approx(0.12)
        return {"success": False, "failure_category": "goal_ik_no_solution"}

    monkeypatch.setattr(session.planning, "plan_next", inspect_actual_start)
    row = session.step()
    assert row["planning_success"] is False
    assert row["execution_success"] is False
    assert session.runtime.data.qpos[joint.qposadr[0]] == pytest.approx(0.12)


def test_stale_trajectory_start_cannot_execute_from_actual_state(tmp_path, monkeypatch):
    session = ExperimentSession("handle_pull", "physics", _config(tmp_path), 0,
                                output_dir=tmp_path / "experiment")
    assert session.step()["execution_success"]
    before = session._observation()["state_id"]
    names = tuple(session.runtime.robot_metadata["joint_groups"]["right"])

    def stale_plan():
        session.planning.planned_joint_names = names
        session.planning.planned_q = np.full((2, len(names)), 5.0)
        return {"success": True, "operation": "plan", "interpolation_dt_s": .025}

    monkeypatch.setattr(session.planning, "plan_next", stale_plan)
    row = session.step()
    assert row["planning_success"] is False
    assert row["failure_category"] == "trajectory_start_mismatch"
    assert session._observation()["state_id"] == before


def test_physical_motion_waits_for_actual_endpoint_before_next_stage(tmp_path, monkeypatch):
    """发完控制指令不等于机械臂到位；阶段边界须检查实际关节。"""
    session = ExperimentSession("handle_pull", "physics", _config(tmp_path), 0,
                                output_dir=tmp_path / "experiment")
    assert session.step()["execution_success"]
    names = tuple(session.runtime.robot_metadata["joint_groups"]["right"])
    current = np.array([
        session.runtime.data.qpos[session.runtime.model.joint(name).qposadr[0]]
        for name in names
    ])

    def fake_plan():
        session.planning.planned_q = np.vstack((current, current))
        session.planning.planned_joint_names = names
        return {"success": True, "operation": "plan", "interpolation_dt_s": .025}

    observed = {}

    def fake_execute(**kwargs):
        observed.update(kwargs)
        return {"success": False, "failure_type": "endpoint_tracking_error",
                "endpoint_error_rad": .12}

    monkeypatch.setattr(session.planning, "plan_next", fake_plan)
    monkeypatch.setattr(session.runtime, "execute_joint_trajectory", fake_execute)
    row = session.step()

    assert observed["endpoint_tolerance_rad"] == pytest.approx(.05)
    assert observed["settle_timeout_s"] == pytest.approx(1.5)
    assert row["execution_success"] is False
    assert row["failure_category"] == "endpoint_tracking_error"


def test_planner_exception_becomes_inspectable_stage_failure(tmp_path, monkeypatch):
    session = ExperimentSession("handle_pull", "physics", _config(tmp_path), 0,
                                output_dir=tmp_path / "experiment")
    assert session.step()["execution_success"]

    def broken_plan():
        raise RuntimeError("backend device unavailable")

    monkeypatch.setattr(session.planning, "plan_next", broken_plan)
    report = session.run()
    row = session.rows[-1]
    assert report["status"] == "failed"
    assert row["failure_category"] == "planning_exception"
    assert "backend device unavailable" in row["message"]
    assert report['shared_flow']['failure_code'] == 'planning_exception'
    assert report['shared_flow']['end_state']['state_id'] == report['actual_final']['state_id']
    assert (tmp_path / "experiment" / row["traceback_ref"]).is_file()


def test_execution_exception_keeps_stage_and_actual_state(tmp_path, monkeypatch):
    session = ExperimentSession("handle_pull", "physics", _config(tmp_path), 0,
                                output_dir=tmp_path / "experiment")
    before = session._observation()["state_id"]

    def broken_gripper(**kwargs):
        raise RuntimeError("actuator mapping invalid")

    monkeypatch.setattr(session.runtime, "execute_gripper_command", broken_gripper)
    report = session.run()
    row = session.rows[-1]
    assert report["status"] == "failed"
    assert row["failure_category"] == "execution_exception"
    assert "actuator mapping invalid" in row["message"]
    assert report['shared_flow']['failure_code'] == 'execution_exception'
    assert row["start_state_id"] == before == row["end_state_id"]
    assert (tmp_path / "experiment" / row["traceback_ref"]).is_file()


def test_door_contact_stage_records_frozen_start_contact_order(tmp_path, monkeypatch):
    """推门前先固定实际门角复核，不能假定门在指垫接触前已经转动。"""
    session = ExperimentSession("push_open", "physics", _config(tmp_path), 0,
                                output_dir=tmp_path / "experiment")
    session.planning.phase_index = next(
        index for index, phase in enumerate(session.planning.phases)
        if phase.phase_id == "push_open/01_contact"
    )
    names = tuple(session.runtime.robot_metadata["joint_groups"]["right"])
    current = np.array([
        session.runtime.data.qpos[session.runtime.model.joint(name).qposadr[0]]
        for name in names
    ])

    def fake_plan():
        session.planning.planned_q = np.vstack((current, current))
        session.planning.planned_joint_names = names
        return {"success": True, "operation": "linear_tcp", "interpolation_dt_s": .025}

    monkeypatch.setattr(session.planning, "plan_next", fake_plan)
    monkeypatch.setattr(session.runtime, "execute_joint_trajectory", lambda **_kwargs:
                        {"success": False, "failure_type": "test_stop"})
    row = session.step()

    frozen = row["frozen_start_door_path_probe"]
    assert frozen["status"] == "sampled_nominal_geometry"
    assert frozen["door_start_rad"] == pytest.approx(row["actual_start"]["door_angle_rad"])
    assert frozen["door_target_rad"] == pytest.approx(frozen["door_start_rad"])
    assert frozen["first_contact_order"] in {
        "no_contact", "pad_only", "forbidden_only", "pad_before_forbidden",
        "forbidden_before_pad", "simultaneous",
    }


def test_probe_next_plans_without_physical_execution_or_progress(tmp_path, monkeypatch):
    session = ExperimentSession("push_open", "physics", _config(tmp_path), 0,
                                output_dir=tmp_path / "experiment")
    session.planning.phase_index = next(
        index for index, phase in enumerate(session.planning.phases)
        if phase.phase_id == "push_open/01_contact"
    )
    phase_index = session.planning.phase_index
    state_id = session._observation()["state_id"]
    names = tuple(session.runtime.robot_metadata["joint_groups"]["right"])
    current = np.array([
        session.runtime.data.qpos[session.runtime.model.joint(name).qposadr[0]]
        for name in names
    ])

    def fake_plan():
        session.planning.planned_q = np.vstack((current, current))
        session.planning.planned_joint_names = names
        return {"success": True, "operation": "linear_tcp", "interpolation_dt_s": .025}

    monkeypatch.setattr(session.planning, "plan_next", fake_plan)
    monkeypatch.setattr(session.runtime, "execute_joint_trajectory", lambda **_kwargs:
                        pytest.fail("只规划检查不能执行物理轨迹"))
    result = session.probe_next()

    assert result["status"] == "plan_inspected"
    assert result["planning_success"] is True
    assert result["frozen_start_door_path_probe"]["status"] == "sampled_nominal_geometry"
    assert session._observation()["state_id"] == state_id
    assert session.planning.phase_index == phase_index
    assert session.rows == []
    assert session.stopped is False
    assert (tmp_path / "experiment/probe_next.json").is_file()
    assert not list((tmp_path / "experiment/stages").glob("*.json"))


def test_probe_next_backend_exception_keeps_inspectable_evidence(tmp_path, monkeypatch):
    session = ExperimentSession("push_open", "physics", _config(tmp_path), 0,
                                output_dir=tmp_path / "experiment")
    session.planning.phase_index = next(
        index for index, phase in enumerate(session.planning.phases)
        if phase.phase_id == "push_open/01_contact"
    )
    before = session._observation()["state_id"]
    monkeypatch.setattr(session.planning, "plan_next", lambda: (_ for _ in ()).throw(
        RuntimeError("GPU solver failed")))

    result = session.probe_next()

    assert result["status"] == "planning_exception"
    assert result["planning_success"] is False
    assert "GPU solver failed" in result["message"]
    assert (tmp_path / "experiment" / result["traceback_ref"]).is_file()
    assert session._observation()["state_id"] == before
    assert session.stopped is False
