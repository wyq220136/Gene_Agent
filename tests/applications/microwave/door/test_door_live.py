import numpy as np
import pytest
import json
import mujoco
from gene_agent.tasks.policies.microwave_task.planning.door_session import DEFAULT_CONFIG, DoorDebugSession
from gene_agent.motion import PlannerMode


def test_missing_configured_scene_is_prepared_for_visual_door_session(tmp_path):
    """干净检出没有历史 run 时，门界面仍能建立当前资产的可加载场景。"""
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    scene_dir = tmp_path / "new_scene"
    config["scene_run"] = str(scene_dir)
    path = tmp_path / "door.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    session = DoorDebugSession(config_path=path)

    assert session.run_dir == scene_dir
    assert session.case.output_dir == scene_dir
    assert (scene_dir / "case_meta.json").is_file()
    assert (scene_dir / "source_scene/scene.xml").is_file()
    assert (scene_dir / "collision/collision_snapshot.json").is_file()
    assert session.phases


def test_existing_configured_scene_is_loaded_without_rebuilding(tmp_path):
    """已有 run 继续按原入口加载，并保留它自己的采样种子。"""
    from gene_agent.tasks.policies.microwave_task.planning.bowl import prepare_pick_bowl_scene

    scene_dir = tmp_path / "existing_scene"
    prepare_pick_bowl_scene(scene_dir, seed=17)
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["scene_run"] = str(scene_dir)
    path = tmp_path / "door.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    session = DoorDebugSession(config_path=path)

    assert session.run_dir == scene_dir
    assert session.case.metadata["seed"] == 17
    assert json.loads((scene_dir / "case_meta.json").read_text(encoding="utf-8"))["seed"] == 17


def test_explicit_missing_run_dir_keeps_load_error(tmp_path):
    """调用者显式提供的 run 路径写错时，不能悄悄换成新场景。"""
    scene_dir = tmp_path / "missing_explicit"

    with pytest.raises(FileNotFoundError, match="case_meta.json"):
        DoorDebugSession(run_dir=scene_dir)

    assert not scene_dir.exists()


def test_online_door_phases_match_knowledge_and_angles():
    session = DoorDebugSession()
    expected = (
        ("handle_pull", session.knowledge["handle_pull"].handle_pull_path),
        ("push_open", session.knowledge["push_open"].door_path),
        ("push_close", session.knowledge["push_close"].door_path),
    )
    for skill, path in expected:
        skill_events = {"open"} if skill == "handle_pull" else {"close", "retreat"} if skill == "push_close" else set()
        phases = [phase for phase in session.phases
                  if phase.skill_id == skill and phase.name not in skill_events]
        assert len(phases) == len(path.waypoints)
        for phase, point in zip(phases, path.waypoints, strict=True):
            np.testing.assert_allclose(phase.tcp_pose_world, point.tcp_pose_world, atol=1e-9)
            assert phase.door_angle_rad == pytest.approx(point.angle_rad)
    assert session.phases[-1].door_angle_rad == pytest.approx(0)


def test_legacy_viser_profile_keeps_separate_handle_and_reestimated_push_angles(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(
        pull_angle_deg=30.0, push_start_angle_deg=25.0,
        push_angle_deg=85.0, handle_open_width_m=0.06,
        legacy_short_retractions=False,
    )
    path = tmp_path / "door.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = DoorDebugSession(config_path=path)
    assert session.phases[0].name == "open"
    assert session.phases[0].motion_kind == "gripper"
    assert session.phases[0].gripper_width_m == pytest.approx(0.06)
    handle = [phase for phase in session.phases if phase.skill_id == "handle_pull"]
    assert [round(np.degrees(p.door_angle_rad)) for p in handle if p.name.startswith("pull_")] == [10, 20, 30]
    opening = [phase for phase in session.phases if phase.skill_id == "push_open"]
    assert [round(np.degrees(p.door_angle_rad)) for p in opening] == [25, 25, 45, 65, 85]
    assert all(not phase.name.startswith("retract_") for phase in opening)
    transition = next(p for p in session.phases if p.skill_id == "task_transition")
    transition_index = next(i for i, phase in enumerate(session.phases) if phase is transition)
    assert session.phases[transition_index - 1] is opening[-1]
    closing = [phase for phase in session.phases if phase.skill_id == "push_close"]
    assert closing[0].door_angle_rad == pytest.approx(np.deg2rad(85))


def test_close_skill_matches_old_gripper_close_and_outward_retreat(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(pull_angle_deg=30.0, push_start_angle_deg=25.0,
                  push_angle_deg=85.0, legacy_short_retractions=False)
    path = tmp_path / "door.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = DoorDebugSession(config_path=path)
    closing = [phase for phase in session.phases if phase.skill_id == "push_close"]
    assert closing[0].name == "close"
    assert closing[0].motion_kind == "gripper"
    assert closing[0].gripper_width_m == pytest.approx(0.0)
    assert all(phase.gripper_width_m == pytest.approx(0.0) for phase in closing)
    assert closing[-1].name == "retreat"
    assert closing[-1].motion_kind == "linear_tcp"
    np.testing.assert_allclose(
        closing[-1].tcp_pose_world[:3, 3] - closing[-2].tcp_pose_world[:3, 3],
        session.knowledge["push_close"].door_candidate_sets[-1].candidates[0].surface_normal_world * .03,
        atol=1e-9,
    )


def test_default_door_task_uses_task1_retreat_instead_of_old_viewer_short_retractions():
    session = DoorDebugSession()
    opening = [phase for phase in session.phases if phase.skill_id == "push_open"]
    closing = [phase for phase in session.phases if phase.skill_id == "push_close"]
    assert all(not phase.name.startswith("retract_") for phase in opening)
    assert closing[-1].name == "retreat"
    assert session.config["return_default_tcp_non_linear_fallback"] is True


def test_door_task_keeps_all_six_planner_structures_hot():
    """整轮门任务有六种不同碰撞/夹爪结构，容量四会循环淘汰。"""
    session = DoorDebugSession()
    assert session.planner_pool.max_entries >= 6


def test_open_push_reestimates_door_state_before_exporting_collision(tmp_path, monkeypatch):
    import gene_agent.tasks.policies.microwave_task.planning.door_session as door_debug

    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(pull_angle_deg=30.0, push_start_angle_deg=25.0,
                  push_angle_deg=85.0, legacy_short_retractions=False)
    path = tmp_path / "door.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    session = DoorDebugSession(config_path=path)
    session.phase_index = next(i for i, phase in enumerate(session.phases)
                               if phase.skill_id == "push_open" and phase.name == "pre_contact")
    joint = session.case.model.joint("microwave_door_joint")
    session.case.data.qpos[joint.qposadr[0]] = np.deg2rad(30)
    mujoco.mj_forward(session.case.model, session.case.data)
    seen = []

    def inspect_collision_scene(model, data, *_args, **_kwargs):
        seen.append(float(data.qpos[model.joint("microwave_door_joint").qposadr[0]]))
        raise RuntimeError("已检查导出边界")

    monkeypatch.setattr(door_debug, "export_collision_scene", inspect_collision_scene)
    with pytest.raises(RuntimeError, match="已检查导出边界"):
        session.plan_next()
    assert seen == pytest.approx([np.deg2rad(25)])


def test_handle_stage_exports_old_task1_disabled_microwave_policy(tmp_path, monkeypatch):
    import gene_agent.tasks.policies.microwave_task.planning.door_session as door_debug

    session = DoorDebugSession()
    session.phase_index = next(i for i, phase in enumerate(session.phases)
                               if phase.skill_id == "handle_pull" and phase.name == "pregrasp")
    observed = []

    def inspect_collision_scene(*_args, **kwargs):
        observed.append(kwargs)
        raise RuntimeError("已检查碰撞策略")

    monkeypatch.setattr(door_debug, "export_collision_scene", inspect_collision_scene)
    with pytest.raises(RuntimeError, match="已检查碰撞策略"):
        session.plan_next()
    assert observed[0]["disabled_body_names"] == ("microwave", "microwave_support")
    assert observed[0]["moving_body_names"] == ()


def test_task1_return_to_default_tcp_records_explicit_old_nonlinear_fallback(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import gene_agent.tasks.policies.microwave_task.planning.door_session as door_debug

    session = DoorDebugSession()
    session.config["return_default_tcp_non_linear_fallback"] = True
    session.phase_index = next(i for i, phase in enumerate(session.phases)
                               if phase.phase_id == "task/open_to_close/return_default_tcp")
    session.output_dir = tmp_path
    names = tuple(session.case.metadata["robot"]["joint_groups"][session.group])
    start = session._current_joint_positions(names)
    linear = {"success": False, "failure_category": "constraint_violation",
              "status": "linear_path_rejected", "message": "TCP轨迹未通过直线度复核"}
    calls = []

    class Planner:
        def plan(self, called_names, called_start, called_target, *, target_frame):
            calls.append((tuple(called_names), np.asarray(called_start),
                          np.asarray(called_target), target_frame))
            return {"success": True, "positions_rad": np.vstack((start, start + .001)),
                    "interpolation_dt_s": .025, "duration_s": .1}

    monkeypatch.setattr(door_debug, "export_collision_scene", lambda *_args, **_kwargs: {
        "scene_model": {}, "scope": "test", "snapshot_sha256": "test",
        "records": [], "mounting_contact_policy": None, "gripper_snapshot": None,
    })
    monkeypatch.setattr(door_debug, "load_realman_motion_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(session.planner_pool, "acquire", lambda *_args, **_kwargs:
                        SimpleNamespace(planner=Planner(), setup_duration_s=0.,
                                        cache_hit=False, rebuild_reason="test"))
    monkeypatch.setattr(door_debug, "plan_with_approximate_ik", lambda *_args, **_kwargs: dict(linear))
    row = session.plan_next()
    assert row["success"] is True
    assert row["operation"] == "plan"
    assert row["requested_operation"] == "linear_tcp"
    assert row["fallback"]["source_status"] == "linear_path_rejected"
    assert calls[0][0] == names
    np.testing.assert_allclose(calls[0][1], start)
    assert calls[0][3] == "world_tcp"


def test_close_pre_contact_retries_only_configured_fixed_pose_after_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import gene_agent.tasks.policies.microwave_task.planning.door_session as door_debug

    session = DoorDebugSession()
    session.phase_index = next(i for i, phase in enumerate(session.phases)
                               if phase.skill_id == 'push_close' and phase.name == 'pre_contact')
    session.output_dir = tmp_path
    fallback_pose = np.eye(4)
    fallback_pose[:3, 3] = [.17, .59, .23]
    session.config['close_pre_contact_fallback_pose_world'] = fallback_pose.tolist()
    original = session.phases[session.phase_index].tcp_pose_world.copy()
    names = tuple(session.case.metadata['robot']['joint_groups'][session.group])
    start = session._current_joint_positions(names)
    attempts = []

    def fake_plan(_planner, _names, _q, target, **_kwargs):
        attempts.append(np.asarray(target).copy())
        if len(attempts) == 1:
            return {'success': False, 'status': 'goal_ik_no_solution',
                    'failure_category': 'goal_ik_no_solution'}
        return {'success': True, 'positions_rad': np.vstack((start, start + .001)),
                'interpolation_dt_s': .025, 'duration_s': .1}

    monkeypatch.setattr(door_debug, 'export_collision_scene', lambda *_a, **_k: {
        'scene_model': {}, 'scope': 'test', 'snapshot_sha256': 'test',
        'records': [], 'mounting_contact_policy': None, 'gripper_snapshot': None})
    monkeypatch.setattr(door_debug, 'load_realman_motion_config', lambda *_a, **_k: {})
    monkeypatch.setattr(session.planner_pool, 'acquire', lambda *_a, **_k:
        SimpleNamespace(planner=object(), setup_duration_s=0., cache_hit=False,
                        rebuild_reason='test'))
    monkeypatch.setattr(door_debug, 'plan_with_local_rotation_search', fake_plan)
    monkeypatch.setattr(door_debug, 'plan_with_approximate_ik', fake_plan)
    row = session.plan_next()
    assert row['success'] is True
    assert row['fallback']['source_status'] == 'goal_ik_no_solution'
    assert row['fallback']['success'] is True
    assert len(attempts) == 2
    np.testing.assert_allclose(attempts[0], original)
    np.testing.assert_allclose(attempts[1], fallback_pose)
    np.testing.assert_allclose(session.phases[session.phase_index].tcp_pose_world, fallback_pose)


def test_door_session_enables_cuda_graph_by_default_and_allows_explicit_development_mode():
    accelerated = DoorDebugSession()
    diagnostic = DoorDebugSession(planner_mode=PlannerMode.DEVELOPMENT)

    assert accelerated.planner_mode is PlannerMode.CUDA_GRAPH_BENCHMARK
    assert accelerated.planner_options() == {
        "planner_mode": PlannerMode.CUDA_GRAPH_BENCHMARK
    }
    assert diagnostic.planner_mode is PlannerMode.DEVELOPMENT


def test_task1_transition_returns_to_robot_default_tcp_but_is_not_part_of_single_skill():
    session = DoorDebugSession()
    transitions = [phase for phase in session.phases if phase.skill_id == "task_transition"]
    assert [phase.phase_id for phase in transitions] == [
        "task/open_to_close/return_default_tcp"
    ]
    transition = transitions[0]
    np.testing.assert_allclose(transition.tcp_pose_world, session.default_tcp_pose_world)
    assert transition.motion_kind == "linear_tcp"
    assert transition.contact_policy == "door_contact"

    session.start_skill("push_open")
    selected = session.phases[session.phase_index:session.stop_index]
    assert selected
    assert {phase.skill_id for phase in selected} == {"push_open"}

    session.start_skill(None)
    selected = session.phases[session.phase_index:session.stop_index]
    assert any(phase.skill_id == "task_transition" for phase in selected)


def test_edit_rebuilds_world_targets_and_invalidates_old_plan():
    session = DoorDebugSession()
    original = next(p for p in session.phases if p.name == "pregrasp").tcp_pose_world.copy()
    revision = session.revision
    session.plan_revision = revision
    new_pose = session.knowledge["handle_pull"].microwave_pose_world.copy()
    new_pose[0, 3] += .025
    session.set_microwave_pose(new_pose)
    assert session.revision == revision + 1
    assert session.plan_revision is None
    phase = next(p for p in session.phases if p.name == "pregrasp")
    assert not np.allclose(phase.tcp_pose_world[:3, 3], original[:3, 3])
    edited = phase.tcp_pose_world.copy()
    edited[2, 3] += .01
    session.set_stage_override(phase.phase_id, edited)
    np.testing.assert_allclose(session.phase(phase.phase_id).tcp_pose_world, edited)
    session.set_stage_override(phase.phase_id, None)
    assert not np.allclose(session.phase(phase.phase_id).tcp_pose_world, edited)


def test_contact_targets_cannot_be_overridden():
    session = DoorDebugSession()
    contact = next(phase for phase in session.phases if phase.name == "contact")
    with pytest.raises(ValueError, match="仅可编辑"):
        session.set_stage_override(contact.phase_id, contact.tcp_pose_world)


def test_independent_push_starts_at_configured_angle():
    session = DoorDebugSession()
    session.start_skill("push_open")
    assert session.phases[session.phase_index].skill_id == "push_open"
    assert session.case.data.qpos[session.case.model.joint("microwave_door_joint").qposadr[0]] == pytest.approx(
        np.deg2rad(25))


def test_approximate_budget_is_explicit_and_contact_defaults_to_strict():
    session = DoorDebugSession()
    pre = next(p for p in session.phases if p.name == "pre_contact")
    contact = next(p for p in session.phases if p.name == "contact")
    assert session.budget_for(pre.phase_id) == {
        "max_position_error_m": .02,
        "max_rotation_error_rad": pytest.approx(np.deg2rad(10)),
    }
    assert session.budget_for(contact.phase_id) is None
    session.set_stage_budget(pre.phase_id, None)
    assert session.budget_for(pre.phase_id) is None


def test_push_waypoints_inherit_bounded_approximate_ik_but_contact_stays_strict():
    """push_* 应统一继承 push 预算，不能因阶段带编号而意外回到严格 IK。"""
    session = DoorDebugSession()
    contact = next(p for p in session.phases if p.name == "contact")
    push = next(p for p in session.phases if p.name.startswith("push_"))

    assert session.budget_for(contact.phase_id) is None
    assert session.budget_for(push.phase_id) == {
        "max_position_error_m": .02,
        "max_rotation_error_rad": pytest.approx(np.deg2rad(10)),
    }


def test_door_push_precontact_defaults_to_task1_fifty_millimetres():
    """在线 Skill 必须复用旧 Task1 已验证的 50 mm 门面法向 pre-contact。"""
    session = DoorDebugSession()
    for skill_id in ("push_open", "push_close"):
        phases = [p for p in session.phases if p.skill_id == skill_id]
        pre_contact = next(p for p in phases if p.name == "pre_contact")
        contact = next(p for p in phases if p.name == "contact")
        assert np.linalg.norm(
            pre_contact.tcp_pose_world[:3, 3] - contact.tcp_pose_world[:3, 3]
        ) == pytest.approx(.05)


def test_local_rotation_search_is_explicit_for_transition_and_push_stages():
    session = DoorDebugSession()
    pre = next(p for p in session.phases if p.name == "pre_contact")
    contact = next(p for p in session.phases if p.name == "contact")
    push = next(p for p in session.phases if p.name.startswith("push_"))
    assert session.local_rotation_search_for(pre.phase_id)["y_degrees"][:3] == [0, -5, 5]
    assert session.local_rotation_search_for(contact.phase_id)["x_degrees"][:3] == [0, -5, 5]
    assert session.local_rotation_search_for(push.phase_id)["y_degrees"][:3] == [0, -5, 5]


def test_task_transition_override_uses_world_frame_and_can_be_restored():
    session = DoorDebugSession()
    transition = next(p for p in session.phases if p.skill_id == "task_transition")
    original = transition.tcp_pose_world.copy()
    edited = original.copy()
    edited[2, 3] += .025
    session.set_stage_override(transition.phase_id, edited)
    np.testing.assert_allclose(session.phase(transition.phase_id).tcp_pose_world, edited)
    assert session.config["task_transition_overrides_world"][transition.phase_id] == edited.tolist()
    session.set_stage_override(transition.phase_id, None)
    np.testing.assert_allclose(session.phase(transition.phase_id).tcp_pose_world, original)


def test_invalid_planner_trajectory_is_reported_as_failure_in_saved_result(tmp_path, monkeypatch):
    import json
    import gene_agent.tasks.policies.microwave_task.planning.door_session as door_debug

    session = DoorDebugSession()
    session.phase_index = next(i for i, p in enumerate(session.phases) if p.name == "pregrasp")
    session.output_dir = tmp_path
    monkeypatch.setattr(door_debug, "export_collision_scene", lambda *args, **kwargs: {
            "scene_model": {}, "scope": "scene", "snapshot_sha256": "test",
        "mounting_contact_policy": None, "gripper_snapshot": None})
    monkeypatch.setattr(door_debug, "load_realman_motion_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(door_debug, "SynchronousPlanner", lambda **kwargs: object())
    monkeypatch.setattr(door_debug, "plan_with_approximate_ik", lambda *args, **kwargs: {
        "success": True, "positions_rad": [0.], "duration_s": 0.01})
    monkeypatch.setattr(door_debug, "plan_with_local_rotation_search", lambda *args, **kwargs: {
        "success": True, "positions_rad": [0.], "duration_s": 0.01})
    row = session.plan_next()
    saved = json.loads((tmp_path / "revision_001/phase_01/result.json").read_text(encoding="utf-8"))
    assert row["success"] is False
    assert saved["success"] is False
    assert saved["failure_category"] == "invalid_trajectory"


def test_plan_result_keeps_backend_diagnostics_and_stage_wall_timing(tmp_path, monkeypatch):
    import gene_agent.tasks.policies.microwave_task.planning.door_session as door_debug

    session = DoorDebugSession()
    session.phase_index = next(i for i, p in enumerate(session.phases) if p.name == "pregrasp")
    session.output_dir = tmp_path
    snapshot = {
        "scene_model": {}, "scope": "scene", "snapshot_sha256": "test",
        "mounting_contact_policy": None, "gripper_snapshot": None,
        "records": [],
    }
    monkeypatch.setattr(door_debug, "export_collision_scene", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(door_debug, "load_realman_motion_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(door_debug, "SynchronousPlanner", lambda **kwargs: object())
    backend = {
        "success": False,
        "status": "goal_ik_no_solution",
        "message": "cuRobo did not find a successful IK seed",
        "failure_category": "goal_ik_no_solution",
        # fake backend 不实际等待，因此内部耗时也设为 0；真实后端使用同一
        # perf_counter，阶段墙钟才具有可比较含义。
        "duration_s": 0.0,
        "ik_diagnostic": {
            "seed_count": 16, "successful_seed_count": 0,
            "selected_seed_index": 0,
            "seeds": [{"index": 0, "position_error_m": .1,
                       "rotation_error_rad": .1}],
            "collision_check": {"scene_collision": False, "self_collision": False,
                                "joint_limit_violation": False,
                                "colliding_obstacles": []},
            "world_obstruction_probe": {"world_collision_disabled_for_probe": True,
                                         "relaxed_ik_success": False},
        },
    }
    monkeypatch.setattr(door_debug, "plan_with_local_rotation_search",
                        lambda *args, **kwargs: dict(backend))
    row = session.plan_next()
    assert row["message"] == backend["message"]
    assert row["status"] == backend["status"]
    assert row["ik_diagnostic"]["seed_count"] == 16
    assert row["failure_summary"]["kind"] == "goal_ik_not_converged"
    assert row["timing"]["backend_s"] == pytest.approx(0.0)
    assert row["timing"]["stage_wall_s"] >= row["timing"]["backend_s"]


def test_full_sequence_builds_seekable_continuous_timeline_without_gpu(monkeypatch):
    session = DoorDebugSession()

    def fake_plan_next():
        phase = session.phases[session.phase_index]
        if phase.motion_kind == "gripper":
            session._set_gripper_width(phase.gripper_width_m)
            session.phase_index += 1
            row = {"phase_id": phase.phase_id, "success": True,
                   "operation": "gripper", "gripper_width_m": phase.gripper_width_m}
        else:
            names = tuple(session.case.metadata["robot"]["joint_groups"][session.group])
            q = session._current_joint_positions(names)
            session.planned_q = np.vstack((q, q + .001))
            session.planned_joint_names = names
            session.animation_index = 0
            session.plan_revision = session.revision
            joint = session.case.model.joint("microwave_door_joint")
            session.animation_door_start_rad = float(
                session.case.data.qpos[joint.qposadr[0]])
            row = {"phase_id": phase.phase_id, "success": True,
                   "operation": phase.motion_kind, "interpolation_dt_s": .025}
        session.results.append(row)
        return row

    monkeypatch.setattr(session, "plan_next", fake_plan_next)
    report = session.plan_sequence()
    assert report["success"] is True
    assert session.timeline is not None
    assert any(segment.skill_id == "task_transition"
               for segment in session.timeline.segments)
    assert session.timeline.duration_s > 0
    final = session.seek_timeline(session.timeline.duration_s)
    assert final.stage_id == session.timeline.segments[-1].stage_id


def test_failed_sequence_keeps_successful_prefix_for_debugging(monkeypatch):
    session = DoorDebugSession()
    calls = 0

    def fake_plan_next():
        nonlocal calls
        phase = session.phases[session.phase_index]
        calls += 1
        if calls == 2:
            row = {"phase_id": phase.phase_id, "success": False,
                   "operation": phase.motion_kind, "failure_category": "goal_ik_no_solution"}
            session.results.append(row)
            return row
        names = tuple(session.case.metadata["robot"]["joint_groups"][session.group])
        q = session._current_joint_positions(names)
        session.planned_q = np.vstack((q, q + .001))
        session.planned_joint_names = names
        session.animation_index = 0
        session.plan_revision = session.revision
        joint = session.case.model.joint("microwave_door_joint")
        session.animation_door_start_rad = float(session.case.data.qpos[joint.qposadr[0]])
        row = {"phase_id": phase.phase_id, "success": True,
               "operation": phase.motion_kind, "interpolation_dt_s": .025}
        session.results.append(row)
        return row

    monkeypatch.setattr(session, "plan_next", fake_plan_next)
    report = session.plan_sequence()
    assert report["success"] is False
    assert report["completed_segments"] == 1
    assert report["first_failure"]["failure_category"] == "goal_ik_no_solution"
    assert len(session.timeline.segments) == 1


def test_sequence_planning_can_advance_one_stage_per_ui_tick(monkeypatch):
    """在线界面每次 tick 只规划一段，才能持续显示阶段进度。"""
    session = DoorDebugSession()
    calls = []

    def fake_plan_next():
        phase = session.phases[session.phase_index]
        calls.append(phase.phase_id)
        if phase.motion_kind == "gripper":
            session._set_gripper_width(phase.gripper_width_m)
            session.phase_index += 1
            row = {"phase_id": phase.phase_id, "success": True,
                   "operation": "gripper", "gripper_width_m": phase.gripper_width_m,
                   "timing": {"stage_wall_s": .0, "planner_cache_hit": True}}
            session.results.append(row)
            return row
        names = tuple(session.case.metadata["robot"]["joint_groups"][session.group])
        q = session._current_joint_positions(names)
        session.planned_q = np.vstack((q, q + .001))
        session.planned_joint_names = names
        session.animation_index = 0
        session.plan_revision = session.revision
        joint = session.case.model.joint("microwave_door_joint")
        session.animation_door_start_rad = float(session.case.data.qpos[joint.qposadr[0]])
        row = {"phase_id": phase.phase_id, "success": True,
               "operation": phase.motion_kind, "interpolation_dt_s": .025,
               "timing": {"stage_wall_s": .5, "planner_cache_hit": True}}
        session.results.append(row)
        return row

    monkeypatch.setattr(session, "plan_next", fake_plan_next)
    started = session.begin_plan_sequence("push_close")
    assert started["done"] is False
    assert started["total_stages"] == len(
        [phase for phase in session.phases if phase.skill_id == "push_close"])

    first = session.plan_sequence_step()
    assert len(calls) == 1
    assert first["done"] is False
    assert first["completed_segments"] == 1
    assert first["last_result"]["phase_id"] == calls[0]

    report = first
    while not report["done"]:
        report = session.plan_sequence_step()
    assert report["success"] is True
    assert report["completed_segments"] == started["total_stages"]
    assert session.timeline is not None
    assert session.timeline_selection == "push_close"


def test_incremental_sequence_failure_returns_structured_summary(monkeypatch):
    session = DoorDebugSession()

    def fail():
        phase = session.phases[session.phase_index]
        row = {
            "phase_id": phase.phase_id,
            "success": False,
            "operation": phase.motion_kind,
            "failure_category": "goal_ik_no_solution",
            "failure_summary": {
                "title": "目标 IK 未收敛",
                "summary": "16 个 seed 均未达到目标位姿。",
                "evidence": ["没有场景碰撞证据", "没有自碰撞证据"],
                "suggestions": ["检查初始状态与目标位姿"],
            },
        }
        session.results.append(row)
        return row

    monkeypatch.setattr(session, "plan_next", fail)
    session.begin_plan_sequence("push_close")
    report = session.plan_sequence_step()
    assert report["done"] is True
    assert report["success"] is False
    assert report["first_failure"]["failure_summary"]["title"] == "目标 IK 未收敛"
    assert report["completed_segments"] == 0
