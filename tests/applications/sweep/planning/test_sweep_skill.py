import numpy as np

from gene_agent.knowledge.sweep import SweepLayout
from gene_agent.skills.library.sweep.skill import run_sweep_skill
from gene_agent.tasks.policies.sweep_cube_task.simulation import approximate_ik_budget_for_stage

from pathlib import Path as _ContactPath
from gene_agent.skills.library.sweep import SweepContacts
SWEEP_CONTACTS_PROFILE = SweepContacts.load(_ContactPath(__file__).resolve().parents[4] / 'config/tasks/sweep/contacts.json')
GRIP_CONTACTS = SWEEP_CONTACTS_PROFILE.grip_pairs
TABLE_CONTACTS = SWEEP_CONTACTS_PROFILE.table_pairs


def test_sweep_approximate_ik_only_for_non_contact_stages():
    assert approximate_ik_budget_for_stage("cube_red__retreat") == {
        "max_position_error_m": .003, "max_rotation_error_rad": .08}
    assert approximate_ik_budget_for_stage("cube_red__move_behind_cube") is not None
    assert approximate_ik_budget_for_stage("pregrasp") is not None
    assert approximate_ik_budget_for_stage("cube_red__sweep") is None
    assert approximate_ik_budget_for_stage("grasp") is None


def test_sweep_skill_uses_actual_brush_state_and_stage_contact_policy():
    layout = SweepLayout(np.array([0., 1.]), np.array([.4, -.1, .03]),
                         np.array([.4, .2, .03]), np.array([.4, .22, 0.]))
    calls = []

    def move(name, target, path_kind, duration_s, *, allowed_contacts):
        calls.append((name, target.copy(), path_kind, tuple(allowed_contacts)))
        return {"success": True}

    result = run_sweep_skill(
        layout, contacts=SWEEP_CONTACTS_PROFILE, sweep_targets=('cube',), initial_brush_z=.03, move_stage=move,
        target_for_brush=lambda xyz: xyz.copy(),
        current_brush_xyz=lambda: np.array([.4, .21, .03]))

    assert result["success"]
    assert [row[0] for row in calls] == [
        "move_behind_cube", "lower_to_contact", "sweep", "retreat"]
    assert calls[0][2] == "joint"
    assert all(row[2] == "linear_tcp" for row in calls[1:])
    assert calls[2][1][1] == .2
    assert calls[3][1][2] == .12
    assert ("brush", "cube") in calls[2][3]
    assert ("brush", "cube") in calls[1][3]
    assert ("brush", "dustpan") not in calls[1][3]


def test_sweep_skill_stops_on_failed_stage():
    layout = SweepLayout(np.array([0., 1.]), np.zeros(3), np.ones(3), np.ones(3))
    calls = []

    def move(name, target, path_kind, duration_s, *, allowed_contacts):
        calls.append(name)
        return {"success": name != "lower_to_contact", "failure_type": "collision"}

    result = run_sweep_skill(
        layout, contacts=SWEEP_CONTACTS_PROFILE, sweep_targets=('cube',), initial_brush_z=0., move_stage=move,
        target_for_brush=lambda xyz: xyz,
        current_brush_xyz=lambda: np.zeros(3))
    assert not result["success"]
    assert result["failure_stage"] == "lower_to_contact"
    assert calls == ["move_behind_cube", "lower_to_contact"]


def test_sweep_skill_labels_each_target_and_allows_target_block_contacts():
    layout = SweepLayout(np.array([0., 1.]), np.zeros(3), np.ones(3), np.ones(3))
    calls = []

    def move(name, target, path_kind, duration_s, *, allowed_contacts):
        calls.append((name, tuple(allowed_contacts)))
        return {"success": True}

    result = run_sweep_skill(
        layout, contacts=SWEEP_CONTACTS_PROFILE, initial_brush_z=0., move_stage=move,
        target_for_brush=lambda xyz: xyz,
        current_brush_xyz=lambda: np.zeros(3),
        stage_prefix="cube_red__", sweep_targets=("cube", "cube_red", "cube_blue"))
    assert result["success"]
    assert calls[0][0] == "cube_red__move_behind_cube"
    assert calls[2][0] == "cube_red__sweep"
    assert ("brush", "cube_blue") in calls[2][1]


def test_sweep_skill_uses_selected_contact_pose_and_actual_brush_height():
    layout = SweepLayout(np.array([0., 1.]), np.array([.4, -.1, .03]),
                         np.array([.4, .2, .03]), np.array([.4, .22, 0.]))
    calls = []
    positions = iter((np.array([.4, -.1, .041]), np.array([.4, .2, .041])))

    def move(name, target, path_kind, duration_s, *, allowed_contacts):
        calls.append((name, np.asarray(target).copy()))
        return {"success": True}

    result = run_sweep_skill(
        layout, contacts=SWEEP_CONTACTS_PROFILE, sweep_targets=('cube',), initial_brush_z=.03, move_stage=move,
        target_for_brush=lambda xyz: xyz.copy(),
        contact_target_for_brush=lambda xyz: xyz + np.array([0., 0., .011]),
        current_brush_xyz=lambda: next(positions))
    assert result["success"]
    assert np.isclose(calls[1][1][2], .041)
    assert np.isclose(calls[2][1][2], .041)


def test_settling_merge_preserves_raw_execution_and_real_end_state():
    from gene_agent.tasks.policies.sweep_cube_task.simulation import _merge_contact_settling
    motion = {"success": True, "start_state_id": "a", "end_state_id": "b", "sim_duration_s": 1.,
              "replay_samples": [{"state_id": "b"}]}
    hold = {"success": False, "failure_type": "hold_failed", "start_state_id": "b",
            "end_state_id": "c", "sim_duration_s": .5, "replay_samples": [{"state_id": "c"}]}
    result = _merge_contact_settling(motion, hold)
    assert result["end_state_id"] == result["replay_samples"][-1]["state_id"] == "c"
    assert result["sim_duration_s"] == 1.5
    assert result["failure_type"] == "hold_failed" and not result["success"]
    assert motion["end_state_id"] == "b" and len(motion["replay_samples"]) == 1


def test_sweep_defers_only_unchanged_preflight_and_retries_after_progress():
    from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun
    from gene_agent.tasks.policies.sweep_cube_task.policy import SweepPolicy
    run = object.__new__(SweepCubeRun)
    controller = SweepPolicy(run)
    run.report = {'failure_stage': None}
    run._all_goals = lambda: {'targets': {name: {'success': False} for name in ('blue', 'red')}}
    state = [0]
    completed = []
    calls = []
    run.observe = lambda: {'state_id': str(state[0])}

    def attempt(name, label):
        calls.append((name, label))
        if name == 'blue' and 'red' not in completed:
            run.report['failure_stage'] = label + '_align__sequence_screen'
            return False
        completed.append(name)
        state[0] += 1
        return True

    controller._sweep_target_with_alignment = attempt
    assert controller._sweep_pending_targets(['blue', 'red'])
    assert completed == ['red', 'blue']
    assert len(set(label for _, label in calls)) == 3
    assert run.report['deferred_targets'][0]['target_name'] == 'blue'
    assert run.report['failure_stage'] is None


def test_sweep_order_stops_when_all_preflights_blocked():
    from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun
    from gene_agent.tasks.policies.sweep_cube_task.policy import SweepPolicy
    run = object.__new__(SweepCubeRun)
    controller = SweepPolicy(run)
    run.report = {'failure_stage': None}
    run._all_goals = lambda: {'targets': {name: {'success': False} for name in ('blue', 'red')}}
    run.observe = lambda: {'state_id': 'unchanged'}
    calls = []
    def attempt(name, label):
        calls.append(name)
        run.report['failure_stage'] = label + '__sequence_screen'
        return False
    controller._sweep_target_with_alignment = attempt
    assert not controller._sweep_pending_targets(['blue', 'red'])
    assert calls == ['blue', 'red']
    assert run.report['failure_stage'] == 'red__sequence_screen'


def test_sweep_order_does_not_skip_execution_failure_or_changed_state():
    from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun
    from gene_agent.tasks.policies.sweep_cube_task.policy import SweepPolicy
    for failure_suffix, moved in [('__sweep', False), ('__sequence_screen', True)]:
        run = object.__new__(SweepCubeRun)
        controller = SweepPolicy(run)
        run.report = {'failure_stage': None}
        run._all_goals = lambda: {'targets': {name: {'success': False} for name in ('blue', 'red')}}
        state = [0]
        calls = []
        run.observe = lambda: {'state_id': str(state[0])}
        def attempt(name, label):
            calls.append(name)
            state[0] += int(moved)
            run.report['failure_stage'] = label + failure_suffix
            return False
        controller._sweep_target_with_alignment = attempt
        assert not controller._sweep_pending_targets(['blue', 'red'])
        assert calls == ['blue']


def test_sweep_deferred_target_already_in_pan_is_observed_not_reswept():
    from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun
    from gene_agent.tasks.policies.sweep_cube_task.policy import SweepPolicy
    run = object.__new__(SweepCubeRun)
    controller = SweepPolicy(run)
    run.report = {'failure_stage': None}
    state = [0]
    calls = []
    run.observe = lambda: {'state_id': str(state[0])}
    run._all_goals = lambda: {'targets': {name: {'success': state[0] > 0} for name in ('blue', 'red')}}
    def attempt(name, label):
        calls.append(name)
        if name == 'blue':
            run.report['failure_stage'] = label + '__sequence_screen'
            return False
        state[0] += 1
        return True
    controller._sweep_target_with_alignment = attempt
    assert controller._sweep_pending_targets(['blue', 'red'])
    assert calls == ['blue', 'red']
    assert run.report['executed_target_order'] == ['red']
    evidence = run.report['observed_completed_targets'][0]
    assert evidence['target_name'] == 'blue' and evidence['goal']['success']
    assert evidence['state_id'] == '1'


def test_sweep_can_defer_after_completed_alignment_with_pure_next_preflight():
    from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun
    from gene_agent.tasks.policies.sweep_cube_task.policy import SweepPolicy
    run = object.__new__(SweepCubeRun)
    controller = SweepPolicy(run)
    run.report = {'failure_stage': None, 'knowledge': {}}
    state = [0]
    completed = []
    run.observe = lambda: {'state_id': str(state[0])}
    run._all_goals = lambda: {'targets': {name: {'success': False} for name in ('blue', 'red')}}
    def attempt(name, label):
        if name == 'blue' and not completed:
            state[0] += 1  # 先前align已成功并撤离，入盘预检未运动。
            run.report['failure_stage'] = label + '__sequence_screen'
            run.report['knowledge'][label] = {'sequence_screen': {
                'evidence_mode': 'kinematic_preview_only', 'start_state_id': str(state[0])}}
            return False
        completed.append(name)
        state[0] += 1
        return True
    controller._sweep_target_with_alignment = attempt
    assert controller._sweep_pending_targets(['blue', 'red'])
    assert completed == ['red', 'blue']



def test_sweep_gravity_compensation_copies_scene_and_only_changes_robot(tmp_path):
    import hashlib, json, mujoco
    from pathlib import Path
    from gene_agent.tasks.policies.sweep_cube_task.simulation import scene_with_robot_gravity_compensation
    scene = Path(__file__).resolve().parents[4] / 'data/realworld/brush_cube_dustpan_0'
    source = scene / 'scene.xml'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    meta = json.loads((scene / 'robot/robot_metadata.json').read_text())
    target = tmp_path / 'scene.xml'
    scene_with_robot_gravity_compensation(source, target, meta)
    model = mujoco.MjModel.from_xml_path(str(target))
    original = mujoco.MjModel.from_xml_path(str(source))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert model.nq == original.nq and model.ngravcomp > 0
    for name in ('brush', 'cube', 'cube_red', 'cube_blue', 'dustpan'):
        assert model.body_gravcomp[model.body(name).id] == 0
    for name in ('link_2', 'link_7', 'pika_gripper_base_link'):
        assert model.body_gravcomp[model.body(name).id] == 1
    np.testing.assert_allclose(model.body_pos, original.body_pos)
    np.testing.assert_allclose(model.geom_pos, original.geom_pos)
