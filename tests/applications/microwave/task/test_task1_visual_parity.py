"""完整微波炉纯视觉任务的顺序和状态交接回归。"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from gene_agent.tasks.policies.microwave_task.visual import (
    TASK1_STEP_ORDER, TaskState, TaskStepResult, run_task1_visual,
    write_task1_report, plan_microwave_task1_visual,
)
from gene_agent.visualization.microwave.catalog import apply_task_state, capture_task_state, door_partition_bounds, run_door_partition, state_after_insert_report, build_task1_catalog, run_rotate_knob_partition, legacy_skipped_stage_two_phases, select_legacy_push_candidate
from gene_agent.tasks.policies.microwave_task.planning.bowl import prepare_pick_bowl_scene


def test_verified_legacy85_viewer_config_uses_recorded_microwave_pose(tmp_path):
    """旧配置的历史 XY 字段不能被 Viewer 忽略后退回默认摆位。"""
    from gene_agent.tasks.policies.microwave_task.visual_session import Task1VisualSession

    # 使用最小显式输入验证旧分支，避免依赖不再跟踪的历史验收快照。
    config_path = tmp_path / "legacy85_config.json"
    config_path.write_text(json.dumps({
        "variant": "legacy_viser", "knob_profile_variant": "tip_x_5_azimuth_240",
        "object_poses_world": {"microwave": [
            [0., 1., 0., .52], [-1., 0., 0., .31],
            [0., 0., 1., .04], [0., 0., 0., 1.],
        ]},
    }), encoding="utf-8")
    session = Task1VisualSession(
        output_dir=tmp_path / "legacy85",
        config_path=config_path,
    )
    pose = np.asarray(session.catalog.configuration["microwave_pose_world"], dtype=float)
    np.testing.assert_allclose(pose[:3, 3], [0.52, 0.31, 0.04], atol=1e-9)


def _state(*, joint=0.0, door=0.0, knob=0.0, width=0.08, bowl_x=0.0,
           model_hash="same-scene"):
    bowl = np.eye(4)
    bowl[0, 3] = bowl_x
    return TaskState(
        model_sha256=model_hash, joint_names=("j1",),
        joint_positions_rad=(joint,), gripper_width_m=width,
        door_angle_rad=door, knob_angle_rad=knob,
        object_poses_world={"bowl": bowl}, attachments=(),
    )


def test_full_task_runs_legacy_order_and_hands_off_every_actual_end_state():
    """删掉一个边界或暗中重置到 home 时，后继接收到的状态会错误。"""
    expected = (
        "open_door_first", "insert_bowl", "close_door_first",
        "rotate_knob", "open_door_second", "extract_place_bowl",
        "close_door_second",
    )
    seen = []

    def adapter(name):
        def run(start):
            seen.append((name, start))
            end = _state(
                joint=start.joint_positions_rad[0] + 1.0,
                door={"open_door_first": 1.0, "close_door_first": 0.0,
                      "open_door_second": 1.0, "close_door_second": 0.0}.get(
                          name, start.door_angle_rad),
                knob=0.7853981633974483 if name == "rotate_knob" else start.knob_angle_rad,
                width=0.0 if name == "insert_bowl" else 0.08,
                bowl_x=2.0 if name == "extract_place_bowl" else
                       start.object_poses_world["bowl"][0, 3],
            )
            return TaskStepResult(name, start, end, True, ({"name": name, "success": True},))
        return run

    report = run_task1_visual(_state(), {name: adapter(name) for name in expected})

    assert TASK1_STEP_ORDER == expected
    assert report.success
    assert [name for name, _state_seen in seen] == list(expected)
    assert [state.joint_positions_rad[0] for _name, state in seen] == list(range(7))
    assert report.final_state.joint_positions_rad == (7.0,)
    assert report.final_state.door_angle_rad == 0.0
    assert report.final_state.knob_angle_rad == 0.7853981633974483
    assert report.final_state.object_poses_world["bowl"][0, 3] == 2.0
    assert report.steps[0]["start_state"]["door_angle_rad"] == 0.0
    assert report.steps[0]["end_state"]["door_angle_rad"] == 1.0
    assert report.steps[1]["start_state"]["door_angle_rad"] == 1.0


def test_task_stops_before_accepting_a_reset_or_changed_asset_frame():
    """独立适配器若从旧场景或 home 重启，不能把其轨迹拼到 Task 中。"""
    invoked = []

    def open_door(start):
        return TaskStepResult("open_door_first", start, _state(joint=1.0, door=1.0), True,
                              ({"name": "open", "success": True},))

    def insert_bowl(start):
        invoked.append("insert_bowl")
        return TaskStepResult("insert_bowl", _state(model_hash="different-asset"),
                              _state(joint=2.0), True,
                              ({"name": "pick", "success": True},))

    def should_not_run(_start):
        invoked.append("later")
        raise AssertionError("失败后不得继续规划")

    runners = {name: should_not_run for name in TASK1_STEP_ORDER}
    runners.update(open_door_first=open_door, insert_bowl=insert_bowl)
    report = run_task1_visual(_state(), runners)

    assert not report.success
    assert report.first_failure["step"] == "insert_bowl"
    assert report.first_failure["code"] == "state_discontinuity"
    assert invoked == ["insert_bowl"]
    assert report.final_state.joint_positions_rad == (1.0,)
    assert len(report.steps) == 2
    assert report.steps[-1]["status"] == "failed"


def test_task_stops_at_first_planning_failure_without_inventing_later_motion():
    """旋钮无解时必须保留前缀与原失败事实，不能把后续门和取出写成功。"""
    called = []

    def runner(name):
        def run(start):
            called.append(name)
            if name == "rotate_knob":
                return TaskStepResult(name, start, start, False, (),
                                      {"code": "goal_ik_no_solution", "message": "0/6 IK"})
            return TaskStepResult(name, start, replace(
                start, joint_positions_rad=(start.joint_positions_rad[0] + 1.0,)),
                True, ({"name": name, "success": True},))
        return run

    report = run_task1_visual(_state(), {name: runner(name) for name in TASK1_STEP_ORDER})
    assert called == list(TASK1_STEP_ORDER[:4])
    assert [step["status"] for step in report.steps] == ["planned"] * 3 + ["failed"]
    assert report.first_failure == {
        "step": "rotate_knob", "code": "goal_ik_no_solution", "message": "0/6 IK",
    }
    assert report.final_state.joint_positions_rad == (3.0,)


def test_catalog_roundtrips_robot_gripper_door_knob_and_object_pose(tmp_path):
    """下一 Skill 的真实 MuJoCo 起点必须恢复上一段的全部可见状态。"""
    case = prepare_pick_bowl_scene(tmp_path / "scene", seed=123)
    model, data = case.model, case.data
    robot = case.metadata["robot"]
    first = capture_task_state(case)
    for name in robot["joint_groups"]["right"]:
        data.qpos[model.joint(name).qposadr[0]] += 0.02
    for name in robot["grippers"]["right"]["joint_names"]:
        data.qpos[model.joint(name).qposadr[0]] = 0.01
    data.qpos[model.joint("microwave_door_joint").qposadr[0]] = 0.5
    data.qpos[model.joint("knob_0_joint").qposadr[0]] = 0.2
    bowl_joint = model.joint("bowl_free")
    data.qpos[bowl_joint.qposadr[0]] += 0.03
    import mujoco
    mujoco.mj_forward(model, data)
    changed = capture_task_state(case)

    apply_task_state(case, first)
    assert capture_task_state(case).discontinuity(first) is None
    apply_task_state(case, changed)
    assert capture_task_state(case).discontinuity(changed) is None
    assert changed.discontinuity(first) == "robot_joint_reset"


def test_door_partition_keeps_task_return_transition_before_insert():
    """旧门 cycle 的 return_default_tcp 属于放入前的开门半段。"""
    class Phase:
        def __init__(self, skill_id):
            self.skill_id = skill_id

    phases = tuple(Phase(name) for name in (
        "handle_pull", "handle_pull", "push_open", "push_open",
        "task_transition", "push_close", "push_close",
    ))
    assert door_partition_bounds(phases, "open") == (0, 5)
    assert door_partition_bounds(phases, "close") == (5, 7)


def test_legacy_85_degree_branch_omits_inside_push_and_return_to_default_tcp():
    """旧同步 Task1 直接从把手 clearance 接物体抓取，门角另作场景更新。"""
    from types import SimpleNamespace

    phases = tuple(SimpleNamespace(skill_id=name, phase_id=f"phase_{index}")
                   for index, name in enumerate((
                       "handle_pull", "handle_pull", "push_open", "push_open",
                       "task_transition", "push_close", "push_close",
                   )))
    selected = legacy_skipped_stage_two_phases(phases)
    assert [phase.skill_id for phase in selected] == [
        "handle_pull", "handle_pull", "push_close", "push_close",
    ]
    assert door_partition_bounds(selected, "open") == (0, 2)
    assert door_partition_bounds(selected, "close") == (2, 4)


def test_legacy_skip_records_external_25_to_85_degree_door_update(tmp_path):
    """把手轨迹仍止于 25°，其后只有门场景状态更新，没有虚构的推门轨迹。"""
    from types import SimpleNamespace

    case = prepare_pick_bowl_scene(tmp_path / "scene", seed=123)
    initial = capture_task_state(case)
    phases = (SimpleNamespace(skill_id="handle_pull", phase_id="handle_pull/end"),
              SimpleNamespace(skill_id="push_close", phase_id="push_close/start"))

    class FakeDoorSession:
        def __init__(self):
            self.case = case
            self.phases = phases
            self.results = []
            self.timeline = None
            self.phase_index = 0
            self.stop_index = len(phases)

        def begin_plan_sequence(self, _skill_id=None):
            self.phase_index = 0
            self._sequence_state = {"initial_qpos": case.data.qpos.copy()}

        def plan_sequence_step(self):
            joint = case.model.joint("microwave_door_joint")
            case.data.qpos[joint.qposadr[0]] = np.deg2rad(25.)
            self.results.append({"phase_id": phases[0].phase_id, "success": True})
            self.phase_index = 1
            return {"done": True, "success": True}

    result = run_door_partition(FakeDoorSession(), "open_door_first", initial,
                                part="open", skipped_stage_two_angle_deg=85.)
    assert result.success
    assert np.isclose(np.rad2deg(result.end_state.door_angle_rad), 85.)
    assert result.start_state.discontinuity(initial) is None
    assert result.stages[0]["phase_id"] == "handle_pull/end"
    assert result.stages[1]["operation"] == "external_door_state_update"
    assert result.stages[1]["from_angle_deg"] == 25.
    assert result.stages[1]["to_angle_deg"] == 85.


def test_old_push_candidate_selection_checks_first_16_then_remaining():
    """旧同步 Knowledge 每个门角先 batch 16 个，找到可达候选便停止。"""
    from types import SimpleNamespace

    candidates = []
    for index in range(20):
        pose = np.eye(4)
        pose[0, 3] = index / 100.0
        candidates.append(SimpleNamespace(candidate_id=f"rank_{index}",
                                          preference_rank=index, tcp_pose_world=pose))

    class FakePlanner:
        def __init__(self, reachable):
            self.reachable = reachable
            self.batches = []

        def solve_ik_batch(self, names, q, targets, *, diagnose_failures):
            self.batches.append([round(float(pose[0, 3]) * 100) for pose in targets])
            return tuple({"success": index in self.reachable}
                         for index in self.batches[-1])

    first = FakePlanner({13, 17})
    chosen, facts = select_legacy_push_candidate(first, ("j1",), (0.,), candidates)
    assert chosen.candidate_id == "rank_13"
    assert first.batches == [list(range(16))]
    assert facts["checked_count"] == 16

    second = FakePlanner({17})
    chosen, facts = select_legacy_push_candidate(second, ("j1",), (0.,), candidates)
    assert chosen.candidate_id == "rank_17"
    assert second.batches == [list(range(16)), list(range(16, 20))]
    assert facts["checked_count"] == 20


def test_task1_push_selection_keeps_legacy_candidate_order_when_tip_tilts_differ():
    """旧同步 Task1 未开启朝下优先；评分全为零，同批按候选原顺序取首解。"""
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation

    candidates = []
    for index in range(16):
        pose = np.eye(4)
        # 第二个可达姿态的 TCP 更朝下，但旧 Task1 没有开启该偏好。
        tilt_deg = 145.0 if index == 0 else 150.0
        pose[:3, :3] = Rotation.from_euler("x", tilt_deg, degrees=True).as_matrix()
        pose[0, 3] = index / 100.0
        candidates.append(SimpleNamespace(
            candidate_id=f"rank_{index}", preference_rank=index, tcp_pose_world=pose,
        ))

    class FakePlanner:
        def solve_ik_batch(self, names, q, targets, *, diagnose_failures):
            return tuple({"success": round(float(pose[0, 3]) * 100) in {0, 6}}
                         for pose in targets)

    assert -candidates[0].tcp_pose_world[2, 2] < -candidates[6].tcp_pose_world[2, 2]
    chosen, facts = select_legacy_push_candidate(FakePlanner(), ("j1",), (0.,), candidates)
    assert chosen.candidate_id == "rank_0"
    assert facts["preference_rank"] == 0
    assert facts["checked_count"] == 16


def test_door_partition_restores_incoming_state_after_session_reset(tmp_path):
    """门会话公开 begin 会重置机器人；Task 必须在规划前重新注入实终态。"""
    from types import SimpleNamespace

    case = prepare_pick_bowl_scene(tmp_path / "scene", seed=123)
    initial = capture_task_state(case)
    incoming = replace(
        initial,
        joint_positions_rad=tuple(value + 0.1 for value in initial.joint_positions_rad),
        door_angle_rad=1.0,
    )
    phases = tuple(SimpleNamespace(skill_id=name, phase_id=f"phase_{index}")
                   for index, name in enumerate((
                       "handle_pull", "push_open", "task_transition",
                       "push_close", "push_close",
                   )))

    class FakeDoorSession:
        def __init__(self):
            self.case = case
            self.phases = phases
            self.results = []
            self.timeline = None
            self.phase_index = 0
            self.stop_index = len(phases)
            self.starts = []

        def begin_plan_sequence(self, skill_id=None):
            apply_task_state(case, initial)
            self.phase_index = 3 if skill_id == "push_close" else 0
            self._sequence_state = {"initial_qpos": case.data.qpos.copy(), "total_stages": 2}
            return {"done": False}

        def plan_sequence_step(self):
            self.starts.append(capture_task_state(case))
            self.results.append({"phase_id": self.phases[self.phase_index].phase_id,
                                 "success": True})
            self.phase_index += 1
            if self.phase_index == self.stop_index:
                self.timeline = SimpleNamespace(duration_s=1.0)
                return {"done": True, "success": True}
            return {"done": False, "success": True}

        def seek_timeline(self, _time_s):
            return None

    session = FakeDoorSession()
    result = run_door_partition(session, "close_door_first", incoming, part="close")
    assert result.success
    assert len(result.stages) == 2
    assert session.starts[0].discontinuity(incoming) is None
    assert result.start_state.discontinuity(incoming) is None


def test_insert_handoff_uses_actual_retreat_endpoint_and_released_bowl_pose():
    """放入结束不能用标称 TCP 或原桌面 Bowl pose 充当下一轮关门初态。"""
    start = _state(joint=0.1, door=1.0, width=0.007)
    released = np.eye(4)
    released[0, 3] = 0.25
    report = {
        "success": True,
        "actual_released_object_world": released.tolist(),
        "stages": [
            {"name": "release", "status": "planned", "details": {"width_m": 0.08}},
            {"name": "retreat_2", "status": "planned",
             "result": {"goal_state": {"joint_names": ["j1"], "positions_rad": [1.2]}}},
        ],
    }
    end = state_after_insert_report(start, report)
    assert end.joint_positions_rad == (1.2,)
    assert end.gripper_width_m == 0.08
    assert end.door_angle_rad == 1.0
    assert end.object_poses_world["bowl"][0, 3] == 0.25
    assert end.attachments == ()


def test_task_report_persists_first_failure_and_last_verified_state(tmp_path):
    """GPU 探针失败时报告必须可复核，不能只在终端打印异常。"""
    import json

    report = run_task1_visual(_state(), {})
    path = write_task1_report(tmp_path / "task_report.json", report)
    saved = json.loads(path.read_text())
    assert saved["success"] is False
    assert saved["first_failure"]["step"] == "open_door_first"
    assert saved["first_failure"]["code"] == "adapter_unavailable"
    assert saved["final_state"]["joint_positions_rad"] == [0.0]
    assert saved["final_state"]["object_poses_world"]["bowl"][0][3] == 0.0


def test_catalog_uses_one_mujoco_asset_for_all_task_boundaries(tmp_path):
    """门和碗不得各建一份无法交接物体 pose 的场景。"""
    catalog = build_task1_catalog(tmp_path / "task")
    assert catalog.initial_state.model_sha256 == capture_task_state(
        catalog.door_session.case,
    ).model_sha256
    assert set(catalog.adapters) == set(TASK1_STEP_ORDER)
    assert catalog.case.model is catalog.door_session.case.model
    assert catalog.case.data is catalog.door_session.case.data
    assert catalog.configuration["door_push_candidate_selection"] == "progressive_16_then_remaining"


def test_legacy_catalog_is_explicit_85_degree_skipped_branch(tmp_path):
    """旧基线与显式推门配置分离，不修改 Knowledge 候选或调试器算法。"""
    catalog = build_task1_catalog(tmp_path / "legacy", variant="legacy_viser")
    assert catalog.configuration["variant"] == "legacy_viser"
    assert catalog.configuration["door_pull_angle_deg"] == 25.
    assert catalog.configuration["door_push_start_angle_deg"] == 85.
    assert catalog.configuration["door_push_angle_deg"] == 85.
    assert all(phase.skill_id != "push_open" for phase in catalog.door_session.phases)
    assert all(phase.skill_id != "task_transition" for phase in catalog.door_session.phases)
    assert catalog.configuration["door_push_candidate_selection"] == "legacy85_skip_open_static_close"


def test_task_catalog_applies_shared_table_before_first_state_and_wires_extract(tmp_path, monkeypatch):
    """旧桌面须在整个 Task 之前原位生效，取出不得中途克隆场景。"""
    import gene_agent.tasks.policies.microwave_task.planning.extract as object_extract_transfer

    catalog = build_task1_catalog(tmp_path / "task", variant="legacy_viser")
    table = catalog.case.model.geom("scene_table")
    facts = catalog.configuration["shared_table_alignment"]
    assert facts["model_identity_preserved"]
    assert np.allclose(table.pos, facts["legacy_table_center_world_m"])
    assert np.allclose(2 * table.size, facts["legacy_table_size_xyz_m"])
    assert catalog.case.revision == facts["scene_revision"]
    assert catalog.initial_state.discontinuity(capture_task_state(catalog.case)) is None
    incoming = replace(
        catalog.initial_state,
        joint_positions_rad=tuple(value + .01 for value in catalog.initial_state.joint_positions_rad),
    )
    seen = []

    def fake_extract(case, state, *, seed, output_dir):
        seen.append((case.model, case.data, state, capture_task_state(case), seed, output_dir))
        return TaskStepResult("extract_place_bowl", state, state, True,
                              ({"name": "release", "status": "planned"},))

    monkeypatch.setattr(object_extract_transfer, "run_extract_bowl_from_current_case",
                        fake_extract)
    result = catalog.adapters["extract_place_bowl"](incoming)
    assert result.success
    assert seen[0][0] is catalog.case.model and seen[0][1] is catalog.case.data
    assert seen[0][2].discontinuity(incoming) is None
    assert seen[0][3].discontinuity(incoming) is None
    assert seen[0][4] == 123


def test_explicit_tip5_azimuth240_profile_is_saved_without_overwriting_default(tmp_path):
    """成功验收摆位只产生 run 内显式 profile；旧 0/6 默认可重跑。"""
    import json

    catalog = build_task1_catalog(
        tmp_path / "task", variant="legacy_viser",
        knob_profile_variant="tip_x_5_azimuth_240",
    )
    task = json.loads((catalog.output_dir / "knob_task_config.json").read_text())
    profile = json.loads((catalog.output_dir / "knob_grasp_profile.json").read_text())
    params = profile["profiles"]["microwave_knob_0"]["parameters"]
    assert params["azimuth_deg"]["values"] == [240]
    assert params["tip_x_deg"]["values"] == [5]
    assert task["grasp_profile_path"] == str(catalog.output_dir / "knob_grasp_profile.json")
    assert catalog.configuration["knob_profile_variant"] == "tip_x_5_azimuth_240"
    assert task["grasp_profile_path"] != "data/task_1/knowledge/grasp_profiles.microwave_calibrated.json"


def test_catalog_applies_scene_edits_to_one_case_before_first_state(tmp_path):
    """在线编辑的微波炉、Bowl、桌面必须落到同一共享模型与碰撞快照。"""
    import json
    import gene_agent.visualization.microwave.catalog as visual_catalog
    from gene_agent.visualization.viser_sim.model import SceneEdits

    source = json.loads((visual_catalog.ROOT / "data/task_1/skills/door_task.json").read_text())
    microwave = np.asarray(source["microwave_pose_world"], dtype=float)
    microwave[:2, 3] = (.52, .31)
    bowl = np.eye(4)
    bowl[:3, 3] = (.28, .13, -.078)
    table = np.eye(4)
    table[:3, :3] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    table[:3, 3] = (.11, .31, -.21601678639650346)
    edits = SceneEdits(
        object_poses_world={"microwave": microwave, "bowl": bowl, "table": table},
        table_size_xy_m=(.62, .58),
    )
    catalog = build_task1_catalog(
        tmp_path / "task", variant="legacy_viser",
        knob_profile_variant="tip_x_5_azimuth_240", scene_edits=edits,
        scene_revision=7,
    )
    case = catalog.case
    assert case is catalog.door_session.case
    assert case.revision == 9  # 构建版本 + 旧桌面修正 + 在线桌面编辑。
    assert np.allclose(case.data.body("microwave").xpos, microwave[:3, 3])
    assert np.allclose(case.data.body("bowl").xpos, bowl[:3, 3])
    assert np.allclose(case.model.geom("scene_table").pos, table[:3, 3])
    assert np.allclose(case.data.geom("scene_table").xmat.reshape(3, 3), table[:3, :3])
    assert np.allclose(2 * case.model.geom("scene_table").size[:2], (.62, .58))
    assert np.allclose(case.metadata["objects"]["sample_poses"]["table"], table)
    assert np.allclose(case.metadata["objects"]["table"]["size_xy_m"], (.62, .58))
    assert catalog.configuration["scene_revision"] == 9
    assert catalog.initial_state.discontinuity(capture_task_state(case)) is None
    task = json.loads((catalog.output_dir / "knob_task_config.json").read_text())
    assert np.allclose(task["microwave_pose_world"], microwave)


def test_task_knob_adapter_hands_off_same_model_robot_door_bowl_and_gripper(tmp_path, monkeypatch):
    """第一次关门终态必须成为旋钮 IK 的实际输入，成功后读取同模型实终态。"""
    import gene_agent.tasks.policies.microwave_task.planning.knob as rotate_knob_skill

    case = prepare_pick_bowl_scene(tmp_path / "scene", seed=123)
    initial = capture_task_state(case)
    incoming = replace(
        initial,
        joint_positions_rad=tuple(value + 0.01 for value in initial.joint_positions_rad),
        gripper_width_m=0.0, door_angle_rad=0.0,
    )
    observed = []

    def fake_knob(output, *, case, task_path):
        observed.append((case.model, case.data, capture_task_state(case)))
        joint = case.model.joint("knob_0_joint")
        case.data.qpos[joint.qposadr[0]] = 0.45
        return {"success": True, "stages": [{"name": "rotate_26", "status": "planned"}]}

    monkeypatch.setattr(rotate_knob_skill, "run_rotate_knob_0_planning", fake_knob)
    result = run_rotate_knob_partition(case, incoming, output_dir=tmp_path / "knob")
    assert result.success
    assert observed[0][0] is case.model and observed[0][1] is case.data
    assert observed[0][2].discontinuity(incoming) is None
    assert result.start_state.discontinuity(incoming) is None
    assert result.end_state.knob_angle_rad == 0.45
    assert result.end_state.gripper_width_m == incoming.gripper_width_m
    assert result.end_state.door_angle_rad == incoming.door_angle_rad


def test_task_entry_saves_a_fail_closed_report_when_first_adapter_fails(tmp_path, monkeypatch):
    """CLI 读取在线会话的最后真实状态，不回退到上一完整分区。"""
    import json
    from types import SimpleNamespace
    import gene_agent.tasks.policies.microwave_task.visual_session as task1_visual_session
    from gene_agent.tasks.policies.microwave_task.visual import Task1VisualReport

    initial = _state()
    partial = replace(initial, gripper_width_m=.04)

    class FakeSession:
        def __init__(self, *, output_dir, config_path):
            self.catalog = SimpleNamespace(output_dir=output_dir / 'revision_000')
            self.catalog.output_dir.mkdir()

        def begin_plan(self, selection):
            assert selection == 'microwave_task1'

        def plan_step(self):
            write_task1_report(self.catalog.output_dir / 'task1_visual_result.json',
                Task1VisualReport(False, ({'step': 'open_door_first', 'status': 'failed'},),
                    partial, {'code': 'goal_ik_no_solution', 'message': '无解'},
                    {'variant': 'task1_visual_explicit'}))
            return {'done': True}

    monkeypatch.setattr(task1_visual_session, 'Task1VisualSession', FakeSession)
    report = plan_microwave_task1_visual(tmp_path / "run")
    saved = json.loads((tmp_path / "run/task1_visual_result.json").read_text(encoding='utf-8'))
    assert not report.success
    assert saved["first_failure"]["code"] == "goal_ik_no_solution"
    assert saved["configuration"]["variant"] == "task1_visual_explicit"
    assert saved['final_state']['gripper_width_m'] == .04
    assert len(saved["steps"]) == 1


def test_task_entry_forwards_explicit_pose_and_knob_profile(tmp_path, monkeypatch):
    """显式调整只经配置入口传递，失败报告同时保存实际采用的值。"""
    import json
    from types import SimpleNamespace
    import gene_agent.tasks.policies.microwave_task.visual_session as task1_visual_session
    from gene_agent.tasks.policies.microwave_task.visual import Task1VisualReport

    seen = []

    class FakeSession:
        def __init__(self, *, output_dir, config_path):
            config = json.loads(config_path.read_text(encoding='utf-8'))
            seen.append(config)
            self.catalog = SimpleNamespace(output_dir=output_dir / 'revision_000')
            self.catalog.output_dir.mkdir()

        def begin_plan(self, selection):
            assert selection == 'microwave_task1'

        def plan_step(self):
            write_task1_report(self.catalog.output_dir / 'task1_visual_result.json',
                Task1VisualReport(False, (), _state(),
                    {'code': 'goal_ik_no_solution'},
                    {'variant': seen[0]['variant'],
                     'knob_profile_variant': seen[0]['knob_profile_variant'],
                     'microwave_xy_override_m': [.52, .29]}))
            return {'done': True}

    monkeypatch.setattr(task1_visual_session, 'Task1VisualSession', FakeSession)
    report = plan_microwave_task1_visual(
        tmp_path / "run", variant="legacy_viser",
        knob_profile_variant="tip_x_5_azimuth_240", microwave_xy_m=(.52, .29),
    )
    assert seen[0]['variant'] == 'legacy_viser'
    assert seen[0]['knob_profile_variant'] == 'tip_x_5_azimuth_240'
    assert np.asarray(seen[0]['object_poses_world']['microwave'])[:2, 3].tolist() == [.52, .29]
    assert report.configuration["microwave_xy_override_m"] == [.52, .29]
