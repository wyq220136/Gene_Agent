"""Skill 规划的确定性阶段测试；不依赖 cuRobo、GPU 或 MuJoCo。"""

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.grasp.model import GraspProposal
from gene_agent.motion.model import (
    JointState, JointTrajectory, MotionFailure, MotionFailureKind,
    MotionOperation, MotionResult, PlannerMode, PlanningSceneRef,
)
from gene_agent.skills import GraspSkill, PickSkill, SkillContext, GraspSettings, load_grasp_settings


def _proposal(name, x=0.3):
    pose = np.eye(4)
    pose[:3, 3] = [x, 0.1, 0.25]
    return GraspProposal(name, "ring", "rim", pose, 0.012, {})


def _context():
    return SkillContext("right", JointState(("j1",), (0.0,)), PlanningSceneRef(0, "hash", "scene"), "bowl")


class Planner:
    def __init__(self, *, linear_success=False, fail_pregrasp_x=()):
        self.requests = []
        self.linear_success = linear_success
        self.fail_pregrasp_x = fail_pregrasp_x

    def fk(self, state):
        pose = np.eye(4)
        pose[0, 3] = state.positions_rad[0]
        return pose

    def _result(self, request, *, success=True):
        if not success:
            kind = (MotionFailureKind.INVALID_REQUEST if request.operation is MotionOperation.LINEAR_TCP
                    else MotionFailureKind.GOAL_IK_NO_SOLUTION)
            summary = "当前 backend 不支持 LINEAR_TCP operation" if request.operation is MotionOperation.LINEAR_TCP else "无解"
            return MotionResult(request.request_id, request.operation, False, "fake", PlannerMode.DEVELOPMENT, 0.01,
                                failure=MotionFailure(kind, summary))
        state = JointState(("j1",), (float(request.target.matrix[0, 3]),))
        trajectory = None
        if request.operation is not MotionOperation.IK:
            trajectory = JointTrajectory(("j1",), (0.0, 1.0), ((request.start.positions_rad[0],), state.positions_rad))
        return MotionResult(request.request_id, request.operation, True, "fake", PlannerMode.DEVELOPMENT, 0.01,
                            goal_state=state, trajectory=trajectory)

    def evaluate_candidates(self, requests):
        self.requests.extend(requests)
        return tuple(self._result(request, success=request.request_id.startswith("b:")) for request in requests)

    def solve_ik(self, request):
        self.requests.append(request)
        return self._result(request)

    def plan(self, request):
        self.requests.append(request)
        valid = (request.operation is not MotionOperation.LINEAR_TCP or self.linear_success)
        valid = valid and not any(abs(request.target.matrix[0, 3] - x) < 1e-8
                                  for x in self.fail_pregrasp_x)
        return self._result(request, success=valid)


def test_final_ik_screening_precedes_pregrasp_and_preserves_order():
    planner = Planner()
    result = GraspSkill().plan(planner, (_proposal("a"), _proposal("b", .4)), _context(), GraspSettings())
    assert [request.operation for request in planner.requests] == [MotionOperation.IK, MotionOperation.IK,
                                                                    MotionOperation.PLAN, MotionOperation.LINEAR_TCP]
    assert result.selected_candidate_id == "b"
    assert result.stage("approach").status == "unsupported"
    assert result.stage("close").status == "not_executed"
    assert result.stage("pregrasp").status == "planned"
    np.testing.assert_allclose(result.stage("pregrasp").target.matrix[:3, 3], [.4, .1, .17])
    assert result.to_dict()["attempts"][0]["candidate_id"] == "a"


def test_grasp_skill_can_opt_in_pregrasp_approximate_ik_only():
    planner = Planner()
    settings = GraspSettings(pregrasp_approximate_ik_budget={
        "max_position_error_m": .02, "max_rotation_error_rad": .1,
    })
    GraspSkill().plan(planner, (_proposal("b"),), _context(), settings)
    pregrasp = next(request for request in planner.requests if request.operation is MotionOperation.PLAN)
    approach = next(request for request in planner.requests if request.operation is MotionOperation.LINEAR_TCP)
    assert pregrasp.approximate_ik_budget["max_position_error_m"] == .02
    assert approach.approximate_ik_budget is None


def test_pick_wraps_grasp_and_does_not_claim_lift_when_approach_failed():
    result = PickSkill().plan(Planner(), (_proposal("b"),), _context(), GraspSettings())
    assert [stage.name for stage in result.stages] == [
        "open", "pregrasp", "approach", "close", "lift", "hold",
    ]
    assert result.to_dict()["stage_contract"] == [
        {"name": "open", "operation": "gripper_command"},
        {"name": "pregrasp", "operation": "plan"},
        {"name": "approach", "operation": "linear_tcp"},
        {"name": "close", "operation": "gripper_command"},
        {"name": "lift", "operation": "linear_tcp"},
        {"name": "hold", "operation": "hold"},
    ]
    assert result.stage("open").details["width_m"] == .08
    assert result.stage("lift").status == "not_executed"
    assert result.stage("approach").status == "unsupported"
    assert result.success is False
    assert result.to_dict()["claim"] == "规划阶段结果；不代表夹持或物理抬升成功"


def test_candidate_budget_is_explicit_and_stable():
    planner = Planner()
    result = GraspSkill().plan(planner, (_proposal("a"), _proposal("b")), _context(),
                               GraspSettings(max_grasp_candidates=1))
    assert len(planner.requests) == 1
    assert result.selected_candidate_id is None
    assert result.stage("pregrasp").status == "not_executed"
    assert result.attempts[0]["stage"] == "final_grasp_ik"


def test_pregrasp_failure_is_recorded_and_next_ik_candidate_is_tried():
    class TwoReachable(Planner):
        def evaluate_candidates(self, requests):
            self.requests.extend(requests)
            return tuple(self._result(request) for request in requests)

    planner = TwoReachable(fail_pregrasp_x=(.3,))
    result = GraspSkill().plan(planner, (_proposal("a"), _proposal("b", .4)), _context(), GraspSettings())
    assert result.selected_candidate_id == "b"
    assert [(item["candidate_id"], item["stage"]) for item in result.attempts] == [
        ("a", "final_grasp_ik"), ("b", "final_grasp_ik"),
        ("a", "pregrasp"), ("b", "pregrasp"), ("b", "approach"),
    ]


def test_batch_result_order_mismatch_is_not_silently_assigned():
    class Reversing(Planner):
        def evaluate_candidates(self, requests):
            return tuple(reversed(super().evaluate_candidates(requests)))

    import pytest
    with pytest.raises(RuntimeError, match="顺序"):
        GraspSkill().plan(Reversing(), (_proposal("a"), _proposal("b")), _context(), GraspSettings())


def test_legacy_visual_profile_preserves_original_pick_distances():
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "data/task_1/skills/skill_profiles.json"
    settings = load_grasp_settings(path, "pick_bowl_legacy_visual")
    # 旧 MicrowaveObjectInsertTask 从对象 profile 读 50 mm 的 pregrasp；
    # 旧 PickSkill 默认沿 pregrasp 方向撤离 100 mm。
    assert settings.approach_distance_m == .05
    assert settings.lift_distance_m == .10
    assert settings.candidate_selection == "legacy_ik"
    np.testing.assert_allclose(
        np.rad2deg(settings.max_pregrasp_joint_travel_rad), 120.0,
    )
    np.testing.assert_allclose(
        np.rad2deg(settings.max_approach_joint_travel_rad), 90.0,
    )


def test_opt_in_pregrasp_travel_limit_skips_reverse_branch_and_tries_next_candidate():
    class BranchPlanner(Planner):
        def evaluate_candidates(self, requests):
            self.requests.extend(requests)
            results = []
            for request in requests:
                success = (request.request_id.endswith(":final_ik")
                           or request.request_id.endswith(":pregrasp_ik_00"))
                results.append(self._result(request, success=success))
            return tuple(results)

        def plan(self, request):
            self.requests.append(request)
            if request.operation is MotionOperation.LINEAR_TCP:
                return self._result(request, success=True)
            endpoint = 2.5 if request.request_id.startswith("a:") else 0.5
            state = JointState(("j1",), (endpoint,))
            trajectory = JointTrajectory(
                ("j1",), (0.0, 1.0), ((request.start.positions_rad[0],), (endpoint,)),
            )
            return MotionResult(
                request.request_id, request.operation, True, "fake",
                PlannerMode.DEVELOPMENT, 0.01, goal_state=state,
                trajectory=trajectory,
            )

    planner = BranchPlanner(linear_success=True)
    result = GraspSkill().plan(
        planner, (_proposal("a"), _proposal("b", .4)), _context(),
        GraspSettings(candidate_selection="legacy_ik",
                      max_pregrasp_joint_travel_rad=2.0),
    )

    assert result.selected_candidate_id == "b"
    rejected = [item for item in result.attempts
                if item["stage"] == "pregrasp_posture_check"
                and not item["result"]["success"]]
    assert len(rejected) == 1
    assert rejected[0]["candidate_id"] == "a"
    assert rejected[0]["result"]["reason_code"] == "pregrasp_joint_travel_exceeded"
    assert rejected[0]["result"]["maximum_joint_name"] == "j1"


def test_opt_in_pregrasp_travel_limit_continues_after_approach_failure():
    class ApproachFallbackPlanner(Planner):
        def evaluate_candidates(self, requests):
            self.requests.extend(requests)
            return tuple(self._result(request, success=True) for request in requests)

        def plan(self, request):
            self.requests.append(request)
            if (request.operation is MotionOperation.LINEAR_TCP
                    and request.request_id.startswith("a:")):
                return MotionResult(
                    request.request_id, request.operation, False, "fake",
                    PlannerMode.DEVELOPMENT, 0.01,
                    failure=MotionFailure(
                        MotionFailureKind.PATH_NOT_FOUND,
                        "候选 a 的直线接近无路径",
                    ),
                )
            return self._result(request, success=True)

    planner = ApproachFallbackPlanner()
    result = GraspSkill().plan(
        planner, (_proposal("a"), _proposal("b", .4)), _context(),
        GraspSettings(
            candidate_selection="legacy_ik",
            max_pregrasp_joint_travel_rad=2.0,
        ),
    )

    assert result.selected_candidate_id == "b"
    assert result.stage("approach").status == "planned"
    approach_attempts = [
        item for item in result.attempts if item["stage"] == "approach"
    ]
    assert approach_attempts[0]["candidate_id"] == "a"
    assert approach_attempts[0]["result"]["success"] is False
    assert approach_attempts[-1]["candidate_id"] == "b"
    assert approach_attempts[-1]["result"]["success"] is True


def test_opt_in_approach_travel_limit_rejects_reverse_branch():
    class ApproachPosturePlanner(Planner):
        def evaluate_candidates(self, requests):
            self.requests.extend(requests)
            return tuple(self._result(request, success=True) for request in requests)

        def plan(self, request):
            self.requests.append(request)
            if (request.operation is MotionOperation.LINEAR_TCP
                    and request.request_id.startswith("a:")):
                state = JointState(("j1",), (2.0,))
                trajectory = JointTrajectory(
                    ("j1",), (0.0, 1.0),
                    ((request.start.positions_rad[0],), (2.0,)),
                )
                return MotionResult(
                    request.request_id, request.operation, True, "fake",
                    PlannerMode.DEVELOPMENT, 0.01, goal_state=state,
                    trajectory=trajectory,
                )
            return self._result(request, success=True)

    planner = ApproachPosturePlanner()
    result = GraspSkill().plan(
        planner, (_proposal("a"), _proposal("b", .4)), _context(),
        GraspSettings(
            candidate_selection="legacy_ik",
            max_pregrasp_joint_travel_rad=2.0,
            max_approach_joint_travel_rad=1.5,
        ),
    )

    assert result.selected_candidate_id == "b"
    rejected = [
        item for item in result.attempts
        if item["stage"] == "approach_posture_check"
        and not item["result"]["success"]
    ]
    assert rejected
    assert all(item["candidate_id"] == "a" for item in rejected)
    assert rejected[0]["result"]["reason_code"] == "approach_joint_travel_exceeded"


def test_grasp_uses_candidate_opening_widths_before_and_after_contact():
    planner = Planner(linear_success=True)
    pose = np.eye(4)
    pose[:3, 3] = [.3, .1, .25]
    candidate = GraspProposal(
        "b", "ring", "rim", pose, .007,
        {"pregrasp_opening_width_m": .041},
    )
    result = GraspSkill().plan(planner, (candidate,), _context(), GraspSettings())
    assert result.stage("open").details["width_m"] == .041
    assert result.stage("close").details["width_m"] == .007
    assert result.stage("approach").status == "planned"


def test_grasp_keeps_pregrasp_strict_but_routes_target_contact_to_its_scene():
    """抓碗时允许碰目标的最终抓取/接近，不能污染 pregrasp 避碰。"""
    strict = Planner(linear_success=False)
    target_contact = Planner(linear_success=True)
    strict_context = _context()
    contact_scene = PlanningSceneRef(1, "contact-hash", "bowl contact")

    result = PickSkill().plan(
        strict, (_proposal("b"),), strict_context, GraspSettings(),
        contact_planner=target_contact, contact_scene=contact_scene,
    )

    assert result.stage("approach").status == "planned"
    assert [request.request_id for request in strict.requests] == ["b:pregrasp"]
    assert all(request.scene == strict_context.scene for request in strict.requests)
    assert [request.request_id for request in target_contact.requests] == [
        "b:final_ik", "b:approach",
    ]
    assert all(request.scene == contact_scene for request in target_contact.requests)
    assert result.attempts[0]["request"]["scene"]["snapshot_sha256"] == "contact-hash"
    assert result.stage("pregrasp").request.scene == strict_context.scene


def test_legacy_visual_selects_rotated_pregrasp_by_ik_before_planning_path():
    class LegacyPlanner(Planner):
        def evaluate_candidates(self, requests):
            self.requests.extend(requests)
            return tuple(self._result(
                request,
                success=(request.request_id.endswith(":final_ik")
                         or request.request_id == "a:pregrasp_ik_01"),
            ) for request in requests)

    planner = LegacyPlanner(linear_success=True)
    settings = GraspSettings(candidate_selection="legacy_ik")
    result = GraspSkill().plan(
        planner, (_proposal("a"), _proposal("b", .4)), _context(), settings,
    )
    assert result.selected_candidate_id == "a"
    assert [request.operation for request in planner.requests[:2]] == [
        MotionOperation.IK, MotionOperation.IK,
    ]
    prechecks = [request for request in planner.requests if "pregrasp_ik" in request.request_id]
    assert len(prechecks) == 34  # 两个 final IK 候选各 17 个原版局部姿态。
    assert all(request.operation is MotionOperation.IK for request in prechecks)
    assert [request.operation for request in planner.requests[-2:]] == [
        MotionOperation.PLAN, MotionOperation.LINEAR_TCP,
    ]
    selected = result.stage("pregrasp").target.matrix
    expected_rotation = Rotation.from_euler("x", 5, degrees=True).as_matrix()
    np.testing.assert_allclose(selected[:3, :3], expected_rotation, atol=1e-9)
    np.testing.assert_allclose(selected[:3, 3],
                               np.array([.3, .1, .25]) - .08 * selected[:3, 2])


def test_legacy_visual_does_not_switch_grasp_after_selected_path_failure():
    class LegacyPlanner(Planner):
        def evaluate_candidates(self, requests):
            self.requests.extend(requests)
            return tuple(self._result(
                request,
                    success=(request.request_id.endswith(":final_ik")
                             or request.request_id == "a:pregrasp_ik_00"),
            ) for request in requests)

    planner = LegacyPlanner(fail_pregrasp_x=(.3,))
    result = GraspSkill().plan(
        planner, (_proposal("a"), _proposal("b", .4)), _context(),
        GraspSettings(candidate_selection="legacy_ik"),
    )
    assert result.selected_candidate_id == "a"
    assert result.stage("pregrasp").status == "failed"
    assert result.stage("approach").status == "not_executed"
    assert not any(request.request_id == "b:pregrasp" for request in planner.requests)


def test_visual_retreat_uses_attached_scene_and_previous_approach_endpoint():
    planner = Planner(linear_success=True)
    settings = GraspSettings(approach_distance_m=.05, lift_distance_m=.10)
    initial = PickSkill().plan(planner, (_proposal("b"),), _context(), settings)
    assert initial.stage("lift").status == "not_executed"
    approach_end = initial.stage("approach").result.goal_state
    attached = SkillContext("right", approach_end,
                            PlanningSceneRef(1, "attached", "with_bowl"), "bowl")
    before = len(planner.requests)
    result = PickSkill().plan_visual_retreat(planner, initial, attached, settings)
    retreat_requests = planner.requests[before:]
    assert [request.operation for request in retreat_requests] == [
        MotionOperation.PLAN, MotionOperation.LINEAR_TCP,
    ]
    assert all(request.start == approach_end and request.scene == attached.scene
               for request in retreat_requests)
    assert result.stage("lift").status == "planned"
    assert result.stage("hold").status == "planned"
    np.testing.assert_allclose(result.stage("lift").target.matrix[:3, 3], [.3, .1, .15])
    assert result.stage("lift").result.trajectory is not None


def test_visual_retreat_keeps_old_first_feasible_target_when_linear_path_fails():
    planner = Planner(linear_success=False)
    settings = GraspSettings()
    # 前四段需要已有直线轨迹；只在 post-pick 阶段令直线规划失败。
    planner.linear_success = True
    initial = PickSkill().plan(planner, (_proposal("b"),), _context(), settings)
    planner.linear_success = False
    attached = SkillContext("right", initial.stage("approach").result.goal_state,
                            PlanningSceneRef(1, "attached", "with_bowl"), "bowl")
    before = len(planner.requests)
    result = PickSkill().plan_visual_retreat(planner, initial, attached, settings)
    assert result.stage("lift").status == "failed"
    assert result.stage("hold").status == "not_executed"
    assert [request.operation for request in planner.requests[before:]] == [
        MotionOperation.PLAN, MotionOperation.LINEAR_TCP,
    ]


def test_postpick_candidates_keep_old_pregrasp_direction_and_local_rotation_order():
    """Skill 的目标生成要供视觉和在线物理端共用，不能退化成世界 Z 抬升。"""
    from gene_agent.skills.library.pick.planning import make_postpick_candidates

    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    grasp[:3, 3] = [.20, .10, -.07]
    pregrasp = grasp.copy()
    pregrasp[:3, 3] += [.02, .01, .045]

    candidates = make_postpick_candidates(grasp, pregrasp, .10)

    assert len(candidates) == 17
    direction = np.array([.02, .01, .045])
    expected_position = grasp[:3, 3] + .10 * direction / np.linalg.norm(direction)
    for candidate in candidates:
        np.testing.assert_allclose(candidate[:3, 3], expected_position, atol=1e-12)
    np.testing.assert_allclose(candidates[0][:3, :3], grasp[:3, :3], atol=1e-12)
    np.testing.assert_allclose(
        candidates[1][:3, :3],
        grasp[:3, :3] @ Rotation.from_euler("x", 5, degrees=True).as_matrix(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        candidates[9][:3, :3],
        grasp[:3, :3] @ Rotation.from_euler("y", 5, degrees=True).as_matrix(),
        atol=1e-12,
    )


def test_postpick_candidates_can_use_explicit_world_vertical_lift():
    from gene_agent.skills.library.pick.planning import make_postpick_candidates

    grasp = np.eye(4)
    pregrasp = np.eye(4)
    pregrasp[:3, 3] = [.05, 0., .01]
    candidates = make_postpick_candidates(
        grasp, pregrasp, .10, translation_direction_world=[0., 0., 1.],
    )

    assert len(candidates) == 17
    for candidate in candidates:
        np.testing.assert_allclose(candidate[:3, 3], [0., 0., .10], atol=1e-12)


def test_pick_viewer_reads_attached_snapshot_only_after_close(tmp_path):
    import json
    from gene_agent.feedback.skill_run import load_skill_run

    (tmp_path / "collision").mkdir()
    (tmp_path / "visual_attachment").mkdir()
    (tmp_path / "collision/collision_snapshot.json").write_text(
        json.dumps({"snapshot_sha256": "before"}), encoding="utf-8",
    )
    (tmp_path / "visual_attachment/collision_snapshot.json").write_text(
        json.dumps({"snapshot_sha256": "after", "transferred_objects": {"bowl": [[0, 0, 0, .002]]}}),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1, "skill": "pick", "task": "pick_bowl_skill_planning",
        "instance_id": "bowl", "snapshot_sha256": "before",
        "visual_attachment": {
            "snapshot_sha256": "after",
            "collision_snapshot_ref": str(tmp_path / "visual_attachment/collision_snapshot.json"),
        },
        "stages": [
            {"name": "approach", "status": "planned", "request": {
                "operation": "linear_tcp", "scene": {"snapshot_sha256": "before"}},
             "result": {"operation": "linear_tcp"}},
            {"name": "lift", "status": "planned", "request": {
                "operation": "linear_tcp", "scene": {"snapshot_sha256": "after"}},
             "result": {"operation": "linear_tcp"}},
        ],
    }
    (tmp_path / "skill_pick_bowl_result.json").write_text(
        json.dumps(report), encoding="utf-8",
    )
    view = load_skill_run(tmp_path)
    assert view.stages[0].collision_snapshot_ref == "collision/collision_snapshot.json"
    assert view.stages[1].collision_snapshot_ref == "visual_attachment/collision_snapshot.json"
    assert view.stages[1].attachment_ref == "visual_attachment/collision_snapshot.json"


def test_pick_viewer_reads_and_verifies_each_stage_collision_snapshot(tmp_path):
    """不同开口/接触策略的阶段不能被单个初始碰撞哈希替代。"""
    import json
    from gene_agent.feedback.skill_run import load_skill_run

    for relative, digest in (
        ('collision/collision_snapshot.json', 'initial'),
        ('collision_grasp_open/strict/collision_snapshot.json', 'strict41'),
        ('collision_grasp_open/target_contact/collision_snapshot.json', 'contact41'),
        ('visual_attachment/collision_snapshot.json', 'attached'),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'snapshot_sha256': digest}), encoding='utf-8')
    report = {
        'schema_version': 1, 'skill': 'pick', 'instance_id': 'bowl',
        'snapshot_sha256': 'initial',
        'visual_attachment': {
            'snapshot_sha256': 'attached',
            'collision_snapshot_ref': str(tmp_path / 'visual_attachment/collision_snapshot.json'),
        },
        'stages': [
            {'name': 'pregrasp', 'status': 'planned',
             'collision_snapshot_ref': 'collision_grasp_open/strict/collision_snapshot.json',
             'request': {'operation': 'plan', 'scene': {'snapshot_sha256': 'strict41'}},
             'result': {'operation': 'plan'}},
            {'name': 'approach', 'status': 'planned',
             'collision_snapshot_ref': 'collision_grasp_open/target_contact/collision_snapshot.json',
             'request': {'operation': 'linear_tcp', 'scene': {'snapshot_sha256': 'contact41'}},
             'result': {'operation': 'linear_tcp'}},
            {'name': 'lift', 'status': 'planned',
             'request': {'operation': 'linear_tcp', 'scene': {'snapshot_sha256': 'attached'}},
             'result': {'operation': 'linear_tcp'}},
        ],
    }
    (tmp_path / 'skill_pick_bowl_result.json').write_text(json.dumps(report), encoding='utf-8')
    view = load_skill_run(tmp_path)

    assert [stage.collision_snapshot_ref for stage in view.stages] == [
        'collision_grasp_open/strict/collision_snapshot.json',
        'collision_grasp_open/target_contact/collision_snapshot.json',
        'visual_attachment/collision_snapshot.json',
    ]
