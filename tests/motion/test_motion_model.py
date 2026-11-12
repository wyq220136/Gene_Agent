"""Motion 公共契约测试；不导入 cuRobo、MuJoCo 或 Viser。"""

import json

import numpy as np
import pytest

from gene_agent.motion.model import (
    CollisionAllowance,
    CollisionPolicy,
    JointState,
    JointTrajectory,
    MotionFailure,
    MotionFailureKind,
    MotionOperation,
    MotionRequest,
    MotionResult,
    PlannerMode,
    PlanningSceneRef,
    PoseTarget,
    motion_request_from_dict,
)


def _request() -> MotionRequest:
    target = np.eye(4)
    target[:3, 3] = [0.42, -0.08, 0.31]
    return MotionRequest(
        request_id="pick-bowl.grasp.003",
        operation=MotionOperation.IK,
        group="right",
        start=JointState(("joint_a", "joint_b"), (0.1, -0.2)),
        target=PoseTarget("world_tcp", target),
        scene=PlanningSceneRef(
            revision=7,
            snapshot_sha256="a" * 64,
            scope="桌面、碗、微波炉和静止左臂",
        ),
        collision_policy=CollisionPolicy(
            policy_id="grasp-contact",
            allowances=(
                CollisionAllowance(
                    first="right_finger_pad",
                    second="bowl",
                    reason="最终 grasp 阶段的预期接触",
                ),
            ),
        ),
        seed=17,
        planner_mode=PlannerMode.DEVELOPMENT,
    )


def test_motion_request_has_stable_json_contract() -> None:
    payload = _request().to_dict()

    assert payload["request_id"] == "pick-bowl.grasp.003"
    assert payload["operation"] == "ik"
    assert payload["start"]["joint_names"] == ["joint_a", "joint_b"]
    assert payload["target"]["frame"] == "world_tcp"
    assert payload["scene"]["revision"] == 7
    assert payload["collision_policy"]["allowances"][0]["reason"] == "最终 grasp 阶段的预期接触"
    assert payload["planner_mode"] == "development"
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


def test_motion_request_roundtrips_optional_near_ik_budget():
    from dataclasses import replace
    request = replace(_request(), operation=MotionOperation.PLAN,
                      approximate_ik_budget={"max_position_error_m": .02,
                                             "max_rotation_error_rad": .1})
    restored = motion_request_from_dict(request.to_dict())
    assert restored.approximate_ik_budget == request.approximate_ik_budget


@pytest.mark.parametrize(
    "names, positions",
    [
        (("a", "a"), (0.0, 0.1)),
        (("a", "b"), (0.0,)),
        (("a", "b"), (0.0, float("nan"))),
    ],
)
def test_joint_state_rejects_ambiguous_or_non_finite_values(names, positions) -> None:
    with pytest.raises(ValueError):
        JointState(names, positions)


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((3, 4)),
        np.diag([1.0, 1.0, 2.0, 1.0]),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], dtype=float),
        np.array([[1, 0, 0, np.nan], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
    ],
)
def test_pose_target_rejects_non_rigid_transform(matrix) -> None:
    with pytest.raises(ValueError):
        PoseTarget("world_tcp", matrix)


@pytest.mark.parametrize(
    "first, second, reason",
    [("", "bowl", "contact"), ("finger", "", "contact"), ("finger", "bowl", "")],
)
def test_collision_allowance_requires_pair_and_reason(first, second, reason) -> None:
    with pytest.raises(ValueError):
        CollisionAllowance(first, second, reason)


def test_motion_result_enforces_success_failure_exclusivity() -> None:
    request = _request()
    trajectory = JointTrajectory(
        joint_names=("joint_a", "joint_b"),
        times_s=(0.0, 0.1),
        positions_rad=((0.1, -0.2), (0.2, -0.1)),
    )
    success = MotionResult(
        request_id=request.request_id,
        operation=MotionOperation.PLAN,
        success=True,
        backend="synthetic",
        planner_mode=PlannerMode.DEVELOPMENT,
        duration_s=0.02,
        trajectory=trajectory,
    )
    assert success.to_dict()["failure"] is None

    failure = MotionFailure(
        kind=MotionFailureKind.GOAL_IK_NO_SOLUTION,
        summary="最终 grasp pose 没有可用 IK",
        backend_status="solver_rejected",
    )
    failed = MotionResult(
        request_id=request.request_id,
        operation=MotionOperation.IK,
        success=False,
        backend="synthetic",
        planner_mode=PlannerMode.DEVELOPMENT,
        duration_s=0.01,
        failure=failure,
    )
    assert failed.to_dict()["failure"]["kind"] == "goal_ik_no_solution"

    with pytest.raises(ValueError, match="成功结果"):
        MotionResult(
            request_id="bad-success",
            operation=MotionOperation.IK,
            success=True,
            backend="synthetic",
            planner_mode=PlannerMode.DEVELOPMENT,
            duration_s=0.0,
            failure=failure,
        )
    with pytest.raises(ValueError, match="失败结果"):
        MotionResult(
            request_id="bad-failure",
            operation=MotionOperation.IK,
            success=False,
            backend="synthetic",
            planner_mode=PlannerMode.DEVELOPMENT,
            duration_s=0.0,
        )


def test_trajectory_rejects_wrong_shape_or_non_monotonic_time() -> None:
    with pytest.raises(ValueError):
        JointTrajectory(("a", "b"), (0.0, 0.1), ((0.0,), (1.0,)))
    with pytest.raises(ValueError):
        JointTrajectory(("a",), (0.0, 0.0), ((0.0,), (1.0,)))


def test_default_motion_request_enables_cuda_graph_but_preserves_explicit_diagnostic():
    original = _request()
    default = MotionRequest(
        "default", original.operation, original.group, original.start,
        original.target, original.scene,
    )
    assert default.planner_mode is PlannerMode.CUDA_GRAPH_BENCHMARK
    assert motion_request_from_dict(default.to_dict()).planner_mode is default.planner_mode
    assert original.planner_mode is PlannerMode.DEVELOPMENT
