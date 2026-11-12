"""失败 IK 的预览必须选有约束证据且最接近目标的候选。"""

import numpy as np

from gene_agent.visualization.grasp_tuner.curobo_pose_ik import diagnose_mujoco_candidate, select_preview_candidate
from gene_agent.visualization.grasp_tuner.pose_ik import PoseIKResult


class FakePlanner:
    def fk(self, names, q):
        pose = np.eye(4)
        pose[0, 3] = q[0]
        return pose

    def _collision_diagnostic(self, q):
        return {"joint_limit_violation": False, "self_collision": True,
                "scene_collision": False}


def test_preview_prefers_feasible_candidate_over_first_invalid_seed():
    goal = np.eye(4)
    selected = select_preview_candidate(
        FakePlanner(), ("joint",), goal,
        np.array([[.12], [.015], [.002]]), [False, True, False],
    )
    assert selected["index"] == 1
    assert selected["constraint_feasible"] is True
    assert np.isclose(selected["position_error_m"], .015)


def test_preview_reports_invalid_when_no_seed_is_feasible():
    selected = select_preview_candidate(
        FakePlanner(), ("joint",), np.eye(4),
        np.array([[.12], [.04]]), [False, False],
    )
    assert selected["index"] == 1
    assert selected["constraint_feasible"] is False


def test_failed_mujoco_candidate_is_still_checked_for_curobo_self_collision():
    report = diagnose_mujoco_candidate(
        np.eye(4), PoseIKResult(False, .012, 5., (.012,)),
        FakePlanner(), ("joint",),
        position_tolerance_m=.005, rotation_tolerance_rad=.05,
    )
    assert report["conclusion"] == "mujoco_also_not_strict"
    assert report["candidate_check"]["self_collision"] is True
    assert np.isclose(report["curobo_fk_position_error_m"], .012)
