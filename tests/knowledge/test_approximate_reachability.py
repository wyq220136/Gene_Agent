"""Knowledge 保留严格失败但误差预算内的 pose 候选。"""
from gene_agent.knowledge.reachability import assess_pose_ik
from gene_agent.visualization.grasp_tuner.pose_ik import PoseIKResult


def test_knowledge_reports_approximate_candidate_without_faking_strict_success():
    result = assess_pose_ik(PoseIKResult(False, .012, 7., (0.,)),
                            max_position_error_m=.02, max_rotation_error_deg=10.)
    assert result["acceptable"] and result["approximate"]
    assert result["strict_ik_success"] is False
    assert result["position_error_m"] == .012


def test_knowledge_rejects_candidate_outside_either_budget():
    for position, rotation in ((.021, 2.), (.005, 11.)):
        result = assess_pose_ik(PoseIKResult(False, position, rotation, (0.,)),
                                max_position_error_m=.02, max_rotation_error_deg=10.)
        assert not result["acceptable"]
