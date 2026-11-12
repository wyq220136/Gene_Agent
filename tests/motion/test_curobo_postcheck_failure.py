"""直线约束和碰撞复核同时失败时，保留两种原因。"""

from gene_agent.motion.curobo.core import _record_postcheck_failure


def test_collision_failure_does_not_hide_earlier_linear_rejection():
    result = {"success": False, "status": "linear_path_rejected",
              "failure_category": "constraint_violation",
              "message": "TCP 轨迹不够直"}
    _record_postcheck_failure(result, category="collision_postcheck_rejected",
                              status="collision_postcheck_rejected",
                              message="离散碰撞复核失败")
    assert result["failure_category"] == "constraint_violation"
    assert result["message"] == "TCP 轨迹不够直"
    assert result["additional_failure_categories"] == ["collision_postcheck_rejected"]


def test_collision_failure_is_primary_when_no_prior_failure():
    result = {"success": True, "status": "success"}
    _record_postcheck_failure(result, category="collision_postcheck_rejected",
                              status="collision_postcheck_rejected",
                              message="离散碰撞复核失败")
    assert result["success"] is False
    assert result["failure_category"] == "collision_postcheck_rejected"
    assert result["message"] == "离散碰撞复核失败"
