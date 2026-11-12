"""阶段判定必须来自实际物理采样，而不是目标姿态或规划成功标志。"""

import pytest

from gene_agent.simulation.checks import evaluate_checks


def _door_check():
    return {
        "kind": "door_angle",
        "target_rad": 0.40,
        "tolerance_rad": 0.05,
        "stable_window_s": 0.20,
        "required_contact_key": "pad_handle_contact_count",
    }


def test_door_angle_requires_contact_and_stable_actual_angle():
    # 如果实现误把目标角当作实际角，或只看最后一帧，这项测试会失效。
    samples = [
        {"sim_time_s": 0.0, "door_angle_rad": 0.0, "pad_handle_contact_count": 0},
        {"sim_time_s": 0.5, "door_angle_rad": 0.35, "pad_handle_contact_count": 2},
        {"sim_time_s": 0.8, "door_angle_rad": 0.38, "pad_handle_contact_count": 0},
        {"sim_time_s": 0.9, "door_angle_rad": 0.40, "pad_handle_contact_count": 0},
        {"sim_time_s": 1.0, "door_angle_rad": 0.41, "pad_handle_contact_count": 0},
    ]
    result = evaluate_checks([_door_check()], samples)[0]
    assert result["status"] == "pass"
    assert result["code"] == "goal_reached"
    assert result["actual_rad"] == pytest.approx(0.41)
    assert result["contact_sample_count"] == 1


def test_door_angle_does_not_pass_from_a_single_transient_target_hit():
    samples = [
        {"sim_time_s": 0.0, "door_angle_rad": 0.0, "pad_handle_contact_count": 0},
        {"sim_time_s": 0.5, "door_angle_rad": 0.40, "pad_handle_contact_count": 1},
        {"sim_time_s": 0.8, "door_angle_rad": 0.25, "pad_handle_contact_count": 0},
        {"sim_time_s": 1.0, "door_angle_rad": 0.20, "pad_handle_contact_count": 0},
    ]
    result = evaluate_checks([_door_check()], samples)[0]
    assert result["status"] == "fail"
    assert result["code"] == "angle_not_stable"


def test_door_angle_reports_no_contact_even_if_goal_angle_is_reached():
    samples = [
        {"sim_time_s": 0.0, "door_angle_rad": 0.0, "pad_handle_contact_count": 0},
        {"sim_time_s": 0.8, "door_angle_rad": 0.40, "pad_handle_contact_count": 0},
        {"sim_time_s": 1.0, "door_angle_rad": 0.40, "pad_handle_contact_count": 0},
    ]
    result = evaluate_checks([_door_check()], samples)[0]
    assert result["status"] == "fail"
    assert result["code"] == "no_contact"


def test_door_angle_needs_enough_observation_time():
    samples = [
        {"sim_time_s": 0.0, "door_angle_rad": 0.39, "pad_handle_contact_count": 1},
        {"sim_time_s": 0.1, "door_angle_rad": 0.40, "pad_handle_contact_count": 0},
    ]
    result = evaluate_checks([_door_check()], samples)[0]
    assert result["status"] == "unknown"
    assert result["code"] == "insufficient_observation"
