"""人工准备的持碗状态有独立来源，不伪造 Pick 执行记录。"""

import numpy as np
import pytest

from gene_agent.realworld.microwave.attachment import make_manual_held_bowl_report
from gene_agent.realworld.microwave.attachment import validate_manual_held_bowl_report
from gene_agent.tasks.checks import CheckResult


def _scene():
    return {"snapshot": {"snapshot_sha256": "scene-a", "active_group": "right"},
            "provenance": {"physics_stepped": False, "pose_frame_verified": True,
                           "collision_snapshot_sha256": "scene-a",
                           "source_frame_number": 10, "source_timestamp_ms": 10000.,
                           "aligned_object_poses_world": {
                               "Bowl/Bowl_0": np.eye(4).tolist()}}}


def test_manual_hold_report_names_real_source_and_frame():
    check = CheckResult("pass", "manual_terminal", {"answer": "y"}, "已准备持碗")
    report = make_manual_held_bowl_report(_scene(), check)
    assert report["source_format"] == "manual_prepared_held_bowl"
    assert report["object_id"] == "Bowl/Bowl_0"
    assert report["source_collision_snapshot_sha256"] == "scene-a"
    assert report["source_frame_number"] == 10
    assert report["postcondition"]["status"] == "pass"
    assert report["postcondition"]["evidence"] == {
        "answer": "y", "collision_snapshot_sha256": "scene-a",
        "source_frame_number": 10, "source_timestamp_ms": 10000.,
    }
    assert "grasp_success" not in report


def test_manual_hold_report_rejects_old_answer_even_if_scene_is_current():
    with pytest.raises(ValueError, match="持碗检查证据与当前现场快照不一致"):
        make_manual_held_bowl_report(
            _scene(), CheckResult(
                "pass", "vision", {"source_frame_number": 9}, "旧帧显示已夹住",
            ),
        )


def test_manual_hold_report_rejects_unbound_check_when_reused_on_same_frame():
    report = make_manual_held_bowl_report(
        _scene(), CheckResult("pass", "manual_terminal", {"answer": "y"}),
    )
    report["postcondition"]["evidence"].pop("source_frame_number")
    with pytest.raises(ValueError, match="当前现场快照"):
        validate_manual_held_bowl_report(_scene(), report)


@pytest.mark.parametrize("status,source", [
    ("unknown", "vision"), ("fail", "manual_terminal"),
    ("pass", "simulation_privileged_weld"),
])
def test_manual_hold_report_rejects_unconfirmed_or_simulation_evidence(status, source):
    with pytest.raises(ValueError):
        make_manual_held_bowl_report(
            _scene(), CheckResult(status, source, {}, "没有现场持碗证据"),
        )


def test_manual_hold_evidence_expires_on_new_frame_with_same_collision_hash():
    first = _scene()
    report = make_manual_held_bowl_report(
        first, CheckResult("pass", "manual_terminal", {"answer": "y"}),
    )
    later = _scene()
    later["provenance"].update(source_frame_number=11, source_timestamp_ms=11000.)
    with pytest.raises(ValueError, match="重新确认"):
        validate_manual_held_bowl_report(later, report)
    with pytest.raises(ValueError, match="当前现场快照"):
        validate_manual_held_bowl_report(
            later, report,
            current_hold_check={"status": "pass", "source": "manual_terminal",
                                "evidence": {"source_frame_number": 10,
                                             "source_timestamp_ms": 10000.,
                                             "collision_snapshot_sha256": "scene-a"}},
        )
    check = validate_manual_held_bowl_report(
        later, report,
        current_hold_check={"status": "pass", "source": "manual_terminal",
                            "evidence": {"source_frame_number": 11,
                                         "source_timestamp_ms": 11000.,
                                         "collision_snapshot_sha256": "scene-a"}},
    )
    assert check["evidence"]["source_frame_number"] == 11
