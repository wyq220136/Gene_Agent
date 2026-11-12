"""Ring 抓取生成器的局部几何与新深度语义。"""

import numpy as np
import pytest

from gene_agent.knowledge.geometry import RingConcept
from gene_agent.knowledge.grasp import ClosingWidthSpec, GraspProfile, generate_grasps


def _profile(*, depths=(0.001,), azimuths=(0.0,), tips=(0.0,)) -> GraspProfile:
    return GraspProfile(
        name="ring_test",
        generator="ring",
        part_name="rim",
        parameters={
            "surface": ("bottom",),
            "azimuth_rad": azimuths,
            "tip_y_rad": tips,
            "depth_m": depths,
        },
        closing_width=ClosingWidthSpec(mode="geometry", clearance_m=0.002),
    )


def test_ring_uses_neutral_local_formula_and_tcp_axes() -> None:
    concept = RingConcept(0.045, 0.05, 0.02, [0, 0, 1], np.eye(4))

    proposal = generate_grasps(concept, np.eye(4), _profile()).proposals[0]

    expected = np.eye(4)
    expected[:3, 3] = [0.0475, 0.0, -0.009]
    np.testing.assert_allclose(proposal.tcp_pose_world, expected, atol=1e-12)
    np.testing.assert_allclose(proposal.tcp_pose_world[:3, 0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(proposal.tcp_pose_world[:3, 2], [0, 0, 1], atol=1e-12)
    assert proposal.closing_width_m == pytest.approx(0.009)


def test_ring_shallow_depth_is_generated_and_only_changes_tcp_position() -> None:
    concept = RingConcept(0.045, 0.05, 0.02, [0, 0, 1], np.eye(4))
    result = generate_grasps(concept, np.eye(4), _profile(depths=(0.001, 0.004)))

    assert len(result.proposals) == 2
    np.testing.assert_allclose(
        result.proposals[0].tcp_pose_world[:3, :3],
        result.proposals[1].tcp_pose_world[:3, :3],
    )
    np.testing.assert_allclose(
        result.proposals[1].tcp_pose_world[:3, 3]
        - result.proposals[0].tcp_pose_world[:3, 3],
        [0.0, 0.0, 0.003],
    )


def test_ring_keeps_declared_center_out_order_and_stable_ids() -> None:
    concept = RingConcept(0.045, 0.05, 0.02, [0, 0, 1], np.eye(4))
    profile = _profile(tips=(0.0, -0.2, 0.2), azimuths=(0.0, np.pi / 2.0))

    first = generate_grasps(concept, np.eye(4), profile)
    second = generate_grasps(concept, np.eye(4), profile)

    assert [p.parameters["tip_y_rad"] for p in first.proposals] == [
        0.0,
        -0.2,
        0.2,
        0.0,
        -0.2,
        0.2,
    ]
    assert [p.candidate_id for p in first.proposals] == [
        p.candidate_id for p in second.proposals
    ]
    assert "azimuth_rad=1" in first.proposals[3].candidate_id
    assert "tip_y_rad=0" in first.proposals[3].candidate_id


def test_partial_ring_keeps_boundary_azimuths_and_reports_each_out_of_arc_combination() -> None:
    concept = RingConcept(
        0.045,
        0.05,
        0.02,
        [0, 0, 1],
        np.eye(4),
        angle_span_rad=np.pi / 2.0,
    )
    profile = _profile(azimuths=(0.0, np.pi / 2.0, np.pi, -0.1))

    result = generate_grasps(concept, np.eye(4), profile)

    assert [item.parameters["azimuth_rad"] for item in result.proposals] == [
        0.0,
        np.pi / 2.0,
    ]
    assert len(result.diagnostics) == 2
    assert {item.code for item in result.diagnostics} == {"RING_AZIMUTH_OUT_OF_ARC"}
    assert [item.context["azimuth_rad"] for item in result.diagnostics] == [np.pi, -0.1]
    assert "azimuth_rad=2" in result.diagnostics[0].candidate_id
    assert "azimuth_rad=3" in result.diagnostics[1].candidate_id


def test_full_ring_treats_azimuth_as_periodic() -> None:
    concept = RingConcept(0.045, 0.05, 0.02, [0, 0, 1], np.eye(4))
    profile = _profile(azimuths=(-np.pi / 2.0, 2.0 * np.pi, 3.0 * np.pi / 2.0))

    result = generate_grasps(concept, np.eye(4), profile)

    assert len(result.proposals) == 3
    assert result.diagnostics == ()
    np.testing.assert_allclose(
        result.proposals[0].tcp_pose_world,
        result.proposals[2].tcp_pose_world,
        atol=1e-12,
    )


def test_zero_degree_ring_returns_no_proposals_and_clear_diagnostics() -> None:
    concept = RingConcept(
        0.045,
        0.05,
        0.02,
        [0, 0, 1],
        np.eye(4),
        angle_span_rad=0.0,
    )

    result = generate_grasps(
        concept,
        np.eye(4),
        _profile(azimuths=(0.0, np.pi / 4.0)),
    )

    assert result.proposals == ()
    assert len(result.diagnostics) == 2
    assert all(item.code == "RING_AZIMUTH_OUT_OF_ARC" for item in result.diagnostics)
    assert all("空弧" in item.message for item in result.diagnostics)
