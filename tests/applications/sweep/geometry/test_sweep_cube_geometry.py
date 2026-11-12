"""清扫目标由 Concept 与当前物体 pose 决定，避免把名义世界 +Y 写死。"""

import json
from pathlib import Path

import numpy as np
import pytest

from gene_agent.knowledge.sweep import (make_brush_grasp_pose, make_sweep_layout,
                                       make_sweep_contact_pose_candidates)
from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun


SCENE = Path(__file__).resolve().parents[4] / "data/realworld/brush_cube_dustpan_0"


def test_rotated_contact_candidates_keep_cube_gap_and_center_alignment():
    from gene_agent.knowledge.sweep import align_sweep_contact_candidates
    from gene_agent.knowledge.geometry import load_concept
    scene = json.loads((SCENE / "scene.json").read_text())
    poses = {n: np.asarray(r["T_world_object"]) for n, r in scene["objects"].items()}
    tcp = make_brush_grasp_pose(poses["brush"], SCENE / "concepts/brush.json")
    layout = make_sweep_layout(poses, SCENE / "concepts")
    base = tcp.copy(); base[:3, 3] += layout.brush_start_world_xyz - poses["brush"][:3, 3]
    rows = make_sweep_contact_pose_candidates(base, tcp, poses["brush"],
        SCENE / "concepts/brush.json", sweep_direction_world_xy=layout.direction_world_xy,
        table_top_z_m=0.)
    fixed = align_sweep_contact_candidates(rows, tcp, poses, SCENE / "concepts")
    face = load_concept(SCENE / "concepts/brush.json", part_name="bristle_face").concept
    cube = load_concept(SCENE / "concepts/cube.json", part_name="body").concept
    grip = np.linalg.inv(tcp) @ poses["brush"]
    c = poses["cube"] @ cube.pose_in_object
    direction = layout.direction_world_xy; lateral = np.array([-direction[1], direction[0]])
    for old, row in zip(rows, fixed):
        f = np.asarray(row["target_tcp_world"]) @ grip @ face.pose_in_object
        radius = .5 * np.sum(np.abs(direction @ f[:2, :3]) * face.size_xyz_m)
        cube_radius = .5 * np.sum(np.abs(direction @ c[:2, :3]) * cube.size_xyz_m)
        assert np.dot(direction, c[:2, 3] - f[:2, 3]) - radius - cube_radius == pytest.approx(.005)
        assert np.dot(lateral, c[:2, 3] - f[:2, 3]) == pytest.approx(0., abs=1e-9)
        assert row["face_low_z_m"] == old["face_low_z_m"]


def test_contact_candidates_keep_functional_brush_face_near_table():
    tcp = np.eye(4)
    brush = np.eye(4)
    base = np.eye(4)
    base[2, 3] = .07
    candidates = make_sweep_contact_pose_candidates(
        base, tcp, brush, SCENE / "concepts/brush.json",
        sweep_direction_world_xy=np.array([0., 1.]), table_top_z_m=0.)
    assert candidates[0]["tilt_axis"] == "original"
    assert [row["tilt_deg"] for row in candidates[1:6]] == [0., 5., 10., 15., 20.]
    assert all(row["face_low_z_m"] == pytest.approx(.002, abs=1e-8)
               for row in candidates[1:])
    assert candidates[3]["target_tcp_world"][0][0] == pytest.approx(1.)
    assert candidates[3]["tilt_axis"] == "world_x"
    assert candidates[3]["target_tcp_world"][1][2] < 0.


def test_sweep_layout_uses_dustpan_opening_and_functional_parts():
    scene = json.loads((SCENE / "scene.json").read_text())
    poses = {name: np.asarray(row["T_world_object"])
             for name, row in scene["objects"].items()}
    layout = make_sweep_layout(poses, SCENE / "concepts")
    assert np.dot(layout.direction_world_xy,
                  poses["dustpan"][:2, 3] - poses["cube"][:2, 3]) > 0
    assert layout.brush_start_world_xyz[1] < poses["cube"][1, 3]
    assert layout.brush_end_world_xyz[1] > layout.brush_start_world_xyz[1] + .10
    grasp = make_brush_grasp_pose(poses["brush"], SCENE / "concepts/brush.json")
    assert np.isclose(grasp[2, 3], .078, atol=.002)
    assert np.dot(grasp[:3, 0], [0, 0, -1]) > .9


def test_sweep_target_keeps_rotated_cube_footprint_inside_opening():
    scene = json.loads((SCENE / "scene.json").read_text())
    poses = {name: np.asarray(row["T_world_object"])
             for name, row in scene["objects"].items()}
    layout = make_sweep_layout(poses, SCENE / "concepts")
    from gene_agent.knowledge.geometry import load_concept
    cube = load_concept(SCENE / "concepts/cube.json", part_name="body").concept
    opening = poses["dustpan"][:2, 0]
    radius = .5 * np.sum(np.abs(opening @ (poses["cube"] @ cube.pose_in_object)[:2, :3]) * cube.size_xyz_m)
    local_x = np.dot(layout.cube_goal_world_xyz[:2] - poses["dustpan"][:2, 3], opening)
    assert .082 - local_x - radius == pytest.approx(.016)


def test_sweep_layout_can_target_another_block():
    scene = json.loads((SCENE / "scene.json").read_text())
    poses = {name: np.asarray(row["T_world_object"])
             for name, row in scene["objects"].items()}
    layout = make_sweep_layout(poses, SCENE / "concepts", target_name="cube_red")
    red = poses["cube_red"][:2, 3]
    assert np.dot(layout.direction_world_xy,
                  layout.cube_goal_world_xyz[:2] - red) > 0
    assert np.linalg.norm(layout.brush_start_world_xyz[:2] - red) < .12
    assert layout.brush_end_world_xyz[1] > layout.brush_start_world_xyz[1] + .10


def test_recovery_inside_pan_uses_inward_axis_without_sideways_push():
    scene = json.loads((SCENE / "scene.json").read_text())
    poses = {name: np.asarray(row["T_world_object"]).copy()
             for name, row in scene["objects"].items()}
    poses["cube_red"][:2, 3] = [.493, .180]
    with pytest.raises(ValueError, match="簸箕开口"):
        make_sweep_layout(poses, SCENE / "concepts", target_name="cube_red")
    layout = make_sweep_layout(poses, SCENE / "concepts", target_name="cube_red",
                               recovery_inside_pan=True)
    assert np.allclose(layout.direction_world_xy, [0., 1.], atol=1e-6)
    assert np.isclose(layout.cube_goal_world_xyz[0], .493, atol=1e-6)


def test_sweep_run_names_the_target_when_knowledge_rejects_opening_direction():
    scene = json.loads((SCENE / "scene.json").read_text())
    poses = {name: np.asarray(row["T_world_object"]).copy()
             for name, row in scene["objects"].items()}
    poses["cube_red"][:2, 3] = [.49, .30]

    class StubRun:
        concept_dir = SCENE / "concepts"
        report = {"knowledge": {}, "sweeps": {}, "failure_stage": None}

        def observe(self):
            return {"objects": {name: {"T_world_object": pose.tolist()}
                                for name, pose in poses.items()}}

    run = StubRun()
    assert not SweepCubeRun._sweep_target(run, "cube_red", "cube_red")
    assert run.report["failure_stage"] == "cube_red__knowledge"
    assert "簸箕开口" in run.report["knowledge"]["cube_red"]["error"]


def test_sequence_orientation_uses_sweep_frame_not_initial_brush_yaw():
    from gene_agent.knowledge.sweep import make_sweep_pose_sequences
    from gene_agent.knowledge.geometry import load_concept
    scene = json.loads((SCENE / "scene.json").read_text())
    poses = {n: np.asarray(r["T_world_object"]) for n, r in scene["objects"].items()}
    tcp = make_brush_grasp_pose(poses["brush"], SCENE / "concepts/brush.json")
    grip = np.linalg.inv(tcp) @ poses["brush"]
    direction = make_sweep_layout(poses, SCENE / "concepts").direction_world_xy
    face = load_concept(SCENE / "concepts/brush.json", part_name="bristle_face").concept
    rows = make_sweep_pose_sequences(tcp, poses, SCENE / "concepts", orientation_policy="functional")
    assert len(rows) == 6
    for row in rows:
        world_face = np.asarray(row["targets"]["lower_to_contact"]) @ grip @ face.pose_in_object
        assert np.dot(world_face[:2, 0], direction) == pytest.approx(0., abs=1e-9)
        assert world_face[2, 0] == pytest.approx(0., abs=1e-9)
        assert row["face_low_z_m"] == pytest.approx(.002)


def test_oblique_front_approach_is_not_rejected_by_arbitrary_angle():
    scene = json.loads((SCENE / "scene.json").read_text())
    poses = {n: np.asarray(r["T_world_object"]).copy() for n, r in scene["objects"].items()}
    poses["cube"][:2, 3] = [.281, .0004]
    layout = make_sweep_layout(poses, SCENE / "concepts")
    cosine = float(np.dot(layout.direction_world_xy, -poses["dustpan"][:2, 0]))
    assert 0. < cosine < .7


def test_explicit_staging_goal_allows_sideways_sweep_outside_pan():
    scene = json.loads((SCENE / "scene.json").read_text())
    poses = {n: np.asarray(r["T_world_object"]).copy() for n, r in scene["objects"].items()}
    poses["cube"][:3, 3] = [.28, -.05, 0.]
    goal = np.array([.38, -.05, .0112])
    layout = make_sweep_layout(poses, SCENE / "concepts", cube_goal_world_xyz=goal)
    np.testing.assert_allclose(layout.direction_world_xy, [1., 0.])
    np.testing.assert_allclose(layout.cube_goal_world_xyz, goal)


def test_fixed_brush_reference_prevents_accumulated_tilt():
    from scipy.spatial.transform import Rotation
    from gene_agent.knowledge.sweep import make_sweep_pose_sequences
    scene = json.loads((SCENE / "scene.json").read_text())
    poses = {n: np.asarray(r["T_world_object"]).copy() for n, r in scene["objects"].items()}
    tcp = make_brush_grasp_pose(poses["brush"], SCENE / "concepts/brush.json")
    reference = poses["brush"][:3, :3].copy()
    original = make_sweep_pose_sequences(tcp, poses, SCENE / "concepts", reference_brush_rotation=reference)
    grip = np.linalg.inv(tcp) @ poses["brush"]
    poses["brush"][:3, :3] = Rotation.from_euler("x", 20., degrees=True).as_matrix() @ reference
    tcp2 = poses["brush"] @ np.linalg.inv(grip)
    repeated = make_sweep_pose_sequences(tcp2, poses, SCENE / "concepts", reference_brush_rotation=reference)
    assert len(original) == len(repeated)
    for one, two in zip(original, repeated):
        np.testing.assert_allclose(np.asarray(one["targets"]["sweep"])[:3, :3],
                                   np.asarray(two["targets"]["sweep"])[:3, :3], atol=1e-8)
