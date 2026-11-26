"""冻结同一场景：真机保留 AffordToolAct 候选，仿真继续使用其几何筛选。"""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gene_agent.realworld.microwave import pick_planning as real
from gene_agent.simulation import MujocoRuntime
from gene_agent.tasks.policies.microwave_task.planning import grasp
from gene_agent.tasks.policies.microwave_task.planning.bowl import prepare_pick_bowl_scene, load_pick_bowl_case
from gene_agent.tasks.policies.microwave_task.planning.bowl_candidates import auto_bowl_grasp_parameters
from gene_agent.tasks.policies.microwave_task.simulation_bowl import pick as sim

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("dx,dy,yaw_deg", [(0., 0., 0.), (.07, -.03, 35.), (-.03, .06, -60.)])
def test_real_afford_candidates_bypass_simulation_geometry_filter(tmp_path, monkeypatch, dx, dy, yaw_deg):
    config = json.loads((ROOT / "config/scenes/task_1.json").read_text())
    bowl = np.array(config["objects"]["bowl"]["simulation_nominal_pose"], dtype=float)
    bowl[:2, 3] += [dx, dy]
    bowl[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ bowl[:3, :3]
    case = prepare_pick_bowl_scene(tmp_path / "scene", pose_overrides={"bowl": bowl})
    runtime = MujocoRuntime(case.model, case.data, case.metadata)
    frozen_qpos = case.data.qpos.copy()
    frozen_qvel = case.data.qvel.copy()
    table_pose = np.eye(4)
    table_pose[:3, :3] = case.data.geom("scene_table").xmat.reshape(3, 3)
    table_pose[:3, 3] = case.data.geom("scene_table").xpos
    scene = {
        "snapshot": case.snapshot, "scene_xml_path": case.metadata["xml_path"],
        "scene_metadata": case.metadata,
        "provenance": {
            "physics_stepped": False, "pose_frame_verified": True,
            "collision_snapshot_sha256": case.snapshot["snapshot_sha256"],
            "aligned_object_poses_world": {"Bowl/Bowl_0": case.bowl_pose.tolist()},
            "source_frame_number": 1, "source_timestamp_ms": 1000.,
            "scene_config_sha256": "synthetic-same-scene",
            "T_world_sdk": np.eye(4).tolist(),
            "table_pose_world": table_pose.tolist(),
            "accepted_table_thickness_m": config["table"]["thickness_m"],
        },
    }
    seen = []

    class UnselectedPlan:
        selected_candidate_id = None
        attempts = ()

        def stage(self, name):
            return SimpleNamespace(status="failed")

        def to_dict(self):
            return {"selected_candidate_id": None, "stages": []}

    def select(current, settings, *, seed, enforce_table_angle=True, proposals=None):
        _, base = grasp.active_arm_base_reference(current)
        if proposals is None:
            assert len(seen) == 0
            assert enforce_table_angle is True
            proposals = grasp.base_side_bowl_proposals(
                current.candidates, object_pose_world=current.bowl_pose,
                base_position_world=base,
                table_normal_world=current.data.geom("scene_table").xmat.reshape(3, 3)[:, 2],
                enforce_table_angle=enforce_table_angle,
            )
        else:
            assert len(current.candidates) == 160
            assert len(proposals) == 160
            assert enforce_table_angle is False
            assert [p.candidate_id for p in proposals] == [c["candidate_id"] for c in current.candidates]
        seen.append((current.candidates, proposals, settings, seed))
        return SimpleNamespace(
            plan=UnselectedPlan(), proposals=proposals,
            strict_snapshot={"snapshot_sha256": "strict"},
            contact_snapshot={"snapshot_sha256": "contact"},
        )

    monkeypatch.setattr(grasp, "plan_bowl_grasp_stages", select)
    monkeypatch.setattr(real, "plan_bowl_grasp_stages", select)
    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **k: pytest.fail("candidate generation stepped physics"))
    simulated = sim.run_experiment(
        tmp_path / "sim", runtime=runtime, candidate_id="auto",
        skill_profile="pick_bowl_legacy_visual", seed=123,
    )
    report = real.plan_observed_bowl_grasp_stages(scene, output_dir=tmp_path / "real", seed=123)
    assert simulated["failure_stage"] == "candidate_selection"
    assert report["physical_pick_success"] is False
    assert len(seen) == 2
    sim_candidates, sim_proposals, sim_settings, sim_seed = seen[0]
    real_candidates, real_proposals, real_settings, real_seed = seen[1]
    assert len(sim_candidates) == len(case.candidates) == 48
    assert len(real_candidates) == 160
    assert len(real_proposals) == 160
    reopened = load_pick_bowl_case(case.output_dir)
    for expected, displayed, loaded in zip(sim_candidates, case.candidates, reopened.candidates):
        assert displayed["candidate_id"] == loaded["candidate_id"] == expected["candidate_id"]
        np.testing.assert_allclose(displayed["T_world_tcp"], expected["T_world_tcp"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(loaded["T_world_tcp"], expected["T_world_tcp"], atol=1e-12, rtol=0)
    assert len({item["candidate_id"] for item in real_candidates}) == 160
    assert {item["parameters"]["depth_m"] for item in real_candidates} == {.015, .025, .028, .03}
    for actual in real_candidates:
        assert actual["pose_source"] == "observed_scene_asset_aligned"
        assert len(actual["parameters"]["tip_y_rad"]) == 5
        assert "tilt_pivot_tcp_m" not in actual["parameters"]["explicit_overrides"]
        np.testing.assert_allclose(actual["parameters"]["tilt_pivot_tcp_m"], 0.)
        np.testing.assert_allclose(np.asarray(actual["T_world_tcp"]),
                                   bowl @ np.asarray(actual["T_object_tcp"]), atol=1e-12)
    displayed = real.build_observed_bowl_grasp_proposals(scene)["proposals"]
    assert [p.candidate_id for p in displayed] == [p.candidate_id for p in real_proposals]
    assert real_settings.max_grasp_candidates == 160
    assert real_settings.pregrasp_search == "batch_all"
    assert real_settings.align_before_approach is True
    assert real_seed == sim_seed == 123
    assert report["generated_candidate_count"] == 160
    assert report["geometry_filtered_candidate_count"] == 160
    assert report["grasp_table_angle_constraint_enabled"] is False
    assert report["grasp_table_angle_allowed_deg"] is None
    assert report["grasp_table_angle_preferred_deg"] is None
    assert report["simulation_geometry_pruning_enabled"] is False
    assert {item.parameters["depth_m"] for item in real_proposals} == {.015, .025, .028, .03}
    angles = []
    for item in real_proposals:
        metrics = grasp.bowl_grasp_geometry_metrics(
            item, object_pose_world=case.bowl_pose,
            base_position_world=grasp.active_arm_base_reference(case)[1],
            table_normal_world=case.data.geom("scene_table").xmat.reshape(3, 3)[:, 2],
        )
        angles.append(metrics["table_angle_deg"])
    assert any(angle > 70. for angle in angles)
    evidence = json.loads((tmp_path / "real/candidate_selection.json").read_text())
    assert evidence["simulation_geometry_pruning_enabled"] is False
    assert evidence["grasp_table_angle_constraint_enabled"] is False
    assert evidence["geometry_filtered_candidate_count"] == 160
    assert evidence["geometry_rejected_candidate_count"] == 0
    assert len(report["candidate_parameters"]["candidate_targets"]) == 160
    assert len(report["candidate_parameters"]["profile_sha256"]) == 64
    np.testing.assert_array_equal(case.data.qpos, frozen_qpos)
    np.testing.assert_array_equal(case.data.qvel, frozen_qvel)
    assert case.data.time == 0.
    # 自动筛选与仿真保持相同的 profile 限制，不悄悄换一套动作参数。
    with pytest.raises(ValueError, match="自动候选筛选"):
        real.plan_observed_bowl_grasp_stages(scene, output_dir=tmp_path / "invalid", profile_name="pick_bowl")
    assert len(seen) == 2


def test_auto_pivot_is_tcp_local_and_depth_override_is_preserved():
    rotation = Rotation.from_euler("xyz", [.2, -.4, .6]).as_matrix()
    position = np.array([.3, .5, .7])
    pivot = np.array([0., .001, -.02])
    pad_center = position + rotation @ pivot
    pads = np.array([pad_center + rotation[:, 0] * .02, pad_center - rotation[:, 0] * .02])
    data = SimpleNamespace(
        site=lambda name: SimpleNamespace(xmat=rotation.reshape(-1), xpos=position),
        geom_xpos=pads,
    )
    model = SimpleNamespace(geom=lambda name: SimpleNamespace(id={"positive": 0, "negative": 1}[name]))
    robot = {"tcp_sites": {"right": "tcp"}, "grippers": {"right": {
        "pad_geom_names": ["positive", "negative"]}}}
    params = auto_bowl_grasp_parameters(model, data, robot, depth_m=.018)
    np.testing.assert_allclose(params["tilt_pivot_tcp_m"], pivot, atol=1e-12)
    assert params["depth_m"] == .018
    np.testing.assert_allclose(np.rad2deg(params["tip_y_rad"]), [-35., -40., -30., -45., -25., -21.])
    assert params["tcp_world_offset_m"] == [0., 0., -.012]
    with pytest.raises(NotImplementedError):
        auto_bowl_grasp_parameters(model, data, robot, group="left")


def test_reopening_historical_case_keeps_original_candidate_geometry(tmp_path):
    """旧记录未保存 auto 参数时，不能把同名候选悄悄换成新的倾角。"""
    from gene_agent.knowledge import bowl_grasp_candidates
    case = prepare_pick_bowl_scene(tmp_path / "scene")
    path = case.output_dir / "case_meta.json"
    meta = json.loads(path.read_text())
    assert "candidate_parameters" in meta
    meta.pop("candidate_parameters")
    path.write_text(json.dumps(meta))
    reopened = load_pick_bowl_case(case.output_dir)
    original = bowl_grasp_candidates(case.bowl_pose)
    assert len(reopened.candidates) == len(original) == 40
    for expected, actual in zip(original, reopened.candidates):
        assert actual["candidate_id"] == expected["candidate_id"]
        np.testing.assert_allclose(actual["T_world_tcp"], expected["T_world_tcp"], atol=1e-12, rtol=0)
