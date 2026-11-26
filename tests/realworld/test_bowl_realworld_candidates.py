"""真机历史失败输入的几何阶段回归；不连接设备、不调用 GPU。"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gene_agent.knowledge import bowl_grasp_candidates
from gene_agent.knowledge.geometry import load_concept
from gene_agent.realworld.microwave.bowl_candidates import observed_bowl_candidates
from gene_agent.realworld.microwave.bowl_pose_check import BowlPoseTableError
from gene_agent.tasks.policies.microwave_task.planning import grasp


# 来自两次历史失败的 observed_scene_provenance，不作为策略特权输入。
FAILURE_BOWL_POSES = [
    [[-.6190934324662887, .025388668376606284, .7849068188335134, .14115592711144553],
     [-.7853122802646457, -.02360206906321638, -.6186498591791235, .13439812618408742],
     [.0028185804632173393, -.999398852693444, .034549863978653506, -.06513460301239471],
     [0., 0., 0., 1.]],
    [[-.5770154118026375, -.06583515256047548, .8140754908230636, .14933335545553045],
     [-.8108627784853598, .16546953588266974, -.5613566602141263, .13849768167909426],
     [-.09774781740732938, -.9840148471324609, -.148861857490478, -.025057320607768774],
     [0., 0., 0., 1.]],
]


def _scene(world_sdk=None):
    world_sdk = np.eye(4) if world_sdk is None else world_sdk
    table = np.eye(4)
    table[:3, 3] = [.1, .3, -.045846251994371416]
    return {"snapshot": {"world_frame": "right_sdk_base"}, "provenance": {
        "T_world_sdk": world_sdk.tolist(), "table_pose_world": (world_sdk @ table).tolist(),
        "accepted_table_thickness_m": .02,
    }}


@pytest.mark.parametrize("bowl", FAILURE_BOWL_POSES)
def test_old_geometry_failure_does_not_remove_afford_depth_candidates(bowl):
    bowl = np.asarray(bowl)
    # 冻结旧自动参数；实测支点近似值只用于说明高度筛选根因，不参与新策略。
    old = bowl_grasp_candidates(bowl, parameters={
        "tip_y_rad": np.deg2rad([-35., -40., -30., -45., -25., -21.]).tolist(),
        "tcp_world_offset_m": [0., 0., -.012], "tilt_pivot_tcp_m": [0., 0., -.010451],
    })
    heights = [grasp.bowl_grasp_geometry_metrics(
        p, object_pose_world=bowl, base_position_world=np.zeros(3),
        table_normal_world=np.array([0., 0., 1.]),
    )["contact_height_m"] for p in grasp.legacy_bowl_proposals(old)]
    assert np.all(np.abs(heights) > .012)  # Ring 半高10 mm + 高度容差2 mm。
    with pytest.raises(ValueError, match="没有满足右臂基座侧、外侧碗沿约束"):
        grasp.base_side_bowl_proposals(
            old, object_pose_world=bowl, base_position_world=np.zeros(3),
            enforce_table_angle=False,
        )
    try:
        candidates, evidence = observed_bowl_candidates(
            _scene(), bowl, pose_source="frozen_observation",
        )
    except BowlPoseTableError:
        # 第一份历史输入已被同帧桌面一致性检查确认上下颠倒；候选策略
        # 不得绕过该感知错误继续生成动作。
        assert np.allclose(bowl, np.asarray(FAILURE_BOWL_POSES[0]))
        return
    assert len(candidates) == len(evidence["candidate_targets"]) == 160
    assert {c["parameters"]["depth_m"] for c in candidates} == {.015, .025, .028, .03}
    table_z = evidence["table_height_sdk_m"]
    for candidate, adjustment in zip(candidates, evidence["candidate_targets"]):
        target = np.asarray(candidate["T_world_tcp"])
        original = np.asarray(adjustment["T_sdk_tcp_before_table_clamp"])
        np.testing.assert_allclose(target[:2], original[:2], atol=1e-12)
        np.testing.assert_allclose(target[:3, :3], original[:3, :3], atol=1e-12)
        assert table_z + .05 <= target[2, 3] <= table_z + .07
        assert target[2, 3] - original[2, 3] == pytest.approx(adjustment["table_clamp_delta_sdk_z_m"])
        np.testing.assert_allclose(bowl @ np.asarray(candidate["T_object_tcp"]), target, atol=1e-12)
        concept = load_concept(grasp.ROOT / "assets/objects/bowl_concept.json", part_name="bowl_rim").concept
        np.testing.assert_allclose(bowl @ concept.pose_in_object @ np.asarray(candidate["T_primitive_tcp"]),
                                   target, atol=1e-12)


def test_real_candidates_are_invariant_to_planning_world_frame():
    bowl = np.asarray(FAILURE_BOWL_POSES[1])
    world_sdk = np.eye(4)
    world_sdk[:3, :3] = Rotation.from_euler("xyz", [.3, -.2, .8]).as_matrix()
    world_sdk[:3, 3] = [.5, -.1, .7]
    reference, _ = observed_bowl_candidates(_scene(), bowl, pose_source="test")
    transformed_scene = _scene(world_sdk)
    transformed_scene["snapshot"]["world_frame"] = "calibrated_world"
    transformed, _ = observed_bowl_candidates(transformed_scene, world_sdk @ bowl, pose_source="test")
    assert [c["candidate_id"] for c in reference] == [c["candidate_id"] for c in transformed]
    for ref, actual in zip(reference, transformed):
        np.testing.assert_allclose(actual["T_world_tcp"], world_sdk @ ref["T_world_tcp"], atol=1e-12)


def test_real_candidates_require_measured_table_and_known_frame():
    scene = _scene()
    del scene["provenance"]["table_pose_world"]
    with pytest.raises(ValueError, match="已确认的桌面"):
        observed_bowl_candidates(scene, FAILURE_BOWL_POSES[0], pose_source="test")
    scene = _scene()
    scene["snapshot"]["world_frame"] = "unknown"
    del scene["provenance"]["T_world_sdk"]
    with pytest.raises(ValueError, match="缺少 T_world_sdk"):
        observed_bowl_candidates(scene, FAILURE_BOWL_POSES[0], pose_source="test")


def test_geometry_metrics_do_not_reject_inward_or_degenerate_base():
    bowl = np.asarray(FAILURE_BOWL_POSES[1])
    candidates, _ = observed_bowl_candidates(_scene(), bowl, pose_source="test")
    ring = load_concept(grasp.ROOT / "assets/objects/bowl_concept.json", part_name="bowl_rim").concept
    center = (bowl @ ring.pose_in_object)[:3, 3]
    metrics = [grasp.bowl_grasp_geometry_metrics(
        p, object_pose_world=bowl, base_position_world=center, table_normal_world=np.array([0., 0., 1.]),
    ) for p in grasp.legacy_bowl_proposals(candidates)]
    assert all(item["base_alignment"] is None for item in metrics)
    assert any(item["external_approach_alignment"] < 0 for item in metrics)
    with pytest.raises(ValueError, match="桌面法向"):
        grasp.bowl_grasp_geometry_metrics(
            grasp.legacy_bowl_proposals(candidates)[0], object_pose_world=bowl,
            base_position_world=center, table_normal_world=np.zeros(3),
        )


def test_explicit_proposals_skip_only_geometry_and_still_prepare_collision(monkeypatch):
    bowl = np.asarray(FAILURE_BOWL_POSES[1])
    candidates, _ = observed_bowl_candidates(_scene(), bowl, pose_source="test")
    case = SimpleNamespace(
        metadata={"robot": {"joint_groups": {"right": ["joint"]}}},
        data=SimpleNamespace(qpos=[0.], geom=lambda name: SimpleNamespace(xmat=np.eye(3))),
        model=SimpleNamespace(joint=lambda name: SimpleNamespace(qposadr=[0])),
    )
    monkeypatch.setattr(grasp, "active_arm_base_reference", lambda *a: ("base", np.zeros(3)))
    monkeypatch.setattr(grasp, "base_side_bowl_proposals", lambda *a, **k: pytest.fail("仿真筛选被调用"))

    class CollisionPreparationReached(Exception):
        pass

    def prepare(current, opening, *, group):
        assert current is case and group == "right" and opening == .041
        raise CollisionPreparationReached

    monkeypatch.setattr(grasp, "prepare_grasp_stage_snapshots", prepare)
    with pytest.raises(CollisionPreparationReached):
        grasp.plan_bowl_grasp_stages(
            case, SimpleNamespace(max_grasp_candidates=160, open_width_m=.041),
            proposals=grasp.legacy_bowl_proposals(candidates), enforce_table_angle=False,
        )


def test_live_selection_keeps_all_candidates_from_old_zero_candidate_case():
    from gene_agent.realworld.microwave.pick_planning import _affordtoolact_proposals
    bowl = np.asarray(FAILURE_BOWL_POSES[1])
    candidates, _ = observed_bowl_candidates(_scene(), bowl, pose_source="frozen_observation")
    # 同一历史初态在旧真机仿真过滤中160→0；新入口应保持全部候选及原顺序。
    with pytest.raises(ValueError, match="没有满足"):
        grasp.base_side_bowl_proposals(
            candidates, object_pose_world=bowl, base_position_world=np.zeros(3),
            table_normal_world=np.array([0., 0., 1.]),
            min_external_approach_alignment=-1., rim_height_tolerance_m=.03,
        )
    proposals = _affordtoolact_proposals(SimpleNamespace(candidates=candidates))
    assert len(proposals) == 160
    assert [p.candidate_id for p in proposals] == [c["candidate_id"] for c in candidates]
    for candidate, proposal in zip(candidates, proposals):
        np.testing.assert_array_equal(candidate["T_world_tcp"], proposal.tcp_pose_world)
