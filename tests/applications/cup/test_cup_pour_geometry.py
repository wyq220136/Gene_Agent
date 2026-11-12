"""杯把从上向下抓取的 frame 与实际接触语义。"""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

from gene_agent.tasks.policies.cup_pour_task.pick_backend import (
    CupPickRun, classify_cup_handle_contacts, top_down_cup_handle_grasp)


SCENE = Path(__file__).resolve().parents[3] / "data/realworld/my_scene_0/scenes/pour"


def test_cup_pick_uses_source_container_role_and_covers_handle(tmp_path):
    run = CupPickRun(SCENE, tmp_path / "run", simulation_weld=True)
    assert run.object_name == "cup_green"
    assert run.report["skill"] == "pick_cup_handle_top_down"
    assert run.object_spheres.shape[1] == 4
    assert len(run.object_spheres) > 10
    allowed = run._allowed_contacts("lift")
    assert ("cup_green", "table") in allowed
    assert ("cup_green", "pour_ball_0") in allowed


def test_top_down_grasp_follows_cup_yaw_and_keeps_arm_above_handle():
    cup = np.eye(4)
    cup[:3, :3] = Rotation.from_euler("z", 25, degrees=True).as_matrix()
    cup[:3, 3] = [.43, -.16, 0.]
    handle = np.eye(4)
    handle[:3, 3] = [0., .068, .071]
    grasp = top_down_cup_handle_grasp(cup, handle)
    expected = (cup @ np.r_[np.array([0., .074, .0815]), 1.])[:3]
    assert np.allclose(grasp[:3, 3], expected)
    assert np.allclose(grasp[:3, 0], [0., 0., -1.], atol=1e-8)
    assert np.allclose(grasp[:3, 1], cup[:3, 0], atol=1e-8)
    assert np.linalg.det(grasp[:3, :3]) > .999


def test_only_bilateral_handle_proxy_contacts_count_as_cup_grasp():
    sites = {"left": np.array([.004, -.09, .095]),
             "right": np.array([-.004, -.09, .095])}
    center = np.array([0., -.09, .095])
    def contact(side, geom):
        return {"body_names": ["cup_green", f"pika_gripper_{side}_link"],
                "geom_names": [geom, f"pika_gripper_{side}_link_collision_1"],
                "position_world_m": sites[side].tolist(), "dist_m": -.0001}
    good = [contact("left", "cup_green_handle_top_proxy"),
            contact("right", "cup_green_handle_top_proxy")]
    assert classify_cup_handle_contacts(good, sites, center)["success"]
    bad = [good[0], contact("right", "cup_green_wall_0_proxy")]
    result = classify_cup_handle_contacts(bad, sites, center)
    assert not result["success"]
    assert not result["bilateral_pad_contact"]
