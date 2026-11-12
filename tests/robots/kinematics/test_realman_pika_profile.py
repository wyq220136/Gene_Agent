"""第二套机器人配置：正装单臂、Pika 夹爪与米制开口。"""

import mujoco
import numpy as np
import pytest

from gene_agent.robots import build_robot, measure_gripper_width, set_gripper_width
from gene_agent.simulation import MujocoRuntime, build_scene
from gene_agent.motion import load_realman_motion_config
from gene_agent.motion.curobo.scene import export_collision_scene


def test_single_pika_robot_has_only_one_upright_arm_and_movable_fingers(tmp_path):
    metadata = build_robot(tmp_path / "robot", profile="realman_single_pika")
    model = mujoco.MjModel.from_xml_path(metadata["xml_path"])
    assert metadata["profile"] == "realman_single_pika"
    assert tuple(metadata["joint_groups"]) == ("left",)
    assert len(metadata["joint_groups"]["left"]) == 7
    assert model.njnt == 9
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_body") < 0
    assert np.allclose(metadata["transforms"]["left"]["world_from_urdf_base"], np.eye(4))
    assert model.site(metadata["tcp_sites"]["left"]).id >= 0


def test_single_pika_width_command_moves_both_fingers_without_teleport(tmp_path):
    metadata = build_robot(tmp_path / "robot", profile="realman_single_pika")
    model = mujoco.MjModel.from_xml_path(metadata["xml_path"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gripper = metadata["grippers"]["left"]
    assert gripper["mode"] == "pika_urdf"
    maximum = gripper["width_range_m"][1]
    assert .09 < maximum < .10
    assert measure_gripper_width(model, data, metadata, "left")["width_m"] == pytest.approx(maximum, abs=1e-4)
    before = data.qpos.copy()
    set_gripper_width(model, data, metadata, "left", .04)
    np.testing.assert_array_equal(data.qpos, before)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    result = measure_gripper_width(model, data, metadata, "left")
    assert result["width_m"] == pytest.approx(.04, abs=.002)
    assert result["symmetry_error_m"] < .002
    assert np.isfinite(data.qpos).all()


def test_scene_and_runtime_query_select_single_pika_profile(tmp_path):
    model, data, metadata = build_scene(tmp_path / "scene", robot_profile="realman_single_pika")
    assert metadata["robot"]["profile"] == "realman_single_pika"
    assert metadata["world_frame"] == "single_robot_base"
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_body") < 0
    runtime = MujocoRuntime(model, data, metadata)
    robot = runtime.query(robot_groups=("left",))["robots"]["left"]
    assert len(robot["joint_names"]) == 7
    assert robot["gripper"]["available"] is True
    assert robot["gripper"]["width_m"] == pytest.approx(
        metadata["robot"]["grippers"]["left"]["width_range_m"][1], abs=1e-4)
    result = runtime.execute_gripper_command(
        group="left", width_m=.04, duration_s=2.0,
        planned_state_id=runtime.query(robot_groups=("left",))["state_id"],
    )
    assert result["success"] is True
    assert runtime.query(robot_groups=("left",))["robots"]["left"]["gripper"]["width_m"] == pytest.approx(.04, abs=.003)


def test_pika_planner_config_uses_its_own_urdf_and_current_width(tmp_path):
    model, data, metadata = build_scene(tmp_path / "scene", robot_profile="realman_single_pika")
    snapshot = export_collision_scene(model, data, metadata["robot"], "left", tmp_path / "collision")
    settings = load_realman_motion_config("left", metadata["robot"],
                                          gripper_snapshot=snapshot["gripper_snapshot"])
    kin = settings["robot_config"]["kinematics"]
    assert kin["urdf_path"] == metadata["robot"]["planning_urdf_path"]
    assert "pika_gripper_base_link" in kin["collision_spheres"]
    assert "l_inspire_jaw_base_link" not in kin["collision_spheres"]
    assert set(kin["lock_joints"]) == set(metadata["robot"]["grippers"]["left"]["joint_names"])


def test_pika_base_proxy_covers_hull_without_filling_empty_aabb_cells():
    """斜面外空格应裁掉；顶点、表面与内部采样仍须覆盖。"""
    import trimesh
    from gene_agent.motion.curobo.profiles.pika import cover_convex_mesh
    # 四面体的斜面外有大量AABB空格，不能用一个填满包围盒的代理。
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]],
                           faces=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    spheres = cover_convex_mesh(mesh, max_cell_m=.02)
    centers = np.array([row['center'] for row in spheres])
    radii = np.array([row['radius'] for row in spheres])
    rng = np.random.default_rng(31)
    weights = rng.dirichlet(np.ones(4), 2000)
    points = np.vstack([mesh.vertices, weights @ mesh.vertices])
    assert np.all(np.min(np.linalg.norm(points[:, None] - centers, axis=-1) - radii, axis=1) <= 1e-8)
    # 已知净空点不该被保守代理填满；已知内部点必须仍碰撞。
    assert np.min(np.linalg.norm(centers - [.09, .09, .09], axis=1) - radii) > .02
    assert np.min(np.linalg.norm(centers - [.02, .02, .02], axis=1) - radii) < 0


def test_pika_proxy_known_sweep_clearance_and_mesh_coverage():
    """seed38归拢后：物理mesh净空30.732mm，旧AABB球却重叠0.225mm。"""
    import trimesh
    from gene_agent.motion.curobo.profiles.pika import _pika_config, MESH_ROOT
    config = _pika_config({'planning_urdf_path': '/tmp/pika-clearance-regression.urdf'})
    spheres = config['kinematics']['collision_spheres']
    base = spheres['pika_gripper_base_link']
    centers = np.array([s['center'] for s in base])
    radii = np.array([s['radius'] for s in base])
    # T_link2_pika_base，由失败时真实qpos的MuJoCo FK得到；不依赖历史run。
    transform = np.array([
        [.7241905159390749, -.6669032565736757, .1754655037188884, -.031128808504469502],
        [.676097576783442, .6365375364820274, -.3710957171902551, -.027385706922975844],
        [.1357945628199256, .38737580076821815, .9118661226779551, .11089315625557704],
        [0, 0, 0, 1]])
    base_in_link2 = centers @ transform[:3, :3].T + transform[:3, 3]
    arm_centers = np.array([s['center'] for s in spheres['link_2']])
    arm_radii = np.array([s['radius'] for s in spheres['link_2']])
    gaps = np.linalg.norm(base_in_link2[:, None] - arm_centers, axis=-1) - radii[:, None] - arm_radii
    assert gaps.min() > .003
    vertices = np.unique(trimesh.load_mesh(MESH_ROOT / 'gripper_base_link.STL', process=False).vertices, axis=0)
    for batch in np.array_split(vertices, 30):
        assert np.all(np.min(np.linalg.norm(batch[:, None] - centers, axis=-1) - radii, axis=1) <= 1e-8)
