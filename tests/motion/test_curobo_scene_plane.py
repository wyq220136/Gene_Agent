import mujoco
import numpy as np

from gene_agent.motion.curobo.scene import export_collision_scene, filter_collision_snapshot


def test_plane_export_keeps_top_surface_as_obstacle(tmp_path):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="floor" type="plane" pos="0 0 -0.04" size="2 2 0.01"/>
      <body name="robot_root"><joint name="arm_joint" type="hinge"/>
        <geom name="robot_link" type="sphere" size="0.01" pos="0 0 0.3"/>
      </body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    metadata = {
        "joint_groups": {"left": ["arm_joint"]},
        "group_root_bodies": {"left": "robot_root"},
        "transforms": {"left": {"world_from_urdf_base": np.eye(4).tolist()}},
        "world_frame": "world", "grippers": {},
    }
    result = export_collision_scene(model, data, metadata, "left", tmp_path)
    obstacle = result["scene_model"]["cuboid"]["mj_geom_0_floor"]
    assert obstacle["dims"][0] >= 4.
    assert obstacle["dims"][1] >= 4.
    assert obstacle["pose"][2] + obstacle["dims"][2] / 2 == -0.04
    assert result["records"][0]["approximation"] == "finite_plane_support_cuboid"

    filtered = filter_collision_snapshot(
        result, tmp_path / "filtered", keep_body_names=None,
        rejected_body_names=None, rejected_geom_names={"floor"},
        attached_object_id=None, scope="contact stage: floor checked by MuJoCo")
    assert filtered["scene_model"]["cuboid"] == {}
    assert filtered["ignored_world_obstacles"] == ["mj_geom_0_floor"]
