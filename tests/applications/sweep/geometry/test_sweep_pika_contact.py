"""Pika 接触证据来自两侧真实 finger mesh 的内侧区域。"""

import json
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.robots.contact.pika import classify_pika_brush_grasp
from gene_agent.feedback.state import snapshot
from gene_agent.knowledge.geometry import load_concept
from gene_agent.knowledge.sweep import make_brush_grasp_pose
from gene_agent.motion.mujoco_ik import MujocoPosePlanner
from gene_agent.robots.gripper import set_gripper_width


SCENE = Path(__file__).resolve().parents[4] / "data/realworld/brush_cube_dustpan_0"


def test_pika_brush_grasp_needs_both_inner_finger_contacts():
    model = mujoco.MjModel.from_xml_path(str(SCENE / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    robot = json.loads((SCENE / "robot/robot_metadata.json").read_text())
    mujoco.mj_forward(model, data)
    brush = np.eye(4)
    brush[:3, :3] = data.body("brush").xmat.reshape(3, 3)
    brush[:3, 3] = data.body("brush").xpos
    planner = MujocoPosePlanner(model, data, robot, "left")
    target = make_brush_grasp_pose(brush, SCENE / "concepts/brush.json")
    joints, position_error, rotation_error = planner._solve(
        planner._copy(), target, np.asarray(data.qpos[planner.qpos_addresses]))
    assert position_error < .005 and rotation_error < .04
    data.qpos[planner.qpos_addresses] = joints
    set_gripper_width(model, data, robot, "left", .01)
    for name in robot["grippers"]["left"]["joint_names"]:
        joint = model.joint(name)
        data.qpos[joint.qposadr[0]] = data.ctrl[model.actuator(name + "_position").id]
    mujoco.mj_forward(model, data)
    contacts = snapshot(model, data)["contacts"]
    sites = {side: data.site(f"pika_{side}_inner").xpos.copy()
             for side in ("left", "right")}
    handle_part = load_concept(SCENE / "concepts/brush.json", part_name="grip_handle").concept
    handle = (brush @ handle_part.pose_in_object)[:3, 3]

    both = classify_pika_brush_grasp(contacts, sites, handle)
    left_only = classify_pika_brush_grasp(
        [row for row in contacts if "pika_gripper_right_link" not in row["body_names"]],
        sites, handle)

    assert both["success"] and both["bilateral_pad_contact"]
    assert both["contact_mode"] == "pika_inner_finger_mesh_region"
    assert not left_only["success"]
