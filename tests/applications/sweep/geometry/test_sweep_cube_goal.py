"""目标成功必须由实际 MuJoCo 物体状态和托盘接触支持。"""

from pathlib import Path

import mujoco
import numpy as np

from benchmarks.sweep_cube_task.evaluation import evaluate_cube_in_dustpan


SCENE = Path(__file__).resolve().parents[4] / "data/realworld/brush_cube_dustpan_0"


def test_cube_goal_requires_inside_and_tray_support():
    model = mujoco.MjModel.from_xml_path(str(SCENE / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    assert not evaluate_cube_in_dustpan(model, data, SCENE / "concepts")["success"]

    cube_joint = model.joint("cube_free")
    pan = data.body("dustpan")
    inward = -pan.xmat.reshape(3, 3)[:3, 0]
    data.qpos[cube_joint.qposadr[0]:cube_joint.qposadr[0] + 3] = (
        pan.xpos + inward * .02 + [0., 0., .031])
    mujoco.mj_forward(model, data)
    for _ in range(400):
        mujoco.mj_step(model, data)
    goal = evaluate_cube_in_dustpan(model, data, SCENE / "concepts")
    assert goal["inside_xy"], goal
    assert goal["tray_contact"], goal
    assert goal["success"], goal
    assert not evaluate_cube_in_dustpan(model, data, SCENE / "concepts",
                                        object_name="cube_red")["success"]


def test_rotated_cube_corner_outside_tray_is_rejected():
    from scipy.spatial.transform import Rotation
    model = mujoco.MjModel.from_xml_path(str(SCENE / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    pan = data.body("dustpan")
    pan_rotation = pan.xmat.reshape(3, 3).copy()
    address = int(model.joint("cube_free").qposadr[0])
    data.qpos[address:address + 3] = pan.xpos + pan_rotation @ [.066, 0., .012]
    rotation = pan_rotation @ Rotation.from_euler("z", 45., degrees=True).as_matrix()
    data.qpos[address + 3:address + 7] = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)
    result = evaluate_cube_in_dustpan(model, data, SCENE / "concepts")
    assert not result["inside_xy"]
