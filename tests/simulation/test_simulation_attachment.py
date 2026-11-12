"""仿真特权固定必须有接触证据，并保持真实状态连续。"""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from gene_agent.simulation import MujocoRuntime, build_scene


def test_contact_gated_weld_preserves_pose_and_can_release(tmp_path: Path):
    model, data, metadata = build_scene(
        tmp_path / "scene", simulation_attachment_pairs={"right": "bowl"},
    )
    runtime = MujocoRuntime(model, data, metadata)
    before = data.body("bowl").xpos.copy()
    with pytest.raises(ValueError, match="接触"):
        runtime.attach_simulation_object("right", "bowl", contact_evidence={})
    assert not runtime.simulation_attachments()
    assert np.allclose(data.body("bowl").xpos, before)

    evidence = {"success": True, "bilateral_pad_contact": True,
                "contact_count": 2, "undesired_gripper_contacts": []}
    attached = runtime.attach_simulation_object("right", "bowl", contact_evidence=evidence)
    assert attached["mode"] == "simulation_privileged_weld"
    assert runtime.simulation_attachments()[0]["object_instance_id"] == "bowl"
    assert runtime.query(object_names=("bowl",), robot_groups=("right",))[
        "simulation_attachments"][0]["object_instance_id"] == "bowl"
    assert np.linalg.norm(data.body("bowl").xpos - before) < 1e-6
    for _ in range(20):
        mujoco.mj_step(model, data)
    runtime.release_simulation_object("right", "bowl")
    assert not runtime.simulation_attachments()


def test_attached_state_restores_on_fresh_model(tmp_path: Path):
    from gene_agent.feedback.state import restore_state, save_state

    scene_dir = tmp_path / "scene"
    model, data, metadata = build_scene(
        scene_dir, simulation_attachment_pairs={"right": "bowl"},
    )
    runtime = MujocoRuntime(model, data, metadata)
    runtime.attach_simulation_object(
        "right", "bowl", contact_evidence={"success": True,
        "bilateral_pad_contact": True, "contact_count": 2},
    )
    save_state(model, data, tmp_path / "attached.npz")
    fresh = mujoco.MjModel.from_xml_path(str(scene_dir / "scene.xml"))
    restored = mujoco.MjData(fresh)
    restore_state(fresh, restored, tmp_path / "attached.npz")
    eq_id = mujoco.mj_name2id(fresh, mujoco.mjtObj.mjOBJ_EQUALITY,
                              "sim_attach_right_bowl")
    assert restored.eq_active[eq_id] == 1
    assert np.allclose(fresh.eq_data[eq_id], model.eq_data[eq_id])


def test_gripper_snapshot_tolerates_contact_width_overshoot(tmp_path: Path):
    """门板推挤指垫时几十微米的关节弹性不能中断下一段规划。"""
    from gene_agent.motion.curobo.gripper import export_gripper_snapshot

    model, data, metadata = build_scene(tmp_path / "scene")
    descriptor = metadata["robot"]["grippers"]["right"]
    for name in descriptor["joint_names"]:
        data.qpos[model.joint(name).qposadr[0]] = 0.040012
    mujoco.mj_forward(model, data)
    snapshot = export_gripper_snapshot(model, data, metadata["robot"], "right")
    assert snapshot["width_m"] > descriptor["width_range_m"][1]
