"""物理开门后夹爪靠近最大开口的微小越界不能中断下一段 Skill。"""

import mujoco
import pytest


def test_real_mujoco_gripper_accepts_micrometer_limit_overshoot_but_rejects_large_one(tmp_path):
    from gene_agent.feedback.state import state_id
    from gene_agent.simulation import MujocoRuntime, build_scene

    model, data, metadata = build_scene(tmp_path / "scene")
    runtime = MujocoRuntime(model, data, metadata)
    positive = model.joint("right_gripper_positive_slide").qposadr[0]
    negative = model.joint("right_gripper_negative_slide").qposadr[0]
    data.qpos[positive] = .040006
    data.qpos[negative] = .04
    mujoco.mj_forward(model, data)

    result = runtime.execute_gripper_command(
        group="right", width_m=.041, duration_s=.02,
        planned_state_id=state_id(model, data),
    )
    assert result["success"]
    assert result["start_width_m"] > .08

    data.qpos[positive] = .041
    data.qpos[negative] = .04
    mujoco.mj_forward(model, data)
    with pytest.raises(ValueError, match="实测夹爪开口"):
        runtime.execute_gripper_command(
            group="right", width_m=.041, duration_s=.02,
            planned_state_id=state_id(model, data),
        )


def test_real_mujoco_gripper_can_open_after_small_negative_dynamic_overshoot(tmp_path):
    from gene_agent.feedback.state import state_id
    from gene_agent.simulation import MujocoRuntime, build_scene

    model, data, metadata = build_scene(tmp_path / "scene")
    runtime = MujocoRuntime(model, data, metadata)
    positive = model.joint("right_gripper_positive_slide").qposadr[0]
    negative = model.joint("right_gripper_negative_slide").qposadr[0]
    data.qpos[positive] = -.00008
    data.qpos[negative] = 0.
    mujoco.mj_forward(model, data)

    result = runtime.execute_gripper_command(
        group="right", width_m=.02, duration_s=.02,
        planned_state_id=state_id(model, data),
    )
    assert result["success"]
    assert result["start_width_m"] == pytest.approx(-.00008)
    assert min(sample["reference_width_m"] for sample in result["samples"]) >= 0.
