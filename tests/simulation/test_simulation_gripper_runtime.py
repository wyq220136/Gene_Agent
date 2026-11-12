"""夹爪执行接口使用米制总开口、状态版本和真实 actuator 反馈。"""

import mujoco
import numpy as np
import pytest

from gene_agent.feedback.state import integration_state, state_id
from gene_agent.simulation import MujocoRuntime


def _runtime_fixture():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep=".002"/>
          <worldbody>
            <body name="arm"><joint name="arm_joint"/><geom size=".01"/><site name="tcp"/>
              <body name="positive"><joint name="positive_slide" type="slide" axis="1 0 0" damping="2"/>
                <geom name="positive_pad" type="box" size=".004 .01 .02"/><site name="positive_inner"/>
              </body>
              <body name="negative"><joint name="negative_slide" type="slide" axis="-1 0 0" damping="2"/>
                <geom name="negative_pad" type="box" size=".004 .01 .02"/><site name="negative_inner"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position name="arm_position" joint="arm_joint" kp="10"/>
            <position name="positive_position" joint="positive_slide" kp="300"/>
            <position name="negative_position" joint="negative_slide" kp="300"/>
          </actuator>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qpos[model.joint("positive_slide").qposadr[0]] = .02
    data.qpos[model.joint("negative_slide").qposadr[0]] = .02
    mujoco.mj_forward(model, data)
    robot = {
        "world_frame": "world",
        "joint_groups": {"right": ["arm_joint"]},
        "tcp_sites": {"right": "tcp"},
        "grippers": {
            "right": {
                "mode": "parallel_slide_proxy",
                "joint_names": ["positive_slide", "negative_slide"],
                "actuator_names": ["positive_position", "negative_position"],
                "width_range_m": [0.0, 0.08],
                "fingers": {
                    "positive": {"inner_site_name": "positive_inner"},
                    "negative": {"inner_site_name": "negative_inner"},
                },
            }
        },
    }
    return MujocoRuntime(model, data, {"world_frame": "world", "robot": robot})


def test_execute_gripper_command_interpolates_ctrl_and_reports_measured_width():
    runtime = _runtime_fixture()
    planned = runtime.query(robot_groups=("right",))["state_id"]
    initial = runtime.data.qpos.copy()

    result = runtime.execute_gripper_command(
        group="right", width_m=.02, duration_s=.01, planned_state_id=planned,
        replay_object_names=(), replay_interval_s=.004,
    )

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["start_state_id"] == planned
    assert result["end_state_id"] == state_id(runtime.model, runtime.data)
    assert len(result["replay_samples"]) >= 2
    assert result["replay_samples"][0]["state_id"] == result["start_state_id"]
    assert result["replay_samples"][-1]["state_id"] == result["end_state_id"]
    assert result["replay_samples"][-1]["robots"]["right"]["gripper"]["width_m"] == pytest.approx(
        result["samples"][-1]["actual_width_m"]
    )
    assert len(result["samples"]) == 5
    assert result["samples"][0]["reference_width_m"] == pytest.approx(.036)
    assert result["samples"][-1]["reference_width_m"] == pytest.approx(.02)
    assert np.isfinite(result["samples"][-1]["actual_width_m"])
    assert runtime.data.ctrl[runtime.model.actuator("positive_position").id] == pytest.approx(.01)
    assert runtime.data.ctrl[runtime.model.actuator("negative_position").id] == pytest.approx(.01)
    assert not np.array_equal(runtime.data.qpos, initial)
    assert runtime.data.qpos[runtime.model.joint("positive_slide").qposadr[0]] != pytest.approx(.01)


def test_gripper_command_accepts_roundoff_at_open_limit():
    """测量值比名义上限多几个浮点尾数时，控制参考仍应是合法开口。"""
    runtime = _runtime_fixture()
    for name in ("positive_slide", "negative_slide"):
        runtime.data.qpos[runtime.model.joint(name).qposadr[0]] = .040000002
    mujoco.mj_forward(runtime.model, runtime.data)
    planned = runtime.query(robot_groups=("right",))["state_id"]
    result = runtime.execute_gripper_command(
        group="right", width_m=.08, duration_s=.004, planned_state_id=planned,
    )
    assert result["success"] is True
    assert all(sample["reference_width_m"] <= .08 for sample in result["samples"])


def test_gripper_observer_receives_isolated_actual_states():
    runtime = _runtime_fixture()
    baseline = _runtime_fixture()
    observed = []

    def observer(model, frozen, sample):
        observed.append((frozen.time, sample["actual_width_m"]))
        frozen.qpos[:] = 99

    args = dict(group="right", width_m=.03, duration_s=.006)
    result = runtime.execute_gripper_command(
        **args,
        planned_state_id=runtime.query(robot_groups=("right",))["state_id"],
        step_observer=observer,
    )
    baseline.execute_gripper_command(
        **args,
        planned_state_id=baseline.query(robot_groups=("right",))["state_id"],
    )

    assert len(observed) == len(result["samples"]) == 3
    np.testing.assert_array_equal(runtime.data.qpos, baseline.data.qpos)


@pytest.mark.parametrize(
    "overrides",
    [
        {"planned_state_id": "stale"},
        {"width_m": -.001},
        {"width_m": np.nan},
        {"duration_s": 0.0},
        {"timeout_s": 0.0},
        {"step_observer": "not callable"},
    ],
)
def test_invalid_gripper_command_is_rejected_before_state_change(overrides):
    runtime = _runtime_fixture()
    arguments = dict(
        group="right",
        width_m=.02,
        duration_s=.01,
        planned_state_id=runtime.query(robot_groups=("right",))["state_id"],
        timeout_s=None,
        step_observer=None,
    )
    arguments.update(overrides)
    before = integration_state(runtime.model, runtime.data).copy()
    ctrl = runtime.data.ctrl.copy()

    with pytest.raises(ValueError):
        runtime.execute_gripper_command(**arguments)

    np.testing.assert_array_equal(integration_state(runtime.model, runtime.data), before)
    np.testing.assert_array_equal(runtime.data.ctrl, ctrl)


def test_gripper_timeout_stops_before_first_over_budget_step():
    runtime = _runtime_fixture()
    result = runtime.execute_gripper_command(
        group="right",
        width_m=.02,
        duration_s=.01,
        planned_state_id=runtime.query(robot_groups=("right",))["state_id"],
        timeout_s=.001,
    )
    assert result["success"] is False
    assert result["failure_type"] == "execution_timeout"
    assert result["samples"] == []
