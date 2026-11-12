import mujoco
import numpy as np
import pytest
from pathlib import Path
import subprocess
import sys

from gene_agent.feedback.state import integration_state, state_id
from gene_agent.simulation import MujocoRuntime


def test_simulation_package_imports_without_prior_feedback_import():
    """若 feedback 便捷导出反向导入 runtime，干净进程会出现循环导入。"""
    completed = subprocess.run(
        [sys.executable, "-c", "import gene_agent.simulation"],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _runtime_fixture():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco model="runtime_test">
          <option timestep="0.002"/>
          <worldbody>
            <body name="arm_base">
              <joint name="joint_b" type="hinge" axis="0 1 0" damping="1"/>
              <geom name="base_geom" type="capsule" size=".015 .08"/>
              <body name="arm_tip" pos="0 0 .2">
                <joint name="joint_a" type="hinge" axis="1 0 0" damping="1"/>
                <geom name="arm_geom" type="sphere" size=".02"/>
                <site name="tcp" pos="0 0 .1"/>
              </body>
            </body>
            <body name="bowl" pos=".4 .1 .2">
              <freejoint name="bowl_free"/>
              <geom name="bowl_geom" type="sphere" size=".05"/>
            </body>
          </worldbody>
          <actuator>
            <position name="joint_a_position" joint="joint_a" kp="20" gear="2"/>
            <position name="joint_b_position" joint="joint_b" kp="20"/>
          </actuator>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qpos[model.joint("joint_a").qposadr[0]] = 0.15
    data.qpos[model.joint("joint_b").qposadr[0]] = -0.25
    data.qvel[model.joint("joint_a").dofadr[0]] = 0.03
    mujoco.mj_forward(model, data)
    metadata = {
        "world_frame": "world",
        "robot": {
            "world_frame": "world",
            "joint_groups": {"right": ["joint_a", "joint_b"]},
            "tcp_sites": {"right": "tcp"},
            "grippers": {},
        },
    }
    return MujocoRuntime(model, data, metadata, scene_revision=7), model, data


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_query_returns_named_operational_state_without_mutating_simulation():
    """若 query 错在原 data 上 forward，这个测试会捕获积分状态变化。"""
    runtime, model, data = _runtime_fixture()
    before = integration_state(model, data).copy()
    before_id = state_id(model, data)

    observation = runtime.query(object_names=("bowl",), robot_groups=("right",))

    np.testing.assert_array_equal(integration_state(model, data), before)
    assert state_id(model, data) == before_id == observation["state_id"]
    assert observation["schema_version"] == 1
    assert observation["scene_revision"] == 7
    assert observation["sim_time_s"] == 0.0
    assert observation["world_frame"] == "world"
    np.testing.assert_allclose(
        np.asarray(observation["objects"]["bowl"]["T_world_object"])[:3, 3],
        [0.4, 0.1, 0.2],
    )
    assert observation["robots"]["right"]["joint_names"] == ["joint_a", "joint_b"]
    np.testing.assert_allclose(observation["robots"]["right"]["joint_positions_rad"], [0.15, -0.25])
    np.testing.assert_allclose(observation["robots"]["right"]["joint_velocities_rad_s"], [0.03, 0.0])
    assert observation["robots"]["right"]["gripper"] == {"available": False}
    assert np.asarray(observation["robots"]["right"]["T_world_tcp"]).shape == (4, 4)

    forbidden = {"contacts", "images", "body_id", "geom_id", "actuator_forces_N"}
    assert forbidden.isdisjoint(set(_all_keys(observation)))


def test_replay_samples_match_actual_step_state_and_boundary():
    """中间帧的 ID、完整 qpos、物体 pose 和 TCP 必须来自同一个真实物理步。"""
    runtime, model, data = _runtime_fixture()
    observed = {}

    def observe_step(model, frozen, sample):
        observed[round(float(frozen.time), 9)] = (
            state_id(model, frozen), frozen.qpos.tolist(),
            frozen.body("bowl").xpos.tolist(), frozen.site("tcp").xpos.tolist(),
        )

    result = runtime.execute_joint_trajectory(
        group="right", joint_names=("joint_a", "joint_b"),
        times_s=(0.0, .01), positions_rad=((.15, -.25), (.2, -.2)),
        planned_state_id=state_id(model, data), step_observer=observe_step,
        replay_object_names=("bowl",), replay_interval_s=.004,
    )
    assert len(result["replay_samples"]) >= 3
    assert result["replay_samples"][0]["state_id"] == result["start_state_id"]
    assert result["replay_samples"][0]["sim_time_s"] == pytest.approx(0.0)
    for row in result["replay_samples"][1:]:
        expected_id, expected_qpos, bowl_position, tcp_position = observed[round(row["sim_time_s"], 9)]
        assert row["state_id"] == expected_id
        np.testing.assert_allclose(row["qpos"], expected_qpos)
        np.testing.assert_allclose(np.asarray(row["objects"]["bowl"]["T_world_object"])[:3, 3], bowl_position)
        np.testing.assert_allclose(np.asarray(row["robots"]["right"]["T_world_tcp"])[:3, 3], tcp_position)
        assert row["scene_revision"] == 7
    assert result["replay_samples"][-1]["state_id"] == result["end_state_id"]


def test_hold_replay_samples_follow_real_mujoco_steps():
    runtime, model, data = _runtime_fixture()
    result = runtime.execute_hold(
        duration_s=.01, planned_state_id=state_id(model, data),
        replay_object_names=("bowl",), replay_group="right", replay_interval_s=.004,
    )
    assert result["success"]
    assert len(result["replay_samples"]) >= 2
    assert result["replay_samples"][0]["state_id"] == result["start_state_id"]
    assert result["replay_samples"][-1]["state_id"] == result["end_state_id"]
    assert result["replay_samples"][-1]["sim_time_s"] == pytest.approx(data.time)
    assert result["replay_samples"][-1]["qpos"] == data.qpos.tolist()


@pytest.mark.parametrize(
    ("objects", "groups", "missing"),
    [(("missing_object",), (), "missing_object"), ((), ("missing_group",), "missing_group")],
)
def test_query_rejects_unknown_names_without_partial_state_change(objects, groups, missing):
    """若未知名称被静默忽略，Pipeline 会把缺失对象误认为有效空结果。"""
    runtime, model, data = _runtime_fixture()
    before = integration_state(model, data).copy()

    with pytest.raises(ValueError, match=missing):
        runtime.query(object_names=objects, robot_groups=groups)

    np.testing.assert_array_equal(integration_state(model, data), before)


def test_execute_named_trajectory_maps_reordered_joints_and_does_not_teleport():
    """若实现按数组顺序写错 actuator 或直接写终点 qpos，本测试会失败。"""
    runtime, model, data = _runtime_fixture()
    observation = runtime.query(robot_groups=("right",))
    initial_qpos = data.qpos.copy()

    result = runtime.execute_joint_trajectory(
        group="right",
        joint_names=("joint_b", "joint_a"),
        times_s=(0.0, 0.002),
        positions_rad=((-0.25, 0.15), (0.10, 0.60)),
        planned_state_id=observation["state_id"],
    )

    assert result["status"] == "completed"
    assert result["success"] is True
    assert result["start_state_id"] == observation["state_id"]
    assert result["end_state_id"] == state_id(model, data)
    assert result["joint_names"] == ["joint_a", "joint_b"]
    np.testing.assert_allclose(result["samples"][-1]["reference_positions_rad"], [0.60, 0.10])
    assert result["max_tracking_error_rad"] > 0.1
    assert data.time == pytest.approx(0.002)
    assert data.ctrl[model.actuator("joint_a_position").id] == pytest.approx(1.2)
    assert data.ctrl[model.actuator("joint_b_position").id] == pytest.approx(0.1)
    assert abs(data.qpos[model.joint("joint_a").qposadr[0]] - 0.60) > 0.1
    assert not np.array_equal(data.qpos, initial_qpos)


def test_execution_endpoint_gate_rejects_unreached_target():
    """轨迹指令发完不等于机械臂真正到位；下一阶段不能沿错误起点继续。"""
    runtime, _, _ = _runtime_fixture()
    planned = runtime.query(robot_groups=("right",))["state_id"]
    result = runtime.execute_joint_trajectory(
        group="right", joint_names=("joint_a", "joint_b"),
        times_s=(0.0, 0.002), positions_rad=((.15, -.25), (.60, .10)),
        planned_state_id=planned, endpoint_tolerance_rad=.02,
        settle_timeout_s=0.0,
    )
    assert result["success"] is False
    assert result["failure_type"] == "endpoint_tracking_error"
    assert result["endpoint_error_rad"] > .1
    assert result["settle_duration_s"] == 0


def test_execution_endpoint_gate_can_wait_for_settling():
    runtime, _, _ = _runtime_fixture()
    planned = runtime.query(robot_groups=("right",))["state_id"]
    result = runtime.execute_joint_trajectory(
        group="right", joint_names=("joint_a", "joint_b"),
        times_s=(0.0, 0.002), positions_rad=((.15, -.25), (.15, -.25)),
        planned_state_id=planned, endpoint_tolerance_rad=.01,
        settle_timeout_s=.1,
    )
    assert result["success"] is True
    assert result["endpoint_error_rad"] <= .01


def test_hold_uses_live_controls_and_actual_state_chain():
    runtime, model, data = _runtime_fixture()
    start = runtime.query(robot_groups=("right",))["state_id"]
    control = data.ctrl.copy()
    result = runtime.execute_hold(duration_s=.01, planned_state_id=start)
    assert result["success"] is True
    assert result["start_state_id"] == start
    assert result["end_state_id"] == state_id(model, data)
    assert result["sim_duration_s"] == pytest.approx(.01)
    np.testing.assert_array_equal(data.ctrl, control)
    with pytest.raises(ValueError, match="过期"):
        runtime.execute_hold(duration_s=.01, planned_state_id=start)


def test_execute_step_observer_receives_isolated_actual_states():
    runtime, model, data = _runtime_fixture()
    baseline_runtime, _, baseline_data = _runtime_fixture()
    planned = runtime.query(robot_groups=("right",))["state_id"]
    baseline_planned = baseline_runtime.query(robot_groups=("right",))["state_id"]
    observed = []

    def observer(observed_model, frozen, sample):
        assert observed_model is model
        observed.append((frozen.time, sample["actual_positions_rad"]))
        frozen.qpos[:] = 99  # 录像/诊断回调不能反向改变在线仿真。

    arguments = dict(
        group="right",
        joint_names=("joint_a", "joint_b"),
        times_s=(0.0, 0.006),
        positions_rad=((0.15, -0.25), (0.30, -0.10)),
    )
    result = runtime.execute_joint_trajectory(
        **arguments, planned_state_id=planned, step_observer=observer,
    )
    baseline_runtime.execute_joint_trajectory(
        **arguments, planned_state_id=baseline_planned,
    )

    assert len(observed) == len(result["samples"]) == 3
    assert [item[0] for item in observed] == [0.002, 0.004, 0.006]
    np.testing.assert_array_equal(data.qpos, baseline_data.qpos)


def test_execute_stops_at_first_structured_collision_condition():
    """真实碰撞回调在当前物理步后停止，不能继续执行剩余轨迹。"""
    runtime, _, data = _runtime_fixture()
    planned = runtime.query(robot_groups=("right",))["state_id"]
    observed_times = []

    def stop_at_second_step(_model, frozen, sample):
        observed_times.append(float(frozen.time))
        if sample["time_s"] >= .004:
            return {
                "failure_type": "forbidden_contact_during_execution",
                "contact_pair": ["arm_geom", "bowl_geom"],
            }
        return None

    result = runtime.execute_joint_trajectory(
        group="right", joint_names=("joint_a", "joint_b"),
        times_s=(0.0, .008),
        positions_rad=((.15, -.25), (.30, -.10)),
        planned_state_id=planned, stop_condition=stop_at_second_step,
        replay_object_names=("bowl",), replay_interval_s=.002,
    )

    assert result["success"] is False
    assert result["failure_type"] == "forbidden_contact_during_execution"
    assert result["failure_step_index"] == 1
    assert result["failure_evidence"]["contact_pair"] == ["arm_geom", "bowl_geom"]
    assert observed_times == pytest.approx([.002, .004])
    assert data.time == pytest.approx(.004)
    assert len(result["samples"]) == 2
    assert result["replay_samples"][-1]["sim_time_s"] == pytest.approx(.004)


@pytest.mark.parametrize(
    "overrides",
    [
        {"planned_state_id": "stale"},
        {"joint_names": ("joint_a", "joint_a")},
        {"joint_names": ("joint_a",)},
        {"times_s": (0.0, 0.0)},
        {"positions_rad": ((0.15, -0.25), (np.nan, 0.0))},
        {"timeout_s": 0.0},
    ],
)
def test_execute_rejects_invalid_trajectory_before_changing_state(overrides):
    """若输入验证发生在执行之后，ctrl 或积分状态会留下部分更新。"""
    runtime, model, data = _runtime_fixture()
    observation = runtime.query(robot_groups=("right",))
    arguments = {
        "group": "right",
        "joint_names": ("joint_a", "joint_b"),
        "times_s": (0.0, 0.002),
        "positions_rad": ((0.15, -0.25), (0.20, -0.20)),
        "planned_state_id": observation["state_id"],
        "timeout_s": None,
    }
    arguments.update(overrides)
    before = integration_state(model, data).copy()
    before_ctrl = data.ctrl.copy()

    with pytest.raises(ValueError):
        runtime.execute_joint_trajectory(**arguments)

    np.testing.assert_array_equal(integration_state(model, data), before)
    np.testing.assert_array_equal(data.ctrl, before_ctrl)


def test_execute_rejects_joint_without_position_actuator():
    """若 motor 被误当成位置目标，ctrl 的单位和含义都会错误。"""
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody><body><joint name="joint"/><geom size=".02"/><site name="tcp"/></body></worldbody>
          <actuator><motor name="joint_motor" joint="joint"/></actuator>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    runtime = MujocoRuntime(
        model,
        data,
        {"robot": {"joint_groups": {"arm": ["joint"]}, "tcp_sites": {"arm": "tcp"}}},
    )
    planned = runtime.query(robot_groups=("arm",))["state_id"]

    with pytest.raises(ValueError, match="position actuator"):
        runtime.execute_joint_trajectory(
            group="arm",
            joint_names=("joint",),
            times_s=(0.0, 0.002),
            positions_rad=((0.0,), (0.1,)),
            planned_state_id=planned,
        )


def test_execute_stops_with_structured_timeout_at_first_budget_boundary():
    """若 timeout 只校验参数却不终止执行，轨迹会越过调用方预算。"""
    runtime, _, _ = _runtime_fixture()
    planned = runtime.query(robot_groups=("right",))["state_id"]

    result = runtime.execute_joint_trajectory(
        group="right",
        joint_names=("joint_a", "joint_b"),
        times_s=(0.0, 0.004),
        positions_rad=((0.15, -0.25), (0.30, -0.10)),
        planned_state_id=planned,
        timeout_s=0.001,
    )

    assert result["status"] == "failed"
    assert result["failure_type"] == "execution_timeout"
    assert result["sim_duration_s"] == pytest.approx(0.0)


def test_execute_does_not_report_final_step_past_timeout_as_completed():
    """若 final-step 特判绕过预算，单步轨迹会在超时后仍报告成功。"""
    runtime, _, data = _runtime_fixture()
    planned = runtime.query(robot_groups=("right",))["state_id"]

    result = runtime.execute_joint_trajectory(
        group="right",
        joint_names=("joint_a", "joint_b"),
        times_s=(0.0, 0.002),
        positions_rad=((0.15, -0.25), (0.30, -0.10)),
        planned_state_id=planned,
        timeout_s=0.001,
    )

    assert result["status"] == "failed"
    assert result["failure_type"] == "execution_timeout"
    assert data.time == pytest.approx(0.0)


def test_execute_stops_when_mujoco_state_becomes_non_finite(monkeypatch):
    """若每步不检查实际状态，NaN 会继续污染后续控制与反馈。"""
    runtime, _, data = _runtime_fixture()
    planned = runtime.query(robot_groups=("right",))["state_id"]
    real_step = mujoco.mj_step

    def corrupt_after_step(model, current):
        real_step(model, current)
        current.qpos[0] = np.nan

    monkeypatch.setattr(mujoco, "mj_step", corrupt_after_step)
    result = runtime.execute_joint_trajectory(
        group="right",
        joint_names=("joint_a", "joint_b"),
        times_s=(0.0, 0.004),
        positions_rad=((0.15, -0.25), (0.30, -0.10)),
        planned_state_id=planned,
    )

    assert np.isnan(data.qpos[0])
    assert result["status"] == "failed"
    assert result["failure_type"] == "non_finite_state"
    assert len(result["samples"]) == 1


def test_execute_rejects_non_finite_initial_state_before_mujoco_can_reset_it():
    """MuJoCo 可能在 step 中自动 reset NaN；runtime 必须保留并报告原始失败。"""
    runtime, model, data = _runtime_fixture()
    data.qpos[0] = np.nan
    planned = state_id(model, data)

    result = runtime.execute_joint_trajectory(
        group="right",
        joint_names=("joint_a", "joint_b"),
        times_s=(0.0, 0.002),
        positions_rad=((0.15, -0.25), (0.30, -0.10)),
        planned_state_id=planned,
    )

    assert result["status"] == "failed"
    assert result["failure_type"] == "non_finite_state"
    assert result["samples"] == []
    assert data.time == pytest.approx(0.0)
    assert np.isnan(data.qpos[0])


def test_execute_stops_when_mujoco_reports_instability(monkeypatch):
    """若 step 自动恢复成有限状态，warning 计数仍必须阻止 completed。"""
    runtime, _, _ = _runtime_fixture()
    planned = runtime.query(robot_groups=("right",))["state_id"]
    real_step = mujoco.mj_step

    def warn_after_step(model, current):
        real_step(model, current)
        current.warning[mujoco.mjtWarning.mjWARN_BADQPOS].number += 1

    monkeypatch.setattr(mujoco, "mj_step", warn_after_step)
    result = runtime.execute_joint_trajectory(
        group="right",
        joint_names=("joint_a", "joint_b"),
        times_s=(0.0, 0.002),
        positions_rad=((0.15, -0.25), (0.30, -0.10)),
        planned_state_id=planned,
    )

    assert result["status"] == "failed"
    assert result["failure_type"] == "mujoco_instability"


@pytest.mark.parametrize("joint_type", ["slide", "ball"])
def test_runtime_rejects_non_hinge_arm_joints_for_radian_contract(joint_type):
    """若 slide/ball 进入 rad 标注的接口，数值的单位或维度会静默错误。"""
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco><worldbody><body><joint name="joint" type="{joint_type}"/>
          <geom size=".02"/><site name="tcp"/></body></worldbody>
          <actuator><position name="joint_position" joint="joint" kp="20"/></actuator>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    runtime = MujocoRuntime(
        model,
        data,
        {"robot": {"joint_groups": {"arm": ["joint"]}, "tcp_sites": {"arm": "tcp"}}},
    )

    with pytest.raises(ValueError, match="hinge"):
        runtime.query(robot_groups=("arm",))
    planned = runtime.query()["state_id"]
    with pytest.raises(ValueError, match="hinge"):
        runtime.execute_joint_trajectory(
            group="arm",
            joint_names=("joint",),
            times_s=(0.0, 0.002),
            positions_rad=((0.0,), (0.1,)),
            planned_state_id=planned,
        )
