"""公共夹爪终态时序与复用回归；不连接或移动真机。"""
from types import SimpleNamespace

import pytest
import numpy as np

from gene_agent.motion.model import JointState
from gene_agent.realworld.devices.grippers.realman.adapter import RealmanGripperAdapter
from gene_agent.realworld.runtime import RealWorldRuntime
from gene_agent.realworld.state_sampling import RobotStateSampler


class SDK:
    def __init__(self):
        self.calls = []
        self.reads = 0
        self.modes = []
        self.width = 800
        self.fail = False

    def command(self, name, block):
        self.calls.append(name)
        if self.fail:
            return 17
        self.width = 500
        # 阻塞返回后还可能运动，必须等到 mode=3 才交接。
        self.modes = [4, 5, 4, 3, 3, 3]
        return 0

    def rm_set_gripper_position(self, position, block, timeout):
        assert 0 <= position <= 1000
        code = self.command("position", block)
        if code == 0:
            self.width = position
        return code

    def rm_set_gripper_pick_on(self, speed, force, block, timeout):
        return self.command("grasp", block)

    def rm_set_gripper_release(self, speed, block, timeout):
        return self.command("release", block)

    def rm_get_gripper_state(self):
        self.reads += 1
        return 0, {"actpos": self.width, "status": 1, "error": 0,
                   "mode": self.modes.pop(0) if self.modes else 3}


def rig(sdk):
    gripper = RealmanGripperAdapter(sdk, max_width_m=.08)
    start = JointState(tuple(f"right_joint_{i}" for i in range(1, 8)), (0.,)*7)
    robot = SimpleNamespace(group="right", read_joint_state=lambda: start,
                            tcp_pose_from_sdk_fk=lambda q: np.eye(4))
    def runtime():
        return RealWorldRuntime(robot, gripper=gripper, scene_snapshot_sha256="scene",
                                max_start_error_rad=.02, tcp_height_fn=lambda _: .2,
                                min_tcp_z_m=.08)
    kw = dict(group="right", target_width_m=.04, planned_start=start,
              planned_start_width_m=.064, max_width_error_m=.005,
              planned_scene_snapshot_sha256="scene", execute=True)
    return gripper, robot, runtime, kw


@pytest.mark.parametrize("mode", ["position", "grasp", "release"])
def test_all_commands_confirm_motion_then_reuse_across_runtimes_and_observations(monkeypatch, mode):
    monkeypatch.setattr("gene_agent.realworld.runtime.time.sleep", lambda _: None)
    sdk = SDK()
    gripper, robot, runtime, kw = rig(sdk)
    result = runtime().run_gripper_stage("stage", command_mode=mode, **kw)
    actual = .0792 if mode == "release" else .04
    assert result["success"] and result["measured_end_width_m"] == pytest.approx(actual)
    assert sdk.calls == ["position" if mode == "release" else mode]
    if mode == "grasp":
        assert result["hardware_command"]["blocking"] is False
        assert result["hardware_command"]["completion_source"] == "sdk_motion_feedback"
    assert result["endpoint_confirmation"]["poll_count"] == 4
    reads = sdk.reads
    assert runtime()._read_start_width(None) == pytest.approx(actual)
    # 每一阶段创建新的 Runtime；夹爪连接和已确认终态跨阶段保留。
    other = SimpleNamespace(confirmed_endpoint_width_m=.08)
    sampler = RobotStateSampler({"left": robot, "right": robot},
                                {"left": other, "right": gripper})
    try:
        for _ in range(3):
            with sampler.sample() as pending:
                samples = pending.result()
            assert samples["right"]["width_m"] == pytest.approx(actual)
            assert samples["right"]["width_source"] == "confirmed_execution_endpoint"
    finally:
        sampler.close()
    assert sdk.reads == reads
    # 下一条命令失败也必须撤销旧终态，不能继续使用前一动作的值。
    sdk.fail = True
    failed = runtime().run_gripper_stage("next", **{**kw, "planned_start_width_m": actual})
    assert not failed["success"] and gripper.confirmed_endpoint_width_m is None


@pytest.mark.parametrize("mode", ["position", "grasp", "release"])
def test_direct_adapter_commands_invalidate_cached_endpoint(mode):
    sdk = SDK()
    gripper, _, _, _ = rig(sdk)
    gripper.confirmed_endpoint_width_m = .04
    gripper.initial_observed_width_m = .064
    if mode == "position":
        gripper.command_width_m(.04)
    else:
        getattr(gripper, mode)()
    assert gripper.confirmed_endpoint_width_m is None
    assert gripper.initial_observed_width_m is None


@pytest.mark.parametrize("target,intermediate,finished,moving_mode", [
    (.0792, 854, 991, 5),  # 原 all/handle_pull/00_open 失败轨迹
    (.041, 703, 508, 4),   # 原 pick/open 失败轨迹
])
def test_replay_intermediate_feedback_waits_then_publishes_final_only(
        monkeypatch, target, intermediate, finished, moving_mode):
    sdk = SDK()
    gripper, _, runtime, kw = rig(sdk)
    rows = iter([(800, 3), (intermediate, moving_mode), (finished, 3)])
    def feedback():
        position, mode = next(rows)
        assert gripper.confirmed_endpoint_width_m is None
        return 0, {"actpos": position, "mode": mode, "error": 0}
    sdk.rm_get_gripper_state = feedback
    waits = []
    monkeypatch.setattr("gene_agent.realworld.runtime.time.sleep", waits.append)
    result = runtime().run_gripper_stage("open", **{**kw, "target_width_m": target})
    assert result["success"] and sdk.calls == ["position"]
    assert result["measured_end_width_m"] == pytest.approx(finished / 1000 * .08)
    assert gripper.confirmed_endpoint_width_m == result["measured_end_width_m"]
    assert waits == [.05]
    confirmation = result["endpoint_confirmation"]
    assert confirmation["poll_count"] == 2 and confirmation["motion_complete"] is True
    assert confirmation["readings"][0]["motion_complete"] is False


def test_motion_not_finished_never_compares_target_or_caches_even_if_width_matches(monkeypatch):
    sdk = SDK()
    gripper, _, runtime, kw = rig(sdk)
    sdk.rm_get_gripper_state = lambda: (0, {"actpos": 500, "mode": 5})
    ticks = iter(i * .05 for i in range(1000))
    monkeypatch.setattr("gene_agent.realworld.runtime.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("gene_agent.realworld.runtime.time.sleep", lambda _: None)
    result = runtime().run_gripper_stage("open", **{**kw, "planned_start_width_m": .04,
                                                  "endpoint_stability_timeout_s": .15})
    assert not result["success"]
    assert "超时" in result["failure"]["message"]
    assert result["endpoint_confirmation"]["motion_complete"] is False
    assert gripper.confirmed_endpoint_width_m is None
    assert sdk.calls == ["position"]


def test_stopped_at_wrong_width_still_fails_with_original_tolerance(monkeypatch):
    sdk = SDK()
    gripper, _, runtime, kw = rig(sdk)
    rows = iter([(800, 3), (854, 5), (880, 3)])
    sdk.rm_get_gripper_state = lambda: (0, dict(zip(("actpos", "mode"), next(rows))))
    monkeypatch.setattr("gene_agent.realworld.runtime.time.sleep", lambda _: None)
    result = runtime().run_gripper_stage("open", **{**kw, "target_width_m": .0792})
    assert not result["success"] and "运动完成后" in result["failure"]["message"]
    assert result["endpoint_confirmation"]["motion_complete"] is True
    assert result["hardware_command"]["target_width_m"] == pytest.approx(.0792)
    assert gripper.confirmed_endpoint_width_m is None
    assert sdk.calls == ["position"]


def test_force_grasp_waits_for_feedback_not_blocking_arrival_event(monkeypatch):
    sdk = SDK()
    gripper, _, runtime, kw = rig(sdk)
    issued = []
    def grasp(speed, force, block, timeout):
        issued.append((speed, force, block, timeout))
        if block:
            return -4  # 原运行的SDK到位事件等待超时
        sdk.width = 400
        sdk.modes = [4, 6]
        return 0
    sdk.rm_set_gripper_pick_on = grasp
    monkeypatch.setattr("gene_agent.realworld.runtime.time.sleep", lambda _: None)
    result = runtime().run_gripper_stage("close", command_mode="grasp", **kw)
    assert result["success"]
    assert issued == [(1000, 500, False, 3)]
    assert result["endpoint_confirmation"]["poll_count"] == 2
    assert result["measured_end_width_m"] == pytest.approx(.032)
