"""公共夹爪终态时序与复用回归；不连接或移动真机。"""
from types import SimpleNamespace

import pytest

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
        assert block is True
        self.calls.append(name)
        if self.fail:
            return 17
        self.width = 500
        # 即使开口已经连续相同，mode=4/5 也不能宣告完成。
        self.modes = [4, 5, 4, 3, 3, 3]
        return 0

    def rm_set_gripper_position(self, position, block, timeout):
        assert 0 <= position <= 1000
        return self.command("position", block)

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
    robot = SimpleNamespace(group="right", read_joint_state=lambda: start)
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
    assert result["success"] and result["measured_end_width_m"] == .04
    assert sdk.calls == [mode]
    assert len(result["endpoint_confirmation"]["readings"]) == 6
    reads = sdk.reads
    assert runtime()._read_start_width(None) == .04
    # 每一阶段创建新的 Runtime；夹爪连接和已确认终态跨阶段保留。
    other = SimpleNamespace(confirmed_endpoint_width_m=.08)
    sampler = RobotStateSampler({"left": robot, "right": robot},
                                {"left": other, "right": gripper})
    try:
        for _ in range(3):
            with sampler.sample() as pending:
                samples = pending.result()
            assert samples["right"]["width_m"] == .04
            assert samples["right"]["width_source"] == "confirmed_execution_endpoint"
    finally:
        sampler.close()
    assert sdk.reads == reads
    # 下一条命令失败也必须撤销旧终态，不能继续使用前一动作的值。
    sdk.fail = True
    failed = runtime().run_gripper_stage("next", **{**kw, "planned_start_width_m": .04})
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


def test_motion_that_never_finishes_cannot_publish_endpoint(monkeypatch):
    sdk = SDK()
    gripper, _, runtime, kw = rig(sdk)
    sdk.rm_get_gripper_state = lambda: (0, {"actpos": 500, "mode": 4})
    ticks = iter(i * .1 for i in range(1000))
    monkeypatch.setattr("gene_agent.realworld.runtime.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("gene_agent.realworld.runtime.time.sleep", lambda _: None)
    result = runtime().run_gripper_stage("open", **{**kw, "planned_start_width_m": .04})
    assert result["failure"]["code"] == "gripper_execution_failed"
    assert not result["endpoint_confirmation"]["confirmed"]
    assert gripper.confirmed_endpoint_width_m is None
    assert sdk.calls == ["position"]
