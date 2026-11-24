"""真机闭爪反馈的执行时序回归；只使用假设备，不连接硬件。"""
from types import SimpleNamespace

import pytest

from gene_agent.motion.model import JointState
from gene_agent.realworld.checks import execution_handoff_error
from gene_agent.realworld.runtime import RealWorldRuntime

NAMES = tuple(f"right_joint_{i}" for i in range(1, 8))
START = JointState(NAMES, (0.,)*7)


def _scene(frame, width, q=None):
    return {"snapshot": {"active_group": "right"},
            "robot_metadata": {"joint_groups": {"right": NAMES}},
            "provenance": {"source_frame_number": frame, "source_timestamp_ms": frame*1000.,
                "joint_positions_by_group_rad": {"right": list(q or START.positions_rad)},
                "gripper_widths_m": {"right": width}}}


def _rig(monkeypatch, *, fault=False, invalid_end=False):
    events=[]
    class Gripper:
        max_width_m=.08
        closing=False
        settled=False
        def read_width_m(self):
            if not self.closing: return .041
            events.append("read_end")
            return float("nan") if invalid_end else .024 if self.settled else .040
        def grasp(self, **kwargs):
            events.append("grasp")
            if fault: raise RuntimeError("SDK 17")
            self.closing=True
        def command_width_m(self, width):
            events.append("position")
    gripper=Gripper()
    robot=SimpleNamespace(group="right",read_joint_state=lambda: START)
    def sleep(seconds):
        events.append(("settle",seconds)); gripper.settled=True
    monkeypatch.setattr("gene_agent.realworld.runtime.time.sleep",sleep)
    runtime=RealWorldRuntime(robot,scene_snapshot_sha256="scene",max_start_error_rad=.02,
        tcp_height_fn=lambda q:.2,min_tcp_z_m=.0,gripper=gripper)
    options=dict(group="right",target_width_m=.007,planned_start=START,
                 planned_start_width_m=.041,max_width_error_m=.005,
                 planned_scene_snapshot_sha256="scene",command_mode="grasp")
    return runtime,options,events


def test_force_grasp_waits_before_freezing_end_width_and_handoff(monkeypatch):
    runtime,kw,events=_rig(monkeypatch)
    execution=runtime.run_gripper_stage("close",execute=True,**kw)
    assert execution["status"]=="executed"
    assert execution["measured_end_width_m"]==pytest.approx(.024)
    assert events==["grasp",("settle",1.5),"read_end"]
    assert execution_handoff_error(_scene(1,.041),_scene(2,.024),execution,
        max_joint_error_rad=.02,max_width_error_m=.005) is None
    # 力控夹持由物体决定宽度，不能拿规划的7mm闭合宽度当作到位条件。
    assert abs(execution["measured_end_width_m"]-execution["target_width_m"])>.005


def test_width_change_after_settling_still_blocks_handoff(monkeypatch):
    runtime,kw,_=_rig(monkeypatch)
    execution=runtime.run_gripper_stage("close",execute=True,**kw)
    error=execution_handoff_error(_scene(1,.041),_scene(2,.044),execution,
        max_joint_error_rad=.02,max_width_error_m=.005)
    assert error and "开口" in error


def test_settle_does_not_hide_arm_drift(monkeypatch):
    runtime,kw,_=_rig(monkeypatch)
    execution=runtime.run_gripper_stage("close",execute=True,**kw)
    error=execution_handoff_error(_scene(1,.041),_scene(2,.024,q=(.05,)*7),execution,
        max_joint_error_rad=.02,max_width_error_m=.005)
    assert error and "关节" in error


def test_preview_never_closes_or_waits(monkeypatch):
    runtime,kw,events=_rig(monkeypatch)
    execution=runtime.run_gripper_stage("close",execute=False,**kw)
    assert execution["status"]=="validated"
    assert execution["measured_end_width_m"] is None and events==[]


def test_sdk_failure_does_not_wait_or_retry_command(monkeypatch):
    runtime,kw,events=_rig(monkeypatch,fault=True)
    execution=runtime.run_gripper_stage("close",execute=True,**kw)
    assert execution["failure"]["code"]=="gripper_execution_failed"
    assert events==["grasp"]


def test_invalid_end_width_cannot_be_reported_as_executed(monkeypatch):
    runtime,kw,_=_rig(monkeypatch,invalid_end=True)
    result=runtime.run_gripper_stage("close",execute=True,**kw)
    assert result["success"] is False
    assert result["failure"]["code"]=="gripper_execution_failed"


@pytest.mark.parametrize("duration",[-1.,float("nan"),float("inf")])
def test_invalid_settle_duration_never_sends_command(monkeypatch,duration):
    runtime,kw,events=_rig(monkeypatch)
    with pytest.raises(ValueError,match="等待"):
        runtime.run_gripper_stage("close",execute=True,grasp_settle_time_s=duration,**kw)
    assert events==[]


def test_motion_handoff_rejects_invalid_gripper_feedback():
    execution={"source_format":"realworld_motion_stage","measured_end":START.to_dict()}
    error=execution_handoff_error(_scene(1,.024),_scene(2,float("nan")),execution,
        max_joint_error_rad=.02,max_width_error_m=.005)
    assert error and "夹爪" in error


@pytest.mark.parametrize("width,allowed,expected",[(.024,True,None),(.024,False,"开口"),
                                                  (float("nan"),True,"读数")])
def test_knob_wait_checks_match_the_active_physical_phase(width,allowed,expected):
    from gene_agent.realworld.microwave.knob import _wait_handoff_error
    error=_wait_handoff_error(_scene(1,.041),_scene(2,width),
        max_joint_error_rad=.02,max_width_error_m=.005,allow_gripper_motion=allowed)
    assert error is None if expected is None else expected in error


def test_knob_grasp_settle_still_rejects_arm_motion():
    from gene_agent.realworld.microwave.knob import _wait_handoff_error
    error=_wait_handoff_error(_scene(1,.041),_scene(2,.024,q=(.05,)*7),
        max_joint_error_rad=.02,max_width_error_m=.005,allow_gripper_motion=True)
    assert error and "关节" in error
