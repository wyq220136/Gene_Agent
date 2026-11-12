import json

import mujoco
import numpy as np
import pytest

from gene_agent.feedback import failure as failure_module
from gene_agent.feedback.failure import capture_failure
from gene_agent.feedback.state import integration_state, state_id
from gene_agent.simulation import MujocoRuntime
from gene_agent.simulation.diagnostics import query_diagnostics


def _contact_runtime():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco model="failure_feedback">
          <visual><global offwidth="320" offheight="240"/></visual>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 .1"/>
            <body name="arm"><joint name="joint" damping="1"/><geom type="sphere" size=".02"/>
              <site name="tcp" pos="0 0 .1"/>
            </body>
            <body name="bowl" pos=".25 0 .04"><freejoint name="bowl_free"/>
              <geom name="bowl_geom" type="sphere" size=".05" rgba=".8 .2 .1 1"/>
            </body>
          </worldbody>
          <actuator><position name="joint_position" joint="joint" kp="20"/></actuator>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    metadata = {
        "world_frame": "world",
        "robot": {
            "world_frame": "world",
            "joint_groups": {"right": ["joint"]},
            "tcp_sites": {"right": "tcp"},
            "grippers": {},
        },
    }
    return MujocoRuntime(model, data, metadata), model, data


def test_diagnostics_are_privileged_and_share_operational_state_id():
    """若 contacts 泄漏进 operational observation，Pipeline 会依赖仿真特权信息。"""
    runtime, model, data = _contact_runtime()
    before = integration_state(model, data).copy()

    operational = runtime.query(object_names=("bowl",), robot_groups=("right",))
    diagnostics = query_diagnostics(runtime, body_names=("bowl",))

    assert "contacts" not in operational
    assert diagnostics["state_id"] == operational["state_id"]
    assert diagnostics["contacts"]
    contact = diagnostics["contacts"][0]
    assert "bowl" in contact["body_names"]
    assert {"geom_ids", "geom_names", "dist_m", "force_contact_frame"} <= set(contact)
    assert set(diagnostics["bodies"]) == {"bowl"}
    np.testing.assert_array_equal(integration_state(model, data), before)


def test_capture_failure_saves_consistent_operational_numeric_and_visual_evidence(tmp_path):
    """若各证据从不同状态读取，Codex 会把时序差异误诊为物理错误。"""
    runtime, model, data = _contact_runtime()
    before = integration_state(model, data).copy()
    output = tmp_path / "failure"

    result = capture_failure(
        runtime,
        output,
        stage_id="close_gripper",
        failure_type="postcondition_failed",
        message="bowl was not retained",
        object_names=("bowl",),
        robot_groups=("right",),
        target_body="bowl",
        details={"held": False},
    )

    expected = {
        "failure.json",
        "operational_observation.json",
        "diagnostics.json",
        "state.npz",
        "images",
    }
    assert expected <= {path.name for path in output.iterdir()}
    failure = json.loads((output / "failure.json").read_text())
    operational = json.loads((output / "operational_observation.json").read_text())
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    views = json.loads((output / "images/views.json").read_text())
    identifiers = {failure["state_id"], operational["state_id"], diagnostics["state_id"], views["state_id"]}
    assert identifiers == {state_id(model, data)}
    assert failure["evidence_incomplete"] is False
    assert failure["details"] == {"held": False}
    assert result["state_id"] == failure["state_id"]
    assert {"overview", "task", "side", "contact", "contact_side"} <= set(views["views"])
    assert all((output / "images" / view["path"]).is_file() for view in views["views"].values())
    assert "contacts" not in operational
    np.testing.assert_array_equal(integration_state(model, data), before)


def test_capture_failure_keeps_numeric_evidence_when_rendering_fails(tmp_path, monkeypatch):
    """若 renderer 异常中断整个捕获，最关键的可恢复状态也会丢失。"""
    runtime, _, _ = _contact_runtime()

    def fail_render(*args, **kwargs):
        raise RuntimeError("synthetic render failure")

    monkeypatch.setattr(failure_module, "render_views", fail_render)
    try:
        raise LookupError("synthetic pipeline error")
    except LookupError as exception:
        result = capture_failure(
            runtime,
            tmp_path / "failure",
            stage_id="plan",
            failure_type="python_exception",
            message=str(exception),
            object_names=("bowl",),
            exception=exception,
        )

    output = tmp_path / "failure"
    assert result["evidence_incomplete"] is True
    assert "render_views" in result["evidence_errors"]
    for name in ("failure.json", "operational_observation.json", "diagnostics.json", "state.npz", "traceback.txt"):
        assert (output / name).is_file()
    traceback_text = (output / "traceback.txt").read_text()
    assert "LookupError" in traceback_text
    assert "synthetic pipeline error" in traceback_text


def test_capture_failure_refuses_to_overwrite_existing_evidence(tmp_path):
    """若复用目录会覆盖旧 run，Codex 将失去修改前后的对照。"""
    runtime, _, _ = _contact_runtime()
    output = tmp_path / "failure"
    output.mkdir()

    with pytest.raises(FileExistsError):
        capture_failure(
            runtime,
            output,
            stage_id="plan",
            failure_type="planner_failed",
            message="no path",
        )
