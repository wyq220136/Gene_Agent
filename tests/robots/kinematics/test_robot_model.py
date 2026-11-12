"""独立解析原 URDF 验证双臂 FK、TCP、资产和关节映射。"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from gene_agent.robots import build_robot

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/robots/realman"


def _axis_rotation(axis, angle):
    """Rodrigues 独立实现，不复用构建器的 scipy 旋转或 MuJoCo FK。"""
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew@skew)


def _urdf_fk(side, values):
    tree = ET.parse(ASSETS / "urdf" / f"realman_{side}.urdf").getroot()
    transforms = {"base_link": np.eye(4)}
    pending = list(tree.findall("joint"))
    while pending:
        progress = False
        for joint in pending[:]:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in transforms:
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            r, p, y = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            local = np.eye(4)
            local[:3, :3] = _axis_rotation([0, 0, 1], y) @ _axis_rotation([0, 1, 0], p) @ _axis_rotation([1, 0, 0], r)
            local[:3, 3] = xyz
            motion = np.eye(4)
            if joint.get("type") != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                motion[:3, :3] = _axis_rotation(axis, values[f"{side}_{joint.get('name')}"])
            transforms[child] = transforms[parent] @ local @ motion
            pending.remove(joint)
            progress = True
        assert progress, "URDF 链断开或存在环"
    return transforms


@pytest.fixture(scope="module")
def robot(tmp_path_factory):
    info = build_robot(tmp_path_factory.mktemp("realman"), gripper_mode="fixed_urdf")
    return info, mujoco.MjModel.from_xml_path(info["xml_path"])


def test_frozen_sources():
    """迁移资产必须与已记录源哈希一致，防止无说明更换坐标或几何。"""
    payload = json.loads((ASSETS / "provenance.json").read_text())
    for filename, item in payload["files"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == item["sha256"]


@pytest.mark.parametrize("case", range(8))
def test_all_link_and_tcp_fk(robot, case):
    info, model = robot
    data = mujoco.MjData(model)
    values = info["home_joint_positions"].copy()
    if case == 0:
        values = dict.fromkeys(values, 0.0)
    elif case > 1:
        rng = np.random.default_rng(1700 + case)
        for name in values:
            joint = model.joint(name)
            low, high = model.jnt_range[joint.id]
            values[name] = rng.uniform(low * .8, high * .8)
    for name, q in values.items():
        data.qpos[model.joint(name).qposadr[0]] = q
    mujoco.mj_forward(model, data)
    installation = json.loads((ASSETS / "installation.json").read_text())["mounted"]
    world_body = np.linalg.inv(installation["body_from_sdk"]["right"])
    tcp = np.diag([-1., -1., 1., 1.])
    tcp[2, 3] = .15
    for side in info["joint_groups"]:
        world_base = world_body @ installation["body_from_sdk"][side] @ np.linalg.inv(installation["sdk_to_curobo"][side])
        transforms = _urdf_fk(side, values)
        for link, base_link in transforms.items():
            expected = world_base @ base_link
            actual = data.body(f"{side}_{link}")
            np.testing.assert_allclose(actual.xpos, expected[:3, 3], atol=2e-9)
            np.testing.assert_allclose(actual.xmat.reshape(3, 3), expected[:3, :3], atol=2e-9)
        expected_tcp = world_base @ transforms["link_7"] @ np.linalg.inv(installation["sdk_flange_to_curobo_frame"][side]) @ tcp
        site = data.site(info["tcp_sites"][side])
        np.testing.assert_allclose(site.xpos, expected_tcp[:3, 3], atol=2e-9)
        np.testing.assert_allclose(site.xmat.reshape(3, 3), expected_tcp[:3, :3], atol=2e-9)


def test_include_and_named_joint_mapping(robot, tmp_path):
    info, model = robot
    combined = mujoco.MjModel.from_xml_string(f'<mujoco><include file="{info["xml_path"]}"/><option timestep="0.002"/></mujoco>')
    assert set(info["joint_groups"]) == {"left", "right"}
    assert len(info["joint_names"]) == model.nq == combined.nq
    assert model.nu == len(info["joint_names"])
    assert info["gripper"]["mode"] == "fixed_urdf"
    for name in info["joint_names"]:
        assert model.joint(name).id >= 0
    assert info["world_frame"] == "right_sdk_base"


def test_recorded_sdk_tcp_baseline(robot):
    """保存的实机默认姿态提供外部锚点，避免两个算法共用错误安装链。"""
    info, model = robot
    data = mujoco.MjData(model)
    for name, q in info["home_joint_positions"].items():
        data.qpos[model.joint(name).qposadr[0]] = q
    mujoco.mj_forward(model, data)
    installation = json.loads((ASSETS / "installation.json").read_text())["mounted"]
    for side, site_name in info["tcp_sites"].items():
        recorded = json.loads((ROOT / "config/robots" / f"realman_{side}_default_pose.json").read_text())
        expected = np.asarray(info["world_from_body"]) @ installation["body_from_sdk"][side] @ recorded["tcp_pose_sdk"]
        site = data.site(site_name)
        assert np.linalg.norm(site.xpos - expected[:3, 3]) < 1e-4
        # 记录来自不同时间的读取且旋转矩阵为单精度，因此采用 1 mrad 基线。
        delta = site.xmat.reshape(3, 3).T @ expected[:3, :3]
        assert np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)) < 1e-3
