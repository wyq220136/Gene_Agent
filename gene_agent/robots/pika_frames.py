"""Pika 离线 frame 核对；不连接设备，也不选择新的生产 TCP。

T_A_B 将 B 中的坐标映射到 A；所有长度单位为米。
参考控制 frame、URDF 原点、仿真 TCP、指垫中心和指尖几何分别保留。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.knowledge.geometry import validate_pose


def load_frame_profile(path: str | Path) -> dict:
    """加载命名 frame，派生操作 TCP；参考控制点不参与操作点的隐式选择。"""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") not in {
        "nominal_audit_only", "site_verified"
    } or not str(payload.get("source", "")).strip():
        raise ValueError("Pika frame profile 缺少版本、状态或来源")
    names = ("T_link7_sdk_flange", "T_link7_pika_model", "T_pika_model_operation_tcp",
             "T_sdk_flange_reference_control")
    result = {name: validate_pose(payload[name], field_name=name) for name in names}
    result["T_link7_operation_tcp"] = result["T_link7_pika_model"] @ result["T_pika_model_operation_tcp"]
    result["T_sdk_flange_operation_tcp"] = (np.linalg.inv(result["T_link7_sdk_flange"])
                                             @ result["T_link7_operation_tcp"])
    return {**result, "status": payload["status"], "source": payload["source"],
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_reference_transform(source: Path) -> dict:
    """只解析两个顶层字面量，绝不 import/exec 外部真机控制代码。

仅支持当前参考代码的 np.array(字面量) 写法；源代码改变后应显式复核，
不能静默猜测新的值。返回值复现其 T_RM_PIKA 常量，不声称端点是最远指尖。
"""
    source = Path(source).resolve()
    raw = source.read_bytes()
    tree = ast.parse(raw, filename=str(source))
    constants = {}
    lines = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id not in {"RM_TO_PIKA_ROT", "TCP_OFFSET"}:
                continue
            name = target.id
            if name in constants:
                raise ValueError(f"{name} 有多次赋值，需人工核对生效分支")
            value = node.value
            if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name) and value.func.value.id == "np"
                    and value.func.attr == "array" and len(value.args) == 1):
                raise ValueError(f"{name} 必须是 np.array(数值字面量)")
            try:
                constants[name] = np.asarray(ast.literal_eval(value.args[0]), dtype=float)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{name} 不是数值字面量") from exc
            lines[name] = node.lineno
    if set(constants) != {"RM_TO_PIKA_ROT", "TCP_OFFSET"}:
        raise ValueError("参考代码缺少 RM_TO_PIKA_ROT 或 TCP_OFFSET")
    if constants["RM_TO_PIKA_ROT"].shape != (3, 3) or constants["TCP_OFFSET"].shape != (3,):
        raise ValueError("参考旋转必须为 3×3，平移必须为 3 维")
    transform = np.eye(4)
    transform[:3, :3] = constants["RM_TO_PIKA_ROT"].T
    transform[:3, 3] = constants["TCP_OFFSET"]
    validate_pose(transform, field_name="T_sdk_flange_reference_control")
    return {"source": str(source), "sha256": hashlib.sha256(raw).hexdigest(),
            "constant_lines": lines,
            "T_sdk_flange_reference_control": transform.tolist(),
            "interpretation": "Literal control transform; physical point identity not calibrated"}


def pose_from_mujoco(item) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(item.xmat).reshape(3, 3)
    pose[:3, 3] = item.xpos
    return pose


def pose_error(expected, actual) -> dict:
    delta = np.linalg.inv(validate_pose(expected)) @ validate_pose(actual)
    return {"translation_m": float(np.linalg.norm(delta[:3, 3])),
            "rotation_deg": float(np.rad2deg(np.arccos(np.clip(
                (np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))}


def model_frames(model, data, robot: dict, joints_rad, width_m: float) -> dict:
    """仅执行 FK。开口用现有双 prismatic 模型的解析值，不做物理夹持。

指尖点为两指网格沿模型 +X 的最大延伸处中心，非接触面标定。
指垫中心复用生产模型的两个 inner site；不要将其等同于物理抓取中心。
"""
    group, = robot["joint_groups"]
    joints = np.asarray(joints_rad, dtype=float)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError("需要七个有限关节角（rad）")
    grip = robot["grippers"][group]
    maximum = grip["width_range_m"][1]
    if not np.isfinite(width_m) or not 0 <= width_m <= maximum:
        raise ValueError(f"开口必须在 [0, {maximum}] m")
    for name, value in zip(robot["joint_groups"][group], joints):
        if not model.joint(name).range[0] <= value <= model.joint(name).range[1]:
            raise ValueError(f"{name} 超出关节限位")
        data.qpos[model.joint(name).qposadr[0]] = value
    closure = (maximum - width_m) / 2
    for name, value in zip(grip["joint_names"], (-closure, closure)):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    base = pose_from_mujoco(data.body("pika_gripper_base_link"))
    pads = [data.site(grip[key]).xpos.copy()
            for key in ("positive_inner_site_name", "negative_inner_site_name")]
    center = base.copy()
    center[:3, 3] = np.mean(pads, axis=0)
    frames = {"urdf_link7": pose_from_mujoco(data.body("link_7")),
              "pika_model_origin": base,
              "simulation_tcp": pose_from_mujoco(data.site(robot["tcp_sites"][group])),
              "pad_center": center}
    # 编译 mesh 顶点在 geom frame；xmat/xpos 已包含 MuJoCo 质心修正。
    vertices = []
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if not (name.startswith(("pika_gripper_left_link_visual", "pika_gripper_right_link_visual"))
                and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        local = model.mesh_vert[start:start + count]
        world = local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        vertices.append((world - base[:3, 3]) @ base[:3, :3])
    if not vertices:
        raise ValueError("Pika 指部 visual mesh 不存在，无法检查最远几何位置")
    points = np.concatenate(vertices)
    lower, upper = points.min(axis=0), points.max(axis=0)
    tip = base.copy()
    distal = (lower + upper) / 2
    distal[0] = upper[0]
    tip[:3, 3] = base[:3, :3] @ distal + base[:3, 3]
    frames["finger_distal_center"] = tip
    return {"joints_rad": joints.tolist(), "requested_width_m": width_m,
            "pad_gap_m": float(np.linalg.norm(pads[0] - pads[1])),
            "finger_extent_in_pika_model_m": {"min": lower.tolist(), "max": upper.tolist()},
            "frames": {key: value.tolist() for key, value in frames.items()}}
