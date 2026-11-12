"""正装 Realman 单臂与 Pika 原始 URDF 的可选仿真配置。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from .pika_geometry import INNER_SITE_LOCAL_M

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
ARM_URDF = ROOT / "assets/robots/realman/urdf/realman_left.urdf"
PIKA_URDF = ROOT / "assets/robots/pika/urdf/pika_gripper_description.urdf"
CONFIG = ROOT / "config/robots/realman_single_pika.json"


def _numbers(values) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def assemble_single_pika_urdf(output: Path, config: dict) -> Path:
    """移除旧夹爪与相机，连接原始 Pika URDF；不更改源资产。"""
    arm = ET.parse(ARM_URDF).getroot()
    keep_links = {"base_link", *(f"link_{index}" for index in range(1, 8))}
    for node in list(arm):
        if node.tag == "link" and node.get("name") not in keep_links:
            arm.remove(node)
        elif node.tag == "joint" and (
            node.find("parent").get("link") not in keep_links
            or node.find("child").get("link") not in keep_links
        ):
            arm.remove(node)
    for mesh in arm.findall(".//mesh"):
        mesh.set("filename", str((ARM_URDF.parent / mesh.get("filename")).resolve()))

    pika = ET.parse(PIKA_URDF).getroot()
    for mesh in pika.findall(".//mesh"):
        mesh.set("filename", str((PIKA_URDF.parent.parent / "meshes" / Path(mesh.get("filename")).name).resolve()))
    for link in pika.findall("link"):
        link.set("name", f"pika_{link.get('name')}")
        arm.append(link)
    for joint in pika.findall("joint"):
        joint.set("name", f"pika_{joint.get('name')}")
        joint.find("parent").set("link", f"pika_{joint.find('parent').get('link')}")
        joint.find("child").set("link", f"pika_{joint.find('child').get('link')}")
        arm.append(joint)
    mount = ET.SubElement(arm, "joint", name="pika_mount", type="fixed")
    ET.SubElement(mount, "origin",
                  xyz=_numbers(config["T_link7_pika_base_xyz_m"]),
                  rpy=_numbers(config["T_link7_pika_base_rpy_rad"]))
    ET.SubElement(mount, "parent", link="link_7")
    ET.SubElement(mount, "child", link="pika_gripper_base_link")
    extension = arm.find("mujoco")
    if extension is None:
        extension = ET.SubElement(arm, "mujoco")
    ET.SubElement(extension, "compiler", discardvisual="false", fusestatic="false", strippath="false")
    path = output / "realman_single_pika.urdf"
    ET.indent(arm)
    ET.ElementTree(arm).write(path, encoding="unicode")
    return path


def _link_from_tcp(config: dict) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", config["T_link7_pika_base_rpy_rad"]).as_matrix()
    transform[:3, 3] = config["T_link7_pika_base_xyz_m"]
    base_tcp = np.eye(4)
    base_tcp[:3, 3] = config["T_pika_base_tcp_xyz_m"]
    return transform @ base_tcp


def build_pika_robot(output_dir: Path, *, world_frame: str | None = None,
                     frame_profile_path: str | Path | None = None) -> dict:
    """返回与旧 build_robot 同形的具名元数据，但只包含单臂与 Pika。"""
    import mujoco
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    profile = None
    if frame_profile_path is not None:
        from .pika_frames import load_frame_profile
        profile = load_frame_profile(frame_profile_path)
        mount = profile["T_link7_pika_model"]
        config["T_link7_pika_base_xyz_m"] = mount[:3, 3].tolist()
        config["T_link7_pika_base_rpy_rad"] = Rotation.from_matrix(mount[:3,:3]).as_euler("xyz").tolist()
        config["T_pika_base_tcp_xyz_m"] = profile["T_pika_model_operation_tcp"][:3,3].tolist()
    if world_frame is not None and world_frame != config["world_frame"]:
        raise ValueError(f"Pika 单臂 world_frame 必须为 {config['world_frame']}")
    urdf = assemble_single_pika_urdf(output, config)
    imported = mujoco.MjModel.from_xml_path(str(urdf))
    xml_path = output / "robot.xml"
    mujoco.mj_saveLastXML(str(xml_path), imported)
    root = ET.parse(xml_path).getroot()
    base = root.find(".//body[@name='pika_gripper_base_link']")
    left = root.find(".//body[@name='pika_gripper_left_link']")
    right = root.find(".//body[@name='pika_gripper_right_link']")
    if base is None or left is None or right is None:
        raise ValueError("Pika URDF 导入后缺少预期 link")
    for body in root.findall(".//body"):
        for index, geom in enumerate(body.findall("geom")):
            role = "visual" if geom.get("contype", "1") == "0" else "collision"
            geom.set("name", f"{body.get('name')}_{role}_{index}")
            geom.set("group", "1" if role == "visual" else "3")
            if role == "collision":
                geom.set("rgba", "0.2 0.8 0.3 0")
                geom.set("friction", "0.8 0.01 0.001")
                geom.set("condim", "4")
    tcp_attributes = {}
    if profile is not None:
        tcp_attributes["quat"] = _numbers(Rotation.from_matrix(
            profile["T_pika_model_operation_tcp"][:3,:3]).as_quat()[[3,0,1,2]])
    ET.SubElement(base, "site", name="pika_tcp", pos=_numbers(config["T_pika_base_tcp_xyz_m"]),
                  size=".005", rgba="1 .25 .05 1", **tcp_attributes)
    # 内面位置来自 Pika 指端 STL 的 Y 边界；宽度是这两个 site 的距离。
    sites = {"positive": (left, "pika_left_inner", INNER_SITE_LOCAL_M['left']),
             "negative": (right, "pika_right_inner", INNER_SITE_LOCAL_M['right'])}
    for body, name, position in sites.values():
        ET.SubElement(body, "site", name=name, pos=_numbers(position), size=".002", rgba="0 1 0 1")
    actuators = ET.SubElement(root, "actuator")
    arm_names = [f"joint_{index}" for index in range(1, 8)]
    finger_names = ["pika_gripper_left_joint", "pika_gripper_right_joint"]
    for name in arm_names + finger_names:
        joint = root.find(f".//joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"导入模型缺少关节 {name}")
        finger = name in finger_names
        # URDF importer 不会自动给未声明的关节补稳定化参数。机械臂沿用现有
        # Realman 模型的阻尼/armature；Pika 的轻指节使用明确的仿真近似。
        joint.set("damping", ".5" if finger else "1")
        joint.set("armature", ".05" if finger else ".02")
        ET.SubElement(actuators, "position", name=f"{name}_position", joint=name,
                      kp=str(config["gripper_position_kp"] if finger else 150),
                      kv=str(config["gripper_position_kv"] if finger else 20),
                      ctrlrange=joint.get("range"),
                      forcerange=_numbers([-config["gripper_force_limit_N"], config["gripper_force_limit_N"]])
                      if finger else joint.get("actuatorfrcrange"))
    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", body1="base_link", body2="link_1")
    ET.indent(root)
    ET.ElementTree(root).write(xml_path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    expected = float(config["open_gap_from_urdf_mesh_m"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    measured = float(data.site("pika_left_inner").xpos @ data.body("pika_gripper_base_link").xmat.reshape(3, 3)[:, 1]
                     - data.site("pika_right_inner").xpos @ data.body("pika_gripper_base_link").xmat.reshape(3, 3)[:, 1])
    if not np.isclose(measured, expected, atol=1e-4):
        raise ValueError(f"Pika URDF 开口与资产配置不符: {measured} vs {expected}")
    group = config["arm_group"]
    gripper = {
        "mode": "pika_urdf", "joint_names": finger_names,
        "actuator_names": [f"{name}_position" for name in finger_names],
        "width_range_m": [0.0, measured], "home_width_m": measured,
        "positive_inner_site_name": "pika_left_inner", "negative_inner_site_name": "pika_right_inner",
        "closing_axis_tcp": [0, 1, 0],
        "pad_geom_names": [],
        "source_urdf_path": str(PIKA_URDF),
        "limitations": ["开口 site 来自原 STL 边界，未经实机内面标定。",
                        "MuJoCo 使用原 URDF mesh 的凸接触几何，尚未分解或标定摩擦。"],
    }
    if profile is not None:
        gripper["closing_axis_tcp"] = (
            profile["T_pika_model_operation_tcp"][:3,:3].T @ np.array([0.,1.,0.])).tolist()
    metadata = {
        "profile": config["profile"], "xml_path": str(xml_path), "planning_urdf_path": str(urdf),
        "world_frame": config["world_frame"], "joint_groups": {group: arm_names},
        "joint_names": arm_names, "group_root_bodies": {group: "base_link"},
        "tcp_sites": {group: "pika_tcp"},
        "home_joint_positions": {name: 0.0 for name in arm_names + finger_names},
        "transforms": {group: {"world_from_urdf_base": np.eye(4).tolist(),
                                "link_from_tcp": _link_from_tcp(config).tolist()}},
        "grippers": {group: gripper}, "gripper": {"name": "pika", "mode": "pika_urdf",
                                             "max_opening_width_m": measured},
        "mounting_contact_exceptions": {},
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (ARM_URDF, PIKA_URDF, CONFIG,
                                       *(PIKA_URDF.parent.parent / "meshes").glob("*.STL"))},
        "limitations": [config["tcp_source"], *gripper["limitations"],
                        "Pika 视觉装配方向已由用户确认；法兰实际平移仍需实物标定。",
                        "cuRobo 使用 Pika 原始 STL 外接球近似；碰撞与净空样例仍需审计。"],
    }
    if profile is not None:
        metadata["transforms"][group]["link_from_tcp"] = profile["T_link7_operation_tcp"].tolist()
        metadata["frame_profile"] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                      for key,value in profile.items()}
        metadata["source_hashes"][profile["path"]] = profile["sha256"]
    (output / "robot_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata
