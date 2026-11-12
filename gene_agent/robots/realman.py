"""保真导入 Realman URDF，并显式组合安装、TCP 和场景本体。

此模块只构建仿真资产，不导入厂商 SDK，也不建立网络连接。
世界坐标默认定义为右臂 SDK base；所有长度为米、关节角为弧度。
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .gripper import generate_gripper_urdf

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/robots/realman"
CONFIG = ROOT / "config/robots"


def _numbers(values) -> str:
    """保留双精度，避免 MuJoCo 默认 XML 导出的小数截断累积到 TCP。"""
    return " ".join(format(float(x), ".17g") for x in values)


def _pose_attributes(transform: np.ndarray) -> dict[str, str]:
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return {"pos": _numbers(transform[:3, 3]), "quat": _numbers(quaternion[[3, 0, 1, 2]])}


def _origin(element: ET.Element | None) -> np.ndarray:
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        # URDF 固定轴 rpy 等于 Rz(yaw) @ Ry(pitch) @ Rx(roll)。
        result[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return result


def load_robot_config() -> dict:
    """返回本模型唯一安装和夹爪配置来源；不隐式选择右臂为世界。"""
    return json.loads((CONFIG / "realman.json").read_text(encoding="utf-8"))


def build_robot(output_dir: Path, *, world_frame: str | None = None,
                gripper_mode: str | None = None,
                profile: str = "realman_dual_standard") -> dict:
    """生成可独立编译、也可通过 include 合并的 robot.xml 和元数据。

    关节顺序来自各组 URDF，调用方应按名字映射 qpos，不能假定每组七轴。
    返回路径为绝对路径；生成 XML 引用仓库中的已冻结 mesh，不访问旧项目。
    """
    if profile == "realman_single_pika":
        if gripper_mode is not None:
            raise ValueError("Pika 配置由 profile 选择，不接受旧 gripper_mode")
        from .realman_pika import build_pika_robot
        return build_pika_robot(output_dir, world_frame=world_frame)
    if profile != "realman_dual_standard":
        raise ValueError(f"未知机器人 profile: {profile}")
    world_frame = world_frame or "right_sdk_base"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_robot_config()
    gripper_mode = gripper_mode or config.get("gripper_mode", "parallel_slide_proxy")
    if gripper_mode not in {"parallel_slide_proxy", "fixed_urdf"}:
        raise ValueError(f"未知 gripper_mode: {gripper_mode}")
    installation = json.loads((ASSETS / "installation.json").read_text(encoding="utf-8"))[config["installation"]]
    if world_frame == "right_sdk_base":
        world_from_body = np.linalg.inv(installation["body_from_sdk"]["right"])
    elif world_frame == "installation_body":
        world_from_body = np.eye(4)
    else:
        raise ValueError(f"未知 world_frame: {world_frame}")
    gripper = json.loads((CONFIG / "standard_gripper.json").read_text(encoding="utf-8"))
    flange_tcp = np.eye(4)
    flange_tcp[:3, :3] = Rotation.from_euler("z", gripper["flange_to_tcp_rotation_z_deg"], degrees=True).as_matrix()
    flange_tcp[2, 3] = gripper["flange_to_tcp_z_m"]
    root = ET.Element("mujoco", model="realman_dual_arm")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    actuators = ET.SubElement(root, "actuator")
    groups, tcp_sites, home, transforms, grippers = {}, {}, {}, {}, {}
    equality = ET.SubElement(root, "equality")
    for group in config["groups"]:
        source = ASSETS / "urdf" / f"realman_{group}.urdf"
        frozen_urdf = ET.parse(source).getroot()
        # arm关节组只取冻结源的可动关节；新增夹爪不能进入7DOF规划组或SDK映射。
        arm_names = [j.get("name") for j in frozen_urdf.findall("joint") if j.get("type") != "fixed"]
        link_tcp = np.linalg.inv(installation["sdk_flange_to_curobo_frame"][group]) @ flange_tcp
        derived = None
        if gripper_mode == "parallel_slide_proxy":
            source, derived = generate_gripper_urdf(group, link_tcp)
        urdf = ET.parse(source).getroot()
        # 使用 MuJoCo 自带 URDF 解析器保留 visual、collision、惯性及 fixed 链。
        # strippath=false 保留各资产目录，fusestatic=false 保留可追溯 link 名。
        extension = urdf.find("mujoco")
        if extension is None:
            extension = ET.SubElement(urdf, "mujoco")
        compiler = extension.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(extension, "compiler")
        compiler.attrib.update(discardvisual="false", fusestatic="false", strippath="false")
        for mesh in urdf.findall(".//mesh"):
            mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
        imported_model = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding="unicode"))
        imported_path = output_dir / f"{group}_imported.xml"
        mujoco.mj_saveLastXML(str(imported_path), imported_model)
        imported = ET.parse(imported_path).getroot()
        # 导出器仅输出六位有效数字。直接取原 URDF 的 joint origin / axis 回填，
        # 不修正原资产中的 1.5708 等近似值，也不重新定义末端姿态。
        for joint in urdf.findall("joint"):
            body = imported.find(f".//body[@name='{joint.find('child').get('link')}']")
            body.attrib.update(_pose_attributes(_origin(joint.find("origin"))))
            if joint.get("type") != "fixed":
                mj_joint = body.find("joint")
                axis = joint.find("axis")
                mj_joint.set("axis", "1 0 0" if axis is None else axis.get("xyz"))
                mj_joint.set("damping", str(config["joint_damping"]))
                mj_joint.set("armature", str(config["joint_armature"]))
        names = arm_names
        groups[group] = [f"{group}_{name}" for name in names]
        recorded = json.loads((CONFIG / f"realman_{group}_default_pose.json").read_text(encoding="utf-8"))
        if len(recorded["joints_rad"]) != len(names):
            raise ValueError(f"{group} 默认关节数量与 URDF 不符")
        home.update(zip(groups[group], recorded["joints_rad"]))
        # 只给已有名称和引用加组前缀；左右两臂不会共享 joint / mesh 命名空间。
        for node in imported.iter():
            for attr in ("name", "mesh", "material"):
                if attr in node.attrib:
                    node.set(attr, f"{group}_{node.get(attr)}")
        for mesh in imported.find("asset"):
            assets.append(mesh)
        body_from_urdf = np.asarray(installation["body_from_sdk"][group]) @ np.linalg.inv(installation["sdk_to_curobo"][group])
        mount = ET.SubElement(world, "body", name=f"{group}_mount", **_pose_attributes(world_from_body @ body_from_urdf))
        for body in imported.find("worldbody"):
            mount.append(body)
        flange = mount.find(f".//body[@name='{group}_{config['flange_link']}']")
        link_tcp = np.linalg.inv(installation["sdk_flange_to_curobo_frame"][group]) @ flange_tcp
        tcp_sites[group] = f"{group}_tcp"
        ET.SubElement(flange, "site", name=tcp_sites[group], size="0.006", rgba="1 0.25 0.05 1", **_pose_attributes(link_tcp))
        transforms[group] = {"body_from_urdf_base": body_from_urdf.tolist(), "world_from_urdf_base": (world_from_body @ body_from_urdf).tolist(), "link_from_tcp": link_tcp.tolist()}
        for name in groups[group]:
            joint = root.find(f".//joint[@name='{name}']")
            ET.SubElement(actuators, "position", name=f"{name}_position", joint=name,
                          kp=str(config["position_kp"]), kv=str(config["position_kv"]),
                          ctrlrange=joint.get("range"), forcerange=joint.get("actuatorfrcrange"))
        for link_body in mount.findall(".//body"):
            for index, geom in enumerate(link_body.findall("geom")):
                role = "visual" if geom.get("contype") == "0" else "collision"
                geom.set("name", f"{link_body.get('name')}_{role}_{index}")
        if derived is not None:
            proxy = derived["config"]
            descriptor = {"mode": gripper_mode, "joint_names": [], "actuator_names": [],
                          "pad_geom_names": [], "collision_geom_names": [], "planning_link_by_body": {},
                          "width_range_m": proxy["width_range_m"], "home_width_m": proxy["home_width_m"],
                          "width_mapping": "w=q_positive+q_negative; command q_positive=q_negative=w/2",
                          "pad_size_m": [proxy["pad_thickness_m"], proxy["pad_y_width_m"], proxy["pad_z_length_m"]],
                          "pad_tip_z_m": proxy["pad_tip_z_m"], "actuator_force_limit_N": proxy["actuator_force_limit_N"],
                          "finger_mass_kg": proxy["finger_mass_kg"], "fingers": {},
                          "generated_urdf_path": derived["generated_urdf_path"],
                          "source_hashes": {key: derived[key] for key in ["generated_urdf_sha256", "source_urdf_sha256", "config_sha256"]},
                          "config": proxy, "assumptions": proxy["assumptions"]}
            for label, finger in derived["fingers"].items():
                joint_name = f"{group}_{finger['joint_name']}"
                body_name = f"{group}_{finger['link_name']}"
                actuator_name = f"{joint_name}_position"
                finger_body = mount.find(f".//body[@name='{body_name}']")
                joint = finger_body.find("joint")
                joint.set("damping", str(proxy["joint_damping_Ns_per_m"]))
                joint.set("armature", str(proxy["joint_armature_kg"]))
                pad = next(geom for geom in finger_body.findall("geom") if geom.get("contype") != "0")
                pad_name = f"{group}_gripper_{label}_pad"
                pad.set("name", pad_name)
                pad.attrib.update(friction=_numbers(proxy["friction"]), condim="4", solref=_numbers(proxy["solref"]), solimp=_numbers(proxy["solimp"]))
                site_name = f"{group}_gripper_{label}_inner"
                ET.SubElement(finger_body, "site", name=site_name, pos=_numbers(finger["inner_site_local_m"]), size=".0015", rgba="0 1 0 1")
                ET.SubElement(actuators, "position", name=actuator_name, joint=joint_name,
                              kp=str(proxy["actuator_kp_N_per_m"]), kv=str(proxy["actuator_kv_Ns_per_m"]),
                              ctrlrange=_numbers(np.array(proxy["width_range_m"])/2),
                              forcerange=_numbers([-proxy["actuator_force_limit_N"], proxy["actuator_force_limit_N"]]))
                home[joint_name] = proxy["home_width_m"]/2
                descriptor["joint_names"].append(joint_name)
                descriptor["actuator_names"].append(actuator_name)
                descriptor["pad_geom_names"].append(pad_name)
                collision_names=[]
                # URDF按pad、shank、carriage顺序插入；保留所有物理实体的可追踪名称。
                physical=[geom for geom in finger_body.findall("geom") if geom.get("contype") != "0"]
                parts=["pad", *proxy["moving_structure_boxes"]]
                if len(physical)!=len(parts):raise ValueError("夹爪碰撞数量与参数不一致")
                for geom,part in zip(physical,parts):
                    name=f"{group}_gripper_{label}_{part}"
                    geom.set("name",name)
                    geom.attrib.update(friction=_numbers(proxy["friction"]), condim="4", solref=_numbers(proxy["solref"]), solimp=_numbers(proxy["solimp"]))
                    collision_names.append(name)
                descriptor["collision_geom_names"].extend(collision_names)
                descriptor["planning_link_by_body"][body_name] = finger["source_fixed_link_name"]
                descriptor["fingers"][label] = {**finger, "joint_name": joint_name, "body_name": body_name,
                                                "actuator_name": actuator_name, "pad_geom_name": pad_name,
                                                "inner_site_name": site_name, "collision_geom_names": collision_names}
            ET.SubElement(equality, "joint", name=f"{group}_gripper_symmetry", joint1=descriptor["joint_names"][1],
                          joint2=descriptor["joint_names"][0], polycoef="0 1 0 0 0", solref=_numbers(proxy["solref"]))
            grippers[group] = descriptor
        else:
            grippers[group] = {"mode": "fixed_urdf", "joint_names": [], "actuator_names": [], "pad_geom_names": [],
                               "source_urdf_path": str(source), "limitations": ["冻结原始fixed夹爪，不可驱动开合"]}
        for geom in mount.findall(".//geom"):
            if geom.get("contype") != "0":
                # group=3 仅是显示分层，接触仍保留；viewer 可隐藏重复碰撞外观。
                geom.set("group", "3")
    ET.SubElement(assets, "mesh", name="robot_body_visual", file=str(ASSETS / "meshes/body/realman_body_without_arms.obj"))
    asset_offset = np.eye(4)
    asset_offset[2, 3] = config["body_asset_z_offset_m"]
    body = ET.SubElement(world, "body", name="robot_body", **_pose_attributes(world_from_body @ asset_offset))
    ET.SubElement(body, "geom", name="robot_body_visual", type="mesh", mesh="robot_body_visual", contype="0", conaffinity="0", group="1", rgba="0.65 0.69 0.74 1")
    for box in json.loads((ASSETS / "collision/robot_body.json").read_text(encoding="utf-8"))["cuboid"]:
        ET.SubElement(body, "geom", name=box["name"], type="box", pos=_numbers(box["pose"][:3]), quat=_numbers(box["pose"][3:]), size=_numbers(np.asarray(box["dims"])/2), group="3", rgba="0.3 0.5 0.6 0.25")
    # 本体代理与安装座、首节机械臂在真实装配里相交；它们不应产生推开
    # 机械臂的物理约束。与下方给 cuRobo 的安装例外使用同一组 link 名。
    mounting_links = ("base_link", "link_1")
    contact = ET.SubElement(root, "contact")
    for group in groups:
        for link in mounting_links:
            ET.SubElement(contact, "exclude", body1="robot_body", body2=f"{group}_{link}")
        # URDF 导入后的 base/link_1 碰撞 mesh 在关节处只有纳米级交叠，
        # MuJoCo 仍会施加强接触约束；相邻安装节不应靠这个接触承载。
        ET.SubElement(contact, "exclude", body1=f"{group}_base_link",
                      body2=f"{group}_link_1")
    xml_path = output_dir / "robot.xml"
    ET.indent(root)
    ET.ElementTree(root).write(xml_path, encoding="unicode")
    # 构建入口失败即抛异常，不能留下“看似成功”的不可编译模型供场景使用。
    mujoco.MjModel.from_xml_path(str(xml_path))
    provenance = json.loads((ASSETS / "provenance.json").read_text(encoding="utf-8"))
    result = dict(profile=profile, xml_path=str(xml_path), joint_names=sum(groups.values(), []), joint_groups=groups,
                  tcp_sites=tcp_sites, group_root_bodies={group: f"{group}_mount" for group in groups}, home_joint_positions=home, world_frame=world_frame, world_from_body=world_from_body.tolist(),
                  collision_protection_body_names=["robot_body"],
                  transforms=transforms,
                  mounting_contact_exceptions={"robot_links": list(mounting_links),
                    "source_geom_names": ["robot_body_body", "robot_body_head", "robot_body_back"],
                    "source": "AffordToolAct/tamp/curobo/planner_backend.py:241-258,3040-3076",
                    "semantics": "旧优化默认屏蔽两link；新最终复核仅允许这些link与指定本体proxy的安装接触"},
                  gripper=dict(name=gripper["name"], mode=gripper_mode, max_opening_width_m=gripper["max_opening_width_m"]), grippers=grippers,
                  provenance=provenance, limitations=[*config["limitations"], *([] if gripper_mode == "fixed_urdf" else next(iter(grippers.values()))["assumptions"])])
    (output_dir / "robot_metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
