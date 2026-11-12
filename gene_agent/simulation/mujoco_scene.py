"""同步 MuJoCo 场景装配；机器人与物体的构建逻辑各自独立。"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .runtime import MujocoRuntime


def compose_scene(robot_xml: Path, objects_xml: Path, output_path: Path,
                  *, simulation_attachment_pairs: dict[str, str] | None = None,
                  tcp_sites: dict[str, str] | None = None) -> Path:
    """组合已明确位于同一个 world 的模型，保留各自的碰撞定义。

    不在此处补偿任何坐标。资产生成器必须显式声明 world frame，调用方在组合前检查。
    include 使用绝对路径，避免命令从不同目录运行时加载错误资产。
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", model="gene_agent_scene")
    for path in (robot_xml, objects_xml):
        ET.SubElement(root, "include", file=str(Path(path).resolve()))
    if simulation_attachment_pairs:
        robot_tree = ET.parse(robot_xml)
        equality = ET.SubElement(root, "equality")
        for group, object_name in simulation_attachment_pairs.items():
            site_name = (tcp_sites or {}).get(group)
            if not site_name:
                raise ValueError(f"机器人组 {group} 缺少 TCP site")
            site = robot_tree.find(f".//site[@name='{site_name}']")
            if site is None:
                raise ValueError(f"机器人组 {group} 的 TCP site 不在机器人模型中")
            parent = next((body for body in robot_tree.iter("body") if site in body), None)
            if parent is None:
                raise ValueError(f"TCP site {site_name} 缺少父 body")
            ET.SubElement(equality, "weld", name=f"sim_attach_{group}_{object_name}",
                          body1=parent.attrib["name"], body2=object_name, active="false")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    from gene_agent.visualization import apply_scene_style
    # 展示样式集中配置，只添加照明与背景；不能借此改变接触或机器人几何。
    apply_scene_style(root)
    ET.indent(root)
    ET.ElementTree(root).write(output_path, encoding="unicode")
    return output_path


def configure_robot_gravity_compensation(robot_xml: Path, robot_metadata: dict,
                                         groups: tuple[str, ...]) -> None:
    """仅修改本次生成的机器人XML；给具名机器人组的所有后代body加重力补偿。

    真实机械臂控制器通常补偿本体重力。这个仿真开关只作用于机器人，
    不作用于碗、门等操作对象；XML落盘后模型指纹与回放状态仍可复现。
    """
    if not groups:
        return
    roots = robot_metadata.get("group_root_bodies", {})
    tree = ET.parse(robot_xml)
    for group in groups:
        if group not in roots:
            raise ValueError(f"未知重力补偿机器人组：{group}")
        root = tree.find(f".//body[@name='{roots[group]}']")
        if root is None:
            raise ValueError(f"机器人组{group}缺少根body：{roots[group]}")
        for body in root.iter("body"):
            body.set("gravcomp", "1")
    ET.indent(tree.getroot())
    tree.write(robot_xml, encoding="unicode")


def set_joint_positions(model, data, positions: dict[str, float]):
    """按名称初始化标量关节，拒绝未知名称及 free/ball，避免自由物体位姿被覆盖。

    这是 reset/检查工具，不能用于把执行轨迹瞬移成物理成功。
    先完整验证，再更新状态，防止错误输入留下半更新状态。
    """
    updates = []
    for name, value in positions.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        # pybind enum 对 numpy 整数的反向比较并不对称，统一整数后判断成员。
        if joint_id < 0 or int(model.jnt_type[joint_id]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        ):
            raise ValueError(f"未知或非标量关节: {name}")
        if not np.isfinite(value):
            raise ValueError(f"关节值必须有限: {name}")
        updates.append((joint_id, float(value)))
    for joint_id, value in updates:
        data.qpos[model.jnt_qposadr[joint_id]] = value
        data.qvel[model.jnt_dofadr[joint_id]] = 0
        # position actuator 以关节位置为目标；仅在 reset 同步，后续 step 不强写 qpos。
        for actuator_id in range(model.nu):
            kp = model.actuator_gainprm[actuator_id, 0]
            is_position = (model.actuator_gaintype[actuator_id] == mujoco.mjtGain.mjGAIN_FIXED
                           and model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_AFFINE
                           and kp > 0 and model.actuator_biasprm[actuator_id, 0] == 0
                           and model.actuator_biasprm[actuator_id, 1] == -kp
                           and model.actuator_dyntype[actuator_id] == mujoco.mjtDyn.mjDYN_NONE)
            if (is_position and model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT
                    and model.actuator_trnid[actuator_id, 0] == joint_id):
                # position 的目标是传动长度，gear 不一定为 1；motor/velocity 不应被赋成位置。
                data.ctrl[actuator_id] = model.actuator_gear[actuator_id, 0] * value
    mujoco.mj_forward(model, data)


def build_scene(output_dir: Path, *, seed: int = 0, perturb: bool = False,
                pose_overrides: dict[str, np.ndarray] | None = None,
                table_size_xy_m: tuple[float, float] | None = None,
                turntable_height_offset_m: float = 0.,
                microwave_floor_height_offset_m: float = 0.,
                robot_profile: str = "realman_dual_standard",
                gravity_compensated_groups: tuple[str, ...] = (),
                simulation_attachment_pairs: dict[str, str] | None = None,
                table_config: dict | None = None):
    """生成首批场景并初始化机器人 home，返回 model、data、可保存的元数据。"""
    from gene_agent.robots import build_robot
    from gene_agent.scene import build_objects

    output_dir = Path(output_dir).resolve()
    robot = build_robot(output_dir / "robot", profile=robot_profile)
    configure_robot_gravity_compensation(Path(robot["xml_path"]), robot,
                                         tuple(gravity_compensated_groups))
    objects = build_objects(output_dir / "objects", seed=seed, perturb=perturb,
                            pose_overrides=pose_overrides, table_size_xy_m=table_size_xy_m, table_config=table_config,
                            turntable_height_offset_m=turntable_height_offset_m,
                            microwave_floor_height_offset_m=microwave_floor_height_offset_m,
                            world_frame=robot["world_frame"])
    if robot["world_frame"] != objects["world_frame"]:
        raise ValueError(f"场景坐标不一致: {robot['world_frame']} != {objects['world_frame']}")
    path = compose_scene(
        robot["xml_path"], objects["xml_path"], output_dir / "scene.xml",
        simulation_attachment_pairs=simulation_attachment_pairs,
        tcp_sites=robot["tcp_sites"],
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    set_joint_positions(model, data, robot["home_joint_positions"])
    return model, data, {"xml_path": str(path), "robot": robot, "objects": objects,
                         "seed": seed, "perturb": perturb, "world_frame": robot["world_frame"],
                         "gravity_compensated_groups": list(gravity_compensated_groups),
                         "simulation_attachment_pairs": dict(simulation_attachment_pairs or {})}
