"""旧刷/长刷资产的同一现场适配；只做 FK，不执行 MuJoCo 动力学。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.robots.pika_frames import load_frame_profile, model_frames, pose_error, pose_from_mujoco
from gene_agent.robots.realman_pika import build_pika_robot
from .observation import validate_snapshot

ROOT = Path(__file__).resolve().parents[3]


def read_site(path: Path) -> dict:
    path = Path(path).resolve()
    site = json.loads(path.read_text(encoding="utf-8"))
    for key in ("scene_template", "frame_profile"):
        item = Path(site[key])
        site[key] = str((ROOT / item).resolve() if not item.is_absolute() else item)
        if not Path(site[key]).is_file():
            raise FileNotFoundError(site[key])
    for role, row in site["objects"].items():
        item = Path(row["concept_path"])
        row["concept_path"] = str((ROOT/item).resolve() if not item.is_absolute() else item)
        validate_pose(row["T_perception_asset"],field_name=f"T_perception_{role}_asset")
    if set(site["objects"]) != {"brush","cube","dustpan"}:
        raise ValueError("第一版只支持人工持刷、一个方块和一个簸箕")
    for key in ("T_sdk_urdf_base", "T_sdk_camera", "T_sdk_table"):
        validate_pose(site[key],field_name=key)
    if site.get("calibration_status") not in {"nominal_audit_only","site_verified"}:
        raise ValueError("请填写现场变换并设置 calibration_status，模板不能直接当标定结果")
    if not str(site.get("calibration_source","")).strip():
        raise ValueError("必须记录标定来源")
    if site["schema_version"] != 1:
        raise ValueError("不支持的现场配置版本")
    if not isinstance(site["max_candidates"],int) or not 1 <= site["max_candidates"] <= 64:
        raise ValueError("max_candidates 必须为 1–64")
    for key in ("max_flange_position_error_m","max_flange_rotation_error_deg"):
        if not np.isfinite(site[key]) or site[key] <= 0:
            raise ValueError(f"{key} 必须是正的有限阈值")
    if not np.isfinite(site["max_snapshot_age_s"]) or site["max_snapshot_age_s"] <= 0:
        raise ValueError("max_snapshot_age_s 必须为正")
    dimensions = np.asarray(site["table_size_xyz_m"],float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("桌面需要三个正的米制尺寸")
    # 当前 Sweep Knowledge 使用 world Z 为桌面法向；斜桌面不静默投影。
    table = np.linalg.inv(site["T_sdk_urdf_base"]) @ site["T_sdk_table"]
    if not np.allclose(table[:3,2],[0,0,1],atol=1e-5):
        raise ValueError("当前清扫 Knowledge 只支持水平桌面")
    for obstacle in site.get("extra_obstacles",[]):
        validate_pose(obstacle["T_sdk_obstacle"],field_name="T_sdk_obstacle")
        size=np.asarray(obstacle["size_xyz_m"],float)
        if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):
            raise ValueError("额外障碍物需要三个正的米制尺寸")
    if site["calibration_status"]=="site_verified":
        for role,row in site["objects"].items():
            if not str(row.get("alignment_source","")).strip():
                raise ValueError(f"{role} 需要 mesh→资产对齐的来源说明；相同 mesh 也须显式注明")
    site["site_file"] = str(path)
    site["site_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return site


def _attributes(pose):
    pose = validate_pose(pose)
    return {"pos":" ".join(map(str,pose[:3,3])),
            "quat":" ".join(map(str,Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]))}


def set_free_body(model,data,name,pose):
    body = model.body(name).id
    joint = int(model.body_jntadr[body])
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{name} 必须是 free body")
    address = model.jnt_qposadr[joint]
    data.qpos[address:address+3] = pose[:3,3]
    data.qpos[address+3:address+7] = Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]


def build_observed_scene(site: dict,snapshot: dict,output: Path, *,allow_synthetic=False,
                         allow_first_frame_preview=False):
    validate_snapshot(snapshot,max_age_s=site["max_snapshot_age_s"],allow_synthetic=allow_synthetic,
                      allow_first_frame_preview=allow_first_frame_preview)
    profile = load_frame_profile(site["frame_profile"])
    synthetic = snapshot["source"] == "synthetic_fixture"
    preview = snapshot["source"] == "first_frame_preview"
    if not synthetic and not preview and (profile["status"] != "site_verified" or site["calibration_status"] != "site_verified"):
        raise ValueError("真实现场规划需要经现场核验的 frame profile 和 site 配置")
    if not synthetic and not preview:
        if snapshot.get("site_sha256")!=site["site_sha256"] or snapshot.get("frame_profile_sha256")!=profile["sha256"]:
            raise ValueError("现场快照与当前标定/配置版本不同，请重新采集")
        if snapshot["camera_serial"]!=site["devices"]["camera"]["serial"]:
            raise ValueError("快照的相机与现场配置不同")
    output.mkdir(parents=True,exist_ok=True)
    robot = build_pika_robot(output / "robot",frame_profile_path=site["frame_profile"])
    tree = ET.parse(site["scene_template"])
    root = tree.getroot()
    for element in root.iter():
        if element.get("file"):
            element.set("file",str((Path(site["scene_template"]).parent/element.get("file")).resolve()))
    robot_xml = ET.parse(robot["xml_path"]).getroot()
    asset = root.find("asset")
    asset.extend(copy.deepcopy(list(robot_xml.find("asset"))))
    # 重新编译时旧 robot mesh 未引用无害；新的 mesh 名重复时使用新机器人来源。
    seen = set()
    for element in list(asset)[::-1]:
        identity = (element.tag,element.get("name"))
        if identity in seen:
            asset.remove(element)
        else:
            seen.add(identity)
    world = root.find("worldbody")
    original_bodies = {name: site["objects"][name]["body_name"] for name in site["objects"]}
    kept = set(original_bodies.values())
    omitted = []
    for node in list(world):
        if node.tag == "body" and node.get("name") not in kept:
            omitted.append(node.get("name")); world.remove(node)
        elif node.tag == "geom" and node.get("name") != "scene_table":
            omitted.append(node.get("name")); world.remove(node)
    world.insert(0,copy.deepcopy(robot_xml.find("worldbody/body")))
    for section in ("actuator","contact","equality","keyframe"):
        for element in root.findall(section):
            root.remove(element)
    for section in ("actuator","contact"):
        root.append(copy.deepcopy(robot_xml.find(section)))
    world_from_sdk = np.linalg.inv(site["T_sdk_urdf_base"])
    poses = {}
    concepts = output / "concepts"; concepts.mkdir(exist_ok=True)
    for role,row in site["objects"].items():
        body = world.find(f"body[@name='{row['body_name']}']")
        if body is None:
            raise ValueError(f"资产模板缺少角色 {role} 的 body")
        body.set("name",role)
        scale = float(row.get("uniform_scale", 1.))
        if not np.isfinite(scale) or not .1 <= scale <= 10:
            raise ValueError(f"{role} 资产缩放无效")
        if scale != 1.:
            # 仅显式现场覆盖；历史仿真资产不改。mesh 顶点缩放与 geom offset 各一次。
            scaled_meshes = set()
            for node in body.iter():
                if node is not body and node.get("pos"):
                    node.set("pos", " ".join(map(str, np.fromstring(node.get("pos"), sep=" ")*scale)))
                if node.tag == "geom" and node.get("size"):
                    node.set("size", " ".join(map(str, np.fromstring(node.get("size"), sep=" ")*scale)))
                mesh_name = node.get("mesh") if node.tag == "geom" else None
                if mesh_name and mesh_name not in scaled_meshes:
                    mesh_asset = asset.find(f"mesh[@name='{mesh_name}']")
                    original = np.fromstring(mesh_asset.get("scale", "1 1 1"), sep=" ")
                    mesh_asset.set("scale", " ".join(map(str, original*scale)))
                    scaled_meshes.add(mesh_name)
        poses[role] = world_from_sdk @ snapshot["T_sdk_object"][role] @ np.asarray(row["T_perception_asset"])
        body.attrib.update(_attributes(poses[role]))
        for free in body.findall("freejoint"):
            free.set("name",f"{role}_free")
        for joint in body.findall("joint"):
            if joint.get("type") == "free": joint.set("name",f"{role}_free")
        for geom in body.findall("geom"):
            if role == "brush" and geom.get("name","").endswith("head_proxy"):
                geom.set("name","brush_collision_head")
            elif role == "brush" and geom.get("name","").endswith("handle_proxy"):
                geom.set("name","brush_collision_handle")
        concept = json.loads(Path(row["concept_path"]).read_text(encoding="utf-8"))
        for part in concept["conceptualization"]:
            part["component"] = row.get("concept_part_aliases",{}).get(part["component"],part["component"])
            if scale != 1.:
                parameters = part["parameters"]
                parameters["size"] = (np.asarray(parameters["size"])*scale).tolist()
                transform = np.asarray(parameters["pose_matrix"], float)
                transform[:3,3] *= scale
                parameters["pose_matrix"] = transform.tolist()
        (concepts/f"{role}.json").write_text(json.dumps(concept,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    table_pose = world_from_sdk @ site["T_sdk_table"]
    table = world.find("geom[@name='scene_table']")
    table.attrib.update(_attributes(table_pose))
    table.set("size"," ".join(map(str,np.asarray(site["table_size_xyz_m"])/2)))
    for index,obstacle in enumerate(site.get("extra_obstacles",[])):
        ET.SubElement(world,"geom",name=f"site_obstacle_{index}",type="box",
            size=" ".join(map(str,np.asarray(obstacle["size_xyz_m"])/2)),
            rgba=".5 .5 .5 1",**_attributes(world_from_sdk@np.asarray(obstacle["T_sdk_obstacle"])))
    path = output / "scene.xml"; tree.write(path,encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path)); data = mujoco.MjData(model)
    model_frames(model,data,robot,snapshot["joint_positions_rad"],snapshot["gripper_width_m"])
    for role,pose in poses.items():
        if role != "dustpan": set_free_body(model,data,role,pose)
    mujoco.mj_forward(model,data)
    support_adjustment = None
    policy = site.get("tabletop_support_policy", {})
    if preview and policy.get("enabled"):
        from gene_agent.scene.tabletop_support import correct_tabletop_support
        support_adjustment = correct_tabletop_support(model, data, ("brush","cube","dustpan"),
            "scene_table", max_adjustment_m=policy.get("max_adjustment_m", .03),
            clearance_m=policy.get("clearance_m", .001))
        # 固定簸箕的变换也要写进 XML；free-body 实际初态另存 npz。
        mujoco.mj_saveLastXML(str(path), model)
        site["support_adjustment"] = support_adjustment
        (output/"support_adjustment.json").write_text(json.dumps(support_adjustment,
            ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    link = data.body("link_7")
    nominal = np.eye(4); nominal[:3,:3]=link.xmat.reshape(3,3);nominal[:3,3]=link.xpos
    sdk_in_world = world_from_sdk @ snapshot["T_sdk_flange"]
    flange_error = pose_error(nominal @ profile["T_link7_sdk_flange"],sdk_in_world)
    if (flange_error["translation_m"] > site["max_flange_position_error_m"]
            or flange_error["rotation_deg"] > site["max_flange_rotation_error_deg"]):
        raise ValueError(f"SDK 法兰与当前模型 FK 不一致：{flange_error}")
    tcp = np.eye(4);tcp[:3,:3]=data.site("pika_tcp").xmat.reshape(3,3);tcp[:3,3]=data.site("pika_tcp").xpos
    observed_tcp = sdk_in_world @ profile["T_sdk_flange_operation_tcp"]
    observed_grip = np.linalg.inv(observed_tcp) @ poses["brush"]
    # 把同一份观测刷子绑定到当前规划模型的 TCP。保留 SDK 原始关系与
    # 这次注册误差，不把两套 FK 的差值悄悄变成预测开始时的物体移动。
    # 这是本帧的规划注册，不是修正机器人 DH，也不授权按此开环执行。
    model_from_observed = np.linalg.inv(tcp) @ observed_tcp
    grip = np.linalg.inv(tcp) @ pose_from_mujoco(data.body("brush"))
    if site.get("base_table_mounting_contact") is True:
        robot["mounting_contact_exceptions"] = {"robot_links":["base_link"],
            "source_geom_names":["scene_table"],"source":"explicit_site_fixed_base_table_mount",
            "semantics":"仅固定底座与桌面安装交叠"}
    (output/"robot/robot_metadata.json").write_text(json.dumps(robot,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    np.savez(output/"initial_state.npz",qpos=data.qpos.copy())
    provenance = {"physics_stepped":False,"source":snapshot["source"],"omitted_template_items":omitted,
                  "support_adjustment":support_adjustment,
                  "execution_enabled":False,"initial_brush_state":snapshot.get("initial_brush_state","held_confirmed"),
                  "robot_state_source":snapshot.get("robot_state_source","snapshot"),
                  "site_sha256":site["site_sha256"],"frame_profile_sha256":profile["sha256"],
                  "flange_fk_error":flange_error,"T_tcp_brush":grip.tolist(),
                  "T_world_tcp_observed_from_sdk":observed_tcp.tolist(),
                  "T_world_tcp_model_fk":tcp.tolist(),
                  "T_observed_sdk_tcp_brush":observed_grip.tolist(),
                  "sdk_model_tcp_registration":{"applied_for_planning":True,
                      "T_model_fk_tcp_observed_sdk_tcp":model_from_observed.tolist(),
                      "source":"initial_observation_registration_not_robot_calibration",
                      "error":pose_error(tcp,observed_tcp)},
                  "asset_sources_sha256":{**robot["source_hashes"],**{str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                      for p in [site["scene_template"],*(row["concept_path"] for row in site["objects"].values())]}},
                  "observed_object_poses_world":{role:pose.tolist() for role,pose in poses.items()},
                  "initial_object_poses_world":{role:pose_from_mujoco(data.body(role)).tolist() for role in poses}}
    (output/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return model,data,robot,grip,provenance
