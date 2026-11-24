"""以本次冻结 RGB-D 和已注册 data mesh 构建真机规划预览；不调用感知或设备。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.motion.curobo.scene import export_collision_scene
from .perception_setup import ROOT, BOWL, MICROWAVE


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _mesh_source(raw, setup):
    path = Path(raw)
    if not path.is_file():
        # 只支持已带哈希的工程迁移；不能换成任意同名 mesh。
        marker = "/data/task_1/"
        suffix = str(path).replace("\\", "/").partition(marker)[2]
        expected = setup.get("input_sha256", {}).get(str(raw))
        candidate = ROOT/"data/task_1"/suffix
        if not suffix or not expected or not candidate.is_file() or _sha(candidate) != expected:
            raise FileNotFoundError(f"本次注册 mesh 不存在或迁移后哈希不一致: {raw}")
        path = candidate
    expected = setup.get("input_sha256", {}).get(str(raw))
    if expected and _sha(path) != expected:
        raise ValueError(f"注册后的 mesh 已改变: {path}")
    return path.resolve()


def capture_cloud(perception_dir, setup, *, T_world_sdk):
    """T_world_camera = T_world_sdk @ T_sdk_camera，仅施加一次，单位米。"""
    folder = Path(perception_dir).resolve()/"capture"
    paths = {name:folder/name for name in ("rgb.png","depth.png","camera.json")}
    camera = json.loads(paths["camera.json"].read_text(encoding="utf-8"))
    rgb = np.asarray(Image.open(paths["rgb.png"]).convert("RGB"))
    raw = np.asarray(Image.open(paths["depth.png"]), dtype=float)
    scale = {"millimeter":.001,"millimeters":.001,"mm":.001,
             "meter":1.,"meters":1.,"m":1.}.get(camera.get("depth_unit"))
    if scale is None:
        raise ValueError("capture/camera.json 必须明确 depth_unit")
    depth = raw*scale
    if depth.shape != rgb.shape[:2]:
        raise ValueError("RGB 和深度尺寸不同")
    K = np.asarray(camera["intrinsic_matrix"], float)
    declared = setup["camera"]
    if (camera.get("serial") != declared["serial"]
            or not np.allclose(K,declared["intrinsic_matrix"],atol=1e-6,rtol=0)
            or declared.get("height") not in (None,depth.shape[0])
            or declared.get("width") not in (None,depth.shape[1])):
        raise ValueError("PERCEPTION 冻结帧与本次标定的相机/内参/分辨率不一致")
    if camera.get("frame_number") != setup["reference_frame"]["frame_number"]:
        raise ValueError("PERCEPTION capture 与关闭参考不是同一冻结帧")
    for path in paths.values():
        expected = setup.get("input_sha256",{}).get(str(path))
        if expected is None:
            matches=[v for k,v in setup.get("input_sha256",{}).items()
                     if k.replace("\\","/").endswith("/capture/"+path.name)]
            expected=matches[0] if len(matches)==1 else None
        if expected and _sha(path)!=expected:
            raise ValueError(f"已确认的冻结帧被修改: {path}")
    transform = validate_pose(T_world_sdk)@validate_pose(declared["T_sdk_camera"])
    rows,cols = np.indices(depth.shape)
    camera_points = np.stack(((cols-K[0,2])*depth/K[0,0],
                              (rows-K[1,2])*depth/K[1,1],depth),axis=-1)
    points = camera_points@transform[:3,:3].T+transform[:3,3]
    valid = np.isfinite(depth)&(depth>.1)&(depth<=3.)
    return points,rgb,valid,transform,{
        "capture_dir":str(folder),"frame_number":camera["frame_number"],
        "timestamp_ms":camera.get("captured_at_unix_ms",camera.get("timestamp_ms")),
        "world_frame":"right_sdk_base","units":"m",
        "T_world_camera":transform.tolist(),
        "input_sha256":{str(p):_sha(p) for p in paths.values()},
        "depth_scale_m":scale,"intrinsic_matrix":K.tolist(),
        "image_size_wh":[depth.shape[1],depth.shape[0]]}


def fit_observed_table(points, camera_position, *, thickness_m):
    """从可见点云选最大近水平平面，使用其真实倾角和可见范围，不套历史 XY。"""
    points = np.asarray(points,float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points)>16000:
        points=points[np.linspace(0,len(points)-1,16000,dtype=int)]
    if len(points)<500:
        raise ValueError("桌面拟合缺少有效 RGB-D 点")
    rng=np.random.default_rng(0)
    triples=points[rng.integers(0,len(points),size=(384,3))]
    normals=np.cross(triples[:,1]-triples[:,0],triples[:,2]-triples[:,0])
    lengths=np.linalg.norm(normals,axis=1)
    normals=normals/np.maximum(lengths[:,None],1e-12)
    allowed=(lengths>1e-5)&(np.abs(normals[:,2])>=np.cos(np.deg2rad(25.)))
    normals=normals[allowed]
    offsets=np.sum(normals*triples[allowed,0],axis=1)
    if not len(normals):
        raise ValueError("点云中没有可支持桌面假设的平面；请检查标定")
    best_count,best_normal,best_offset=0,None,None
    for start in range(0,len(normals),32):
        distances=np.abs(points@normals[start:start+32].T-offsets[start:start+32])
        counts=(distances<.006).sum(axis=0)
        index=int(np.argmax(counts))
        if counts[index]>best_count:
            best_count=int(counts[index]);best_normal=normals[start+index];best_offset=offsets[start+index]
    inliers=points[np.abs(points@best_normal-best_offset)<.006]
    if len(inliers)<500 or len(inliers)<.08*len(points):
        raise ValueError("点云桌面平面支持不足，不能生成历史矩形桌面替代")
    center=inliers.mean(axis=0)
    _,_,vt=np.linalg.svd(inliers-center,full_matrices=False)
    normal=vt[-1]
    if np.dot(normal,np.asarray(camera_position)-center)<0: normal=-normal
    x=np.array([1.,0.,0.]);x-=normal*np.dot(x,normal);x/=np.linalg.norm(x)
    y=np.cross(normal,x); rotation=np.column_stack((x,y,normal))
    xy=(inliers-center)@rotation[:,:2]
    low,high=np.quantile(xy,[.005,.995],axis=0)
    size=high-low
    if np.min(size)<.2 or np.max(size)>3.:
        raise ValueError(f"桌面可见范围异常: {size.tolist()}")
    top_center=center+rotation[:,:2]@((low+high)/2)
    pose=np.eye(4);pose[:3,:3]=rotation
    pose[:3,3]=top_center-normal*float(thickness_m)/2
    residual=(inliers-center)@normal
    return {"table_pose_world":pose.tolist(),"table_size_xy_m":size.tolist(),
        "accepted_table_thickness_m":float(thickness_m),
        "top_center_world_m":top_center.tolist(),"normal_world":normal.tolist(),
        "source":"PERCEPTION_RGBD_visible_plane","inlier_count":len(inliers),
        "input_point_count":len(points),"rms_m":float(np.sqrt(np.mean(residual**2))),
        "footprint_is_visible_extent":True,
        "world_up_consistent":bool(normal[2]>.995),
        "limitations":["范围仅覆盖本次可见桌面；遮挡/视野外边界未知。",
                        "厚度沿用已确认值，不是单视角点云测量结果。",
                        "未翻转/平移相机标定以迎合仿真世界朝上约定。"]}


def rgbd_surface(points, valid, *, max_edge_m=.035):
    """只连接相邻有效深度点；不跨遮挡、深度跳变或缺失区补面。"""
    h,w=valid.shape
    ids=np.arange(h*w).reshape(h,w)
    a,b,c,d=ids[:-1,:-1],ids[1:,:-1],ids[:-1,1:],ids[1:,1:]
    faces=np.concatenate((np.stack((a,b,c),-1).reshape(-1,3),
                          np.stack((c,b,d),-1).reshape(-1,3)))
    vertices=points.reshape(-1,3)
    keep=valid.reshape(-1)[faces].all(axis=1)
    face_points=vertices[faces]
    edge=np.maximum.reduce([np.linalg.norm(face_points[:,i]-face_points[:,j],axis=1)
                            for i,j in ((0,1),(1,2),(2,0))])
    faces=faces[keep&(edge<=max_edge_m)]
    if len(faces)<100:
        raise ValueError("有效深度表面太少，无法重建预览")
    return vertices,faces


def _xml_pose(pose):
    return {"pos":" ".join(map(str,pose[:3,3])),
            "quat":" ".join(map(str,Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]))}


def reconstruct_plan_scene(initial_scene, *, perception_dir, setup, output_dir, T_world_sdk):
    """复制本次影子模型并替换现场显示/桌面，不改原记录、关节、门轴和其他碰撞保护。"""
    output=Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("点云重建输出目录必须为空")
    old_root=Path(initial_scene["scene_xml_path"]).resolve().parent
    if output.is_relative_to(old_root):
        raise ValueError("重建输出不能位于原场景内部")
    output.mkdir(parents=True,exist_ok=True)
    scene=deepcopy(initial_scene)
    points,rgb,valid,T_world_camera,source=capture_cloud(perception_dir,setup,T_world_sdk=T_world_sdk)
    step=4
    grid=points[::step,::step].copy()
    colors=rgb[::step,::step].copy()
    measured=valid[::step,::step].copy()
    background=measured.copy()
    bowl_path=_mesh_source(setup["meshes"][BOWL],setup)
    mw_dir=_mesh_source(setup["meshes"][MICROWAVE],setup).parent
    mesh_sources={"bowl":bowl_path,"cabinet":mw_dir/"obj_mesh.obj",
                  "door":mw_dir/"handle_door_mesh.obj","handle":mw_dir/"handle_mesh.obj"}
    for p in mesh_sources.values():
        if not p.is_file(): raise FileNotFoundError(f"缺少 data 实际 mesh: {p}")
    # 去掉冻结帧中待操作对象，避免开门/移碗后仍留下关闭门或原碗的影像。
    for name,path in mesh_sources.items():
        instance=BOWL if name=="bowl" else MICROWAVE
        world_data=validate_pose(T_world_sdk)@validate_pose(setup["closed_reference_poses_sdk"][instance])
        local=(grid-world_data[:3,3])@world_data[:3,:3]
        bounds=trimesh.load_mesh(path,process=False).bounds
        background &= ~np.all((local>=bounds[0]-.025)&(local<=bounds[1]+.025),axis=-1)
    # 拟合前排除物体影像，避免误选微波炉顶面；不再调用分割服务。
    table=fit_observed_table(grid[background],T_world_camera[:3,3],
                            thickness_m=scene["provenance"]["accepted_table_thickness_m"])
    source.update(table=table,source_meshes={k:{"path":str(v),"sha256":_sha(v)} for k,v in mesh_sources.items()})
    source["warnings"] = ([] if table["world_up_consistent"] else
        ["实测桌面朝上法向不满足规划世界 +Z；请核对相机外参与 SDK 坐标，不能自动翻转点云。"])
    source["claim"]="本次 PERCEPTION 冻结点云和 data mesh；非新感知、非物理执行"
    _write(output/"reconstruction.json",source)
    old_root=Path(scene["scene_xml_path"]).resolve().parent
    new_root=output/"scene"
    shutil.copytree(old_root,new_root)
    # 已生成的 include 指向旧运行目录；新副本只引用自己的 objects/robot。
    for p in new_root.rglob("*.xml"):
        tree=ET.parse(p)
        for node in tree.iter():
            file=node.get("file")
            if file:
                path=Path(file)
                if path.is_absolute() and path.is_relative_to(old_root):
                    node.set("file",str(new_root/path.relative_to(old_root)))
        tree.write(p,encoding="unicode")
    old_objects=Path(scene["scene_metadata"]["objects"]["xml_path"])
    objects_path=new_root/old_objects.relative_to(old_root)
    tree=ET.parse(objects_path);root=tree.getroot();asset=root.find("asset")
    generated=new_root/"objects/observed_assets";generated.mkdir(parents=True,exist_ok=True)
    # 保留机器人初态，但从背景中删除其初始几何，避免规划运动时重影。
    old_model=mujoco.MjModel.from_xml_path(str(scene["scene_xml_path"]))
    data=mujoco.MjData(old_model);data.qpos[:]=scene["snapshot"]["qpos_snapshot"]
    mujoco.mj_forward(old_model,data)
    flat=grid.reshape(-1,3)
    for gid in range(old_model.ngeom):
        body=old_model.body(int(old_model.geom_bodyid[gid])).name
        if not body.startswith(("left_","right_","robot_body")):continue
        local=(flat-data.geom_xpos[gid])@data.geom_xmat[gid].reshape(3,3)
        kind=int(old_model.geom_type[gid])
        if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
            mid=int(old_model.geom_dataid[gid]);start=int(old_model.mesh_vertadr[mid]);count=int(old_model.mesh_vertnum[mid])
            verts=old_model.mesh_vert[start:start+count]
            lo,hi=verts.min(0)-.015,verts.max(0)+.015
        else:
            half=np.asarray(old_model.geom_size[gid]).copy()+.015
            if kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE),int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
                half=np.array([half[0],half[0],half[0]+half[1]])
            lo,hi=-half,half
        background &= ~np.all((local>=lo)&(local<=hi),axis=1).reshape(background.shape)
    vertices,faces=rgbd_surface(grid,background)
    h,w=background.shape
    vv,uu=np.indices((h,w)); uv=np.column_stack((uu.reshape(-1)/max(w-1,1),1-vv.reshape(-1)/max(h-1,1)))
    obj=generated/"observed_environment.obj"
    with obj.open("w") as f:
        for v in vertices:f.write("v "+" ".join(format(float(x),".9g") for x in v)+"\n")
        for coord in uv:f.write(f"vt {coord[0]:.9g} {coord[1]:.9g}\n")
        for face in faces:f.write("f "+" ".join(f"{i+1}/{i+1}" for i in face)+"\n")
    Image.fromarray(rgb).save(generated/"observed_rgb.png")
    ET.SubElement(asset,"texture",name="observed_rgbd_texture",type="2d",file=str(generated/"observed_rgb.png"))
    ET.SubElement(asset,"material",name="observed_rgbd_material",texture="observed_rgbd_texture",texuniform="false",rgba="1 1 1 1",specular="0",shininess="0")
    ET.SubElement(asset,"mesh",name="observed_environment_mesh",file=str(obj))
    ET.SubElement(root.find("worldbody"),"geom",name="observed_rgbd_surface",type="mesh",
                  mesh="observed_environment_mesh",material="observed_rgbd_material",
                  contype="0",conaffinity="0",density="0",group="2")
    np.savez_compressed(output/"observed_pointcloud.npz",points_world=grid[measured],
                        colors=colors[measured],background_points_world=grid[background],
                        background_colors=colors[background])
    # 数据碗使用注册时的 object frame，不做 canonicalize/落稳/名义姿态修正。
    bowl_copy=generated/"data_bowl.obj";shutil.copyfile(bowl_path,bowl_copy)
    asset.find("mesh[@name='bowl_visual_mesh']").set("file",str(bowl_copy))
    root.find(".//geom[@name='bowl_visual']").set("rgba",".2 .5 .9 1")
    # data 微波炉 mesh 属于旧感知 object frame，显式逆 alignment 后挂到可动 link。
    asset_data=np.linalg.inv(validate_pose(setup["assets"]["alignment"]))
    door_joint=validate_pose(setup["assets"]["joints"]["door"]["T_asset_joint"])
    for body_name in ("microwave","microwave_door"):
        for geom in root.find(f".//body[@name='{body_name}']").findall("geom"):
            if geom.get("group")=="2" and "turntable" not in geom.get("name",""):
                geom.set("rgba","0 0 0 0")
    for name in ("cabinet","door","handle"):
        target=generated/f"data_microwave_{name}.obj";shutil.copyfile(mesh_sources[name],target)
        mesh_name=f"observed_microwave_{name}"
        ET.SubElement(asset,"mesh",name=mesh_name,file=str(target))
        parent=root.find(".//body[@name='microwave']" if name=="cabinet" else ".//body[@name='microwave_door']")
        transform=asset_data if name=="cabinet" else np.linalg.inv(door_joint)@asset_data
        ET.SubElement(parent,"geom",name=mesh_name+"_visual",type="mesh",mesh=mesh_name,
                      contype="0",conaffinity="0",density="0",group="2",
                      rgba="1 .55 .65 1" if name=="handle" else ".85 .85 .8 1",**_xml_pose(transform))
    table_geom=root.find(".//geom[@name='scene_table']")
    table_pose=np.asarray(table["table_pose_world"])
    table_geom.attrib.update(_xml_pose(table_pose))
    table_geom.set("size"," ".join(map(str,[*(np.array(table["table_size_xy_m"])/2),table["accepted_table_thickness_m"]/2])))
    table_geom.set("rgba",".2 .6 1 0");table_geom.set("group","3")
    # 实际纸箱显示来自深度表面；其既有规划保护代理仍保留。
    support=root.find(".//geom[@name='microwave_support_visual']")
    if support is not None:support.set("rgba","0 0 0 0")
    robot_xml=new_root/"robot"/Path(scene["robot_metadata"]["xml_path"]).name
    if robot_xml.is_file():
        robot_tree=ET.parse(robot_xml)
        proxy=robot_tree.find(".//geom[@name='robot_body_visual']")
        if proxy is not None:proxy.set("rgba","0 0 0 0")
        robot_tree.write(robot_xml,encoding="unicode")
    ET.indent(root);tree.write(objects_path,encoding="unicode")
    xml_path=new_root/Path(scene["scene_xml_path"]).relative_to(old_root)
    camera_tree=ET.parse(xml_path)
    camera_body=ET.SubElement(camera_tree.getroot(),"worldbody")
    T_world_render=T_world_camera@np.diag([1.,-1.,-1.,1.])
    fovy=np.degrees(2*np.arctan(source["image_size_wh"][1]/(2*source["intrinsic_matrix"][1][1])))
    ET.SubElement(camera_body,"camera",name="observed_rgbd_camera",fovy=str(fovy),
                  **_xml_pose(T_world_render))
    camera_tree.write(xml_path,encoding="unicode")
    model=mujoco.MjModel.from_xml_path(str(xml_path))
    if model.nq!=old_model.nq:
        raise ValueError("点云重建不应改变机器人或物体的关节结构")
    data=mujoco.MjData(model);data.qpos[:]=scene["snapshot"]["qpos_snapshot"]
    mujoco.mj_forward(model,data)
    snapshot=export_collision_scene(model,data,scene["robot_metadata"],scene["snapshot"]["active_group"],output/"collision")
    scene["snapshot"]=snapshot;scene["scene_xml_path"]=str(xml_path)
    scene["scene_metadata"]["xml_path"]=str(xml_path)
    scene["scene_metadata"]["objects"]["xml_path"]=str(objects_path)
    scene["scene_metadata"]["objects"]["sample_poses"]["table"]=table["table_pose_world"]
    scene["scene_metadata"]["objects"]["table"].update(table)
    scene["provenance"].update(table_pose_world=table["table_pose_world"],
        table_size_xy_m=table["table_size_xy_m"],collision_snapshot_sha256=snapshot["snapshot_sha256"],
        observed_geometry={"record":str(output/"reconstruction.json"),
            "pointcloud":str(output/"observed_pointcloud.npz"),
            "source_frame_number":source["frame_number"],
            "table_world_up_consistent":table["world_up_consistent"],
            "table_normal_world":table["normal_world"],"warnings":source["warnings"],
            "claim":"RGB-D 表面用于显示；桌面碰撞来自点云平面；其他碰撞仍为原保护代理"})
    source["background_point_count"]=int(background.sum());source["surface_triangle_count"]=len(faces)
    source["collision_snapshot_sha256"]=snapshot["snapshot_sha256"]
    _write(output/"reconstruction.json",source)
    _write(output/"initial_scene.json",scene)
    return scene
