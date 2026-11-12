"""把 MuJoCo 当前接触几何冻结为规划组 URDF base 中的 cuRobo 场景。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import copy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from ..configuration import CollisionMode, resolve_collision_mode


def planner_scene_options(snapshot: dict) -> dict:
    """唯一的快照→solver 场景参数适配；保留快速模式的机器人保护。"""
    result = {"scene_model": snapshot["scene_model"], "scene_scope": snapshot["scope"],
              "scene_snapshot_sha256": snapshot["snapshot_sha256"],
              "mounting_contact_policy": snapshot.get("mounting_contact_policy") or {}}
    if snapshot.get("collision_mode") == CollisionMode.ROBOT_ONLY.value:
        result.update(collision_mode=CollisionMode.ROBOT_ONLY.value,
                      robot_protection_scene=snapshot["scene_model"])
    return result


def export_self_collision_only_snapshot(
    source_snapshot: dict,
    output_dir: Path,
    *,
    attached_object_id: str | None = None,
    gripper_snapshot: dict | None = None,
) -> dict:
    """显式记录旧 ``ignore_scene_collision`` 的 world 切换。

    旧 InsertObjectSkill 从 pre-place 开始忽略外部场景障碍，但机器人
    自碰撞依然由 cuRobo 配置检查。这里生成独立 revision/哈希，不在
    原快照上原地删除障碍；视觉界面仍能看到哪些 world obstacle 被忽略。
    """
    source = dict(source_snapshot)
    if source.get("collision_mode") == CollisionMode.ROBOT_ONLY.value:
        return _write_robot_only_derived(source, output_dir, gripper_snapshot=gripper_snapshot)
    transferred = source.get("transferred_objects") or {}
    if attached_object_id is not None and attached_object_id not in transferred:
        raise ValueError("附着物必须存在于来源碰撞快照的转移记录")
    gripper = copy.deepcopy(gripper_snapshot or source.get("gripper_snapshot"))
    if not isinstance(gripper, dict) or not gripper.get("snapshot_id"):
        raise ValueError("仅自身碰撞快照需要明确的夹爪开口快照")
    ignored = sorted(str(row["obstacle_name"]) for row in source.get("records", ()))
    scope = "仅机器人自碰撞；旧 ignore_scene_collision=True；world 障碍停用"
    payload = {
        "source_snapshot_sha256": source.get("snapshot_sha256"),
        "gripper_snapshot_id": gripper["snapshot_id"],
        "attached_object_id": attached_object_id,
        "ignored_world_obstacles": ignored,
        "scope": scope,
    }
    digest = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    result = {
        "snapshot_sha256": digest,
        "source_snapshot_sha256": source.get("snapshot_sha256"),
        "scene_model": {},
        "scope": scope,
        "records": [],
        "excluded": [
            {**copy.deepcopy(row), "reason": "旧阶段 ignore_scene_collision=True"}
            for row in source.get("records", ())
        ],
        "ignored_world_obstacles": ignored,
        "gripper_snapshot": gripper,
        "mounting_contact_policy": {},
        "attached_object_id": attached_object_id,
        "transferred_objects": (
            {attached_object_id: copy.deepcopy(transferred[attached_object_id])}
            if attached_object_id is not None else {}
        ),
        "limitations": ["仅关闭 world 碰撞；机器人自碰撞仍启用；不代表物理接触成功"],
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "collision_snapshot.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def filter_collision_snapshot(
    source: dict, output_dir: Path, *,
    keep_body_names: set[str] | None,
    rejected_body_names: set[str] | None,
    rejected_geom_names: set[str] | None = None,
    attached_object_id: str | None,
    scope: str,
) -> dict:
    """从完整碰撞快照派生旧 Skill 所需的阶段 world，不改源快照。

    机器人基座在 cuRobo 中属于 world 障碍：调用方可只保留该 body，
    而不是把“忽略普通场景”误解为完全空 world。附着球独立保留。
    """
    if source.get("collision_mode") == CollisionMode.ROBOT_ONLY.value:
        return _write_robot_only_derived(source, output_dir)
    if keep_body_names is not None and rejected_body_names is not None:
        raise ValueError("保留集合和排除集合不能同时提供")
    if keep_body_names is not None and rejected_geom_names is not None:
        raise ValueError("保留 body 集合和排除 geom 集合不能同时提供")
    records = list(source["records"])
    included = [row for row in records if (
        (keep_body_names is None or row["body_name"] in keep_body_names)
        and (rejected_body_names is None or row["body_name"] not in rejected_body_names)
        and (rejected_geom_names is None or row["geom_name"] not in rejected_geom_names)
    )]
    names = {row["obstacle_name"] for row in included}
    scene = {
        kind: {name: copy.deepcopy(item)
               for name, item in source["scene_model"].get(kind, {}).items()
               if name in names}
        for kind in ("cuboid", "mesh")
    }
    unknown = names - set(scene["cuboid"]) - set(scene["mesh"])
    if unknown:
        raise ValueError(f"碰撞记录在 scene_model 中缺失：{sorted(unknown)}")
    transfers = source.get("transferred_objects", {})
    if attached_object_id is None:
        transfers = {}
    else:
        if attached_object_id not in transfers:
            raise ValueError(f"快照中没有附着物 {attached_object_id}")
        transfers = {attached_object_id: copy.deepcopy(transfers[attached_object_id])}
    result = copy.deepcopy(source)
    result["scene_model"] = scene
    result["records"] = included
    result["transferred_objects"] = transfers
    # 某阶段移除桌面等允许工具接触的障碍后，安装例外不能引用已不存在的
    # world 名称；保留源快照，派生策略只缩小到本阶段仍存在的障碍。
    mounting = copy.deepcopy(result.get("mounting_contact_policy") or {})
    mounting["obstacle_names"] = [name for name in mounting.get("obstacle_names", ()) if name in names]
    if not mounting["obstacle_names"]:
        mounting["robot_links"] = []
    result["mounting_contact_policy"] = mounting
    result["ignored_world_obstacles"] = [
        row["obstacle_name"] for row in records if row["obstacle_name"] not in names
    ]
    result["scope"] = scope
    payload = {
        "parent_sha256": source["snapshot_sha256"],
        "scene_model": scene,
        "records": included,
        "gripper_snapshot": result.get("gripper_snapshot"),
        "transferred_objects": transfers,
        "mounting_contact_policy": mounting,
        "scope": scope,
    }
    result["snapshot_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "collision_snapshot.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _write_robot_only_derived(source: dict, output_dir: Path, *, gripper_snapshot: dict | None = None) -> dict:
    """旧阶段筛选不能移除 robot_only 中的本体/另一臂保护。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = copy.deepcopy(source)
    if gripper_snapshot is not None:
        if not gripper_snapshot.get("snapshot_id"):
            raise ValueError("规划夹爪快照缺少 snapshot_id")
        result["gripper_snapshot"] = copy.deepcopy(gripper_snapshot)
        payload = {"parent_sha256": source["snapshot_sha256"], "gripper_snapshot": gripper_snapshot}
        result["snapshot_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    (output / "collision_snapshot.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


def _descendant(model, body_id, root_id):
    while body_id != 0:
        if body_id == root_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return root_id == 0


def _pose_list(transform):
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return [*transform[:3, 3].tolist(), *quaternion.tolist()]


def _mesh_for_geom(model, geom_id, margin=0.0):
    """读取编译后 mesh 局部顶点，不能再应用源 STL 的 scale/mesh pose。

    MuJoCo 已把 mesh 的质心/主轴变换合并到 geom pose；data.geom_xmat 与
    model.mesh_vert 直接配对才是接触世界几何。mesh 接触按凸包导出，保持
    当前 MuJoCo 默认凸碰撞语义，而不是让凹原始网格突然变成另一套形状。
    """
    kind = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id].copy()
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        mesh = trimesh.convex.convex_hull(np.array(model.mesh_vert[start:start+count], dtype=float))
        if margin:
            center=mesh.center_mass
            face_distances=np.einsum("ij,ij->i",mesh.face_normals,mesh.triangles_center-center)
            minimum=float(face_distances.min())
            if minimum<=0:raise ValueError("凸mesh无法构造保守margin膨胀")
            mesh.vertices=center+(mesh.vertices-center)*(1+margin/minimum)
        return mesh, "compiled_mesh_convex_hull"
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        # cuRobo当前后端原生仅支持cuboid/mesh。采用64段外接多边形避免圆周内缩；
        # 最大半径增加 sec(pi/64)-1≈0.12%，高度保持原值。
        radius = float(size[0]+margin)/np.cos(np.pi/64)
        return trimesh.creation.cylinder(radius=radius, height=2*(size[1]+margin), sections=64), "cylinder_circumscribed_64_segments"
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return _mesh_for_sphere(float(size[0]+margin))
    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        sphere, _ = _mesh_for_sphere(float(size[0]+margin))
        vertices = np.concatenate([sphere.vertices+[0, 0, size[1]], sphere.vertices-[0, 0, size[1]]])
        return trimesh.convex.convex_hull(vertices), "capsule_convex_hull_of_circumscribed_spheres"
    raise ValueError(f"不支持启用接触的 MuJoCo geom 类型 {kind}，geom_id={geom_id}")


def _mesh_for_sphere(radius):
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    distances = np.einsum("ij,ij->i", mesh.face_normals, mesh.triangles_center)
    mesh.apply_scale(radius/distances.min())
    return mesh, "sphere_circumscribed_icosphere_subdivision3"


def export_collision_scene(model, data, robot_metadata: dict, active_group: str, output_dir: Path,
                           *, transferred_objects: dict[str, list[list[float]]] | None = None,
                           moving_body_names: tuple[str, ...] = (),
                           disabled_body_names: tuple[str, ...] = (),
                           collision_mode: CollisionMode | str | None = None) -> dict:
    """显式冻结全场景当前接触几何，不生成允许碰撞列表。

    active_group所属body子树由原机器人碰撞球负责，因此只排除该子树。
    其他组、动态碗、桌面、本体等全部作为当前时刻静态障碍导出。
    所有包含/排除都有逐geom记录；未知启用类型直接抛错，不静默漏掉。
    此导出不模拟另一臂运动；执行前必须验证快照状态仍一致。
    """
    if resolve_collision_mode(collision_mode) is CollisionMode.ROBOT_ONLY:
        from .robot_scene import export_robot_only_snapshot
        return export_robot_only_snapshot(model, data, robot_metadata, active_group, output_dir)
    transferred_objects = transferred_objects or {}
    moving_body_names = tuple(moving_body_names)
    disabled_body_names = tuple(disabled_body_names)
    if len(set(moving_body_names)) != len(moving_body_names):
        raise ValueError("运动关节 body 不能重复")
    if len(set(disabled_body_names)) != len(disabled_body_names):
        raise ValueError("阶段关闭碰撞的 body 不能重复")
    transfer_roots = {}
    for object_id, spheres in transferred_objects.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        sphere_array = np.asarray(spheres, dtype=float)
        if body_id < 0 or sphere_array.ndim != 2 or sphere_array.shape[1] != 4 \
                or len(sphere_array) == 0 or not np.all(np.isfinite(sphere_array)) \
                or np.any(sphere_array[:, 3] <= 0):
            raise ValueError(f"附着物{object_id}必须存在且提供正半径世界坐标球代理")
        transfer_roots[object_id] = body_id
    if np.any(model.geom_gap != 0) or np.any(model.pair_gap != 0):
        raise NotImplementedError("非零 MuJoCo margin/gap 中gap尚不支持，禁止静默忽略")
    if np.any(model.geom_margin < 0) or np.any(model.pair_margin < 0):
        raise NotImplementedError("负接触margin尚不支持")
    if active_group not in robot_metadata["joint_groups"]:
        raise ValueError(f"未知 active_group: {active_group}")
    root_name = robot_metadata["group_root_bodies"][active_group]
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root_id < 0:
        raise ValueError(f"缺少规划组根body {root_name}")
    moving_roots = {}
    for name in moving_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"运动关节 body 不存在：{name}")
        if _descendant(model, body_id, root_id):
            raise ValueError(f"不能把活动组 body 当作运动场景障碍排除：{name}")
        if any(_descendant(model, body_id, other) or _descendant(model, other, body_id)
               for other in transfer_roots.values()):
            raise ValueError(f"运动障碍不能同时作为附着物：{name}")
        moving_roots[name] = body_id
    disabled_roots = {}
    for name in disabled_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"阶段关闭碰撞的 body 不存在：{name}")
        if _descendant(model, body_id, root_id):
            raise ValueError(f"不能关闭活动组 body 的原机器人碰撞球：{name}")
        if any(_descendant(model, body_id, other) or _descendant(model, other, body_id)
               for other in (*moving_roots.values(), *transfer_roots.values())):
            raise ValueError(f"阶段关闭碰撞的 body 不能同时是运动障碍或附着物：{name}")
        disabled_roots[name] = body_id
    if any(_descendant(model, body_id, root_id) for body_id in transfer_roots.values()):
        raise ValueError("附着物不能属于活动机器人组")
    effective_margins=np.array(model.geom_margin,copy=True)
    static_pairs=[]
    for pair_id in range(model.npair):
        ids=[int(model.pair_geom1[pair_id]),int(model.pair_geom2[pair_id])]
        if any(_descendant(model, int(model.geom_bodyid[g]), object_body)
               for g in ids for object_body in transfer_roots.values()):
            raise NotImplementedError("显式contact pair涉及转移中的附着物，不能静默忽略")
        if any(_descendant(model,int(model.geom_bodyid[g]),root_id) for g in ids):
            raise NotImplementedError("涉及活动组的显式contact pair尚不支持")
        if any(model.geom_contype[g]==0 and model.geom_conaffinity[g]==0 for g in ids):
            raise NotImplementedError("显式contact pair包含双零mask几何，不能静默遗漏")
        margin=float(model.pair_margin[pair_id])
        for g in ids:effective_margins[g]=max(effective_margins[g],margin)
        static_pairs.append({"pair_id":pair_id,"geom_ids":ids,"geom_names":[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) for g in ids],
                             "margin_m":margin,"scope":"非活动组静态障碍内部接触；几何全部导出，不模拟障碍间动力学"})
    if any(effective_margins[g]>0 and _descendant(model,int(model.geom_bodyid[g]),root_id) for g in range(model.ngeom)):
        raise NotImplementedError("活动组非零margin未映射到原机器人球")
    # mj_step 更新qpos后，geom_xpos可能仍为积分前缓存。复制整个data再forward，
    # 保证所有导出字段来自同一个当前状态，同时保留调用方warmstart等积分内部量。
    snapshot_data = mujoco.MjData(model)
    mujoco.mj_copyData(snapshot_data, model, data)
    mujoco.mj_forward(model, snapshot_data)
    data = snapshot_data
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    world_from_base = np.asarray(robot_metadata["transforms"][active_group]["world_from_urdf_base"])
    base_from_world = np.linalg.inv(world_from_base)
    scene, records, excluded = {"cuboid": {}, "mesh": {}}, [], []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        source = dict(geom_id=geom_id, geom_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                      body_id=body_id, body_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                      mujoco_type=int(model.geom_type[geom_id]), contype=int(model.geom_contype[geom_id]),
                      conaffinity=int(model.geom_conaffinity[geom_id]))
        if source["contype"] == 0 and source["conaffinity"] == 0:
            excluded.append(dict(source, reason="contact_masks_both_zero"))
            continue
        if _descendant(model, body_id, root_id):
            excluded.append(dict(source, reason="active_group_uses_original_robot_collision_spheres"))
            continue
        disabled_name = next((name for name, disabled_root in disabled_roots.items()
                              if _descendant(model, body_id, disabled_root)), None)
        if disabled_name is not None:
            excluded.append(dict(source, reason="stage_collision_policy_disabled",
                                 disabled_body_name=disabled_name))
            continue
        moving_name = next((name for name, moving_root in moving_roots.items()
                            if _descendant(model, body_id, moving_root)), None)
        if moving_name is not None:
            excluded.append(dict(source,
                                 reason="moving_articulation_requires_per_sample_validation",
                                 moving_body_name=moving_name))
            continue
        transferred_name = next((name for name, object_body in transfer_roots.items()
                                 if _descendant(model, body_id, object_body)), None)
        if transferred_name is not None:
            excluded.append(dict(source, reason="transferred_to_attached_robot_proxy",
                                 object_instance_id=transferred_name))
            continue
        world_geom = np.eye(4)
        world_geom[:3, :3] = data.geom_xmat[geom_id].reshape(3, 3)
        world_geom[:3, 3] = data.geom_xpos[geom_id]
        base_geom = base_from_world @ world_geom
        name = f"mj_geom_{geom_id}_{source['geom_name'] or 'unnamed'}"
        record = dict(source, obstacle_name=name, T_world_geom=world_geom.tolist(), T_base_geom=base_geom.tolist(),
                      geom_size=model.geom_size[geom_id].tolist(), contact_margin_m=float(effective_margins[geom_id]), snapshot_static=True)
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            # MuJoCo plane 数学上无限；cuRobo 只接受有限 cuboid/mesh。
            # 此处覆盖当前桌面工作区的 4 m 方形支撑面，并把其顶面与
            # 原 plane 严格对齐。超过此范围的路径必须由调用方拒绝。
            half_xy = np.maximum(model.geom_size[geom_id, :2], 2.)
            depth = max(float(2 * model.geom_size[geom_id, 2]), .02)
            support_pose = base_geom.copy()
            support_pose[:3, 3] -= support_pose[:3, 2] * (depth / 2)
            margin = float(effective_margins[geom_id])
            scene["cuboid"][name] = {
                "dims": [float(2 * (half_xy[0] + margin)),
                         float(2 * (half_xy[1] + margin)), depth + 2 * margin],
                "pose": _pose_list(support_pose),
            }
            record.update(export_kind="cuboid", approximation="finite_plane_support_cuboid",
                          finite_support_half_xy_m=half_xy.tolist(),
                          plane_top_world_z_m=float(world_geom[2, 3]))
        elif int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            scene["cuboid"][name] = {"dims": (2*(model.geom_size[geom_id]+effective_margins[geom_id])).tolist(), "pose": _pose_list(base_geom)}
            record.update(export_kind="cuboid", approximation="margin_box_expansion" if effective_margins[geom_id] else "none")
        else:
            mesh, approximation = _mesh_for_geom(model, geom_id, effective_margins[geom_id])
            mesh_path = output_dir / f"geom_{geom_id}.obj"
            mesh.export(mesh_path)
            scene["mesh"][name] = {"file_path": str(mesh_path), "pose": _pose_list(base_geom)}
            record.update(export_kind="mesh", approximation=approximation,
                          mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(), mesh_path=str(mesh_path))
        records.append(record)
    # 哈希不含输出目录，避免同一状态在不同run目录被当成不同快照；包括所有碰撞mask和
    # 源几何/pose、原始qpos以及世界映射，覆盖静态body位置变化及其他臂状态变化。
    digest_records = [{k:v for k,v in r.items() if k != "mesh_path"} for r in records]
    gripper_snapshot = None
    if robot_metadata.get("grippers", {}).get(active_group, {}).get("mode") == "parallel_slide_proxy":
        from .gripper import export_gripper_snapshot
        gripper_snapshot = export_gripper_snapshot(model, data, robot_metadata, active_group)
    elif robot_metadata.get("grippers", {}).get(active_group, {}).get("mode") == "pika_urdf":
        from .profiles.pika import export_pika_gripper_snapshot
        gripper_snapshot = export_pika_gripper_snapshot(model, data, robot_metadata, active_group)
    snapshot = dict(gripper_snapshot_id=gripper_snapshot["snapshot_id"] if gripper_snapshot else None, active_group=active_group, world_frame=robot_metadata["world_frame"],
                    base_from_world=base_from_world.tolist(), qpos=data.qpos.tolist(),
                    records=digest_records, excluded=excluded, static_internal_contact_pairs=static_pairs,
                    transferred_objects=transferred_objects,
                    moving_body_names=list(moving_body_names),
                    disabled_body_names=list(disabled_body_names),
                    mounting_contact_exceptions=robot_metadata.get("mounting_contact_exceptions", {}))
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    exception = robot_metadata.get("mounting_contact_exceptions", {})
    mounting_policy = {"robot_links": exception.get("robot_links", []),
                       "obstacle_names": [r["obstacle_name"] for r in records if r["geom_name"] in exception.get("source_geom_names", [])],
                       "source": exception.get("source"), "semantics": exception.get("semantics")}
    result = dict(gripper_snapshot=gripper_snapshot, static_internal_contact_pairs=static_pairs, mounting_contact_policy=mounting_policy, scene_model=scene, records=records, excluded=excluded, snapshot_sha256=digest,
                  transferred_objects=transferred_objects,
                  moving_body_names=list(moving_body_names),
                  disabled_body_names=list(disabled_body_names),
                  active_group=active_group, active_group_root_body=root_name,
                  world_frame=robot_metadata["world_frame"], base_from_world=base_from_world.tolist(),
                  other_arm_frozen=True, frozen_groups=[x for x in robot_metadata["joint_groups"] if x != active_group],
                  simulation_time_s=float(data.time), qpos_snapshot=data.qpos.tolist(),
                  scope=("MuJoCo启用接触的非活动组几何；其他臂与剩余动态物体冻结；"
                         "活动组采用原cuRobo球与ignore；"
                         + (f"按本阶段旧 Skill 策略关闭 {disabled_body_names} 的场景碰撞。"
                            if disabled_body_names else "")
                         + (f"运动关节 {moving_body_names} 不参与静态规划，必须逐帧复核。"
                            if moving_body_names else "")
                         + (f"已显式转移附着物{sorted(transferred_objects)}到末端球代理。"
                            if transferred_objects else "无附着物转移。")),
                  limitations=["Mesh使用编译顶点凸包；圆柱/球/胶囊采用明确的保守多面体代理。",
                               "MuJoCo接触mask只决定是否导出，不复现逐几何位运算pair filter；导出障碍对活动组保守生效。",
                               "原MuJoCo已禁用接触的knob等geom被逐条记录排除，不能声称这些部件已有碰撞保护。",
                               "快照不是双臂协同规划；状态变化后必须重建，当前接口不提供自动同步。",
                               *( ["运动关节几何不在 cuRobo 静态障碍内；仅逐帧 MuJoCo 接触复核，未证明连续扫掠无碰撞。"]
                                  if moving_body_names else [] ),
                               "非活动组内部显式pair逐对验证并记录，非零margin按max(geom,pair)保守膨胀；gap、活动组pair或双零mask pair仍拒绝。"])
    (output_dir / "collision_snapshot.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
