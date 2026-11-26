"""单环境cuboid/mesh固定缓冲区批量更新，保留地址与完整障碍集合。"""


def load_batch(self, scene_cfg, env_idx=0, store_reference=True, *, changed_kinds=None):
    # 明确限定已验证的单环境存储，不修改第三方类。
    if self.num_envs != 1:
        raise ValueError("批量更新仅支持单环境")
    if scene_cfg.voxel:
        raise ValueError("批量更新不支持voxel")
    names = [x.name for x in (scene_cfg.cuboid or []) + (scene_cfg.mesh or [])]
    if len(names) != len(set(names)):
        raise ValueError("障碍名称必须全局唯一")
    for name, objects in [
        ("cuboids", scene_cfg.cuboid or []),
        ("meshes", scene_cfg.mesh or []),
    ]:
        storage = getattr(self, name)
        if objects and (storage is None or len(objects) > storage.max_n):
            raise ValueError(f"{name}缓存容量不足")
        if name == "cuboids" and len({x.name for x in objects}) != len(objects):
            raise ValueError("重复cuboid名称")
    if not 0 <= env_idx < self.num_envs:
        raise ValueError("env_idx越界")
    if store_reference:
        self.scene_model = scene_cfg
    if changed_kinds is None:
        self.clear(env_idx)
    if self.cuboids is not None and (changed_kinds is None or "cuboid" in changed_kinds):
        self.cuboids.load_batch(scene_cfg.cuboid or [], env_idx)
    if self.meshes is not None and (changed_kinds is None or "mesh" in changed_kinds):
        self.meshes.load_batch(scene_cfg.mesh or [], env_idx)


def apply_batch(worlds, updates):
    unique = {}
    for world, expected in worlds:
        ordered_names = world.get_obstacle_names()
        if len(ordered_names) != len(set(ordered_names)):
            raise ValueError("障碍名称必须全局唯一")
        names = set(ordered_names)
        if set(expected) - names:
            raise ValueError("碰撞world缺少预期障碍")
        if world.num_envs != 1 or world.data.voxels is not None:
            raise ValueError("批量位姿仅支持单环境cuboid/mesh")
        unique[id(world.data)] = (world, names)
    if not unique or not updates:
        return
    import torch
    from curobo.types import Pose

    ordered = list(updates)
    lookup = {name: i for i, name in enumerate(ordered)}
    device = next(iter(unique.values()))[0].device_cfg
    for world, _ in unique.values():
        if (
            world.device_cfg.device != device.device
            or world.device_cfg.dtype != device.dtype
        ):
            raise ValueError("世界device/dtype不一致")
    inverse = (
        Pose.from_batch_list([updates[name] for name in ordered], device)
        .inverse()
        .get_pose_vector()
    )
    for world, names in unique.values():
        for storage in [world.data.cuboids, world.data.meshes]:
            if storage is None:
                continue
            mapping = {
                name: i for i, name in enumerate(storage.names[0]) if name is not None
            }
            selected = [name for name in ordered if name in mapping]
            if not selected:
                continue
            indices = torch.tensor(
                [mapping[n] for n in selected], device=device.device, dtype=torch.long
            )
            source = torch.tensor(
                [lookup[n] for n in selected], device=device.device, dtype=torch.long
            )
            storage.inv_pose[0, indices, :7] = inverse[source]



def update_solver_worlds(solvers, scene_cfg, *, changed_kinds=None):
    """共享 data 的 world 只加载一次；所有 wrapper 和独立 graph 都同步。"""
    data_ids, graph_ids = set(), set()
    count = 0
    for solver in solvers:
        world = getattr(solver, "scene_collision_checker", None)
        if world is None:
            world = solver.scene_model
        if id(world.data) not in data_ids:
            if changed_kinds is None:
                load_batch(world.data, scene_cfg)
            elif changed_kinds:
                load_batch(world.data, scene_cfg, changed_kinds=changed_kinds)
            else:
                world.data.scene_model = scene_cfg
            data_ids.add(id(world.data))
        world.scene_model = scene_cfg
        count += 1
        graph = getattr(solver, "graph_planner", None)
        if graph is not None and id(graph) not in graph_ids:
            graph.reset_buffer()
            graph_ids.add(id(graph))
    from .optimization import current_optimization
    owner = current_optimization()
    if owner is not None:
        owner.world_sync_requests += count
        owner.world_data_syncs += len(data_ids) if changed_kinds is None or changed_kinds else 0
    return dict(solver_count=count, unique_world_data=len(data_ids), graphs_reset=len(graph_ids))


def update_solver_world(solver, scene_cfg):
    return update_solver_worlds((solver,), scene_cfg)
