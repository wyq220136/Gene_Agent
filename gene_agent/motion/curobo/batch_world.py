"""单环境cuboid/mesh固定缓冲区批量更新，保留地址与完整障碍集合。"""


def load_batch(self, scene_cfg, env_idx=0, store_reference=True):
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
    self.clear(env_idx)
    if self.cuboids is not None:
        self.cuboids.load_batch(scene_cfg.cuboid or [], env_idx)
    if self.meshes is not None:
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
        unique[id(world)] = (world, names)
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


def update_solver_world(solver, scene_cfg):
    world = getattr(solver, "scene_collision_checker", None)
    if world is None:
        world = solver.scene_model
    load_batch(world.data, scene_cfg)
    world.scene_model = scene_cfg
    graph = getattr(solver, "graph_planner", None)
    if graph is not None:
        graph.reset_buffer()
