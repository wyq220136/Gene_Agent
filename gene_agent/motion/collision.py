"""安装接触例外的局部复核：完整外物保护、完整原始自碰撞对不变。"""
from __future__ import annotations
import copy
from contextlib import contextmanager


def split_scene_for_mounting_contacts(scene, policy):
    """只按显式障碍名分组，缺失名称报错；不按模糊前缀猜测本体。"""
    names = set(policy.get('obstacle_names', []))
    actual = {name for obstacles in scene.values() for name in obstacles}
    if names-actual:
        raise ValueError(f"安装例外引用未知障碍: {sorted(names-actual)}")
    body, external = {}, {}
    for kind, obstacles in scene.items():
        body[kind] = {name:copy.deepcopy(value) for name,value in obstacles.items() if name in names}
        external[kind] = {name:copy.deepcopy(value) for name,value in obstacles.items() if name not in names}
    return body,external


@contextmanager
def optimizer_mounting_mask(kinematics, policy):
    """恢复旧优化期近基座link屏蔽，退出严格恢复半径。

    cuRobo当前API不提供link×obstacle过滤，所以优化阶段屏蔽比最终允许策略更宽。
    该结果不能直接放行：调用方必须用本文件的完整复核检查所有外物和自碰撞。
    """
    params=kinematics.config.kinematics_config
    saved=params.link_spheres.clone()
    try:
        for link in policy.get('robot_links',[]):
            params.disable_link_spheres(link)
        yield
    finally:
        params.link_spheres.copy_(saved)


class MountingCollisionValidator:
    """对成功IK/离散轨迹做局部允许对之外的完整球碰撞复核。"""
    def __init__(self, kinematics, scene, policy):
        import torch
        from curobo._src.geom.collision.collision_scene import SceneCollision,SceneCollisionCfg
        from curobo._src.geom.types import SceneCfg
        from curobo.types import DeviceCfg
        self.torch=torch; self.kinematics=kinematics; self.policy=copy.deepcopy(policy); self.device=DeviceCfg()
        body,external=split_scene_for_mounting_contacts(scene,policy)
        self.checkers={}
        for label,world in [('body',body),('external',external)]:
            if any(world.values()):
                self.checkers[label]=SceneCollision.from_config(SceneCollisionCfg(device_cfg=self.device,scene_model=SceneCfg.create(world)))
        params=kinematics.config.kinematics_config
        ids=[params.link_name_to_idx_map[link] for link in policy.get('robot_links',[])]
        self.mounting_mask=torch.zeros_like(params.link_sphere_idx_map,dtype=torch.bool)
        for index in ids:self.mounting_mask|=params.link_sphere_idx_map==index

    def check(self, positions, planner_joint_names):
        """positions=[N,dof]；不更改原球。轨迹只覆盖调用方给出的离散样本。"""
        from curobo.types import JointState
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer
        torch=self.torch
        tensor=torch.as_tensor(positions,device='cuda',dtype=torch.float32).reshape(-1,len(planner_joint_names))
        spheres=self.kinematics.compute_kinematics(JointState.from_position(tensor,joint_names=planner_joint_names)).robot_spheres
        query=spheres.reshape(1,tensor.shape[0],-1,4)
        costs={}
        for label,checker in self.checkers.items():
            values=query.clone()
            if label=='body':values[:,:,self.mounting_mask,3]=-100.
            buffer=CollisionBuffer.from_shape(values.shape,self.device)
            cost=checker.get_sphere_distance_raw(values,buffer,torch.tensor([1.],device='cuda'),torch.tensor([0.],device='cuda'))
            costs[label]=float(cost.max().item())
        # 使用后端实际生成的pair列表及padding，保留所有原self_collision_ignore；
        # 不因为优化时屏蔽base/link1就跳过它们与其他link的有效自碰撞对。
        config=self.kinematics.get_self_collision_config()
        pairs=config.collision_pairs.to(dtype=torch.long)
        maximum=0.
        if len(pairs):
            values=spheres.reshape(tensor.shape[0],-1,4)
            a,b=values[:,pairs[:,0]],values[:,pairs[:,1]]
            padding=config.sphere_padding
            overlap=a[...,3]+b[...,3]+padding[pairs[:,0]]+padding[pairs[:,1]]-torch.linalg.vector_norm(a[...,:3]-b[...,:3],dim=-1)
            enabled=(a[...,3]>0)&(b[...,3]>0)
            maximum=float(torch.where(enabled,overlap,torch.zeros_like(overlap)).max().clamp(min=0).item())
        costs['self']=maximum
        return {'valid': all(value<=1e-7 for value in costs.values()),'max_collision_costs':costs,
                'sample_count':int(tensor.shape[0]),'mounting_contact_policy':self.policy,
                'trajectory_continuous_collision_checked':False}
