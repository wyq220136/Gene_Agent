"""有界的桌面支撑投影；保留原始观测，不作物理执行或碰撞豁免。"""
from __future__ import annotations

import mujoco
import numpy as np
from gene_agent.robots.pika_frames import pose_from_mujoco
from .rigid_geometry import collision_geoms, collision_geom_bounds


def correct_tabletop_support(model, data, body_names, table_geom_name, *,
                             max_adjustment_m=.03, clearance_m=.001):
    """只向上平移穿桌物体，保持 XY/旋转；所有对象通过预算才应用。"""
    if not np.isfinite(max_adjustment_m) or not 0 < max_adjustment_m <= .1:
        raise ValueError('支撑校正上限必须在 (0, 0.1] m')
    if not np.isfinite(clearance_m) or not 0 <= clearance_m <= .003:
        raise ValueError('支撑净空必须在 [0, 0.003] m')
    table = model.geom(table_geom_name)
    rotation = data.geom_xmat[table.id].reshape(3,3)
    if int(table.type[0]) != mujoco.mjtGeom.mjGEOM_BOX or not np.allclose(rotation[:,2], [0,0,1], atol=1e-6):
        raise ValueError('支撑校正仅支持水平 box 桌面')
    top = float(data.geom_xpos[table.id,2] + table.size[2])
    report = {'method':'bounded_upward_world_z_projection', 'planning_assumption':True,
              'max_adjustment_m':max_adjustment_m, 'clearance_m':clearance_m,
              'table_top_z_m':top, 'objects':{}, 'physics_stepped':False}
    for name in body_names:
        geom_ids = list(collision_geoms(model, name))
        if not geom_ids:
            raise ValueError(f'{name} 缺少碰撞几何')
        # body0 为 world：直接取每个编译几何的世界界限，避免旋转局部AABB膨胀。
        low = min(float(collision_geom_bounds(model, data, g, 0)[0,2]) for g in geom_ids)
        shift = max(0., top + clearance_m-low) if low < top else 0.
        if shift > max_adjustment_m:
            raise ValueError(f'{name} 所需桌面支撑校正 {shift*1000:.1f}mm 超出上限 {max_adjustment_m*1000:.1f}mm')
        before = pose_from_mujoco(data.body(name))
        after = before.copy();after[2,3] += shift
        report['objects'][name] = {'shift_world_z_m':shift, 'collision_low_z_before_m':low,
                                  'T_world_object_observed':before.tolist(),
                                  'T_world_object_planning':after.tolist()}
        body_id = model.body(name).id
        if model.body_parentid[body_id] != 0:
            raise ValueError('当前支撑校正仅支持 world 直接子物体')
        joint = int(model.body_jntadr[body_id])
        if joint >= 0 and model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('支撑校正不修改关节约束物体')
    for name, row in report['objects'].items():
        shift = row['shift_world_z_m']
        if not shift:
            continue
        body_id = model.body(name).id
        joint = int(model.body_jntadr[body_id])
        model.body_pos[body_id,2] += shift
        if joint >= 0:
            data.qpos[int(model.jnt_qposadr[joint])+2] += shift
    mujoco.mj_forward(model, data)
    return report
