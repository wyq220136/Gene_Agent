"""敲击面几何对齐：输入当前工具/TCP/目标pose，不读取仿真任务状态。"""
import numpy as np
from scipy.spatial.transform import Rotation
from gene_agent.knowledge.geometry.transforms import validate_pose


def align_hit_tcp(world_tool, world_tcp, world_target, *, strike_point, strike_normal,
                  target_point, clearance_m):
    """将工具作用面法向朝向目标局部-Z，沿目标+Z留净空；保留最小转角的平面内朝向。"""
    tool=validate_pose(world_tool);tcp=validate_pose(world_tcp);target=validate_pose(world_target)
    point=np.asarray(strike_point,dtype=float);normal=np.asarray(strike_normal,dtype=float)
    target_point=np.asarray(target_point,dtype=float)
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (point,normal,target_point)) or not np.isclose(np.linalg.norm(normal),1.):
        raise ValueError('敲击点必须有限三维，敲击法向必须单位向量')
    if not np.isfinite(clearance_m) or clearance_m<=0:raise ValueError('净空必须为正米制距离')
    a=tool[:3,:3]@normal;b=-target[:3,2]
    cross=np.cross(a,b);s=np.linalg.norm(cross);dot=np.clip(a@b,-1,1)
    if s>1e-8:rotation=Rotation.from_rotvec(cross/s*np.arctan2(s,dot)).as_matrix()
    elif dot>0:rotation=np.eye(3)
    else:
        axis=tool[:3,0] if abs(tool[:3,0]@a)<.9 else tool[:3,1]
        axis=np.cross(a,axis);axis/=np.linalg.norm(axis)
        rotation=Rotation.from_rotvec(axis*np.pi).as_matrix()
    goal_tool=np.eye(4);goal_tool[:3,:3]=rotation@tool[:3,:3]
    contact=target[:3,:3]@target_point+target[:3,3]
    goal_tool[:3,3]=contact+target[:3,2]*clearance_m-goal_tool[:3,:3]@point
    return goal_tool@np.linalg.inv(tool)@tcp


def align_hit_tcp_candidates(world_tool, world_tcp, world_target, *, yaw_offsets_rad=(0.,), **surface):
    """绕目标法向采样等价敲击朝向；保持作用点、作用面法向和净空不变。"""
    offsets=np.asarray(yaw_offsets_rad,float)
    if offsets.ndim!=1 or not len(offsets) or not np.isfinite(offsets).all():
        raise ValueError('敲击朝向候选必须为非空有限弧度列表')
    target=validate_pose(world_target)
    original=align_hit_tcp(world_tool,world_tcp,target,**surface)
    point=np.asarray(surface['target_point'],float)
    pivot=target[:3,:3]@point+target[:3,3]+target[:3,2]*surface['clearance_m']
    goals=[]
    for angle in offsets:
        rotation=Rotation.from_rotvec(target[:3,2]*angle).as_matrix()
        goal=original.copy();goal[:3,:3]=rotation@original[:3,:3]
        goal[:3,3]=pivot+rotation@(original[:3,3]-pivot)
        goals.append(goal)
    return tuple(goals)
