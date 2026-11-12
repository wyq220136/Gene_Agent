"""Pika 模型的夹持区域；纯 URDF 几何，不依赖物理引擎或设备。"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


# 与模型内侧 site 相同的选定区域，米；它与操作 TCP 和最远指尖不同。
INNER_SITE_LOCAL_M = {'left': [.085, -.04149731, 0.], 'right': [.085, .04149731, 0.]}


def _origin(node):
    result = np.eye(4)
    origin = node.find('origin')
    if origin is not None:
        result[:3,3] = np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        result[:3,:3] = Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return result


def pika_grasp_center_tcp(robot_metadata: dict, group: str) -> np.ndarray:
    """选定内侧夹持区，用实际安装 URDF 和 T_link7_TCP 表达；不是操作 TCP。

    内侧接触检测 site 向指尖侧 6mm 的区域，避免旧刷柄尾先碰指根。
    该选择已检查真实双侧接触；不代表夹持力/抗转能力已验证。
    """
    descriptor = robot_metadata['grippers'][group]
    if descriptor['mode'] != 'pika_urdf':
        raise ValueError('当前夹持区域只对应 Pika 原始手指模型')
    root = ET.parse(Path(robot_metadata['planning_urdf_path'])).getroot()
    mount = root.find("joint[@name='pika_mount']")
    if mount is None:
        raise ValueError('Pika 安装 URDF 缺少 pika_mount，不能猜测夹持中心')
    points = []
    for side, point in INNER_SITE_LOCAL_M.items():
        joint = root.find(f"joint[@name='pika_gripper_{side}_joint']")
        if joint is None:
            raise ValueError(f'Pika URDF 缺少 {side} 手指关节')
        point = np.asarray(point) + [.006, 0., 0.]
        points.append((_origin(joint) @ np.r_[point,1.])[:3])
    link_tcp = np.asarray(robot_metadata['transforms'][group]['link_from_tcp'],dtype=float)
    tcp_pika = np.linalg.inv(link_tcp) @ _origin(mount)
    return (tcp_pika @ np.r_[np.mean(points,axis=0),1.])[:3]


def pika_predict_contact_width(robot_metadata, group, T_tcp_object, mesh_path, *, open_width_m):
    """静态几何预览：对完整手指投射闭合方向射线，停在首次接触开口。

    用资产顶点及面中心进行确定性采样；物体固定、双指对称运动，不预测
    物体受力后的平移、旋转或稳定夹持。局部几何查询不调用物理引擎。
    """
    import trimesh
    descriptor = robot_metadata['grippers'][group]
    if descriptor['mode'] != 'pika_urdf':
        raise ValueError('完整手指开口预测只支持 Pika 模型')
    low, high = descriptor['width_range_m']
    if not np.isfinite(open_width_m) or not low <= open_width_m <= high:
        raise ValueError('预览初始开口超出夹爪范围')
    urdf_path = Path(robot_metadata['planning_urdf_path'])
    root = ET.parse(urdf_path).getroot()
    tcp_pika = np.linalg.inv(np.asarray(robot_metadata['transforms'][group]['link_from_tcp'])) @ _origin(root.find("joint[@name='pika_mount']"))
    mesh = mesh_path.copy() if isinstance(mesh_path, trimesh.Trimesh) else trimesh.load_mesh(mesh_path, process=False)
    mesh.apply_transform(np.asarray(T_tcp_object, dtype=float))
    points = np.concatenate([mesh.vertices, mesh.triangles_center])
    points = points[::max(1, len(points)//12000)]
    widths = {}
    for side, sign in [('left', 1), ('right', -1)]:
        joint = root.find(f"joint[@name='pika_gripper_{side}_joint']")
        transform = _origin(joint)
        axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
        transform[:3,3] += axis * (-sign*(high-open_width_m)/2)
        direction = sign * (tcp_pika[:3,:3] @ transform[:3,:3] @ axis)
        candidates = []
        link = root.find(f"link[@name='pika_gripper_{side}_link']")
        for collision in link.findall('collision'):
            node = collision.find('geometry/mesh')
            if node is None:
                raise ValueError('Pika 完整手指查询要求 mesh 碰撞模型')
            path = Path(node.get('filename'))
            if not path.is_absolute(): path = urdf_path.parent/path
            # STL 各三角面重复顶点；合并同点后才能按封闭表面判断内外。
            finger = trimesh.load_mesh(path, process=True)
            finger.apply_scale(np.fromstring(node.get('scale','1 1 1'),sep=' '))
            finger.apply_transform(tcp_pika @ transform @ _origin(collision))
            if not finger.is_watertight:
                raise ValueError('Pika 手指 mesh 不封闭，不能验证初始开口穿透')
            bounds = finger.bounds
            inside_bounds = np.all((points > bounds[0]) & (points < bounds[1]),axis=1)
            if np.any(finger.contains(points[inside_bounds])):
                raise ValueError(f'{side} 初始开口已与物体穿透，不能预测闭爪接触')
            hits, indices, _ = finger.ray.intersects_location(
                points, np.tile(direction,(len(points),1)), multiple_hits=False)
            distances = (hits-points[indices]) @ direction
            distances = distances[(distances >= 0.) & (distances <= open_width_m/2)]
            if len(distances): candidates.append(float(distances.min()))
        if not candidates:
            raise ValueError(f'{side} 完整手指闭合范围内未发现物体，不能假定夹住')
        widths[side] = float(open_width_m - 2*min(candidates))
    return {'predicted_width_m':max(widths.values()), 'per_finger_contact_width_m':widths,
            'source':'full_finger_mesh_first_contact', 'physical_hold_verified':False,
            'assumption':'fixed_object_symmetric_closure_surface_sampling'}
