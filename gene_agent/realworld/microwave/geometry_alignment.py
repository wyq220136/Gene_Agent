"""真机输入的纯几何消歧；不使用仿真名义位姿，不连接设备。"""
import numpy as np


class CameraSDKAxisError(ValueError):
    code = "camera_sdk_axis_inconsistent"


def fitted_table_pose(points, camera_position, center_xy, size_xy, thickness_m):
    """保留 Afford 固定 XY 范围，拟合真实平面和法向，厚度沿法向延伸。"""
    points = np.asarray(points, float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 10:
        raise ValueError("固定桌面范围内不足 10 个点，不能估计平面方向")
    sample = points[np.linspace(0, len(points)-1, min(len(points), 7000), dtype=int)]
    rng = np.random.default_rng(5)
    best = None
    for _ in range(300):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        # 与现有近水平桌面假设相同，仅放开实际倾角；不把竖直背景当桌面。
        if abs(normal[2]) < .85:
            continue
        mask = np.abs((sample-a) @ normal) < .005
        if best is None or mask.sum() > best.sum():
            best = mask
    if best is None or best.sum() < 10 or best.mean() < .20:
        raise ValueError("固定 XY 范围内没有足够支持的桌面平面")
    selected = sample[best]
    center = selected.mean(0)
    _, _, vt = np.linalg.svd(selected-center, full_matrices=False)
    normal = vt[-1]
    if normal @ (np.asarray(camera_position)-center) < 0:
        normal = -normal
    top = np.array([*np.asarray(center_xy, float), 0.])
    top[2] = center[2] - (normal[:2] @ (top[:2]-center[:2])) / normal[2]
    x = np.array([1., 0., 0.]); x -= normal * x.dot(normal); x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((x, np.cross(normal, x), normal))
    pose[:3, 3] = top-normal*float(thickness_m)/2
    return dict(table_pose_world=pose.tolist(), table_size_xy_m=np.asarray(size_xy).tolist(),
        accepted_table_thickness_m=float(thickness_m), table_top_z_m=float(top[2]),
        top_center_world_m=top.tolist(), normal_world=normal.tolist(),
        source="registered_rgbd_fixed_table_geometry", frame="right_sdk_base",
        method="affordtoolact_fixed_xy_with_observed_3d_plane", footprint_is_visible_extent=False,
        sdk_up_consistent=bool(normal[2] > 0),
        inlier_count=int((np.abs((points-center)@normal)<.005).sum()), input_point_count=len(points), sample_point_count=len(sample),
        rms_m=float(np.sqrt(np.mean(((selected-center)@normal)**2))),
        distance_threshold_m=.005, assumption="固定 XY；可见桌面上表面朝相机；倾角由原始 RGB-D 测量")


def handle_halfturn(mesh_vertices, mesh_center):
    """验证实际 mesh 存在等价半转；不能对任意非对称资产强加 180°。"""
    from scipy.spatial import cKDTree
    vertices = np.asarray(mesh_vertices, float)
    center = np.asarray(mesh_center, float)
    turn = np.eye(4); turn[:3, :3] = np.diag([-1., -1., 1.])
    turn[:3, 3] = center-turn[:3, :3]@center
    flipped = vertices@turn[:3, :3].T+turn[:3, 3]
    residual = float(cKDTree(vertices).query(flipped)[0].max())
    if residual > 1e-6:
        raise ValueError("注册把手 mesh 不满足局部 Z 半转对称，不能套用本消歧规则")
    return turn, residual


def canonical_handle_pose(raw_pose, config):
    """只在等价把手分支中选择与观测桌面上方一致的固定柜体姿态。"""
    raw = np.asarray(raw_pose, float)
    turn = np.asarray(config["T_mesh_equivalent_halfturn"], float)
    alignment = np.asarray(config["T_mesh_asset"], float)
    up = np.array(config["observed_up_sdk"], dtype=float, copy=True)
    up /= np.linalg.norm(up)
    candidates = (raw, raw@turn)
    dots = [float((candidate@alignment)[:3, 2]@up) for candidate in candidates]
    choice = int(np.argmax(dots))
    if dots[choice] < .85 or abs(dots[0]-dots[1]) < .5:
        raise ValueError("把手等价姿态不能确定直立柜体；需核对完整柜体/桌面观测")
    return candidates[choice], dict(selected_equivalent_branch=choice, cabinet_up_dot=dots,
        raw_pose_sdk=raw.tolist(), canonical_pose_sdk=candidates[choice].tolist(),
        basis="observed_table_up; upright stationary cabinet; equivalent handle geometry",
        pose_estimator_output_modified=False)


def table_top_z_at_xy(table_pose, thickness_m, xy):
    """求候选 XY 处的实际平面高度，不能用桌中心高度代替倾斜桌面。"""
    table = np.asarray(table_pose, float)
    normal = table[:3, 2]
    if normal[2] <= 0:
        raise CameraSDKAxisError("camera_sdk_axis_inconsistent: 桌面上法向与 SDK +Z 反向，不能生成真机目标")
    top = table[:3, 3] + normal*float(thickness_m)/2
    return float(top[2] - normal[:2] @ (np.asarray(xy)-top[:2])/normal[2])
