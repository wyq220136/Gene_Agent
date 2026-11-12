"""固定相机首帧的输入检查、自动分割与注册；不包含任何运动控制。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.perception_modules.local_service import environment, model_service, ROOT
from .planning import write_json

ROLES = ('brush', 'cube', 'dustpan')
DEFAULT_PROMPTS = {'brush': 'a cleaning brush', 'cube': 'a green cube', 'dustpan': 'a dustpan'}


def registration_quality(mesh_path, pose, mask, depth, intrinsic):
    """可见深度到估计表面的单向残差；诊断用途，不是 GT 位姿精度。"""
    import trimesh
    from scipy.spatial import cKDTree
    mesh = trimesh.load(mesh_path, force='mesh')
    valid = mask & (depth > .08) & (depth < 3)
    v, u = np.where(valid)
    z = depth[valid]
    k = np.asarray(intrinsic)
    points = np.column_stack(((u-k[0,2])*z/k[0,0], (v-k[1,2])*z/k[1,1], z))
    surface = trimesh.sample.sample_surface(mesh, 30000, seed=0)[0]
    surface = surface @ pose[:3,:3].T + pose[:3,3]
    distances = cKDTree(surface).query(points)[0]
    median, p90 = np.quantile(distances, [.5,.9])
    return {'median_depth_to_surface_m': float(median), 'p90_depth_to_surface_m': float(p90),
            'depth_points': len(points), 'mesh_extent_m': mesh.extents.tolist(),
            'warning': bool(median > .015 or p90 > .04),
            'warning_thresholds_m': {'median': .015, 'p90': .04},
            'interpretation': 'one_way_visible_depth_residual_not_ground_truth_pose_error'}


def load_calibration(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding='utf-8'))
    k = np.asarray(raw['intrinsic_matrix'], float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('标定内参无效')
    serial = str(raw.get('serial', '')).strip()
    if not serial:
        raise ValueError('标定文件缺少相机序列号 serial')
    return {**raw, 'serial': serial, 'intrinsic_matrix': k.tolist(),
            'T_sdk_camera': validate_pose(raw['extrinsic_matrix']).tolist(),
            'calibration_path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def check_capture(calibration, camera, shape):
    if str(camera.get('serial')) != calibration['serial']:
        raise ValueError('采集相机与标定文件的序列号不同')
    if tuple(shape) != (calibration['height'], calibration['width']):
        raise ValueError('图像尺寸与标定不同；请提供对应分辨率标定，不自动缩放外参/内参')
    if not np.allclose(camera['intrinsic_matrix'], calibration['intrinsic_matrix'], rtol=.002, atol=.5):
        raise ValueError('采集流内参与标定不一致')


def capture(calibration, output, *, reuse_capture=None, endpoint=None):
    """默认直接启动单帧采集子进程，退出后相机释放，不要求常驻 server。"""
    import shutil
    output = Path(output)
    if reuse_capture is not None:
        output.mkdir(parents=True)
        for name in ('rgb.png', 'depth.png', 'camera.json'):
            shutil.copy2(Path(reuse_capture)/name, output/name)
    elif endpoint:
        from gene_agent.realworld.pose_tracking import capture_frozen_scene_frame
        capture_frozen_scene_frame(output_dir=output, endpoint=endpoint, expected_serial=calibration['serial'])
    else:
        python, env = environment('robotwin_full')
        command = [str(python), '-m', 'tools.run.capture_realsense_frame',
                   '--serial', calibration['serial'], '--output-dir', str(output),
                   '--width', str(calibration['width']), '--height', str(calibration['height'])]
        subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=180)
    camera = json.loads((output/'camera.json').read_text(encoding='utf-8'))
    rgb = np.asarray(Image.open(output/'rgb.png').convert('RGB'))
    raw = np.asarray(Image.open(output/'depth.png'))
    check_capture(calibration, camera, raw.shape)
    if rgb.shape[:2] != raw.shape or camera.get('depth_unit', 'millimeter') != 'millimeter':
        raise ValueError('RGB 与毫米深度不匹配')
    depth = raw.astype(float)/1000
    return camera, rgb, depth


def estimate_table(depth, excluded, calibration, size_xyz, *, top_z=None):
    """world Z 水平面强约束，排除物体 mask 后选择背景中支持最多的 8mm 高度带。"""
    k = np.asarray(calibration['intrinsic_matrix'])
    v, u = np.indices(depth.shape)
    valid = (depth > .08) & (depth < 3) & ~excluded
    z = depth[valid][::4]
    if len(z) < 300:
        raise ValueError('桌面有效背景深度不足')
    xyz = np.column_stack(((u[valid][::4]-k[0, 2])*z/k[0, 0],
                           (v[valid][::4]-k[1, 2])*z/k[1, 1], z))
    t = np.asarray(calibration['T_sdk_camera'])
    world = xyz @ t[:3, :3].T + t[:3, 3]
    if top_z is None:
        edges = np.arange(world[:, 2].min()-.004, world[:, 2].max()+.012, .004)
        counts, edges = np.histogram(world[:, 2], edges)
        i = int(np.argmax(counts)); height = (edges[i]+edges[i+1])/2
    else:
        height = float(top_z)
        if not np.isfinite(height):
            raise ValueError('桌面高度必须有限')
    inliers = np.abs(world[:, 2]-height) < .008
    ratio = float(inliers.mean())
    if inliers.sum() < 300 or ratio < .25:
        raise ValueError(f'无法确认水平桌面（支持比例 {ratio:.3f}）；核对外参或 --table-top-z')
    if top_z is None:
        height = float(np.median(world[inliers, 2]))
    spread = np.ptp(world[inliers, :2], axis=0)
    if np.min(spread) < .12:
        raise ValueError('桌面支持点横向覆盖不足')
    pose = np.eye(4)
    pose[:2, 3] = np.median(world[inliers, :2], axis=0)
    pose[2, 3] = height-float(size_xyz[2])/2
    return pose, {'method': 'masked_background_horizontal_height_histogram',
                  'top_z_m': height, 'inlier_ratio': ratio, 'support_points': int(inliers.sum()),
                  'xy_spread_m': spread.tolist(), 'size_xyz_m': list(size_xyz),
                  'size_source': 'site_configuration_proxy_not_measured_table_boundary',
                  'height_source': 'explicit_with_depth_support' if top_z is not None else 'depth'}


def perceive(calibration, capture_dir, meshes, output, prompts):
    """所有物体注册用同一帧；模型服务依次退出，最后才交给 cuRobo。"""
    from gene_agent.perception_modules.segmentation.seg_client import segment_image
    from gene_agent.perception_modules.pose_est.pose_client import PoseClient
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    capture_dir = Path(capture_dir)
    before = {name: hashlib.sha256((capture_dir/name).read_bytes()).hexdigest()
              for name in ('rgb.png', 'depth.png', 'camera.json')}
    depth_raw = np.asarray(Image.open(capture_dir/'depth.png'))
    rgb = np.asarray(Image.open(capture_dir/'rgb.png').convert('RGB'))
    masks = {}
    timings = {}
    start = time.monotonic()
    with model_service('seg', output) as server:
        for role in ROLES:
            print(f'SAM3 首帧分割：{role} / {prompts[role]}', flush=True)
            mask_path = output/f'{role}_mask.png'
            status, response = segment_image(server, capture_dir/'rgb.png',
                [{'category': role, 'prompt': prompts[role], 'mask_path': str(mask_path)}], timeout=300)
            write_json(output/f'{role}_seg_response.json', response)
            if status != 200 or not response.get('ok') or not response.get('success', True) or not mask_path.is_file():
                raise RuntimeError(f'{role} 分割失败；查看 {output}')
            mask = np.asarray(Image.open(mask_path).convert('L')) > 0
            if mask.shape != depth_raw.shape or mask.sum() < 25 or mask.mean() > .6:
                raise ValueError(f'{role} mask 空、尺寸错误或过大；可通过 --{role}-prompt 换提示词')
            if np.count_nonzero(mask & (depth_raw > 0)) < max(25, .3*mask.sum()):
                raise ValueError(f'{role} mask 内有效深度不足')
            for other, previous in masks.items():
                if np.count_nonzero(mask & previous)/min(mask.sum(), previous.sum()) > .2:
                    raise ValueError(f'{role} 与 {other} 的分割重叠过多，拒绝错误场景；请修改 prompt')
            masks[role] = mask
            overlay = rgb.copy();overlay[mask] = (.45*overlay[mask]+.55*np.array([255, 50, 50])).astype(np.uint8)
            Image.fromarray(overlay).save(output/f'{role}_overlay.png')
    timings['segmentation_s'] = time.monotonic()-start
    poses = {}
    quality = {}
    start = time.monotonic()
    with model_service('pose', output) as server:
        client = PoseClient(server, timeout=600)
        for role in ROLES:
            print(f'FoundationPose 首帧注册：{role}', flush=True)
            session = f'first_frame_{role}'
            try:
                response = client.register(session_id=session, rgb_path=capture_dir/'rgb.png',
                    depth_path=capture_dir/'depth.png', mask_path=output/f'{role}_mask.png',
                    mesh_path=meshes[role], camera_path=capture_dir/'camera.json')
                write_json(output/f'{role}_pose_response.json', response)
                if response.get('session_id') != session:
                    raise ValueError('FoundationPose 返回 session 不匹配')
                pose = validate_pose(response['T_camera_object'])
                if pose[2, 3] <= 0:
                    raise ValueError(f'{role} 位姿位于相机后方')
                camera = json.loads((capture_dir/'camera.json').read_text(encoding='utf-8'))
                quality[role] = registration_quality(meshes[role], pose, masks[role],
                    depth_raw.astype(float)/1000, camera['intrinsic_matrix'])
                write_json(output/'registration_quality.json', quality)
                if quality[role]['warning']:
                    print(f'注意：{role} 深度与 mesh 残差偏大，需在三维预览核对；'
                          f'中位数={quality[role]["median_depth_to_surface_m"]*1000:.1f} mm', flush=True)
                poses[role] = np.asarray(calibration['T_sdk_camera']) @ pose
            finally:
                # 注册失败时也交由 context manager 彻底清理自己拥有的进程。
                try:
                    client.stop(session)
                except RuntimeError:
                    pass
    timings['pose_estimation_s'] = time.monotonic()-start
    after = {name: hashlib.sha256((capture_dir/name).read_bytes()).hexdigest() for name in before}
    if after != before:
        raise ValueError('感知期间冻结帧文件被修改')
    write_json(output/'manifest.json', {'capture_sha256': before, 'calibration_sha256': calibration['sha256'],
        'prompts': prompts, 'timing': timings, 'T_sdk_object': {k: v.tolist() for k, v in poses.items()},
        'registration_quality': quality,
        'mesh_sha256': {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in meshes.items()},
        'method': 'SAM3 masks + FoundationPose register, no tracking, no GT poses'})
    return poses, np.logical_or.reduce(list(masks.values()))
