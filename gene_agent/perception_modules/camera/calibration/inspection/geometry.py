"""基座XYZ裁剪、相机身份绑定和水平主平面诊断；统计始终使用完整像素。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from gene_agent.knowledge.geometry.transforms import validate_pose
from ..storage import resolve_calibration_path


@dataclass
class InspectionFrame:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    points: np.ndarray
    valid: np.ndarray
    calibration: dict
    calibration_path: Path
    calibration_sha256: str
    source: str
    max_depth: float


def validate_crop(crop):
    lo, hi = np.asarray(crop['min'], float), np.asarray(crop['max'], float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
        raise ValueError('裁剪范围必须是有限三维XYZ，且每轴min < max')
    return lo, hi


def crop_mask(points, valid, crop):
    """与PIKA实际guide一致：基座坐标闭区间裁剪，不改变像素对应关系。"""
    if points.shape != valid.shape + (3,):
        raise ValueError('点云与有效像素掩码尺寸不匹配')
    lo, hi = validate_crop(crop)
    return valid & np.isfinite(points).all(axis=-1) & ((points >= lo) & (points <= hi)).all(axis=-1)


def backproject_base(depth, K, transform, *, max_depth=1.2):
    depth, K = np.asarray(depth), np.asarray(K, float)
    transform = validate_pose(np.asarray(transform, float))
    if depth.ndim != 2 or not np.isfinite(max_depth) or max_depth <= 0:
        raise ValueError('需要二维米制深度和有限正数深度上限')
    if (K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError('内参K无效')
    v, u = np.indices(depth.shape)
    rays = np.stack([(u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones(depth.shape)], axis=-1)
    valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth)
    safe_depth = np.where(valid, depth, 0)
    points = (rays * safe_depth[..., None]) @ transform[:3, :3].T + transform[:3, 3]
    points[~valid] = np.nan
    return points.astype(np.float32), valid


def make_frame(rgb, depth, K, sample_serial, parameters_path, *, source, max_depth=1.2):
    path = Path(parameters_path).resolve()
    raw = path.read_bytes()
    parameters = json.loads(raw)
    coefficients = np.asarray(parameters.get('D', parameters.get('distortion_coefficients', [])), float)
    if not np.isfinite(coefficients).all() or np.any(coefficients != 0):
        raise ValueError('检查需要已去畸变的RGB-D；不能忽略显式提供的非零畸变参数')
    if parameters.get('quality_passed') is False:
        raise ValueError('指定标定未通过质量检查')
    serial = parameters.get('serial', parameters.get('camera_serial'))
    if not serial or sample_serial != serial:
        raise ValueError(f'相机serial不匹配：标定{serial}，图像{sample_serial}')
    rgb, depth, K = np.asarray(rgb), np.asarray(depth), np.asarray(K, float)
    if rgb.shape != depth.shape + (3,) or rgb.dtype != np.uint8:
        raise ValueError('RGB/深度尺寸或RGB类型不一致')
    if parameters.get('width') != rgb.shape[1] or parameters.get('height') != rgb.shape[0]:
        raise ValueError('原图与标定分辨率不一致')
    if not np.allclose(K, parameters['intrinsic_matrix'], atol=1e-6, rtol=0):
        raise ValueError('实际图像内参与标定不一致')
    transform = parameters.get('extrinsic_matrix', parameters.get('T_sdk_third'))
    if 'extrinsic_matrix' in parameters and 'T_sdk_third' in parameters:
        if not np.allclose(parameters['extrinsic_matrix'], parameters['T_sdk_third'], atol=1e-8, rtol=0):
            raise ValueError('标定中的新旧外参字段冲突')
    points, valid = backproject_base(depth, K, transform, max_depth=max_depth)
    if not valid.any():
        raise ValueError('深度范围内没有有效点；请检查深度或增大--max-depth')
    parameters['serial'] = serial
    return InspectionFrame(rgb.copy(), depth.copy(), K, points, valid, parameters, path,
                           hashlib.sha256(raw).hexdigest(), str(source), max_depth)


def load_frame(data_dir, *, parameters_path=None, sample=1, max_depth=1.2):
    directory = resolve_calibration_path(data_dir).resolve()
    if not (directory/'session.json').is_file() and (directory/'inspection_report.json').is_file():
        report = json.loads((directory/'inspection_report.json').read_text())
        parameters_path = parameters_path or resolve_calibration_path(report['camera_parameters'])
        if 'saved_capture' in report:
            source = directory/report['saved_capture']
        else:
            source = resolve_calibration_path(report['source'])
        with np.load(source) as raw:
            return make_frame(raw['rgb'], raw['depth_m'], raw['K'], report['camera_serial'], parameters_path,
                              source=source, max_depth=max_depth)
    parameters_path = resolve_calibration_path(parameters_path) if parameters_path else directory/'third_view_extrinsic.json'
    session = json.loads((directory/'session.json').read_text())
    samples = session.get('samples')
    if samples is None:
        samples = [session['stage_a'], *session.get('validation_samples', [])]
    if not 1 <= sample <= len(samples):
        raise ValueError(f'板位编号须在1到{len(samples)}之间')
    item = samples[sample-1].get('third', samples[sample-1])
    with np.load(directory/item['rgbd_file']) as raw:
        if not np.allclose(raw['K'], item['K'], atol=1e-6, rtol=0):
            raise ValueError('原始图像与会话K不一致')
        return make_frame(raw['rgb'], raw['depth_m'], raw['K'], item['serial'], parameters_path,
                          source=directory/item['rgbd_file'], max_depth=max_depth)


def overlay_rgb(frame, kept):
    overlay = frame.rgb.copy()
    overlay[frame.valid & ~kept] = (overlay[frame.valid & ~kept] * .15).astype(np.uint8)
    overlay[~frame.valid] = [90, 40, 90]
    return overlay


def plane_diagnostics(points, *, table_z=None, threshold_m=.005):
    """找接近基座水平的主平面候选，不输出绝对标定通过标志。"""
    if table_z is not None and not np.isfinite(table_z):
        raise ValueError('实测桌高必须是有限米数')
    points = np.asarray(points, float)
    points = points[np.isfinite(points).all(axis=1)]
    unavailable = {'available': False, 'reason': '裁剪区有效平面点不足或没有接近水平的主平面',
                   'fit_distance_threshold_m': threshold_m, 'candidate_max_tilt_deg': 30}
    if len(points) < 100:
        return unavailable
    rng = np.random.default_rng(0)
    probe = points[rng.choice(len(points), min(8000, len(points)), replace=False)]
    best, best_count = None, 0
    for _ in range(100):
        a, b, c = probe[rng.choice(len(probe), 3, replace=False)]
        n = np.cross(b-a, c-a)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            continue
        n /= norm
        if abs(n[2]) < np.cos(np.deg2rad(30)):
            continue
        d = -n@a
        mask = np.abs(probe@n+d) <= threshold_m
        count = int(mask.sum())
        if count > best_count:
            best, best_count = mask, count
    if best is None or best_count < max(100, len(probe)*.15):
        return unavailable
    selected = probe[best]
    center = selected.mean(0)
    _, _, vt = np.linalg.svd(selected-center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    if normal[2] < np.cos(np.deg2rad(30)):
        return unavailable
    d = -normal@center
    distance = np.abs(points@normal+d)
    inliers = distance <= threshold_m
    height = float(np.median(points[inliers, 2]))
    return {'available': True, 'normal_sdk': normal.tolist(), 'offset_m': float(d),
            'fit_distance_threshold_m': threshold_m, 'candidate_max_tilt_deg': 30, 'seed': 0,
            'fit_sample_count': len(probe),
            'tilt_deg': float(np.rad2deg(np.arccos(np.clip(normal[2], -1, 1)))),
            'median_height_m': height, 'rmse_mm': float(np.sqrt(np.mean(distance[inliers]**2))*1000),
            'inlier_count': int(inliers.sum()), 'inlier_fraction': float(inliers.mean()),
            'height_range_p05_p95_m': np.quantile(points[inliers, 2], [.05, .95]).tolist(),
            'expected_table_z_m': table_z,
            'height_error_mm': None if table_z is None else (height-table_z)*1000,
            'interpretation': '水平主平面候选；平面拟合残差不是绝对标定精度，单平面不能约束XY平移及绕Z旋转。'}


def load_crop_config(path, frame):
    config = json.loads(Path(path).read_text())
    expected = {'format': 'pika_point_cloud_processing', 'schema_version': 1, 'units': 'm',
                'coordinate_frame': f"robot_base:{frame.calibration.get('robot_group', 'unspecified')}",
                'camera_serial': frame.calibration['serial'], 'camera_parameters_sha256': frame.calibration_sha256}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'裁剪配置{key}与当前相机/基座/标定不匹配；请重新检查并保存范围')
    validate_crop(config['crop'])
    return config['crop']
