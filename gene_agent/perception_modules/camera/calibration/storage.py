"""标定会话统一存储；旧路径只在读取时映射，不改写历史采集内容。"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[4]
RUN_KINDS = {'handeye', 'moving_board', 'fixed_board', 'board_camera', 'crosscheck', 'manual'}
LEGACY_PREFIXES = {
    'calibration_wrist_': ('runs', 'handeye'),
    'calibration_moving_board_': ('runs', 'moving_board'),
    'calibration_fixed_board_': ('runs', 'fixed_board'),
    'calibration_board_camera_': ('runs', 'board_camera'),
    'calibration_third_crosscheck_': ('runs', 'crosscheck'),
    'calibration_single_': ('runs', 'manual'),
    'calibration_diagnostics_': ('diagnostics',),
}
RESULT_KINDS = {
    'third_camera': ('third_view_extrinsic.json', 'extrinsic_matrix', '第三相机 → 基座'),
    'handeye': ('handeye.json', 'T_flange_camera', '手腕相机 → 法兰'),
    'manual': ('cam_param.json', 'extrinsic_matrix', '量尺方案相机 → 基座'),
    'board_camera': ('camera_to_board.json', 'T_camera_board', '标定板 → 相机'),
}


def default_run_dir(kind: str) -> Path:
    if kind not in RUN_KINDS:
        raise ValueError(f'未知标定用途：{kind}')
    return Path('data/calibration/runs') / kind / f'calibration_{kind}_{datetime.now():%Y%m%d_%H%M%S}'


def resolve_calibration_path(value: str | Path, *, root=None) -> Path:
    """已有路径优先；仅兼容本仓库data根目录下已整体迁移的标定会话。"""
    path = Path(value)
    if path.exists():
        return path
    root = Path(root) if root is not None else ROOT
    try:
        suffix = path.absolute().relative_to(root / 'data')
    except ValueError:
        return path
    if not suffix.parts:
        return path
    for prefix, folders in LEGACY_PREFIXES.items():
        if suffix.parts[0].startswith(prefix):
            destination = root / 'data/calibration' / Path(*folders) / suffix
            return destination if destination.exists() else path
    return path


def resolve_cli_paths(args, *fields):
    for field in fields:
        value = getattr(args, field, None)
        if value is not None:
            setattr(args, field, resolve_calibration_path(value))


def same_recorded_config(actual: dict, expected: dict, path_keys=()) -> bool:
    """迁移前后路径等价，其余配置仍严格比较；不修改输入字典。"""
    def normalize(record):
        result = dict(record)
        for key in path_keys:
            if result.get(key) is not None:
                result[key] = str(resolve_calibration_path(result[key]).resolve())
        return result
    return normalize(actual) == normalize(expected)


def publish_result(data_dir, kind: str, *, root=None):
    """仅在操作者确认后调用；固定入口指向整轮目录，保留相对来源文件关系。"""
    root = Path(root) if root is not None else ROOT
    run = Path(data_dir).resolve()
    # 外部目录由调用方自行管理，不把其他项目或离线临时数据写进本仓库目录索引。
    if not run.is_relative_to(root):
        return None
    filename, matrix_key, title = RESULT_KINDS[kind]
    result = json.loads((run / filename).read_text())
    if result.get('quality_passed') is False or matrix_key not in result:
        raise ValueError('不能把失败或缺少变换的结果设为最近正式标定')
    from gene_agent.knowledge.geometry.transforms import validate_pose
    import numpy as np
    validate_pose(np.asarray(result[matrix_key], dtype=float))
    catalog = root / 'data/calibration/latest'
    catalog.mkdir(parents=True, exist_ok=True)
    alias = catalog / kind
    if alias.exists() and not alias.is_symlink():
        raise FileExistsError(f'固定结果入口已被普通目录占用：{alias}')
    temporary = catalog / f'.{kind}.{uuid4().hex}.tmp'
    try:
        temporary.symlink_to(os.path.relpath(run, catalog), target_is_directory=True)
        os.replace(temporary, alias)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'\n===== 已保存正式标定：{title} =====')
    print(f'固定结果入口：{alias / filename}')
    print(f'原始采集目录：{run}')
    print('随时查看：python -m gene_agent.perception_modules.camera.calibration results\n')
    if kind in ('third_camera', 'manual'):
        extra = '' if kind == 'third_camera' else f' --live --camera-params {alias / filename}'
        print(f'标定后桌面/裁剪检查：python -m gene_agent.perception_modules.camera.calibration inspect{extra}\n')
    return alias


def show_results():
    print('最近确认保存的标定结果（候选和失败采集不会替换这里）：')
    for kind, (filename, _, title) in RESULT_KINDS.items():
        path = ROOT / 'data/calibration/latest' / kind / filename
        if path.is_file():
            print(f'\n{title}\n  固定入口：{path}\n  来源：{path.resolve()}')
        else:
            print(f'\n{title}：尚无登记结果')
    return 0


def protect_latest_result(data_dir):
    """人工流程会先写候选，禁止原地覆盖固定入口正在引用的正式会话。"""
    run = Path(data_dir).resolve()
    for kind in RESULT_KINDS:
        alias = ROOT / 'data/calibration/latest' / kind
        if alias.is_symlink() and alias.resolve() == run:
            raise FileExistsError('该目录是最近正式标定结果，请新建输出目录，避免候选覆盖正式结果')


def standard_camera_parameters(data_dir, camera_sample, result):
    """正式基座外参沿用cam_param字段；内参与分辨率必须取同一台相机的原始采集。"""
    import numpy as np
    from gene_agent.knowledge.geometry.transforms import validate_pose

    payload = dict(result)
    raw_transform = payload.get('extrinsic_matrix', payload.get('T_sdk_third'))
    transform = validate_pose(np.asarray(raw_transform, dtype=float))
    if 'T_sdk_third' in payload and not np.allclose(transform, payload['T_sdk_third'], atol=1e-8, rtol=0):
        raise ValueError('同一结果中的外参字段不一致')
    with np.load(Path(data_dir) / camera_sample['rgbd_file']) as raw:
        rgb = raw['rgb']
        K = np.asarray(raw['K'], dtype=float)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('缺少有效RGB尺寸，不能补写相机参数')
        height, width = rgb.shape[:2]
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('相机内参无效')
    if not np.allclose(K, camera_sample['K'], atol=1e-6, rtol=0):
        raise ValueError('保存的原图内参与会话不一致')
    serial = camera_sample['serial']
    specified = payload.get('serial', payload.get('camera_serial'))
    if not serial or specified not in (None, serial):
        raise ValueError('结果与原始观测的相机serial不一致')
    payload.pop('T_sdk_third', None)
    payload.pop('camera_serial', None)
    return {**payload, 'width': int(width), 'height': int(height), 'intrinsic_matrix': K.tolist(),
            'serial': serial, 'extrinsic_matrix': transform.tolist()}
