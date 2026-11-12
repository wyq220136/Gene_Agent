"""相机外参标定。

统一入口：python -m gene_agent.perception_modules.camera.calibration --help。
路线B：handeye → moving-board；路线A：manual（内含板相机步骤）。
数据默认集中在data/calibration/runs/；其他流程和坐标约定见README.md。
导入本包不加载设备SDK或启动运动。
"""

from importlib import import_module

_EXPORTS = {
    "calibrate_camera_to_board": "camera_to_board",
    "calibrate_board_to_robot": "board_to_robot",
    "compute_extrinsic": "compute_extrinsic",
    "convert_left_to_right": "convert_to_right_arm",
    "visualize_with_pointcloud": "convert_to_right_arm",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """只在实际运行该标定步骤时加载 OpenCV 等硬件依赖。"""
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
