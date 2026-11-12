"""FoundationPose 位姿估计与 tracking 模块。"""

from importlib import import_module

_EXPORTS = {
    "FoundationPoseBackend": "fp_pose_est",
    "PoseSession": "fp_pose_est",
    "PoseClient": "pose_client",
    "estimate_task_poses": "task_pose_est",
    "correct_microwave_pose_with_door_plane": "microwave_pose_postprocess",
    "correct_pose_with_external_knowledge": "batch_est_pose",
    "load_camera_parameters": "batch_est_pose",
    "save_pose_json": "batch_est_pose",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
