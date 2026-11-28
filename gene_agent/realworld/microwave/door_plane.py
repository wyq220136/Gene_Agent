"""AffordToolAct 门板 ROI/RANSAC/yaw 校正，使用当前标定 URDF 的坐标。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.perception_modules.pose_est.microwave_pose_postprocess import (
    _load_rgbd_points, select_door_region_points, fit_plane_ransac,
    correct_pose_yaw_from_plane,
)


def _origin(element):
    result = np.eye(4)
    origin = element.find('origin')
    if origin is not None:
        result[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        result[:3, :3] = Rotation.from_euler('xyz', np.fromstring(
            origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return result


class DoorPlaneObserver:
    def __init__(self, assets):
        self.assets = assets
        urdfs = [Path(name) for name in assets.get('sources', {}) if name.endswith('.urdf')]
        self.available = len(urdfs) == 1 and urdfs[0].is_file()
        if not self.available:
            return
        urdf = ET.parse(urdfs[0]).getroot()
        door = urdf.find("link[@name='door']")
        panel = door.find("collision[@name='door_panel_collision']")
        handle = door.find("collision[@name='handle_grip_collision']")
        mesh_joint = np.asarray(assets['alignment']) @ np.asarray(assets['joints']['door']['T_asset_joint'])
        # URDF panel box is X/Y(thickness)/Z. The Afford ROI expects X/Y/Z(normal).
        plane_axes = np.eye(4)
        plane_axes[:3, :3] = Rotation.from_euler('x', 90, degrees=True).as_matrix()
        panel_size = np.fromstring(panel.find('geometry/box').get('size'), sep=' ')
        self.door = SimpleNamespace(pose=mesh_joint @ _origin(panel) @ plane_axes,
                                    size_xyz=panel_size[[0, 2, 1]])
        handle_size = np.fromstring(handle.find('geometry/box').get('size'), sep=' ')
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        self.handle = SimpleNamespace(pose=mesh_joint @ _origin(handle),
                                      local_vertices=signs * handle_size / 2)

    def correct(self, pose_sdk, capture, T_sdk_camera):
        pose = np.asarray(pose_sdk)
        diagnostic = {'method': 'AffordToolAct_door_roi_ransac_yaw', 'applied': False,
                      'raw_pose_sdk': pose.tolist(), 'position_modified': False}
        if not self.available:
            return pose.copy(), {**diagnostic, 'reason': 'calibrated_urdf_unavailable'}
        capture = Path(capture)
        if not all((capture/name).is_file() for name in ('rgb.png', 'depth.png', 'camera.json')):
            return pose.copy(), {**diagnostic, 'reason': 'auxiliary_rgbd_unavailable'}
        camera = json.loads((capture / 'camera.json').read_text(encoding='utf-8'))
        points = _load_rgbd_points(capture/'rgb.png', capture/'depth.png', camera,
                                  np.asarray(T_sdk_camera), side='right')
        try:
            mask, bounds = select_door_region_points(points, pose, self.door, self.handle)
        except ValueError as error:
            return pose.copy(), {**diagnostic, 'reason': 'door_roi_unavailable', 'detail': str(error)}
        selected = points[mask]
        diagnostic.update(selected_point_count=len(selected), roi_bounds_m=bounds.tolist())
        if len(selected) < 40:
            return pose.copy(), {**diagnostic, 'reason': 'door_plane_occluded_or_sparse'}
        if len(selected) > 2000:
            selected = selected[np.linspace(0, len(selected)-1, 2000, dtype=int)]
        try:
            normal, offset, inliers = fit_plane_ransac(selected, iterations=300)
        except (ValueError, RuntimeError) as error:
            return pose.copy(), {**diagnostic, 'reason': 'plane_fit_unavailable', 'detail': str(error)}
        fraction = float(inliers.mean())
        rmse = float(np.sqrt(np.mean((selected[inliers] @ normal + offset)**2)))
        corrected, yaw_deg = correct_pose_yaw_from_plane(pose, self.door.pose, normal)
        expected = pose[:3, :3] @ self.door.pose[:3, 2]
        if expected @ normal < 0:
            normal = -normal
        diagnostic.update(normal_sdk=normal.tolist(), yaw_correction_deg=yaw_deg,
                          inlier_fraction=fraction, rmse_m=rmse,
                          max_yaw_correction_deg=15., corrected_pose_sdk=corrected.tolist())
        if fraction < .6 or rmse > .006 or abs(normal[2]) > .35 or abs(yaw_deg) > 15.:
            return pose.copy(), {**diagnostic, 'reason': 'plane_quality_or_correction_out_of_bounds'}
        return corrected, {**diagnostic, 'applied': True, 'reason': 'bounded_door_plane_yaw'}

    def reconstruct_closed(self, pose_sdk, capture, T_sdk_camera, up_sdk):
        """准备阶段重建闭门机身；不改变原有平面采纳门限，辅助失败不退出。"""
        pose = np.array(pose_sdk, dtype=float, copy=True)
        diagnostic = {
            "method": "closed_door_plane_table_up_and_handle_center",
            "applied": False, "fallback": "original_registered_reference",
        }
        try:
            _, plane = self.correct(pose, capture, T_sdk_camera)
            diagnostic["door_plane"] = plane
            if not plane["applied"]:
                return pose, {**diagnostic, "reason": plane.get("reason", "plane_not_applied")}

            alignment = np.asarray(self.assets["alignment"], dtype=float)
            inverse_alignment = np.linalg.inv(alignment)
            asset_handle = np.asarray(self.assets["T_asset_handle"], dtype=float)
            # 把手注册的抓握中心作平移锚点；不用部分可见点云的质心替代它。
            handle_sdk = (pose @ alignment @ asset_handle)[:3, 3]
            up = np.asarray(up_sdk, dtype=float)
            up = up / np.linalg.norm(up)
            normal = np.asarray(plane["normal_sdk"], dtype=float)
            normal = normal - up * float(normal @ up)
            normal = normal / np.linalg.norm(normal)
            # door.pose是mesh→门板；先回到asset坐标，再与实测法向建立对应。
            asset_up = np.array([0., 0., 1.])
            asset_normal = inverse_alignment[:3, :3] @ self.door.pose[:3, 2]
            asset_normal -= asset_up * float(asset_normal @ asset_up)
            asset_normal /= np.linalg.norm(asset_normal)
            measured_basis = np.column_stack((normal, up, np.cross(normal, up)))
            asset_basis = np.column_stack((asset_normal, asset_up,
                                          np.cross(asset_normal, asset_up)))
            root = np.eye(4)
            root[:3, :3] = measured_basis @ asset_basis.T
            root[:3, 3] = handle_sdk - root[:3, :3] @ asset_handle[:3, 3]
            reference = root @ inverse_alignment
            if not np.isfinite(reference).all():
                return pose, {**diagnostic, "reason": "degenerate_reconstruction_geometry"}

            joint = self.assets["joints"]["door"]
            door_joint = root @ np.asarray(joint["T_asset_joint"])
            axis = root[:3, :3] @ np.asarray(joint["axis_asset"])
            diagnostic.update(applied=True, reason="accepted_plane_rebuilt_closed_frame",
                fallback=None, T_sdk_scene_asset=root.tolist(),
                T_sdk_door_joint=door_joint.tolist(), door_axis_sdk=axis.tolist(),
                handle_center_sdk=handle_sdk.tolist(),
                handle_center_after_sdk=(root @ asset_handle)[:3, 3].tolist(),
                table_up_sdk=up.tolist(), corrected_reference_pose_sdk=reference.tolist())
            return reference, diagnostic
        except (ValueError, TypeError, KeyError, AttributeError, OSError, RuntimeError,
                np.linalg.LinAlgError) as error:
            # 可选几何辅助不添加任务失败条件；现有注册/输入检查保持原样。
            return pose, {**diagnostic, "reason": "auxiliary_reconstruction_unavailable",
                          "error_type": type(error).__name__, "detail": str(error)}

    @staticmethod
    def reference_capture(setup):
        paths = [Path(name).parent for name in setup.get('input_sha256', {}) if name.endswith('/rgb.png')]
        return next((p for p in paths if (p/'depth.png').is_file() and (p/'camera.json').is_file()), None)
