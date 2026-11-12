"""场景调参器与 Knowledge 实验台共用的候选/夹爪示意层。"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from gene_agent.knowledge.grasp import GraspProposal
from .app import _colored_box, _viser_pose, build_proxy_geometry, PALM_COLOR_RGBA, FINGER_COLOR_RGBA


class CandidateOverlay:
    def __init__(self, server: Any, *, prefix: str = "/knowledge") -> None:
        self.server = server
        self.prefix = prefix
        self.handles: list[Any] = []
        self.frames: list[Any] = []
        self.proposals: tuple[GraspProposal, ...] = ()
        self.proxy_visible = True
        self.reachability: tuple[bool, ...] | None = None

    def update(self, proposals: Sequence[GraspProposal], *, show_all: bool = False, selected: int = 0) -> None:
        if not proposals:
            raise ValueError("没有可展示的 grasp proposal")
        self.clear()
        self.proposals = tuple(proposals)
        self.reachability = None
        scene = self.server.scene
        # 一次绘制全部 TCP 位置和接近方向；拖动参数时整个分布同步更新。
        # 单个真实夹爪仍用于观察当前候选的指尖细节，避免几十个 mesh 遮住物体。
        points = np.asarray([item.tcp_pose_world[:3, 3] for item in self.proposals], dtype=np.float32)
        directions = np.asarray([item.tcp_pose_world[:3, 2] for item in self.proposals], dtype=np.float32)
        self.cloud = scene.add_point_cloud(f"{self.prefix}/all/positions", points, (24, 154, 178),
                                           point_size=.0035, point_shape="circle", visible=show_all)
        segments = np.stack((points, points + .013 * directions), axis=1)
        self.approaches = scene.add_line_segments(f"{self.prefix}/all/approaches", segments, (24, 154, 178),
                                                   thickness=.001, visible=show_all)
        self.handles.extend((self.cloud, self.approaches))
        self.selected_frame = scene.add_frame(f"{self.prefix}/selected", axes_length=.045, axes_radius=.0015)
        self.handles.append(self.selected_frame)
        self.palm = scene.add_mesh_trimesh(f"{self.prefix}/selected/palm", _colored_box((.075,.018,.014), PALM_COLOR_RGBA))
        self.handles.append(self.palm)
        finger = _colored_box((.008,.014,.055), FINGER_COLOR_RGBA)
        self.left = scene.add_mesh_trimesh(f"{self.prefix}/selected/left", finger)
        self.right = scene.add_mesh_trimesh(f"{self.prefix}/selected/right", finger.copy())
        self.handles.extend((self.left, self.right))
        self.set_proxy_visible(self.proxy_visible)
        self.select(selected)

    def clear(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear(); self.frames.clear()
        self.proposals = ()
        self.reachability = None

    def toggle_all(self, visible: bool) -> None:
        for frame in self.frames:
            frame.visible = visible
        for handle in (getattr(self, "cloud", None), getattr(self, "approaches", None)):
            if handle is not None:
                handle.visible = visible

    def set_reachability(self, results: Sequence[bool] | None) -> None:
        """绿/红显示纯 IK 通过/失败；参数或物体移动后清除过期结论。"""
        self.reachability = None if results is None else tuple(bool(x) for x in results)
        if not self.proposals:
            return
        colors = np.asarray([(24, 154, 178) if self.reachability is None else
                             ((38, 171, 96) if passed else (221, 77, 67))
                             for passed in (self.reachability or (False,) * len(self.proposals))], dtype=np.uint8)
        self.cloud.colors = colors
        self.approaches.colors = np.repeat(colors[:, None, :], 2, axis=1)

    def set_proxy_visible(self, visible: bool) -> None:
        self.proxy_visible = visible
        for name in ("palm", "left", "right"):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.visible = visible

    def select(self, index: int) -> None:
        proposal = self.proposals[max(0, min(index, len(self.proposals)-1))]
        wxyz, position = _viser_pose(proposal.tcp_pose_world)
        self.selected_frame.wxyz = wxyz
        self.selected_frame.position = position
        geometry = build_proxy_geometry(proposal.closing_width_m)
        self.palm.position = geometry.palm_position
        self.left.position = geometry.left_finger_position
        self.right.position = geometry.right_finger_position


__all__ = ["CandidateOverlay"]
