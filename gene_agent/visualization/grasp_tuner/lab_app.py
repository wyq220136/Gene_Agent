"""无机器人、无任务场景的多类型抓取 Knowledge 公式实验台。"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import trimesh

from gene_agent.knowledge.geometry import CuboidConcept, CylinderConcept, RingConcept, TrifoldConcept
from gene_agent.knowledge.grasp import generate_grasps, parse_grasp_profiles
from .controls import GraspParameterControls
from .overlay import CandidateOverlay
from .app import _viser_pose
from .selection import closest_candidate_index
from .feedback import canonical_sha256, write_live_state
from .gripper_preview import GripperMeshPreview


KINDS = ("ring", "cylinder_axial", "cylinder", "cuboidal_block", "trifold_handle")
# 实验台的尺寸由 Concept 类型决定。size_x/size_y 对 Cylinder 是半径，
# 对 Cuboid/Trifold 却是全长，不能像旧界面那样共用一组默认数值。
_COMMON_GEOMETRY = {
    "size_x": .03, "size_y": .03, "size_z": .03,
    "inner_radius": .04, "outer_radius": .055, "arc_deg": 360.,
    "connector_z": .021, "separation": .16,
}
GEOMETRY_DEFAULTS = {
    "ring": _COMMON_GEOMETRY | {"size_z": .014},
    "cylinder_axial": _COMMON_GEOMETRY | {"size_x": .0275, "size_y": .0275, "size_z": .01},
    "cylinder": _COMMON_GEOMETRY | {"size_z": .20},
    "cuboidal_block": _COMMON_GEOMETRY.copy(),
    "trifold_handle": _COMMON_GEOMETRY | {"size_x": .014, "size_y": .15, "size_z": .014},
}
DEFAULT_PARAMETERS = {
    "ring": {"surface": ["top"], "azimuth_deg": {"values": [0.]}, "tip_x_deg": {"values": [0.]}, "tip_y_deg": {"values": [0.]}, "vertical_offset_m": {"values": [.012]}},
    "cylinder_axial": {"surface": ["top"], "depth_m": {"values": [.005]}, "azimuth_deg": {"values": [0.]}, "tip_x_deg": {"values": [0.]}},
    "cylinder": {"cylinder_rotation_deg": {"values": [0.]}, "tip_x_deg": {"values": [0.]}, "vertical_offset_m": {"values": [0.]}, "depth_m": {"values": [.02]}},
    "cuboidal_block": {"face": ["front"], "rotate_z_90": {"values": [False]}, "surface_parallel_offset_m": {"values": [0.]}, "depth_m": {"values": [.01]}, "tip_x_deg": {"values": [0.]}},
    "trifold_handle": {"vertical_offset_m": {"values": [.05]}, "tip_x_deg": {"values": [-60.]}, "tip_y_deg": {"values": [0.]}, "depth_m": {"values": [.013]}},
}


def annular_sector(inner: float, outer: float, height: float, angle_rad: float) -> trimesh.Trimesh | None:
    """用真实的扇形环 mesh 表示部分 Ring；0 弧不伪装成完整环。"""
    if angle_rad <= 0:
        return None
    if math.isclose(angle_rad, 2 * math.pi, abs_tol=1e-12):
        return trimesh.creation.annulus(inner, outer, height=height, sections=96)
    count = max(2, int(math.ceil(angle_rad / (2*math.pi) * 96)))
    vertices = []
    for theta in np.linspace(0., angle_rad, count + 1):
        c, s = math.cos(theta), math.sin(theta)
        vertices.extend(((inner*c,inner*s,-height/2),(outer*c,outer*s,-height/2),(inner*c,inner*s,height/2),(outer*c,outer*s,height/2)))
    faces = []
    for index in range(count):
        a, b = 4*index, 4*(index+1)
        for x,y,z,w in ((a,a+1,b+1,b),(a+2,b+2,b+3,a+3),(a,b,b+2,a+2),(a+1,a+3,b+3,b+1)):
            faces.extend(((x,y,z),(x,z,w)))
    faces.extend(((0,2,3),(0,3,1)))
    last = 4*count
    faces.extend(((last,last+1,last+3),(last,last+3,last+2)))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class KnowledgeGraspLabApp:
    """概念几何和采样参数均由滑块控制，候选只由正式 Knowledge API 生成。"""

    def __init__(self, server: Any, *, initial_kind: str = "ring", state_path: str | None = None, restored: dict | None = None, gripper_case: Any | None = None) -> None:
        if initial_kind not in KINDS:
            raise ValueError(f"未知 generator: {initial_kind}")
        if restored and restored.get("kind") != initial_kind:
            raise ValueError("恢复状态的 grasp 类型与当前实验台不一致")
        self.server = server
        self.gripper_run = str(gripper_case.output_dir) if gripper_case is not None else None
        self.state_path = state_path
        self.input_error: str | None = None
        self._syncing = True
        self.object_handles: list[Any] = []
        self.overlay = CandidateOverlay(server, prefix="/knowledge_lab")
        self.gripper_preview = GripperMeshPreview(server, gripper_case, prefix="/knowledge_lab/real_gripper") if gripper_case is not None else None
        self.overlay.set_proxy_visible(self.gripper_preview is None or not self.gripper_preview.available)
        if self.gripper_preview is not None:
            self.gripper_preview.set_palm_visible(False)
        self._active_kind = initial_kind
        self._geometry_by_kind: dict[str, dict[str, float]] = {}
        self._parameters_by_kind: dict[str, dict] = {}
        server.scene.set_up_direction("+z")
        server.scene.set_background_image(np.full((8,8,3), 250, dtype=np.uint8), format="png")
        server.gui.configure_theme(dark_mode=False, control_width="large", show_logo=False, show_share_button=False, brand_color=(53,112,169))
        server.scene.add_grid("/lab/grid", width=.65, height=.65, cell_size=.025, plane="xy", position=(0,0,-.07))
        server.scene.add_frame("/lab/world", axes_length=.07, axes_radius=.0015)
        server.initial_camera.position = (.25,-.32,.24)
        server.initial_camera.look_at = (0.,0.,0.)
        server.initial_camera.up = (0.,0.,1.)

        @server.on_client_connect
        def _camera(client: Any) -> None:
            client.camera.position = (.25,-.32,.24)
            client.camera.look_at = (0.,0.,0.)
            client.camera.up_direction = (0.,0.,1.)
            client.camera.fov = .75
            client.flush()
            client.get_render(height=8, width=8, transport_format="png", timeout=8.0)
            print(f"[knowledge-lab] scene-ready: {self.kind.value}", flush=True)

        gui = server.gui
        gui.add_markdown("### Grasp Knowledge 实验台\n独立查看 Concept 几何与 grasp 参数；生成公式不依赖 MuJoCo 或 cuRobo。")
        self.kind = gui.add_dropdown("抓取类型", options=KINDS, initial_value=initial_kind)
        self.status = gui.add_markdown("")
        self.show_all = gui.add_checkbox("显示全部候选 TCP", initial_value=True)
        self.palm_toggle = gui.add_checkbox("显示真实夹爪掌部", initial_value=False)
        self.palm_toggle.disabled = self.gripper_preview is None or not self.gripper_preview.available
        self.palm_toggle.on_update(lambda event: self._toggle_palm())
        self.candidate = gui.add_slider("查看候选", min=0, max=0, step=1, initial_value=0)
        self.quick_summary = gui.add_markdown("")
        self.show_all.on_update(lambda event: self.overlay.toggle_all(bool(self.show_all.value)))
        self.candidate.on_update(lambda event: self._select())
        with gui.add_folder("Concept 几何", expand_by_default=True):
            self.dimensions = {}
            for key,label,low,high,initial in (
                ("size_x","X / 半径 X (m)",.005,.2,GEOMETRY_DEFAULTS[initial_kind]["size_x"]), ("size_y","Y / 半径 Y (m)",.005,.2,GEOMETRY_DEFAULTS[initial_kind]["size_y"]),
                ("size_z","高度 Z (m)",.005,.3,GEOMETRY_DEFAULTS[initial_kind]["size_z"]), ("inner_radius","Ring 内半径 (m)",.001,.19,.04),
                ("outer_radius","Ring 外半径 (m)",.005,.2,.055), ("arc_deg","Ring 弧角 (deg)",0.,360.,360.),
                ("connector_z","把手连接段高度 (m)",.005,.15,.021), ("separation","把手两连接段间距 (m)",.02,.4,.16),
            ):
                step = 1. if key == "arc_deg" else (.0005 if key in {"size_x", "size_y"} else .001)
                control = gui.add_slider(label, min=low, max=high, step=step, initial_value=initial)
                self.dimensions[key] = control
                control.on_update(lambda event: self._update())
        self.surface = gui.add_dropdown("接近面", options=("top","bottom"), initial_value="top")
        self.face = gui.add_dropdown("Cuboid 面", options=("front","back","left","right","top","bottom"), initial_value="front")
        self.rotate = gui.add_checkbox("Cuboid 局部 Z 旋转 90°", initial_value=False)
        for control in (self.surface,self.face,self.rotate):control.on_update(lambda event: self._update())
        self.parameters = GraspParameterControls(gui, DEFAULT_PARAMETERS[initial_kind], self._update, self._parameter_error)
        self.kind.on_update(lambda event: self._change_kind())
        self._syncing = False
        self._update_visibility()
        self._update()
        if restored:
            self._restore(restored)

    def _restore(self, restored: dict) -> None:
        """用保存的几何和采样规格重建同一候选。"""
        self._syncing = True
        try:
            for name, value in restored["geometry"].items():
                self.dimensions[name].value = float(value)
            self.surface.value = restored["surface"]
            self.face.value = restored["face"]
            self.rotate.value = bool(restored["rotate_z_90"])
            self.parameters.remove()
            self.parameters = GraspParameterControls(self.server.gui, restored["parameters"], self._update, self._parameter_error)
            self.show_all.value = bool(restored.get("show_all_candidates", True))
            self.palm_toggle.value = bool(restored.get("gripper_palm_visible", False))
            self._toggle_palm()
        finally:
            self._syncing = False
        self._update()
        index = int(restored["selected_index"])
        if not 0 <= index < len(self.overlay.proposals):
            raise ValueError("恢复状态的候选序号超出当前参数范围")
        self.candidate.value = index
        self._select()
        self._geometry_by_kind[str(self.kind.value)] = {
            name: float(control.value) for name, control in self.dimensions.items()
        }
        self._parameters_by_kind[str(self.kind.value)] = self.parameters.read()
        current = self.overlay.proposals[index]
        expected = restored["selected_candidate"]
        if current.candidate_id != expected["candidate_id"] or not np.allclose(current.tcp_pose_world, expected["tcp_pose_world"], atol=1e-9):
            raise ValueError("恢复结果与截图候选不一致；请核对 Knowledge 代码版本")

    def _change_kind(self) -> None:
        if self._syncing:return
        self._syncing = True
        next_kind = str(self.kind.value)
        self._geometry_by_kind[self._active_kind] = {
            name: float(control.value) for name, control in self.dimensions.items()
        }
        try:
            self._parameters_by_kind[self._active_kind] = self.parameters.read()
        except (ValueError, TypeError):
            # 用户尚未修好的 JSON 草稿不阻止切换类型；仅缓存最后有效值。
            pass
        try:
            for name, value in self._geometry_by_kind.get(next_kind, GEOMETRY_DEFAULTS[next_kind]).items():
                self.dimensions[name].value = value
            self.parameters.remove()
            self.parameters = GraspParameterControls(
                self.server.gui, self._parameters_by_kind.get(next_kind, DEFAULT_PARAMETERS[next_kind]),
                self._update, self._parameter_error,
            )
            self._active_kind = next_kind
        finally:
            self._syncing = False
        self._update_visibility()
        self._update()

    def _update_visibility(self) -> None:
        """只显示当前 Concept 和 generator 有意义的控件，避免旧界面式参数堆积。"""
        kind = str(self.kind.value)
        ring = kind == "ring"
        trifold = kind == "trifold_handle"
        for name, control in self.dimensions.items():
            control.visible = name in {"size_x", "size_y", "size_z"} or (ring and name in {"inner_radius", "outer_radius", "arc_deg"}) or (trifold and name in {"connector_z", "separation"})
        self.dimensions["size_x"].visible = not ring
        self.dimensions["size_y"].visible = not ring
        self.surface.visible = kind in {"ring", "cylinder_axial"}
        self.face.visible = kind == "cuboidal_block"
        self.rotate.visible = kind == "cuboidal_block"

    def _concept(self):
        d = {name: float(control.value) for name,control in self.dimensions.items()}
        identity = np.eye(4)
        kind = str(self.kind.value)
        if kind == "ring":
            return RingConcept(d["inner_radius"],d["outer_radius"],d["size_z"],[0,0,1],identity,math.radians(d["arc_deg"]))
        if kind in ("cylinder","cylinder_axial"):
            return CylinderConcept([d["size_x"],d["size_y"]],d["size_z"],identity)
        if kind == "cuboidal_block":
            return CuboidConcept([d["size_x"],d["size_y"],d["size_z"]],identity)
        return TrifoldConcept([d["size_x"],d["size_y"],d["size_z"]], [d["size_x"],.014,d["connector_z"]], d["separation"], identity)

    def _draw_object(self, concept) -> None:
        for handle in reversed(self.object_handles):handle.remove()
        self.object_handles.clear()
        scene=self.server.scene
        color=(67,185,197,255)
        if isinstance(concept,RingConcept):
            mesh=annular_sector(concept.inner_radius_m,concept.outer_radius_m,concept.height_m,concept.angle_span_rad)
            if mesh is None:return
            mesh.visual.face_colors=color
            self.object_handles.append(scene.add_mesh_trimesh("/lab/object/ring",mesh))
        elif isinstance(concept,CylinderConcept):
            mesh=trimesh.creation.cylinder(radius=1.,height=concept.height_m,sections=64)
            mesh.apply_scale((concept.radius_xy_m[0],concept.radius_xy_m[1],1.))
            mesh.visual.face_colors=color
            self.object_handles.append(scene.add_mesh_trimesh("/lab/object/cylinder",mesh))
        elif isinstance(concept,CuboidConcept):
            mesh=trimesh.creation.box(extents=concept.size_xyz_m);mesh.visual.face_colors=color
            self.object_handles.append(scene.add_mesh_trimesh("/lab/object/cuboid",mesh))
        else:
            for name,part in (("grip",concept.grip_segment),("plus",concept.positive_connector),("minus",concept.negative_connector)):
                mesh=trimesh.creation.box(extents=part.size_xyz_m);mesh.visual.face_colors=color
                wxyz,position=_viser_pose(part.pose_in_trifold)
                self.object_handles.append(scene.add_mesh_trimesh(f"/lab/object/{name}",mesh,wxyz=wxyz,position=position))

    def _update(self) -> None:
        if self._syncing:return
        try:
            previous = self.overlay.proposals[int(self.candidate.value)] if self.overlay.proposals and int(self.candidate.value) < len(self.overlay.proposals) else None
            kind=str(self.kind.value)
            concept=self._concept()
            parameters=self.parameters.read()
            if "surface" in parameters:parameters["surface"]=[str(self.surface.value)]
            if "face" in parameters:parameters["face"]=[str(self.face.value)]
            if "rotate_z_90" in parameters:parameters["rotate_z_90"]={"values":[bool(self.rotate.value)]}
            payload={"profiles":{"lab":{"generator":kind,"part":"lab_object","parameters":parameters,"closing_width":{"mode":"geometry","clearance_m":.001}}}}
            profile=parse_grasp_profiles(payload)["lab"]
            count=math.prod(len(values) for values in profile.parameters.values())
            if count>20_000:raise ValueError(f"参数组合将生成 {count} 个候选，超过 Viser 调参上限 20000")
            result=generate_grasps(concept,np.eye(4),profile)
            self._draw_object(concept)
            self.candidate.max=max(0,len(result.proposals)-1)
            self.candidate.value=closest_candidate_index(result.proposals, previous)
            if result.proposals:self.overlay.update(result.proposals,show_all=bool(self.show_all.value),selected=int(self.candidate.value))
            else:self.overlay.clear()
            self._select()
            reasons=", ".join(sorted({item.code for item in result.diagnostics}))
            self.status.content=f"**候选：** {len(result.proposals)}；**拒绝：** {len(result.diagnostics)}"+(f"（{reasons}）" if reasons else "")
            self.input_error = None
            self._write_state()
            print(f"[knowledge-lab] {kind}: {len(result.proposals)} candidates; {len(result.diagnostics)} rejected",flush=True)
        except (ValueError,TypeError,KeyError) as error:
            self.input_error = str(error)
            self.status.content=f"❌ 当前输入无效：{error}。画面仍是上一次有效结果。"
            print(f"[knowledge-lab] invalid: {error}",flush=True)
            self._write_state()

    def _select(self) -> None:
        if self.overlay.proposals:
            self.overlay.select(int(self.candidate.value))
            current = self.overlay.proposals[min(int(self.candidate.value), len(self.overlay.proposals)-1)]
            if self.gripper_preview is not None:
                self.gripper_preview.select(current.tcp_pose_world, opening_width_m=current.closing_width_m)
            self.quick_summary.content = (
                f"**候选 {int(self.candidate.value)+1}/{len(self.overlay.proposals)}** · "
                f"TCP `{np.round(current.tcp_pose_world[:3,3], 3).tolist()}` m"
            )
        self._write_state()

    def _parameter_error(self, message: str) -> None:
        self.input_error = message
        self.status.content = f"❌ 无法切换采样方式：{message}。画面仍是上一次有效结果。"
        print(f"[knowledge-lab] mode switch rejected: {message}", flush=True)
        self._write_state()

    def _toggle_palm(self) -> None:
        if self.gripper_preview is not None:
            self.gripper_preview.set_palm_visible(bool(self.palm_toggle.value))
        self._write_state()

    def _write_state(self) -> None:
        if self.state_path is None or not self.overlay.proposals:
            return
        selected = self.overlay.proposals[min(int(self.candidate.value), len(self.overlay.proposals)-1)]
        geometry = {name: float(control.value) for name, control in self.dimensions.items()}
        parameters = self.parameters.read() if self.input_error is None else None
        payload = {
            "schema_version": 1, "tool": "lab", "kind": str(self.kind.value),
            "geometry": geometry, "parameters": parameters,
            "surface": str(self.surface.value), "face": str(self.face.value),
            "rotate_z_90": bool(self.rotate.value),
            "candidate_count": len(self.overlay.proposals),
            "selected_index": int(self.candidate.value), "selected_candidate": selected.to_dict(),
            "gripper_run": self.gripper_run,
            "gripper_visual": "mujoco_mesh" if self.gripper_preview and self.gripper_preview.available else "box_proxy",
            "gripper_palm_visible": bool(self.palm_toggle.value),
            "show_all_candidates": bool(self.show_all.value),
            "view": "default", "camera": {"position": [.25,-.32,.24], "look_at": [0.,0.,0.], "up": [0.,0.,1.], "fov": .75},
            "input_valid": self.input_error is None, "input_error": self.input_error,
        }
        payload["configuration_sha256"] = canonical_sha256({"kind": payload["kind"], "geometry": geometry, "parameters": parameters})
        write_live_state(self.state_path, payload)


__all__=["KnowledgeGraspLabApp","annular_sector"]
