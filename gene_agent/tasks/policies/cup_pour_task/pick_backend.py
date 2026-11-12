"""从杯把上方抓取：Concept 定位、cuRobo 规划、MuJoCo 双指接触。"""

from __future__ import annotations

from gene_agent.knowledge.grasp.cup_handle import top_down_cup_handle_grasp

import mujoco
import numpy as np

from gene_agent.tasks.tabletop.pick import TabletopPickRun, _body_pose
from gene_agent.robots.contact.pika import classify_pika_brush_grasp
from gene_agent.motion.curobo.gripper import cover_bounds
from gene_agent.motion.curobo.scene import filter_collision_snapshot




def classify_cup_handle_contacts(contacts: list[dict], sites_world: dict,
                                 handle_top_world: np.ndarray) -> dict:
    """双指都必须接触杯把上缘，杯壁接触不能算作抓杯把。"""
    handle_name = "cup_green_handle_top_proxy"
    handle_rows, wrong_rows = [], []
    for row in contacts:
        bodies = set(row.get("body_names", ()))
        if "cup_green" not in bodies or not any(
                name.startswith("pika_gripper_") for name in bodies):
            continue
        if handle_name in row.get("geom_names", ()):
            handle_rows.append(row)
        else:
            wrong_rows.append(row)
    result = classify_pika_brush_grasp(
        handle_rows, sites_world, handle_top_world,
        site_radius_m=.022, handle_radius_m=.020, object_name="cup_green")
    if wrong_rows:
        result["undesired_gripper_contacts"].extend(wrong_rows)
        result["success"] = False
        result["bilateral_pad_contact"] = False
    result["contact_mode"] = "cup_handle_top_proxy_bilateral_inner_finger"
    result["wrong_cup_contact_count"] = len(wrong_rows)
    return result


class CupPickRun(TabletopPickRun):
    """复用同步 Pick 契约，但用杯把自身的抓取目标与碰撞/接触语义。"""

    BENCHMARK_ID = "cup_pour_task"
    EXTRA_CODE_SOURCES = ("gene_agent/tasks/policies/cup_pour_task/pick_backend.py",
                          "gene_agent/knowledge/grasp/cup_handle.py")
    SUPPORTED_OBJECTS = frozenset({"cup_green"})
    OBJECT_ROLE = "source_container"
    SKILL_NAME = "pick_cup_handle_top_down"
    GRASP_CANDIDATE_SUFFIX = "grasp_handle_top_down"

    def _collision_spheres(self, *, max_cell_m: float = .025) -> np.ndarray:
        rows = []
        body_id = self.model.body(self.object_name).id
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) != body_id:
                continue
            if (self.model.geom_contype[geom_id] == 0 and
                    self.model.geom_conaffinity[geom_id] == 0):
                continue
            geom_type = int(self.model.geom_type[geom_id])
            size = np.asarray(self.model.geom_size[geom_id], dtype=float)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                half = size
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                half = np.array([size[0], size[0], size[1]])
            else:
                raise ValueError("杯子附着代理只支持 box/cylinder 碰撞几何")
            center = np.asarray(self.model.geom_pos[geom_id], dtype=float)
            rows.extend([(*sphere["center"], sphere["radius"])
                         for sphere in cover_bounds(
                             np.stack((center - half, center + half)),
                             max_cell_m=max_cell_m)])
        if not rows:
            raise ValueError("杯子没有规划附着碰撞代理")
        return np.asarray(rows, dtype=float)

    def _grasp_target(self, body_pose: np.ndarray) -> np.ndarray:
        pose = top_down_cup_handle_grasp(body_pose, self.grip.pose_in_object)
        pose[:3, 3] += body_pose[:3, :3] @ [self.grasp_handle_x_offset_m, 0., 0.]
        return pose

    def _contact_classifier(self, contacts, _descriptor, *, object_body: str) -> dict:
        frozen = self.runtime.copy_data()
        sites = {side: frozen.site(f"pika_{side}_inner").xpos.copy()
                 for side in ("left", "right")}
        handle_top = (_body_pose(frozen.body(self.object_name)) @
                      np.array([0., .068, .093, 1.]))[:3]
        return classify_cup_handle_contacts(contacts, sites, handle_top)

    def _snapshot_filter(self, stage: str, snapshot: dict) -> dict:
        if stage != "lift":
            return snapshot
        # 杯内球及杯底与桌面是本阶段预期接触；只从附着物规划
        # 快照排除。整条轨迹仍由 MuJoCo 完整几何逐点复核。
        return filter_collision_snapshot(
            snapshot, self.output_dir / "motion_curobo" / "collision" /
            "lift_cup_contents_scope", keep_body_names=None,
            rejected_body_names={self.model.body(i).name for i in range(self.model.nbody)
                                 if self.model.body(i).name.startswith("pour_ball_")},
            rejected_geom_names={"scene_table"},
            attached_object_id=self.object_name,
            scope="杯内球与杯底桌面接触的阶段性规划排除；完整 MuJoCo 碰撞复核",
        )

    def _allowed_contacts(self, stage: str) -> list[tuple[str, str]]:
        allowed = super()._allowed_contacts(stage)
        if stage == "lift":
            allowed.extend((self.object_name, self.model.body(i).name)
                           for i in range(self.model.nbody)
                           if self.model.body(i).name.startswith("pour_ball_"))
        return allowed

    def _attachment(self, stage: str, observation: dict) -> dict | None:
        result = super()._attachment(stage, observation)
        if result is not None:
            result["coverage"] = "cup_bottom_wall_handle_box_cylinder_aabb_cells_25mm"
        return result
