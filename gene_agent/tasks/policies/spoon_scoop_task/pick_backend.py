"""勺子专用抓取目标、碰撞代理与接触检查。"""
from __future__ import annotations
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from gene_agent.motion.curobo.gripper import cover_bounds
from gene_agent.motion.curobo.scene import filter_collision_snapshot
from gene_agent.robots.contact.pika import classify_pika_brush_grasp
from gene_agent.tasks.tabletop.pick import TabletopPickRun, _body_pose, _write

def _spoon_collision_spheres(model, data, object_name: str) -> np.ndarray:
    """完整覆盖勺柄与勺头 box 碰撞代理，返回勺子 body frame 的球。"""
    object_id = model.body(object_name).id
    rows = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != object_id:
            continue
        if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError("勺子规划附着代理要求 box 碰撞几何")
        center = np.asarray(model.geom_pos[geom_id], dtype=float)
        half = np.asarray(model.geom_size[geom_id], dtype=float)
        rows.extend([(*sphere["center"], sphere["radius"])
                     for sphere in cover_bounds(np.stack((center - half, center + half)),
                                                max_cell_m=.022)])
    if not rows:
        raise ValueError("勺子没有可用于规划附着的碰撞几何")
    return np.asarray(rows, dtype=float)


class SpoonPickRun(TabletopPickRun):
    BENCHMARK_ID = "spoon_scoop_task"
    EXTRA_CODE_SOURCES = ("gene_agent/tasks/policies/spoon_scoop_task/pick_backend.py",)
    SUPPORTED_OBJECTS = frozenset({"spoon_orange", "spoon_pink"})
    OBJECT_ROLE = "tool"
    SKILL_NAME = "pick_spoon_handle"
    GRASP_CANDIDATE_SUFFIX = "grasp_handle_diagonal"

    def _collision_spheres(self) -> np.ndarray:
        return _spoon_collision_spheres(self.model, self.data, self.object_name)

    def _grasp_target(self, body_pose: np.ndarray) -> np.ndarray:
        grasp = body_pose @ self.grip.pose_in_object
        grasp[:3, 3] += body_pose[:3, :3] @ [self.grasp_handle_x_offset_m, 0., 0.]
        grasp[2, 3] -= .002
        grasp[:3, :3] = Rotation.from_euler("y", 45, degrees=True).as_matrix()
        return grasp

    def _contact_classifier(self, contacts, _descriptor, *, object_body: str) -> dict:
        frozen = self.runtime.copy_data()
        sites = {side: frozen.site(f"pika_{side}_inner").xpos.copy()
                 for side in ("left", "right")}
        center = (_body_pose(frozen.body(self.object_name)) @
                  self.grip.pose_in_object)[:3, 3]
        return classify_pika_brush_grasp(
            contacts, sites, center, object_name=object_body,
            site_radius_m=.04, handle_radius_m=.04)

    def _snapshot_filter(self, stage: str, snapshot: dict) -> dict:
        if stage not in {"approach", "lift"}:
            return snapshot
        # 勺柄贴桌，Pika 规划球对桌面过度保守；完整 MuJoCo 几何
        # 逐点复核所有 robot-table 接触，不能用移除快照掩盖穿透。
        return filter_collision_snapshot(
            snapshot, self.output_dir / "motion_curobo" / "collision" /
            f"{stage}_table_contact_scope",
            keep_body_names=None, rejected_body_names=set(),
            rejected_geom_names={"scene_table"},
            attached_object_id=self.object_name if stage == "lift" else None,
            scope="勺柄贴桌阶段的规划球桌面净空；整条轨迹由完整 MuJoCo 几何复核",
        )
