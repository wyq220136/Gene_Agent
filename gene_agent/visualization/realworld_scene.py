"""AffordToolAct 场景界面与 Gene 真机冻结输入的只读适配。"""
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
import tarfile
from tempfile import TemporaryDirectory

import numpy as np

from gene_agent.realworld.scene_archive import load_scene_archive
from gene_agent.visualization.afford_scene import (
    SceneHandles, add_coordinate_frames, add_world_origin_frame, _wxyz,
)
from gene_agent.visualization.pick_bowl.scene import PickBowlScene

BOWL = "Bowl/Bowl_0"


@contextmanager
def scene_bundle(source):
    """只提取唯一 initial_scene 包；链接、越界、重复路径都拒绝。"""
    source = Path(source).expanduser().resolve()
    if source.is_dir() or source.name == "manifest.json":
        bundle = source if source.is_file() or (source / "manifest.json").is_file() else source / "initial_scene"
        yield bundle
        return
    with tarfile.open(source, "r:*") as archive, TemporaryDirectory(prefix="gene-realworld-scene-") as temporary:
        members = archive.getmembers()
        manifests = [m for m in members if PurePosixPath(m.name).parts[-2:] == ("initial_scene", "manifest.json")]
        if len(manifests) != 1:
            raise ValueError("日志中 initial_scene/manifest.json 缺失或不唯一；请指定单次运行")
        prefix = PurePosixPath(manifests[0].name).parent
        destination = Path(temporary)
        seen = set()
        total = 0
        for member in members:
            path = PurePosixPath(member.name)
            if not path.is_relative_to(prefix):
                continue
            relative = path.relative_to(prefix)
            if ".." in path.parts or "\\" in str(path) or path.is_absolute():
                raise ValueError("压缩包包含越界路径")
            if member.isdir():
                continue
            if not member.isfile() or relative in seen:
                raise ValueError("现场包包含链接、特殊文件或重复路径")
            seen.add(relative)
            total += member.size
            if total > 1024 ** 3 or len(seen) > 20000:
                raise ValueError("现场包超过显示工具的解包限制")
            target = destination / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream:
                target.write_bytes(stream.read())
        yield destination


def load_observation(bundle):
    """加载当时消费的模型/qpos/碰撞；RGB-D 不完整只标缺失，不换帧。"""
    scene, model, data = load_scene_archive(bundle)
    rgbd, warning = None, None
    try:
        from gene_agent.visualization.realworld_alignment import load_inputs
        rgbd = load_inputs(run=Path(scene["offline_replay"]["source_manifest"]).parent)
        if rgbd["source"]["frame_number"] != scene["provenance"]["source_frame_number"]:
            raise ValueError("RGB-D 与操作场景不是同帧")
    except (ValueError, KeyError, OSError) as error:
        rgbd = None
        warning = f"原始 RGB-D 对照不可用：{type(error).__name__}: {error}"
    return scene, model, data, rgbd, warning


def observation_summary(scene, rgbd=None, warning=None):
    provenance = scene["provenance"]
    table = np.asarray(provenance["table_pose_world"], float)
    top = table.copy()
    top[:3, 3] += table[:3, 2] * float(provenance["accepted_table_thickness_m"]) / 2
    return dict(mode="read_only_realworld_observation", physics_stepped=False,
        snapshot_sha256=scene["snapshot"]["snapshot_sha256"],
        frame_number=provenance.get("source_frame_number"),
        active_group=provenance["active_group"], world_frame=scene["snapshot"].get("world_frame"),
        T_world_sdk=provenance["T_world_sdk"],
        T_world_bowl=provenance["aligned_object_poses_world"][BOWL],
        T_world_table_center=table.tolist(), T_world_configured_table_top=top.tolist(),
        table_size_xy_m=provenance["table_size_xy_m"],
        table_thickness_m=provenance["accepted_table_thickness_m"],
        gripper_widths_m=provenance.get("gripper_widths_m"),
        joint_positions_by_group_rad=provenance.get("joint_positions_by_group_rad"),
        pose_source_modes=provenance.get("pose_source_modes"),
        object_observations=(provenance.get("perception_refresh") or {}).get("object_observations"),
        source_paths=provenance.get("source_paths"),
        active_robot_spheres_count=len(scene["snapshot"].get("active_robot_spheres", [])),
        rgbd_available=rgbd is not None, warning=warning)


def checkbox(server, label, handles, *, visible=True):
    SceneHandles._set_visible(handles, visible)
    control = server.gui.add_checkbox(label, initial_value=visible)
    control.on_update(lambda _: SceneHandles._set_visible(handles, control.value))
    return control


class RealworldSceneApp:
    def __init__(self, server, scene, model, data, rgbd=None, warning=None):
        self.server = server
        self.summary = observation_summary(scene, rgbd, warning)
        provenance = scene["provenance"]
        server.scene.set_up_direction("+z")
        server.gui.add_markdown("## 真机操作输入 · AffordToolAct Viser\n\n"
            "只读冻结观测；单位米。模型/qpos/规划障碍均来自本轮 initial_scene，"
            "未落稳、未推进物理、未重新生成候选或规划。SDK +Z 不等于已测得的重力方向。")
        server.gui.add_markdown(f"**活动臂：{self.summary['active_group']}**；"
            f"frame={self.summary['frame_number']}\n\n"
            f"snapshot: `{self.summary['snapshot_sha256']}`\n\n"
            "微波炉可能保留此前的操作状态；各物体来源见下方记录，不标为全部同帧。")
        # 所有图层使用同一 world：绝不把 SDK mesh 再乘一次 base_from_world。
        self.geometry = PickBowlScene(server, model, data, scene.get("robot_metadata", {}), scene["snapshot"])
        import mujoco
        robot_visuals, object_visuals = [], []
        for geom_id, handle in self.geometry.visual_handles.items():
            geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
            group = object_visuals if any(word in (body + "/" + geom).lower() for word in ("bowl", "table", "microwave")) else robot_visuals
            group.append(handle)
        spheres = [h for n, h in self.geometry.collision_handles.items() if n.startswith("robot_sphere_")]
        obstacles = [h for n, h in self.geometry.collision_handles.items() if not n.startswith("robot_sphere_")]
        self.layers = SceneHandles(robot_visuals, spheres, object_visuals, obstacles)
        self.layers.add_layer_controls(server, labels=("机器人模型（两臂/夹爪/本体）", "已记录的活动臂碰撞球", "操作物体模型（碗/桌/微波炉）", "实际规划障碍（红色半透明）"))
        SceneHandles._set_visible(obstacles, True)
        SceneHandles._set_visible(spheres, True)
        if not spheres:
            server.gui.add_markdown("本快照未记录活动臂碰撞球，不能从当前 URDF 重算冒充当时的球。规划障碍包含桌面和其他外物。")
        table_top = np.asarray(self.summary["T_world_configured_table_top"])
        frames = {"active_sdk": np.asarray(provenance["T_world_sdk"]),
                  "bowl_consumed": np.asarray(self.summary["T_world_bowl"]),
                  "configured_table_center": np.asarray(self.summary["T_world_table_center"]),
                  "configured_table_top": table_top}
        frame_handles = [add_world_origin_frame(server, np.eye(4))]
        if rgbd is not None:
            frames["camera"] = frames["active_sdk"] @ rgbd["T_sdk_camera"]
        frame_handles += add_coordinate_frames(server, frames, axes_length=.10)
        for name, pose in frames.items():
            frame_handles.append(server.scene.add_label(f"/labels/{name}", text=name, position=pose[:3, 3]))
        with server.gui.add_folder("感知与桌面对照"):
            checkbox(server, "坐标轴和名称（X红/Y绿/Z蓝）", frame_handles)
            half = np.asarray(provenance["table_size_xy_m"]) / 2
            local = np.array([[-half[0], -half[1], 0], [half[0], -half[1], 0], [half[0], half[1], 0], [-half[0], half[1], 0]])
            top_handle = server.scene.add_mesh_simple("/configured_table_top", local, np.array([[0, 1, 2], [0, 2, 3]]),
                color=(255, 176, 32), opacity=.55, side="double", position=table_top[:3, 3], wxyz=_wxyz(table_top))
            checkbox(server, "配置桌面顶面（橙色，未拟合/修正）", [top_handle])
            if rgbd is not None:
                self._add_rgbd(server, rgbd, frames["camera"], frames["bowl_consumed"])
            else:
                server.gui.add_markdown(warning or "没有原始 RGB-D；只显示已冻结的操作模型。")
        with server.gui.add_folder("实际输入和来源（矩阵平移单位米）"):
            server.gui.add_markdown("```json\n" + json.dumps(self.summary, ensure_ascii=False, indent=2) + "\n```")
            server.gui.add_markdown("优先检查：原始注册 mesh 是否贴合点云中的实碗；"
                "配置桌面是否贴合可见桌面；SDK/相机与独立测量参照是否一致。"
                "仅凭碗和桌面的相对矛盾，不能唯一证明外参有误。")
        focus = frames["bowl_consumed"][:3, 3]
        server.initial_camera.look_at = tuple(focus)
        server.initial_camera.position = tuple(focus + np.array([-.8, -.9, .65]))
        server.initial_camera.up = (0., 0., 1.)
        server.initial_camera.fov = .8
        view_up = server.gui.add_dropdown("仅改变浏览器视图的上方向", options=("+z", "-z", "+y", "-y"), initial_value="+z")
        view_up.on_update(lambda _: server.scene.set_up_direction(view_up.value))

    def _add_rgbd(self, server, rgbd, camera_pose, consumed_bowl):
        import trimesh
        indices = np.flatnonzero(rgbd["valid"])
        indices = indices[np.linspace(0, len(indices) - 1, min(80000, len(indices)), dtype=int)]
        cloud = server.scene.add_point_cloud("/observed_rgbd", points=rgbd["cloud"].reshape(-1, 3)[indices],
            colors=rgbd["rgb"].reshape(-1, 3)[indices], point_size=.0025,
            position=camera_pose[:3, 3], wxyz=_wxyz(camera_pose))
        checkbox(server, "同帧原始 RGB-D 点云（彩色）", [cloud])
        mesh = trimesh.load(rgbd["mesh_file"], force="mesh", process=False)
        raw_world = camera_pose @ rgbd["T_camera_bowl"]
        raw = server.scene.add_mesh_simple("/registered_bowl_raw", np.asarray(mesh.vertices), np.asarray(mesh.faces),
            color=(20, 210, 255), opacity=.40, side="double", position=raw_world[:3, 3], wxyz=_wxyz(raw_world))
        checkbox(server, "原始注册碗 mesh（青色，不追加 Concept）", [raw])
        delta = np.linalg.inv(consumed_bowl) @ raw_world
        server.gui.add_markdown("原始注册链：T_world_sdk × T_sdk_camera × T_camera_bowl；"
            f"与操作碗 pose 的平移差 {np.linalg.norm(delta[:3, 3])*1000:.6f} mm，"
            f"转角差 {np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3])-1)/2, -1, 1))):.6f}°。")
        h, w = rgbd["valid"].shape
        frustum = server.scene.add_camera_frustum("/observed_camera", fov=float(2*np.arctan(h/(2*rgbd["K"][1, 1]))),
            aspect=w/h, scale=.12, image=rgbd["rgb"], position=camera_pose[:3, 3], wxyz=_wxyz(camera_pose))
        checkbox(server, "采集相机视锥（示意视场）", [frustum], visible=False)
        # RGB 图像不使用中心视锥近似，保留原始内参供像素核对。
        server.gui.add_image(rgbd["rgb"], label=f"原始 RGB · frame {rgbd['source']['frame_number']}")
        server.gui.add_markdown("RGB-D/注册资产来源：\n```json\n" + json.dumps(rgbd["source"], ensure_ascii=False, indent=2) + "\n```")
