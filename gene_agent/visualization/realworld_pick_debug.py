"""真机抓碗候选、失败证据与已保存轨迹的只读 Viser 叠加层。"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np

from gene_agent.visualization.afford_scene import _wxyz


STATUS_COLORS = {
    "selected": (38, 184, 91),
    "rejected": (226, 74, 74),
    "evaluated": (50, 132, 214),
    "not_evaluated": (145, 151, 161),
}


def _add_line_segments_compat(scene: Any, name: str, **kwargs):
    """兼容不支持线宽参数的旧版 Viser，同时保留其他 TypeError。"""
    try:
        return scene.add_line_segments(name, **kwargs)
    except TypeError as error:
        message = str(error)
        unsupported_width = (
            "unexpected keyword argument 'thickness'" in message
            or 'unexpected keyword argument "thickness"' in message
            or "unexpected keyword argument 'thickness_units'" in message
            or 'unexpected keyword argument "thickness_units"' in message
        )
        if not unsupported_width:
            raise
        fallback = dict(kwargs)
        fallback.pop("thickness", None)
        fallback.pop("thickness_units", None)
        return scene.add_line_segments(name, **fallback)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 不是 JSON object")
    return value


def _selection_paths(run: Path, explicit: Path | None) -> list[Path]:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"抓取选择文件不存在：{path}")
        return [path]
    if not run.is_dir():
        return []
    paths = []
    for name in ("selection.json", "candidate_selection.json"):
        for path in run.rglob(name):
            relative = path.relative_to(run)
            if "initial_scene" not in relative.parts:
                paths.append(path.resolve())
    # selection.json 包含真机包装层的补充字段，优先于同目录 candidate_selection.json。
    return sorted(set(paths), key=lambda path: (path.name != "selection.json", str(path)))


def _failure_text(failure: dict | None) -> str:
    failure = failure or {}
    return str(failure.get("summary") or failure.get("message")
               or failure.get("backend_status") or failure.get("kind") or "未知失败")


def _trajectory_rows(value: Any, *, source: str, context: str = ""):
    if isinstance(value, dict):
        trajectory = value.get("trajectory")
        if isinstance(trajectory, dict) and trajectory.get("positions_rad"):
            label = str(value.get("name") or value.get("stage") or value.get("request_id")
                        or context or "trajectory")
            yield label, source, trajectory
        for key, child in value.items():
            if key == "trajectory":
                continue
            child_context = str(value.get("name") or value.get("stage") or context or key)
            yield from _trajectory_rows(child, source=source, context=child_context)
    elif isinstance(value, list):
        for child in value:
            yield from _trajectory_rows(child, source=source, context=context)


def _load_trajectories(run: Path, selection_path: Path, selection: dict) -> list[dict]:
    sources = [(selection_path, selection)]
    if run.is_dir():
        for path in sorted(run.rglob("*_result.json")):
            if "initial_scene" in path.relative_to(run).parts or path.resolve() == selection_path:
                continue
            try:
                sources.append((path, _read_json(path)))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    rows, seen = [], set()
    for path, value in sources:
        for label, source, trajectory in _trajectory_rows(
                value, source=str(path), context=path.stem):
            names = tuple(str(name) for name in trajectory.get("joint_names", ()))
            positions = np.asarray(trajectory.get("positions_rad"), dtype=float)
            if not names or positions.ndim != 2 or positions.shape[1] != len(names):
                continue
            times = np.asarray(trajectory.get("times_s", range(len(positions))), dtype=float)
            if times.shape != (len(positions),):
                times = np.arange(len(positions), dtype=float)
            digest = hashlib.sha256(
                json.dumps([names, positions.tolist()], separators=(",", ":")).encode()
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rows.append(dict(label=label, source=source, joint_names=names,
                             positions_rad=positions, times_s=times))
    return rows


def load_pick_debug(run_source, *, selection=None, scene_snapshot_sha256=None):
    """只读加载候选和轨迹；未记录的信息保持缺失，不重新求解。"""
    run = Path(run_source).expanduser().resolve()
    if run.is_file() and run.name == "manifest.json":
        run = run.parent.parent
    elif run.is_dir() and run.name == "initial_scene":
        run = run.parent
    explicit = None if selection is None else Path(selection)
    paths = _selection_paths(run, explicit)
    if not paths:
        return None
    selection_path = paths[0]
    saved = _read_json(selection_path)
    plan = saved.get("plan") if isinstance(saved.get("plan"), dict) else saved
    attempts = plan.get("attempts", [])
    attempt_by_candidate: dict[str, list[dict]] = {}
    request_targets = {}
    for attempt in attempts:
        candidate_id = attempt.get("candidate_id")
        if not candidate_id:
            continue
        attempt_by_candidate.setdefault(candidate_id, []).append(attempt)
        target = (attempt.get("request") or {}).get("target", {}).get("matrix")
        if target is not None:
            request_targets.setdefault(candidate_id, target)
    target_rows = (saved.get("candidate_parameters") or {}).get("candidate_targets") or []
    targets = {row.get("candidate_id"): row for row in target_rows if row.get("candidate_id")}
    order = list(saved.get("candidate_order") or plan.get("candidate_order") or targets)
    for candidate_id in attempt_by_candidate:
        if candidate_id not in order:
            order.append(candidate_id)
    selected = saved.get("selected_candidate_id") or plan.get("selected_candidate_id")
    candidates = []
    failure_counts = Counter()
    failure_stage_counts = Counter()
    for index, candidate_id in enumerate(order):
        row = targets.get(candidate_id, {})
        matrix = row.get("T_world_tcp") or request_targets.get(candidate_id)
        if matrix is None:
            continue
        pose = np.asarray(matrix, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            continue
        candidate_attempts = attempt_by_candidate.get(candidate_id, [])
        failed = next((attempt for attempt in candidate_attempts
                       if (attempt.get("result") or {}).get("success") is False), None)
        if candidate_id == selected:
            status = "selected"
        elif failed is not None:
            status = "rejected"
        elif candidate_attempts:
            status = "evaluated"
        else:
            status = "not_evaluated"
        failure = None if failed is None else (failed.get("result") or {}).get("failure") or {}
        if failure is not None:
            failure_counts[(str(failure.get("kind") or "unknown"),
                            str(failure.get("backend_status") or ""),
                            _failure_text(failure))] += 1
            failure_stage_counts[str(failed.get("stage") or "unknown")] += 1
        before = row.get("T_sdk_tcp_before_table_clamp")
        before_pose = None if before is None else np.asarray(before, dtype=float)
        candidates.append(dict(candidate_id=candidate_id, index=index, status=status,
            pose=pose, before_clamp_pose=(before_pose if before_pose is not None
                and before_pose.shape == (4, 4) and np.isfinite(before_pose).all() else None),
            table_clamp_delta_m=row.get("table_clamp_delta_sdk_z_m"),
            attempts=candidate_attempts, first_failure_attempt=failed, failure=failure))
    trajectories = _load_trajectories(run, selection_path, saved)
    snapshot_ids = {str(value) for value in (
        saved.get("input_state_id"), saved.get("base_collision_snapshot_sha256"),
        saved.get("stage_collision_snapshot_sha256")) if value}
    snapshot_match = (None if not scene_snapshot_sha256 or not snapshot_ids else
                      str(scene_snapshot_sha256) in snapshot_ids)
    top_failures = [dict(kind=key[0], backend_status=key[1], reason=key[2], count=count)
                    for key, count in failure_counts.most_common(8)]
    clamp_values = [float(row["table_clamp_delta_m"]) for row in candidates
                    if row.get("table_clamp_delta_m") is not None]
    if trajectories:
        no_trajectory_reason = None
    elif failure_stage_counts:
        stage, count = failure_stage_counts.most_common(1)[0]
        no_trajectory_reason = (
            f"没有生成关节轨迹：{count} 个候选在 {stage} 阶段失败，运行未进入轨迹规划。")
    else:
        no_trajectory_reason = "没有生成关节轨迹：运行未进入轨迹规划，或旧记录未保存轨迹。"
    return dict(
        selection_path=str(selection_path), available_selection_paths=[str(path) for path in paths],
        candidate_count=len(candidates), selected_candidate_id=selected,
        status_counts=dict(Counter(row["status"] for row in candidates)),
        candidates=candidates, trajectories=trajectories,
        trajectory_count=len(trajectories), failure_stage_counts=dict(failure_stage_counts),
        top_failures=top_failures, snapshot_ids=sorted(snapshot_ids),
        scene_snapshot_sha256=scene_snapshot_sha256, snapshot_match=snapshot_match,
        clamp_summary=(None if not clamp_values else dict(count=len(clamp_values),
            min_m=min(clamp_values), max_m=max(clamp_values), mean_m=float(np.mean(clamp_values)))),
        no_trajectory_reason=no_trajectory_reason,
    )


def pick_debug_summary(debug: dict | None) -> dict:
    if debug is None:
        return dict(available=False, message="本运行未找到抓取 selection.json")
    return dict(available=True, selection_path=debug["selection_path"],
        candidate_count=debug["candidate_count"], status_counts=debug["status_counts"],
        selected_candidate_id=debug["selected_candidate_id"],
        failure_stage_counts=debug["failure_stage_counts"],
        top_failures=debug["top_failures"], trajectory_count=debug["trajectory_count"],
        no_trajectory_reason=debug["no_trajectory_reason"],
        clamp_summary=debug["clamp_summary"], snapshot_match=debug["snapshot_match"])


class RealworldPickDebugOverlay:
    """将已保存的候选、失败和轨迹叠加到冻结现场。"""

    def __init__(self, server: Any, model: Any, data: Any, scene: dict,
                 renderer: Any, debug: dict):
        self.server, self.model, self.data = server, model, data
        self.scene, self.renderer, self.debug = scene, renderer, debug
        self.base_qpos = np.asarray(data.qpos).copy()
        self._playing = False
        self._last_tick = time.monotonic()
        self._trajectory_line = None
        self._candidate_frame = server.scene.add_frame(
            "/realworld_pick/current_candidate", axes_length=.065, axes_radius=.0025)
        self._candidate_label = server.scene.add_label(
            "/realworld_pick/current_candidate/label", text="candidate", position=(0., 0., 0.))
        self._build_candidate_geometry()
        with server.gui.add_folder("抓碗失败定位 · AffordToolAct 粒度"):
            warning = ""
            if debug["snapshot_match"] is False:
                warning = ("\n\n⚠️ **selection 与当前 initial_scene 的快照哈希不一致。** "
                           "可查看几何关系，但不能声称它们来自同一轮现场。")
            server.gui.add_markdown(
                f"候选 **{debug['candidate_count']}**；状态 `{debug['status_counts']}`；"
                f"轨迹 **{debug['trajectory_count']}**。{warning}")
            self.candidate_control = server.gui.add_dropdown(
                "候选 TCP", options=tuple(row["candidate_id"] for row in debug["candidates"]),
                initial_value=debug["selected_candidate_id"] or debug["candidates"][0]["candidate_id"])
            self.candidate_detail = server.gui.add_markdown("")
            show_axes = server.gui.add_checkbox("显示全部候选抓取轴", initial_value=True)
            show_clamp = server.gui.add_checkbox("显示桌面抬升修正线", initial_value=True)
            show_axes.on_update(lambda _: self._set_visible(self._axis_handles, show_axes.value))
            show_clamp.on_update(lambda _: self._set_visible(self._clamp_handles, show_clamp.value))
            self.trajectory_control = server.gui.add_dropdown(
                "已保存轨迹", options=tuple(row["label"] for row in debug["trajectories"]) or ("无轨迹",),
                initial_value=(debug["trajectories"][0]["label"] if debug["trajectories"] else "无轨迹"),
                disabled=not debug["trajectories"])
            self.frame_control = server.gui.add_slider(
                "轨迹帧", min=0, max=max(1, self._trajectory_length() - 1), step=1,
                initial_value=0, disabled=not debug["trajectories"])
            self.play_button = server.gui.add_button("播放 / 暂停", disabled=not debug["trajectories"])
            self.rewind_button = server.gui.add_button("回到轨迹起点", disabled=not debug["trajectories"])
            self.playback_detail = server.gui.add_markdown(
                debug["no_trajectory_reason"] or "蓝线为 TCP 正运动学路径；机器人为已保存关节轨迹。")
        self.candidate_control.on_update(lambda _: self._show_candidate())
        self.trajectory_control.on_update(lambda _: self._select_trajectory())
        self.frame_control.on_update(lambda _: self._show_trajectory_frame())
        self.play_button.on_click(lambda _: self._toggle_play())
        self.rewind_button.on_click(lambda _: self._rewind())
        self._show_candidate()
        if debug["trajectories"]:
            self._select_trajectory()

    @staticmethod
    def _set_visible(handles, visible):
        for handle in handles:
            handle.visible = bool(visible)

    def _build_candidate_geometry(self):
        self._axis_handles, self._clamp_handles = [], []
        for status, color in STATUS_COLORS.items():
            rows = [row for row in self.debug["candidates"] if row["status"] == status]
            if rows:
                self.server.scene.add_point_cloud(
                    f"/realworld_pick/candidates/{status}",
                    points=np.asarray([row["pose"][:3, 3] for row in rows]),
                    colors=color, point_size=.009)
        axis_segments, clamp_segments = [], []
        for row in self.debug["candidates"]:
            pose = row["pose"]
            axis_segments.append(np.stack((pose[:3, 3] - .055 * pose[:3, 2], pose[:3, 3])))
            before = row["before_clamp_pose"]
            if before is not None and not np.allclose(before[:3, 3], pose[:3, 3]):
                clamp_segments.append(np.stack((before[:3, 3], pose[:3, 3])))
        if axis_segments:
            self._axis_handles.append(_add_line_segments_compat(
                self.server.scene,
                "/realworld_pick/approach_axes", points=np.asarray(axis_segments),
                colors=(116, 182, 255), thickness=.0015, thickness_units="world"))
        if clamp_segments:
            self._clamp_handles.append(_add_line_segments_compat(
                self.server.scene,
                "/realworld_pick/table_clamp", points=np.asarray(clamp_segments),
                colors=(255, 158, 42), thickness=.002, thickness_units="world"))

    def _candidate(self):
        selected = self.candidate_control.value
        return next(row for row in self.debug["candidates"] if row["candidate_id"] == selected)

    def _show_candidate(self):
        row = self._candidate()
        pose = row["pose"]
        self._candidate_frame.wxyz = _wxyz(pose)
        self._candidate_frame.position = tuple(pose[:3, 3])
        self._candidate_label.position = tuple(pose[:3, 3] + np.array([0., 0., .035]))
        self._candidate_label.text = row["candidate_id"]
        failed = row["first_failure_attempt"] or {}
        failure = row["failure"] or {}
        clamp = row.get("table_clamp_delta_m")
        detail = (
            f"**{row['candidate_id']}** · `{row['status']}` · 顺序 {row['index']}\n\n"
            f"首个失败阶段：`{failed.get('stage', '未失败/未评估')}`\n\n"
            f"失败类型：`{failure.get('kind', '')}` / `{failure.get('backend_status', '')}`\n\n"
            f"原因：{_failure_text(failure) if failure else '无已保存失败'}\n\n")
        detail += (f"桌面抬升修正：{float(clamp) * 1000:.2f} mm" if clamp is not None
                   else "未记录桌面抬升修正。")
        self.candidate_detail.content = detail

    def _trajectory(self):
        label = self.trajectory_control.value
        return next((row for row in self.debug["trajectories"] if row["label"] == label), None)

    def _trajectory_length(self):
        row = self._trajectory() if hasattr(self, "trajectory_control") else (
            self.debug["trajectories"][0] if self.debug["trajectories"] else None)
        return 0 if row is None else len(row["positions_rad"])

    def _tcp_points(self, trajectory):
        site_name = self.scene["robot_metadata"]["tcp_sites"][self.scene["snapshot"]["active_group"]]
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.base_qpos
        points = []
        for positions in trajectory["positions_rad"]:
            for name, value in zip(trajectory["joint_names"], positions, strict=True):
                scratch.qpos[self.model.joint(name).qposadr[0]] = float(value)
            mujoco.mj_forward(self.model, scratch)
            points.append(np.asarray(scratch.site(site_name).xpos).copy())
        return np.asarray(points)

    def _select_trajectory(self):
        trajectory = self._trajectory()
        if trajectory is None:
            return
        self._playing = False
        self.frame_control.max = max(1, len(trajectory["positions_rad"]) - 1)
        self.frame_control.value = 0
        if self._trajectory_line is not None:
            self._trajectory_line.remove()
            self._trajectory_line = None
        points = self._tcp_points(trajectory)
        if len(points) >= 2:
            self._trajectory_line = _add_line_segments_compat(
                self.server.scene,
                "/realworld_pick/tcp_path", points=np.stack((points[:-1], points[1:]), axis=1),
                colors=(31, 135, 212), thickness=.004, thickness_units="world")
        self.playback_detail.content = (
            f"轨迹 `{trajectory['label']}`：{len(points)} 帧；来源 `{trajectory['source']}`。\n\n"
            "蓝线为 TCP 正运动学路径。碰撞代理和蓝色碰撞球是规划起点快照；"
            "轨迹第 1 帧后自动隐藏碰撞球，避免误读为逐帧碰撞结果。")
        self._show_trajectory_frame()

    def _show_trajectory_frame(self):
        trajectory = self._trajectory()
        if trajectory is None:
            return
        index = min(int(self.frame_control.value), len(trajectory["positions_rad"]) - 1)
        self.data.qpos[:] = self.base_qpos
        for name, value in zip(trajectory["joint_names"], trajectory["positions_rad"][index], strict=True):
            self.data.qpos[self.model.joint(name).qposadr[0]] = float(value)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_visuals(self.data)
        self.renderer.set_robot_spheres_visible(index == 0)

    def _toggle_play(self):
        self._playing = not self._playing
        self._last_tick = time.monotonic()

    def _rewind(self):
        self._playing = False
        self.frame_control.value = 0

    def tick(self):
        if not self._playing or not self.debug["trajectories"]:
            return
        now = time.monotonic()
        if now - self._last_tick < .08:
            return
        self._last_tick = now
        length = self._trajectory_length()
        next_index = int(self.frame_control.value) + 1
        if next_index >= length:
            next_index = 0
        self.frame_control.value = next_index
