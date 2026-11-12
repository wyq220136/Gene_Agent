"""只读 Skill 回放时间轴。规划轨迹与 MuJoCo 实际采样分开读取。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from gene_agent.feedback.skill_run import SkillRunView, StageView, load_skill_run


@dataclass(frozen=True)
class ReplayFrame:
    stage_id: str
    mode: str
    global_time_s: float
    local_time_s: float
    sample_time_s: float
    sample_quality: str
    joint_names: tuple[str, ...]
    joint_positions_rad: tuple[float, ...] | None
    gripper_width_m: float | None
    object_poses_world: dict[str, Any] | None
    state_id: str | None
    state_ref: str | None
    collision_snapshot_ref: str | None
    path_kind: str | None
    attachment: dict[str, Any] | None
    qpos: tuple[float, ...] | None = None
    tcp_poses_world: dict[str, Any] | None = None
    scene_revision: int | None = None
    sim_time_s: float | None = None


@dataclass(frozen=True)
class _Sample:
    time_s: float
    quality: str
    joints: tuple[str, ...]
    positions: tuple[float, ...] | None
    width: float | None = None
    objects: dict[str, Any] | None = None
    state_id: str | None = None
    state_ref: str | None = None
    attachment: dict[str, Any] | None = None
    qpos: tuple[float, ...] | None = None
    tcp_poses_world: dict[str, Any] | None = None
    scene_revision: int | None = None
    sim_time_s: float | None = None


@dataclass(frozen=True)
class _Segment:
    stage: StageView
    start_s: float
    duration_s: float
    samples: tuple[_Sample, ...]


class ReplayTimeline:
    """同一份阶段/采样索引供 Viser 与 MuJoCo viewer 使用。

    seek 只计算和返回不可变帧，不会把回放状态写入 live runtime。
    新录制的 replay_samples 保存每个采样步的真实完整状态；旧录制的
    samples 只含部分机器人反馈，物体 pose 和 state ID 继续标为未知。
    """

    def __init__(self, root: Path, view: SkillRunView, *, recorded_samples=None) -> None:
        self.root = root
        self.view = view
        self._recorded_samples = recorded_samples
        self._selected_stage: str | None = None
        self._mode = ("actual" if recorded_samples is not None else
                      "actual" if view.source_format in {"legacy_execution_probe", "skill_execution"}
                      and all(stage.execution_status == "completed" for stage in view.stages)
                      else "planned")
        self._segments: tuple[_Segment, ...] = ()
        self.select(mode=self._mode)

    @classmethod
    def load(cls, run_dir: str | Path, *, skill: str | None = None) -> ReplayTimeline:
        root = Path(run_dir).resolve()
        from gene_agent.feedback.microwave_actual_run import load_microwave_actual_run
        recorded = load_microwave_actual_run(root, skill=skill)
        if recorded is not None:
            view, samples = recorded
            return cls(root, view, recorded_samples=samples)
        from gene_agent.feedback.replay_adapters import load_tabletop_actual_run
        recorded = load_tabletop_actual_run(root, skill=skill)
        if recorded is not None:
            view, samples = recorded
            return cls(root, view, recorded_samples=samples)
        return cls(root, load_skill_run(root, skill=skill))

    def _read(self, relative: str) -> dict[str, Any]:
        value = json.loads((self.root / relative).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"回放产物不是JSON对象：{relative}")
        return value

    def _observation(self, relative: str, time_s: float) -> _Sample:
        row = self._read(relative)
        robots = row.get("robots", {})
        group = next(iter(robots), None)
        robot = robots.get(group, {})
        objects = {name: obj.get("T_world_object") for name, obj in row.get("objects", {}).items()}
        gripper = robot.get("gripper", {})
        return _Sample(time_s, "boundary_state", tuple(robot.get("joint_names", ())),
                       tuple(robot.get("joint_positions_rad", ())) if group else None,
                       gripper.get("width_m"), objects, row.get("state_id"), relative,
                       row.get("attachment"),
                       tuple(row["qpos"]) if "qpos" in row else None,
                       {name: value.get("T_world_tcp") for name, value in robots.items()},
                       row.get("scene_revision"), row.get("sim_time_s"))

    def _planned_samples(self, stage: StageView) -> tuple[_Sample, ...]:
        if stage.result_ref is None:
            return (_Sample(0.0, "stage_marker", (), None),)
        report = self._read(stage.result_ref)
        if self.view.source_format == "skill_plan":
            index = int(stage.stage_id.split("_", 1)[0])
            row = report["stages"][index]
            # 规划记录的开/闭爪是零时长状态事件。回放任意单独阶段时仍
            # 必须从前面最近一次夹爪命令恢复宽度，否则随机拖动时间轴
            # 会沿用上一帧的宽度，误判是否真正夹住物体。
            widths = [
                float(previous["details"]["width_m"])
                for previous in report["stages"][:index + 1]
                if "width_m" in (previous.get("details") or {})
            ]
            width = widths[-1] if widths else None
            if width is None and self.view.skill == "insert":
                # 放入 Skill 从上一个 Pick 的闭爪终点接续，规划阶段也要沿用闭爪宽度。
                pick = self._read("skill_pick_bowl_result.json")
                prior_widths = [
                    float(previous["details"]["width_m"])
                    for previous in pick["stages"]
                    if "width_m" in (previous.get("details") or {})
                ]
                width = prior_widths[-1] if prior_widths else None
            objects = ({self.view.instance_id: row["object_pose_world"]}
                       if row.get("object_pose_world") is not None else None)
            preview = row.get("planned_replay_samples") or []
            if preview:
                # 旋钮运动学回放保存完整 qpos；机器人关节来自 cuRobo 轨迹，
                # 旋钮关节仅是明确标记的预览跟随，不能被当作实际 MuJoCo 积分。
                return tuple(_Sample(
                    float(sample["time_s"]), "planned_kinematic_preview",
                    tuple(sample["joint_names"]), tuple(sample["joint_positions_rad"]),
                    float(sample["gripper_width_m"]), sample.get("object_poses_world"),
                    qpos=tuple(sample["qpos"]),
                    tcp_poses_world={sample["robot_group"]: sample["tcp_pose_world"]},
                ) for sample in preview)
            request, result = row.get("request") or {}, row.get("result") or {}
            trajectory = result.get("trajectory") or {}
            names = tuple(trajectory.get("joint_names") or request.get("start", {}).get("joint_names") or ())
            times = trajectory.get("times_s") or []
            positions = trajectory.get("positions_rad") or []
            if len(times) != len(positions):
                raise ValueError(f"阶段{stage.stage_id}轨迹时间与关节采样数不一致")
            if positions:
                return tuple(_Sample(float(t), "planned_trajectory", names, tuple(q), width, objects)
                             for t, q in zip(times, positions))
            start = request.get("start") or {}
            if not start:
                # 开/闭爪与 hold 是状态事件，不能因为它们没有自己的路径
                # 就回到 home。独立查看 close 时应停在 approach 的实际
                # 规划终点；独立查看 hold 时应停在 lift 的终点。
                for previous in reversed(report["stages"][:index]):
                    previous_result = previous.get("result") or {}
                    endpoint = previous_result.get("goal_state") or {}
                    if endpoint.get("joint_names") and endpoint.get("positions_rad"):
                        start = endpoint
                        break
                if not start:
                    for following in report["stages"][index + 1:]:
                        next_start = (following.get("request") or {}).get("start") or {}
                        if next_start.get("joint_names") and next_start.get("positions_rad"):
                            start = next_start
                            break
            names = tuple(start.get("joint_names") or names)
            return (_Sample(0.0, "stage_marker", names,
                            tuple(start["positions_rad"]) if "positions_rad" in start else None,
                            width, objects),)
        names = tuple(report.get("joint_names") or ())
        if not report.get("success"):
            return (_Sample(0.0, "failed_stage_marker", names, None),)
        positions = report.get("positions_rad") or []
        dt = float(report.get("interpolation_dt_s") or 0.0)
        if positions and dt <= 0:
            raise ValueError(f"旧阶段{stage.stage_id}缺少规划采样间隔")
        return tuple(_Sample(i * dt, "planned_trajectory", names, tuple(q))
                     for i, q in enumerate(positions)) or (_Sample(0.0, "stage_marker", names, None),)

    def _actual_samples(self, stage: StageView) -> tuple[_Sample, ...]:
        if self._recorded_samples is not None:
            rows = self._recorded_samples[stage.stage_id]
            start = float(rows[0]['sim_time_s'])
            samples = []
            for row in rows:
                robots = row.get('robots') or {}
                robot = next(iter(robots.values()), {})
                samples.append(_Sample(
                    float(row['sim_time_s']) - start, row.get('sample_quality', 'complete_actual_sample'),
                    tuple(robot.get('joint_names', ())),
                    tuple(robot['joint_positions_rad']) if 'joint_positions_rad' in robot else None,
                    robot.get('gripper', {}).get('width_m'),
                    {key: obj['T_world_object'] for key, obj in row.get('objects', {}).items()},
                    row.get('state_id'), attachment=row.get('attachment'), qpos=tuple(row['qpos']),
                    tcp_poses_world={key: robot.get('T_world_tcp') for key, robot in robots.items()},
                    scene_revision=row.get('scene_revision'), sim_time_s=float(row['sim_time_s'])))
            return tuple(samples)
        if stage.execution_status != "completed":
            raise ValueError(f"阶段{stage.stage_id}没有可回放的实际执行")
        if stage.start_state_ref is None or stage.end_state_ref is None:
            raise ValueError(f"阶段{stage.stage_id}缺少实际状态边界")
        start = self._observation(stage.start_state_ref, 0.0)
        end_data = self._read(stage.end_state_ref)
        duration = float(end_data["sim_time_s"] - self._read(stage.start_state_ref)["sim_time_s"])
        if duration < 0:
            raise ValueError(f"阶段{stage.stage_id}实际时间倒退")
        execution = self._read(stage.execution_ref) if stage.execution_ref else {}
        if stage.execution_ref == "result.json":
            execution = execution["stages"][stage.name]
        intermediate = []
        if execution.get("replay_samples"):
            start_time = float(self._read(stage.start_state_ref)["sim_time_s"])
            for row in execution["replay_samples"]:
                required = {"state_id", "sim_time_s", "scene_revision", "qpos", "objects", "robots"}
                if not required.issubset(row):
                    raise ValueError(f"阶段{stage.stage_id}完整实际采样缺少{sorted(required - row.keys())}")
                t = float(row["sim_time_s"]) - start_time
                if not 0 < t < duration:
                    continue
                robots = row["robots"]
                group = next(iter(robots), None)
                if group is None or not row["state_id"] or not row["qpos"]:
                    raise ValueError(f"阶段{stage.stage_id}完整实际采样缺少机器人或状态")
                robot = robots[group]
                intermediate.append(_Sample(
                    t, "complete_actual_sample", tuple(robot["joint_names"]),
                    tuple(robot["joint_positions_rad"]),
                    robot.get("gripper", {}).get("width_m"),
                    {name: obj["T_world_object"] for name, obj in row["objects"].items()},
                    row["state_id"], None, row.get("attachment"),
                    tuple(row["qpos"]),
                    {name: value["T_world_tcp"] for name, value in robots.items()},
                    row["scene_revision"], row["sim_time_s"],
                ))
        else:
            for row in execution.get("samples", []):
                t = float(row["time_s"])
                if not 0 < t < duration:
                    continue
                positions = row.get("actual_positions_rad")
                intermediate.append(_Sample(
                    t, "partial_actual_sample", tuple(execution.get("joint_names") or start.joints),
                    tuple(positions) if positions is not None else None,
                    row.get("actual_width_m"), None,
                ))
        end = self._observation(stage.end_state_ref, duration)
        return (start, *intermediate, end)

    def select(self, stage_id: str | None = None, mode: str = "actual") -> ReplayTimeline:
        """切换全Skill/单阶段及事实来源；单阶段时间从零开始。"""
        if mode not in {"actual", "planned"}:
            raise ValueError("mode只能是actual或planned")
        stages = self.view.stages if stage_id is None else tuple(
            stage for stage in self.view.stages if stage.stage_id == stage_id)
        if not stages:
            raise KeyError(f"未知Skill阶段：{stage_id}")
        if self._recorded_samples is not None and mode != 'actual':
            raise ValueError('实际记录没有对应规划时间轴；请打开独立规划记录')
        if self._recorded_samples is None and mode == "actual" and any(stage.execution_status != "completed" for stage in stages):
            raise ValueError("所选阶段没有实际执行采样，不能切换为actual")
        segments = []
        offset = 0.0
        for index, stage in enumerate(stages):
            samples = self._actual_samples(stage) if mode == "actual" else self._planned_samples(stage)
            if any(b.time_s < a.time_s for a, b in zip(samples, samples[1:])):
                raise ValueError(f"阶段{stage.stage_id}采样时间倒退")
            duration = max(samples[-1].time_s, 0.001)
            if self._recorded_samples is not None:
                duration = max(samples[-1].time_s, 0.)
            if self._recorded_samples is not None and index + 1 < len(stages):
                next_start = self._recorded_samples[stages[index + 1].stage_id][0]['sim_time_s']
                duration = max(float(next_start) - float(
                    self._recorded_samples[stage.stage_id][0]['sim_time_s']), .001)
            segments.append(_Segment(stage, offset, duration, samples))
            offset += duration
        self._selected_stage, self._mode, self._segments = stage_id, mode, tuple(segments)
        return self

    @property
    def duration_s(self) -> float:
        last = self._segments[-1]
        return last.start_s + last.duration_s

    @property
    def mode(self) -> str:
        return self._mode

    def seek(self, time_s: float) -> ReplayFrame:
        """取不晚于请求时刻的真实采样；绝不插值伪造物体状态。"""
        if not 0 <= time_s <= self.duration_s:
            raise ValueError(f"回放时间超出范围：{time_s}，总长{self.duration_s}")
        segment = next((s for s in self._segments
                        if time_s < s.start_s + s.duration_s), self._segments[-1])
        local = min(time_s - segment.start_s, segment.duration_s)
        sample = next((s for s in reversed(segment.samples) if s.time_s <= local + 1e-9),
                      segment.samples[0])
        attachment = (self._read(segment.stage.attachment_ref)
                      if segment.stage.attachment_ref else sample.attachment)
        return ReplayFrame(
            segment.stage.stage_id, self._mode, time_s, local, sample.time_s,
            sample.quality, sample.joints, sample.positions, sample.width,
            sample.objects, sample.state_id, sample.state_ref,
            segment.stage.collision_snapshot_ref, segment.stage.path_kind,
            attachment,
            sample.qpos, sample.tcp_poses_world, sample.scene_revision,
            sample.sim_time_s,
        )
