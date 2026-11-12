"""Bowl 在线视觉会话：重建 GT 场景，并转换真实 Skill 规划结果。

本层只适配已有应用规划器，不修改候选、碰撞策略或 IK 成功判定。
attachment 表示纯视觉规划状态，不能作为物理夹持成功的证据。
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from gene_agent.visualization.viser_sim.model import SceneEdits, VisualRun, VisualStage
from gene_agent.visualization.viser_sim.playback import VisualPlayback, VisualState
from gene_agent.visualization.skill_inspector.planned_timeline import PlannedSegment
from gene_agent.tasks.policies.microwave_task.planning.bowl import prepare_pick_bowl_scene, _body_pose

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / 'data/task_1/visualization/bowl_visual_debug.json'
OBJECT_SKILLS = {'pick_bowl': '抓碗', 'insert_bowl': '放入碗'}
OBJECT_TASKS = {'microwave_object_insert': '抓碗 → 放入微波炉'}


def object_visual_run(report: dict, selection: str, root: Path, *,
                      initial_attachments: tuple[str, ...] = ()) -> VisualRun:
    """保留阶段顺序、失败前缀、真实轨迹以及显式 attach/detach 事实。"""
    root = Path(root).resolve()
    contracts = {row['name']: row['operation'] for row in report['stage_contract']}
    attached = initial_attachments
    stages = []
    object_id = report.get('instance_id', 'bowl')
    for row in report['stages']:
        name = row['name']
        operation = contracts[name]
        result = row.get('result') or {}
        details = row.get('details') or {}
        success = row['status'] == 'planned' and result.get('success', True)
        before = attached
        # Pick 的既有 adapter 在闭爪的实际 FK 处建立 attachment。未获得
        # attachment 证据时不能仅凭阶段名称让碗跟随夹爪。
        if success and name == 'close' and report.get('visual_attachment'):
            attached = (object_id,)
        if success and name == 'detach':
            attached = ()
        if success and name == 'attach':
            attached = (object_id,)
        reference = row.get('collision_snapshot_ref')
        if reference is None:
            reference = ((report.get('visual_attachment') or {}).get('collision_snapshot_ref')
                         if before else None) or 'collision/collision_snapshot.json'
        path = Path(reference)
        if not path.is_absolute():
            path = root / path
        snapshot = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        enabled = tuple(str(record['obstacle_name']) for record in snapshot.get('records', [])
                        if record.get('obstacle_name'))
        allowed = tuple(str(record.get('object_instance_id') or record.get('body_name'))
                        for record in snapshot.get('excluded', [])
                        if (record.get('object_instance_id') or record.get('body_name'))
                        and record.get('reason') not in {
                            'contact_masks_both_zero', 'active_group_uses_original_robot_collision_spheres',
                            'transferred_to_attached_robot_proxy'})
        kind = {'gripper_command': 'gripper', 'hold': 'attachment'}.get(operation, operation)
        trajectory = None
        if success and kind in {'plan', 'linear_tcp', 'joint'}:
            raw = result.get('trajectory') or {}
            times = np.asarray(raw.get('times_s', []), dtype=float)
            if len(times) < 2 or not np.allclose(np.diff(times), times[1] - times[0], atol=1e-7):
                raise ValueError(f'{selection}/{name} 缺少等间隔真实规划轨迹，拒绝补造')
            trajectory = PlannedSegment(
                stage_id=f'{selection}/{name}', skill_id=selection,
                joint_names=tuple(raw['joint_names']),
                positions_rad=np.asarray(raw['positions_rad']),
                sample_dt_s=float(times[1] - times[0]), channels={}, details=result,
            )
        target = (row.get('target') or {}).get('matrix')
        failure = None if success else (result.get('failure') or {
            'category': row['status'], 'message': details.get('reason', '该阶段未完成规划'),
            'attempts': report.get('attempts', []),
        })
        stages.append(VisualStage(
            stage_id=f'{selection}/{name}', skill_id=selection, motion_kind=kind,
            success=bool(success), trajectory=trajectory,
            target_tcp_world=None if target is None else np.asarray(target),
            object_instance_id=object_id,
            object_pose_world=None if row.get('object_pose_world') is None else np.asarray(row['object_pose_world']),
            collision_enabled=enabled, collision_allowed=allowed,
            collision_snapshot_ref=str(path) if path.is_file() else None,
            attach_before=before, attach_after=attached,
            gripper_command=name if kind == 'gripper' else None,
            gripper_width_m=details.get('width_m'),
            state_duration_s=(float(details.get('duration_s', .25)) if kind in {'gripper', 'attachment'} else None),
            planning_duration_s=result.get('duration_s'), failure=failure,
            diagnostics={'source_status': row['status'], 'details': details,
                         'metrics': result.get('metrics', {}),
                         'collision_scope': snapshot.get('scope', '未提供'),
                         'attachment_evidence': report.get('visual_attachment'),
                         'visual_only': True},
        ))
        if not success:
            break
    return VisualRun(selection, int(report['scene_revision']), tuple(stages))


class ObjectVisualSession:
    """一个可重建、可重新规划的 Skill 场景；同一配置生成唯一 GT 和碰撞快照。"""

    def __init__(self, *, output_dir: Path | None = None, config_path: Path | None = None):
        self.output_dir = Path(output_dir or ROOT / 'runs' / f'object-online-{datetime.now():%Y%m%d-%H%M%S-%f}').resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = json.loads(Path(config_path or DEFAULT_CONFIG).read_text(encoding='utf-8'))
        self._plan_state = None
        self.run = None
        self.playback = None
        self.prerequisite_report = None
        self.case = self._build(self.config, 0)
        self.initial = self._initial_state()

    def _build(self, config: dict, revision: int):
        return prepare_pick_bowl_scene(
            self.output_dir / f'revision_{revision:03d}', revision=revision,
            seed=int(config.get('seed', 123)),
            pose_overrides={k: np.asarray(v) for k, v in config.get('object_poses_world', {}).items()},
            joint_overrides=config.get('object_joint_positions_rad', {}),
            table_size_xy_m=config.get('table_size_xy_m'),
        )

    def _initial_state(self) -> VisualState:
        case = self.case
        robot = case.metadata['robot']
        names = tuple(robot['joint_groups']['right'])
        positions = np.asarray([float(case.data.qpos[case.model.joint(n).qposadr[0]]) for n in names])
        width = sum(float(case.data.qpos[case.model.joint(n).qposadr[0]])
                    for n in robot['grippers']['right']['joint_names'])
        return VisualState(names, positions, width, {'bowl': _body_pose(case.data, 'bowl')},
                           float(case.data.qpos[case.model.joint('microwave_door_joint').qposadr[0]]))

    def tcp_pose(self, names, positions):
        scratch = mujoco.MjData(self.case.model)
        scratch.qpos[:] = self.case.data.qpos
        for name, value in zip(names, positions, strict=True):
            scratch.qpos[self.case.model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(self.case.model, scratch)
        site = scratch.site(self.case.metadata['robot']['tcp_sites']['right'])
        pose = np.eye(4)
        pose[:3, :3] = site.xmat.reshape(3, 3)
        pose[:3, 3] = site.xpos
        return pose

    def apply_edits(self, edits: SceneEdits) -> None:
        """整体验证新场景后再提交，避免坏输入破坏当前可用场景。"""
        if edits.robot_joint_positions_rad:
            raise ValueError('Bowl 在线入口暂不支持编辑机器人初始关节')
        config = json.loads(json.dumps(self.config))
        config.setdefault('object_poses_world', {}).update({k: v.tolist() for k, v in edits.object_poses_world.items()})
        config.setdefault('object_joint_positions_rad', {}).update(edits.object_joint_positions_rad)
        if edits.table_size_xy_m is not None:
            config['table_size_xy_m'] = list(edits.table_size_xy_m)
        # 失败的构建目录保留诊断；下次使用未占用的 revision，绝不覆盖历史。
        revision = self.case.revision + 1
        while (self.output_dir / f'revision_{revision:03d}').exists():
            revision += 1
        case = self._build(config, revision)
        self.config, self.case = config, case
        self.initial = self._initial_state()
        self.run = self.playback = self.prerequisite_report = None
        self._plan_state = None
        self.save_config()

    def save_config(self) -> Path:
        path = self.output_dir / 'scene_config.json'
        path.write_text(json.dumps(self.config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return path

    def begin_plan(self, selection: str) -> dict:
        """只准备同步规划状态；GPU 计算由后续 tick 的 plan_step 调用。"""
        if selection not in OBJECT_SKILLS and selection not in OBJECT_TASKS:
            raise ValueError(f'未接入在线规划的 Skill：{selection}')
        if self.run is not None:
            self.apply_edits(SceneEdits())
        self._plan_state = {'selection': selection, 'step': 'pick',
                            'initial': self._initial_state(), 'task_initial': self._initial_state()}
        return {'done': False, 'message': ('正在准备抓碗' if selection == 'pick_bowl'
                else '正在准备同场景 Pick，作为放入的合法初态')}

    def cancel_plan(self) -> None:
        """在相邻同步步骤之间取消；不尝试中断单次 cuRobo 求解。"""
        self._plan_state = None

    def _finish_run(self, run: VisualRun, initial: VisualState) -> dict:
        self.run = run
        self.initial = initial
        self.playback = VisualPlayback(run, initial, self.tcp_pose)
        self._plan_state = None
        (self.case.output_dir / 'visual_run.json').write_text(
            json.dumps(run.manifest(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        self.save_config()
        return {'done': True, 'run': run}

    def plan_step(self) -> dict:
        """每次最多执行一个已有 Skill adapter，阶段之间允许界面刷新。"""
        from gene_agent.tasks.policies.microwave_task.planning.grasp import run_pick_bowl_skill_planning
        from gene_agent.tasks.policies.microwave_task.planning.insert import run_insert_bowl_skill_planning
        if self._plan_state is None:
            raise RuntimeError('请先 begin_plan')
        state = self._plan_state
        selection = state['selection']
        initial = state['initial']
        if state['step'] == 'pick':
            pick = run_pick_bowl_skill_planning(self.case, seed=int(self.config.get('seed', 123)))
            pick_run = object_visual_run(pick, 'pick_bowl', self.case.output_dir)
            if selection == 'pick_bowl':
                return self._finish_run(pick_run, initial)
            self.prerequisite_report = pick
            if not pick_run.success:
                if selection in OBJECT_TASKS:
                    return self._finish_run(VisualRun(selection, self.case.revision, pick_run.stages), initial)
                failed = VisualStage('insert_bowl/prerequisite_pick', selection, 'attachment', False,
                    failure={'category': 'prerequisite_failed',
                             'message': '同场景 Pick 前置规划失败，放入未执行',
                             'pick_failure': pick_run.stages[-1].failure})
                return self._finish_run(VisualRun(selection, self.case.revision, (failed,)), initial)
            playback = VisualPlayback(pick_run, initial, self.tcp_pose)
            state['pick_run'] = pick_run
            state['initial'] = playback.sample(playback.duration_s).state
            state['step'] = 'insert'
            return {'done': False,
                    'message': 'Pick 前置成功；下一步从其实际终态与 attachment 规划放入',
                    'partial_run': (VisualRun(selection, self.case.revision, pick_run.stages)
                                    if selection in OBJECT_TASKS else pick_run),
                    'partial_playback': playback}
        report = run_insert_bowl_skill_planning(self.case, seed=int(self.config.get('seed', 123)))
        run = object_visual_run(report, 'insert_bowl', self.case.output_dir,
                               initial_attachments=initial.attachments)
        if selection in OBJECT_TASKS:
            # 保留同一场景、同一 Pick 结果的原始阶段；VisualRun 验证真实轨迹衔接，
            # VisualPlayback 从最初未抓取状态重放所有 attach/detach 事件。
            run = VisualRun(selection, self.case.revision, state['pick_run'].stages + run.stages)
            initial = state['task_initial']
        return self._finish_run(run, initial)

    def plan(self, selection: str, progress: Callable[[str], None] | None = None) -> VisualRun:
        """CLI 同步包装，和 Viser tick 复用相同的逐 Skill 状态转移。"""
        notify = progress or (lambda message: None)
        report = self.begin_plan(selection)
        while not report['done']:
            notify(report.get('message', '正在规划'))
            report = self.plan_step()
        return report['run']
