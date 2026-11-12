"""完整 Task1 的在线同步会话；冻结真实 partition 输出供共享场景播放。"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gene_agent.visualization.viser_sim.model import SceneEdits, VisualRun, VisualStage
from gene_agent.visualization.viser_sim.playback import VisualPlayback, VisualState
from gene_agent.visualization.skill_inspector.planned_timeline import PlannedSegment, PlannedTimeline
from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession, object_visual_run, ROOT
from .visual import TASK1_STEP_ORDER, Task1VisualReport, write_task1_report
from gene_agent.visualization.microwave.door_session import _collision_facts, door_visual_run_from_session


TASK1_SELECTION = 'microwave_task1'
DEFAULT_TASK1_CONFIG = {
    'variant': 'legacy_viser', 'knob_profile_variant': 'tip_x_5_azimuth_240',
    'object_poses_world': {'microwave': [[0., 1., 0., .52], [-1., 0., 0., .31],
                                        [0., 0., 1., .04], [0., 0., 0., 1.]]},
}


def prefix_stages(partition: str, stages) -> tuple[VisualStage, ...]:
    """两轮门 Skill 名称相同，以 partition 前缀区分，轨迹数值保持原样。"""
    return tuple(replace(
        stage, stage_id=f'{partition}/{stage.stage_id}',
        trajectory=None if stage.trajectory is None else replace(
            stage.trajectory, stage_id=f'{partition}/{stage.stage_id}'),
        diagnostics={**stage.diagnostics, 'task_partition': partition,
            'collision_scope': (json.loads(Path(stage.collision_snapshot_ref).read_text(encoding='utf-8')).get('scope', '未提供')
                if stage.collision_snapshot_ref else '状态事件；未调用 planner')},
    ) for stage in stages)


def visual_state(state) -> VisualState:
    return VisualState(state.joint_names, np.asarray(state.joint_positions_rad),
                       state.gripper_width_m, dict(state.object_poses_world),
                       state.door_angle_rad, state.attachments,
                       {'knob_0_joint': state.knob_angle_rad})



def task_state_from_visual(basis, state: VisualState):
    """报告终态来自真实成功前缀的回放状态；不使用 cancel 恢复后的场景。

    model 标识来自同一 Task 基线，所有可见状态均来自已保留的实际轨迹与
    事件。这里只描述最后一个规划状态，不把未完成的目标当作已执行结果。
    """
    return replace(basis, joint_names=state.joint_names,
        joint_positions_rad=tuple(float(value) for value in state.joint_positions_rad),
        gripper_width_m=state.gripper_width_m, door_angle_rad=state.door_angle_rad,
        knob_angle_rad=state.object_joint_positions_rad.get('knob_0_joint', basis.knob_angle_rad),
        object_poses_world=dict(state.object_poses_world), attachments=state.attachments)


def knob_visual_run(report: dict, root: Path) -> VisualRun:
    """读取真实轨迹及离散 knob commit；绝不在机器人转动时伪造 knob 联动。"""
    root = Path(root)
    stages = []
    joint = report.get('joint_name', 'knob_0_joint')
    for row in report['stages']:
        if row['status'] == 'not_executed':
            break
        name = row['name']
        identifier = f'rotate_knob_0/{name}'
        result = row.get('result') or {}
        success = row['status'] == 'planned' and result.get('success', True)
        path = Path(row['collision_snapshot_ref']) if row.get('collision_snapshot_ref') else None
        if path is not None and not path.is_absolute():
            path = root / path
        reference = str(path) if path is not None and path.is_file() else None
        enabled, allowed, disabled, groups, count = _collision_facts(reference)
        common = dict(skill_id='rotate_knob_0', collision_snapshot_ref=reference,
                      collision_enabled=enabled, collision_allowed=allowed,
                      diagnostics={'collision_scope': row.get('collision_scope', '未提供'),
                          'collision_disabled_body_names': disabled, 'collision_display_groups': groups,
                          'collision_obstacle_count': count, 'direct_joint_command': row.get('direct_joint_command'),
                          'visual_only': True})
        target = (row.get('target') or {}).get('matrix')
        if not success:
            stages.append(VisualStage(identifier, motion_kind='object_state', success=False,
                target_tcp_world=None if target is None else np.asarray(target),
                failure=result.get('failure') or {'code': 'knob_planning_failed',
                    'message': row.get('axis_path_failure') or report.get('failure_reason') or '旋钮阶段失败'},
                **common))
            break
        samples = list(row.get('planned_replay_samples') or ())
        if not samples:
            raise ValueError(f'{identifier} 缺少实际 replay samples')
        commit = samples.pop() if row.get('visual_angle_commit') else None
        action = row.get('action')
        if action in {'gripper_open', 'gripper_grasp', 'gripper_release', 'wait'}:
            # 旧旋钮立即写开口再等待，不能套用门夹爪的持续插值语义。
            stages.append(VisualStage(identifier, motion_kind='object_state', success=True,
                gripper_command=action, gripper_width_m=float(samples[-1]['gripper_width_m']),
                object_joint_positions_rad={joint: float(samples[-1]['articulation_angle_rad'])},
                state_duration_s=float(row.get('settle_time_s', 0.)), **common))
        else:
            if action == 'last_joint_rotation':
                raw = {'joint_names': samples[0]['joint_names'],
                       'times_s': [s['time_s'] for s in samples],
                       'positions_rad': [s['joint_positions_rad'] for s in samples]}
                kind = 'joint'
            else:
                raw = result.get('trajectory') or {}
                kind = row.get('operation') or 'plan'
            times = np.asarray(raw.get('times_s', []), dtype=float)
            if len(times) < 2 or not np.allclose(np.diff(times), times[1]-times[0], atol=1e-8):
                raise ValueError(f'{identifier} 缺少等间隔真实轨迹')
            # 保存的 robot samples 必须与实际 planner/直接命令逐点一致。
            positions = np.asarray(raw['positions_rad'])
            np.testing.assert_allclose(positions, [s['joint_positions_rad'] for s in samples], atol=1e-7)
            if any(abs(float(s['articulation_angle_rad']) - float(samples[0]['articulation_angle_rad'])) > 1e-9 for s in samples):
                raise ValueError('旋钮运动中存在未支持的非离散角度通道')
            segment = PlannedSegment(identifier, 'rotate_knob_0', tuple(raw['joint_names']), positions,
                                     float(times[1]-times[0]), {}, details={'source': action})
            stages.append(VisualStage(identifier, motion_kind=kind, success=True, trajectory=segment,
                target_tcp_world=None if target is None else np.asarray(target),
                planning_duration_s=result.get('duration_s'), **common))
        if commit is not None:
            delay = float(commit['time_s']) - float(samples[-1]['time_s'])
            if delay <= 0:
                raise ValueError('旋钮 commit 必须在实际轨迹末帧之后')
            # 只接受机器人未发生跳变的离散提交；周期 wrap 需另做显式无损映射。
            np.testing.assert_allclose(commit['joint_positions_rad'], samples[-1]['joint_positions_rad'], atol=1e-7)
            stages.append(VisualStage(identifier + '/visual_angle_commit', motion_kind='object_state',
                success=True, state_duration_s=delay, state_update_at_end=True,
                object_joint_positions_rad={joint: float(commit['articulation_angle_rad'])}, **common))
    return VisualRun('microwave_knob_rotate', int(report.get('scene_revision', 0)), tuple(stages))


class Task1VisualSession(ObjectVisualSession):
    """编辑后重建一份 catalog；每次 tick 同步规划一个 Task partition。"""

    def __init__(self, *, output_dir=None, config_path=None):
        self.output_dir = Path(output_dir or ROOT / 'runs' / f'task1-online-{datetime.now():%Y%m%d-%H%M%S-%f}').resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = (json.loads(Path(config_path).read_text(encoding='utf-8')) if config_path
                       else json.loads(json.dumps(DEFAULT_TASK1_CONFIG)))
        self._plan_state = None
        self.run = self.playback = self.prerequisite_report = None
        self.case = self._build(self.config, 0)
        self.initial = self._initial_state()

    def _build(self, config, revision):
        from gene_agent.visualization.microwave.catalog import build_task1_catalog
        catalog = build_task1_catalog(self.output_dir / f'revision_{revision:03d}',
            variant=config.get('variant', 'legacy_viser'),
            knob_profile_variant=config.get('knob_profile_variant', 'tip_x_5_azimuth_240'),
            pull_angle_deg=config.get('pull_angle_deg'),
            push_angle_deg=config.get('push_angle_deg'),
            push_start_angle_deg=config.get('push_start_angle_deg'),
            scene_edits=SceneEdits(
                object_poses_world={k: np.asarray(v) for k,v in config.get('object_poses_world', {}).items()},
                object_joint_positions_rad=config.get('object_joint_positions_rad', {}),
                table_size_xy_m=config.get('table_size_xy_m')), scene_revision=revision,
            on_door_phase_complete=self._freeze_door_phase)
        self.catalog = catalog
        self._attempted = False
        return catalog.case

    def _initial_state(self):
        return visual_state(self.catalog.initial_state)

    def _finish_run(self, run, initial):
        from gene_agent.visualization.microwave.catalog import apply_task_state
        # 规划只产出结果；渲染器的基线与 Knowledge 视图始终恢复本轮名义初态。
        apply_task_state(self.case, self.catalog.initial_state)
        return super()._finish_run(run, initial)

    def begin_plan(self, selection):
        if selection != TASK1_SELECTION:
            raise ValueError(f'完整 Task1 会话不支持 {selection}')
        if self.run is not None or self._attempted:
            self.apply_edits(SceneEdits())
        self._attempted = True
        self._plan_state = {'index': 0, 'current': self.catalog.initial_state,
                            'stages': [], 'rows': [], 'initial': self._initial_state()}
        return {'done': False, 'message': '准备从同一场景规划完整 Task1（7 个分区）'}

    def _freeze_door_phase(self, door):
        """在下一次候选筛选之前复制当前已完成 phase 的实际轨迹与状态通道。

        候选无解会 cancel 并清空 _sequence_state；不能等 partition 返回后
        才从最终 timeline 取前缀。此回调只读现有结果，绝不生成或再规划轨迹。
        """
        row = door.results[-1]
        phase_id = row['phase_id']
        phase = next(phase for phase in door.phases if phase.phase_id == phase_id)
        source = (door._sequence_state['segments'] if door._sequence_state is not None
                  else () if door.timeline is None else door.timeline.segments)
        segment = next((segment for segment in source if segment.stage_id == phase_id), None)
        # PlannedSegment.__post_init__ 会复制 positions/channels，避免后续 session
        # 对原 ndarray 的写入、取消、重复门 Skill 覆盖冻结的证据。
        frozen = None if segment is None else replace(segment)
        view = SimpleNamespace(results=(row,), phases=(phase,), revision=door.revision,
            timeline=None if frozen is None else PlannedTimeline((frozen,)))
        stage = door_visual_run_from_session(view, None, first_row=0,
                    report={'success': bool(row.get('success'))}).stages[0]
        if row.get('knowledge_candidate_selection') is not None:
            stage = replace(stage, diagnostics={**stage.diagnostics,
                            'knowledge_candidate_selection': row['knowledge_candidate_selection']})
        self._door_frozen[phase_id] = stage

    def _convert(self, result, first_door_row):
        name = result.step
        root = self.catalog.output_dir
        if name.startswith(('open_door_', 'close_door_')):
            stages = []
            for row in result.stages:
                phase_id = row['phase_id']
                if row.get('operation') == 'external_door_state_update':
                    stages.append(VisualStage(phase_id, 'task_transition', 'door_state', True,
                        door_angle_rad=float(np.deg2rad(row['to_angle_deg'])),
                        diagnostics={'external_state_update': True,
                                     'collision_scope': '外部门状态事件；未调用 planner'}))
                elif row.get('operation') == 'knowledge_candidate_selection':
                    # 此失败行只存在于 TaskStepResult，不在 session.results 中。
                    # 它没有规划轨迹；保留原始 IK 失败及候选筛选所用快照。
                    facts = row['candidate_selection']
                    reference = facts.get('collision_snapshot_ref')
                    enabled, allowed, disabled, groups, count = _collision_facts(reference)
                    phase = next(phase for phase in self.catalog.door_session.phases
                                 if phase.phase_id == phase_id)
                    stages.append(VisualStage(phase_id, phase.skill_id, 'object_state', False,
                        failure=dict(result.failure), target_tcp_world=phase.tcp_pose_world,
                        collision_snapshot_ref=reference, collision_enabled=enabled,
                        collision_allowed=allowed, diagnostics={
                            'knowledge_candidate_selection': facts,
                            'collision_disabled_body_names': disabled,
                            'collision_display_groups': groups, 'collision_obstacle_count': count}))
                else:
                    stages.append(self._door_frozen[phase_id])
            return prefix_stages(name, stages)
        if name == 'insert_bowl':
            pick = json.loads((self.case.output_dir / 'skill_pick_bowl_result.json').read_text(encoding='utf-8'))
            # Pick 使用开门实终态导出的 before_insert world，不能显示初始关门快照。
            base_ref = str(self.case.output_dir / 'task1_collision/before_insert/collision_snapshot.json')
            for row in pick['stages']:
                if row['name'] in {'open', 'pregrasp', 'approach', 'close'}:
                    row['collision_snapshot_ref'] = base_ref
            first = object_visual_run(pick, 'pick_bowl', self.case.output_dir)
            stages = list(first.stages)
            if first.success:
                insert = json.loads((self.case.output_dir / 'skill_insert_bowl_result.json').read_text(encoding='utf-8'))
                stages.extend(object_visual_run(insert, 'insert_bowl', self.case.output_dir,
                                                initial_attachments=('bowl',)).stages)
            return prefix_stages(name, stages)
        if name == 'rotate_knob':
            report = json.loads((root / 'knob/skill_rotate_knob_0_result.json').read_text(encoding='utf-8'))
            return prefix_stages(name, knob_visual_run(report, root / 'knob').stages)
        if name == 'extract_place_bowl':
            from gene_agent.tasks.policies.microwave_task.planning.extract import extract_visual_run
            report = json.loads((root / 'extract/skill_extract_bowl_result.json').read_text(encoding='utf-8'))
            return prefix_stages(name, extract_visual_run(report, root / 'extract').stages)
        raise ValueError(f'未支持 Task partition: {name}')

    def plan_step(self):
        """适配器或转换异常同样保留已完成前缀，禁止 UI 清空后只报异常。"""
        if self._plan_state is None:
            raise RuntimeError('请先 begin_plan')
        state = self._plan_state
        previous = tuple(state['stages'])
        index = state['index']
        try:
            return self._plan_step_impl()
        except Exception as error:
            name = TASK1_STEP_ORDER[index]
            # 门分区在候选筛选或 planner 抛异常前，可能已经完成若干真实
            # 阶段。回调在每段结束时冻结了轨迹；此处必须把本分区前缀接到
            # 已完成的上一个分区之后，否则回放和报告都会退回旧终态。
            frozen = (prefix_stages(name, self._door_frozen.values())
                      if name.startswith(('open_door_', 'close_door_')) else ())
            failure = {'code': 'task_adapter_or_visual_conversion_error',
                       'message': f'{type(error).__name__}: {error}', 'step': name}
            failed = VisualStage(f'{name}/error', name, 'object_state', False, failure=failure,
                                 diagnostics={'task_partition': name})
            run = VisualRun(TASK1_SELECTION, self.case.revision, previous + frozen + (failed,))
            playback = VisualPlayback(run, state['initial'], self.tcp_pose)
            final_state = task_state_from_visual(state['current'],
                playback.sample(playback.duration_s).state)
            report = Task1VisualReport(False, tuple(state['rows']) + ({'step': name,
                'status': 'failed', 'failure': failure, 'end_state': final_state.to_dict(),
                'end_state_source': 'last_planned_visual_state'},), final_state, failure,
                dict(self.catalog.configuration))
            write_task1_report(self.catalog.output_dir / 'task1_visual_result.json', report)
            return self._finish_run(run, state['initial'])

    def _plan_step_impl(self):
        if self._plan_state is None:
            raise RuntimeError('请先 begin_plan')
        state = self._plan_state
        name = TASK1_STEP_ORDER[state['index']]
        first_row = len(self.catalog.door_session.results)
        # 每个 partition 独立收集，避免第二轮同名门 phase 沿用第一轮记录。
        self._door_frozen = {}
        result = self.catalog.adapters[name](state['current'])
        mismatch = state['current'].discontinuity(result.start_state)
        if mismatch:
            raise ValueError(f'{name} Task 边界不连续：{mismatch}')
        try:
            stages = self._convert(result, first_row)
        except (FileNotFoundError, RuntimeError) as error:
            if result.success:
                raise
            stages = (VisualStage(f'{name}/failed', name, 'object_state', False,
                       failure={**result.failure, 'conversion_note': str(error)},
                       diagnostics={'task_partition': name}),)
        state['stages'].extend(stages)
        row = {'step': name, 'status': 'planned' if result.success else 'failed',
               'start_state': result.start_state.to_dict(), 'stages': result.stages}
        if result.success:
            row['end_state'] = result.end_state.to_dict()
            state['current'] = result.end_state
        else:
            row['failure'] = result.failure
        state['rows'].append(row)
        run = VisualRun(TASK1_SELECTION, self.case.revision, tuple(state['stages']))
        playback = VisualPlayback(run, state['initial'], self.tcp_pose)
        observed = playback.sample(playback.duration_s).state
        if result.success:
            expected = visual_state(result.end_state)
            np.testing.assert_allclose(observed.joint_positions_rad, expected.joint_positions_rad, atol=1e-5)
            np.testing.assert_allclose(observed.gripper_width_m, expected.gripper_width_m, atol=1e-5)
            np.testing.assert_allclose(observed.door_angle_rad, expected.door_angle_rad, atol=1e-5)
            np.testing.assert_allclose(observed.object_joint_positions_rad['knob_0_joint'],
                                       expected.object_joint_positions_rad['knob_0_joint'], atol=1e-5)
            for object_name, pose in expected.object_poses_world.items():
                np.testing.assert_allclose(observed.object_poses_world[object_name], pose, atol=1e-5)
            if observed.attachments != expected.attachments:
                raise ValueError(f'{name} 回放 attachment 与 adapter 终态不同')
        else:
            state['current'] = task_state_from_visual(state['current'], observed)
            row['end_state'] = state['current'].to_dict()
            row['end_state_source'] = 'last_planned_visual_state'
        state['index'] += 1
        report = Task1VisualReport(result.success and state['index'] == len(TASK1_STEP_ORDER),
            tuple(state['rows']), state['current'], None if result.success else result.failure,
            dict(self.catalog.configuration))
        write_task1_report(self.catalog.output_dir / 'task1_visual_result.json', report)
        if not result.success or state['index'] == len(TASK1_STEP_ORDER):
            initial = state['initial']
            # 额外冻结门段真实轨迹，避免仅保存诊断 rows 后丢失完整回放证据。
            samples = [{**stage.manifest_row(), 'trajectory': None if stage.trajectory is None else {
                'joint_names': list(stage.trajectory.joint_names),
                'positions_rad': stage.trajectory.positions_rad.tolist(),
                'sample_dt_s': stage.trajectory.sample_dt_s,
                'channels': {k:v.tolist() for k,v in stage.trajectory.channels.items()}}} for stage in run.stages]
            (self.catalog.output_dir / 'visual_run_samples.json').write_text(json.dumps(
                {'initial_state': self.catalog.initial_state.to_dict(), 'stages': samples},
                ensure_ascii=False, indent=2), encoding='utf-8')
            return self._finish_run(run, initial)
        return {'done': False, 'message': f'{state["index"]}/7 分区完成：{name}；下一步 {TASK1_STEP_ORDER[state["index"]]}',
                'partial_run': run, 'partial_playback': playback}
