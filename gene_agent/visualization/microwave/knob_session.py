"""旋钮在线会话：同一 GT 场景重新生成 Knowledge、碰撞与真实轨迹。"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.knowledge.geometry import load_concept
from gene_agent.knowledge.grasp import generate_grasps, load_grasp_profiles
from gene_agent.visualization.viser_sim.model import SceneEdits
from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession, ROOT
from gene_agent.tasks.policies.microwave_task.visual_session import knob_visual_run

KNOB_SKILLS = {'rotate_knob': '旋转旋钮'}
KNOB_TASKS = {'microwave_knob_rotate': '微波炉旋钮子任务'}
KNOB_PROFILES = {'tip_x_5_azimuth_240': 'tip X 5° / azimuth 240°', 'default': '旧默认六候选（保留失败基线）'}
DEFAULT_KNOB_CONFIG = {
    'knob_profile_variant': 'tip_x_5_azimuth_240',
    'object_poses_world': {
        'microwave': [[0.,1.,0.,.52],[-1.,0.,0.,.31],[0.,0.,1.,.04],[0.,0.,0.,1.]],
        'bowl': [[1.,0.,0.,.285],[0.,1.,0.,.12],[0.,0.,1.,-.078],[0.,0.,0.,1.]],
    },
    'object_joint_positions_rad': {'microwave_door_joint': 0., 'knob_0_joint': 0.},
}


class KnobVisualSession(ObjectVisualSession):
    """共享场景使用真实 home 初态；Skill 与单分区 Task 仅检查范围名称不同。"""

    def __init__(self, *, output_dir=None, config_path=None):
        self.output_dir = Path(output_dir or ROOT / 'runs' / f'knob-online-{datetime.now():%Y%m%d-%H%M%S-%f}').resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config = json.loads(json.dumps(DEFAULT_KNOB_CONFIG))
        supplied = json.loads(Path(config_path).read_text(encoding='utf-8')) if config_path else {}
        for key in ('object_poses_world', 'object_joint_positions_rad'):
            config[key].update(supplied.pop(key, {}))
        config.update(supplied)
        self.config = config
        self._plan_state = self.run = self.playback = self.prerequisite_report = None
        self.case = self._build(config, 0)
        self.initial = self._initial_state()

    def _build(self, config, revision):
        variant = config.get('knob_profile_variant', 'tip_x_5_azimuth_240')
        if variant not in KNOB_PROFILES:
            raise ValueError(f'未知旋钮 profile：{variant}')
        case = super()._build(config, revision)
        task = json.loads((ROOT / 'data/task_1/skills/knob_0_task.json').read_text(encoding='utf-8'))
        pose = np.eye(4)
        pose[:3,:3] = case.data.body('microwave').xmat.reshape(3,3)
        pose[:3,3] = case.data.body('microwave').xpos
        task['microwave_pose_world'] = pose.tolist()
        if variant == 'tip_x_5_azimuth_240':
            # 仅显式选择已经验证的 profile；不修改原文件，也不替失败候选做兜底。
            profiles = json.loads((ROOT / task['grasp_profile_path']).read_text(encoding='utf-8'))
            parameters = profiles['profiles'][task['grasp_profile_name']]['parameters']
            parameters['azimuth_deg']['values'] = [240]
            parameters['tip_x_deg']['values'] = [5]
            profile_path = case.output_dir / 'knob_profile.json'
            profile_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding='utf-8')
            task['grasp_profile_path'] = str(profile_path)
        task_path = case.output_dir / 'knob_task_config.json'
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding='utf-8')
        concept = load_concept(ROOT / task['concept_path'], part_name='knob_0')
        profile = load_grasp_profiles(ROOT / task['grasp_profile_path'])[task['grasp_profile_name']]
        self.knob_candidates = generate_grasps(concept.concept, pose, profile).proposals
        self.knob_task_path = task_path
        self._nominal_qpos = case.data.qpos.copy()
        self._attempted = False
        return case

    def _initial_state(self):
        state = super()._initial_state()
        angle = float(self.case.data.qpos[self.case.model.joint('knob_0_joint').qposadr[0]])
        return replace(state, object_joint_positions_rad={'knob_0_joint': angle})

    def set_profile(self, variant):
        if variant not in KNOB_PROFILES:
            raise ValueError(f'未知旋钮 profile：{variant}')
        previous = self.config
        self.config = {**previous, 'knob_profile_variant': variant}
        try:
            self.apply_edits(SceneEdits())
        except Exception:
            self.config = previous
            raise

    def begin_plan(self, selection):
        if selection not in KNOB_SKILLS and selection not in KNOB_TASKS:
            raise ValueError(f'不支持的旋钮入口：{selection}')
        if self.run is not None or self._attempted:
            self.apply_edits(SceneEdits())
        self._attempted = True
        self._plan_state = {'selection': selection, 'initial': self._initial_state()}
        return {'done': False, 'message': '正在同一 GT 场景规划旋钮；保留真实失败诊断'}

    def plan_step(self):
        from gene_agent.tasks.policies.microwave_task.planning.knob import run_rotate_knob_0_planning
        if self._plan_state is None:
            raise RuntimeError('请先 begin_plan')
        state = self._plan_state
        root = self.case.output_dir / 'knob'
        report = run_rotate_knob_0_planning(root, task_path=self.knob_task_path, case=self.case)
        self.report = report
        run = knob_visual_run(report, root)
        if not run.success:
            last = run.stages[-1]
            last = replace(last, failure={**last.failure,
                'candidate_count': report.get('candidate_count'),
                'grasp_reachable_count': report.get('grasp_reachable_count'),
                'report_ref': str(root / 'skill_rotate_knob_0_result.json')})
            run = replace(run, stages=(*run.stages[:-1], last))
        run = replace(run, selection_id=state['selection'], scene_revision=self.case.revision)
        # 不让 adapter 的规划终态污染 Knowledge 初态。播放层独立应用保存的真实样本。
        self.case.data.qpos[:] = self._nominal_qpos
        mujoco.mj_forward(self.case.model, self.case.data)
        return self._finish_run(run, state['initial'])
