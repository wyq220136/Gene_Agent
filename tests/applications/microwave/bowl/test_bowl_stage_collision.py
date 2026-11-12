"""抓碗阶段碰撞快照与目标接触复核。"""

import json

import pytest


def test_bowl_grasp_stage_snapshots_use_commanded_opening_and_only_exclude_target(tmp_path):
    """规划接近时不能沿用 home 开口，也不能把桌面和微波炉一起屏蔽。"""
    from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession
    from gene_agent.tasks.policies.microwave_task.planning.bowl import prepare_grasp_stage_snapshots
    from gene_agent.robots.gripper import measure_gripper_width

    session = ObjectVisualSession(output_dir=tmp_path / 'stage-scenes')
    case = session.case
    original = measure_gripper_width(case.model, case.data, case.metadata['robot'], 'right')['width_m']
    strict, contact = prepare_grasp_stage_snapshots(case, .041, group='right')

    assert original == pytest.approx(.08, abs=.001)
    assert measure_gripper_width(case.model, case.data, case.metadata['robot'], 'right')['width_m'] == pytest.approx(original)
    assert strict['gripper_snapshot']['width_m'] == pytest.approx(.041, abs=.001)
    assert contact['gripper_snapshot']['width_m'] == pytest.approx(.041, abs=.001)
    strict_objects = {row['obstacle_name']: row['body_name'] for row in strict['records']}
    contact_objects = {row['obstacle_name']: row['body_name'] for row in contact['records']}
    assert any(body == 'bowl' for body in strict_objects.values())
    assert contact_objects == {name: body for name, body in strict_objects.items() if body != 'bowl'}
    assert contact['disabled_body_names'] == ['bowl']
    assert strict['snapshot_sha256'] != contact['snapshot_sha256']



def test_bowl_visual_skill_records_the_actual_collision_scene_for_each_stage(tmp_path, monkeypatch):
    """失败报告也必须让用户看出 pregrasp 与接近阶段用了哪些碰撞文件。"""
    from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession
    import gene_agent.tasks.policies.microwave_task.planning.grasp as planning
    from gene_agent.skills.runtime.planning import SkillPlan, SkillStage

    session = ObjectVisualSession(output_dir=tmp_path / 'skill-scenes')
    case = session.case
    initial_hash = case.snapshot['snapshot_sha256']

    class NoGpuPlanner:
        def __init__(self, **_kwargs):
            pass

    class StoppedPick:
        def plan(self, _planner, _proposals, _context, _settings, **_kwargs):
            return SkillPlan('pick', None, (
                SkillStage('open', 'planned', details={'width_m': .041}),
                SkillStage('pregrasp', 'failed'),
                SkillStage('approach', 'not_executed'),
                SkillStage('close', 'not_executed'),
                SkillStage('lift', 'not_executed'),
                SkillStage('hold', 'not_executed'),
            ), ())

    monkeypatch.setattr(planning, 'SynchronousPlanner', NoGpuPlanner)
    monkeypatch.setattr(planning, 'CuroboPlannerAdapter', lambda planner, **_kwargs: planner)
    monkeypatch.setattr(planning, 'PickSkill', StoppedPick)
    report = planning.run_pick_bowl_skill_planning(case)

    assert report['snapshot_sha256'] == initial_hash
    stages = {row['name']: row for row in report['stages']}
    strict_path = case.output_dir / stages['pregrasp']['collision_snapshot_ref']
    contact_path = case.output_dir / stages['approach']['collision_snapshot_ref']
    strict = json.loads(strict_path.read_text(encoding='utf-8'))
    contact = json.loads(contact_path.read_text(encoding='utf-8'))
    assert strict['gripper_snapshot']['width_m'] == pytest.approx(.041, abs=.001)
    assert strict['disabled_body_names'] == []
    assert contact['disabled_body_names'] == ['bowl']
    assert report['stage_collision_snapshots']['pregrasp']['snapshot_sha256'] == strict['snapshot_sha256']
    assert report['stage_collision_snapshots']['approach']['snapshot_sha256'] == contact['snapshot_sha256']



def test_planned_target_contact_rejects_palm_while_allowing_pad():
    """目标物规划豁免不能让非指垫部位穿过目标而被误报成功。"""
    import mujoco
    from gene_agent.motion.model import JointTrajectory
    from gene_agent.simulation.planned_contact import validate_planned_target_contacts

    model = mujoco.MjModel.from_xml_string('''
    <mujoco>
      <worldbody>
        <body name="arm_root">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom name="palm" type="box" size="0.01 0.01 0.01" mass="1"/>
          <geom name="pad" type="box" pos="0.1 0 0" size="0.01 0.01 0.01" mass="0.1"/>
        </body>
        <body name="target">
          <geom name="target_geom" type="sphere" pos="0.1 0 0" size="0.025"/>
        </body>
      </worldbody>
    </mujoco>''')
    data = mujoco.MjData(model)
    robot = {'group_root_bodies': {'right': 'arm_root'}}
    permitted = validate_planned_target_contacts(
        model, data, robot, group='right', object_body='target',
        trajectory=JointTrajectory(('slide',), (0., 1.), ((0.,), (0.,))),
        gripper_joint_positions_m={}, allowed_geom_names=('pad',),
    )
    rejected = validate_planned_target_contacts(
        model, data, robot, group='right', object_body='target',
        trajectory=JointTrajectory(('slide',), (0., 1.), ((0.,), (.1,))),
        gripper_joint_positions_m={}, allowed_geom_names=('pad',),
    )

    assert permitted['success'] is True
    assert permitted['allowed_contact_count'] > 0
    assert permitted['forbidden_contacts'] == []
    assert rejected['success'] is False
    assert any(row['robot_geom_name'] == 'palm' for row in rejected['forbidden_contacts'])
    assert data.qpos[model.joint('slide').qposadr[0]] == 0.



def test_forbidden_approach_contact_stops_pick_before_close():
    """规划器已返回轨迹也不能在掌部碰碗后继续声称闭爪成功。"""
    from gene_agent.tasks.policies.microwave_task.planning.grasp import apply_approach_contact_postcheck
    from gene_agent.skills.runtime.planning import SkillPlan, SkillStage

    plan = SkillPlan('pick', 'bowl_ring_02_03', (
        SkillStage('open', 'planned'), SkillStage('pregrasp', 'planned'),
        SkillStage('approach', 'planned'), SkillStage('close', 'planned'),
        SkillStage('lift', 'not_executed'), SkillStage('hold', 'not_executed'),
    ), ())
    evidence = {'success': False, 'forbidden_contacts': [
        {'robot_geom_name': 'palm', 'object_geom_name': 'bowl_wall'}],
        'checked_waypoints': 41,
    }
    stopped = apply_approach_contact_postcheck(plan, evidence)

    assert stopped.stage('pregrasp').status == 'planned'
    assert stopped.stage('approach').status == 'failed'
    assert stopped.stage('close').status == 'not_executed'
    assert stopped.stage('approach').details['target_contact_postcheck'] == evidence
    assert stopped.attempts[-1]['stage'] == 'approach_contact_postcheck'
    assert not stopped.success
