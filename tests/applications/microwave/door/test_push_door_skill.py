"""推门 Skill 继承已验证任务的阶段与运动约束。"""
import numpy as np
import pytest

from gene_agent.motion.model import MotionOperation
from gene_agent.skills.library.door.planning import PushDoorSkill
from gene_agent.skills.library.door.planning import DoorSkill, DoorSkillSettings
from gene_agent.knowledge.operation import AxisSpec


def test_push_door_keeps_precontact_free_and_every_push_linear():
    precontact = np.eye(4)
    contacts = [np.eye(4) for _ in range(3)]
    for index, pose in enumerate(contacts):
        pose[0, 3] = .01 * (index + 1)
    stages = PushDoorSkill().targets(precontact, contacts, [20., 50., 90.])
    assert stages[0].name == 'precontact'
    assert stages[0].operation is MotionOperation.PLAN
    assert [stage.operation for stage in stages[1:]] == [MotionOperation.LINEAR_TCP] * 3
    assert [stage.target_angle_deg for stage in stages[1:]] == [20., 50., 90.]


def test_push_door_requires_matching_nonempty_waypoints():
    with pytest.raises(ValueError, match='数量'):
        PushDoorSkill().targets(np.eye(4), [np.eye(4)], [])


def test_handle_pull_keeps_each_hinge_segment_linear():
    axis = AxisSpec(np.zeros(3), np.array([0., 0., 1.]), "world", "test")
    settings = DoorSkillSettings(.03, np.deg2rad(20), np.deg2rad(5), .08, .015, .08)
    stages = DoorSkill().targets(np.eye(4), axis, start_angle_rad=0., settings=settings)
    assert all(stage.operation is MotionOperation.LINEAR_TCP
               for stage in stages if stage.name.startswith("swing_"))
