"""模型描述必须捕获关节错配，不能假定七轴或允许隐式型号。"""
from copy import deepcopy
import numpy as np
import pytest


def metadata(count=6):
    """独立的具名旋转臂组；特意不叫 left/right。"""
    names = [f'axis_{i}' for i in range(count)]
    return dict(profile='test_arm', embodiment_id='fixture', world_frame='world',
                joint_groups={'working': names}, planner_joint_names={'working': names},
                joint_types={'working': ['revolute'] * count},
                joint_limits={'working': [[-2., 2.]] * count},
                tool_links={'working': 'tip'},
                transforms={'working': {'world_from_urdf_base': np.eye(4).tolist(),
                                        'link_from_tcp': np.eye(4).tolist()}},
                grippers={'working': {'width_range_m': [0., .08]}})


@pytest.mark.parametrize('count', [6, 7])
def test_group_mapping_roundtrip_and_input_isolation(count):
    from gene_agent.robots.model import RobotDescription
    original = metadata(count)
    robot = RobotDescription.from_metadata(original)
    original['joint_groups']['working'].reverse()
    assert robot.groups['working'].joint_names == tuple(f'axis_{i}' for i in range(count))
    saved = robot.to_metadata()
    assert saved == metadata(count)
    saved['transforms']['working']['link_from_tcp'][0][3] = 3.
    assert robot.to_metadata()['transforms']['working']['link_from_tcp'][0][3] == 0.


@pytest.mark.parametrize('mutation', ['duplicate', 'planner', 'pose', 'nan', 'limit', 'prismatic', 'profile'])
def test_invalid_description_rejected_before_planning(mutation):
    from gene_agent.robots.model import RobotDescription
    raw = metadata()
    if mutation == 'duplicate': raw['joint_groups']['working'][1] = 'axis_0'
    if mutation == 'planner': raw['planner_joint_names']['working'].pop()
    if mutation == 'pose': raw['transforms']['working']['link_from_tcp'][0][0] = 2.
    if mutation == 'nan': raw['transforms']['working']['world_from_urdf_base'][0][3] = float('nan')
    if mutation == 'limit': raw['joint_limits']['working'][0] = [1., -1.]
    if mutation == 'prismatic': raw['joint_types']['working'][0] = 'prismatic'
    if mutation == 'profile': del raw['profile']
    with pytest.raises(ValueError): RobotDescription.from_metadata(raw)


def test_explicit_legacy_profile_preserves_extra_metadata():
    from gene_agent.robots.model import RobotDescription
    raw = metadata(7)
    raw['profile'] = 'realman_single_pika'
    del raw['joint_types']; del raw['joint_limits']; del raw['planner_joint_names']
    raw['provenance'] = {'note': 'retained'}
    assert RobotDescription.from_metadata(raw).to_metadata() == raw
