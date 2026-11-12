"""夹持中心与操作 TCP 分开；几何检查不调用 IK 或物理引擎。"""
import json
from pathlib import Path
import numpy as np
import pytest

from gene_agent.robots.pika_geometry import pika_grasp_center_tcp
from gene_agent.knowledge.sweep import make_brush_grasp_pose
from gene_agent.knowledge.geometry import load_concept
from gene_agent.skills.library.pick.brush import brush_grasp_cross_section

SCENE = Path('data/realworld/brush_cube_dustpan_0')


def test_pika_grasp_center_uses_selected_inner_region_without_changing_tcp():
    metadata = json.loads((SCENE/'robot/robot_metadata.json').read_text())
    point = pika_grasp_center_tcp(metadata, 'left')
    np.testing.assert_allclose(point, [-.0075, -.0000095, .0064182], atol=1e-7)
    assert metadata['transforms']['left']['link_from_tcp'] == json.loads((SCENE/'robot/robot_metadata.json').read_text())['transforms']['left']['link_from_tcp']


@pytest.mark.parametrize('yaw', [0., .7, -1.1])
def test_grasp_aligns_actual_finger_center_with_handle(yaw):
    from scipy.spatial.transform import Rotation
    brush = np.eye(4);brush[:3,:3] = Rotation.from_euler('z', yaw).as_matrix()
    brush[:3,3] = [.35,-.18,.01]
    point = np.array([-.0135, -.0000095, .0064182])
    grasp = make_brush_grasp_pose(brush, SCENE/'concepts/brush.json', grasp_center_tcp_m=point)
    part = load_concept(SCENE/'concepts/brush.json',part_name='grip_handle').concept
    np.testing.assert_allclose(grasp[:3,3]+grasp[:3,:3]@point,(brush@part.pose_in_object)[:3,3],atol=1e-10)


def test_preview_uses_outer_handle_surface_not_internal_mesh_shell():
    point = np.array([-.0135, -.0000095, .0064182])
    brush = np.eye(4)
    grasp = make_brush_grasp_pose(brush,SCENE/'concepts/brush.json',grasp_center_tcp_m=point)
    result = brush_grasp_cross_section(brush, SCENE/'assets/brush.obj',grasp,point)
    assert .015 < result['predicted_width_m'] < .018
    assert result['physical_hold_verified'] is False
    # 旧浅抓的实际夹持中心位于刷柄上方，不能假装验证通过。
    old = make_brush_grasp_pose(brush, SCENE/'concepts/brush.json')
    with pytest.raises(ValueError,match='夹持中心'):
        brush_grasp_cross_section(brush,SCENE/'assets/brush.obj',old,point)


def test_current_visual_knowledge_uses_the_robot_grasp_center(tmp_path):
    from gene_agent.scene.visual_inputs import load_visual_inputs
    from gene_agent.tasks.planning.knowledge import proposals
    inputs = load_visual_inputs('sweep',SCENE,tmp_path)
    row = proposals(inputs,'brush_grasp')[0]
    point = pika_grasp_center_tcp(inputs.robots['left'].metadata,'left')
    part = load_concept(SCENE/'concepts/brush.json',part_name='grip_handle').concept
    expected = (inputs.objects['brush']['T_world_object']@part.pose_in_object)[:3,3]
    np.testing.assert_allclose(row['pose'][:3,3]+row['pose'][:3,:3]@point,expected,atol=1e-10)


def test_complete_fingers_stop_before_local_handle_section_width():
    from gene_agent.robots.pika_geometry import pika_predict_contact_width
    metadata=json.loads((SCENE/'robot/robot_metadata.json').read_text())
    center=pika_grasp_center_tcp(metadata,'left')
    grasp=make_brush_grasp_pose(np.eye(4),SCENE/'concepts/brush.json',grasp_center_tcp_m=center)
    result=pika_predict_contact_width(metadata,'left',np.linalg.inv(grasp),SCENE/'assets/brush.obj',open_width_m=.04)
    assert .022 < result['predicted_width_m'] < .028
    assert result['physical_hold_verified'] is False
    assert result['source']=='full_finger_mesh_first_contact'
    assert result['predicted_width_m']==max(result['per_finger_contact_width_m'].values())


def test_preview_rejects_finger_overlap_at_initial_opening():
    from gene_agent.robots.pika_geometry import pika_predict_contact_width
    metadata=json.loads((SCENE/'robot/robot_metadata.json').read_text())
    grasp=make_brush_grasp_pose(np.eye(4),SCENE/'concepts/brush.json',
        grasp_center_tcp_m=pika_grasp_center_tcp(metadata,'left'))
    shifted=np.linalg.inv(grasp);shifted[1,3]+=.01
    with pytest.raises(ValueError,match='初始.*穿透'):
        pika_predict_contact_width(metadata,'left',shifted,SCENE/'assets/brush.obj',open_width_m=.04)
