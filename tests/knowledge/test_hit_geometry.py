import numpy as np
import pytest
from scipy.spatial.transform import Rotation

def test_alignment_uses_actual_tool_in_tcp_and_observed_target():
    from gene_agent.knowledge.operation import hit
    assert hasattr(hit,'align_hit_tcp')
    tool=np.eye(4);tool[:3,:3]=Rotation.from_euler('xy',[.4,.3]).as_matrix();tool[:3,3]=[.1,.2,.3]
    tcp=tool.copy();tcp[:3,3]+=[.03,-.04,.02]
    target=np.eye(4);target[:3,3]=[-.2,.3,.8]
    point=np.array([0,.06,.05]);normal=np.array([0,0,1])
    goal=hit.align_hit_tcp(tool,tcp,target,strike_point=point,strike_normal=normal,target_point=[0,0,.008],clearance_m=.05)
    achieved=goal@np.linalg.inv(tcp)@tool
    assert achieved[:3,:3]@normal==pytest.approx([0,0,-1])
    assert achieved[:3,:3]@point+achieved[:3,3]==pytest.approx([-.2,.3,.858])


def test_invalid_surface_normal_rejected():
    from gene_agent.knowledge.operation import hit
    with pytest.raises(ValueError):
        hit.align_hit_tcp(np.eye(4),np.eye(4),np.eye(4),strike_point=[0,0,0],strike_normal=[0,0,0],target_point=[0,0,0],clearance_m=.05)


def test_yaw_candidates_preserve_strike_point_normal_and_clearance():
    from gene_agent.knowledge.operation.hit import align_hit_tcp_candidates
    tool=np.eye(4);tool[:3,3]=[.1,.2,.8]
    tcp=tool.copy();tcp[:3,3]+=[.03,.02,.01]
    target=np.eye(4);target[:3,3]=[-.2,.3,.8]
    point=np.array([0,.06,.05]);normal=np.array([0,0,1])
    goals=align_hit_tcp_candidates(tool,tcp,target,strike_point=point,strike_normal=normal,target_point=[0,0,.008],clearance_m=.05,yaw_offsets_rad=[0,-.3,.3])
    assert len(goals)==3
    for goal in goals:
        actual=goal@np.linalg.inv(tcp)@tool
        assert actual[:3,:3]@normal==pytest.approx([0,0,-1])
        assert actual[:3,:3]@point+actual[:3,3]==pytest.approx([-.2,.3,.858])
