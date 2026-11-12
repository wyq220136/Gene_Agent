"""同一轨迹全部采样批量 FK；CPU 替身检查坐标、顺序及单次后端调用。"""
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch
from gene_agent.motion.curobo.core import SynchronousPlanner


def test_stationary_pivot_uses_start_tcp_axis_without_weakening_linear_validation(monkeypatch):
    from scipy.spatial.transform import Rotation
    p = SynchronousPlanner.__new__(SynchronousPlanner)
    p.world_from_base = np.eye(4)
    p.link_from_tcp = np.eye(4)
    p.tool_frame, p.tcp_frame = "tool", "tcp"
    p._goal = lambda *args: None
    p.torch = SimpleNamespace(float32=np.float32, tensor=lambda value, **kw: np.asarray(value))
    monkeypatch.setitem(sys.modules, "curobo.types", SimpleNamespace(
        Pose=lambda **kw: SimpleNamespace(**kw),
        GoalToolPose=SimpleNamespace(from_poses=lambda poses, **kw: poses)))
    start = np.eye(4)
    target = start.copy()
    target[:3, :3] = Rotation.from_rotvec([0, 0, .04]).as_matrix()
    with pytest.raises(ValueError):
        p._linear_goal(target, "world_tcp", start)
    goal = p._linear_goal(target, "world_tcp", start, stationary_rotation=True)
    np.testing.assert_array_equal(goal["tcp"].position, [[0, 0, 0]])
    # 起点TCP的Z轴作为无平移时的辅助轴；实际末端仍要求旋转后的目标姿态。
    quaternion = goal["tcp"].quaternion[0]
    np.testing.assert_allclose(Rotation.from_quat(quaternion[[1, 2, 3, 0]]).apply([0, 0, 1]),
                               [0, 0, 1], atol=1e-12)
    assert abs(goal["tool"].quaternion[0, 3]) > .019


def fake_planner(monkeypatch):
    p=SynchronousPlanner.__new__(SynchronousPlanner)
    p.joint_names=['a','b','c'];p.planner_joint_names=['u','v','w'];calls=[]
    p.torch=SimpleNamespace(float32=torch.float32,
        as_tensor=lambda value,**kw:torch.as_tensor(value,dtype=kw['dtype']))
    monkeypatch.setitem(sys.modules,'curobo.types',SimpleNamespace(JointState=SimpleNamespace(
        from_position=lambda value,**kw:SimpleNamespace(position=value,joint_names=kw['joint_names']))))
    def compute(state):
        calls.append(state)
        poses=SimpleNamespace(position=state.position,quaternion=torch.tensor([[1.,0.,0.,0.]]).expand(len(state.position),-1))
        return SimpleNamespace(tool_poses=SimpleNamespace(get_link_pose=lambda name:poses))
    p.kinematics=SimpleNamespace(compute_kinematics=compute);p.tool_frame='tool'
    p.world_from_base=np.eye(4);p.world_from_base[:3,3]=[1.,2.,3.]
    p.link_from_tcp=np.eye(4);p.link_from_tcp[:3,3]=[.1,.2,.3]
    return p,calls


def test_batch_fk_reorders_all_rows_and_applies_frames_once(monkeypatch):
    p,calls=fake_planner(monkeypatch)
    result=p.fk_batch(['c','a','b'],[[3.,1.,2.],[6.,4.,5.]])
    assert len(calls)==1 and calls[0].joint_names==['u','v','w']
    assert calls[0].position.is_contiguous()
    np.testing.assert_allclose(result[:,:3,3],[[2.1,4.2,6.3],[5.1,7.2,9.3]])
    np.testing.assert_array_equal(result[:,3,:],[[0,0,0,1],[0,0,0,1]])


def test_empty_batch_fk_has_no_backend_call(monkeypatch):
    p,calls=fake_planner(monkeypatch)
    assert p.fk_batch(p.joint_names,[]).shape==(0,4,4)
    assert not calls


@pytest.mark.parametrize('values',[[[1.,2.]],[[1.,2.,float('nan')]]])
def test_batch_fk_rejects_invalid_joint_matrix(monkeypatch,values):
    p,calls=fake_planner(monkeypatch)
    with pytest.raises(ValueError):p.fk_batch(p.joint_names,values)
    assert not calls


def test_solution_extraction_removes_locked_joints_preserving_batch_seed_axes():
    p=SynchronousPlanner.__new__(SynchronousPlanner);p.planner_joint_names=['a','b']
    values=torch.tensor([[[90.,2.,1.]],[[91.,4.,3.]]])
    source=SimpleNamespace(reorder=lambda names:SimpleNamespace(position=values[:,:,[2,1]]))
    np.testing.assert_array_equal(p._solution_positions(source),[[[1.,2.]],[[3.,4.]]])


def test_world_probe_releases_temporary_solver_when_solve_raises(monkeypatch):
    from gene_agent.motion.configuration import CollisionMode
    destroyed=[]
    class Probe:
        def __init__(self,*args):pass
        def solve_pose(self,*args,**kwargs):raise RuntimeError('probe failed')
        def destroy(self):destroyed.append(True)
    monkeypatch.setitem(sys.modules,'curobo.inverse_kinematics',SimpleNamespace(
        InverseKinematics=Probe,InverseKinematicsCfg=SimpleNamespace(create=lambda **kw:None)))
    p=SynchronousPlanner.__new__(SynchronousPlanner)
    p.collision_mode=CollisionMode.FULL;p.robot_config={};p.runtime_options={}
    with pytest.raises(RuntimeError,match='probe failed'):p._diagnose_world_obstruction(None,None)
    assert destroyed==[True]
