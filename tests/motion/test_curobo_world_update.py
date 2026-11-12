"""世界增量更新不得保留过期几何或漏更新碰撞检查器。"""
from copy import deepcopy
import pytest
from gene_agent.motion.curobo import world_update


def scene():
    return {'mesh': {'hammer': {'file_path': '/mesh.obj', 'pose': [0,0,0,1,0,0,0], 'scale': [1,1,1]}}, 'cuboid': {}}


def test_pose_change_only_and_geometry_changes_require_reload():
    old=scene();new=deepcopy(old);new['mesh']['hammer']['pose'][0]=.2
    assert world_update.pose_updates(old,new)=={'hammer':new['mesh']['hammer']['pose']}
    for field,value in [('scale',[2,1,1]),('file_path','/different.obj')]:
        changed=deepcopy(new);changed['mesh']['hammer'][field]=value
        assert world_update.pose_updates(old,changed) is None
    changed=deepcopy(new);changed['mesh']['new']=changed['mesh'].pop('hammer')
    assert world_update.pose_updates(old,changed) is None
    assert old==scene()


def test_invalid_pose_is_rejected_before_world_mutation():
    for pose in ([0,0,0], [float('nan'),0,0,1,0,0,0], [0,0,0,0,0,0,0]):
        new=scene();new['mesh']['hammer']['pose']=pose
        with pytest.raises(ValueError):world_update.pose_updates(scene(),new)


class World:
    def __init__(self,names):self.poses={name:None for name in names};self.calls=0
    def get_obstacle_names(self):return list(self.poses)
    def update_obstacle_pose(self,name,pose):self.poses[name]=pose;self.calls+=1


def test_updates_all_worlds_once_and_respects_validator_partitions():
    a,b,c=World(['hammer','table']),World(['hammer']),World(['table'])
    worlds=[(a,{'hammer','table'}),(a,{'hammer','table'}),(b,{'hammer'}),(c,{'table'})]
    world_update.apply_pose_updates(worlds,{'hammer':[1,0,0,1,0,0,0]},lambda p:tuple(p))
    assert a.poses['hammer']==b.poses['hammer']==(1,0,0,1,0,0,0)
    assert (a.calls,b.calls,c.calls)==(1,1,0)


def test_missing_world_obstacle_fails_before_any_update():
    a,b=World(['hammer']),World([])
    with pytest.raises(ValueError,match='hammer'):
        world_update.apply_pose_updates([(a,{'hammer'}),(b,{'hammer'})],{'hammer':[1,0,0,1,0,0,0]},tuple)
    assert a.calls==0
