"""世界增量更新不得保留过期几何或漏更新碰撞检查器。"""
from copy import deepcopy
import pytest
from gene_agent.motion.curobo import world_update


def scene():
    return {'mesh': {'hammer': {'file_path': '/mesh.obj', 'pose': [0,0,0,1,0,0,0], 'scale': [1,1,1]}}, 'cuboid': {}}


def test_pose_change_only_and_geometry_changes_require_reload(tmp_path):
    path=tmp_path/'mesh.obj';path.write_bytes(b'geometry')
    old=scene();old['mesh']['hammer']['file_path']=str(path)
    original=deepcopy(old)
    loaded=world_update.mesh_content_hashes(old)
    new=deepcopy(old);new['mesh']['hammer']['pose'][0]=.2
    assert world_update.pose_updates(old,new,previous_mesh_hashes=loaded,
        current_mesh_hashes=world_update.mesh_content_hashes(new))=={'hammer':new['mesh']['hammer']['pose']}
    for field,value in [('scale',[2,1,1]),('file_path','/different.obj')]:
        changed=deepcopy(new);changed['mesh']['hammer'][field]=value
        assert world_update.pose_updates(old,changed,previous_mesh_hashes=loaded,
            current_mesh_hashes=world_update.mesh_content_hashes(changed)) is None
    changed=deepcopy(new);changed['mesh']['new']=changed['mesh'].pop('hammer')
    assert world_update.pose_updates(old,changed,previous_mesh_hashes=loaded,
        current_mesh_hashes=world_update.mesh_content_hashes(changed)) is None
    assert old==original


def test_invalid_pose_is_rejected_before_world_mutation(tmp_path):
    path=tmp_path/'mesh.obj';path.write_bytes(b'geometry')
    old=scene();old['mesh']['hammer']['file_path']=str(path)
    loaded=world_update.mesh_content_hashes(old)
    for pose in ([0,0,0], [float('nan'),0,0,1,0,0,0], [0,0,0,0,0,0,0]):
        new=deepcopy(old);new['mesh']['hammer']['pose']=pose
        with pytest.raises(ValueError):world_update.pose_updates(old,new,
            previous_mesh_hashes=loaded,current_mesh_hashes=world_update.mesh_content_hashes(new))


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


def test_same_mesh_bytes_at_new_episode_path_only_updates_pose(tmp_path):
    from gene_agent.motion.curobo.world_update import pose_updates
    before=tmp_path/'episode1.obj';after=tmp_path/'episode2.obj'
    before.write_bytes(b'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    after.write_bytes(before.read_bytes())
    old={'mesh':{'bin':{'file_path':str(before),'scale':[1.,1.,1.],'pose':[0.,0.,0.,1.,0.,0.,0.]}}}
    new={'mesh':{'bin':{'file_path':str(after),'scale':[1.,1.,1.],'pose':[.1,0.,0.,1.,0.,0.,0.]}}}
    loaded=world_update.mesh_content_hashes(old)
    assert pose_updates(old,new,previous_mesh_hashes=loaded,
        current_mesh_hashes=world_update.mesh_content_hashes(new))=={'bin':new['mesh']['bin']['pose']}
    after.write_bytes(b'different actual geometry')
    assert pose_updates(old,new,previous_mesh_hashes=loaded,
        current_mesh_hashes=world_update.mesh_content_hashes(new)) is None


def test_missing_old_mesh_file_requires_reload_even_if_name_looks_identical(tmp_path):
    from gene_agent.motion.curobo.world_update import pose_updates
    after=tmp_path/'new.obj';after.write_bytes(b'known bytes')
    row={'file_path':str(after),'scale':[1.,1.,1.],'pose':[0.,0.,0.,1.,0.,0.,0.]}
    old={'mesh':{'bin':{**row,'file_path':str(tmp_path/'removed.obj')}}}
    assert pose_updates(old,{'mesh':{'bin':row}}) is None


def test_same_gltf_bytes_can_reference_different_geometry_and_require_reload(tmp_path):
    # glTF主文件相同不代表相对buffer相同，不能只比较主文件后复用旧几何。
    first=tmp_path/'first';second=tmp_path/'second'
    first.mkdir();second.mkdir()
    for directory,geometry in ((first,b'old vertices'),(second,b'new vertices')):
        (directory/'model.gltf').write_text('{"buffers":[{"uri":"mesh.bin"}]}')
        (directory/'mesh.bin').write_bytes(geometry)
    old=scene();new=deepcopy(old)
    old['mesh']['hammer']['file_path']=str(first/'model.gltf')
    new['mesh']['hammer']['file_path']=str(second/'model.gltf')
    assert world_update.mesh_content_hashes(old)=={'hammer':None}
    assert world_update.pose_updates(old,new,previous_mesh_hashes=world_update.mesh_content_hashes(old),
        current_mesh_hashes=world_update.mesh_content_hashes(new)) is None


def test_exported_mesh_path_change_with_identical_bytes_uses_pose_update(tmp_path):
    a,b=tmp_path/'a.obj',tmp_path/'b.obj'
    a.write_bytes(b'v 0 0 0\nv 1 0 0\n');b.write_bytes(a.read_bytes())
    old=scene();old['mesh']['hammer']['file_path']=str(a)
    new=deepcopy(old);new['mesh']['hammer']['file_path']=str(b)
    new['mesh']['hammer']['pose'][0]=.1
    loaded = world_update.mesh_content_hashes(old)
    def updates():
        return world_update.pose_updates(old,new,previous_mesh_hashes=loaded,
            current_mesh_hashes=world_update.mesh_content_hashes(new))
    assert updates()=={'hammer':new['mesh']['hammer']['pose']}
    # 路径不变但内容在下一次比较前改变，不能沿用上一次文件身份缓存。
    b.write_bytes(b'v 9 0 0\nv 1 0 0\n')
    assert updates() is None
    b.write_bytes(a.read_bytes());new['mesh']['hammer']['scale']=[2,1,1]
    assert updates() is None


def test_identical_mesh_files_still_validate_new_pose(tmp_path):
    a,b=tmp_path/'a.obj',tmp_path/'b.obj'
    a.write_bytes(b'geometry');b.write_bytes(a.read_bytes())
    old=scene();old['mesh']['hammer']['file_path']=str(a)
    new=deepcopy(old);new['mesh']['hammer']['file_path']=str(b)
    new['mesh']['hammer']['pose']=[0,0,0,0,0,0,0]
    with pytest.raises(ValueError):world_update.pose_updates(old,new,
        previous_mesh_hashes=world_update.mesh_content_hashes(old),
        current_mesh_hashes=world_update.mesh_content_hashes(new))


def test_rewritten_old_mesh_path_is_not_evidence_of_loaded_geometry(tmp_path):
    a,b=tmp_path/'a.obj',tmp_path/'b.obj'
    a.write_bytes(b'old geometry')
    old=scene();old['mesh']['hammer']['file_path']=str(a)
    loaded = world_update.mesh_content_hashes(old)
    a.write_bytes(b'new geometry');b.write_bytes(a.read_bytes())
    new=deepcopy(old);new['mesh']['hammer']['file_path']=str(b)
    new['mesh']['hammer']['pose'][0]=.2
    # 没有加载时内容身份，两个当前文件相同也不能证明solver里的几何相同。
    assert world_update.pose_updates(old,new) is None
    assert world_update.pose_updates(old,new,previous_mesh_hashes=loaded,
        current_mesh_hashes=world_update.mesh_content_hashes(new)) is None


def test_loaded_identity_detects_same_path_overwrite(tmp_path):
    path=tmp_path/'mesh.obj';path.write_bytes(b'original')
    old=scene();old['mesh']['hammer']['file_path']=str(path)
    loaded=world_update.mesh_content_hashes(old)
    path.write_bytes(b'replaced')
    new=deepcopy(old);new['mesh']['hammer']['pose'][0]=.2
    assert world_update.pose_updates(old,new,previous_mesh_hashes=loaded,
        current_mesh_hashes=world_update.mesh_content_hashes(new)) is None



def test_same_mesh_path_without_loaded_identity_requires_reload(tmp_path):
    path=tmp_path/'mesh.obj';path.write_bytes(b'geometry')
    old=scene();old['mesh']['hammer']['file_path']=str(path)
    new=deepcopy(old);new['mesh']['hammer']['pose'][0]=.2
    assert world_update.pose_updates(old,new) is None


def test_same_bytes_with_different_loader_format_requires_reload(tmp_path):
    a,b=tmp_path/'mesh.obj',tmp_path/'mesh.stl'
    a.write_bytes(b'same bytes');b.write_bytes(a.read_bytes())
    old=scene();old['mesh']['hammer']['file_path']=str(a)
    new=deepcopy(old);new['mesh']['hammer']['file_path']=str(b)
    assert world_update.pose_updates(old,new,previous_mesh_hashes=world_update.mesh_content_hashes(old),
        current_mesh_hashes=world_update.mesh_content_hashes(new)) is None


def test_core_scene_pose_sync_batches_all_solver_and_validator_worlds(monkeypatch):
    """规划、IK、批IK及接触分区必须同步，不能漏掉热缓存中的检查器。"""
    from types import SimpleNamespace as NS
    from gene_agent.motion.curobo.core import SynchronousPlanner, scene_collision_cache
    from gene_agent.motion.curobo import batch_world
    old = {'cuboid': {'table': {'dims': [1., 1., .1], 'pose': [0, 0, 0, 1, 0, 0, 0]}}}
    new = deepcopy(old)
    new['cuboid']['table']['pose'][0] = .2
    worlds = [object() for _ in range(6)]
    p = object.__new__(SynchronousPlanner)
    p.scene_model = old
    p.collision_cache = scene_collision_cache(old)
    p.mounting_contact_policy = {'obstacle_names': ['table']}
    p._planner, p._ik = [NS(scene_collision_checker=w) for w in worlds[:2]]
    p._ik_batches = {3: NS(scene_collision_checker=worlds[2])}
    p._collision_checker = NS(scene_model=worlds[3])
    p._validator = NS(checkers={'body': worlds[4], 'other': worlds[5]})
    p._start_validation_cache = {'stale': True}
    p.collision_mode = 'full'
    calls = []
    monkeypatch.setattr(batch_world, 'apply_batch', lambda ws, updates: calls.append((ws, updates)))
    p.update_scene(scene_model=new, scene_scope='hit', scene_snapshot_sha256='new',
                   mounting_contact_policy=p.mounting_contact_policy)
    assert calls == [([(w, {'table'} if i < 5 else set()) for i, w in enumerate(worlds)],
                      {'table': new['cuboid']['table']['pose']})]
    assert not p._start_validation_cache
    assert p.scene_model == new
