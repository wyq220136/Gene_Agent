"""动态夹爪同步的缓存复用；不允许因缓存忽略实际碰撞几何。"""
from copy import deepcopy
from gene_agent.motion.curobo.session import CuroboPlannerPool

class FakePlanner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.seed_resets = 0
    def update_scene(self, **kwargs):
        pass
    def refresh_scene_metadata(self, **kwargs):
        pass
    def reset_seed(self):
        self.seed_resets += 1
    def close(self):
        self.closed = True

def _settings():
    return {"robot_config": {"kinematics": {
        "collision_spheres": {"tool": [{"center": [0,0,0], "radius": .01}]},
        "lock_joints": {"finger": .04}}}}

def _snapshot():
    return {"scene_model": {"cuboid": {}, "mesh": {}},
            "scope": "test", "snapshot_sha256": "scene"}

def _dynamic_settings(offset=0.):
    from gene_agent.motion.curobo.gripper import _digest
    settings = _settings()
    added = [{"center": [offset, 0., 0.], "radius": .004}]
    settings["robot_config"]["kinematics"]["collision_spheres"]["tool"].extend(added)
    gripper = {"schema_version": 1, "mode": "parallel_slide_proxy", "group": "right",
               "source_hashes": {"urdf": "unchanged"}, "source_robot_config_sha256": "source",
               "width_m": .08 + offset, "added_spheres_by_link": {"tool": added},
               "merged_sphere_sha256": _digest(settings["robot_config"]["kinematics"]["collision_spheres"])}
    gripper["snapshot_id"] = _digest(gripper)
    settings["gripper_snapshot"] = gripper
    return settings


class UpdatingPlanner(FakePlanner):
    def update_gripper_geometry(self, **kwargs):
        self.kwargs.update(kwargs)


def test_dynamic_gripper_reuse_updates_exact_spheres_before_planning():
    pool = CuroboPlannerPool(planner_factory=UpdatingPlanner, reuse_gripper_geometry=True)
    first = pool.acquire(_dynamic_settings(), _snapshot())
    second_settings = _dynamic_settings(.0001)
    second = pool.acquire(second_settings, _snapshot())
    assert second.cache_hit
    assert second.planner is first.planner
    assert second.planner.kwargs["robot_config"] == second_settings["robot_config"]
    assert second.planner.kwargs["gripper_snapshot"] == second_settings["gripper_snapshot"]


def test_dynamic_reuse_keeps_arm_spheres_and_lock_joints_in_identity():
    pool = CuroboPlannerPool(planner_factory=UpdatingPlanner, reuse_gripper_geometry=True)
    first = pool.acquire(_dynamic_settings(), _snapshot())
    changed = _dynamic_settings(.0001)
    changed["robot_config"]["kinematics"]["lock_joints"]["finger"] = .02
    assert not pool.acquire(changed, _snapshot()).cache_hit
    changed = _dynamic_settings(.0002)
    changed["robot_config"]["kinematics"]["collision_spheres"]["tool"][0]["radius"] = .02
    from gene_agent.motion.curobo.gripper import _digest
    changed["gripper_snapshot"]["merged_sphere_sha256"] = _digest(changed["robot_config"]["kinematics"]["collision_spheres"])
    changed["gripper_snapshot"]["snapshot_id"] = _digest({k:v for k,v in changed["gripper_snapshot"].items() if k != "snapshot_id"})
    assert not pool.acquire(changed, _snapshot()).cache_hit
    assert not first.planner.closed


def test_failed_dynamic_update_discards_all_cached_solver_state():
    class Broken(UpdatingPlanner):
        def update_gripper_geometry(self, **kwargs):
            raise RuntimeError("partially updated")
    pool = CuroboPlannerPool(planner_factory=Broken, reuse_gripper_geometry=True)
    first = pool.acquire(_dynamic_settings(), _snapshot())
    second = pool.acquire(_dynamic_settings(.0001), _snapshot())
    assert not second.cache_hit
    assert second.rebuild_reason == "gripper_update_failed"
    assert first.planner.closed
    assert second.planner is not first.planner


def test_reusing_new_gripper_state_restarts_sampling_without_discarding_graph():
    pool = CuroboPlannerPool(planner_factory=UpdatingPlanner, reuse_gripper_geometry=True)
    first = pool.acquire(_dynamic_settings(), _snapshot())
    second = pool.acquire(_dynamic_settings(.0001), _snapshot())
    assert second.cache_hit
    assert second.planner.seed_resets == 1
    pool.acquire(_dynamic_settings(.0001), _snapshot())
    assert second.planner.seed_resets == 1


def test_pool_defaults_to_exact_gripper_reuse_with_explicit_opt_out():
    pool = CuroboPlannerPool(planner_factory=UpdatingPlanner)
    first = pool.acquire(_dynamic_settings(), _snapshot())
    second = pool.acquire(_dynamic_settings(.0001), _snapshot())
    assert second.cache_hit and second.planner is first.planner
    diagnostic = CuroboPlannerPool(planner_factory=UpdatingPlanner, reuse_gripper_geometry=False)
    diagnostic.acquire(_dynamic_settings(), _snapshot())
    assert not diagnostic.acquire(_dynamic_settings(.0001), _snapshot()).cache_hit


def test_pool_context_disposes_cached_planners_on_exception():
    import pytest
    with pytest.raises(RuntimeError,match='stage failed'):
        with CuroboPlannerPool(planner_factory=FakePlanner) as pool:
            lease=pool.acquire(_settings(),_snapshot())
            assert lease.to_dict()['cache_hit'] is False
            raise RuntimeError('stage failed')
    assert lease.planner.closed and not pool._entries


def _aloha_settings(width=.02):
    from gene_agent.robots.aloha import digest
    s=_settings();kin=s['robot_config']['kinematics']
    kin['lock_joints']={'finger':width,'fixed_other':.1}
    kin['cspace']={'joint_names':['arm','finger'],'default_joint_position':[0.,width]}
    g=dict(schema_version=1,mode='aloha_parallel',group='right',source_hashes={'urdf':'same'},source_robot_config_sha256='source',joint_names=['finger'],joint_positions_m={'finger':width},width_m=width,merged_sphere_sha256=digest(kin['collision_spheres']))
    g['snapshot_id']=digest(g);s['gripper_snapshot']=g
    return s


def test_aloha_exact_lock_update_reuses_but_other_locks_do_not():
    pool=CuroboPlannerPool(planner_factory=UpdatingPlanner)
    first=pool.acquire(_aloha_settings(),_snapshot())
    changed=_aloha_settings(.021)
    second=pool.acquire(changed,_snapshot())
    assert second.cache_hit and second.planner is first.planner
    assert second.planner.kwargs['robot_config']==changed['robot_config']
    changed=_aloha_settings(.022);changed['robot_config']['kinematics']['lock_joints']['fixed_other']=.2
    assert not pool.acquire(changed,_snapshot()).cache_hit


class HeldUpdatingPlanner(UpdatingPlanner):
    def update_held_geometry(self,**kwargs):self.kwargs.update(kwargs)


def _held_settings(x=0.,radius=.01):
    s=_aloha_settings();s['held_object']=dict(object_id='tool',finger_links=['finger_link'],spheres_link=[[x,0.,0.,radius]],sphere_sources=[{'source_shape_id':'original'}])
    return s


def test_held_centers_update_even_when_gripper_and_snapshot_are_unchanged():
    pool=CuroboPlannerPool(planner_factory=HeldUpdatingPlanner)
    first=pool.acquire(_held_settings(),_snapshot())
    new=_held_settings(.005)
    second=pool.acquire(new,_snapshot())
    assert second.cache_hit and second.planner is first.planner
    assert second.planner.kwargs['held_object']==new['held_object']
    assert second.planner.seed_resets==1
    assert not pool.acquire(_held_settings(.005,.02),_snapshot()).cache_hit
    other=_held_settings();other['held_object']['object_id']='different'
    assert not pool.acquire(other,_snapshot()).cache_hit


def test_failed_held_update_discards_instance():
    class Broken(HeldUpdatingPlanner):
        def update_held_geometry(self,**kwargs):raise RuntimeError('partial update')
    pool=CuroboPlannerPool(planner_factory=Broken)
    first=pool.acquire(_held_settings(),_snapshot())
    second=pool.acquire(_held_settings(.005),_snapshot())
    assert not second.cache_hit and first.planner.closed
    assert second.rebuild_reason=='held_geometry_update_failed'


def test_held_source_and_contact_policy_are_not_merged():
    pool=CuroboPlannerPool(planner_factory=HeldUpdatingPlanner)
    pool.acquire(_held_settings(),_snapshot())
    changed=_held_settings(.001)
    changed['held_object']['sphere_sources'][0]['source_shape_id']='different_mesh'
    assert not pool.acquire(changed,_snapshot()).cache_hit
    snapshot=_snapshot();snapshot['mounting_contact_policy']={'robot_links':['gene_held_tool'],'obstacle_names':['table']}
    assert not pool.acquire(_held_settings(),snapshot).cache_hit


def test_dynamic_held_opt_out_and_invalid_centers():
    import pytest
    pool=CuroboPlannerPool(planner_factory=HeldUpdatingPlanner,reuse_gripper_geometry=False)
    pool.acquire(_held_settings(),_snapshot())
    assert not pool.acquire(_held_settings(.001),_snapshot()).cache_hit
    invalid=_held_settings(float('nan'))
    with pytest.raises(ValueError):
        CuroboPlannerPool(planner_factory=HeldUpdatingPlanner).acquire(invalid,_snapshot())


def test_aloha_lock_snapshot_mismatch_is_rejected():
    import pytest
    settings=_aloha_settings();settings['robot_config']['kinematics']['lock_joints']['finger']=.03
    with pytest.raises(ValueError,match='锁关节'):
        CuroboPlannerPool(planner_factory=UpdatingPlanner).acquire(settings,_snapshot())


class PolicyUpdatingPlanner(HeldUpdatingPlanner):
    supports_contact_policy_update = True
    def update_contact_policy(self, policy):
        self.kwargs['mounting_contact_policy'] = deepcopy(policy)


def test_supported_policy_switch_reuses_resources_but_replaces_rules():
    pool=CuroboPlannerPool(planner_factory=PolicyUpdatingPlanner)
    allowed=_snapshot();allowed['mounting_contact_policy']={'robot_links':['tool'],'obstacle_names':['table']}
    first=pool.acquire(_held_settings(),allowed)
    strict=pool.acquire(_held_settings(),_snapshot())
    assert strict.cache_hit and strict.planner is first.planner
    assert strict.planner.kwargs['mounting_contact_policy']=={}
    assert strict.planner.seed_resets==1
    back=pool.acquire(_held_settings(),allowed)
    assert back.cache_hit
    assert back.planner.kwargs['mounting_contact_policy']==allowed['mounting_contact_policy']
    assert back.planner.seed_resets==2


def test_failed_policy_switch_discards_solver_and_opt_out_keeps_isolation():
    class Broken(PolicyUpdatingPlanner):
        def update_contact_policy(self,policy):raise RuntimeError('partial policy update')
    allowed=_snapshot();allowed['mounting_contact_policy']={'robot_links':['tool'],'obstacle_names':['table']}
    pool=CuroboPlannerPool(planner_factory=Broken)
    first=pool.acquire(_held_settings(),allowed)
    second=pool.acquire(_held_settings(),_snapshot())
    assert first.planner.closed and not second.cache_hit
    assert second.rebuild_reason=='contact_policy_update_failed'
    isolated=CuroboPlannerPool(planner_factory=PolicyUpdatingPlanner,reuse_contact_policy=False)
    isolated.acquire(_held_settings(),allowed)
    assert not isolated.acquire(_held_settings(),_snapshot()).cache_hit
