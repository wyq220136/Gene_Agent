"""统一快速规划模式的隔离与真实导出行为；CPU 测试不代替 GPU 验收。"""
from copy import deepcopy

import mujoco
import pytest

from gene_agent.motion.configuration import (
    CollisionMode, current_motion_config, motion_configuration, configured_motion,
)


def test_config_scope_restores_and_nested_entry_inherits():
    @configured_motion("site")
    def entry(site):
        return current_motion_config().collision_mode
    assert entry({}) is CollisionMode.FULL
    with motion_configuration({"collision_mode": "robot_only"}):
        assert entry({}) is CollisionMode.ROBOT_ONLY
        assert entry({"motion": {"collision_mode": "full"}}) is CollisionMode.ROBOT_ONLY
    assert entry({}) is CollisionMode.FULL
    with pytest.raises(ValueError):
        with motion_configuration({"collision_mode": "typo"}):
            pass


def _scene():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="arm"><joint name="j" type="hinge"/>
        <geom type="box" size=".01 .01 .1" mass="1"/></body>
      <body name="support" pos="0 0 -.3"><geom name="base_proxy" type="box" size=".1 .1 .1"/></body>
      <body name="object"><freejoint/><geom name="environment" type="sphere" size=".2"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    meta = {"joint_groups": {"right": ["j"]}, "group_root_bodies": {"right": "arm"},
            "world_frame": "world", "transforms": {"right": {"world_from_urdf_base":
                [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}},
            "collision_protection_body_names": ["support"]}
    return model, data, meta


def test_robot_only_does_not_export_scene_or_transfer_objects(tmp_path, monkeypatch):
    from gene_agent.motion.curobo import scene
    model, data, meta = _scene()
    monkeypatch.setattr(scene, "_mesh_for_geom", lambda *a: pytest.fail("不应网格化场景"))
    with motion_configuration({"collision_mode": "robot_only"}):
        snap = scene.export_collision_scene(model, data, meta, "right", tmp_path,
            transferred_objects={"object": [[0,0,0,.1]]})
        derived = scene.filter_collision_snapshot(snap, tmp_path / "derived",
            keep_body_names=set(), rejected_body_names=None, attached_object_id="object", scope="stage")
        self_only = scene.export_self_collision_only_snapshot(snap, tmp_path / "self", attached_object_id="object")
    assert snap["collision_mode"] == "robot_only"
    assert snap['disabled_body_names'] == snap['moving_body_names'] == []
    assert snap['active_group_root_body'] == 'arm'
    assert snap['static_internal_contact_pairs'] == []
    assert not snap["transferred_objects"]
    assert [r["geom_name"] for r in snap["records"]] == ["base_proxy"]
    assert derived["scene_model"] == snap["scene_model"] == self_only["scene_model"]
    assert list(tmp_path.glob("*.obj")) == []
    assert all(p.suffix == ".json" for p in tmp_path.rglob("*.*"))


def test_pool_mode_switch_isolates_cached_world_and_attachment():
    from gene_agent.motion.curobo.session import CuroboPlannerPool
    class Planner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def close(self):
            pass
    pool = CuroboPlannerPool(planner_factory=Planner)
    settings = {"attachment_spheres_tool": [[0,0,0,.1]]}
    snap = {"scene_model": {"cuboid": {"obstacle": {"dims": [1,1,1], "pose": [0,0,0,1,0,0,0]}}},
            "scope": "full", "snapshot_sha256": "full"}
    full = pool.acquire(settings, snap)
    robot = deepcopy(snap)
    robot.update(collision_mode="robot_only", scene_model={}, scope="robot", snapshot_sha256="robot")
    fast = pool.acquire(settings, robot)
    assert fast.planner is not full.planner
    assert fast.planner.kwargs.get("attachment_spheres_tool") is None
    assert fast.planner.kwargs["collision_mode"] == "robot_only"
    assert pool.acquire(settings, robot).cache_hit
    assert pool.acquire(settings, snap).planner is full.planner


def test_bridge_fast_mode_skips_attachment_and_environment_callbacks(tmp_path, monkeypatch):
    from gene_agent.motion.curobo import runtime_bridge as bridge
    from tests.motion.test_motion_runtime_bridge import Runtime
    runtime = Runtime()
    snap = {"scene_model": {}, "scope": "robot", "snapshot_sha256": "robot",
            "collision_mode": "robot_only", "gripper_snapshot": {}, "mounting_contact_policy": {}}
    monkeypatch.setattr(bridge, "export_collision_scene", lambda *a, **kw: snap)
    monkeypatch.setattr(bridge, "load_realman_motion_config", lambda *a, **kw: {})
    class Planner:
        def __init__(self, **kw):
            assert kw.get("attachment_spheres_tool") is None
        def plan(self, *a, **kw):
            return {"success": True, "joint_names": ["j1"],
                    "positions_rad": [[.2],[.3]], "interpolation_dt_s": .025}
    monkeypatch.setattr(bridge, "SynchronousPlanner", Planner)
    def unexpected(*a):
        pytest.fail("robot_only 不应调用 attachment/场景筛选/场景几何预检查")
    with motion_configuration({"collision_mode": "robot_only"}):
        result = bridge.CuroboRuntimeBridge(runtime, tmp_path, attachment_for_stage=unexpected,
            snapshot_filter_for_stage=unexpected, contact_policy_for_stage=unexpected,
            plan_postcheck_for_stage=unexpected).plan_only("test", __import__('numpy').eye(4), "state-1")
    assert result["success"]
    assert result["plan"]["collision_mode"] == "robot_only"


def test_fast_scene_options_and_surface_sampling_do_not_add_attachment(tmp_path):
    import numpy as np
    from gene_agent.motion.curobo.scene import planner_scene_options, export_collision_scene
    from gene_agent.motion.curobo.attachment import build_attachment_surface_spheres_from_mesh
    model, data, meta = _scene()
    with motion_configuration({"collision_mode": "robot_only"}):
        snapshot = export_collision_scene(model, data, meta, 'right', tmp_path)
        centers, radii, _ = build_attachment_surface_spheres_from_mesh(None, num_spheres=32, radius=.002)
    options = planner_scene_options(snapshot)
    assert options['robot_protection_scene'] == snapshot['scene_model']
    assert centers.shape == (0,3) and radii.shape == (0,)
    # 转换物体刚性关系仍然可用；空代理不会导致坐标计算维度错位。
    assert (np.eye(3) @ centers.T).T.shape == (0,3)


def test_fast_derived_scene_keeps_explicit_planning_gripper_width(tmp_path):
    from gene_agent.motion.curobo.scene import export_self_collision_only_snapshot
    source = {"collision_mode": "robot_only", "snapshot_sha256": "before", "scene_model": {},
              "gripper_snapshot": {"snapshot_id": "closed"}}
    updated = export_self_collision_only_snapshot(source, tmp_path,
        gripper_snapshot={"snapshot_id": "open"})
    assert updated["gripper_snapshot"]["snapshot_id"] == "open"
    assert updated["snapshot_sha256"] != source["snapshot_sha256"]
    assert source["gripper_snapshot"]["snapshot_id"] == "closed"


def test_pika_static_geometry_cached_but_source_edits_invalidate(tmp_path, monkeypatch):
    from gene_agent.motion.curobo.profiles import pika
    source = tmp_path/'robot.yml'
    source.write_bytes(pika.SOURCE_CONFIG.read_bytes())
    monkeypatch.setattr(pika, 'SOURCE_CONFIG', source)
    meta = {'planning_urdf_path': '/tmp/offline_robot.urdf'}
    calls = []
    original = pika.trimesh.load_mesh
    def read(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(pika.trimesh, 'load_mesh', read)
    first = pika._pika_config(meta)
    second = pika._pika_config(meta)
    assert len(calls) == 3
    first['kinematics']['collision_spheres'].clear()
    assert second['kinematics']['collision_spheres']  # 缓存值不能被调用方污染
    source.write_bytes(source.read_bytes()+b'\n# cache invalidation\n')
    assert pika._pika_config(meta) == second
    assert len(calls) == 6


def test_robot_only_scene_identity_survives_constructor_for_typed_adapter(monkeypatch):
    import sys
    from types import SimpleNamespace
    import numpy as np
    from gene_agent.motion.curobo.core import SynchronousPlanner, CuroboPlannerAdapter
    from gene_agent.motion.curobo.scene import planner_scene_options
    from gene_agent.motion import JointState, MotionRequest, MotionOperation, PoseTarget, PlanningSceneRef
    monkeypatch.setitem(sys.modules, 'curobo.kinematics', SimpleNamespace(
        Kinematics=lambda cfg: SimpleNamespace(joint_names=['j']),
        KinematicsCfg=SimpleNamespace(from_robot_yaml_file=lambda cfg: cfg)))
    snapshot = {'collision_mode': 'robot_only', 'scene_model': {},
                'scope': 'robot_only explicit protection scope', 'snapshot_sha256': 'same-scene'}
    planner = SynchronousPlanner(robot_config={'kinematics': {}}, joint_names=['j'],
        planner_joint_names=['j'], world_from_base=np.eye(4), link_from_tcp=np.eye(4), tool_frame='tool',
        **planner_scene_options(snapshot))
    request = MotionRequest('test', MotionOperation.IK, 'right', JointState(('j',),(0.,)),
        PoseTarget('world_tcp',np.eye(4)), PlanningSceneRef(0,'same-scene',snapshot['scope']), seed=123)
    adapter = CuroboPlannerAdapter(planner,group='right',scene_revision=0)
    assert adapter._validate(request,MotionOperation.IK) is None


def test_visual_insert_skips_missing_attachment_spheres_in_robot_only():
    import numpy as np
    from gene_agent.tasks.policies.microwave_task.planning.insert import _sphere_link_from_world
    snapshot = {'collision_mode': 'robot_only', 'transferred_objects': {}}
    assert _sphere_link_from_world(snapshot,np.eye(4),{}) is None
