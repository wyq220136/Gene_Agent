"""小网格距离回归：远方、真实穿透、球覆盖网格及连续碰撞均与解析盒一致。"""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from gene_agent.motion.curobo.mesh_sdf import ensure_mesh_sdf_compatibility


def test_native_mesh_query_is_not_replaced(monkeypatch):
    data_mesh = pytest.importorskip("curobo._src.geom.data.data_mesh")
    def native(obs_set, env_idx, local_idx, local_pt, query_distance):
        raise AssertionError("只检查签名，不应执行查询")
    monkeypatch.setattr(data_mesh, "compute_local_sdf_with_grad", native)
    assert ensure_mesh_sdf_compatibility()["mode"] == "native_query_radius"
    assert data_mesh.compute_local_sdf_with_grad is native


def test_unknown_mesh_api_fails_closed(monkeypatch):
    data_mesh = pytest.importorskip("curobo._src.geom.data.data_mesh")
    monkeypatch.setattr(data_mesh, "compute_local_sdf_with_grad", lambda unexpected: None)
    with pytest.raises(RuntimeError, match="不支持.*mesh SDF"):
        ensure_mesh_sdf_compatibility()


def test_legacy_patch_updates_both_loaded_kernels_once(monkeypatch):
    from gene_agent.motion.curobo import mesh_sdf
    data_mesh = pytest.importorskip("curobo._src.geom.data.data_mesh")
    def legacy(obs_set, env_idx, local_idx, local_pt):
        raise AssertionError("旧查询不得执行")
    monkeypatch.setattr(data_mesh, "compute_local_sdf_with_grad", legacy)
    calls = []
    for name in mesh_sdf._KERNEL_MODULES:
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__name__=name))
    monkeypatch.setattr(mesh_sdf, "_register_mesh_overload",
                        lambda wp, kernel, function: calls.append(kernel.__name__))
    assert ensure_mesh_sdf_compatibility()["mode"] == "legacy_exact_surface_query"
    assert ensure_mesh_sdf_compatibility()["mode"] == "legacy_exact_surface_query"
    assert calls == list(mesh_sdf._KERNEL_MODULES)


def _collision_cost(scene, spheres, activation, swept):
    import torch
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
    from curobo.types import DeviceCfg
    device = DeviceCfg()
    checker = SceneCollision.from_config(SceneCollisionCfg(device_cfg=device, scene_model=scene))
    query = torch.tensor(spheres, device=device.device, dtype=torch.float32, requires_grad=True)
    buffer = CollisionBuffer.from_shape(query.shape, device)
    arguments = dict(query_spheres=query, collision_buffer=buffer,
                     weight=torch.tensor([1.], device=device.device),
                     activation_distance=torch.tensor([activation], device=device.device))
    if swept:
        costs = checker.get_swept_sphere_distance_raw(
            **arguments, trajectory_dt=torch.tensor([.1], device=device.device),
        )
    else:
        costs = checker.get_sphere_distance_raw(**arguments)
    result = costs.detach().clone()
    costs.sum().backward()
    return result, query.grad.detach().clone()


@pytest.mark.parametrize("activation", [0., .01])
@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize("offset", [False, True])
def test_gpu_mesh_matches_analytic_box_without_ignoring_small_objects(activation, swept, offset):
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("需要真实 CUDA 碰撞查询")
    ensure_mesh_sdf_compatibility()
    from curobo._src.geom.types import Cuboid, Mesh, SceneCfg
    pose = [.3, -.2, .1, 1., 0., 0., 0.]
    box = Cuboid(name="small_box", dims=[.05]*3, pose=pose)
    geometry = box.get_trimesh_mesh()
    shift = np.array([.2, -.1, .3]) if offset else np.zeros(3)
    # 相同世界几何，mesh 局部坐标故意不居中，检验查询界的推导。
    mesh = Mesh(name="small_box", vertices=(geometry.vertices + shift).tolist(),
                faces=geometry.faces.reshape(-1).tolist(),
                pose=[*(np.asarray(pose[:3])-shift), *pose[3:]])
    x = [.005, .04, .08, .10, .50, 1.00]
    row = [[pose[0]+value, pose[1], pose[2], .05] for value in x]
    spheres = [[row, row]] if swept else [[row]]
    expected, expected_grad = _collision_cost(SceneCfg(cuboid=[box]), spheres, activation, swept)
    observed, observed_grad = _collision_cost(SceneCfg(mesh=[mesh]), spheres, activation, swept)
    torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-4)
    torch.testing.assert_close(observed_grad[..., :3], expected_grad[..., :3], atol=1e-4, rtol=1e-3)
    assert (observed[..., :2] > 0).all()  # 球覆盖小网格、真实表面穿透必须拦截。
    assert (observed[..., 3:] == 0).all()  # 距离 0.1/0.5/1m 的球均应放行。


def test_gpu_sweep_detects_crossing_with_clear_endpoints():
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("需要真实 CUDA 碰撞查询")
    ensure_mesh_sdf_compatibility()
    from curobo._src.geom.types import Cuboid, Mesh, SceneCfg
    box = Cuboid(name="wall", dims=[.05]*3, pose=[0., 0., 0., 1., 0., 0., 0.])
    geometry = box.get_trimesh_mesh()
    mesh = Mesh(name="wall", vertices=geometry.vertices.tolist(),
                faces=geometry.faces.reshape(-1).tolist(), pose=box.pose)
    spheres = [[[[-.1, 0., 0., .05]], [[.1, 0., 0., .05]]]]
    scene = SceneCfg(mesh=[mesh])
    static, _ = _collision_cost(scene, spheres, 0., False)
    swept, _ = _collision_cost(scene, spheres, 0., True)
    assert (static == 0).all()
    assert (swept > 0).any()
