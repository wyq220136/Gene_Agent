"""刷子/托盘允许接触不能放行侧壁穿透。"""
from pathlib import Path
import numpy as np
import mujoco
import pytest
from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun
from gene_agent.tasks.policies.sweep_cube_task.execution.curobo import check_brush_pan_walls, check_observed_brush_pan_walls

SCENE = Path(__file__).resolve().parents[4] / "data/realworld/brush_cube_dustpan_0"


@pytest.mark.parametrize("wall_name", ["dustpan_wall_-1_collision", "dustpan_back_collision"])
def test_wall_check_rejects_overlap_and_keeps_live_state(tmp_path, monkeypatch, wall_name):
    monkeypatch.setattr(SweepCubeRun, "_render", lambda *args: None)
    run = SweepCubeRun(SCENE, tmp_path / "run")
    before = run.data.qpos.copy()
    observation = run.observe()
    tcp = np.asarray(observation["robots"][run.group]["T_world_tcp"])
    brush = np.asarray(observation["objects"]["brush"]["T_world_object"]).copy()
    # 将实际刷子凸包中心对准墙中心，构造确定的穿透样本。
    wall = run.model.geom(wall_name).id
    geom = run.model.geom("brush_collision_00").id
    brush[:3, 3] += run.data.geom_xpos[wall] - run.data.geom_xpos[geom]
    q = run.data.qpos[run.planner.qpos_addresses].copy()
    collisions = check_brush_pan_walls(run.planner, [q], np.linalg.inv(tcp) @ brush)
    assert collisions
    preview = run.planner._copy()
    run.planner._set_attached(preview, np.linalg.inv(tcp) @ brush, "brush")
    mujoco.mj_forward(run.model, preview)
    assert check_observed_brush_pan_walls(run.planner, [{"qpos": preview.qpos.copy()}])
    assert all(row["reason"] == "brush_penetrates_rigid_dustpan_wall" for row in collisions)
    brush[2, 3] += .5
    assert not check_brush_pan_walls(run.planner, [q], np.linalg.inv(tcp) @ brush)
    np.testing.assert_array_equal(run.data.qpos, before)


def test_contact_height_uses_mesh_and_moves_whole_sequence():
    from gene_agent.tasks.policies.sweep_cube_task.execution.curobo import correct_sequence_table_clearance
    low = np.eye(4); low[2, 3] = -.005
    high = low.copy(); high[2, 3] += .05
    row = {"targets": {"lower_to_contact": low.tolist(), "retreat": high.tolist()},
           "face_low_z_m": .002, "face_center_world_xyz": [0., 0., .012]}
    corrected = correct_sequence_table_clearance(row, np.eye(4), np.array([[0., 0., 0.]]), 0.)
    assert abs(corrected["collision_low_z_m"] - .001) < 1e-9
    assert abs(corrected["targets"]["retreat"][2][3] - .051) < 1e-9
    assert row["targets"]["lower_to_contact"][2][3] == -.005


def test_push_contact_uses_actual_surface_normal_and_does_not_step_physics():
    from types import SimpleNamespace
    from gene_agent.tasks.policies.sweep_cube_task.execution.curobo import inspect_push_contact
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="brush"><freejoint name="brush_free"/><geom name="brush_collision_00" type="box" size=".01 .01 .01"/></body>
      <body name="cube"><geom name="cube_collision" type="box" size=".01 .01 .01"/></body>
      </worldbody></mujoco>''')
    planner = SimpleNamespace(model=model, _copy=lambda: mujoco.MjData(model))
    start = np.eye(4); start[0, 3] = -.03
    end = start.copy(); end[0, 3] += .1
    good = inspect_push_contact(planner, start, end, np.eye(4), "cube")
    assert good["success"] and good["alignment_cosine"] > .99
    short_end = start.copy(); short_end[0, 3] += .005
    assert not inspect_push_contact(planner, start, short_end, np.eye(4), "cube")["success"]
    start[:2, 3] = [-.035, -.015]
    end = start.copy(); end[:2, 3] += .1
    bad = inspect_push_contact(planner, start, end, np.eye(4), "cube")
    assert not bad["success"]
    assert bad["reason"] == "push_face_not_aligned"


def test_fine_brush_proxy_preserves_mesh_vertex_coverage():
    from gene_agent.tasks.policies.sweep_cube_task.execution.curobo import brush_collision_vertices_object, brush_collision_spheres_object
    model = mujoco.MjModel.from_xml_path(str(SCENE / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    vertices = brush_collision_vertices_object(model, data)
    spheres = brush_collision_spheres_object(model, data, max_cell_m=.012)
    distances = np.linalg.norm(vertices[:, None, :] - spheres[None, :, :3], axis=-1)
    assert np.all(np.any(distances <= spheres[None, :, 3] + 1e-7, axis=1))
    assert spheres[:, 3].max() < .011


def test_push_face_alignment_separates_lateral_slip_from_vertical_tilt():
    from gene_agent.tasks.policies.sweep_cube_task.execution.curobo import push_normal_alignment
    direction = np.array([1., 0., 0.])
    # 16度下倾，但水平方向完全正确，不应当作侧向推歪。
    good = push_normal_alignment([[.96, 0., -.28]], direction)
    assert good['success']
    assert good['horizontal_alignment_cosine'] == pytest.approx(1.)
    assert good['alignment_cosine'] == pytest.approx(.96)
    # 水平偏20度仍拒绝；过于竖直的接触也不能通过。
    assert not push_normal_alignment([[.94, .342, 0.]], direction)['success']
    assert not push_normal_alignment([[.6, 0., -.8]], direction)['success']
    assert not push_normal_alignment([[0., 0., 1.]], direction)['success']
