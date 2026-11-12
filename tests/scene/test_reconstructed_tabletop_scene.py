"""实景资产的桌面场景应有稳定局部 frame，两个 viewer 读取同一布局。"""

import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from gene_agent.scene.reconstructed_tabletop import (
    SOURCE_ORDER, canonicalize_mesh, build_tabletop_scene,
)
from gene_agent.scene.tabletop_randomization import sample_tabletop_layout


def test_canonical_mesh_uses_long_axis_forward_and_bottom_on_table(tmp_path: Path):
    source = tmp_path / "source.obj"
    destination = tmp_path / "canonical.obj"
    trimesh.creation.box(extents=[0.04, 0.02, 0.20]).export(source)

    result = canonicalize_mesh(source, destination)
    mesh = trimesh.load_mesh(destination, process=False)

    assert np.allclose(mesh.extents, [0.20, 0.04, 0.02], atol=1e-5)
    assert np.isclose(mesh.bounds[0, 2], 0.0, atol=1e-6)
    assert result["mesh_path"] == str(destination.resolve())


def test_built_scene_compiles_and_places_three_objects_on_support(tmp_path: Path):
    sources = {}
    for name, size in {
        "brush": [0.04, 0.08, 0.23],
        "cube": [0.0224, 0.0224, 0.0224],
        "dustpan": [0.15, 0.02, 0.17],
    }.items():
        path = tmp_path / f"{name}.obj"
        trimesh.creation.box(extents=size).export(path)
        sources[name] = {"mesh_path": str(path), "source_kind": "test"}

    scene = build_tabletop_scene(tmp_path / "scene", sources=sources)
    task = json.loads((tmp_path / "scene" / "task.json").read_text())
    model = mujoco.MjModel.from_xml_path(scene["scene_xml_path"])
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    assert scene["robot"]["profile"] == "realman_single_pika"
    assert task["status"] == "mujoco_skill_available"
    assert task["execution_status"] == "mujoco_seeded_evaluation_available"
    assert task["success_condition"]["implemented"]
    assert len(scene["objects"]) == 5
    assert {"cube", "cube_red", "cube_blue"} <= set(scene["objects"])
    assert np.allclose(scene["objects"]["cube_red"]["extents_xyz_m"],
                       scene["objects"]["cube"]["extents_xyz_m"])
    assert np.allclose(scene["objects"]["cube_blue"]["extents_xyz_m"],
                       scene["objects"]["cube"]["extents_xyz_m"])
    assert model.geom("cube_red_visual").rgba[0] > model.geom("cube_red_visual").rgba[2]
    assert model.geom("cube_blue_visual").rgba[2] > model.geom("cube_blue_visual").rgba[0]
    assert all(scene["reachability_probe"][name]["position_error_m"] < 0.01
               for name in ("brush", "cube", "dustpan"))
    assert [scene["objects"][name]["T_world_object"][1][3]
            for name in ("brush", "cube", "dustpan")] == [-0.26, -0.09, 0.12]
    brush_size = scene["objects"]["brush"]["extents_xyz_m"]
    assert brush_size[0] > 2.5 * brush_size[2]
    assert brush_size[2] > brush_size[1]  # 刷头厚度立起来，刷毛面朝桌面。
    assert scene["objects"]["brush"]["brush_face_normal_object"] == [0, 0, -1]
    brush_mesh = trimesh.load_mesh(tmp_path / "scene" / scene["objects"]["brush"]["mesh_path"])
    assert np.isclose(brush_mesh.bounds[0, 2], 0.0, atol=1e-6)
    source_brush = trimesh.load_mesh(sources["brush"]["mesh_path"])
    transform = np.asarray(scene["objects"]["brush"]["T_object_source_mesh"])
    transformed = source_brush.vertices @ transform[:3, :3].T + transform[:3, 3]
    assert np.allclose(np.vstack((transformed.min(axis=0), transformed.max(axis=0))),
                       brush_mesh.bounds, atol=1e-6)
    # 簸箕的局部 +X 是铲口；世界朝向应指向位于 -Y 的方块。
    dustpan_front = np.asarray(scene["objects"]["dustpan"]["T_world_object"])[:3, 0]
    assert np.dot(dustpan_front, [0, -1, 0]) > 0.99
    assert np.linalg.norm(data.body("dustpan").xmat.reshape(3, 3)[:, 0] - dustpan_front) < 1e-6
    assert "dustpan_back_collision" in [model.geom(i).name for i in range(model.ngeom)]
    assert model.geom("dustpan_back_collision").pos[0] < 0
    tray = model.geom("dustpan_tray_collision")
    assert tray.pos[2] + tray.size[2] <= .001  # 铲口与桌面齐平，方块可滑入。
    for name in ("brush", "cube", "dustpan"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
        assert abs(float(data.body(name).xpos[2])) < 0.01
    assert "pika_tcp" in [model.site(i).name for i in range(model.nsite)]
    assert np.linalg.norm(data.site("pika_tcp").xpos - [0.22, 0, 0.46]) < 0.02
    for index in range(1, 8):
        joint = model.joint(f"joint_{index}")
        actuator = model.actuator(f"joint_{index}_position")
        assert np.isclose(data.ctrl[actuator.id], data.qpos[joint.qposadr[0]])


def test_real_brush_collision_preserves_handle_gap_and_mass(tmp_path: Path):
    """防止把刷子退回整件包围 box，导致柄附近的空隙被填满。"""
    original = Path(__file__).resolve().parents[2] / "data/realworld/brush_cube_dustpan_0"
    existing = json.loads((original / "scene.json").read_text())
    sources = {
        name: {"mesh_path": row["source_mesh_path"], "source_kind": "test"}
        for name, row in existing["objects"].items()
        if name in SOURCE_ORDER
    }
    scene = build_tabletop_scene(tmp_path / "scene", sources=sources)
    model = mujoco.MjModel.from_xml_path(scene["scene_xml_path"])
    collision = scene["objects"]["brush"]["collision"]
    parts = scene["objects"]["brush"]["collision_parts"]

    assert collision == "vhacd_convex_parts"
    assert 2 <= len(parts) <= 4
    assert np.isclose(model.body("brush").mass, 0.12, atol=1e-6)
    weld = model.equality("sim_attach_left_brush")
    assert int(weld.type[0]) == int(mujoco.mjtEq.mjEQ_WELD)
    assert not bool(weld.active0[0])
    assert all((tmp_path / "scene" / part["mesh_path"]).is_file() for part in parts)
    assert all(model.geom(part["geom_name"]).type == mujoco.mjtGeom.mjGEOM_MESH
               for part in parts)
    # 真实刷子是长柄连接薄刷头；凸块总体积必须明显小于整件包围盒。
    box_volume = np.prod(scene["objects"]["brush"]["extents_xyz_m"])
    assert sum(part["volume_m3"] for part in parts) < 0.5 * box_volume


def test_approved_wide_randomization_can_put_cube_off_pan_axis():
    scene_dir = Path(__file__).resolve().parents[2] / "data/realworld/brush_cube_dustpan_0"
    scene = json.loads((scene_dir / "scene.json").read_text())
    names = tuple(scene["objects"])
    nominal = {}
    footprint = {}
    for name in names:
        pose = np.asarray(scene["objects"][name]["T_world_object"])
        nominal[name] = (pose[0, 3], pose[1, 3], np.arctan2(pose[1, 0], pose[0, 0]))
        footprint[name] = scene["objects"][name]["extents_xyz_m"][:2]
    config = scene["randomization"]
    sample = sample_tabletop_layout(
        nominal, config["ranges_xyyaw_rad"], footprint,
        table_bounds_xy=config["table_bounds_xy_m"], seed=23,
        min_clearance_m=config["min_clearance_m"],
        max_attempts=config["max_attempts"])

    # 红块与簸箕中心 X 相差超过 10 cm，刷子也不再保持名义朝向。
    assert abs(sample["poses_xyyaw_rad"]["cube_red"][0] - nominal["dustpan"][0]) > .10
    assert abs(sample["poses_xyyaw_rad"]["brush"][2]) > .20
