from pathlib import Path

import mujoco
import numpy as np

from gene_agent.tasks.policies.sweep_cube_task.execution.curobo import brush_collision_spheres_object


SCENE = Path(__file__).resolve().parents[4] / "data/realworld/brush_cube_dustpan_0/scene.xml"


def test_brush_attachment_spheres_cover_compiled_collision_vertices():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    spheres = brush_collision_spheres_object(model, data)
    assert spheres.ndim == 2 and spheres.shape[1] == 4
    assert len(spheres) >= 8
    brush = data.body("brush")
    world_from_brush = np.eye(4)
    world_from_brush[:3, :3] = brush.xmat.reshape(3, 3)
    world_from_brush[:3, 3] = brush.xpos
    brush_from_world = np.linalg.inv(world_from_brush)
    checked = 0
    for geom_id in range(model.ngeom):
        if (model.body(int(model.geom_bodyid[geom_id])).name != "brush"
                or (model.geom_contype[geom_id] == 0
                    and model.geom_conaffinity[geom_id] == 0)):
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start:start + count], dtype=float)
        world = (data.geom_xmat[geom_id].reshape(3, 3) @ vertices.T).T + data.geom_xpos[geom_id]
        local = (brush_from_world[:3, :3] @ world.T).T + brush_from_world[:3, 3]
        clearance = np.linalg.norm(local[:, None, :] - spheres[None, :, :3], axis=2) - spheres[None, :, 3]
        assert np.max(np.min(clearance, axis=1)) <= 1e-6
        checked += count
    assert checked > 0
