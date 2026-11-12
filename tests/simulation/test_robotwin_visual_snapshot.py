from types import SimpleNamespace
import numpy as np
import trimesh
from gene_agent.simulation.robotwin.visual_snapshot import save_visual_snapshot


def test_native_box_visual_uses_half_size_and_actual_pose(tmp_path):
    import pytest
    sapien=pytest.importorskip("sapien")
    shape=sapien.render.RenderShapeBox([.1,.2,.3],sapien.render.RenderMaterial(base_color=[.8,.2,.1,1]))
    entity=sapien.Entity();entity.name='box';entity.pose=sapien.Pose([1,2,3])
    component=sapien.render.RenderBodyComponent();component.attach(shape);entity.add_component(component)
    env=SimpleNamespace(scene=SimpleNamespace(entities=[entity]))
    cache={}
    first=save_visual_snapshot(env,{},'left',tmp_path,cache,state_id='s1')
    row=first['objects'][0]
    mesh=trimesh.load(row['file_path'],force='mesh')
    np.testing.assert_allclose(mesh.extents,[.2,.4,.6],rtol=1e-6)
    np.testing.assert_allclose(np.asarray(row['world_from_body'])[:3,3],[1,2,3])
    entity.pose=sapien.Pose([2,3,4])
    second=save_visual_snapshot(env,{},'left',tmp_path,cache,state_id='s2')
    assert row['sha256']==second['objects'][0]['sha256']
    np.testing.assert_allclose(np.asarray(second['objects'][0]['world_from_body'])[:3,3],[2,3,4])
