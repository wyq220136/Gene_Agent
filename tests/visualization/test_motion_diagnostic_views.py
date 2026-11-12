import numpy as np
from gene_agent.visualization.motion_diagnostic_viser import select_view


def bundle():
    a=dict(sphere_index=0,label='left/robot/arm/sphere#0',center_world_m=[0,0,0],effective_radius_m=.04,role='robot')
    b=dict(sphere_index=1,label='left/tool/held/sphere#1',center_world_m=[.06,0,0],effective_radius_m=.04,role='held')
    return dict(configuration_ref='ik_candidate',snapshot={'scene_model':{}},collisions={'spheres':[a,b],'pairs':[dict(kind='self',first=a,second=b,allowed=False)]})


def test_local_camera_covers_both_collision_spheres_and_keeps_identity():
    v=select_view(bundle(),view='local',pair_index=0)
    np.testing.assert_allclose(v['target'],[.03,0,0])
    assert v['distance']>.14
    assert 'left/tool/held/sphere#1' in v['labels']


def test_camera_override_and_missing_collision_are_explicit():
    v=select_view(bundle(),view='global',azimuth=80,elevation=20,distance=2)
    assert v['distance']==2 and v['azimuth']==80
    import pytest
    with pytest.raises(ValueError):select_view(bundle(),pair_index=99)
    b=bundle();b['collisions']['pairs']=[]
    assert select_view(b,view='local')['pair'] is None
    assert select_view(b,view='local')['warnings']


def test_camera_pan_does_not_change_collision_focus():
    v=select_view(bundle(),view='local',look_at=[1,2,3])
    assert v['target']==[1,2,3]
    np.testing.assert_allclose(v['focus'],[.03,0,0])
    assert v['pair']['first']['sphere_index']==0


def test_local_context_keeps_nearby_geometry_but_excludes_distant_geometry():
    from gene_agent.visualization.motion_diagnostic_viser import nearby_bounds
    assert nearby_bounds(np.array([[-1,-1,-.1],[1,1,0]]),[0,0,.05],.2)
    assert not nearby_bounds(np.array([[5,5,5],[6,6,6]]),[0,0,0],.2)


def test_plain_render_has_no_text_panel():
    from gene_agent.visualization.motion_diagnostic_viser import plain_render
    image=plain_render(np.full((80,100,4),255,dtype=np.uint8))
    assert image.size==(100,80)


def test_old_bundle_missing_visual_is_explicit():
    from gene_agent.visualization.motion_diagnostic_visual import load_visual_scene
    scene=load_visual_scene(bundle())
    assert not scene['meshes'] and scene['status']=='missing'
    assert scene['warnings']
