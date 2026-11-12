"""安装例外必须局部、显式，优化期修改退出后完整恢复。"""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gene_agent.motion.collision import split_scene_for_mounting_contacts, optimizer_mounting_mask


def test_body_external_split_preserves_every_obstacle():
    scene={'cuboid':{'torso':{'dims':[1,2,3]},'table':{'dims':[3,2,1]}},'mesh':{'left_arm':{'file_path':'left.obj'}}}
    body,external=split_scene_for_mounting_contacts(scene,{'obstacle_names':['torso']})
    assert set(body['cuboid'])=={'torso'}
    assert set(external['cuboid'])=={'table'} and set(external['mesh'])=={'left_arm'}
    assert body['cuboid']['torso']==scene['cuboid']['torso']
    with pytest.raises(ValueError,match='未知障碍'):
        split_scene_for_mounting_contacts(scene,{'obstacle_names':['missing']})


def test_optimizer_mask_is_restored_even_on_error():
    class Params:
        def __init__(self):self.link_spheres=torch.tensor([[[0.,0.,0.,.1],[0.,0.,1.,.2]]])
        def disable_link_spheres(self,name):
            assert name=='base_link'
            self.link_spheres[:,0,3]=-100.
    params=Params();original=params.link_spheres.clone()
    kin=SimpleNamespace(config=SimpleNamespace(kinematics_config=params))
    with pytest.raises(RuntimeError,match='injected'):
        with optimizer_mounting_mask(kin,{'robot_links':['base_link']}):
            assert params.link_spheres[0,0,3]==-100.
            assert params.link_spheres[0,1,3]==original[0,1,3]
            raise RuntimeError('injected')
    torch.testing.assert_close(params.link_spheres,original)
