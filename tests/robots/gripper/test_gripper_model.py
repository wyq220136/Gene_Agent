"""双指代理URDF、真实内间隙、TCP不漂移及有限力控制契约。"""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
from gene_agent.robots import build_robot,set_gripper_width,measure_gripper_width


@pytest.fixture(scope='module')
def robot(tmp_path_factory):
    meta=build_robot(tmp_path_factory.mktemp('active_gripper'))
    model=mujoco.MjModel.from_xml_path(meta['xml_path'])
    return meta,model


def initialize(model,data,meta):
    for name,value in meta['home_joint_positions'].items():
        data.qpos[model.joint(name).qposadr[0]]=value
        data.ctrl[model.actuator(f'{name}_position').id]=value
    mujoco.mj_forward(model,data)


def inner_gap_from_geoms(model,data,meta,group):
    """独立用碰撞box几何投影计算内面，不复用控制/测量函数的site定义。"""
    tcp=data.site(meta['tcp_sites'][group]);rotation=tcp.xmat.reshape(3,3)
    inner=[]
    for label,sign in [('positive',1),('negative',-1)]:
        geom=data.geom(meta['grippers'][group]['fingers'][label]['pad_geom_name'])
        center=(geom.xpos-tcp.xpos)@rotation
        basis=rotation.T@geom.xmat.reshape(3,3)
        half=float(np.abs(basis[0])@model.geom_size[geom.id])
        inner.append(center[0]-sign*half)
    return inner[0]-inner[1],abs(sum(inner))


@pytest.mark.parametrize('width',[0.,.007,.041,.080])
def test_both_fingers_true_gap_symmetry_and_tcp(robot,width):
    meta,model=robot;data=mujoco.MjData(model);initialize(model,data,meta)
    poses={group:(data.site(site).xpos.copy(),data.site(site).xmat.copy()) for group,site in meta['tcp_sites'].items()}
    for group,gripper in meta['grippers'].items():
        for name in gripper['joint_names']:data.qpos[model.joint(name).qposadr[0]]=width/2
    mujoco.mj_forward(model,data)
    for group in meta['grippers']:
        gap,symmetry=inner_gap_from_geoms(model,data,meta,group)
        assert gap==pytest.approx(width,abs=1e-12) and symmetry<1e-12
        feedback=measure_gripper_width(model,data,meta,group)
        assert feedback['width_m']==pytest.approx(width,abs=1e-12)
        np.testing.assert_allclose(data.site(meta['tcp_sites'][group]).xpos,poses[group][0],atol=1e-12)
        np.testing.assert_allclose(data.site(meta['tcp_sites'][group]).xmat,poses[group][1],atol=1e-12)


def test_derived_urdf_is_independently_compilable_and_mimic_explicit(robot):
    meta,model=robot
    assert len(meta['joint_names'])==14 and model.nq==18 and model.nu==18 and model.neq==2
    for group,gripper in meta['grippers'].items():
        tree=ET.parse(gripper['generated_urdf_path']).getroot()
        sliders=[joint for joint in tree.findall('joint') if joint.get('type')=='prismatic']
        assert len(sliders)==2 and len(tree.findall('.//mimic'))==1
        assert {j.find('axis').get('xyz') for j in sliders}=={'1 0 0','-1 0 0'}
        compiled=mujoco.MjModel.from_xml_path(gripper['generated_urdf_path'])
        assert compiled.nq==9
        for name in gripper['joint_names']:assert name not in meta['joint_groups'][group]
        assert gripper['home_width_m']==.08


def test_default_arm_tcp_matches_legacy(robot,tmp_path):
    meta,model=robot;fixed=build_robot(tmp_path,gripper_mode='fixed_urdf')
    reference=mujoco.MjModel.from_xml_path(fixed['xml_path'])
    data=mujoco.MjData(model);legacy=mujoco.MjData(reference)
    initialize(model,data,meta);initialize(reference,legacy,fixed)
    for group in meta['joint_groups']:
        np.testing.assert_allclose(data.site(meta['tcp_sites'][group]).xpos,legacy.site(fixed['tcp_sites'][group]).xpos,atol=1e-12)
        np.testing.assert_allclose(data.site(meta['tcp_sites'][group]).xmat,legacy.site(fixed['tcp_sites'][group]).xmat,atol=1e-12)
        for i in range(1,8):
            np.testing.assert_allclose(data.body(f'{group}_link_{i}').xpos,legacy.body(f'{group}_link_{i}').xpos,atol=1e-12)


def test_control_only_and_measurement_preserves_integration_state(robot):
    meta,model=robot;data=mujoco.MjData(model);initialize(model,data,meta)
    q=data.qpos.copy();warm=data.qacc_warmstart.copy()
    set_gripper_width(model,data,meta,'right',.007)
    np.testing.assert_array_equal(data.qpos,q)
    for name in meta['grippers']['right']['actuator_names']:assert data.ctrl[model.actuator(name).id]==.0035
    measure_gripper_width(model,data,meta,'right')
    np.testing.assert_array_equal(data.qpos,q);np.testing.assert_array_equal(data.qacc_warmstart,warm)
    for value in [-.001,.081,np.nan,np.inf]:
        with pytest.raises(ValueError):set_gripper_width(model,data,meta,'right',value)
    with pytest.raises(ValueError):set_gripper_width(model,data,meta,'unknown',.01)


def test_dynamic_open_close_uses_finite_force(robot):
    meta,model=robot;data=mujoco.MjData(model);initialize(model,data,meta)
    # 完整双臂模型中所有arm也由有限力position actuator保持；不在step间重写qpos。
    for target in [0.,.007,.041,.08]:
        for group in meta['grippers']:set_gripper_width(model,data,meta,group,target)
        peak=0.
        for _ in range(600):
            mujoco.mj_step(model,data)
            for gripper in meta['grippers'].values():
                forces=[data.actuator_force[model.actuator(n).id] for n in gripper['actuator_names']]
                peak=max(peak,max(abs(x) for x in forces))
        assert peak<=10.00001
        for group in meta['grippers']:
            measured=measure_gripper_width(model,data,meta,group)
            assert measured['width_m']==pytest.approx(target,abs=.0005)
            assert measured['symmetry_error_m']<.00025


def test_visual_reuses_original_meshes_without_proxy_boxes(robot):
    """外观恢复不能新增物理关节，也不能把视觉连杆导入接触层。"""
    meta,model=robot
    for group,gripper in meta['grippers'].items():
        root=ET.parse(gripper['generated_urdf_path']).getroot()
        tcp=root.find("link[@name='gripper_tcp_frame']")
        assert tcp.find('visual') is None
        for label,rod in [('positive','link1'),('negative','link2')]:
            link=root.find(f"link[@name='gripper_{label}_finger']")
            visuals=link.findall('visual')
            assert len(visuals)==2
            meshes=[v.find('geometry/mesh') for v in visuals]
            assert all(m is not None for m in meshes)
            assert any(m.get('filename').endswith(f'jaw_{rod}.STL') for m in meshes)
            assert len(link.findall('collision'))==3
            body=model.body(gripper['fingers'][label]['body_name']).id
            visual_ids=[i for i in range(model.ngeom) if model.geom_bodyid[i]==body and model.geom_contype[i]==0]
            assert len(visual_ids)==2
            assert all(model.geom_conaffinity[i]==0 for i in visual_ids)
