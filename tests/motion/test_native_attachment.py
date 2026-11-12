import copy
import numpy as np

def test_separate_attachment_link_keeps_arm_and_gripper_spheres():
    from gene_agent.motion.curobo import attachment
    assert hasattr(attachment,'with_held_object')
    config={'kinematics':{'collision_spheres':{'wrist':[{'center':[0,0,0],'radius':.02}]},
        'collision_link_names':['wrist'],'self_collision_ignore':{'wrist':['finger']},'extra_links':{}}}
    old=copy.deepcopy(config)
    new=attachment.with_held_object(config,tool_link='wrist',finger_links=['finger'],spheres_link=[[.1,0,0,.01]])
    assert config==old
    k=new['kinematics']
    assert k['collision_spheres']['wrist']==old['kinematics']['collision_spheres']['wrist']
    assert k['extra_links']['gene_held_tool']['parent_link_name']=='wrist'
    assert set(k['self_collision_ignore']['gene_held_tool'])=={'finger'}
    assert k['collision_spheres']['gene_held_tool'][0]['center']==[.1,0.,0.]
