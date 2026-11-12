"""XML显示风格与共享相机配置；不覆盖源材质或修改业务物理参数。"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

CONFIG = Path(__file__).resolve().parents[2]/'config/visualization/studio.json'


def _load_config():
    # cuRobo/Warp initialization can leave a worker process with an ASCII
    # preferred encoding. The repository config is always UTF-8 and contains
    # Chinese provenance text, so never depend on the process locale here.
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def _text(values):
    return ' '.join(str(value) for value in values)


def _child(parent, tag):
    node=parent.find(tag)
    return ET.SubElement(parent,tag) if node is None else node


def _gripper_side_direction(local_z):
    """沿夹爪本地 +Z 跟随，保持相机在世界水平面上方且方位连续。"""
    side=np.asarray(local_z,dtype=float).copy()
    # 不能在 local_z[2] 变号时反转整根轴：那会让方位角跳 180°。
    side[2]=max(side[2],0.)+.18
    return side/np.linalg.norm(side)


def _gripper_side_camera(model, data, target, metadata, group):
    """根据活动组的双指内侧 site 和 TCP 构造随夹爪运动的侧视相机。"""
    descriptor = metadata.get('grippers', {}).get(group)
    tcp_name = metadata.get('tcp_sites', {}).get(group)
    if descriptor is None or tcp_name is None:
        raise ValueError(f'活动夹爪组 {group!r} 缺少 gripper 描述器或 TCP site')
    fingers = descriptor.get('fingers', {})
    names = (
        descriptor.get('positive_inner_site_name') or fingers.get('positive', {}).get('inner_site_name'),
        descriptor.get('negative_inner_site_name') or fingers.get('negative', {}).get('inner_site_name'),
    )
    if any(name is None for name in names):
        raise ValueError(f'活动夹爪组 {group!r} 缺少双指 inner_site_name')
    for name in (tcp_name, *names):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) < 0:
            raise ValueError(f'活动夹爪组 {group!r} 的 site 不存在: {name}')
    positive, negative = (data.site(name).xpos for name in names)
    tcp = data.site(tcp_name)
    opening = positive - negative
    opening_norm = np.linalg.norm(opening)
    if opening_norm < 1e-6:
        # 闭爪时两个内侧 site 可重合，改用 TCP +Y 作为开合轴。
        opening = tcp.xmat.reshape(3, 3)[:, 1]
    else:
        opening = opening / opening_norm
    rotation = tcp.xmat.reshape(3, 3)
    # 水平观察方向与双指开合轴垂直；TCP +Z 在向下抓碗时接近竖直，
    # 直接沿该轴看会变成俯视并被机械臂遮住，无法判断碗沿是否进入双指。
    side = np.cross(opening, [0., 0., 1.])
    if np.linalg.norm(side) < 1e-6:
        side = rotation[:, 2].copy()
        side[2] = 0.
    if np.linalg.norm(side) < 1e-6:
        side = rotation[:, 0].copy()
        side[2] = 0.
    side /= np.linalg.norm(side)
    root_name = metadata.get('group_root_bodies', {}).get(group)
    root_id = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
               if root_name else -1)
    if root_id >= 0 and np.dot(side, tcp.xpos - data.xpos[root_id]) < 0:
        # 相机放在远离机械臂根部的一侧，减少手臂遮挡夹持区域。
        side = -side
    side[2] = .22
    side /= np.linalg.norm(side)
    midpoint = .5 * (positive + negative)
    lookat = .55 * midpoint + .45 * target
    return {
        'distance': .34,
        'azimuth': float(np.degrees(np.arctan2(-side[1], -side[0]))),
        'elevation': float(-np.degrees(np.arcsin(np.clip(side[2], -1., 1.)))),
        'lookat': lookat.tolist(),
    }


def apply_scene_style(root):
    """在compose_scene完成include后调用，可重复应用。

    只修改visual/asset skybox/worldbody light；不添加有质量几何、地面或碰撞，
    不给业务geom强套灰色material。影子由较弱主光产生，补光降低纯黑阴影对比。
    完整MJB指纹会随视觉参数变化，不能绕过原有restore模型身份检查。
    """
    config=_load_config()
    visual=_child(root,'visual')
    _child(visual,'headlight').attrib.update({key:_text(value) for key,value in config['headlight'].items()})
    _child(visual,'quality').attrib.update(offsamples='4',shadowsize='4096')
    _child(visual,'rgba').attrib.update(haze='.9 .93 .96 1',fog='.9 .93 .96 1')
    _child(visual,'map').attrib.update(znear='.001',shadowclip='1',shadowscale='.8')
    _child(visual,'scale').attrib.update(framelength='.12',framewidth='.006')
    asset=_child(root,'asset')
    sky=asset.find("texture[@name='gene_visual_sky']")
    if sky is None:
        sky=ET.SubElement(asset,'texture',name='gene_visual_sky')
    sky.attrib.update(type='skybox',builtin='gradient',rgb1=_text(config['sky_top_rgb']),
                      rgb2=_text(config['sky_bottom_rgb']),width='512',height='3072')
    world=_child(root,'worldbody')
    for light in list(world.findall('light')):
        if light.get('name','') == 'scene_light' or light.get('name','').startswith('gene_visual_'):
            world.remove(light)
    for light in config['lights']:
        attrs={key:('true' if value else 'false') if isinstance(value,bool) else _text(value) if isinstance(value,list) else value for key,value in light.items()}
        ET.SubElement(world,'light',directional='true',specular='.1 .1 .1',**attrs)
    return root


def camera_presets(model,data,target_body='bowl', *, robot_metadata=None, active_group=None):
    """返回world lookat、米距离和度角；数据应先forward，调用方持有状态访问权。"""
    body_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,target_body)
    if body_id<0:
        raise ValueError(f'Unknown target body: {target_body}')
    target=data.xpos[body_id]
    presets={}
    for name,config in _load_config()['cameras'].items():
        anchor=target
        if name=='task':
            microwave_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'microwave')
            if microwave_id>=0:
                anchor=(target+data.xpos[microwave_id])/2
        if name=='top':
            table_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'scene_table')
            if table_id>=0:
                anchor=data.geom_xpos[table_id]
        if name in {'knob','knob_oblique','door_oblique'}:
            site_name = 'microwave_handle_grasp' if name == 'door_oblique' else 'knob_0_site'
            site_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,site_name)
            if site_id<0:
                continue
            anchor=data.site_xpos[site_id]
        presets[name]={key:config[key] for key in ('distance','azimuth','elevation')}
        presets[name]['lookat']=(anchor+config['target_offset_m']).tolist()
        if name in {'knob','knob_oblique','door_oblique'}:
            # 旋钮 site +Z 指向外侧，把手 site 局部 +Y 指向门内侧。
            # 两者从当前关节姿态求相机朝向，开门后镜头仍跟随把手。
            site_rotation=data.site_xmat[site_id].reshape(3,3)
            forward=(site_rotation[:,1] if name == 'door_oblique'
                     else -site_rotation[:,2])
            presets[name]['azimuth']=float(np.degrees(np.arctan2(forward[1],forward[0]))+config['azimuth'])
            presets[name]['elevation']=float(np.degrees(np.arcsin(np.clip(forward[2],-1,1)))+config['elevation'])
    if robot_metadata is not None and active_group is not None:
        presets['gripper_side'] = _gripper_side_camera(
            model, data, target, robot_metadata, active_group,
        )
        return presets
    # 兼容原有 Pika 查看入口：没有指定活动组时仍沿本地 ±Z 观察双指。
    # 目标放在碗与两个指内面之间，避免只追随被碰撞推走的碗。
    finger_sites=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,name)
                  for name in ('pika_left_inner','pika_right_inner')]
    base_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pika_gripper_base_link')
    if base_id>=0 and all(index>=0 for index in finger_sites):
        midpoint=.5*(data.site_xpos[finger_sites[0]]+data.site_xpos[finger_sites[1]])
        side=_gripper_side_direction(data.body(base_id).xmat.reshape(3,3)[:,2])
        lookat=.55*midpoint+.45*target
        presets['gripper_side']={
            'distance':.34,
            'azimuth':float(np.degrees(np.arctan2(-side[1],-side[0]))),
            'elevation':float(-np.degrees(np.arcsin(np.clip(side[2],-1.,1.)))),
            'lookat':lookat.tolist(),
        }
    return presets


def configure_camera(camera,params):
    """离屏与passive viewer调用同一设置函数，避免两个入口偷偷使用不同视角。"""
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=params['lookat']
    for key in ('distance','azimuth','elevation'):
        setattr(camera,key,params[key])
    return camera
