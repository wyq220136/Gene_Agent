"""ALOHA 的 cuRobo 参数适配，保留源球与排除规则并锁定实际夹爪。"""
from copy import deepcopy
from gene_agent.robots.aloha import digest
from gene_agent.robots.model import RobotDescription


def build_aloha_motion_config(group, robot_metadata, *, gripper_snapshot):
    """输出 SynchronousPlanner 参数；pool 对兼容指关节变化精确更新，不兼容结构重建。"""
    robot = RobotDescription.from_metadata(robot_metadata)
    if group not in robot.groups:
        raise ValueError(f'未知机器人组: {group}')
    if not gripper_snapshot or gripper_snapshot.get('group') != group:
        raise ValueError('ALOHA 规划需要当前实际夹爪快照')
    payload = {k:v for k,v in gripper_snapshot.items() if k != 'snapshot_id'}
    if gripper_snapshot.get('snapshot_id') != digest(payload):
        raise ValueError('ALOHA 夹爪快照内容哈希不一致')
    if gripper_snapshot.get('source_hashes') != robot_metadata['source_hashes']:
        raise ValueError('ALOHA 夹爪快照来源与模型不一致')
    cfg = deepcopy(robot_metadata['planning_assets'][group]['robot_config'])
    kin = cfg['kinematics']
    # 原bench使用旧cuRobo配置；只转换已核对的字段，不静默丢弃未知能力。
    if kin.pop('use_usd_kinematics', False):
        raise ValueError('ALOHA 原生适配暂不支持 USD kinematics')
    source_tool = kin.pop('ee_link', robot.groups[group].tool_link)
    if source_tool != robot.groups[group].tool_link:
        raise ValueError('ALOHA ee_link 与 native move_group 不一致')
    kin['tool_frames'] = list(dict.fromkeys([*kin.get('tool_frames', []), source_tool]))
    kin['extra_links'] = kin.get('extra_links') or {}  # 原native模板用null表示无额外link。
    kin['lock_joints'] = dict(gripper_snapshot['joint_positions_m'])
    # cspace 保留指关节和 retract 值；cuRobo 根据 lock_joints 消去指关节。
    cspace = kin['cspace']
    if 'retract_config' in cspace:
        cspace['default_joint_position'] = cspace.pop('retract_config')
    for i,name in enumerate(cspace['joint_names']):
        if name in kin['lock_joints']:
            cspace['default_joint_position'][i] = kin['lock_joints'][name]
    transforms = robot_metadata['transforms'][group]
    return dict(robot_config=cfg,joint_names=list(robot.groups[group].joint_names),
                planner_joint_names=list(robot.groups[group].planner_joint_names),
                gripper_snapshot=deepcopy(gripper_snapshot),
                world_from_base=transforms['world_from_urdf_base'],
                link_from_tcp=transforms['link_from_tcp'],tool_frame=robot.groups[group].tool_link)


def gripper_reuse_identity(settings):
    """仅规范已验证ALOHA指关节的动态值，其他锁关节/源球/变换仍属结构。"""
    g=settings['gripper_snapshot']
    if digest({k:v for k,v in g.items() if k!='snapshot_id'})!=g['snapshot_id']:
        raise ValueError('ALOHA动态快照哈希不一致')
    result=deepcopy(settings);kin=result['robot_config']['kinematics']
    if digest(kin['collision_spheres'])!=g['merged_sphere_sha256']:
        raise ValueError('ALOHA源球与快照不一致')
    names=g['joint_names']
    if not names or set(names)!=set(g['joint_positions_m']):
        raise ValueError('ALOHA动态指关节集合不一致')
    for name in names:
        value=g['joint_positions_m'][name]
        if kin['lock_joints'][name]!=value:
            raise ValueError('ALOHA锁关节与实际快照不一致')
        index=kin['cspace']['joint_names'].index(name)
        if kin['cspace']['default_joint_position'][index]!=value:
            raise ValueError('ALOHA锁关节默认值不一致')
        kin['lock_joints'][name]=0.
        kin['cspace']['default_joint_position'][index]=0.
    result['gripper_snapshot']={k:g[k] for k in ('schema_version','group','mode','source_hashes','source_robot_config_sha256','merged_sphere_sha256','joint_names')}
    result['gripper_snapshot']['dynamic_lock_joints']=list(names)
    return result
