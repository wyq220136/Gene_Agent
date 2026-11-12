"""显式选择 embodiment 的规划配置；旧加载器只服务旧调用。"""


def load_motion_config(group, robot_metadata, *, gripper_snapshot=None, legacy_loader=None):
    """返回现有 planner 参数；不支持的显式 profile 必须报错。

    无 profile 仅保留历史 metadata 调用兼容，新 RobotDescription 不接受它。
    legacy_loader 由原会话传入，保留既有调用方的配置/测试注入边界。
    """
    profile = robot_metadata.get('profile')
    if profile == 'aloha_agilex':
        from .aloha import build_aloha_motion_config
        return build_aloha_motion_config(group, robot_metadata, gripper_snapshot=gripper_snapshot)
    if profile not in {None, 'realman_dual_standard', 'realman_single_pika'}:
        raise ValueError(f'未知机器人 profile: {profile}')
    if legacy_loader is None:
        from ..core import load_realman_motion_config
        legacy_loader = load_realman_motion_config
    return legacy_loader(group, robot_metadata, gripper_snapshot=gripper_snapshot)
