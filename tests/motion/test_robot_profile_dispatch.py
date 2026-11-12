"""未知机器人不能落入 Realman；旧调用的配置和会话行为保持。"""
import pytest


@pytest.mark.parametrize('profile', ['realman_dual_standard', 'realman_single_pika', None])
def test_known_legacy_dispatch_preserves_config(profile):
    from gene_agent.motion.curobo.profiles import load_motion_config
    raw = {'joint_groups': {'working': ['one']}}
    if profile: raw['profile'] = profile
    sentinel = {'tool_frame': 'legacy_tip', 'joint_names': ['one']}
    def legacy(group, metadata, *, gripper_snapshot):
        assert group == 'working' and metadata is raw
        assert gripper_snapshot == {'group': 'working'}
        return sentinel
    assert load_motion_config('working', raw, gripper_snapshot={'group': 'working'}, legacy_loader=legacy) is sentinel


def test_unknown_explicit_profile_fails_without_loading_legacy():
    from gene_agent.motion.curobo.profiles import load_motion_config
    def forbidden(*args, **kwargs):
        raise AssertionError('unknown profile used legacy')
    with pytest.raises(ValueError, match='profile'):
        load_motion_config('arm', {'profile': 'unknown_robot'}, legacy_loader=forbidden)
