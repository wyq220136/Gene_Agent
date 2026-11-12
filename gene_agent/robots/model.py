"""不依赖引擎的旋转臂组描述；保留源 metadata，拒绝隐式关节映射。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from gene_agent.knowledge.geometry.transforms import validate_pose


@dataclass(frozen=True)
class JointGroup:
    """臂组的公共关节顺序与规划器关节顺序显式一对一对应。"""
    joint_names: tuple[str, ...]
    planner_joint_names: tuple[str, ...]
    tool_link: str


def _names(value, label):
    """避免字符串被当作名字序列，拒绝空名和重复名。"""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f'{label} 必须是非空名称列表')
    if any(not isinstance(n, str) or not n.strip() for n in value) or len(set(value)) != len(value):
        raise ValueError(f'{label} 名称无效或重复')
    return tuple(value)


@dataclass(frozen=True)
class RobotDescription:
    """仅验证/冻结当前需要的边界，不替代 URDF 或原生控制器配置。"""
    profile_id: str
    embodiment_id: str
    world_frame: str
    groups: Mapping[str, JointGroup]
    _metadata: dict

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> 'RobotDescription':
        """验证新模型，并兼容明确声明的现有 Realman/Pika metadata。"""
        raw = deepcopy(dict(metadata))
        profile = raw.get('profile')
        world = raw.get('world_frame')
        if not isinstance(profile, str) or not profile.strip():
            raise ValueError('机器人必须显式声明 profile')
        if not isinstance(world, str) or not world.strip():
            raise ValueError('机器人必须显式声明 world_frame')
        groups = raw.get('joint_groups')
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError('机器人缺少 joint_groups')
        legacy = profile in {'realman_dual_standard', 'realman_single_pika'}
        result = {}
        all_names = set()
        for group, values in groups.items():
            if not isinstance(group, str) or not group.strip():
                raise ValueError('臂组名称必须非空')
            names = _names(values, f'{group}.joint_names')
            if all_names.intersection(names):
                raise ValueError('不同臂组不能共享活动关节')
            all_names.update(names)
            planner = raw.get('planner_joint_names', {}).get(group)
            if planner is None and not legacy:
                raise ValueError(f'{group} 缺少 planner_joint_names')
            planner = names if planner is None else _names(planner, f'{group}.planner_joint_names')
            if len(planner) != len(names):
                raise ValueError(f'{group} 公共/规划关节数量不一致')
            types = raw.get('joint_types', {}).get(group)
            limits = raw.get('joint_limits', {}).get(group)
            if not legacy and (types is None or limits is None):
                raise ValueError(f'{group} 缺少关节类型/限位')
            if types is not None and (len(types) != len(names) or any(t not in {'revolute', 'continuous'} for t in types)):
                raise ValueError('当前弧度轨迹仅支持 revolute/continuous 臂组')
            if limits is not None:
                array = np.asarray(limits, dtype=float)
                if array.shape != (len(names), 2) or not np.isfinite(array).all() or np.any(array[:, 0] >= array[:, 1]):
                    raise ValueError(f'{group} 关节限位无效')
            transforms = raw.get('transforms', {}).get(group, {})
            for key in ('world_from_urdf_base', 'link_from_tcp'):
                validate_pose(transforms.get(key), field_name=f'{group}.{key}')
            tool = raw.get('tool_links', {}).get(group, 'link_7' if legacy else None)
            if not isinstance(tool, str) or not tool.strip():
                raise ValueError(f'{group} 缺少 tool_link')
            gripper = raw.get('grippers', {}).get(group, {})
            if 'width_range_m' in gripper:
                width = np.asarray(gripper['width_range_m'], dtype=float)
                if width.shape != (2,) or not np.isfinite(width).all() or not 0 <= width[0] < width[1]:
                    raise ValueError(f'{group} 夹爪开口范围无效')
            result[group] = JointGroup(names, planner, tool)
        return cls(profile, str(raw.get('embodiment_id', profile)), world,
                   MappingProxyType(result), raw)

    def to_metadata(self) -> dict:
        """返回独立副本，调用方修改不能改变已经验证的模型。"""
        return deepcopy(self._metadata)
