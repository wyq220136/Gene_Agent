"""清扫接触角色由调用方提供；不绑定机械臂或场景实例名。"""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class SweepContacts:
    tool: str
    support: str
    receiver: str
    gripper_links: tuple[str, ...]

    def __post_init__(self):
        names = (self.tool, self.support, self.receiver, *self.gripper_links)
        if not self.gripper_links or any(not isinstance(n,str) or not n for n in names):
            raise ValueError('清扫工具、支撑、接收器与夹爪 link 必须明确')
        object.__setattr__(self, 'gripper_links', tuple(self.gripper_links))

    @property
    def grip_pairs(self):
        return tuple((link,self.tool) for link in self.gripper_links)

    @property
    def table_pairs(self):
        return (*self.grip_pairs, (self.tool,self.support))

    @classmethod
    def load(cls, path):
        """读取适配器显式选择的配置，不在 Skill 内猜测场景或机器人。"""
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))
