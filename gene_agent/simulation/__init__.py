"""仿真公共包；仅使用具体 MuJoCo 功能时才加载其引擎。"""
from importlib import import_module

__all__ = ['MujocoRuntime', 'compose_scene', 'configure_robot_gravity_compensation',
           'set_joint_positions', 'build_scene']


def __getattr__(name):
    """保持既有公开入口，原生 benchmark 导入子模块不会加载 MuJoCo。"""
    if name == 'MujocoRuntime':
        return getattr(import_module('.runtime', __name__), name)
    if name in __all__:
        return getattr(import_module('.mujoco_scene', __name__), name)
    raise AttributeError(name)
