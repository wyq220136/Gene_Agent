"""机器人资产、具名关节组与显式代理夹爪控制。"""
__all__ = ["build_robot", "set_gripper_width", "measure_gripper_width"]


def __getattr__(name):
    # URDF/纯视觉模块不应因包初始化而加载物理引擎。
    if name == "build_robot":
        from .realman import build_robot
        return build_robot
    if name in {"set_gripper_width", "measure_gripper_width"}:
        from . import gripper
        return getattr(gripper, name)
    raise AttributeError(name)
