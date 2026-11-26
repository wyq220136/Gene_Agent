"""同步执行链中保留的夹爪实测状态；新命令发出前必须撤销旧值。"""


def confirmed_width(gripper):
    return getattr(gripper, "confirmed_endpoint_width_m", None)


def retained_width(gripper):
    width = confirmed_width(gripper)
    return width if width is not None else getattr(gripper, "initial_observed_width_m", None)


def retained_width_source(gripper, width):
    if (confirmed_width(gripper) is None
            and getattr(gripper, "initial_observed_width_m", None) == width):
        return "retained_initial_observation"
    return "confirmed_execution_endpoint"


def invalidate_endpoint(gripper):
    gripper.confirmed_endpoint_width_m = None
    gripper.initial_observed_width_m = None


def endpoint_overrides(grippers, overrides=None):
    widths = {group: retained_width(gripper) for group, gripper in grippers.items()
              if retained_width(gripper) is not None}
    widths.update(overrides or {})
    return widths
