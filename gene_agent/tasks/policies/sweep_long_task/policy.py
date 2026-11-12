"""长刷沿用清扫策略，仅覆盖既有抓取与排序参数。"""
def operation_parameters(*, aligned_lane=True, grasp_yaw_deg=0., goal_inset_extra_m=.002):
    return dict(grasp_mode="world_minus_z", grasp_yaw_deg=grasp_yaw_deg,
                target_order_policy="farthest_first" if aligned_lane else "closest_first",
                alignment_lateral_trigger_m=.045,
                goal_inset_extra_m=goal_inset_extra_m if aligned_lane else 0.)


def run(operations, *, alignment_lateral_trigger_m=.045,
        target_order_policy="farthest_first") -> bool:
    """长刷复用同一清扫决策；安装/抓取与通道参数由外层适配提供。

    返回操作完成 bool；完整实际状态/评分报告通过 simulation.run_long_brush。
    与普通刷共享失败停止及有限恢复，不维护第二份清扫循环。
    """
    from gene_agent.tasks.policies.sweep_cube_task.policy import run as sweep
    return sweep(operations, alignment_lateral_trigger_m=alignment_lateral_trigger_m,
                 target_order_policy=target_order_policy)
