"""物体 Sub-task 在线入口；与单 Skill 共用编辑、场景和连续时间轴。"""
from gene_agent.visualization.microwave.bowl_session import OBJECT_TASKS
from .object_app import ObjectVisualApp


class ObjectTaskApp(ObjectVisualApp):
    """保留 Task 的全部 Skill 阶段，成功与失败前缀均使用真实规划结果。"""

    selections = OBJECT_TASKS
    selection_label = '完整 Task'
    title = 'Bowl Task · Pick → Insert'
    sample_bowl_candidates = True
    progressive_preview = True

    def __init__(self, server, *, initial_selection='microwave_object_insert', **kwargs):
        super().__init__(server, initial_selection=initial_selection, **kwargs)
