"""炉内取碗到桌面放碗在线子任务界面。"""

from gene_agent.visualization.microwave.extract_session import EXTRACT_SUBTASKS, ExtractTaskVisualSession
from .extract_place_app import ExtractPlaceVisualApp


class ExtractTaskVisualApp(ExtractPlaceVisualApp):
    """共用可编辑 Viser 场景，连续显示 Extract→Place 全部真实阶段。"""

    selections = EXTRACT_SUBTASKS
    selection_label = "子任务"
    title = "Bowl 子任务 · 炉内取出 → 桌面放置"

    def __init__(self, server, *, session=None,
                 initial_selection="microwave_object_extract", config_path=None):
        super().__init__(server, session=session or ExtractTaskVisualSession(
            config_path=config_path), initial_selection=initial_selection)


__all__ = ["ExtractTaskVisualApp"]
