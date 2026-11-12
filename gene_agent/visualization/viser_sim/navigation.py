"""在同一个 Viser 页面中切换在线 Skill 或 Task 检查器。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import Lock
from typing import Any
import traceback


class VisualViewerRouter:
    """同类选择沿用当前场景；跨类选择清理旧 GUI 和场景后重建。"""

    def __init__(
        self, server: Any, *, kind: str, labels: Mapping[str, str],
        families: Mapping[str, str], build: Callable[[str, Any], Any],
        initial_selection: str, initial_config: Any = None,
        on_app_started: Callable[[Any], None] | None = None,
    ) -> None:
        if initial_selection not in labels or set(labels) != set(families):
            raise ValueError("全局 Viewer 选项与类别映射不一致")
        self.server = server
        self.kind = kind
        self.labels = dict(labels)
        self.families = dict(families)
        self.build = build
        self.on_app_started = on_app_started
        self.selection_id = initial_selection
        self.app: Any = None
        self._pending: str | None = None
        self._pending_lock = Lock()
        self._syncing = False
        self._display = {key: f"{label} · {key}" for key, label in self.labels.items()}
        self._by_display = {value: key for key, value in self._display.items()}
        self._mount(initial_selection, initial_config)

    def _mount(self, selection: str, config: Any) -> None:
        self.select = self.server.gui.add_dropdown(
            f"全局选择 {self.kind}", options=tuple(self._by_display),
            initial_value=self._display[selection],
        )
        self.select.on_update(self._queue_selection)
        self.switch_info = self.server.gui.add_markdown(
            "同类切换保留当前场景；跨类别切换加载该类别的默认配置。"
            "需要留存摆位时，请先在当前界面保存配置。"
        )
        self.app = self.build(selection, config)
        # 原有的局部下拉框仍负责同类切换逻辑，但用户只看见全局入口。
        self.app.selection.visible = False
        if self.on_app_started is not None:
            self.on_app_started(self.app)

    def _queue_selection(self, _event: Any) -> None:
        if self._syncing:
            return
        with self._pending_lock:
            self._pending = self._by_display[str(self.select.value)]

    def _planning(self) -> bool:
        if self.app is None:
            return False
        return bool(getattr(self.app, "planning", False) or
                    getattr(getattr(self.app, "controller", None), "planning", False))

    def tick(self) -> None:
        """只在主循环切换，避免 GUI 回调中删除仍在使用的 Viser handle。"""
        with self._pending_lock:
            selected, self._pending = self._pending, None
        if selected is not None and selected != self.selection_id:
            if self._planning():
                self._syncing = True
                try:
                    self.select.value = self._display[self.selection_id]
                finally:
                    self._syncing = False
                self.switch_info.content = "当前正在规划；请在本轮结束后切换。"
            elif (self.app is not None and
                  (getattr(self.app, "_pending_edits", False) or
                   getattr(self.app, "_pose_applied", True) is False)):
                # 草稿尚未进入 GT/Knowledge/collision；切换会丢弃或隐式提交它。
                self._syncing = True
                try:
                    self.select.value = self._display[self.selection_id]
                finally:
                    self._syncing = False
                self.switch_info.content = "位姿草稿尚未应用；请先点击“应用新位姿”，再切换。"
            elif (self.app is not None and
                  self.families[selected] == self.families[self.selection_id]):
                # 设置原局部控件的值会触发该类现有的重建/失效回调。
                self.app.selection.value = selected
                self.selection_id = selected
            else:
                self.app = None
                self.server.gui.reset()
                self.server.scene.reset()
                self.selection_id = selected
                try:
                    self._mount(selected, None)
                except Exception as error:
                    traceback.print_exc()
                    self.server.gui.reset()
                    self.server.scene.reset()
                    self.select = self.server.gui.add_dropdown(
                        f"全局选择 {self.kind}", options=tuple(self._by_display),
                        initial_value=self._display[selected],
                    )
                    self.select.on_update(self._queue_selection)
                    self.switch_info = self.server.gui.add_markdown(
                        f"切换失败：{type(error).__name__}: {error}。请选择其他 {self.kind} 重试。"
                    )
        if self.app is not None:
            self.app.tick()
        self.select.disabled = self._planning()
