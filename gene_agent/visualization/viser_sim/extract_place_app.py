"""Bowl Extract / Place 在线单 Skill Viser 界面。"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import viser

from gene_agent.tasks.policies.microwave_task.planning.door_session import configure_curobo_run_log
from gene_agent.visualization.microwave.extract_session import EXTRACT_PLACE_SKILLS, ExtractPlaceVisualSession
from .object_app import ObjectVisualApp, _serialized


class ExtractPlaceVisualApp(ObjectVisualApp):
    """共用 ObjectVisualApp 的场景编辑、碰撞分层与真实轨迹播放。"""

    selections = EXTRACT_PLACE_SKILLS
    title = "Bowl 炉内取出与桌面放置"

    def __init__(self, server, *, session=None, initial_selection="extract_bowl",
                 config_path=None):
        super().__init__(server, session=session or ExtractPlaceVisualSession(
            config_path=config_path), initial_selection=initial_selection)
        # 旧取出使用一组固定 profile keypose，候选滑块会暗示存在并不属于
        # 本 Skill 的抓取候选筛选；目标预览直接显示实际固定目标。
        self.candidate.visible = False
        self.preview_button.visible = False

    @_serialized
    def _show_knowledge(self):
        if self.planning:
            return
        self.playing = False
        self.scene.renderer._build_collision({})
        self.scene._shown_snapshot_ref = None
        self._shown_stage = None
        if self.scene._path_handle is not None:
            self.scene._path_handle.remove()
            self.scene._path_handle = None
        self.scene._target_handle.visible = False
        target = (self.session.extract_targets.pre_extract_1.matrix
                  if self.selection.value in {"extract_bowl", "microwave_object_extract"}
                  else self.session.table_target.matrix)
        self.ghost.set_visible(True)
        self.ghost.select(target, opening_width_m=self.session.open_width_m)
        self.phase_info.content = (
            f"**任务配置中的固定 TCP 目标** · 场景版本 {self.session.case.revision}；"
            "摆位编辑已更新目标与碰撞；这个预览尚未检查路径可行性。"
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="在线重新规划炉内取碗或桌面放碗")
    parser.add_argument("--skill", choices=tuple(EXTRACT_PLACE_SKILLS),
                        default="extract_bowl")
    parser.add_argument("--config", type=Path, help="Bowl 场景配置 JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args(argv)
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        app = ExtractPlaceVisualApp(server, initial_selection=args.skill,
                                    config_path=args.config)
        configure_curobo_run_log(app.session.output_dir)
        print(f"[skill] {args.skill}：http://{args.host}:{server.get_port()}", flush=True)
        print(f"[skill] 本次记录：{app.session.output_dir}", flush=True)
        while True:
            app.tick()
            time.sleep(.03)
    except KeyboardInterrupt:
        print("[skill] 已退出", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
