"""Skill 在线入口分发；门与 Bowl 使用各自会话，共用 Viser 显示组件。"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import viser

from gene_agent.tasks.policies.microwave_task.planning.door_session import configure_curobo_run_log
from gene_agent.visualization.microwave.bowl_session import OBJECT_SKILLS
from gene_agent.visualization.viser_sim.app import DOOR_SKILLS, DoorVisualApp
from gene_agent.visualization.viser_sim.object_app import ObjectVisualApp
from gene_agent.visualization.viser_sim.knob_app import KnobVisualApp
from gene_agent.visualization.microwave.knob_session import KNOB_SKILLS
from gene_agent.visualization.microwave.extract_session import EXTRACT_PLACE_SKILLS
from gene_agent.visualization.viser_sim.extract_place_app import ExtractPlaceVisualApp
from gene_agent.visualization.viser_sim.navigation import VisualViewerRouter
from gene_agent.visualization.viser_sim.sweep_replay_app import SweepReplayApp, SWEEP_SKILLS
from gene_agent.visualization.viser_sim.spoon_replay_app import SpoonReplayApp
from gene_agent.feedback.tabletop.spoon_run import SPOON_SKILLS
from gene_agent.visualization.viser_sim.cup_replay_app import CupReplayApp
from gene_agent.feedback.tabletop.cup_run import CUP_SKILLS


SKILL_LABELS = {**DOOR_SKILLS, **OBJECT_SKILLS, **EXTRACT_PLACE_SKILLS, **KNOB_SKILLS,
                **SWEEP_SKILLS, **SPOON_SKILLS, **CUP_SKILLS}
SKILL_FAMILIES = {
    **dict.fromkeys(DOOR_SKILLS, "door"),
    **dict.fromkeys(OBJECT_SKILLS, "object"),
    **dict.fromkeys(EXTRACT_PLACE_SKILLS, "extract_place"),
    **dict.fromkeys(KNOB_SKILLS, "knob"),
    **dict.fromkeys(SWEEP_SKILLS, "sweep"),
    **dict.fromkeys(SPOON_SKILLS, "spoon"),
    **dict.fromkeys(CUP_SKILLS, "cup"),
}


def _build_app(server, skill, config, run_dir=None):
    if skill in CUP_SKILLS:
        return CupReplayApp(server, initial_selection=skill, run_dir=run_dir)
    if skill in SPOON_SKILLS:
        return SpoonReplayApp(server, initial_selection=skill, run_dir=run_dir)
    if skill in SWEEP_SKILLS:
        return SweepReplayApp(server, initial_selection=skill, run_dir=run_dir)
    if skill in KNOB_SKILLS:
        return KnobVisualApp(server, initial_selection=skill, config_path=config)
    if skill in EXTRACT_PLACE_SKILLS:
        return ExtractPlaceVisualApp(server, initial_selection=skill, config_path=config)
    if skill in OBJECT_SKILLS:
        return ObjectVisualApp(server, initial_selection=skill, config_path=config)
    return DoorVisualApp(server, mode='skill', initial_selection=skill, config_path=config)


def main() -> None:
    parser = argparse.ArgumentParser(description='在线检查 Skill；清扫项回放实际 MuJoCo 状态')
    parser.add_argument('--skill', choices=tuple(SKILL_LABELS), default=next(iter(DOOR_SKILLS)))
    parser.add_argument('--config', type=Path, help='门/Bowl/旋钮的 scene_config.json')
    parser.add_argument('--run', type=Path, help='保存记录目录；微波炉记录只读回放，不启动在线规划')
    parser.add_argument('--mode', choices=('actual', 'planned'), help='保存记录的数据类型')
    parser.add_argument('--stage', help='保存记录中的阶段 ID')
    parser.add_argument('--time', type=float, default=0., help='回放时间（秒）')
    parser.add_argument('--play', action='store_true', help='本地 MuJoCo 回放，需要 --run')
    parser.add_argument('--output', type=Path, help='保存当前帧 PNG 与来源清单，需要 --run')
    parser.add_argument('--camera', default='gripper_side', help='保存记录的相机预设')
    parser.add_argument('--show-collision', action='store_true')
    parser.add_argument('--hide-visual', action='store_true')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8092)
    args = parser.parse_args()
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        router = VisualViewerRouter(
            server, kind='Skill', labels=SKILL_LABELS, families=SKILL_FAMILIES,
            build=lambda skill, config: _build_app(server, skill, config, args.run),
            initial_selection=args.skill, initial_config=args.config,
            on_app_started=lambda app: (None if isinstance(app, (SweepReplayApp, SpoonReplayApp))
                                        else configure_curobo_run_log(app.session.output_dir)),
        )
        print(f'[skill] {args.skill}：http://{args.host}:{server.get_port()}', flush=True)
        print(f'[skill] 记录：{router.app.run_dir if isinstance(router.app, (SweepReplayApp, SpoonReplayApp)) else router.app.session.output_dir}', flush=True)
        while True:
            router.tick()
            time.sleep(.03)
    except KeyboardInterrupt:
        print('[skill] 已退出', flush=True)
    finally:
        server.stop()
