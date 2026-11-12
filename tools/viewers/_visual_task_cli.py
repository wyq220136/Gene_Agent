"""Task 在线入口分发：门循环与 Bowl 放入共用同步 Viser 生命周期。"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import viser

from gene_agent.tasks.policies.microwave_task.planning.door_session import configure_curobo_run_log
from gene_agent.visualization.microwave.bowl_session import OBJECT_TASKS
from gene_agent.visualization.viser_sim.app import DOOR_TASKS, DoorVisualApp
from gene_agent.visualization.viser_sim.object_task_app import ObjectTaskApp
from gene_agent.visualization.viser_sim.task1_app import Task1VisualApp
from gene_agent.tasks.policies.microwave_task.visual_session import TASK1_SELECTION
from gene_agent.visualization.microwave.knob_session import KNOB_TASKS
from gene_agent.visualization.viser_sim.knob_app import KnobTaskVisualApp
from gene_agent.visualization.microwave.extract_session import EXTRACT_SUBTASKS
from gene_agent.visualization.viser_sim.extract_task_app import ExtractTaskVisualApp
from gene_agent.visualization.viser_sim.navigation import VisualViewerRouter
from gene_agent.visualization.viser_sim.sweep_replay_app import SweepReplayApp, SWEEP_TASKS
from gene_agent.visualization.viser_sim.spoon_replay_app import SpoonReplayApp
from gene_agent.feedback.tabletop.spoon_run import SPOON_TASKS
from gene_agent.visualization.viser_sim.cup_replay_app import CupReplayApp
from gene_agent.feedback.tabletop.cup_run import CUP_TASKS


TASK_LABELS = {**DOOR_TASKS, **OBJECT_TASKS, **EXTRACT_SUBTASKS, **KNOB_TASKS,
               TASK1_SELECTION: '完整微波炉 Task1', **SWEEP_TASKS, **SPOON_TASKS,
               **CUP_TASKS}
TASK_FAMILIES = {
    **dict.fromkeys(DOOR_TASKS, 'door'),
    **dict.fromkeys(OBJECT_TASKS, 'object'),
    **dict.fromkeys(EXTRACT_SUBTASKS, 'extract'),
    **dict.fromkeys(KNOB_TASKS, 'knob'),
    TASK1_SELECTION: 'task1',
    **dict.fromkeys(SWEEP_TASKS, 'sweep'),
    **dict.fromkeys(SPOON_TASKS, 'spoon'),
    **dict.fromkeys(CUP_TASKS, 'cup'),
}


def _build_app(server, task, config, run_dir=None):
    if task in CUP_TASKS:
        return CupReplayApp(server, initial_selection=task, run_dir=run_dir)
    if task in SPOON_TASKS:
        return SpoonReplayApp(server, initial_selection=task, run_dir=run_dir)
    if task in SWEEP_TASKS:
        return SweepReplayApp(server, initial_selection=task, run_dir=run_dir)
    if task == TASK1_SELECTION:
        return Task1VisualApp(server, config_path=config)
    if task in KNOB_TASKS:
        return KnobTaskVisualApp(server, initial_selection=task, config_path=config)
    if task in EXTRACT_SUBTASKS:
        return ExtractTaskVisualApp(server, initial_selection=task, config_path=config)
    if task in OBJECT_TASKS:
        return ObjectTaskApp(server, initial_selection=task, config_path=config)
    return DoorVisualApp(server, mode='task', initial_selection=task, config_path=config)


def main() -> None:
    parser = argparse.ArgumentParser(description='在线检查 Task；清扫项回放实际 MuJoCo 状态')
    parser.add_argument('--task', choices=tuple(TASK_LABELS), default=next(iter(DOOR_TASKS)))
    parser.add_argument('--config', type=Path, help='门/Bowl/旋钮/Task1 场景配置（scene_config.json）')
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
    parser.add_argument('--port', type=int, default=8093)
    args = parser.parse_args()
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        router = VisualViewerRouter(
            server, kind='Task', labels=TASK_LABELS, families=TASK_FAMILIES,
            build=lambda task, config: _build_app(server, task, config, args.run),
            initial_selection=args.task, initial_config=args.config,
            on_app_started=lambda app: (None if isinstance(app, (SweepReplayApp, SpoonReplayApp))
                                        else configure_curobo_run_log(app.session.output_dir)),
        )
        print(f'[task] {args.task}：http://{args.host}:{server.get_port()}', flush=True)
        print(f'[task] 记录：{router.app.run_dir if isinstance(router.app, (SweepReplayApp, SpoonReplayApp)) else router.app.session.output_dir}', flush=True)
        while True:
            router.tick()
            time.sleep(.03)
    except KeyboardInterrupt:
        print('[task] 已退出', flush=True)
    finally:
        server.stop()
