"""view_skill/view_task 的保存记录分支；复用只读回放，不创建在线会话。"""
from __future__ import annotations

import argparse
from pathlib import Path
import time


_SKILLS = {'pick_bowl': 'pick', 'insert_bowl': 'insert', 'extract_bowl': 'extract',
           'place_bowl': 'place', 'rotate_knob': 'rotate_knob_0',
           'handle_pull': 'handle_pull', 'push_open': 'push_open', 'push_close': 'push_close'}
_TASKS = {'microwave_task1', 'microwave_object_insert', 'microwave_object_extract',
          'microwave_knob_rotate', 'microwave_door_open', 'microwave_door_close',
          'microwave_door_cycle'}
_TASK_FILTERS = {'microwave_knob_rotate': 'rotate_knob_0',
                 'microwave_object_insert': 'object_insert',
                 'microwave_object_extract': 'object_extract',
                 'microwave_door_cycle': 'door_cycle',
                 'microwave_door_open': 'open_door', 'microwave_door_close': 'close_door'}


def dispatch_saved_run(kind, argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--run', type=Path)
    probe.add_argument(f'--{kind}')
    args, _ = probe.parse_known_args(argv)
    selection = getattr(args, kind)
    # 其他任务已有专用记录适配；没有 --run 时保留在线场景编辑入口。
    if args.run is None or (selection and selection not in (_SKILLS if kind == 'skill' else _TASKS)):
        return False
    parser = argparse.ArgumentParser(description='只读检查保存的微波炉 MuJoCo 记录')
    parser.add_argument(f'--{kind}')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8092 if kind == 'skill' else 8093)
    parser.add_argument('--stage', help='保存记录中的阶段 ID')
    parser.add_argument('--mode', choices=('actual', 'planned'), help='默认使用记录本身的数据类型；实际记录为 actual')
    parser.add_argument('--time', type=float, default=0.)
    parser.add_argument('--show-collision', action='store_true')
    parser.add_argument('--hide-visual', action='store_true')
    parser.add_argument('--play', action='store_true', help='打开本地 MuJoCo 只读回放窗口')
    parser.add_argument('--output', type=Path, help='保存当前帧 PNG 与来源 manifest')
    parser.add_argument('--camera', default='gripper_side')
    args = parser.parse_args(argv)
    skill = _SKILLS.get(selection) if kind == 'skill' else _TASK_FILTERS.get(selection)
    from gene_agent.feedback.realworld_record import load_realworld_record
    realworld = load_realworld_record(args.run, skill=skill)
    if realworld is not None:
        if args.mode == 'planned':
            raise ValueError('现场报告面板显示静态观测与实测端点/unknown；不是规划轨迹回放。请去掉 --mode planned 查看原始规划目标和证据')
        from gene_agent.visualization.realworld_record import RealworldRecordApp, save_static_snapshot
        if args.play:
            raise ValueError('现场报告只有静态观测和 SDK 实测端点，没有可播放的完整 actual 轨迹；请用 Viser 报告面板')
        if args.stage or args.time:
            raise ValueError('现场记录没有连续时间轴；请在 Viser 中选择保存报告和静态观测帧')
        if args.output:
            import json
            print(json.dumps(save_static_snapshot(realworld, args.output, camera_name=args.camera), ensure_ascii=False))
            return True
        import viser
        server = viser.ViserServer(host=args.host, port=args.port)
        try:
            app = RealworldRecordApp(server, realworld)
            print(f'[{kind}-evidence] 静态观测 + 实测端点/unknown：http://{args.host}:{server.get_port()}', flush=True)
            while True:
                app.tick()
                time.sleep(.03)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
        return True
    if args.play or args.output is not None:
        from gene_agent.simulation.skill_viewer import SkillMujocoViewer
        viewer = SkillMujocoViewer(args.run, skill=skill)
        viewer.select(args.stage, mode=args.mode)
        viewer.seek(args.time)
        if args.output:
            import json
            print(json.dumps(viewer.save_snapshot(args.output, camera_name=args.camera), ensure_ascii=False))
        if args.play:
            viewer.launch(camera_name=args.camera)
        return True
    import viser
    from gene_agent.visualization.skill_inspector import SkillInspectorApp
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        app = SkillInspectorApp(server, args.run, skill=skill, stage_id=args.stage,
            mode=args.mode, show_collision=args.show_collision, show_visual=not args.hide_visual,
            initial_time_s=args.time, read_only=True)
        print(f'[{kind}-replay] {args.run.resolve()}：http://{args.host}:{server.get_port()}', flush=True)
        while True:
            app.tick()
            time.sleep(.03)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return True
