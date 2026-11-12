"""录制在线纯视觉 Skill/Task 的实际规划轨迹。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import viser

from gene_agent.tasks.policies.microwave_task.planning.door_session import configure_curobo_run_log
from gene_agent.visualization.microwave.bowl_session import OBJECT_SKILLS, OBJECT_TASKS
from gene_agent.visualization.microwave.extract_session import EXTRACT_PLACE_SKILLS
from gene_agent.visualization.microwave.extract_session import EXTRACT_SUBTASKS
from gene_agent.visualization.microwave.knob_session import KNOB_SKILLS, KNOB_TASKS
from gene_agent.tasks.policies.microwave_task.visual_session import TASK1_SELECTION
from gene_agent.visualization.motion_debug.capture import _free_port
from gene_agent.visualization.viser_sim.app import (
    DOOR_SKILLS, DOOR_TASKS, DoorVisualApp,
)
from gene_agent.visualization.viser_sim.object_app import ObjectVisualApp
from gene_agent.visualization.viser_sim.object_task_app import ObjectTaskApp
from gene_agent.visualization.viser_sim.extract_place_app import ExtractPlaceVisualApp
from gene_agent.visualization.viser_sim.extract_task_app import ExtractTaskVisualApp
from gene_agent.visualization.viser_sim.knob_app import KnobVisualApp, KnobTaskVisualApp
from gene_agent.visualization.viser_sim.task1_app import Task1VisualApp
from gene_agent.visualization.viser_sim.recording import (
    CAMERA_LABELS, _source_revision, record_visual_run,
)
from tools.diagnostics.check_visual_acceptance import decode_frame_count


def record_all(root: Path, *, fps: int, resume: bool = False,
               width: int = 1500, height: int = 900,
               software_renderer: bool = False) -> None:
    """逐项调用同一在线录制入口，保留每项自己的成功配置与诊断。"""
    configs = {
        "door": root / "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json",
        "bowl": root / "data/task_1/visualization/bowl_visual_debug.json",
        "knob": root / "data/task_1/visualization/knob_visual_tip5_verified.json",
        "task1": root / "data/task_1/visualization/task1_visual_explicit90_verified.json",
    }
    missing = [str(path) for path in configs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("验收配置不存在：" + "、".join(missing))
    expected_configs = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in configs.items()
    }
    source_revision = _source_revision()
    units = [
        ("skill", "handle_pull", "door"),
        ("skill", "push_open", "door"),
        ("skill", "push_close", "door"),
        ("skill", "pick_bowl", "bowl"),
        ("skill", "insert_bowl", "bowl"),
        ("skill", "extract_bowl", "bowl"),
        ("skill", "place_bowl", "bowl"),
        ("skill", "rotate_knob", "knob"),
        ("task", "microwave_door_open", "door"),
        ("task", "microwave_door_close", "door"),
        ("task", "microwave_door_cycle", "door"),
        ("task", "microwave_object_insert", "bowl"),
        ("task", "microwave_object_extract", "bowl"),
        ("task", "microwave_knob_rotate", "knob"),
        ("task", TASK1_SELECTION, "task1"),
    ]
    failures = []
    for kind, name, config_key in units:
        # overview 是每项的主验收机位；接触 Skill 再补夹爪侧面。
        cameras = ("overview", "gripper_side") if kind == "skill" else ("overview",)
        for camera in cameras:
            output = root / "artifacts/viser_acceptance" / name / f"{camera}.mp4"
            manifest_path = output.with_suffix(".json")
            if resume and output.is_file() and output.stat().st_size > 1024 and manifest_path.is_file():
                try:
                    old = json.loads(manifest_path.read_text(encoding="utf-8"))
                    valid = (old.get("source") == "online_viser_visual_run"
                             and old.get("camera") == camera and old.get("fps") == fps
                             and old.get("width", width) == width
                             and old.get("height", height) == height
                             and old.get("run", {}).get("selection_id") == name
                             and old.get("run", {}).get("success") is True
                             and old.get("source_revision", {}).get("runtime_sha256")
                                 == source_revision["runtime_sha256"]
                             and isinstance(old.get("config"), dict)
                             and all(old["config"].get(key) == value for key, value
                                     in expected_configs[config_key].items()))
                    if valid:
                        valid = decode_frame_count(output) == len(old.get("stage_ids_by_frame", ()))
                except (OSError, ValueError, RuntimeError):
                    valid = False
                if valid:
                    print(f"[batch] 已验收，跳过 {kind}/{name}/{camera}", flush=True)
                    continue
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--kind", kind, "--name", name,
                       "--config", str(configs[config_key]),
                       "--camera", camera, "--fps", str(fps),
                       "--width", str(width), "--height", str(height)]
            if software_renderer:
                command.append("--software-renderer")
            print(f"[batch] {kind}/{name}/{camera}", flush=True)
            result = subprocess.run(command, cwd=root, check=False)
            if result.returncode:
                failures.append(f"{kind}/{name}/{camera}: exit {result.returncode}")
    if failures:
        raise RuntimeError("验收录制未全部成功：" + "；".join(failures))
    print(f"[batch] {len(units)} 项视频及 Skill 侧视已保存到 artifacts/viser_acceptance", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("skill", "task"), default="skill")
    parser.add_argument("--all", action="store_true", help="依次录制全部 8 Skill、6 子任务和完整 Task")
    parser.add_argument("--resume", action="store_true", help="与 --all 合用，跳过已有成功视频与清单")
    parser.add_argument("--name", help="Skill 或 Task 名称；默认首个可用名称")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--camera", choices=tuple(CAMERA_LABELS), default="overview")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1500,
                        help="录制画面宽度；降低数值可缩小视频")
    parser.add_argument("--height", type=int, default=900,
                        help="录制画面高度；降低数值可缩小视频")
    parser.add_argument("--software-renderer", action="store_true",
                        help="无可用 OpenGL 的服务器回退到原 SwiftShader 软件渲染")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.all:
        if args.name or args.config or args.output:
            parser.error("--all 不能与 --name、--config 或 --output 合用")
        record_all(root, fps=args.fps, resume=args.resume,
                   width=args.width, height=args.height,
                   software_renderer=args.software_renderer)
        return
    if args.resume:
        parser.error("--resume 只能与 --all 合用")
    names = ((DOOR_SKILLS | OBJECT_SKILLS | EXTRACT_PLACE_SKILLS | KNOB_SKILLS)
             if args.kind == "skill" else
             (DOOR_TASKS | OBJECT_TASKS | EXTRACT_SUBTASKS | KNOB_TASKS |
              {TASK1_SELECTION: "完整微波炉 Task1"}))
    name = args.name or next(iter(names))
    if name not in names:
        parser.error(f"当前 {args.kind} 仅支持：{', '.join(names)}")
    output = (args.output or root / "artifacts/viser_acceptance" / name /
              f"{args.camera}.mp4").resolve()
    port = _free_port()
    server = viser.ViserServer(host="127.0.0.1", port=port)
    try:
        if args.kind == "skill" and name in OBJECT_SKILLS:
            app = ObjectVisualApp(server, initial_selection=name,
                                  config_path=args.config)
        elif args.kind == "skill" and name in EXTRACT_PLACE_SKILLS:
            app = ExtractPlaceVisualApp(server, initial_selection=name,
                                        config_path=args.config)
        elif args.kind == "skill" and name in KNOB_SKILLS:
            app = KnobVisualApp(server, initial_selection=name,
                                config_path=args.config)
        elif args.kind == "task" and name in OBJECT_TASKS:
            app = ObjectTaskApp(server, initial_selection=name,
                                config_path=args.config)
        elif args.kind == "task" and name in EXTRACT_SUBTASKS:
            app = ExtractTaskVisualApp(server, initial_selection=name,
                                       config_path=args.config)
        elif args.kind == "task" and name in KNOB_TASKS:
            app = KnobTaskVisualApp(server, initial_selection=name,
                                    config_path=args.config)
        elif args.kind == "task" and name == TASK1_SELECTION:
            app = Task1VisualApp(server, initial_selection=name,
                                 config_path=args.config)
        else:
            app = DoorVisualApp(server, mode=args.kind, initial_selection=name,
                                config_path=args.config)
        configure_curobo_run_log(app.session.output_dir)
        if isinstance(app, ObjectVisualApp) and app.preview_partitions is not None:
            # 在线界面在真实计时器驱动下逐分区预览；无界面录制器会紧循环调用
            # tick()，没有用户等待时间，预览时间轴可能长期停在首个分区。
            # 这里只关闭规划中的预览，最终仍从同一完整规划轨迹逐帧录制。
            app.preview_partitions.value = False
        app._start_plan()
        while (app.planning if isinstance(app, ObjectVisualApp)
               else app.controller.planning):
            app.tick()
        run = app.controller.run
        if run is None or not run.success:
            failure = None if run is None else run.stages[-1].failure
            raise RuntimeError(f"{name} 未形成可录制的完整规划：{failure or app.status.content}")
        record_visual_run(app, output, viser_port=port, camera=args.camera,
                          fps=args.fps, width=args.width, height=args.height,
                          software_renderer=args.software_renderer)
        print(f"视频：{output}", flush=True)
        print(f"清单：{output.with_suffix('.json')}", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
