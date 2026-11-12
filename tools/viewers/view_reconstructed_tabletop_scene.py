"""用同一份 scene.xml 离屏渲染，或在 MuJoCo/Viser 打开重建桌面场景。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image


def load_scene(scene_dir: Path):
    path = scene_dir / "scene.json"
    scene = json.loads(path.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(scene_dir / "scene.xml"))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return scene, model, data


def render_views(scene_dir: Path, output_dir: Path) -> list[Path]:
    _, model, data = load_scene(scene_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    saved = []
    with mujoco.Renderer(model, height=800, width=1200) as renderer:
        for name, azimuth, elevation, distance, lookat in (
            ("overview", 145, -28, 1.55, (0.32, 0, 0.31)),
            ("top", 180, -82, 1.24, (0.36, 0, 0.08)),
            ("objects", 130, -38, 0.85, (0.45, 0, 0.05)),
        ):
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.distance = distance
            camera.lookat[:] = lookat
            renderer.update_scene(data, camera=camera, scene_option=option)
            path = output_dir / f"{name}.png"
            Image.fromarray(renderer.render()).save(path)
            saved.append(path)
    return saved


def view_mujoco(scene_dir: Path) -> None:
    import mujoco.viewer

    _, model, data = load_scene(scene_dir)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.32, 0, 0.27)
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -28
        viewer.opt.geomgroup[3] = 0
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def view_viser(scene_dir: Path, *, port: int) -> None:
    import viser

    from gene_agent.visualization.pick_bowl.scene import PickBowlScene

    scene_spec, model, data = load_scene(scene_dir)
    server = viser.ViserServer(host="127.0.0.1", port=port)
    server.scene.set_up_direction("+z")
    # Viser 默认视点距离约十米，会把桌面缩成小点；按现有单臂桌面范围给出初始全景。
    server.initial_camera.position=(-.45,.75,.85)
    server.initial_camera.look_at=(.30,0.,.20)
    server.initial_camera.up=(0.,0.,1.)
    server.initial_camera.fov=.8
    server.scene.set_background_image(np.full((8, 8, 3), 235, dtype=np.uint8), format="png")
    server.scene.add_frame("/world", axes_length=0.12, axes_radius=0.002)
    robot=scene_spec.get("robot")
    if robot is None:
        metadata=scene_dir/"robot/robot_metadata.json"
        robot=json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else {}
    PickBowlScene(server, model, data, robot, {})
    title = {
        "sweep_old": "旧刷子清扫", "sweep_long": "白柄长刷清扫",
        "sweep_small": "蓝色短刷清扫", "pour": "绿色杯倾倒",
        "scoop_orange": "橙色勺子舀取", "scoop_pink": "粉色勺子舀取",
    }.get(scene_spec.get("task"), "重建桌面场景")
    server.gui.add_markdown(
        f"### {title}\n"
        "来自同一份 MuJoCo scene.xml；显示名义桌面排布。"
        "这是场景预览；Skill/Task 的实际执行结果应查看对应 run。"
    )
    print(f"Viser: http://127.0.0.1:{port}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path,
                        default=Path("data/realworld/brush_cube_dustpan_0"))
    parser.add_argument("--mode", choices=("render", "mujoco", "viser"), default="render")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args()
    scene_dir = args.scene_dir.resolve()
    if args.mode == "render":
        for path in render_views(scene_dir, (args.output_dir or scene_dir / "renders").resolve()):
            print(path)
    elif args.mode == "mujoco":
        view_mujoco(scene_dir)
    else:
        view_viser(scene_dir, port=args.port)


if __name__ == "__main__":
    main()
