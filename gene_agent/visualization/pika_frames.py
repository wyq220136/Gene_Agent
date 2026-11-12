"""读取离线 Pika 报告的预览；显示与截图共享同一 MuJoCo 状态。"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.robots.pika_frames import model_frames
from gene_agent.visualization.frame_axes import append_frame_axes


COLORS = {
    "urdf_link7": (150, 150, 150),
    "pika_model_origin": (255, 180, 30),
    "simulation_tcp": (235, 60, 180),
    "pad_center": (30, 190, 110),
    "finger_distal_center": (40, 180, 240),
    "sdk_nominal_flange": (110, 80, 230),
    "reference_control": (245, 75, 40),
}


def load_audit(path: Path):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if report.get("schema_version") != 1 or not report.get("samples"):
        raise ValueError("不是有效的 Pika frame 核对报告")
    robot_path = Path(report["robot_xml"]).with_name("robot_metadata.json")
    robot = json.loads(robot_path.read_text())
    model = mujoco.MjModel.from_xml_path(report["robot_xml"])
    return report, robot, model, mujoco.MjData(model)


def apply_sample(report, robot, model, data, index):
    sample = report["samples"][index]
    current = model_frames(model, data, robot, sample["joints_rad"], sample["requested_width_m"])
    # 拒绝把改动后的模型与历史报告混合显示。
    for name, pose in current["frames"].items():
        if not np.allclose(pose, sample["frames"][name], atol=1e-7):
            raise ValueError(f"报告与模型不一致：{name}；请重新生成报告")
    return {key: np.asarray(value) for key, value in sample["frames"].items()}


def render_audit(report_path: Path, output: Path) -> list[Path]:
    """保存全景、指端、参考控制 frame 和另一姿态，供离线检查。"""
    from PIL import Image, ImageDraw, ImageFont
    report, robot, model, data = load_audit(report_path)
    output.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = 1200
    model.vis.global_.offheight = 900
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    images = []
    for name, index, azimuth, elevation, distance, selected in (
        ("overview", 0, 125, -20, 1.7, tuple(COLORS)),
        ("finger_points", 1, 90, -10, .38,
         ("pika_model_origin", "pad_center", "simulation_tcp", "finger_distal_center")),
        ("control_frames", 1, 165, -25, .46,
         ("urdf_link7", "sdk_nominal_flange", "pika_model_origin", "reference_control", "simulation_tcp")),
        ("bent_pose", 3, 135, -25, .48, tuple(COLORS)),
    ):
        frames = apply_sample(report, robot, model, data, index)
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        if name == "overview":
            camera.lookat[:] = data.body("pika_gripper_base_link").xpos / 2
        else:
            origin = frames["pika_model_origin"]
            camera.lookat[:] = origin[:3, 3] + origin[:3, 0] * .10
        with mujoco.Renderer(model, height=900, width=1200) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            if name != "overview":
                # 局部核对使用半透明外观，避免指垫中心/控制点被实体遮住。
                # 只改 renderer scene，不影响模型、碰撞或保存的状态。
                for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                    geom.rgba[3] = .24
            for frame_name in selected:
                if frame_name not in frames:
                    continue
                append_frame_axes(renderer.scene, {frame_name: frames[frame_name]}, axis_length_m=.028)
                marker = renderer.scene.geoms[renderer.scene.ngeom - 1]
                marker.rgba[:] = [*(np.asarray(COLORS[frame_name])/255), 1]
                marker.size[:] = .0035
            image = Image.fromarray(renderer.render())
        canvas = Image.new("RGB", (1200, 1060), "#f1f3f5")
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 909), f"{name}: sample {index}; width {report['samples'][index]['requested_width_m']*1000:.0f} mm", fill="black", font=font)
        for row, key in enumerate(k for k in selected if k in frames):
            col, line = row % 3, row // 3
            x, y = 15 + col * 395, 943 + line * 29
            draw.ellipse((x, y+5, x+12, y+17), fill=COLORS[key])
            draw.text((x+20, y), key, fill="black", font=font)
        draw.text((15, 1030), "Offline geometry only. SDK base identity assumption; no physical TCP calibration.", fill="#444444", font=font)
        target = output / f"{name}.png"
        canvas.save(target)
        images.append(target)
    return images


def serve_audit(report_path: Path, *, host: str = "127.0.0.1", port: int = 8094):
    import threading
    import viser
    from gene_agent.visualization.pick_bowl.scene import PickBowlScene, _pose
    report, robot, model, data = load_audit(report_path)
    frames = apply_sample(report, robot, model, data, 0)
    server = viser.ViserServer(host=host, port=port)
    server.scene.set_up_direction("+z")
    scene = PickBowlScene(server, model, data, {}, {})
    handles = {}
    for key, pose in frames.items():
        wxyz, position = _pose(pose)
        root = f"/pika_audit/{key}"
        frame = server.scene.add_frame(root, wxyz=wxyz, position=position,
                                       axes_length=.035, axes_radius=.0012)
        server.scene.add_icosphere(root + "/point", radius=.0035, color=COLORS[key])
        server.scene.add_label(root + "/label", key, position=(.0, .008, .008))
        handles[key] = frame
    with server.gui.add_folder("Offline frame audit"):
        server.gui.add_markdown("**仅离线几何预览，无设备连接。**\n\n"
                                "轴色：X 红 / Y 绿 / Z 蓝。点名是不同参考点，不代表实测标定。\n\n"
                                "SDK frame 使用名义型号及同 base 假设；选择样本可查看不同关节与开口。")
        choices = [f"{i}: pose {i//2}, width {s['requested_width_m']*1000:.0f} mm"
                   for i, s in enumerate(report["samples"])]
        selected = server.gui.add_dropdown("Sample", options=choices)
        focus = server.gui.add_checkbox("Focus on Pika", initial_value=True)
        mesh_visible = server.gui.add_checkbox("Show mesh (off to inspect hidden frames)", initial_value=True)
        mesh_visible.on_update(lambda event: scene.set_visual_visible(event.target.value))
        for key, frame in handles.items():
            checkbox = server.gui.add_checkbox(key, initial_value=True)
            checkbox.on_update(lambda event, item=frame: setattr(item, "visible", event.target.value))
    lock = threading.Lock()

    def camera(client):
        origin = data.body("pika_gripper_base_link").xpos.copy()
        if focus.value:
            center = origin + data.body("pika_gripper_base_link").xmat.reshape(3,3)[:,0] * .10
            client.camera.look_at = center
            client.camera.position = center + np.array([.28, -.32, .18])
        else:
            client.camera.look_at = origin / 2
            client.camera.position = origin / 2 + np.array([1.0, -1.0, .6])

    @server.on_client_connect
    def connected(client):
        with lock:
            camera(client)

    @selected.on_update
    def update(_event):
        with lock, server.atomic():
            current = apply_sample(report, robot, model, data, choices.index(selected.value))
            scene.update_visuals(data)
            for key, pose in current.items():
                handles[key].wxyz, handles[key].position = _pose(pose)
            for client in server.get_clients().values():
                camera(client)

    @focus.on_update
    def refocus(_event):
        with lock:
            for client in server.get_clients().values():
                camera(client)
    print(f"Pika frame 预览：http://{host}:{port}；Ctrl+C 退出", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
