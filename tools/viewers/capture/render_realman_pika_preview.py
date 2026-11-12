"""渲染直立 Realman 单臂与 Pika 夹爪的装配预览；不控制真机。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image
from gene_agent.robots import build_robot, measure_gripper_width, set_gripper_width
from gene_agent.visualization.style import apply_scene_style


def build_preview_model(source_xml: Path, output: Path) -> Path:
    path = output / "realman_single_pika_preview.xml"
    root = ET.parse(source_xml).getroot()
    for body in root.findall(".//body"):
        visual_color = ".2 .68 .76 1" if body.get("name", "").startswith("pika_") else ".86 .51 .29 1"
        for geom in body.findall("geom"):
            if geom.get("contype", "1") == "0":
                geom.set("group", "1")
                geom.set("rgba", visual_color)
            else:
                geom.set("group", "3")
                geom.set("rgba", "0.2 0.8 0.3 0")
    world = root.find("worldbody")
    base = world.find(".//body[@name='base_link']")
    if base is None:
        raise ValueError("导入后的 Realman 模型缺少 base_link")
    axis_length_m = .22
    for label, direction, color in (
        ("x", (1, 0, 0), "1 .1 .1 1"),
        ("y", (0, 1, 0), ".1 .85 .1 1"),
        ("z", (0, 0, 1), ".15 .35 1 1"),
    ):
        end = " ".join(str(axis_length_m * value) for value in direction)
        ET.SubElement(base, "geom", name=f"base_axis_{label}", type="capsule",
                      fromto=f"0 0 0 {end}", size=".006", rgba=color,
                      contype="0", conaffinity="0", group="2")
        ET.SubElement(base, "geom", name=f"base_axis_{label}_tip", type="sphere",
                      pos=end, size=".014", rgba=color,
                      contype="0", conaffinity="0", group="2")
    ET.SubElement(base, "geom", name="base_axis_origin", type="sphere", size=".015",
                  rgba="1 1 1 1", contype="0", conaffinity="0", group="2")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.SubElement(world, "geom", name="preview_floor", type="plane", size="1 1 .1",
                  pos="0 0 -0.02", rgba=".78 .8 .83 1")
    apply_scene_style(root)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")
    mujoco.MjModel.from_xml_path(str(path))
    return path


def _set_zero_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for joint_id in range(model.njnt):
        data.qpos[model.jnt_qposadr[joint_id]] = 0
    mujoco.mj_forward(model, data)


def _render(model: mujoco.MjModel, data: mujoco.MjData, *, target: str, azimuth: float) -> Image.Image:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = azimuth
    camera.elevation = -22 if target == "gripper" else -25
    if target == "gripper":
        camera.lookat[:] = data.body("pika_gripper_base_link").xpos
        camera.distance = .39
    elif target == "base":
        camera.lookat[:] = data.body("base_link").xpos + np.array([0, 0, .09])
        camera.distance = .58
    else:
        camera.lookat[:] = .5 * (data.body("base_link").xpos + data.body("pika_gripper_base_link").xpos)
        camera.distance = max(1.3, 2.0 * np.linalg.norm(
            data.body("pika_gripper_base_link").xpos - data.body("base_link").xpos))
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        if target == "base":
            # 仅在底座局部图里把外壳半透明，显示从真实 base_link 原点穿出的 +Z。
            for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM:
                    continue
                name = model.geom(geom.objid).name
                if not name.startswith("base_axis_"):
                    geom.rgba[3] = .24
        return Image.fromarray(renderer.render())


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    robot = build_robot(output / "robot", profile="realman_single_pika")
    xml = build_preview_model(Path(robot["xml_path"]), output)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    _set_zero_pose(model, data)
    _render(model, data, target="overview", azimuth=125).save(output / "zero_overview.png")
    _render(model, data, target="base", azimuth=125).save(output / "zero_base_axes.png")
    _render(model, data, target="gripper", azimuth=135).save(output / "zero_gripper.png")
    set_gripper_width(model, data, robot, "left", .04)
    frames = [_render(model, data, target="gripper", azimuth=135)]
    for step in range(1000):
        mujoco.mj_step(model, data)
        if (step + 1) % 50 == 0:
            frames.append(_render(model, data, target="gripper", azimuth=135))
    frames[-1].save(output / "gripper_after_40mm.png")
    frames[0].save(output / "pika_close_40mm.gif", save_all=True,
                   append_images=frames[1:], duration=90, loop=0)
    actual_width = measure_gripper_width(model, data, robot, "left")["width_m"]
    (output / "preview_notes.json").write_text(json.dumps({
        "arm": "Realman left arm, upright at world origin, no shared dual-arm body",
        "gripper_source": robot["grippers"]["left"]["source_urdf_path"],
        "mount_source": "Pika base at Realman link_7 origin; former left gripper joint orientation plus 180 deg about flange local Z, per user visual feedback",
        "mount_status": "provisional visual fit, not calibrated Pika flange transform",
        "tcp_status": "provisional geometric site in production robot model; physical TCP not calibrated",
        "joint_state": "all 7 Realman arm joints and both Pika finger joints are exactly zero",
        "base_axes": "base_link origin; X red, Y green, Z blue; each 0.22 m",
        "gripper_close_demo": {"target_width_m": .04, "actual_width_m": actual_width,
                               "simulated_seconds": float(data.time)},
        "model": str(xml),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"零位预览完成：{output / 'zero_overview.png'}；{output / 'zero_base_axes.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/realman-single-pika-preview"))
    run(parser.parse_args().output.resolve())
