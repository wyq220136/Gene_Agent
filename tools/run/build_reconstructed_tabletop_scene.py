"""从冻结的 brush_0 RGB-D 自动补齐三物体感知，并生成 Pika 桌面场景。"""

from __future__ import annotations

import argparse
from pathlib import Path

from gene_agent.perception_modules.scene_builder import build_from_capture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=Path("data/realworld/brush_0"))
    parser.add_argument("--scene-dir", type=Path,
                        default=Path("data/realworld/brush_cube_dustpan_0"))
    parser.add_argument("--dustpan-prompt", default="a white plastic dustpan")
    parser.add_argument("--cube-prompt", default="a small green cube")
    parser.add_argument("--cube-scale", type=float, default=0.5,
                        help="方块任务场景线性尺寸相对 mask/深度估计的比例；本场景经用户确认用 0.5")
    parser.add_argument("--no-inference", action="store_true",
                        help="只使用已有 mask 与 mesh；不启动模型服务")
    parser.add_argument("--refresh-perception", action="store_true",
                        help="重新运行 SAM3 和簸箕 SAM3D；绿方块仍从 mask 与深度拟合")
    args = parser.parse_args()
    scene = build_from_capture(
        args.capture_dir, args.scene_dir,
        prompts={"dustpan": args.dustpan_prompt, "cube": args.cube_prompt},
        allow_inference=not args.no_inference,
        refresh_perception=args.refresh_perception,
        cube_scene_scale_factor=args.cube_scale,
    )
    print(f"场景 JSON: {Path(scene['scene_xml_path']).with_name('scene.json')}")
    print(f"MuJoCo/Viser 共用 MJCF: {scene['scene_xml_path']}")
    for name, object_spec in scene["objects"].items():
        xyz = [row[3] for row in object_spec["T_world_object"][:3]]
        print(f"{name}: {xyz} m, size={object_spec['extents_xyz_m']} m")


if __name__ == "__main__":
    main()
