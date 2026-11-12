"""用同一份 Skill 回放记录检查 MuJoCo 画面；只读，不执行 Skill。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_agent.simulation.skill_viewer import SkillMujocoViewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="包含 model/scene.xml 的 Skill run")
    parser.add_argument("--stage", help="阶段 ID；省略时查看完整 Skill")
    parser.add_argument("--mode", choices=("actual", "planned"), default="actual")
    parser.add_argument("--time", type=float, default=0., help="所选时间轴上的秒数")
    parser.add_argument("--output", type=Path, help="PNG 截图路径；同时输出同名 JSON manifest")
    parser.add_argument("--camera", default="gripper_side", help="共享相机预设名称")
    parser.add_argument("--play", action="store_true", help="打开本地 MuJoCo 交互窗口")
    args = parser.parse_args()
    if not args.play and args.output is None:
        parser.error("单帧检查需提供 --output；交互检查需提供 --play")
    viewer = SkillMujocoViewer(args.run)
    viewer.select(args.stage, mode=args.mode)
    frame = viewer.seek(args.time)
    if args.output:
        result = viewer.save_snapshot(args.output, camera_name=args.camera)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if args.play:
        print("空格播放/暂停，左右键逐样本，[ ] 切阶段，A 返回完整 Skill", flush=True)
        viewer.launch(camera_name=args.camera)


if __name__ == "__main__":
    main()
