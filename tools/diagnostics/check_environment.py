"""检查开发环境并留下可查看证据，不运行机器人 Task 或连接真机。

该入口只验证 PyTorch CUDA、MuJoCo 基础物理/离屏渲染，以及旧项目需要的
cuRobo 公共 API 是否可导入。真实机器人资产、IK 和轨迹规划需要单独验收，
不能根据这里的 PASS 推断它们可运行。每项独立捕获错误，便于一次看到全部缺口。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "checks": {}}

    def check(name, operation):
        try:
            report["checks"][name] = {"success": True, "details": operation()}
        except Exception as error:
            report["checks"][name] = {
                "success": False, "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        print(name, "PASS" if report["checks"][name]["success"] else "FAIL", flush=True)

    def cuda():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch 无法使用 CUDA")
        a = torch.eye(4, device="cuda")
        if not torch.equal(a @ a, a):
            raise RuntimeError("CUDA 矩阵计算结果错误")
        torch.cuda.synchronize()
        return {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0)}

    def mujoco_check():
        # 只给当前进程选择离屏后端，不更改整个 conda 环境，保留本地 viewer 的默认行为。
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco
        import numpy as np
        from PIL import Image
        xml = '''<mujoco><option timestep="0.002"/>
        <worldbody><light pos="0 -1 3"/>
        <camera name="check" pos="1 -1 0.8" xyaxes="0.707 0.707 0 -0.348 0.348 0.870"/>
        <geom type="plane" size="1 1 0.1" rgba="0.7 0.7 0.7 1"/>
        <body name="ball" pos="0 0 0.3"><freejoint/>
        <geom type="sphere" size="0.06" mass="0.1" rgba="0.1 0.4 0.9 1"/>
        </body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        height = float(data.body("ball").xpos[2])
        if not 0.04 < height < 0.08 or data.ncon == 0:
            raise RuntimeError(f"落球接触验证失败: z={height}, ncon={data.ncon}")
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera="check")
            pixels = renderer.render().copy()
        if pixels.shape != (480, 640, 3) or float(np.std(pixels)) < 1:
            raise RuntimeError("渲染没有生成有效 RGB 图像")
        path = args.output / "mujoco_environment.png"
        Image.fromarray(pixels).save(path)
        return {"version": mujoco.__version__, "height_m": height,
                "contacts": int(data.ncon), "image": str(path.resolve()),
                "render_backend": os.environ["MUJOCO_GL"]}

    def curobo_check():
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
        return {"version": importlib.metadata.version("nvidia-curobo"),
                "imports": [MotionPlanner.__name__, MotionPlannerCfg.__name__,
                            ToolPoseCriteria.__name__],
                "gpu_planning_tested": False}

    check("torch_cuda", cuda)
    check("mujoco_physics_render", mujoco_check)
    check("curobo_api_import", curobo_check)
    report["success"] = all(item["success"] for item in report["checks"].values())
    (args.output / "environment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
