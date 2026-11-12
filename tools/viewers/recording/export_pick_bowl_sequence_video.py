"""把已有抓碗接口 run 连续重演成 MuJoCo 视频。

该 run 的 pregrasp、grasp、闭爪会从上一阶段的实际仿真状态继续执行；
postgrasp 是沿已执行 grasp 关节轨迹反向撤离的诊断动作，没有重新做
接触状态下的 cuRobo 碰撞规划。视频不代表新 PickSkill 已完成直线接近。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gene_agent.feedback.state import restore_state, state_id
from gene_agent.simulation import MujocoRuntime
from gene_agent.visualization.style import camera_presets, configure_camera


def assert_plan_starts_at_state(actual, planned_rows, *, tolerance_rad: float = .05) -> float:
    """显式拒绝每段从 home 重启或接上不属于当前状态的旧轨迹。"""
    actual_q = np.asarray(actual, dtype=float)
    planned = np.asarray(planned_rows, dtype=float)
    if planned.ndim != 2 or len(planned) < 2 or planned.shape[1:] != actual_q.shape:
        raise ValueError("保存的阶段轨迹维度与当前关节状态不一致")
    error = float(np.max(np.abs(planned[0] - actual_q)))
    if error > tolerance_rad:
        raise ValueError(f"阶段轨迹起点不等于上一阶段实际状态：最大差 {error:.4f} rad")
    return error


def postgrasp_return_trajectory(
    measured_closed_q, grasp_rows, *, duration_s: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """从闭爪后的实测关节位置起步，沿刚走过的关节路径撤离。"""
    measured = np.asarray(measured_closed_q, dtype=float)
    path = np.asarray(grasp_rows, dtype=float)
    if path.ndim != 2 or len(path) < 2 or path.shape[1:] != measured.shape:
        raise ValueError("grasp 轨迹与闭爪后关节状态不匹配")
    if not np.isfinite(path).all() or not np.isfinite(measured).all():
        raise ValueError("postgrasp 轨迹必须全部有限")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("postgrasp 时长必须为正有限数")
    rows = np.vstack((measured, path[-2::-1]))
    times = np.linspace(0., duration_s, len(rows))
    return rows, times


class VideoSink:
    """按仿真时间采样离屏画面，并通过 ffmpeg 写入 H.264。"""

    def __init__(self, model, data, path: Path, *, fps: int = 24, width: int = 960,
                 height: int = 720):
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        self.frames = 0
        self.next_time = 0.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = configure_camera(mujoco.MjvCamera(), camera_presets(model, data)["task"])
        self.font = ImageFont.truetype("DejaVuSans.ttf", 24)
        self.process = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-crf", "22",
             "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    def begin(self, stage: str, data) -> None:
        self.next_time = 0.
        self.record(stage, data, 0.)

    def record(self, stage: str, data, elapsed_s: float) -> None:
        if elapsed_s + 1e-6 < self.next_time:
            return
        self.renderer.update_scene(data, camera=self.camera)
        frame = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rounded_rectangle((16, 16, 500, 78), radius=8, fill=(20, 28, 39))
        draw.text((30, 23), stage, font=self.font, fill="white")
        draw.text((30, 53), f"MuJoCo continuous replay  |  t={elapsed_s:.2f} s",
                  fill=(215, 226, 239))
        assert self.process.stdin is not None
        self.process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1
        self.next_time += 1. / self.fps

    def close(self) -> None:
        self.renderer.close()
        assert self.process.stdin is not None and self.process.stderr is not None
        self.process.stdin.close()
        error = self.process.stderr.read().decode("utf-8", errors="replace")
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg 写视频失败: {error}")


def export_sequence(run_dir: Path, output_dir: Path) -> dict:
    """按 open→pregrasp→grasp→close→postgrasp 连续重演已保存的成功 run。"""
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if not source.get("success"):
        raise ValueError("只接受已有 pregrasp、grasp 和闭爪成功记录的 run")
    model = mujoco.MjModel.from_xml_path(str(run / "model/scene.xml"))
    data = mujoco.MjData(model)
    restore_state(model, data, run / "stages/initial/state.npz")
    robot = json.loads((run / "model/robot/robot_metadata.json").read_text(encoding="utf-8"))
    objects = json.loads((run / "model/objects/scene_metadata.json").read_text(encoding="utf-8"))
    runtime = MujocoRuntime(model, data, {"robot": robot, "objects": objects})
    start_bowl_z = float(data.body("bowl").xpos[2])
    stages: list[dict] = []
    video_path = output / "pick_bowl_sequence.mp4"
    sink = VideoSink(model, data, video_path)
    try:
        def execute_gripper(stage: str, width: float, duration: float) -> None:
            start_id = state_id(model, data)
            sink.begin(stage, data)
            result = runtime.execute_gripper_command(
                group="right", width_m=width, duration_s=duration,
                planned_state_id=start_id, timeout_s=duration + .5,
                step_observer=lambda _m, frozen, sample: sink.record(
                    stage, frozen, sample["time_s"]),
            )
            stages.append({"name": stage, "success": result["success"],
                           "start_state_id": start_id, "end_state_id": result["end_state_id"],
                           "sim_duration_s": result["sim_duration_s"],
                           "bowl_z_m": float(data.body("bowl").xpos[2])})
            if not result["success"]:
                raise RuntimeError(f"{stage} MuJoCo 执行失败: {result['failure_type']}")

        def execute_arm(stage: str, names, rows, times,
                        *, require_saved_start: bool = False) -> None:
            actual = [float(data.qpos[model.joint(name).qposadr[0]]) for name in names]
            continuity = assert_plan_starts_at_state(
                actual, rows, tolerance_rad=.05 if require_saved_start else .07,
            )
            start_id = state_id(model, data)
            sink.begin(stage, data)
            result = runtime.execute_joint_trajectory(
                group="right", joint_names=names, times_s=times, positions_rad=rows,
                planned_state_id=start_id, timeout_s=float(times[-1]) + 1.,
                step_observer=lambda _m, frozen, sample: sink.record(
                    stage, frozen, sample["time_s"]),
            )
            stages.append({"name": stage, "success": result["success"],
                           "start_state_id": start_id, "end_state_id": result["end_state_id"],
                           "start_error_rad": continuity,
                           "max_tracking_error_rad": result["max_tracking_error_rad"],
                           "sim_duration_s": result["sim_duration_s"],
                           "bowl_z_m": float(data.body("bowl").xpos[2])})
            if not result["success"]:
                raise RuntimeError(f"{stage} MuJoCo 执行失败: {result['failure_type']}")

        execute_gripper("OPEN GRIPPER", .08, .3)
        plans = {}
        for stage in ("pregrasp", "final"):
            plan = json.loads((run / "plans" / f"{stage}.json").read_text(encoding="utf-8"))
            if not plan.get("success"):
                raise ValueError(f"源 run 的 {stage} 轨迹未成功")
            plans[stage] = plan
            rows = plan["positions_rad"]
            times = np.arange(len(rows)) * float(plan["interpolation_dt_s"])
            execute_arm("HOME -> PREGRASP" if stage == "pregrasp" else
                        "PREGRASP -> GRASP (JOINT PATH)",
                        plan["joint_names"], rows, times, require_saved_start=True)
        execute_gripper("CLOSE GRIPPER", .0, 1.0)
        names = plans["final"]["joint_names"]
        closed_q = [float(data.qpos[model.joint(name).qposadr[0]]) for name in names]
        rows, times = postgrasp_return_trajectory(
            closed_q, plans["final"]["positions_rad"], duration_s=1.2,
        )
        execute_arm("POSTGRASP RETURN (UNPLANNED)", names, rows, times)
    finally:
        sink.close()
    report = {
        "schema_version": 1,
        "source_run": str(run),
        "video": str(video_path),
        "frame_count": sink.frames,
        "stages": stages,
        "bowl_z_delta_m": float(data.body("bowl").xpos[2] - start_bowl_z),
        "limitations": [
            "视频重演的是旧接口已保存的关节规划；pregrasp 到 grasp 并非笛卡尔直线约束。",
            "postgrasp 沿已执行 grasp 关节路径反向撤离，接触状态下未做 cuRobo 规划。",
            "视频展示 MuJoCo 连续执行，不代表新 PickSkill 的完整实现或真机成功率。",
        ],
    }
    (output / "video_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_sequence(args.run, args.output), ensure_ascii=False))
