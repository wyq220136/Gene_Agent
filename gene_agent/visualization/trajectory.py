"""从实际 MuJoCo 执行状态生成关键帧、普通动画和分色碰撞动画。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import render_views


class TrajectoryRecorder:
    """低频采样执行观察副本；录像失败由调用方作为诊断异常处理。"""

    def __init__(
        self,
        output_dir,
        *,
        target_body: str = "bowl",
        width: int = 640,
        height: int = 480,
        fps: int = 10,
        sample_interval_s: float = 0.1,
        combine_mp4: bool = False,
        detail_view: str | None = None,
        extra_view: str | None = None,
        overview_camera: dict | None = None,
        stage_camera_overrides: dict[str, dict[str, dict]] | None = None,
        robot_metadata: dict | None = None,
        active_group: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.target_body = target_body
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.sample_interval_s = float(sample_interval_s)
        self.combine_mp4 = bool(combine_mp4)
        self.detail_view = detail_view
        self.extra_view = extra_view
        self.overview_camera = dict(overview_camera or {})
        # 阶段近景只改渲染相机；抓取桌面物体与放入炉腔可以各自避开遮挡。
        self.stage_camera_overrides = {
            stage: {view: dict(params) for view, params in views.items()}
            for stage, views in (stage_camera_overrides or {}).items()
        }
        self.robot_metadata = robot_metadata
        self.active_group = active_group
        if detail_view is not None and detail_view == "overview":
            raise ValueError("detail_view 不能重复 overview")
        if extra_view is not None and extra_view in {"overview", detail_view}:
            raise ValueError("extra_view 必须是不同的第三视角")
        if "gripper_side" in {detail_view, extra_view} and (robot_metadata is None or active_group is None):
            raise ValueError("gripper_side 录像需要活动夹爪组和 robot_metadata")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0 or self.sample_interval_s <= 0:
            raise ValueError("录像尺寸、fps 和 sample_interval_s 必须为正值")
        self._stages = {}
        self._next_time = {}
        self._palette = {}

    def record_keyframe(self, name: str, model, data) -> dict:
        return render_views(
            model,
            data,
            self.output_dir / "keyframes" / name,
            target_body=self.target_body,
            width=self.width,
            height=self.height,
            collision_overlay=True,
            site_axes=True,
            contact_debug=True,
            robot_metadata=self.robot_metadata,
            active_group=self.active_group,
        )

    def _capture(self, stage: str, model, data) -> dict:
        index = len(self._stages.setdefault(stage, []))
        frame_dir = self.output_dir / "frames" / stage / f"{index:06d}"
        cameras = ({"overview": dict(self.overview_camera)}
                   if self.overview_camera else {})
        cameras.update(self.stage_camera_overrides.get(stage, {}))
        result = render_views(
            model,
            data,
            frame_dir,
            target_body=self.target_body,
            width=self.width,
            height=self.height,
            view_names=("overview",) + tuple(name for name in (self.detail_view, self.extra_view) if name),
            cameras=cameras or None,
            collision_overlay=True,
            robot_metadata=self.robot_metadata,
            active_group=self.active_group,
        )
        overlay = result["views"]["overview_overlay"]
        self._palette.update(overlay.get("collision_palette", {}))
        record = {
            "index": index,
            "state_id": result["state_id"],
            "sim_time_s": result["sim_time"],
            "raw_path": str((frame_dir / "overview.png").relative_to(self.output_dir)),
            "collision_path": str((frame_dir / "overview_overlay.png").relative_to(self.output_dir)),
            "cameras": {name: result["views"][name]["camera"] for name in
                        ("overview", self.detail_view, self.extra_view) if name is not None},
        }
        if self.detail_view is not None:
            record["detail_path"] = str(
                (frame_dir / f"{self.detail_view}.png").relative_to(self.output_dir)
            )
        if self.extra_view is not None:
            record["extra_path"] = str(
                (frame_dir / f"{self.extra_view}.png").relative_to(self.output_dir)
            )
        self._stages[stage].append(record)
        return record

    def begin_stage(self, stage: str, model, data) -> None:
        if stage in self._stages:
            raise ValueError(f"录像阶段已存在: {stage}")
        self._stages[stage] = []
        self._next_time[stage] = self.sample_interval_s
        self._capture(stage, model, data)

    def observe(self, stage: str, model, data, sample: dict) -> None:
        if stage not in self._stages:
            raise ValueError(f"录像阶段尚未开始: {stage}")
        elapsed = float(sample["time_s"])
        if elapsed + 1e-12 < self._next_time[stage]:
            return
        self._capture(stage, model, data)
        while self._next_time[stage] <= elapsed + 1e-12:
            self._next_time[stage] += self.sample_interval_s

    def end_stage(self, stage: str, model, data) -> None:
        if stage not in self._stages:
            raise ValueError(f"录像阶段尚未开始: {stage}")
        self._capture(stage, model, data)

    def observer(self, stage: str):
        return lambda model, data, sample: self.observe(stage, model, data, sample)

    def _save_gif(self, paths, target: Path) -> None:
        frames = []
        try:
            for path in paths:
                with Image.open(self.output_dir / path) as image:
                    frames.append(image.convert("RGB").copy())
            if not frames:
                raise ValueError(f"没有可写入的录像帧: {target.name}")
            frames[0].save(
                target,
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / self.fps),
                loop=0,
                optimize=False,
            )
        finally:
            for frame in frames:
                frame.close()

    def _save_combined_mp4(self) -> dict:
        """将执行时采到的帧按阶段顺序编码；不重放或重新规划动作。"""
        frame_sources = [
            (stage, frame)
            for stage, frames in self._stages.items() for frame in frames
        ]
        if not frame_sources:
            raise ValueError("没有可导出的视频帧")
        target = self.output_dir / "skill_execution.mp4"
        video_width = self.width * (1 + int(self.detail_view is not None) + int(self.extra_view is not None))
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{video_width}x{self.height}",
            "-r", str(self.fps), "-i", "-", "-c:v", "libx264",
            "-crf", "22", "-pix_fmt", "yuv420p", str(target),
        ]
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as error:
            raise RuntimeError("保存 MP4 需要系统 ffmpeg") from error
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 17)
        except OSError:
            font = ImageFont.load_default()
        assert process.stdin is not None and process.stderr is not None
        for stage, record in frame_sources:
            with Image.open(self.output_dir / record["raw_path"]) as source:
                overview = source.convert("RGB")
            if self.detail_view is not None:
                with Image.open(self.output_dir / record["detail_path"]) as source:
                    detail = source.convert("RGB")
            else:
                detail = None
            if self.extra_view is not None:
                with Image.open(self.output_dir / record["extra_path"]) as source:
                    extra = source.convert("RGB")
            else:
                extra = None
            frame = Image.new("RGB", (video_width, self.height))
            for index, panel in enumerate(panel for panel in (overview, detail, extra) if panel is not None):
                frame.paste(panel, (self.width * index, 0))
                panel.close()
            draw = ImageDraw.Draw(frame)
            draw.rounded_rectangle((8, 8, min(video_width - 8, 270), 39),
                                   radius=4, fill=(18, 26, 37))
            draw.text((17, 13), f"Skill stage: {stage}", font=font, fill="white")
            if self.detail_view is not None:
                draw.text((self.width + 12, 13), self.detail_view,
                          font=font, fill="white", stroke_width=2, stroke_fill="black")
            if self.extra_view is not None:
                draw.text((self.width * (1 + int(self.detail_view is not None)) + 12, 13),
                          self.extra_view, font=font, fill="white", stroke_width=2,
                          stroke_fill="black")
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
            frame.close()
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg 写入 MP4 失败: {stderr}")
        return {"path": target.name, "frame_count": len(frame_sources),
                "duration_s": len(frame_sources) / self.fps,
                "size_px": [video_width, self.height],
                "source": "live_execution_step_observer"}

    def finalize(self) -> dict:
        stages = {}
        for stage, frames in self._stages.items():
            raw_name = f"{stage}.gif"
            collision_name = f"{stage}_collision.gif"
            self._save_gif([frame["raw_path"] for frame in frames], self.output_dir / raw_name)
            self._save_gif(
                [frame["collision_path"] for frame in frames],
                self.output_dir / collision_name,
            )
            stages[stage] = {
                "frame_count": len(frames),
                "gif": raw_name,
                "collision_gif": collision_name,
                "frames": frames,
            }
        result = {
            "schema_version": 1,
            "fps": self.fps,
            "sample_interval_s": self.sample_interval_s,
            "width": self.width,
            "height": self.height,
            "target_body": self.target_body,
            "detail_view": self.detail_view,
            "extra_view": self.extra_view,
            "overview_camera": self.overview_camera,
            "active_group": self.active_group,
            "collision_palette": self._palette,
            "stages": stages,
        }
        if self.combine_mp4 and stages:
            result["combined_video"] = self._save_combined_mp4()
        (self.output_dir / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return result
