"""用有限轮视觉反馈生成可编辑的功能性 Concept。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Callable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from gene_agent.knowledge.geometry import load_concepts


AgentCall = Callable[[str, list[Path], Path], Mapping]
_TEMPLATES = {"Cuboid", "Cylinder", "Ring"}


def _numbers(value: object, count: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是 {count} 个有限数") from exc
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{label} 必须是 {count} 个有限数")
    return result


def parts_to_concept(payload: Mapping, user_prompt: str, bounds: np.ndarray) -> dict:
    """Agent 的六维参数转为当前编辑器的 ``T_object_part`` 格式。"""
    if not isinstance(payload, Mapping):
        raise ValueError("Agent 输出必须是 JSON 对象")
    category = payload.get("category")
    parts = payload.get("parts")
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category 必须是非空文本")
    if not isinstance(parts, list) or not 1 <= len(parts) <= 6:
        raise ValueError("功能 part 数量须为 1–6")
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3):
        raise ValueError("mesh bounds 必须为 (2, 3)")
    extent = np.asarray(bounds[1] - bounds[0], dtype=float)
    if not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError("mesh bounds 无效")
    max_extent = float(np.max(extent))
    result_parts = []
    names: set[str] = set()
    for index, item in enumerate(parts):
        if not isinstance(item, Mapping):
            raise ValueError(f"part {index} 必须是对象")
        name = item.get("name")
        semantic = item.get("semantic")
        template = item.get("template")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"part {index} 缺少名称")
        name = name.strip()
        if name in names:
            raise ValueError(f"part 名称重复：{name}")
        names.add(name)
        if not isinstance(semantic, str) or not semantic.strip():
            raise ValueError(f"part {name} 缺少功能语义")
        if not isinstance(template, str) or template not in _TEMPLATES:
            raise ValueError(f"part {name} 的模板必须是 Cuboid/Cylinder/Ring")
        size = _numbers(item.get("size"), 3, f"part {name}.size")
        if np.any(size <= 0):
            raise ValueError(f"part {name}.size 必须是正数")
        if np.any(size > max_extent * 2.0):
            raise ValueError(f"part {name}.size 明显大于参考 mesh")
        if template == "Ring" and size[0] <= size[1]:
            raise ValueError(f"part {name} Ring 外半径须大于内半径")
        xyz = _numbers(item.get("translation"), 3, f"part {name}.translation")
        if np.any(xyz < bounds[0] - max_extent) or np.any(xyz > bounds[1] + max_extent):
            raise ValueError(f"part {name} 中心明显偏离参考 mesh")
        rpy = _numbers(item.get("rpy_deg"), 3, f"part {name}.rpy_deg")
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
        pose[:3, 3] = xyz
        parameters = {"size": size.tolist(), "pose_matrix": pose.tolist()}
        if template == "Ring":
            parameters["exist_angle"] = [360.0]
        result_parts.append({"template": template, "semantic": semantic.strip(),
                             "component": name.strip(), "parameters": parameters})
    return {"category": category.strip(), "annotation_type": "functional_primitives",
            "units": "m", "user_prompt": user_prompt,
            "conceptualization": result_parts}


def _json_write(path: Path, value: Mapping) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _mesh_bounds(mesh_path: Path) -> np.ndarray:
    loaded = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError("参考 mesh 必须是非空三角网格")
    bounds = np.asarray(loaded.bounds, dtype=float)
    extent = bounds[1] - bounds[0]
    if not np.isfinite(bounds).all() or np.any(extent <= 0) or np.max(extent) > 10:
        raise ValueError("参考 mesh 尺度或 bounds 异常；输入必须是米制 object frame")
    return bounds


def _prompt(user_prompt: str, bounds: np.ndarray, previous: dict | None = None,
            previous_ranges: dict | None = None) -> str:
    frame = ("参考 mesh 位于物体自身坐标系，单位米；bounds="
             f"{np.round(bounds, 5).tolist()}。RGB 是外观参考，坐标以 mesh 三视角为准。")
    rules = ("三视角图从左到右分别是 X-Y、Z-Y、X-Z；中间图的横轴是物体 Z，不是 X。"
             "只保留抓取、接触、施力等操作需要的功能 part。每个 part 只用一个 "
             "Cuboid、Cylinder 或 Ring；总数尽量少，最多 6 个。"
             "Cuboid.size 是 XYZ 全长；Cylinder.size 是两个截面半径和局部 Z 高度；"
             "Ring.size 是外半径、内半径、高度。translation 为 object frame 米，"
             "rpy_deg 是 xyz 欧拉角（度），决定 T_object_part。作用部位必须覆盖实际接触/施力面，"
             "不能只包围连接处。复核时逐个核对三个正交投影的中心和两端，特别核对长轴方向。"
             "不要追求装饰或细节。"
             "只输出符合 schema 的 JSON，不运行工具。")
    if previous is None:
        return f"请生成物体功能 Concept。\n用户要求：{user_prompt}\n{frame}\n{rules}"
    return (f"请快速复核现有 Concept 与参考 mesh 的叠图，只修正明显的功能位置或尺寸错误。"
            f"若已足够用于操作，原样返回同样的 parts。\n用户要求：{user_prompt}\n{frame}\n{rules}"
            f"\n当前参数：{json.dumps(previous, ensure_ascii=False)}"
            f"\n当前各 part 的物体坐标 AABB（毫米）：{json.dumps(previous_ranges, ensure_ascii=False)}")


def annotate_concept(
    rgb_path: str | Path,
    mesh_path: str | Path,
    user_prompt: str,
    output_dir: str | Path,
    *,
    agent: AgentCall,
    review_rounds: int = 1,
) -> Path:
    """一次生成、最多两次视觉复核；返回可由现有编辑器读取的 JSON。"""
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("用户要求不能为空")
    if not 0 <= review_rounds <= 2:
        raise ValueError("review_rounds 须为 0–2")
    rgb = Path(rgb_path).expanduser().resolve()
    mesh = Path(mesh_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not rgb.is_file() or not mesh.is_file():
        raise FileNotFoundError("RGB 或参考 mesh 不存在")
    if output.exists():
        raise FileExistsError(f"运行目录已存在，拒绝覆盖：{output}")
    bounds = _mesh_bounds(mesh)
    output.mkdir(parents=True)
    from tools.viewers.capture.render_concept_overlay import render

    started = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    record = {"schema_version": 1, "status": "running", "user_prompt": user_prompt,
              "rgb": str(rgb), "mesh": str(mesh),
              "rgb_sha256": hashlib.sha256(rgb.read_bytes()).hexdigest(),
              "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
              "pipeline_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "mesh_bounds_object_m": bounds.tolist(), "started_at_utc": started.isoformat(),
              "review_rounds_requested": review_rounds, "rounds": []}
    manifest = output / "annotation_run.json"
    _json_write(manifest, record)
    try:
        reference_image = output / "reference_views.png"
        render(mesh, None, reference_image)
        previous = None
        previous_ranges = None
        for index in range(review_rounds + 1):
            images = [rgb, reference_image if index == 0 else output / f"overlay_{index-1:02d}.png"]
            instruction = _prompt(user_prompt, bounds, previous, previous_ranges)
            (output / f"prompt_{index:02d}.txt").write_text(instruction, encoding="utf-8")
            t0 = time.monotonic()
            payload = agent(instruction, images, output / f"agent_{index:02d}.log")
            concept = parts_to_concept(payload, user_prompt, bounds)
            draft = output / f"concept_{index:02d}.json"
            _json_write(draft, concept)
            load_concepts(draft)
            overlay = output / f"overlay_{index:02d}.png"
            render(mesh, draft, overlay)
            from gene_agent.visualization.concept_comparison import concept_to_mesh
            previous_ranges = {
                part.part_name: np.round(concept_to_mesh(part.concept).bounds * 1000, 1).tolist()
                for part in load_concepts(draft)
            }
            record["rounds"].append({"index": index, "duration_seconds": round(time.monotonic()-t0, 3),
                                     "concept": draft.name, "overlay": overlay.name,
                                     "reason": payload.get("reason", "")})
            _json_write(manifest, record)
            previous = dict(payload)
        final = output / "concept.json"
        final.write_bytes(draft.read_bytes())
        (output / "overlay_final.png").write_bytes(overlay.read_bytes())
        record["status"] = "complete"
        record["final_concept"] = final.name
        return final
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["elapsed_seconds"] = round(time.monotonic() - started_clock, 3)
        _json_write(manifest, record)


def codex_output_schema() -> dict:
    part = {"type": "object", "additionalProperties": False, "required": [
        "name", "semantic", "template", "size", "translation", "rpy_deg"], "properties": {
        "name": {"type": "string"}, "semantic": {"type": "string"},
        "template": {"type": "string", "enum": ["Cuboid", "Cylinder", "Ring"]},
        "size": {"type": "array", "items": {"type": "number"}},
        "translation": {"type": "array", "items": {"type": "number"}},
        "rpy_deg": {"type": "array", "items": {"type": "number"}},
    }}
    return {"type": "object", "additionalProperties": False,
            "required": ["category", "reason", "parts"], "properties": {
                "category": {"type": "string"}, "reason": {"type": "string"},
                "parts": {"type": "array", "items": part},
            }}


class CodexCliAgent:
    """只读 Codex CLI 后端；所有参数由程序验证后才落盘。"""

    def __init__(self, output_dir: Path, *, model: str | None = None,
                 timeout_seconds: int = 300) -> None:
        self.output_dir = output_dir
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.schema = output_dir / "agent_schema.json"

    def __call__(self, instruction: str, images: list[Path], transcript: Path) -> Mapping:
        _json_write(self.schema, codex_output_schema())
        response = transcript.with_suffix(".json")
        command = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
                   "--sandbox", "read-only", "--color", "never", "--output-schema", str(self.schema),
                   "--output-last-message", str(response), "-C", str(self.output_dir),
                   "-c", 'model_reasoning_effort="low"']
        if self.model:
            command.extend(["--model", self.model])
        for image in images:
            command.extend(["--image", str(image)])
        command.append("-")
        try:
            completed = subprocess.run(command, input=instruction, text=True,
                                       capture_output=True, timeout=self.timeout_seconds,
                                       check=False)
        except subprocess.TimeoutExpired as exc:
            transcript.write_text(f"Codex CLI 超时：{exc}\n", encoding="utf-8")
            raise RuntimeError(f"Codex CLI 超过 {self.timeout_seconds} 秒") from exc
        transcript.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not response.is_file():
            raise RuntimeError(f"Codex CLI 失败，退出码 {completed.returncode}；见 {transcript}")
        return json.loads(response.read_text(encoding="utf-8"))
