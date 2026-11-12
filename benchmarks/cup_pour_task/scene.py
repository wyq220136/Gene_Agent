"""本任务的重建场景与初态生成；复用共同的桌面物理资产。"""
from pathlib import Path
import json
from benchmarks.tabletop import build_multi_task_scenes, sample_scene_variants

VARIANTS = ('pour',)

def build(capture_dir, output_dir, *, variants=VARIANTS):
    if not variants or any(v not in VARIANTS for v in variants):
        raise ValueError("此 benchmark 不支持请求的场景变体")
    return build_multi_task_scenes(capture_dir, output_dir, tasks=tuple(variants))

def load_scene(scene_dir):
    return json.loads((Path(scene_dir) / "scene.json").read_text(encoding="utf-8"))

def sample(scene_dir, *, seeds=(0, 7, 21)):
    return sample_scene_variants(scene_dir, seeds=tuple(seeds))
