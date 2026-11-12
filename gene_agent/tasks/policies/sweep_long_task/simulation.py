"""长刷任务装配：固定 benchmark 初态后复用原清扫策略。"""
from pathlib import Path
from .policy import operation_parameters, run as run_policy
import json
from benchmarks.catalog import identity
from gene_agent.tasks.policies import identity as policy_identity
from benchmarks.sweep_long_task.scene import prepare_long_brush_scene as _prepare_benchmark_scene
from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun

def prepare_long_brush_scene(source_dir: Path, run_dir: Path, *,
                             grasp_yaw_deg: float = 0., goal_inset_extra_m: float = .002,
                             **environment) -> Path:
    """旧操作装配的兼容入口；题目先建场景，策略再附加原操作参数。"""
    import math
    if not math.isfinite(grasp_yaw_deg) or not math.isfinite(goal_inset_extra_m):
        raise ValueError("抓取朝向与推进深度必须有限")
    if not 0. <= goal_inset_extra_m <= .02:
        raise ValueError("额外入盘深度必须为 0–20 mm")
    adapter = _prepare_benchmark_scene(source_dir, run_dir, **environment)
    aligned_lane = environment.get('aligned_lane', True)
    path = adapter / 'scene.json'
    scene = json.loads(path.read_text(encoding='utf-8'))
    scene.update(operation_parameters(aligned_lane=aligned_lane, grasp_yaw_deg=grasp_yaw_deg,
                                      goal_inset_extra_m=goal_inset_extra_m))
    path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    path = adapter / 'source_mapping.json'
    mapping = json.loads(path.read_text(encoding='utf-8'))
    mapping.update(grasp_yaw_deg=grasp_yaw_deg, goal_inset_extra_m=goal_inset_extra_m)
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return adapter


def run_long_brush(source_dir: Path, output_dir: Path,
                   *, layout_shift_x_m: float = -.04,
                   aligned_lane: bool = True,
                   grasp_yaw_deg: float = 0.,
                   head_proxy_half_width_y_m: float = .0861275891,
                   layout_seed: int | None = None,
                   layout_distribution: str = "wide",
                   goal_inset_extra_m: float = .002) -> dict:
    adapter = prepare_long_brush_scene(source_dir, output_dir,
                                       layout_shift_x_m=layout_shift_x_m,
                                       aligned_lane=aligned_lane,
                                       grasp_yaw_deg=grasp_yaw_deg,
                                       head_proxy_half_width_y_m=head_proxy_half_width_y_m,
                                       layout_seed=layout_seed,
                                       layout_distribution=layout_distribution,
                                       goal_inset_extra_m=goal_inset_extra_m)
    run = SweepCubeRun(adapter, output_dir, prepared_output_dir=True)
    run.report["benchmark"] = identity("sweep_long_task")
    run.report["policy_id"] = "sweep_long_task/default"
    run.report["policy"] = policy_identity("sweep_long_task")
    run.report["task_variant"] = "sweep_long"
    run.report["source_mapping"] = json.loads((adapter / "source_mapping.json").read_text())
    return run.run(policy=run_policy)

