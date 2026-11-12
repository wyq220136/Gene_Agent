"""题目与策略的显式装配；复用已有仿真入口，不引入新的任务算法。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import traceback

from benchmarks.catalog import load_definition, identity as benchmark_identity
from gene_agent.tasks.policies import identity as policy_identity


ROOT = Path(__file__).resolve().parents[2]
# 显式兼容表，不能从用户提供的模块路径导入或执行代码。
POLICIES = {name + '/default': name for name in (
    'microwave_task', 'sweep_cube_task', 'sweep_long_task',
    'cup_pour_task', 'spoon_scoop_task')}
ENVIRONMENT = {
    'microwave_task': {'seed': None, 'randomize': None},
    'sweep_cube_task': {'seed': 0, 'randomize': False, 'robot_gravity_compensation': False},
    'sweep_long_task': {'layout_shift_x_m': -.04, 'aligned_lane': True,
        'head_proxy_half_width_y_m': .0861275891, 'layout_seed': None,
        'layout_distribution': 'wide'},
    'cup_pour_task': {'simulation_weld': False},
    'spoon_scoop_task': {'simulation_weld': False},
}
PARAMETERS = {
    'microwave_task': {},  # 旧复合 JSON 的操作参数仍由现有适配读取，见文档边界。
    'sweep_cube_task': {'grasp_tilt_deg': -20.},
    'sweep_long_task': {'grasp_yaw_deg': 0., 'goal_inset_extra_m': .002},
    'cup_pour_task': {'until_skill': 'place'},
    'spoon_scoop_task': {'until_skill': 'pour', 'source_above_z_m': .15},
}


@dataclass(frozen=True)
class RunRequest:
    benchmark_id: str
    policy_id: str
    input_path: Path
    environment: dict
    parameters: dict


def _checked(values, defaults):
    values = dict(values or {})
    if unknown := values.keys() - defaults.keys():
        raise ValueError(f'不支持的参数：{sorted(unknown)}')
    result = {**defaults, **values}
    for key, value in result.items():
        if key in {'seed', 'layout_seed'}:
            if value is not None and (type(value) is not int or not 0 <= value < 2**31):
                raise ValueError(f'{key} 必须是 0–2147483647 整数')
        elif key == 'randomize' and value is None:
            continue
        elif isinstance(defaults[key], bool) or key == 'randomize':
            if type(value) is not bool:
                raise ValueError(f'{key} 必须是 bool')
        elif isinstance(defaults[key], float):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{key} 必须是有限数值')
    return result


def resolve_request(benchmark_id, *, policy_id=None, input_path=None,
                    environment=None, parameters=None):
    """可在无 GPU 环境检查选择；不构建场景、不导入执行后端。"""
    definition = load_definition(benchmark_id)
    policy_id = policy_id or benchmark_id + '/default'
    if POLICIES.get(policy_id) != benchmark_id:
        raise ValueError(f'Policy {policy_id} 与 benchmark {benchmark_id} 不兼容')
    env = _checked(environment, {**ENVIRONMENT[benchmark_id], 'collision_mode': 'full'})
    params = _checked(parameters, PARAMETERS[benchmark_id])
    if env['collision_mode'] not in {'full', 'robot_only'}:
        raise ValueError('collision_mode 必须是 full/robot_only')
    if 'layout_distribution' in env and env['layout_distribution'] not in {'small', 'wide', 'stress'}:
        raise ValueError('layout_distribution 必须是 small/wide/stress')
    choices = {'cup_pour_task': {'pick', 'pour', 'place'},
               'spoon_scoop_task': {'pick', 'scoop', 'pour'}}
    if benchmark_id in choices and params['until_skill'] not in choices[benchmark_id]:
        raise ValueError('不支持的 until_skill')
    source = Path(input_path) if input_path is not None else ROOT / definition['scene']['input']
    return RunRequest(benchmark_id, policy_id, source.resolve(), env, params)


def _execute(request, output):
    """环境生命周期在外层；Task 内仍按原顺序读取执行后的真实状态。"""
    from gene_agent.motion.configuration import motion_configuration
    from gene_agent.feedback.live_record import live_physics_recording
    env = dict(request.environment)
    collision = env.pop('collision_mode')
    with motion_configuration({'collision_mode': collision}), live_physics_recording(output, enabled=True):
        if request.benchmark_id == 'microwave_task':
            from .policies.microwave_task.simulation import run_microwave_task1_physical
            return run_microwave_task1_physical(output, config=request.input_path,
                collision_mode=collision, **env)
        if request.benchmark_id == 'sweep_cube_task':
            from .policies.sweep_cube_task.simulation import SweepCubeRun
            return SweepCubeRun(request.input_path, output, **env, **request.parameters).run()
        if request.benchmark_id == 'sweep_long_task':
            from .policies.sweep_long_task.simulation import run_long_brush
            return run_long_brush(request.input_path, output, **env, **request.parameters)
        if request.benchmark_id == 'cup_pour_task':
            from .policies.cup_pour_task.simulation import CupPourTaskRun
            task = CupPourTaskRun(request.input_path, output, **env)
        else:
            from .policies.spoon_scoop_task.simulation import SpoonScoopTaskRun
            task = SpoonScoopTaskRun(request.input_path, output,
                source_above_z_m=request.parameters['source_above_z_m'], **env)
        task.runtime.replay_sample(object_names=(), robot_groups=(task.group,))
        return task.run_task(until_skill=request.parameters['until_skill'])


def run_benchmark(benchmark_id, output_dir, **options):
    request = resolve_request(benchmark_id, **options)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'运行目录已存在：{output}')
    source_files = ([request.input_path] if request.input_path.is_file() else
                    [request.input_path / name for name in ('scene.json', 'scene.xml')])
    manifest = {'schema_version': 1, 'benchmark': benchmark_identity(benchmark_id),
        'policy': policy_identity(benchmark_id), 'input': str(request.input_path),
        'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in source_files if p.is_file()},
        'environment': request.environment, 'parameters': request.parameters,
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'status': 'running'}
    try:
        result = _execute(request, output)
        operation_success = result.get('goal_success', result.get('success'))
        terminal = {'cup_pour_task': 'place', 'spoon_scoop_task': 'pour'}.get(benchmark_id)
        scope = request.parameters.get('until_skill')
        full_task = terminal is None or scope == terminal
        task_success = operation_success if full_task else None
        manifest.update(status=result.get('status'),
            task_success=task_success, operation_success=operation_success,
            requested_scope='full' if full_task else scope,
            failure_stage=result.get('failure_stage', result.get('failure_subtask')))
        return {**result, 'benchmark_task_success': task_success,
                'requested_scope': manifest['requested_scope']}
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'exception',
                        error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        # 原执行器负责创建新目录；失败不得丢掉已执行前缀，亦不重写其判定。
        if output.is_dir():
            (output / 'launch.json').write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
