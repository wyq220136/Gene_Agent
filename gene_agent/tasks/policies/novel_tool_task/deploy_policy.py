"""原 evaluator 的策略回调：共享Pick/Hit执行与原任务评分分别记录。"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import traceback
from uuid import uuid4


@dataclass
class NativePolicy:
    """仅保存显式参数和episode生命周期，不缓存原环境/专家轨迹。"""
    options: dict
    episode_done: bool = False


def get_model(usr_args):
    """无引擎地构建策略；实际机器人/物体观测只能在回调开始后读取。"""
    operation=usr_args.get('gene_operation','hold')
    if operation not in {'hold','motion','pick','hit'}:raise ValueError('未知基础操作')
    allowed={'gene_operation','gene_output_root','gene_planner_python','gene_planner_pythonpath',
             'gene_candidate','gene_stop_after','gene_diagnostic_mode','gene_diagnostic_budget_s','gene_save_motion_requests'}
    options={k:v for k,v in usr_args.items() if k in allowed}
    if not options.get('gene_output_root'):
        options['gene_output_root']=str(Path(usr_args['eval_output_dir'])/'gene')
    return NativePolicy(options)


def reset_model(model):
    """原 evaluator 每episode调用，不能继承上次持物/规划状态。"""
    model.episode_done=False


def eval(TASK_ENV,model,observation):
    """接线层持有原环境；Skill仅取得受限RuntimePort，诊断独立记录。"""
    if model.episode_done:return
    from benchmarks.novel_tool.registry import task_declaration
    from tool_use.policies.motion_plan.execution import NativeEvaluation
    from gene_agent.feedback.native_run import NativeRunRecorder
    from gene_agent.simulation.robotwin.session import robot_metadata
    from gene_agent.simulation.robotwin.runtime import RobotwinRuntime
    from gene_agent.simulation.robotwin.diagnostics import RobotwinDiagnostics,native_episode_identity
    import yaml
    env=TASK_ENV;options=model.options;episode=uuid4().hex
    output=Path(options['gene_output_root'])/episode
    recorder=NativeRunRecorder(output,manifest=dict(episode_id=episode,benchmark=env.task_name,
        requested_scope=options.get('gene_operation','hold'),embodiment='aloha-agilex',
        native_identity=native_episode_identity(env),
        information_access={'object_pose':'gt_current','robot_state':'native_proprioception','expert_actions':False,'hidden_score_to_policy':False}))
    error=None
    try:
        asset=Path(env.robot.left_urdf_path).resolve().parents[1]
        args=dict(left_robot_file=str(asset),left_embodiment_config=yaml.safe_load((asset/'config.yml').read_text()))
        meta=robot_metadata(env,args)
        (output/'robot_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        declaration=task_declaration(env.task_name)
        bindings={'tool':declaration['tool_actor'],'target':declaration['target_actor']}
        if hasattr(env,'board'):bindings['board']='board'
        runtime=RobotwinRuntime(env,episode_id=episode,robot_metadata=meta,allowed_observation_fields=('objects','robots'),object_bindings=bindings,on_sample=recorder.append_state)
        diagnostic=RobotwinDiagnostics(env)
        with NativeEvaluation(env):
            start=runtime.observe(object_names=tuple(bindings),robot_groups=tuple(meta['joint_groups']))
            recorder.append_state('episode_start',start)
            operation=options.get('gene_operation','hold')
            if operation=='hold':
                result=runtime.hold(duration_s=.1,stage_id='hold',expected_state_id=start.state_id)
                recorder.append_state('hold_end',result.end)
                recorder.append_event(dict(episode_id=episode,stage_id='hold',kind='stage_outcome',state_id=result.end.state_id,success=result.success,details=dict(result.details),failure=result.failure))
            else:
                from .execution import run_operation
                from gene_agent.simulation.robotwin.scene import RobotwinSceneProvider
                execution_options={**options,'_failure_image_capture':diagnostic.capture_failure_image,'_target_body':getattr(env,bindings['target']).get_name(),'_scene_provider':RobotwinSceneProvider(env,runtime,robot_metadata=meta,output_dir=output/'collision',object_bodies={k:getattr(env,v).get_name() for k,v in bindings.items()})}
                run_operation(operation,runtime=runtime,robot=meta,start=start,recorder=recorder,options=execution_options,output=output)
        end=runtime.observe(object_names=tuple(bindings),robot_groups=tuple(meta['joint_groups']))
        recorder.append_state('episode_end',end)
        try:
            recorder.append_event(dict(episode_id=episode,kind='diagnostic',**diagnostic.capture(stage_id='final',state_id=end.state_id)))
        except Exception as exc:
            recorder.append_event(dict(episode_id=episode,stage_id='final',kind='diagnostic_unavailable',error=f'{type(exc).__name__}: {exc}'))
    except Exception as exc:
        error=exc
        recorder.append_event(dict(episode_id=episode,stage_id='exception',kind='failure',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
    finally:
        model.episode_done=True
        # 调用原check_success保存原生评分快照；完整统计仍由外部evaluator产出。
        native=output/'native_score.json'
        try:
            native.write_text(json.dumps({'success':bool(env.check_success()),'source':'native_env.check_success','task':env.task_name,'info':env.info},ensure_ascii=False,indent=2,default=str))
        except Exception as exc:
            recorder.append_event(dict(episode_id=episode,stage_id='scoring',kind='score_unavailable',error=f'{type(exc).__name__}: {exc}'))
        recorder.finalize(native_result_ref=native,error=error)
    if error is not None:print(f'Gene_Agent episode error: {type(error).__name__}: {error}',flush=True)
