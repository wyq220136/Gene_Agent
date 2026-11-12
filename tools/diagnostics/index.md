# 按故障选择诊断

诊断入口各有输入前提；读取历史 run 时显式给路径。不会因此授权真机运动。

| 要查什么 | 从这里开始 |
|---|---|
| 新 embodiment 的 native/cuRobo FK、碰撞和实际短轨迹 | [check_embodiment.py](check_embodiment.py) |
| 原生评测阶段、原结果与媒体引用 | [inspect_native_run.py](inspect_native_run.py) |
| 环境是否可用 | [check_environment.py](check_environment.py) |
| 实际 GPU FK/IK 与运动规划 | [check_motion.py](check_motion.py) |
| 学习新Skill怎样调用Motion | [motion_skill_example.py](motion_skill_example.py)、[接口指南](../../gene_agent/motion/README.md) |
| TCP 直线规划 | [check_motion_linear.py](check_motion_linear.py) |
| 动态障碍同步与 planner 复用 | [check_curobo_scene_update.py](check_curobo_scene_update.py) |
| 本体与安装碰撞审计 | [check_body_collision_audit.py](check_body_collision_audit.py) |
| 物体是否跟随夹爪、何时失去接触 | [check_held_object_replay.py](check_held_object_replay.py) |
| Viser/MuJoCo 状态与采样是否一致 | [check_skill_view_consistency.py](check_skill_view_consistency.py)、[check_skill_record_sync.py](check_skill_record_sync.py) |
| Pika 坐标及模型核对 | [check_pika_frames.py](check_pika_frames.py)、[check_realman_pika_motion.py](check_realman_pika_motion.py) |
| 只读检查现场快照 | [inspect_realworld_snapshot.py](inspect_realworld_snapshot.py) |

本体审计当前模型使用 `--current-scene`；旧默认值可能依赖历史场景。
模块用途、参数与输出以脚本 --help/说明为准；临时诊断放 runs/tmp，不新增长期索引。

- `inspect_native_run.py --run <run>`：失败与耗时摘要（不启动GPU）；`--performance --format json`输出计时；`--details`查看全部失败；`--recheck`只复查保存现场。
- 同案例迭代：`inspect_native_run.py --run <新episode> --compare-with <原失败episode> --stage <阶段> --hypothesis "本轮猜想" --format json`。物理/接触假设加`--require-executed`，未测到目标阶段不算验证；输入精度和失败优先顺序见[迭代流程](../../docs/task_development/simulation_iteration.md#阶段精度与失败回归)。
- `replay_motion_request.py --bundle <完整现场> --output <新目录>`：正式cuRobo离线重规划与冷/热对照，无物理执行。碰撞图仍用 `tools/viewers/view_motion_diagnostics.py`，用法与限制见[诊断流程](../../docs/user/motion/curobo_diagnostics.md)。
