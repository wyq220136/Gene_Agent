# 代码地图

按职责找到模块，再用源码搜索追踪实际调用；无需先读完所有说明。

| 要查什么 | 从这里开始 |
|---|---|
| Concept 与操作候选；只生成几何，不定义执行成功 | [knowledge](knowledge) |
| Skill 如何调用、规划与执行区别、实际终态 | [Skill 使用入口](skills/index.md) |
| 子任务组合与状态交接 | 微波炉 [subtasks.py](tasks/policies/microwave_task/subtasks.py)、[调用方式](../docs/user/microwave/mujoco_task1_physical.md)；旧视觉组合 [runner.py](tasks/runner.py) |
| 任务题目、场景初态、独立判据 | [Benchmark 入口](../benchmarks/index.md) |
| 各任务 code policy 与专用执行适配 | [policies](tasks/policies)；[Python 接口与编写范例](../docs/developer/architecture/task_authoring.md)；[显式装配](tasks/launch.py)，从 Benchmark 选择兼容策略 |
| Skill怎样调用运动规划、碰撞与求解器复用 | [Motion使用入口](motion/README.md) |
| 场景、机器人与夹爪装配；新增 embodiment | [scene](scene)、[robots](robots)、[模型/规划/控制边界](../docs/developer/architecture/native_benchmark_and_embodiment.md) |
| 原生 benchmark / MuJoCo 状态、物理执行与接触 | [simulation](simulation)、[原生接入流程](../docs/task_development/native_benchmark.md) |
| 机器人/夹爪设备、观测与真机执行边界 | [README.md](realworld/README.md) |
| 相机、分割、重建与位姿服务 | [readme.md](perception_modules/readme.md) |
| 当前微波炉任务装配 | [simulation.py](tasks/policies/microwave_task/simulation.py) |
| 碗操作目标与仿真执行 | 仿真/真机共用 [bowl.py](tasks/policies/microwave_task/operations/bowl.py)；物理执行在 [simulation_bowl](tasks/policies/microwave_task/simulation_bowl)，现场适配在 [realworld/microwave](realworld/microwave) |
| 微波炉规划、门会话与显示 | [planning](tasks/policies/microwave_task/planning)、[simulation_door](tasks/policies/microwave_task/simulation_door)、[显示](visualization/microwave) |
| 杯/勺共享 Pick 接线与任务结果范围 | [pick.py](tasks/tabletop/pick.py)、[results.py](tasks/results.py)；专用几何仍在各任务 `pick_backend.py` |
| 在线记录流与实时只读查看 | [live_record.py](feedback/live_record.py)、[live_run.py](visualization/live_run.py) |
| 记录、失败证据、回放与视频 | [feedback](feedback)、[visualization](visualization) |
| 无 MuJoCo 的 Knowledge / Skill / Task 规划 | [planning](tasks/planning)、[显示](visualization/pure_planning) |
| 本地开发工作台 | [usage.md](../docs/user/workbench/usage.md) |

跨模块约定见 [开发文档](../docs/developer/index.md)；命令见 [工具入口](../tools/index.md)。
新的实现文件不必登记；只有职责/主要入口变化时更新本页。模块已有 README 时直接复用，不再建子索引。

- cuRobo失败证据：`motion/curobo/diagnostics.py`；摘要与工件：`feedback/motion_diagnostics.py`；离线显示：`visualization/motion_diagnostics.py`、`motion_diagnostic_viser.py`；visual读取/FK：`visualization/motion_diagnostic_visual.py`，原生保存：`simulation/robotwin/visual_snapshot.py`。操作见[诊断流程](../docs/user/motion/curobo_diagnostics.md)。
