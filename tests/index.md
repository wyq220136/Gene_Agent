# 回归怎么选

按改动范围运行，不必先阅读或执行所有测试。源码名称和断言给出更细的覆盖范围。

| 要查什么 | 从这里开始 |
|---|---|
| 改微波炉子任务选择与交接 | [选择/共享目标/源码身份](tasks/test_microwave_subtasks.py)、[实际流程与失败证据](applications/microwave/task/test_physical_task_shared_flow.py) |
| 改共享 Skill 或 Task | [skills](skills)、[tasks](tasks) |
| 改 benchmark 判据、初态或 policy 顺序 | [契约](tasks/test_benchmark_contracts.py)、[启动/参数边界](tasks/test_benchmark_launch.py)、[场景](tasks/test_benchmark_scene.py)、[流程](tasks/test_policy_flow.py) |
| 改 Knowledge 候选与几何 | [knowledge](knowledge) |
| 改 IK、直线、碰撞或 planner 适配 | [motion](motion) |
| 改 MuJoCo 执行、接触或辅助夹持 | [simulation](simulation) |
| 改机器人/TCP/夹爪模型 | [robots](robots) |
| 改现场设备、观测与发令前检查 | [realworld](realworld) |
| 改具体微波炉流程 | [microwave/](applications/microwave/index.md) |
| 改清扫/杯子任务 | [sweep](applications/sweep)、[cup](applications/cup) |
| 改场景、物体位姿或随机分布 | [scene](scene) |
| 改相机采集与恢复 | [runtime](perception/runtime) |
| 改标定（含独立环境前提） | [calibration/](perception/calibration/index.md) |
| 改显示、回放或视频事实 | [visualization](visualization)（含桌面失败前缀实际回放） |
| 改本地开发工作台 | [workbench](workbench) |

```bash
conda run --no-capture-output -n curobo python -m pytest -q tests/skills tests/tasks
conda run --no-capture-output -n curobo python -m pytest -q tests/motion tests/knowledge
```
Viser/HTTP 需要 localhost；物理场景集成需要本机资产。标定额外依赖和 unittest 命令见专项说明。
不再为每个测试源码登记一条说明；临时 test/check 不进入常规集合。测试收集或 --help 不代替行为验收。

Skill 能力库回归集中在 `tests/skills/`：`test_pick_library.py` 检查请求绑定和范围，`test_library_capabilities.py` 检查 Transfer／Door／Knob／Sweep 的独立停止、检查责任和执行证据，`test_operation_runner.py` 检查原始失败与终态，`test_simulation_pick_backend.py` 检查接触后夹持和阶段停止。
