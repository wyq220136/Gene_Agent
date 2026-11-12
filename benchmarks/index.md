# Benchmark 入口

这里保存题目：场景与初始分布、物理条件和独立判据。解法位于 [tasks/policies](../gene_agent/tasks/policies)，通过同名标识对应；Skill 实现保持原接口。

| Benchmark | 场景 / 判据 | Policy 与运行命令 |
|---|---|---|
| [microwave_task](microwave_task/definition.json) | [scene](microwave_task/scene.py)、[evaluation](microwave_task/evaluation.py)；现有 Task1 混合仿真 | [policy](../gene_agent/tasks/policies/microwave_task/policy.py)；`python tools/run/run_microwave_task1_physical.py --output <新目录>` |
| [sweep_cube_task](sweep_cube_task/definition.json) | [scene](sweep_cube_task/scene.py)、[evaluation](sweep_cube_task/evaluation.py)；普通刷三方块入盘 | [policy](../gene_agent/tasks/policies/sweep_cube_task/policy.py)；`python tools/run/run_sweep_cube.py --output-dir <新目录>` |
| [sweep_long_task](sweep_long_task/definition.json) | [scene](sweep_long_task/scene.py)；与普通刷共用入盘判据 | [policy](../gene_agent/tasks/policies/sweep_long_task/policy.py)；`python tools/run/run_sweep_long.py --output-dir <新目录>` |
| [cup_pour_task](cup_pour_task/definition.json) | [scene](cup_pour_task/scene.py)、[evaluation](cup_pour_task/evaluation.py)；倒球并放回 | [policy](../gene_agent/tasks/policies/cup_pour_task/policy.py)；`python tools/run/run_cup_pour_task.py --output <新目录>` |
| [spoon_scoop_task](spoon_scoop_task/definition.json) | [scene](spoon_scoop_task/scene.py)、[evaluation](spoon_scoop_task/evaluation.py)；橙/粉勺为场景变体 | [policy](../gene_agent/tasks/policies/spoon_scoop_task/policy.py)；`python tools/run/run_spoon_scoop_task.py --output <新目录>` |

在 conda `curobo` 环境运行。原 `--until-skill`、独立抓勺、清扫评测、真机和 viewer 命令保持；任务专用执行适配为 policy 旁的 `simulation.py`/`pick_backend.py`，不是新通用 Skill。
通用纯视觉规划入口仍服务任意输入场景，不为它虚构 benchmark。

## 显式选择题目与策略

```bash
python tools/run/run_benchmark.py --benchmark cup_pour_task --policy cup_pour_task/default --inspect
python tools/run/run_benchmark.py --benchmark cup_pour_task --policy cup_pour_task/default \
  --input data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/samples/seed_10 \
  --output runs/tmp/cup-review/attempt-01
```

五个题目均通过此入口调用原仿真执行器，目前每题只登记已有的 `default` 策略；不兼容的组合立即报错，不为展示选择能力新增算法。`--environment <JSON>` 提供 seed/摆位、物理辅助与碰撞模式；`--parameters <JSON>` 提供策略参数，二者不混用。默认不开启杯/勺 weld，历史辅助案例需显式 `{"simulation_weld": true}`。`robot_only` 会改变规划约束，不能当作默认 full 条件的结果。

`--inspect` 显示本题支持的环境与策略参数及默认值，不启动 GPU。已有目录禁止复用；`launch.json` 记录选择、有效参数、输入/源码指纹与异常，原任务报告仍是成败来源。只运行到 pick/scoop 等中间 Skill 时，`operation_success` 可为 true，但完整 `task_success` 为 null（未评测），视频也遵守该边界。新运行自动附加实际帧流。原各任务 CLI 继续可用。

微波炉旧复合 JSON 保持兼容，不复制第二套默认配置；场景仅接收 `environment_settings()` 投影，抓取/旋钮调参归 `operation_parameters()`。cuRobo 碰撞导出与门执行配置由仿真适配生成。新微波炉 run 的 `resolved_inputs.json` 分别记录环境、策略与适配来源；门模板内部仍有历史混合字段，暂未批量拆分，跨策略替换前须明确这一依赖。长刷场景构建只保留初态/物理/Concept，抓取姿态、清扫顺序和推进深度由 policy 给出，`simulation.py` 负责装配；原命令参数保持，applications 目录已移除。

## 独立查看场景

```bash
python tools/viewers/view_benchmark.py --benchmark microwave_task --mode viser
python tools/viewers/view_benchmark.py --benchmark cup_pour_task --mode mujoco
```

`--mode inspect` 只构建/加载并报告模型，不启动窗口或执行动作。`--input` 可替换定义中引用的场景目录/配置；微波炉与长刷的预览副本默认放 `runs/tmp/benchmark-preview/`，`--output` 可指定新目录。
杯/勺变体继续用 `tools/run/build_multi_task_scenes.py` 生成，实际规则在 [tabletop.py](tabletop.py)，不复制资产。

## 当前边界

- `definition.json` 是标识、输入引用和证据说明；资产、原始配置仍用既有路径作为单一真源，未重新复制一套数值。
- `evaluation.py` 可直接用实际状态调用，不导入 policy。微波炉桌面放置仍需要原物理落稳证据；没有把末态截图当作完整任务证明。
- 逐阶段安全/接触检查继续在执行适配中运行，任务成功仍结合完整执行前缀；未宣称所有历史判据已变成通用离线评分器。
- 杯/勺 policy 通过 `pick/pour/place` 或 `pick/scoop/pour` 操作接口组织流程；共享 Pick 接线在 `tasks/tabletop/`，专用几何独立。清扫保留原循环。操作参数与算法保持，未建立通用 Task 基类。
- Workbench 已注册五类 benchmark，默认输入、标题与版本读取本目录定义，保留原 sweep/microwave 网页 ID 兼容历史作业。五类均有实际运行及场景查看入口；纯视觉规划和网页拖拽/分布图目前仍限微波炉与普通刷。杯子/勺子可从初态运行至所选 Skill；长刷使用完整任务入口。新杯/勺运行的实际流支持失败前缀与视频导出，历史记录仍使用原生回放入口。路径白名单继续只接受 data/config/runs 资产，不开放任意 benchmark 代码执行。
- 新 run 记录 benchmark 标识/版本/定义与判据源码指纹、policy 标识/版本与源码指纹。实际场景、输入和 code_sources 继续沿用原记录。
- 固定样本回归不证明位置泛化或真机成功。当前验证与后续顺序见 [总计划](../docs/developer/architecture/incremental_refactor_plan.md)。

最新验证（2026-10-31）：五类固定输入迁移前后实跑结论保持，杯/勺/清扫的阶段目标矩阵差0；微波炉13步骤与4项最终检查通过。杯子仍失败于 upright_30/goal_ik_no_solution，其余四类成功，均保留原仿真辅助标签。818项相关回归及后续定向检查通过；五段 actual 视频已解码并核对阶段覆盖，抽查三张关键帧。具体输入、结果与限制见 [Task 编写约定与验收](../docs/developer/architecture/task_authoring.md)。不代表位置分布成功率或真机验收。


## 外部原生题目

[Novel Tool adapter](novel_tool_task/source.py) 只解析外部 `tool_use` 套件并调用原 evaluator，不复制环境、资产或判据；策略接线在 [novel_tool_task](../gene_agent/tasks/policies/novel_tool_task)。入口为 `tools/run/run_novel_tool.py`，依赖、场景查看与原评分边界见[接入流程](../docs/task_development/native_benchmark.md)。这一路径直接在 SAPIEN/RoboTwin 内运行，不进入上述五个 MuJoCo 题目的装配器。首个验证对象为 `hammer_train/model_id=0` 与原 ALOHA；完整 Pick→Hit 的验收见[hammer 操作说明](../docs/user/novel_tool/hammer.md)。后续`mallet_train`与`hammer_00003`的共享Skill/Task迁移、原评分和证据边界见[Hit实例迁移](../docs/user/novel_tool/hit_transfer.md)，不能据此推断其他实例已通过。
