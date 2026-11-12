# Skill 能力入口

可复用操作集中在 `library/`；执行契约在 `contracts.py`，阶段循环、后端和规划辅助在 `runtime/`。
推荐从下表选能力，按当前环境复用已有适配。根包保留旧规划类导出，旧平铺实现已迁走。实现与证据见 [整理计划](../../docs/developer/architecture/skill_library_plan.md)。

## 能力与入口

| 能力 | 推荐调用／规划辅助 | 完成承诺与边界 |
|---|---|---|
| Pick | [pick.run](library/pick/skill.py)、`PickRequest`；[planning](library/pick/planning.py) | 接近、闭爪接触检查、按请求抬升和持物检查；grasp 前缀不算抓起 |
| Insert／Place | [transfer.insert / place](library/transfer/skill.py)；[insertion](library/transfer/planning/insertion.py)、[placement](library/transfer/planning/placement.py) | 携物到目标、释放、解除辅助、落稳及撤离；需要有效持物输入 |
| Extract | [transfer.extract / resume_extract](library/transfer/skill.py)；[extraction](library/transfer/planning/extraction.py) | 抓取、验证持物并携物退出；grasp 停止不 attach，不撤离 |
| Door | [door.run](library/door/skill.py)、[阶段／门角检查](library/door/sequence.py)；[planning](library/door/planning.py) | 单次拉／推门的阶段与后置检查；Task 决定开关次序及默认 TCP 过渡 |
| Knob | [knob.run](library/knob/skill.py)；[planning](library/knob/planning.py) | 夹持检查后旋转、释放、撤离及终态检查；目标几何不等于物理转动 |
| Hit | [hit.run](library/hit/skill.py)、[作用面几何](../knowledge/operation/hit.py) | 从已验证持物状态对齐、直线敲击、可选撤离；阶段后置检查与原 benchmark 任务评分分开 |
| Sweep | [sweep.run](library/sweep/skill.py)、[SweepContacts](library/sweep/model.py) | 按实际工具位置推进预位、下降、推扫、撤离；方块是否入盘由 Task/benchmark 判断 |

共享基础设施：[执行事实](contracts.py)、[后端协议](runtime/operation.py)、[唯一阶段循环](runtime/runner.py)、[规划事实](runtime/planning.py)、[位姿辅助](runtime/poses.py)。
规划辅助返回 SkillPlan；执行返回 OperationResult。Sweep 保留字典结果兼容既有消费者，明确 success、executed、requested_scope、failure 和 stages。两类结果都不代替 Task 评分。

## 怎么接到任务

Task 作者优先复用既有环境装配，不在 policy 内重写候选、碰撞和证据回调：

- Pick：[MuJoCo backend](../simulation/pick_backend.py)、[桌面场景](../tasks/tabletop/pick.py)、[微波炉](../tasks/policies/microwave_task/simulation_bowl/pick.py)。
- 原生 Pick / Hit：[Novel Tool 接线](../tasks/policies/novel_tool_task/execution.py)、[hammer 操作](../../docs/user/novel_tool/hammer.md)、[Hit实例迁移](../../docs/user/novel_tool/hit_transfer.md)。复用 Pick，不复制抓取候选生成；Hit 使用当前物体与目标 pose。
- 搬运：[微波炉仿真操作](../tasks/policies/microwave_task/simulation_bowl)、[现场接线](../realworld/microwave)。
- Door：[仿真会话](../tasks/policies/microwave_task/simulation_door/session.py)、[现场接线](../realworld/microwave/door.py)。会话逐阶段使用共享执行器，保留原接触检查和 checkpoint。
- Sweep：[仿真接线](../tasks/policies/sweep_cube_task/simulation.py)、[现场接线](../realworld/sweep/execution.py)。工具／接触名称来自适配配置，不藏在 Skill 中。

有了已绑定的 backend，能力层可这样调用（示意，不是脱离场景可运行的脚本）：

```python
from gene_agent.skills.library import pick, transfer, door, knob, sweep

picked = pick.run(pick_backend, pick_request, stop_after="grasp")
inserted = transfer.insert(insert_backend, stop_after="insert")
extracted = transfer.extract(extract_backend, stop_after="grasp")
opened = door.run(door_backend, stages=door_stages)
rotated = knob.run(knob_backend, target_names=selected_stage_names, stop_after="grasp")
```

每个 backend 只绑定其负责的操作；上面的不同 backend 不能随意互换。候选、目标及当前状态由 Knowledge、场景和机器人适配提供。
Pick 的输入显式使用 PickRequest；Transfer 的放置目标、Extract 的内外关键位姿、Door 的门轴与阶段目标、Knob 的旋转目标均先由对应适配绑定，再交给能力层执行。Sweep 直接接收 layout、SweepContacts 与已有运动／状态回调。这一轮保留这种已在使用的装配方式，没有增加全局工厂；新增 Task 应复用上表的场景装配，而不是仅凭旧报告构建 backend。
已有设备接线可显式选择不同检查来源（例如 MuJoCo 轨迹碰撞检查替代现场路径审批），使用能力目录内的内部 flow 参数；普通调用不跳过 required check。仿真混合旋钮许可仅在声明的后端启用，不能用来替代真机接触证据。

## 调试命令

在项目根目录的 curobo 环境，输出使用新的目录：

```bash
# Pick：停在闭爪检查，默认不继续抬升。
python tools/run/run_spoon_pick.py --stop-after grasp --output runs/tmp/skill-review/pick

# Insert：从 fresh reset 执行开门与 Pick 准备前缀，再独立放入。
python tools/run/run_microwave_task1_physical.py --subtask insert_bowl --output runs/tmp/skill-review/insert

# 只执行到释放、解除固定、落稳；不撤离、不关门。
python tools/run/run_microwave_task1_physical.py --subtask insert_bowl --stop-after release --output runs/tmp/skill-review/insert-release

# Extract／Place：同样执行必要前缀，再停在所选操作结束。
python tools/run/run_microwave_task1_physical.py --subtask extract_bowl --output runs/tmp/skill-review/extract
python tools/run/run_microwave_task1_physical.py --subtask place_bowl_on_table --output runs/tmp/skill-review/place

# Door：先停在第一个开爪阶段；省略 until-stage 则完成抓把手拉门。
python tools/run/run_experiment.py --selection handle_pull --until-stage handle_pull/00_open --output runs/tmp/skill-review/door

# Knob：从名义关闭场景运行，不补其他 Task 前缀。
python tools/run/run_microwave_task1_physical.py --subtask rotate_knob --preparation current --output runs/tmp/skill-review/knob

# Sweep：先 Pick，再对指定方块做单次推扫；可限定停止阶段。
python tools/run/run_sweep_cube.py --skill sweep --target cube --stop-after lower_to_contact --output-dir runs/tmp/skill-review/sweep

# 对同一实际记录查看与录视频，无需再次规划。
python tools/viewers/view_skill.py --run <run>
python tools/viewers/recording/record_skill_run_video.py --run <run> --output <本轮目录>/operation.mp4
```

调试范围、输入与结果的含义见 [检查说明](../../docs/user/visualization/skill_viser_inspection.md)。初态不同可能失败；失败前缀也是有效调试证据。

微波炉 `--stop-after` 仅接受单项操作：insert_bowl 的 preplace／insert／release，place_bowl_on_table 的 preplace／place／release，extract_bowl 的 grasp，以及 rotate_knob 的 open／pregrasp／approach／grasp／rotate。准备前缀完整执行；成功暂停时 status=partial、operation_success=true、task_success=null，实际现场保存到 debug_stop_state.npz。该文件用于检查，不表示已经提供跨进程恢复；旋钮仍先沿用完整目标规划，再按范围执行。

## 范围、交接和失败

- Pick 支持 pregrasp／approach／grasp／pick；同一有效 backend 的 grasp 后可 resume，先复核接触。新进程不能凭旧报告继承持物。
- Insert／Place 支持 preplace、到达目标、release、完整流程。release 前缀包含 detach、落稳和释放检查，不会停在开爪但 weld 尚未解除的中间状态。
- Extract 的 grasp 前缀包含接触检查。Knob 支持 open／pregrasp／approach／grasp／rotate，rotate 停在释放之前，不能冒称整项完成。
- Door 暂停使用既有 phase_id；目标可按新实测门角重新生成，不提前固定完整开环轨迹。Task1 多次开关组合不属于 Door Skill。
- Sweep 支持 move_behind_cube／lower_to_contact／sweep／retreat；阶段名保留旧记录兼容，工具角色可由新适配替换。
- 查看 requested_scope、executed、实际终态与失败报告引用。unknown 不当作通过；局部成功不当作完整 Skill/Task 成功，未观测到的终态不填规划值。
- 公共结果的 execution_status 区分 executed、partial、not_executed、unknown、prediction。executed 表示已有执行证据；失败后证据不足时返回 unknown，即使 executed=false 也不能据此断定机器人未动。旧成功回调继续按“已完成动作”的契约兼容；后端应优先传 executed／execution，纯检查阶段使用 OperationStage.observation_only。requires_check 表示必须检查，不代表这一步没有运动。execution_status=partial 表示失败前已有执行，与主动暂停时的顶层 status=partial 分开理解。
- 辅助 weld、近似 IK、旋钮运动学更新都保留标签；没有宣称真机、任意机器人或整个位置分布已验收。

代码整理本身不建立新 Skill 算法：现有回归覆盖碗、杯、勺、刷及微波炉；现场仅离线验证，具体配置的物理成功以本轮运行证据为准。
