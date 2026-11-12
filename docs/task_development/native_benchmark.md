# 在已有仿真 Benchmark 中开发

状态：原生接口与首个 RoboTwin/ALOHA adapter 已实现，已验证场景、双臂 FK/规划和原生短轨迹；hammer 已接通完整 Pick→Hit，当前验收与复现入口见[hammer 操作说明](../user/novel_tool/hammer.md)。详细实现与限制见[接口与 embodiment 设计](../developer/architecture/native_benchmark_and_embodiment.md)。

## 选择这条路线

输入已经是可执行的仿真题目时，直接使用它的原生环境、机器人、初态分布和评分器。不要求 RGB-D 重建或转换到 MuJoCo。只有任务来自真实场景时，才按[原 real2sim 路线](reconstructed_tabletop_scene.md)准备仿真场景。

外部 benchmark 提供原始 API，本项目 adapter 将它连接到共享 Knowledge/Skill/Motion。新机器人按[机器人与夹爪迁移](robot_and_gripper.md)单独验收。

## 恢复已有工作时先做什么

1. 先读最近实际 run 的 `launch.json`、`evaluation_index.json` 和所选 episode 的 `manifest.json` / `native_run.json`；对照原 `_metadata.json`、共享 Skill 的 `requested_scope` 与阶段结果。启动器 completed、原专家成功、pregrasp 成功是三个不同结论。
2. 从记录恢复外部仓库、实例、实际解释器、planner 依赖、候选来源与有效参数；核对文件存在及版本/哈希。不要从当前 shell 的 conda 名称猜配置，也不要把临时 worktree 当正式代码入口。正式代码始终是主项目的 `gene_agent/`、`benchmarks/`、`tools/`。
3. 原生与规划依赖可分别配置。旧记录中的 `runs/tmp/` 私有依赖和实例输入都可能被清理；缺失时按锁定版本重新准备环境/输入，先重跑最小基线，不修改原 benchmark 依赖或用旧文字声明补验收。
4. 写清本轮要推进的一个阶段、已知事实与待验证原因；临时计划/调参记录放当轮 `runs/tmp/` 或 `docs/tmp/`。精确物理 checkpoint 未实现，恢复任务依靠原 reset 与动作前缀，不把 pose JSON 当可继续积分的 checkpoint。

新episode可先读 `development_summary.md`；旧记录使用 `inspect_native_run.py --run <run>` 只读恢复。保存过完整请求时，优先离线重规划验证规划器修改，再重跑真实动作前缀。依赖来源由describe记录给出，缺所需API会在加载候选前报错。

基础接口证据为 `runs/tmp/native-benchmark/acceptance_report.json`；完整 hammer 的最新证据入口为 `runs/tmp/novel_tool/hammer/2026-11-02/acceptance_report.json`，引用原评分、固定开发/验收案例和视频。记录不存在或版本不一致时，应重新取得证据，不能照抄历史数字。长期能力边界见[接口说明](../developer/architecture/native_benchmark_and_embodiment.md)。

## 1. 确认题目身份和基线

读取实际任务/套件声明、入口、模型实例、版本、依赖和资产来源；区分训练清单与当前评测套件。先运行原生基线，确认环境能够复现，不把原专家成绩记作本项目策略成绩。

记录 resolved 配置，而不只记录命令行：机器人与控制器、观测权限、案例筛选、seed、动作与仿真时间预算、成功判据、物理辅助。用户要求严格原 bench 时，不调整摆位、摩擦、碰撞或阈值来提高成功率。

## 2. 交付原生场景检查

从原 reset 生成多个初态，实际打开全景、目标与夹爪相关视角，核对尺度、坐标、机器人安装及随机分布。提供能打开本次原生 3D 场景的实际命令。这仍是第一个交付点。

若原环境没有可用查看入口，补最小原生 viewer/截图适配；不能为显示而另造一套改变题目条件的场景。

## 3. 检查 embodiment 与控制

确认具名关节、限位、base/TCP、夹爪开合和控制频率。多组 FK、已知碰撞与净空、短轨迹和 TCP 直线通过后，才进入接触动作。

区分机器人模型、规划配置、仿真控制器三份事实；同名 TCP 不一定同坐标，关节数量一致也不表示顺序一致。非活动臂仍进入碰撞保护。

## 4. 绑定允许观测和共享 Skill

显式列出 policy 可读字段。当前实际 pose 等可按协议用 GT 替代；seed、setting 名义 pose、专家轨迹和评分器隐藏状态不能变成动作答案。

Concept 对齐实际 object frame；Knowledge 生成候选，Skill 复用阶段顺序，backend 负责真实执行和状态交接。完整 Skill 与停止到某阶段使用同一实现，不复制调试专用策略。

编写前按[Concept与动作](concept_and_actions.md#写第一版代码之前)检查新增输入，接线沿用[Motion示例](../../gene_agent/motion/README.md)。已有合格embodiment证据可复用，TCP/夹爪/资产改变才重验受影响部分。不要把Hammer闭爪参数、ALOHA轴或本机cuRobo路径复制为新任务默认值。

先单动作/Skill，再完整 Task；下一阶段使用执行后的实际状态。目标下发成功与实际到达分开记录，持物状态不得从计划 attachment 推断。

## 5. 用反馈定位失败

从 stage/state/time 定位原始 planner 返回、轨迹跟踪、接触事件、评分条件和原生视频。开发 Agent 可以查询诊断，但这些特权字段不能自动注入运行中 policy。

建议一次调试闭环：保留原失败→列出已观察事实和待验证原因→最小修改→同一开发案例复跑→固定验收案例检验。原因不足时记录 unknown；不要因失败就断言 IK、摩擦或碰撞有问题。

按[阶段精度与失败回归](simulation_iteration.md#阶段精度与失败回归)先验证原失败，明确该轮是否真正进入/执行目标阶段；规划修改先离线，接触修改再fresh reset。原生Pick支持`skill_parameters.pregrasp_approximate_ik_budget`；Hit支持`hit_parameters.approximate_ik_by_stage`的`align`/`withdraw`，单位m/rad，默认不启用，`strike`配置预算会报错。预算字段为`max_position_error_m`和`max_rotation_error_rad`。预检和正式规划同用预算，Hit悬停近似仍需满足后续敲击的侧向/法向要求。旧MuJoCo路线的字段存在不代表原生adapter已经消费，须核对本轮请求与阶段摘要。

慢规划先读 `--performance` 的筛选、setup、后端和失败尝试，区分冷/热。比较代码修复时，用[冻结请求重规划](../user/motion/curobo_diagnostics.md#计时与独立重规划)隔离仿真；不能只比不同场景的热请求，也不能把删除必要检查当作提速。

没有精确 checkpoint 时，通过原 reset 和动作前缀复现，并核对初态。局部恢复与新 reset 的全流程结果分别报告。

## 6. 正式评测和交付

使用原 evaluator 的初始化、预算与评分，记录原结果引用及版本；案例筛选和最终分母完整保留。开发与最终样本预先分开，不挑成功 seed。

交付应包含：

- 可复制的场景查看、独立 Skill 和原生评测命令；命令必须本轮实际验证。
- benchmark/机器人/代码/资产身份、允许观测和全部有效参数。
- 实际 cuRobo 返回、阶段与最终状态、失败前缀、原生评分和耗时。
- 原生截图/MP4 与阶段时间对应；记录实际检查范围，解码检查不等于画面验收。
- 固定样本结果、完整分母、仿真辅助标签和未验证范围。

## 当前首个接入目标

RoboTwin Novel Tool：`hit_tools.yaml` 的 `hammer_train`，对应 `beat_nail_hammer/model_id=0`；当前 clean 配置使用 `aloha-agilex`。

基础接口和完整任务分开验收。当前 hammer 的功能标注、参数、运行命令及实际证据见[hammer 操作说明](../user/novel_tool/hammer.md)。

## 后续实例迁移

1. 标注原生 object frame 下的抓取区、作用面与目标几何，并实际检查叠图；不使用专家点或 setting 答案。
2. 优先调整 Concept / grasp profile / Hit 参数，继续调用共享 Pick 和 Hit。新增动作语义确有需要时才扩展 Skill；展开原评分的继承/组合调用，确认是否要求敲击后仍持物撤离。共享Hit现支持显式 `withdraw_distance_m`，默认0关闭，迁移证据见[Hit实例迁移](../user/novel_tool/hit_transfer.md)。
3. 先独立检查抓取、抬升保持，再运行完整任务；每阶段使用实际终态和原始碰撞几何。
4. 预先固定开发与最终样本，保持原筛选、预算及评分；其他物体不能沿用 hammer 成功声明。

可复用接口和 Knowledge/Skill 进入正式目录；实例输入放 `data/benchmarks/novel_tool/`，调参脚本和运行记录放当轮 `runs/tmp/`，均不追踪。不为每个对象复制执行循环。

## 已实现的命令

从项目根目录运行。`<native-python>` 需要原 RoboTwin/SAPIEN 环境；`<planner-python>` 与 `<planner-path>` 指向本项目使用的 cuRobo 环境。规划依赖以 [environment/requirements.lock.txt](../../environment/requirements.lock.txt) 为准（本轮 cuRobo commit `78fd485fa82d9b9a063fb4985e371814587e666a`、warp-lang 1.12.1）。依赖版本相同时可使用同一解释器、略去 planner-path。本机可用的是 `policy` 环境，并非名为 `curobo` 的 conda 环境；规划进程另用私有 pinned cuRobo，未替换原环境的旧版。

```bash
# 只读确认套件实例，不启动仿真/GPU
python tools/run/run_novel_tool.py --external-root <tool_use> \
  --family hit --instance hammer_train --mode inspect --output runs/tmp/native/inspect-01

# 原生多初态截图；加 --viewer 交互查看单个seed（需要本机显示）
<native-python> tools/run/run_novel_tool.py --external-root <tool_use> \
  --family hit --instance hammer_train --mode scene --python <native-python> \
  --output runs/tmp/native/scene-01

# 新机器人：5组FK、普通/直线规划、实际执行和已知碰撞诊断
<native-python> tools/diagnostics/check_embodiment.py --external-root <tool_use> \
  --group left --checks all --python <native-python> --planner-python <planner-python> \
  --planner-pythonpath <planner-path> --output runs/tmp/native/embodiment-left-01

# 原evaluator + 共享Pick前缀；candidate由本项目Concept/Knowledge或人工几何给出
<native-python> tools/run/run_novel_tool.py --external-root <tool_use> \
  --family hit --instance hammer_train --mode evaluate --operation pick \
  --candidate <candidate.json> --stop-after pregrasp --python <native-python> \
  --planner-python <planner-python> --planner-pythonpath <planner-path> \
  --output runs/tmp/native/pick-01

python tools/diagnostics/inspect_native_run.py --run runs/tmp/native/pick-01 --stage pregrasp
```

候选 JSON 最小字段为 `group`、`T_object_tcp`（4×4、米、原生 actor frame）和来源 `source`；可选 `skill_parameters` 使用共享 `GraspSettings` 字段。ALOHA approach 沿 TCP +X，开合沿 Y；不能套用旧 Pika 的局部 Z 接近假设。`contact_width_m` 只用于几何夹持检查，缺失时检查保持 unknown。候选不应从原专家抓取点/functional point 文件导出。开发候选默认放 `data/benchmarks/novel_tool/` 或当轮 run，均不追踪。

默认 `--operation hold` 仅检查原生保持；`motion` 检查 1cm TCP 直线。`pick` 的 `pregrasp/approach/grasp/pick` 停止范围共用原 Pick 实现。完整 Pick 已将工具的全部原碰撞 shape 转为独立持物碰撞 link，保留机器人自碰撞与场景检查；只允许明确阶段/对象对的接触。`--operation hit` 总是先执行完整 Pick，再对齐与直线敲击，不受 Pick 默认停止前缀影响。

原 evaluator 的专家筛选日志与本项目策略分开：日志内 expert Hit 成功不算本项目成功。启动器 `completed` 只表示原进程正常结束；完整任务以原 `_result.txt`/`_metadata.json` 与 `native_score.json` 为准。仅运行前缀时 Hit 评分为 0/1 是完整任务失败，不能改成未评测或成功。`evaluation_index.json` 只索引真源，不另造评分器。


基础接口验收保留在原报告，完整任务的最新结果见[hammer 操作说明](../user/novel_tool/hammer.md)。原 evaluator 成功时可提前终止 strike；此时动作记录保留 `native_terminated`，任务成功独立读取原评分，不能改写为轨迹终点已到达。

cuRobo失败统一使用[诊断流程](../user/motion/curobo_diagnostics.md)：原生评测可显式开启有预算诊断，先读摘要，再看同一次候选/碰撞现场；不为诊断转换物理引擎。
