# Task 编写约定与渐进迁移方案

2026-10-31 修订。**A/B/C 与 applications 退场已完成，五类固定样本仿真对照通过**。未运行真机。
目标：让人和 Agent 能快速看懂任务、独立调试一个操作、根据实际状态组合已有 Skill。
现有微波炉子任务能力见 [已实施方案](task_subtask_review.md)；本文件维护任务编写约定和迁移验收状态。

## 本轮补充：迁移必须服务的四个目标

1. **原有效果不退化。** 同配置、seed、资产、规划模式与判据比较迁移前后；原来成功的基线必须重新成功，原来失败的样本不得用放宽判据改成成功。出现差异先定位，不能以“只是整理代码”交付。保留原命令、参数、实际回放能力；耗时异常增长也要解释。单次对照不证明全部分布成功率不变。
2. **最终移除 applications。** 已按职责迁移有效实现、更新活动调用并删除旧别名；不恢复同义转发目录。历史 run 保持只读。
3. **调试以实际失败为入口。** 摘要直接说明运行范围、操作/阶段、失败分类、实际终态、原始报告和视频位置。前置检查失败、规划失败、部分执行失败与最终条件失败分别可测；完整和局部调用同一实现。
4. **共享仿真/真机决策逻辑。** 共享的不仅是步骤名称，还包括操作目标生成、阶段次序、前后置条件的语义和恢复决策；后端负责取得观测、规划/执行接线、具体证据采集和资源管理。现场调整这些共享逻辑后先在 MuJoCo 用同一版本验证，再进入真机验收。

迁移前，真机 Insert 从 `applications/microwave/physical_insert.py` 导入纯 keypose 变换。A 批已将这类目标计算提取到共同操作模块。当前仿真和真机共用 `run_microwave_task1_flow`、已有 Skill 阶段循环及碗目标规则，但仍分别管理执行分派和具体证据；不能称为已共用完整执行器。

期望调用关系：`共同 policy → 共同操作决策/阶段目标 → 仿真或真机适配 → 现有 Skill/motion`。按实际重合点提取普通函数，不预建统一后端框架。GT 与实测观测按共同语义输入；允许来源、误差、时效和可用证据不同，不能伪造真机没有的接触/weld 信息。现有第二次开门前是否回默认 TCP 的后端差异先显式保留，不在结构迁移中强行对齐。

### applications 的去向与删除条件

| 原内容 | 迁移归属（按调用职责确认，不按文件名机械搬运） |
|---|---|
| Task 分派、专用操作组合及仿真会话 | `tasks/policies/<name>_task/`，共同操作与仿真接线清楚区分 |
| 可复用动作阶段/执行 | `skills/`；任务特例在命名子目录，不与公共执行器平摊 |
| 纯 Concept/目标几何算法 | `knowledge/`；现场观测和 MuJoCo 不进入该层；任务策略偏移仍归任务 |
| 场景初态、最终判据 | 对应 `benchmarks/<name>_task/` |
| 场景状态/物理辅助、机器人特定接触处理 | 按实际职责归 `simulation/`、`robots/` 或任务仿真适配，不为去目录而泛化 |
| 回放、显示、编辑与诊断 | `visualization/` 对应专题；命令放现有 `tools/` 专题 |
| physical_task1、pour_cup/task_runner 等旧 import 别名 | 更新仍在使用的调用者后删除，不另建同义转发目录 |

迁移清单须覆盖 tasks、realworld、skills、tools、tests、来源哈希和运行时动态路径。旧 run 保持只读，必要时在读取器处理旧来源路径；不能重写历史证据。每批跑入口与回放测试，最后确认活动代码无旧依赖、没有唯一实现遗留，再移除 applications。历史文档提及旧路径可以保留并标注历史。

## 从参考项目取什么

以下来自本地源码/开发说明检查，未运行参考项目，也不引用其论文成绩作为本项目证据。

| 来源与实际机制 | 本项目采用的经验 | 不照搬的部分 |
|---|---|---|
| RoboRSI 的 `visual_pick_place`：抓取后检查 holding、visual_verified、identity_verified，再定位目标并放置；返回 failed_phase 和 trace | 操作之间交接可核验的事实；失败时保留当前持物状态与具体阶段 | 不强制所有操作依赖视觉模型；当前 pipeline 仍可使用 GT pose |
| RoboRSI `LHExecutor`：整段长任务只打开一次环境，原子操作之间状态连续，尝试有上限 | 不因进入下一个子任务而 reset；确有恢复需求才增加有界重试 | 不引入 Planner/Engineer/Reviewer 在线多角色执行框架 |
| RoboRSI compound gate：组合返回成功之后还要独立 simulator verdict | 操作完成与 benchmark 成功分开；局部成功不代替完整连续验收 | 不复制其固定 seed 21/22、两次成功即通过的特定规则 |
| RAX autonomous-control：普通 Python `main(ctx)`、显式参数、操作前重新定位、对象坐标系动作、外层建立 Context | 短 policy＋已有操作；资源外部管理；物体位置变化由观测与参数处理 | 不引入 MCP 服务、字符串工具总线、自动 fallback 或任意点恢复 |
| RAX 经验库：前置条件、适用范围、失败模式和运行证据一起保存 | 留下下一次开发真正需要的适用条件和失败经验 | 不为每个函数创建 SKILL.md，也不把实验数据塞进源码目录 |

源码线索：
- [RoboRSI 组合实现](../../../third_partys/RoboRSI/roborsi/embodied/skills/atomic/libero_pick_place/visual_pick_place/policy.py)、[组合测试](../../../third_partys/RoboRSI/tests/sim/libero/test_short_pick_place_compound.py)、[长任务执行器](../../../third_partys/RoboRSI/roborsi/agents/long_horizon/lh_executor.py)、[独立评测 gate](../../../third_partys/RoboRSI/roborsi/agents/evolution/compound_protocol.py)。本地 HEAD `90c8569`。
- [RAX 开发说明](../../../third_partys/rax/skills/autonomous-control/SKILL.md)、[经验库格式](../../../third_partys/rax/skills/continual-learning/references/library-format.md)。main 是索引；额外通过 `git show` 检查本地 `origin/libero-pro/native`（`1767412`）的 `auto_research_agent/api/runtime.py` 与 `services/controller/formal_worker.py`，确认 Context 与独立 worker 的边界；未切分支、未拉取代码。

## 采用的方案与边界

三个选择：只整理目录不能消除隐式状态；一次统一所有 Task 框架风险过大；**采用少量共同语义＋任务内普通函数，逐个迁移**。
不建立通用 Task/Subtask 基类、插件注册系统、任务图、第三套执行循环或新的 checkpoint 系统。
结构简单且能独立调试的任务可以维持原样；文件长度本身不是拆分依据。

## 新任务最小形态

```text
benchmarks/<name>_task/              # 题目：场景、初态分布、独立成功条件
gene_agent/tasks/policies/<name>_task/
    policy.py                      # 根据观测决定下一步，调用操作，有限恢复
    simulation.py                  # 仿真装配、生命周期、执行与证据适配
    subtasks.py                    # 可选：已有操作的调试分组
    operations/                    # 可选：确有独立修改需求时才提取操作
gene_agent/realworld/<任务专题>/     # 需要真机时再接现场观测和执行
```

这是职责示意，不要求创建空文件。已有 Knowledge 生成几何，Skill 实现动作阶段，motion 负责规划；Task 不复制它们。
CLI 使用 `tools/run/` 现有入口模式，viewer 消费执行记录，不另写任务流程。任务说明进入现有文档索引，不逐目录铺 README/index。

### 从哪里调用 policy

这些是已准备后端的同步 Python 接口，不能代替外层建场景/接设备。新任务优先参照杯/勺的短函数；已有恢复决策参照清扫；复杂操作分组参照微波炉。无需照搬所有文件。

| Task | policy 入口 | 范围与结果 |
|---|---|---|
| 微波炉 | `run_microwave_task1_flow(backend, policy=..., selection="all", preparation="prefix")` | 完整或已有子任务；返回 `OperationResult`，评分/报告引用由 Session 或现场入口给出 |
| 倒水 | `cup_pour_task.policy.run(operations, until_skill="place")` | 从 Pick 执行至 pick/pour/place；返回运行摘要；所需能力见同文件 `CupOperations` |
| 舀取 | `spoon_scoop_task.policy.run(operations, until_skill="pour")` | 从 Pick 执行至 pick/scoop/pour；返回运行摘要；所需能力见 `SpoonOperations` |
| 普通刷、长刷 | 各自 `policy.run(operations, alignment_lateral_trigger_m=..., target_order_policy=...)` | 完整清扫决策，返回操作完成 bool；实际末态、失败和完整报告由 `SweepCubeRun.run()` 外层保存 |

清扫 operations 提供 `pick/observe/begin_sweeping/sweep/goals/finish_sweeping`，以及失败证据和进度记录；policy 不读 MuJoCo、scene 或原始报告。观测的 pose 使用世界坐标米制矩阵，实际 state_id 用于区分预检和执行。`record_progress(**values)` 只保存决策证据，不发起动作；其中列表是本次同步调用内持续追加的进度，适配须保留引用，在调用结束（包括失败/异常）后保存最终内容，不能收到空列表时只序列化一次。当前 `SweepCubeRun` 使用 report.update 并在 finally 落盘。

杯/勺 `until_skill` 是“执行至”，不是任意中段恢复。清扫目前没有新增独立方块恢复 CLI；对单个 Skill 的调试仍使用现有入口。只有微波炉具备 prefix/current 子任务模式，不能把不同任务的局部能力写成全部相同。
外部先看 `operation_success/task_success/actual_final/failure`；局部完成的 task_success 为 null。policy 返回 bool 的清扫任务应通过仿真运行入口获取这些摘要，不能把 bool 当作 benchmark 最终评分。

### 共同约定：约束语义，不强制类名和字段布局

1. **Policy 可读。** 主体能看出对象、操作次序、分支和失败后的处理。场景加载、CUDA solver、控制增益、报告落盘放适配层。允许普通函数、现有回调或已有 runner，不要求为了统一外形重写流程。
2. **每个操作说明交接。** 列清输入对象/目标/参数、可观测前置条件、完成条件、失败后状态。按需传前序证据；纯 pose 不足以证明持物。设备资源留在适配层，不塞进 policy 的“万能 ctx”。
3. **按实际状态推进。** 动作前核对当前观测，动作后检查实际结果，必要时再生成下一阶段目标。长任务可以循环和分支，不预生成所有阶段的开环轨迹。已在持物时失败，不能默认重新抓取或松爪。
4. **观测与评分分开。** Policy 使用真机可获得的语义信息，仿真可用 GT 提供 pose 等；模型指纹、辅助 weld、精确颗粒身份属于仿真校验/评测。包装成一个字典并不能消除特权依赖。真机无法提供某项观测时必须标明缺口。
5. **结果能定位问题。** 复用 `OperationResult`、`OperationState`、`CheckResult`，旧报告用薄适配保留。检查支持 pass/fail/unknown；区分未执行、已执行、部分执行与仅规划，失败也保存实际终态。未知不能自动通过。
6. **参数有来源。** 对象局部方向、偏移和容差带 frame/单位；机器人特例明确标注。参数保留单一配置来源与已验证范围，不复制到多份文档或固定世界坐标序列。
7. **调试不变造事实。** 完整任务与局部入口调用同一操作。简单任务支持已有 until_skill 即可；只有复杂任务才增加选择子任务。fresh reset 前缀、当前已准备状态、恢复状态要区分；当前不承诺跨进程恢复。
8. **失败可复现。** 摘要指向具体操作/阶段、原始 planner/执行失败、实际终态和已有视频。记录近似 IK 与仿真辅助。恢复需重新观测，设次数上限；禁止无条件重试、静默 reset 或覆盖失败记录。

运行摘要逐步对齐这些语义，不新增一套内部结果树：

| 信息 | 建议复用/补充字段 | 含义 |
|---|---|---|
| 本轮运行范围 | requested_scope | 完整任务、指定操作或执行到某阶段 |
| 操作完成 | operation_success | 所选流程实际执行且操作后置检查通过；纯规划不算 |
| 完整任务评分 | task_success | 完整运行后的 benchmark 结论；局部或未评估为 null，证据不足另记 unknown |
| 状态与错误 | actual_final / failure / report_ref | 实际终态、失败阶段及原始证据引用；可复用现有等价字段 |
| 证据与耗时 | evidence_mode / timing | 仿真辅助、规划/实际执行来源；未采集的耗时写 null，不猜测 |

已有 `success` 字段先保留兼容并写清语义，不能直接全库重命名。完整任务执行失败记录 task_success=false；操作成功但最终条件失败也必须为 false。
独立评分是目标；目前微波炉最终碗状态仍依赖先前放置报告，此限制必须保留说明。

### 直观 policy 与 Task 对外接口

用户确认：任务代码应直接体现操作顺序、条件分支和失败处理；新 Agent 应知道如何编写和放置 policy；仿真与真机尽量执行同一份决策代码，并做好 Task 对外接口。本节说明接口约定，具体入口见上表；执行与验收进度见下节。

采用明确的 Python 操作调用，先以微波炉验证。继续使用现有步骤表作为唯一决策入口，会保留阅读跳转负担；另建通用任务图/插件框架不符合本轮规模。既有步骤标识、参数和报告仍供兼容与验收使用，不能再与显式 policy 各维护一套会分歧的动作顺序。

**两个接口边界：**

| 边界 | 约定 |
|---|---|
| CLI、Workbench、其他 policy → Task | 每个任务有明确的 Python 运行入口；输入为已准备的操作适配、任务参数和完整/子任务选择。后端创建、设备授权、场景 reset 与输出目录由外部运行层负责 |
| Task policy → 操作 | 通过 `open_door()`、`load_bowl()` 等具名操作表达顺序；每次操作返回阶段结果和执行后的实际状态/证据。操作内部继续复用已有 Skill 与 motion |

对外保留 `run_microwave_task1_flow(backend, *, policy=MicrowaveTaskPolicy(), selection="all", preparation="prefix") -> OperationResult`，不再另造同义运行入口。操作参数继续由现有适配层接收，避免新增一个与配置重复的 parameters 字典。完整记录仍由现有 Session/现场入口返回；流程结果与带 benchmark 评分的运行摘要是两个明确边界。

**输入约束：**

- Task 参数是操作目标、策略配置及已有预算，不能携带修改 benchmark 成功条件的权限；坐标、单位和参数来源明确。
- 完整任务与子任务都调用同一操作实现。调用 Task 不隐式 reset；独立运行先检查当前真实前置条件。
- 仿真的 `prefix` 准备由外层明确选择和记录；真机沿用当前状态与原有执行授权，不自动补跑仿真前缀。
- 仅规划、仿真实际执行和真机执行的来源显式保留；未知参数、未知操作及后端能力不满足应在动作前拒绝。

**返回约定：**

| 信息 | 用途 |
|---|---|
| 运行标识、范围、执行来源与状态 | 知道调用了哪项任务、是否实际执行、完整还是局部 |
| 操作完成、完整题目得分 | `operation_success` 与 `task_success` 分开；未运行完整题目或未评测时后者为 null。Benchmark 负责评分，policy 不自行伪造评分结论 |
| 已执行步骤与实际终态 | 完成多少、机器人与持物状态是否已知、证据是否可供下一操作使用；失败也保留可取得的实际状态 |
| 失败操作、阶段、类别与原始证据引用 | 外部可以定位规划/执行/后置检查失败，不必逐个猜测私有报告结构；未知原因如实保留 |
| 运行记录、视频与可获取的耗时引用 | 大轨迹、图片和 traceback 留在原记录中，通过稳定引用取得，不全塞进返回对象 |

复用现有 `OperationResult`、Task 摘要和结果工具补齐语义，避免同一事实产生多份不同结论。内部异常保存 traceback 与实际终态；配置错误明确报告。不能用宽泛的异常捕获把部分执行误写成“未执行”。

**决策与迁移验收：** policy 中明确写成功后继续、失败后停止或经过验证的恢复分支。第一轮保持原失败停止行为，不凭空新增回退。后续恢复必须根据执行后状态决定，记录尝试次数和独立调用标识；不能把已经持物的部分失败当成初始状态重跑。

微波炉完整/局部入口、仿真与现场适配接同一 policy；必要的默认 TCP 过渡和已知后端差异显式保留。先用替身适配核对两端调用顺序、参数传递、前置拒绝、部分失败与返回契约，再做同输入 fresh-reset 仿真对照。真机硬件验收另行授权。最后在新任务入口说明公共 API、所需操作能力及一个可参考的现有任务；不将仅有函数名一致称作双后端已打通。

### 微波炉可读 policy：实施与验收清单

当前进度：五类任务的代码优化与独立审查已完成，相关回归 585 项通过；固定样本实际对照结论见下方本轮验收。

**目标：** 打开 policy 就能看到完整操作次序；沿用原完整/局部入口和失败证据，保持当前行为。
**方案：** 使用 Python 具名操作描述和现有同步 `run_operation`。不新增依赖、通用 Task 基类或第二套执行检查。
**依据：** 上述“下一轮”约定；执行时使用 executing-plans 分批推进。这是全任务优化的第一批；后续任务范围见下节，不将微波炉的复杂度强加给简单任务。

**文件落点：** `microwave_task/policy.py` 放可读顺序及运行入口；`subtasks.py` 保留选择与前缀规则；`simulation.py`、`realworld/microwave/task.py` 仅按需调整接线和结果引用。本轮未新增生产源码文件。文档继续维护本文件与已有微波炉使用说明。

#### 1. 固定接口与兼容基线

- [x] 在 `tests/tasks/test_microwave_subtasks.py` 和 `tests/applications/microwave/task/test_physical_task_shared_flow.py` 固定现有完整步骤、各 selection 的 prefix/current、禁用旋钮及非法输入行为。
- [x] 固定 `OperationResult` 的成功/失败、实际终态、已执行前缀语义；用替身测试 Pick 失败、Insert 部分失败、后置条件 unknown 都会停止，后续关门不会执行。
- [x] 保存同输入 fresh-reset 仿真基线；沿用原动作参数、碰撞、辅助夹持、规划模式和判据，不重写历史 run。

#### 2. 让顺序只在一处定义

- [x] 在 `policy.py` 增加微波炉专用 `iter_task_steps(policy: MicrowaveTaskPolicy)`，通过具名的门、过渡、load/unload 描述函数按顺序 yield `MicrowaveTaskStep`。load 展开 Pick+Insert，unload 展开 Extract+Place；不生成整段轨迹。
- [x] 主体直接展示：首次开门 → 开门后过渡 → load → 关门前过渡 → 首次关门 → 可选旋钮 → 可选再次开门前过渡 → 再次开门 → 开门后过渡 → unload → 关门前过渡 → 最后关门。描述函数只是生成操作请求，不能伪称已经执行。
- [x] 删除 `_TASK1_STEPS` 作为第二份顺序定义；保留 `microwave_task_steps(policy)` 作为该迭代器的兼容物化接口，供原子任务选择、显示和适配查阅。步骤 ID、success_key、报告文件名保持。
- [x] `run_microwave_task1_flow` 仍交给现有 `run_operation` 执行、检查并在失败时返回；在入口附近明确该停止规则。每个操作内部仍基于最新实际状态生成目标。当前只有配置条件分支，不把步骤枚举包装成已经支持任意反馈恢复的 policy 引擎。
- [x] 复跑步骤1的测试及 `tests/applications/microwave/task/test_task1_visual_parity.py`，确认完整/局部共用同一顺序来源。

#### 3. 接口可用、失败可定位

- [x] 在原入口 docstring 和本文件说明 backend 所需的 observe/execute、观测来源、参数归属、selection/preparation 和返回值；直接引用现有 OperationBackend 契约，不另造相同协议。
- [x] 检查仿真与现场运行摘要：范围、operation_success、task_success、actual_final、失败阶段及原始报告引用。仅补实际缺失的信息；不要求两端私有报告完全同构，不新建结果树。
- [x] 运行 `tests/applications/microwave/realworld/task/test_realworld_task1.py`、`test_realworld_task_transition.py`、`test_realworld_live_observe.py` 的离线测试，验证两端共用 policy，保留仿真13步/现场默认14步的已有差异。不得连接硬件执行。

#### 4. 实际验收后停止扩展

- [x] 使用同配置 fresh reset 重跑完整微波炉，并用原 `--subtask load_bowl --preparation prefix` 核对局部入口；子任务内部失败定位由替身回归覆盖，不额外制造昂贵的物理失败试验。
- [x] 比较参数、步骤顺序、真实 cuRobo 返回、最终四项判据和耗时；导出完整 actual 视频供用户查看。局部成功不得写成完整成功，保留混合仿真标签。
- [x] 更新已有使用说明中的 Python 入口、局部命令和失败查看位置。不借此改其他 Task、Skill、最终评分、checkpoint、自动恢复或 Workbench UI。

定向测试统一使用 `PYTHONUTF8=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/curobo/bin/python -m pytest <上述测试文件> -q`。
实际命令沿用 `python tools/run/run_microwave_task1_physical.py --output runs/tmp/task-policy/<本轮>/full --video`，局部另用新目录并追加上述 subtask 参数。

**审查重点：** 无双份顺序、无静默 reset、无失败后继续动作、无局部冒充完整成功、无丢失默认 TCP 过渡。未知状态与原始失败原因仍可追溯。
**完成界限：** 这轮改善可读性和可调用性，不宣称真机已验收或全部任务支持反馈恢复；未来确有恢复需求时，用普通 Python 分支与实际结果扩展，再决定是否需要抽取公共能力。

### 所有现有任务：实施与验收清单

用户补充：每个任务都应按原方案优化。范围覆盖五类任务，按批交付；“不过度设计”限制实现复杂度，不缩减为只处理微波炉。上一节的其他任务不改指第一批，不是整个计划的结束条件。

共同验收：policy 能看到顺序、分支和失败处理；公开入口说明输入、运行范围和结果；完整与局部复用操作；设备资源与报告管理在适配层；原功能和证据保持。接口语义一致即可，不强制所有任务采用同一种类或返回对象。

| 批次与任务 | 当前具体问题 | 最小修改与边界 |
|---|---|---|
| 1 微波炉 | 流程藏在步骤表及分派中 | 按上一节实施，保留已有子任务与默认 TCP 过渡 |
| 2 倒水 | `run(operations, until_skill)` 已清楚表达 Pick→Pour→Place，但 operations 能力和结果交接需查适配实现 | 保留短 policy；明确 pick/pour/place 的输入状态、成功含义、失败证据与 finish 返回。检查异常不会丢失实际终态；保留 until_skill，不拆新的子任务体系 |
| 2 舀取 | Pick→Scoop→Pour 已清晰，颗粒身份/持物证据等依赖在仿真侧 | 同样补齐公共操作契约与失败定位，标清仿真特有证据；只提取实际存在的纯决策依赖，不为未有的真机后端创建空实现 |
| 3 普通刷 | policy 混入 self.scene、self.report、私有执行方法及渲染；恢复决策与报告字段耦合 | 保留观察→排序→推扫/归拢→有限恢复→释放的普通 Python 流程；执行、记录与显示通过小范围具名操作适配。策略读取明确参数与操作结果，不靠修改报告清除失败；保留预检失败换顺序、实际执行失败停止的原边界 |
| 4 长刷 | policy.py 目前仅有参数，实际流程需追到普通刷执行链 | 明确任务入口调用共享清扫策略及长刷参数差异；复用批次3的操作接口，不复制清扫循环。保留现有排序、抓取方向、阈值与命令 |

**涉及文件：** 各任务现有 `policy.py`、`simulation.py`；普通刷按需涉及已有 `execution/{curobo,grip,selection}.py`，先核对实际调用再决定是否调整。现场已有清扫入口须核对调用兼容，但不预先承诺与仿真全流程相同。只在已有两处真实复用时才提取到 `tasks/` 公共模块，不新增五套包装器。

- [x] 批次2：核对杯/勺公开 operations 方法和返回契约，补缺失的局部终止、部分执行异常与实际终态测试；完善现有文档中的调用例子。已经清楚的流程不为形式重写。
- [x] 批次3：先固定普通刷的排序、归拢触发、纯预检换序、无进展停止、一次恢复与释放行为，再逐项移出记录/资源依赖。恢复判断必须有原始结果证据，不能仅用布尔成功替代。
- [x] 批次4：长刷接同一清扫策略，固定 aligned_lane、target_order_policy、抓取参数及最终判据；核对普通刷改动没有改变长刷行为。
- [x] 每批使用现有 tests/tasks 及对应任务测试增加有意义的契约回归；共享清扫修改须同时验证普通刷与长刷。不借目录整理改测试组织。
- [x] 每批做同配置 fresh-reset 对照，保留当前倒水已知失败样本的失败原因；成功样本重新成功才交付。分别提供命令、实际视频、失败定位示例与未验证范围，不把单次样本推广为成功率保证。
- [x] 五类任务完成后更新本文件与现有任务入口导航，给新 Agent 一个简单任务范例（杯/勺）和一个复杂任务范例（微波炉/清扫），无需新增独立规范体系。

实施顺序为微波炉→杯/勺→普通刷→长刷。共用代码变化会影响下一批时同步做回归；不等全部完成才验证。各任务均检查仿真/真机边界，但本轮不新实现缺失的真机感知和设备后端，也不执行硬件运动。

### 本轮 policy 优化验收（2026-10-31）

五类任务均已处理：微波炉顺序只有具名生成器一处定义；杯/勺保留短流程，增加操作 Protocol 与调用说明；清扫改为独立策略＋公开操作适配；长刷实际入口调用共享清扫策略。未新增通用 Task 框架、自动恢复、生产源码文件或硬件后端。已有真机适配只做离线回归，不能由接口可替换推断已完成真机迁移。

| 固定样本 | 结果 | 对照 |
|---|---|---|
| 微波炉名义摆位 | 完整成功；13步、4项判据通过 | 本轮 fresh baseline→after；输入投影、步骤与判据一致；墙钟166.79→156.20s，不作性能结论 |
| 微波炉 load_bowl/prefix | 局部成功 | 保留开门和默认 TCP 前缀；task_success=null |
| 倒水 seed10、显式 weld | 保留 upright_30 / goal_ik_no_solution 失败 | 本轮 fresh run 与上一轮同输入记录相比，顺序与目标矩阵一致；policy 执行语句与本轮起始快照 AST 一致 |
| 橙勺 seed21、显式 weld | 成功 | 同上；未为仅类型/文档修改重复执行同一场景 |
| 普通刷 randomize seed36 | 成功 | 本轮 fresh before→after：输入、参数、阶段顺序一致，目标矩阵最大差0 |
| 长刷名义通道 | 成功 | 同上；共用清扫策略且保留长刷参数 |

相关回归命令为 `PYTHONUTF8=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/curobo/bin/python -m pytest tests/skills tests/tasks tests/applications tests/visualization/test_video_export.py -q`，585项通过；独立只读审查未发现阻断问题。初次受沙箱限制的仿真因 No CUDA GPUs 停止，随后本机 GPU 对照正常；没有 CPU fallback。实际返回核对为 cuRobo＋CUDA Graph，四类桌面任务分别保留14/12/26/10份原始 planner 返回。

本轮临时证据位于 `runs/tmp/task-policy/20261031/`：`comparison.json`、`cup-spoon-comparison.json`、`sweep-comparison.json` 和 `video_checks.json`。五段 actual 视频在 `{after-microwave,before-cup,before-spoon,after-sweep,after-long}/video/task.mp4`；before-cup/spoon 命名来自批次脚本，但运行时已包含它们最终的接口修改。视频只核对全段可解码、实际阶段覆盖和结果标签，未逐帧做画面验收；继续保留仿真辅助夹持及微波炉旋钮运动学更新标签。

```bash
python tools/viewers/view_task.py --task microwave_task1 \
  --run runs/tmp/task-policy/20261031/after-microwave --mode actual
```

这轮固定样本通过不代表位置分布成功率或真机验收。新任务从上方“从哪里调用 policy”选择范例，故障恢复只复用已有且验证过的分支。

### 一次新任务开发的完成条件

- 在现有文档中说明：完整命令、可调范围、如何局部调试、前置状态、失败记录与回放入口。
- 首先在同一 Runtime 跑通实际状态交接；再做 fresh reset 完整运行，人工介入的调试结果单独标记。
- 使用固定开发/回归样本核对功能；位置泛化在声明的分布上另行验证，不用一次成功背书。
- 开发笔记只提炼验证过的适用范围、典型失败和排查方法；临时过程留在 runs/tmp，不新造长期日志体系。

## 旧任务迁移计划

执行建议：使用 executing-plans 按批次实施，每批先核对当前源码和其他 Agent 改动。保持原参数、碰撞策略、cuRobo 后端/规划器复用、报告兼容与原成功判据。涉及行为或评分修改必须单独验证。

### A：微波炉碗操作试点（优先执行）

- [x] 复现当前 load/unload 及 fresh reset 完整任务，保存配置、源版本、阶段顺序和实际视频作为对照。
- [x] 核对两端调用链；共同目标变换进入 `operations/bowl.py`，五份 MuJoCo 唯一实现迁入 `simulation_bowl/`，现场接线保持独立，未复制实现。
- [x] pick/retreat 报告、实际起终态与模型绑定合为 `SimulationBowlHandoff`；增益、planner、门会话仍由 Session 管理。
- [x] 扩展现有测试，覆盖缺失/过期/模型变化交接、部分失败终态、控制恢复与局部结果范围；补同会话 Extract→Place。
- [x] 相关回归 503 项通过；cuRobo 从 fresh reset 重跑 load、unload 和完整任务均成功。真机仅离线适配检查，未启动设备。
- [x] 同目标规则的数值测试、现场适配目标对照与观测过期拒绝通过；实际放入四点矩阵与基线完全一致。保留原后端差异，不宣称完整双后端执行器已统一。

验收收益：改放碗流程时只需定位碗装配与已有底层操作；不用同时维护多个交接字典；原 CLI、回放与完整任务保持可用。若提取后仍要传整个 Session 并随意改内部字段，应缩小提取范围，不能只换文件位置。

### A 批实际文件落点

| 当前位置 | 职责与来源 |
|---|---|
| `microwave_task/operations/bowl.py` | 原 object_keyposes、放入深度/撤离旋转、取出 profile 目标、桌面高度变换；仿真和现场共用。门角 Task 准入共用检查，目标由适配显式提供 |
| `microwave_task/simulation_bowl/{pick,insert,extract,retreat,place}.py` | 原五个 physical/pick 模块的唯一 MuJoCo 执行实现；旧模块删除，活动 import 更新；数据/报告路径不改 |
| `microwave_task/simulation.py` | 独立碗操作方法、Task 会话和资源；`SimulationBowlHandoff` 集中报告/实际起终态/模型绑定，指纹不进入现场 policy |
| `realworld/microwave/*_planning.py` | 现场观测、校准和持物证据核验；目标变换调用共同实现，不再从物理执行模块导入 |
| `tasks/policies/__init__.py` | 源身份递归覆盖操作子包；现有通用能力，不额外造追踪框架 |

A 批当时只移出六份实现；后续 B/C 收尾已迁出剩余内容并删除 applications。保持既有真机和仿真准入范围差异及明确的混合仿真辅助，不改评分，不声称真机硬件已验收。

本轮证据在 `runs/tmp/task-refactor/20261031/`（临时、不跟踪）。同配置对照墙钟：load 48.83→48.49 s，unload 133.30→132.33 s，完整任务 154.92→156.88 s；原步骤顺序及完整四项判据保持。完整任务 13 步，真实 cuRobo 返回成功，CUDA Graph 开启，源码身份覆盖当前操作子包。单个固定初态对照不代表位置分布成功率。

实际视频分别为 `after-load_bowl/video/task.mp4`（10 s）、`after-unload_bowl/video/task.mp4`（8.83 s）、`after-all/video/task.mp4`（87.33 s）；导出耗时约 0.78/0.90/3.99 s。全部通过完整解码检查，各抽查一张终态关键帧，不表示逐帧看完。仍是辅助夹持＋旋钮运动学更新的混合仿真。独立审查发现并修复插入报告生成后录像收尾异常掩盖原规划失败的回退，保留原报告及额外异常；失败测试已验证。

### B：杯/勺 policy 与共享 Pick 接线（已实施）

- 杯/勺 policy 只调用公开操作接口，观察、颗粒身份检查、报告与资源管理由 `simulation.py` 负责。原阶段顺序、参数、阈值和 `until_skill` 保持；未新增简单任务的子任务框架。
- `tasks/tabletop/pick.py::TabletopPickRun` 保存两者共用的 Pika 左臂 Pick 接线；杯/勺专用抓取几何、接触分类与碰撞过滤在各自 `pick_backend.py`。`CupPickRun` 不再继承 `SpoonPickRun`。这不是任意机器人通用执行器。
- `tasks/results.py` 统一范围、操作/完整成功语义、实际终态与失败引用。局部运行的 `task_success=null`；失败保留实际观测和原始规划类别。末态观测失败不能返回成功。
- 精确颗粒成员和辅助夹持仍是原有仿真检查，尚未用真机观测替换。不能由 policy 变短推断 sim2real 已完成。

### C：其余任务与 applications 退场（已实施）

- 普通刷保留原观测排序、归拢、重试循环，只补结果语义；专用 motion 接线原样移到 `sweep_cube_task/execution/`。
- 长刷 `policy.py` 给出原抓取/排序参数，场景装配在 `simulation.py`；统一启动器、旧 CLI 和源码身份同步更新。
- 微波炉原门/旋钮/碗规划在 `microwave_task/planning/`；共同门目标规则合入 `operations/door.py`；门仿真会话与执行在 `simulation_door/`，旋钮仿真在 `simulation_knob.py`。既有真实设备接线仍归 `realworld/microwave/`。
- 显示在 `visualization/microwave/`；杯/勺/刷回放在 `feedback/tabletop/`；纯视觉规划在 `tasks/planning/`；Pika 接触分类在 `robots/contact/pika.py`。未复制新 Skill 或通用后端框架。
- 删除旧 aliases 与 applications 目录；活动 import、动态路径、来源哈希及输入 checker 元数据已更新。视觉恢复哈希排除物理适配、保留共同规则变化检测，避免目录搬迁造成无关失效。
- 微波炉 subtask 仍使用专用 CLI；统一启动器并未因此新增 subtask 参数。历史记录不改写，回放由读取器消费旧格式。
- 门目标装配仍复用 `visualization/grasp_tuner/microwave_scene.py` 的已有构建器，没有借迁目录重写其 Knowledge 边界。杯/勺仍依赖仿真颗粒统计；这些限制与独立最终评分、真机观测替换单独处理。

### D：有真实需求再做

同一操作重复调用/恢复时，引入与操作类型分开的调用 ID，避免覆盖历史证据；目前微波炉会拒绝重复 step。这是未来长任务需要的能力，不在 A 中顺便实现。
最终独立评分、checkpoint、旧视觉 run_steps 与实际 run_operation 收敛、真机感知替换分别做行为变更，不能夹在结构重排中。

### 本轮验收

相关回归 818 项通过，机器人另有 45 项通过；随后清理门模块 import 的 196 项、视频完整范围修正的 37 项定向回归通过（含重叠，不累加）。五类场景 inspect、六个长期命令参数入口以及五份旧实际记录读取通过。只读审查和32份直接迁移实现的规范化 AST 对照未发现动作算法变化。

| 固定输入与条件 | 迁移前 → 后 | 实际证据 |
|---|---|---|
| 微波炉 seed123 名义摆位 | 成功 → 成功 | 13步骤、4项最终检查；参数投影及步骤顺序一致 |
| 倒水 seed10，显式 weld | upright_30 IK失败 → 同样失败 | 倒入5球/洒落0，未完成扶正放回 |
| 橙勺 seed21，显式 weld | 成功 → 成功 | 舀起的4球转移到目标碗 |
| 普通刷 randomize seed36 | 成功 → 成功 | 3方块入盘 |
| 长刷名义通道 | 成功 → 成功 | 3方块入盘 |

四类杯/勺/清扫的输入指纹、环境/策略参数、阶段顺序一致，阶段目标矩阵最大绝对差0；均有真实 cuRobo 原始返回和 CUDA Graph 记录。微波炉单次墙钟156.88→157.45 s；这不是系统性能或成功率评测，不宣称缓存加速收益。

本轮临时证据：`runs/tmp/task-refactor/20261031/remainder/` 下 `comparison.json`、`microwave-comparison.json`、`video_checks.json`；五段 actual MP4 在 `after-<cup|spoon|sweep|long|microwave>/video/task.mp4`。时长分别31.75/28.00/36.42/11.92/87.33 s；导出约1.48–4.28 s。全段解码、阶段覆盖与成功标记核对通过，实际打开微波炉撤离、勺子终态、普通刷终态三张关键帧；不表示逐帧观看五段视频。

实跑发现视频把新 `requested_scope=full` 误当局部范围，已通过失败→通过的测试修正，并仅重导出受影响的杯/勺视频；原始运行记录不改写。所有成功均保留接触核验后的仿真辅助夹持标签；微波炉另有旋钮运动学更新，仍不属于全物理或真机成功。

复查本次完整微波炉记录：

```bash
python tools/viewers/view_task.py --task microwave_task1 \
  --run runs/tmp/task-refactor/20261031/remainder/after-microwave --mode actual
```

独立操作仍用原 `run_microwave_task1_physical.py --subtask load_bowl --preparation prefix --output <新目录>`；杯/勺保留 `--until-skill`。仿真和真机共享策略及已提取的目标规则，执行接线仍各自管理，不宣称三后端已共用完整执行器。

## 如何判断迁移值得继续

每批只回答：修改一个操作需动哪些文件；单独运行是否与完整任务复用；失败能否直接定位；原参数/阶段/判据是否保持；运行耗时有无异常增长。
当前已采用公开操作接口、明确交接、少量共用接线和失败摘要，保留各任务自己的细节。没有收益则停止推广，不能为了目录整齐而全库迁移。
