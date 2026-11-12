# Benchmark、Policy 与稳定 Skill 库

2026-10-30 架构讨论与源码审查记录。下文 Skill 问题描述的是重构前快照；benchmark/Task 的当前迁移状态以总计划与 Benchmark 入口为准，拟议的统一 Skill 接口尚未实施。
原共享边界与回归范围见 [shared_operation_runtime.md](shared_operation_runtime.md)。
后续执行顺序以 [渐进重构总计划](incremental_refactor_plan.md) 为准：用户已选择先分离 benchmark/Task，Skill 重构暂缓。

## 已确认的方向

- real2sim 场景经过物理信息、任务目标、初始分布和判据补全后，形成独立 benchmark；不与 policy 混放。
- Benchmark 定义题目，包含资产引用、机器人配置、初态分布/有效性、可用观测与动作、成功失败和时间限制。
- Policy 定义解法，组合感知、Knowledge/Skill，以及任务级重试与异常处理；不能改写自身验收条件。
- 两者通过 benchmark 标识、版本和接口对应；同一 benchmark 可比较不同 policy，同一 policy 可用于兼容场景。
- 成熟 Skill 的核心功能与接口应稳定。新任务主要编写短 policy、配置对象 Knowledge 和参数，不重复后端代码。
- 操作按需查询、同步执行，下一阶段依据实际终态生成；不把整个任务预先展开成开环轨迹。
- Policy 的 GT 输入限于真实世界可获得的语义量。判定器及 Codex 调试端可有更完整的仿真证据，不能借此扩大 policy 观测。
- 当前仿真不要求 policy 通过相机估计 pose；相机可服务于 Codex 调试。未来真实感知保持观测语义并声明误差、时效与来源。

## 当前 Skill 库的具体问题

审查范围为当前工作树，其他 Agent 正在修改，后续迁移前须重新核对。`skills/` 当前有 23 个顶层 Python 文件（含 `__init__`）。

| 问题 | 当前证据 | 影响 |
|---|---|---|
| 多种调用与返回路线 | `pick.py` 返回 `SkillPlan`；`pick_flow.py` 返回 `OperationResult`；`sweep.py` 返回字典；门/旋钮还有纯几何阶段接口 | 调用者需要逐个理解和适配，规划完成与实际操作完成容易混淆 |
| 共享流程仍过薄 | `OperationStage` 只有名称和检查开关；`CallbackOperationBackend` 调用 `perform(stage.name)` | 共享了顺序，但目标构造、动作选择和证据处理大量留在 applications/realworld/simulation |
| Skill 反向依赖 Task | `operation.py` 导入 `tasks.checks`；`door_sequence.py::compose_task1_door_cycle` 导入 `tasks.microwave` | 下层混入上层任务组合；基础检查结果缺少合适归属 |
| 具体后端依赖泄漏 | `target_selection.py` 直接调用 `motion.curobo.approximate`；`executor.py` 转发到 `simulation.pick_backend` | 声称通用的入口仍要求了解 cuRobo/MuJoCo；兼容入口不能当作新公共接口 |
| 机器人与任务假设写进共享 Skill | `sweep.py` 写死 Pika link、brush/table/dustpan；`profiles.py` 加载 sweep 配置并限制 Pika 开口 | 换机器人或对象时仍需改 Skill 源码 |
| 历史动作形态固化 | `insertion.py` 固定 pre-place/place/两段撤离；`extraction.py` 固定五点；`placement.py` 面向旧 profile | 可保留为特定操作方法，但不能直接当作任意插入、取出、放置的稳定接口 |
| 规划与展示混入操作类 | `PickSkill.plan_visual_retreat()` 专门处理视觉 attach 后的规划；门等有独立预览编排 | 增加新 Skill 时容易再维护一套 viewer 流程 |
| 错误信息对上层不够明确 | `operation_runner.py` 将操作失败归为 `stage_failed`；详细证据在 `StageOutcome.details`，`OperationResult.to_dict()` 不包含该字段 | 现场记录可能另有细节，但公共结果不足以让 policy 判断如何恢复 |
| 状态与参数契约不一致 | `SkillContext` 是规划上下文；`OperationState.values` 为通用 Mapping；各动作有不同回调和 profile | 无法仅看入口知道所需观测、夹持状态、单位、能力和缺失信息处理 |
| 缺少使用导航 | `__init__.py` 只导出部分 Skill；实际用法散在调用方和测试中 | 新用户不清楚应该用计划接口还是执行接口 |

已有基础应保留：MotionRequest/MotionResult 与 MotionPlanner、真实终态交接、pass/fail/unknown 检查、近似 IK 证据、共享 Pick/Transfer/Extract/Knob 流程、单 Skill 回放。不是从零另写一个框架。

源码入口： [skills](../../../gene_agent/skills/index.md)、[motion 协议](../../../gene_agent/motion/planner.py)、[Task 流程](../../../gene_agent/tasks/policies/microwave_task/policy.py)、[运行桥](../../../gene_agent/motion/curobo/runtime_bridge.py)。

## 建议的职责边界（待后续设计确认）

| 层 | 负责 | 不应承担 |
|---|---|---|
| Benchmark | 题目、初态生成、独立评分、固定测试样本 | 指定抓取候选或恢复路线；让 policy 修改判据 |
| Task policy | 感知查询、选择/组合 Skill、任务级恢复与预算 | 直接操作 MuJoCo data、SDK、规划器生命周期或编写回放文件 |
| 薄运行上下文 | 提供观测、Skill 调用、能力检查和统一记录；绑定具体后端 | 藏入某个任务的步骤，变成另一套 Task 执行器 |
| Skill | 参数化操作，阶段目标、约束、必要检查与有界局部恢复 | 重置任务、采样初始分布、决定 benchmark 得分 |
| Knowledge | 对象/部件几何、候选与操作路径参数 | 整个任务重试和设备控制 |
| Motion/执行/感知适配 | 规划、实际动作、状态读取；把语义对象映射到模型/link/传感器 | 重新决定 Task 或 Skill 次序 |

Bench 与 policy 分别存放；benchmark 位于 `benchmarks/<name>_task/`，policy 位于 `gene_agent/tasks/policies/<name>_task/`。任务目录统一以 `_task` 结尾，不与 tasks 下的公共 runner 等文件并列；当前目录已建立，具体迁移状态与验证见总计划。
不整体把 applications 搬入 tasks：策略归 Task，题目归 benchmark，后端和 viewer 回到原职责模块。
场景资产可引用共享 assets，不必为每个 benchmark 复制。机器人能力变化要形成明确实验配置，不能跨机器人直接比较未注明条件的成功率。

### Skill 的稳定性

优先稳定语义操作，例如 pick、place、沿铰链操作、转动、工具推扫；按行为族分组，不按微波炉/碗等任务名分组。
核心契约可集中一处；动作多的行为族再建子目录。不要为了层级给每个短文件建目录，也不预造尚不存在的技能。
一份 Skill 导航说明公共入口、输入、结果、适用条件、能力与证据；源码 docstring 说明详细参数。

三种变化应区分：

1. 换物体、位姿或目标：优先改 Knowledge、参数或 policy。
2. 换机器人或执行环境：优先改模型、能力映射与后端适配；不承诺没有控制差异。
3. 新的接触模式、工具或控制需求：可能确实需要新增/修改 Skill，例如从位姿控制变成力控插装。

稳定不意味着只剩固定 pose 序列。Skill 内部需要按实际状态生成阶段目标、核验前后置条件。
现有已验证实现先保留为明确的方法，再由多个实际任务证明可复用后收敛；不要一开始做无限可配置的万能 Skill。

### 公共入口与反馈

建议提供面向 policy 的同步操作入口，参数包含语义对象/目标、操作配置和明确预算；由上下文提供观测、规划与执行服务。
公共返回应含：完成/失败/中止、已执行前缀、实际终态引用、操作后置条件及证据、失败分类、目标与实际误差、近似标记、run 引用。
观察接口须声明 frame、单位、时间/版本、来源与缺失信息；不能把任意字典包装一下就称为统一。
规划轨迹、实际执行和 benchmark 成功分别表达；Skill 不把模拟辅助夹持冒称物理成功。
允许检查/运行单阶段，但从阶段中间开始必须具备真实前置状态，不能伪造已完成前缀。

Skill 用有类型的 motion 请求声明路径与接触语义；近似 IK 的算法留在 motion，阶段是否允许近似及误差预算由操作语义决定。
共享 Skill 不应负责 solver 池、CUDA Graph、MuJoCo geom 或机器人 link 名。执行适配负责这些映射及模型更新。
Viewer 读取共享阶段和记录；预览使用显式预测状态，不另写操作策略。

### 重试与真机边界

- Motion：在同一目标/约束下进行有预算的求解尝试。
- Skill：执行前尝试同一操作的候选；实际动作后重查状态，只做已定义的局部恢复。
- Policy：选择重感知、换操作、改变任务顺序或停止；共享总预算，避免多层重试相乘。
- 执行后端：必要的停止、异常捕获、实际终态和证据保存。掉物或未知夹持时不能盲目重试，更不能默认松爪。
- Benchmark harness：reset、采样和最终评分；真机准备场景不能假装等价于自动 reset。

任务判定器与 Skill 检查可以复用几何谓词，但阈值/证据权限/版本归属独立。Policy 可见的目标可公开，隐藏 seed 和真值判定内部数据不进入 policy。
例如倒入数量：policy 使用可观测估计，benchmark 可用仿真真值评分。实时估计不足时应显式 unknown。
统一上下文有助于迁移，不等于一键 sim2real；真实接触、感知误差和设备能力仍需验收。

## 参考项目：本地代码和文档证据

本轮查阅本地检出版本，未运行这些项目，也未用 README 的论文成绩作为架构有效性的证明。

| 项目 | 具体依据 | 值得借鉴与限制 |
|---|---|---|
| RoboRSI | [项目分层](../../../third_partys/RoboRSI/README.md)、[真机 Env/Backend 与技能入口](../../../third_partys/RoboRSI/roborsi/embodied/embodiment/README.md)、[grasp_object 契约](../../../third_partys/RoboRSI/roborsi/embodied/skills/base/grasp_object/libero/SKILL.md) | long_horizon/atomic/base 层级；技能写清参数、结果、何时使用与恢复约束；返回结果和新观测。其技能也含具体后端与感知实现，不能声称它已做到机器人无关的稳定操作库，更不必照搬多 Agent 系统 |
| RAX | [技能导航](../../../third_partys/rax/skills/README.md)、[可复用程序格式](../../../third_partys/rax/skills/continual-learning/references/library-format.md) | `main(ctx, **params)`、前后置检查、失败和适用范围、draft/verified/deprecated；固化探索成果。此 checkout 主要为主分支导航和 Agent 工作流文档，具体机器人实现位于其他分支，未据此声称审计了完整执行代码 |
| ASPIRE | [环境判定](../../../third_partys/ASPIRE/aspire/sim/cap/envs/base.py)、[公共 API 和文档聚合](../../../third_partys/ASPIRE/aspire/sim/cap/integrations/base_api.py)、[技能积累](../../../third_partys/ASPIRE/aspire/sim/cap/skills/library.py) | 环境判定、暴露给程序的 API、可积累函数库分开；接口文档可从实际 callable 生成。函数出现频率和任务来源可辅助发现复用，但不等价于物理鲁棒性验收，不照搬其自动晋升标准 |

这些仓库中的 Agent SKILL.md 与本项目机器人操作 Skill 不是同一种东西。借鉴说明、能力边界和证据组织，不把提示词目录当成机器人执行模块。
本项目仍把原始实验证据放 runs，不照搬把大量证据图片堆进源码技能目录的布局。

## 后续实施应验证的结果

先用现有微波炉流程分离 benchmark 与 Task，保持 Skill 行为；待上层边界稳定，再以 Pick 和门操作验证 Skill 抽象，而非一次性移动所有文件。单纯分目录不能修复接口；另建庞大插件框架会增加负担；优先在现有共享循环上逐项收敛。

- 同一 Pick 实现能通过配置用于碗和刷子；不同夹爪/接触模式按能力声明，不假装无差异。
- 固定 benchmark/seed/判据，只换 policy 能独立运行与比较；失败样本不静默跳过。
- Policy 调用不需要导入 cuRobo、MuJoCo、SDK，不提供十几个内部执行回调。
- 未执行失败、部分执行失败、unknown 后置检查、近似到达与仿真辅助结果均可定位并正确传递。
- 单 Skill、单阶段和 Task 共用实际记录；原验收命令保持可用，必要时显式迁移。
- 仿真与真机适配共用操作语义，真机能力缺失明确报错；没有硬件验收时不声称真机成功。

具体文件移动与统一类型按总计划逐轮细化；本记录不授权改变已有 benchmark 判据或机器人行为。
