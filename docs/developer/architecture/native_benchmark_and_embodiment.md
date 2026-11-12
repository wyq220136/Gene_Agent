# 新 Embodiment 与原生仿真 Benchmark 接入设计

状态：基础接口已实现，ALOHA 双臂与原生运行已实测。2026-11-01。面向实施与维护者。
操作流程见[接入原生 benchmark](../../task_development/native_benchmark.md)，机器人检查见[机器人与夹爪迁移](../../task_development/robot_and_gripper.md)。

本页保留接口设计与验收约束；当前实现和证据范围见末节。操作命令与依赖选择见接入流程。

## 1. 本轮目标与完成含义

增加两项独立可检查的能力：

1. 接入新的 embodiment：用明确的机器人描述、规划 profile 和控制映射支持新机械臂；本轮以 ALOHA-AgileX 为新增实例，以既有 Realman/Pika 为回归对象。
2. 在已有仿真 benchmark 内开发：直接使用原环境、控制器和评分器，连接本项目 Knowledge/Skill/Motion，并向开发 Agent 提供可定位失败的证据；本轮以 RoboTwin Novel Tool 为新增实例。

MuJoCo 是已有后端，不是外部题目必须转换到的中间格式。机器人与仿真器分别适配，不能把二者揉成一个新任务专用类。

本轮功能验收不等于完整 hammer 成功，不等于三类九物体成功率，也不等于任意机器人即插即用。原生场景、真实 cuRobo 轨迹、真实控制执行、共享 Skill 的接线及可解释的失败记录必须实测；接口 import 或 CPU 替身测试不足以完成验收。

## 2. 用户要求与范围

- 严格沿用原 bench 的场景、机器人、物理、初始化、控制、预算和评分。更换原题目机器人须另标实验条件。
- 可使用 bench 允许的 GT 感知替代，例如当前物体 pose；不能从 seed、setting、专家轨迹或隐藏评分状态计算动作答案。
- 允许读取机器人模型、标定、关节限位和控制映射，不能将其与泄露目标答案混为一谈。
- 同类任务优先共用 Skill；实例差异尽量通过 Concept/profile 调整，这是建议而非强制零样本约束。
- 新开发集中于 Gene_Agent。tool_use 源码和资产只读引用；结果、诊断和可配置缓存写入本项目。
- 正式规划使用 cuRobo，默认 CUDA Graph 与 CuroboPlannerPool；不能以原 bench 专家代跑替代本项目能力验收。
- 第一版支持具名旋转关节臂组和可校验的平行夹爪控制。不将弧度轨迹强行解释为移动底盘、直线关节或灵巧手接口；这些类型以后按实际消费者扩展。
- 不预建远程服务、插件市场、通用任务编排器或所有仿真器适配。仅保留当前调用链需要的 Protocol、数据验证和显式装配。

## 3. 当前实际边界

| 代码 | 当前事实 | 实施方向 |
|---|---|---|
| `knowledge/geometry`、`knowledge/grasp` | 几何和候选主体不依赖 MuJoCo | 直接复用；旧来源默认值不沿用到新环境 |
| `skills/contracts.py`、`skills/runtime/operation.py`、`runner.py` | 已有 OperationState、StageOutcome、CheckResult、OperationBackend 与阶段循环 | 不再建第二套 Skill 生命周期 |
| `skills/library/pick/skill.py` | Pick 阶段通过 backend + PickRequest 执行，原生 adapter 已接同一循环 | 不复制阶段顺序 |
| `simulation/pick_backend.py` | 部分检查和记录访问 model/data，使用 MuJoCo 接触 | 不当作任意后端实现直接复用 |
| `motion/model.py`、`client.py`、`planner.py` | 具名关节、TCP 目标、请求/结果与 StageMotion 已独立 | 保持契约，扩展模型来源 |
| `motion/curobo/session.py`、`core.py` | session 经 load_motion_config 选择 Realman/Pika/ALOHA；core 保留原求解器 | 新模型独立 profile，未知模型拒绝 |
| `motion/curobo/runtime_bridge.py`、`scene.py` | 碰撞与执行接线直接读取 MuJoCo | 外部环境生成同语义快照，不伪造 MuJoCo 对象 |
| `simulation/__init__.py` | lazy export 保留旧入口，导入原生子模块不加载 MuJoCo | 仅旧功能按需加载引擎 |
| `feedback/motion_debug.py` | 可独立保存规划事实 | 继续复用 |
| `feedback/state.py`、`failure.py`、`visualization/video_export.py` | 状态和视频多数依赖 MuJoCo | 原生证据由适配器提供；统一索引不强制统一原始格式 |

无 MuJoCo 的纯规划入口仍使用 Pika 几何，不能把“无引擎依赖”当作“任意机器人支持”。

## 4. 运行结构与生命周期

```text
原生 evaluator ── reset / budget / termination / score
      │ 调用策略入口                    │ 原始结果
      ▼                                 ▼
Task policy → 共享 Skill → StageMotion → cuRobo
                   │          ▲
                   ▼          │ 当前碰撞/机器人快照
          受限 Runtime 接口 ← 原生 adapter
                   │
          原生控制器 / 物理仿真
                   │
          实际观测 + 独立诊断记录 → 开发 Agent
```

原 evaluator 是 episode 生命周期的唯一控制者。policy 回调不得重置环境、重复推进物理或自建另一套评分。原生 adapter 的命令必须经过原控制路径和预算检查，长轨迹也要响应中途终止。

调试入口可从 fresh reset 执行到某阶段；检查前缀不等于完整任务。checkpoint/恢复是可选能力，未支持时明确拒绝，不能把只含 pose 的快照当可继续积分的完整状态。

## 5. Embodiment 契约

### 5.1 描述与动态状态分开

`RobotDescription` 表达模型身份与静态映射；`OperationState` 中的机器人数据表达当前关节、base pose、TCP 和夹爪状态。基座可变时每次观测更新，不永久缓存首次世界变换。

建议模型字段：

| 字段 | 语义 |
|---|---|
| `embodiment_id`、`profile_id`、`source_hashes` | 明确实例、规划适配与源文件版本 |
| `groups` | 任意非空具名臂组；不限定 left/right，也不默认七轴 |
| `joint_names`、`planner_joint_names` | 显式一对一映射；缺失、重复或不匹配报错 |
| `joint_types`、`limits` | 本版臂组只接旋转关节，位置 rad、速度 rad/s；明确连续关节处理 |
| `base_link`、`tool_link`、`T_link_tcp` | 必须来自实际模型与坐标校验，不硬编码 link_7 |
| `gripper` | 模型/关节与 mimic 映射、命令模式、实际开口范围和测量定义 |
| `planning_assets` | URDF、配置、碰撞球与排除规则的可解析来源及内容哈希 |

所有 `T_A_B` 表示 B 到 A；位置 m、角度 rad。边界将原生四元数显式转成矩阵；源序与目标序留在转换器中。夹爪米制开口与 native normalized command 不得直接相等；无实测映射时不能宣称宽度正确。

Realman 现有 metadata 通过显式转换继续使用，不一次重写所有消费者。只为仍被调用的旧 API 保留兼容，新增入口拒绝未知 profile，不回退到 Realman。

### 5.2 规划配置提供者

新增 `load_motion_config(group, robot_metadata, *, gripper_snapshot=None) -> dict`，返回当前 SynchronousPlanner 已消费的 robot_config、关节名称、world_from_base、link_from_tcp、tool_frame 等字段。session 调用此入口，Realman/Pika 分支保留原值，ALOHA 提供单独 profile。

每轮包含非活动臂、本体、夹爪和 attachment 的正确几何。动态夹爪球只在来源、数量、拓扑及更新机制兼容时复用；否则带原因重建。不能为了命中缓存删掉几何哈希。

### 5.3 本轮 ALOHA 来源

当前 `novel_tool_clean.yml` 指定 `aloha-agilex`；`_embodiment_config.yml` 指向相应资产目录。`config.yml` 声明每臂六关节、`fl_link6/fr_link6`、夹爪 base/mimic 关系、delta/global 变换与 robot_pose。

实际配置必须从 resolved setup 与运行时核对，不能仅凭模板构建。`envs/robot/robot.py` 的末端变换和夹爪归一化只是审计入口，需多组 FK/实际开口验证变换次数与方向。原 native robot 的 expert/planner 仅作对照。

## 6. 原生 Runtime 与 Benchmark 边界

### 6.1 最小接口

在 `simulation/contracts.py` 定义少量协议；数据继续使用已有类型：

```python
class RuntimePort(Protocol):
    def observe(self, *, object_names: tuple[str, ...],
                robot_groups: tuple[str, ...]) -> OperationState: ...
    def execute_trajectory(self, trajectory: JointTrajectory, *, group: str,
                           stage_id: str, expected_state_id: str) -> StageOutcome: ...
    def execute_gripper(self, *, group: str, width_m: float,
                        duration_s: float, stage_id: str,
                        expected_state_id: str) -> StageOutcome: ...
    def hold(self, *, duration_s: float, stage_id: str,
             expected_state_id: str) -> StageOutcome: ...

class PlanningSceneProvider(Protocol):
    def snapshot(self, *, state: OperationState, group: str,
                 collision_policy: CollisionPolicy) -> dict: ...
```

`OperationState.values` 的新运行时公共内容为 episode_id、scene_revision、sim_time_s、world_frame、objects、robots、observation_provenance。objects 保存实际 T_world_object；robots 保存具名关节、实际 TCP、夹爪观测；允许字段由该 bench 观测策略筛选。诊断字段不在此对象中。

episode 身份进入 state_id，重置后即使相同 pose 也不能复用旧计划。状态读取不推进物理；启动命令前核对 expected_state_id。执行结果包含实际 start/end、执行范围及原生证据引用，终态读取失败使用 unavailable，不能用计划终点补位。

`JointTrajectory` 沿用当前采样时间和具名弧度位置。native 重采样/插值必须记录实际周期、速度约束与跟踪结果；不能丢弃轨迹时间，或改变原控制器增益来掩盖执行差异。

原生 episode 的终止与阶段结果分开：StageOutcome.details 保留 `terminated`、`truncated` 和原生原因/来源引用；前者为原环境终止，后者为预算等外部截断。它们不等于 task_success；没有明确区分的原接口返回 unknown并保留原字段，不自行推断。原 evaluator 公开的终止信号可以正常结束policy，隐藏评分细节仍只进入诊断。

`PlanningSceneProvider` 的快照复用现有 cuRobo snapshot schema：scene_model、snapshot_sha256、scope、active_group、world_frame、gripper_snapshot 及原生几何来源 records；增加 episode/state 引用。不复制碰撞求解器。接触例外必须按阶段声明，无法支持的过滤明确失败；保留任务必要接触但不能全局关闭目标碰撞。

### 6.2 能力声明

adapter 提供不可变的能力声明：支持的观测字段、控制模式、原生物理步钩子、规划碰撞几何、截图/视频、接触诊断、状态恢复。unsupported 与诊断 unknown 有不同含义；空数组不能冒充已验证无碰撞。

第一版不强制通用 `reset`、`evaluate` 方法：它们由原 evaluator 拥有，本项目启动器负责调用原接口。统一的是请求/产物身份和语义，不复制 evaluator。

### 6.3 Skill 绑定

共享 Pick 使用已有 PickRequest、run_operation、OperationBackend。RoboTwin 适配将阶段目标、RuntimePort、PlanningSceneProvider 和 CuroboPlanningSession 绑定到 backend；不复制阶段顺序。

抓取/持物检查按允许观测提供 CheckResult，例如实际工具与夹爪相对 pose 随抬升保持。bench 内部接触可供开发诊断，只有原协议允许时才能参与在线策略准入。不能把旧 MuJoCo backend 的接触要求静默改为 always true；证据不足按 unknown 失败。

## 7. 开发反馈与防止信息泄漏

运行时 policy、开发 Agent、评分器是三个不同消费者。开发 Agent 可以在开发 run 结束/暂停后看隐藏接触与评分事件来定位错误；这些字段不回传给当前 policy，也不自动驱动其恢复。

新增独立 `DiagnosticsPort.capture(*, stage_id: str, state_id: str) -> dict`。由 orchestrator/记录器持有，不挂在 policy 的 RuntimePort 上；返回可序列化数据/文件引用，不返回 native_env。代码边界不是对恶意 Python 的安全沙箱，验收实际调用链与传入字段。

### 7.1 足够定位问题的证据

| 问题层 | 需要的信息 |
|---|---|
| 几何/坐标 | mesh/Concept、实际 pose、TCP 轴、候选参数与目标来源 |
| IK/规划 | 原始 MotionResult、目标/FK 误差、限位、起点/目标/路径碰撞、失败时刻 |
| 执行 | 计划与实际关节/TCP 曲线、控制周期、实际物体 pose、部分执行位置、时间预算 |
| 接触 | 对象对、位置/法向、时间、可获得的力或冲量及单位；未提供须明确 |
| 原评分 | 原生原始结果、规则身份、可获得的分项/状态机事件；只有 bool 时不编造原因 |
| 媒体 | 原生全景/接触截图、实际 MP4、相机、阶段到媒体时间索引 |
| 异常 | traceback、最后可知状态、未完成阶段及记录器自身错误 |

所有证据通过 run/episode/stage/state/sim_time 关联。按阶段写入并在异常时保留前缀，记录器失败不得覆盖原控制/规划错误。诊断读取不改环境、不推进仿真；若渲染必须用副本则明确来源。

### 7.2 统一索引而非统一原始文件

沿用 MotionDebugBundle、StageOutcome、CheckResult；新增 `feedback/native_run.py` 管理原生记录的索引与引用，不强迫 SAPIEN 导出 MJCF 或 qpos。摘要包含 failure_kind、observed_facts、hypotheses、evidence_refs；未知归因保留 unknown。

索引至少包含：schema_version、benchmark/embodiment/protocol/input 身份、scope、原评分报告引用、事件流引用、planning/execution/media/raw_native 证据列表及内容哈希。原评分是 task_success 来源；基础设施失败和未完成测试不能直接变成任务失败或成功。原 evaluator 的统计保持原样，基础设施异常另外列出并报告分母。

原生视频可直接作为 actual 证据，保留采样率/仿真时间对齐。对现有统一视频入口按后端分派；本轮只接实际消费所需的原生输出，不重写所有 viewer。无数据时不可生成虚构 actual。

### 7.3 复现能力

默认保存 reset 条件、实际初态、源版本、动作前缀和参数。重置同 seed 后核对实际初态，不假定引擎位级确定性。

精确恢复是可选能力；只有完整保存原生物理、控制器、评分器及 RNG 状态并验证后才声明支持。pose 导出只供检查，不能当 checkpoint。局部恢复结果单独标记，最终完整任务来自 fresh reset。

## 8. 原协议与导入隔离

RoboTwin 初版通过独立子进程启动 `script/eval_policy.py`，policy_name 指向 Gene_Agent 模块。外层 evaluate.py 的配置解析必须逐项对应：case 身份、模型实例、工具摆放、seed、专家筛选、动作预算、60秒等原生时间限制、视频与评分。以上数值读取源配置，不在本设计建立第二真源。

子进程以 RoboTwin 为 cwd，顶层 benchmarks 必须解析到原项目；Gene_Agent policy 不再导入本项目顶层 benchmarks。主进程只读套件文件/结果，不 import 外部环境。实际 import 来源进入运行记录。

`hit_tools.yaml` 首对象为 hammer_train / beat_nail_hammer / model_id=0。原配置可能包含旧任务相关选项，必须看 resolved task_setup，不能因模板字段存在就假定对当前任务生效。

开发与最终验收样本预先分开；专家筛选仅用于原协议的案例选择，不向策略提供动作。不能针对失败换 seed 或删分母。

## 9. 文件落点与迁移范围

| 位置 | 内容 |
|---|---|
| `robots/model.py`、`robots/aloha.py` | 最小机器人描述校验、从原模型解析 ALOHA |
| `motion/curobo/profiles/{__init__,aloha}.py` | 配置选择与 ALOHA 规划适配 |
| `simulation/contracts.py` | 无引擎协议、状态/能力字段验证 |
| `simulation/robotwin/{runtime,scene,diagnostics}.py` | 原生运行、规划快照、开发诊断；只按当前职责拆分 |
| `simulation/mujoco_adapter.py` | 对现有 MujocoRuntime 做同语义薄适配，用于最小跨后端验证 |
| `feedback/native_run.py` | 原生事件和证据索引，不复制评分器 |
| `benchmarks/novel_tool_task/source.py` | 外部套件与身份解析，不复制资产/判据 |
| `tasks/policies/novel_tool_task/` | 原生策略回调与共享 Skill 绑定；后续 hit_task/pour_task/sweep_task 按需增加 |
| `tools/run/run_novel_tool.py` | 套件入口；运行/inspect 的明确范围 |
| `tools/diagnostics/check_embodiment.py` | 可复用模型/FK/碰撞检查命令，业务运行不依赖它 |
| `tools/diagnostics/inspect_native_run.py` | 读取证据摘要与媒体索引，不重跑策略 |

本表都是计划路径，不能视为当前可执行入口。现有 MuJoCo 实现先保持原位；仅为解除 eager import 或实际复用边界迁移必要代码，不全库改名。现有 benchmarks/catalog.py 只扫描 *_task，外部套件初版独立启动，不虚构统一注册完成。

## 10. 验收矩阵

| 能力 | 必须实际验证 | 不能由什么替代 |
|---|---|---|
| 模型支持 | ALOHA 六轴与既有七轴配置；多姿态 FK/TCP、夹爪开合、错误映射拒绝 | URDF 可显示 |
| 规划支持 | 已知净空/碰撞、非活动臂、动态夹爪、真实 cuRobo 普通/直线轨迹、缓存更新 | import 成功、CPU IK |
| 原生执行 | 原控制器跟踪、实际终态、暂停/预算终止、状态连续性 | 动画、设置 qpos、命令已发出 |
| Skill 复用 | 一个共享 Skill 在原生后端运行，状态/检查使用允许观测 | 只调用原专家 |
| 解耦 | 公共接口及 RoboTwin 实际路径不加载/调用 MuJoCo；一个旧后端最小对照 | 只移动文件 |
| 反馈 | 一个真实失败可关联原始证据和媒体；能说明失败层或明确未知 | success=false |
| 协议一致 | 原 evaluator、resolved setup、输入/源码/资产身份、预算与评分委托 | 手写等价判断 |

在运行前冻结容差和样本清单。建议初始 FK 数值一致性容差为1mm/0.5°，至少5组有效关节配置；这是模型适配检查建议，不是放宽规划成功阈值或原 bench 判据。实施者若因原生 FK 定义需要不同容差，须在比较前说明并保存。

CPU 协议/错误路径测试与 GPU/物理验收分别报告。修改公共入口时定向回归 Realman/Pika 与既有 Skill，不能由一条新轨迹推断全部旧任务通过。

## 11. 开源、记录与长期积累

稳定的接口、adapter、模型 profile、共用算法、必要测试与使用说明进入 Git。实例输入默认在 `data/benchmarks/novel_tool/`，试验在 `runs/tmp/`，保留证据在 `runs/`，阶段计划在 `docs/tmp/`，均不追踪。

正式结果保存实际生效的 Concept/profile/资产哈希；另发版本化输入包即可复现，不依赖重新调用模型生成相同标注。正式代码不 import runs/tmp、docs/tmp 或历史试验脚本。

长期维护三个入口即可：开发流程说明如何接入；本文维护实际契约与能力边界；机器人迁移文档维护模型验收。经验证的排错经验再加入 human_experience；不得把暂定推测、计划中的能力或某次成功写成普遍结论。

## 12. 风险与实施顺序

主要风险是 ALOHA 末端坐标映射、夹爪动态碰撞、SAPIEN 执行时序及现有反馈链的隐式 MuJoCo import；按模型→执行→共享 Skill→诊断→原评测逐层验证。

先落地模型契约和引擎无关边界，再做原生适配；保留现有消费者兼容。GPU/依赖问题先诊断并留证据，不把 fake planner 测试计为完成。若原接口无法无修改接入，提交具体调用证据和最小替代方案，不直接复制整套 evaluator。


## 13. 当前实现与边界（2026-11-01）

- `robots/model.py` 验证具名旋转臂组、限位、坐标和夹爪描述；`motion/curobo/profiles` 显式选择配置。未知 profile 报错，旧 Realman/Pika 加载路径保留。
- `simulation/contracts.py` 复用已有 `OperationState`、`StageOutcome`、`JointTrajectory`；`robotwin/` 实现原控制器执行、实际状态、全场景碰撞导出和独立诊断。`mujoco_adapter.py` 复用旧执行器，原导入路径通过 lazy export 保持。
- 原 benchmark 与本项目使用不同版本 cuRobo 时，`ProcessPlanningSession` 在独立 worker 内维护原 `CuroboPlanningSession`/pool。配置字段在 ALOHA profile 内转换，原环境不注入新依赖；不提供 CPU fallback。默认 CUDA Graph，模型/锁关节变化按原 pool 规则重建。
- ALOHA TCP 必须包含 SAPIEN 末端 joint 的 `pose_in_child`，再应用 native 的 global/delta/bias。仅抄 `config.yml` 的旋转会产生 180°偏差。此固定关系由原生 API 提供，已用多组关节角核对。
- 原 evaluator 持有 reset、专家可解筛选、预算及最终评分。策略参数只有显式白名单；观测只含当前 pose 与机器人本体状态。接触和原评分归独立诊断，不自动进入 Skill。Python 对象隔离是 API 约束，不是面向恶意代码的安全沙箱。
- 原生视频、原结果和 episode 流由 `evaluation_index.json` 索引；每阶段保存精确实际终态，观测不可用时记录 unknown，不填计划值。`inspect_native_run.py` 可按阶段只读检查。

已验证：原场景 seed 0/7/21；左右各 5 组 native/cuRobo FK（最大约 `2.24e-7 m`、`5.21e-7 rad`）；双臂分别普通/直线规划、125 个原物理步的短轨迹执行、明确新增障碍起点碰撞；原 evaluator 中保持动作 25 步。旧 Realman/Pika 模型已用原 MuJoCo 执行器实跑 cuRobo 短轨迹。最终双臂各13项诊断全部通过，含空手闭合、80mm目标重新张开（实际约78.94mm）、动态锁关节变化后的正常模型重建与规划。共享 Pick 的 open→pregrasp 在原 evaluator 的两个 fresh-reset 案例（原筛选 seed 100000/100001）均成功；完整 Hit 原评分仍为0/2。相关回归420项通过，5项原有失败单列；不是全库、位置泛化或真机验收。

当前范围：原生 adapter 仅支持此固定基座 ALOHA 的旋转臂组和平行夹爪；没有宣称支持移动基座、直线关节、灵巧手或任意 benchmark 自动接入。碰撞导出支持已实测的原 mesh/box/plane/cylinder，圆柱用外接32边形保守覆盖并标记，未适配的解析几何明确报错，不用内接网格低估体积。碰撞源保留原 native 模型；本轮不等于所有碰撞球覆盖、附件或完整接触任务验收。共享 Pick 已接原生执行，后续完整 hammer 的验收见下节；其他实例和任务目前仅能解析源套件，不能据此宣称策略迁移完成。

运行证据在本地 `runs/tmp/native-benchmark/` 下按轮次保存，不追踪 Git；验收失败与成功记录均保留。五项旧测试失败已在未改主目录重现，须与新增回归区分；不修改历史资产/数据来使其通过。


## Hammer 完整操作的增量能力

最新实际验收与复现入口见[hammer 操作说明](../../user/novel_tool/hammer.md)，上节保留的是基础接口历史证据。

- Knowledge：按原 actor frame 标注 handle Cylinder 与 strike-head Cuboid，复用 Cylinder 抓取候选生成；共享 `knowledge/operation/hit.py` 用当前工具/TCP/目标 pose 计算作用面目标。
- Skill：现有 Pick 完整执行后调用共享 Hit 的 align→strike。Novel Tool 的接线与后置检查集中在其策略目录，未复制到通用 Skill。实例 Concept、profile、候选与有效参数在忽略的 `data/benchmarks/novel_tool/`，每次 run 保存内容及哈希。
- 持物碰撞：原工具全部 collision shape 以保守球覆盖迁入独立 `gene_held_tool` link，使用实际工具/link 相对变换；不会给物理系统添加 weld。保留机器人原球与 padding，只允许手指与工具接触，腕部继续检查。
- 阶段接触：lift 仅允许工具/table，strike 仅允许工具/目标；声明需与 scene 和 request 严格一致。优化器屏蔽后的整条离散轨迹仍检查机器人/持物体与其他障碍及自碰撞，不把无碰撞优化结果直接当验收。
- 执行：cuRobo/CUDA Graph/pool；仅对 align 显式放慢执行时间，不变更关节路径。抓取开口为临时证据，lift/hold 检查实际抬升与工具/TCP 漂移，align/正常结束的 strike 检查实际作用点。
- 结果：原成功可能提前中断 strike，动作层保留 `native_terminated` 与实际终态；任务层读取原评分，不伪造轨迹终点。原任务使用其受控 nail 位移规则，不能解释为验证了真实木材穿透。

这些实现已有 hammer 消费者；其他工具、目标形状、机器人和任务族仍需各自验证，不构成通用物理夹持或任意敲击任务的成功保证。
