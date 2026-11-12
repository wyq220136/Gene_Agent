# Gene_Agent 开发规则

本文件面向在此仓库工作的 Codex/开发者，只保留长期有效的约束与当前入口。项目动机和工作流程见 [核心说明](核心说明.md)；已验证状态以实际代码、测试和运行记录为准，历史计划不能覆盖用户最新要求。

## 阅读入口

- 初次理解项目：[核心说明](核心说明.md)，了解目标与模块关系。
- 开发或恢复新任务：[阶段流程](docs/task_development/index.md)，先选真实场景或原生 benchmark 路线，确认已有输入、最近证据和下一步；外部仿真任务的恢复入口见[原生流程](docs/task_development/native_benchmark.md)。
- 诊断失败或积累经验：[人类经验](human_experience.md)，保留用户原意，区分经验建议与强制约束；不把未验证推测写成长期结论。
- 诊断和修复已完成的 Gene_Agent Task 实验、用固定阶段图锚定失败位置、结合实际视频与结构化日志追到上游设计根因，或为新任务建立稳定阶段 SVG：先完整读取 [Gene Agent task failure diagnosis Skill](skills/gene-agent-task-failure-diagnosis/SKILL.md)。用户要求修复时，必须沿真实调用链修改产生根因的代码并用原失败初态回归；不得用放宽合理条件、延长预算或增加重试掩盖坏目标。进入 cuRobo 内部原因时再同时读取下一条专用 Skill。
- 诊断 cuRobo 的 IK、碰撞、路径、执行或任务后验失败，以及提出只读修复建议：先完整读取 [cuRobo failure diagnosis Skill](skills/curobo-failure-diagnosis/SKILL.md)，按其证据路由与报告契约执行；不要仅凭外层异常归因。
- 当前能力以代码与对应任务证据为准；阶段通过须有证据，历史计划和模型的文字声明不能代替验收。
- 编写新任务先读[常见错误与推荐写法](human_experience.md#第一次编写新任务时先避免这些错误)；编写规划接线先读[Motion 入口](gene_agent/motion/README.md)，复用现有 session/StageMotion，按[诊断流程](docs/user/motion/curobo_diagnostics.md)读取失败与耗时证据。

## 目标与工作方式

- 目标是在人类辅助下快速部署新任务，并在声明的物体位姿分布上调到高成功率。真实场景可先在 MuJoCo 用 GT 调试；已有仿真 benchmark 则直接使用原环境、机器人、预算与评分，不要求转换到 MuJoCo。需要 sim2real 时再处理感知与物理差距，真机逻辑尽量与仿真共享。不要把单次规划、动画或截图成功当成任务成功。
- 优先改善 Codex 下一轮开发效率：让问题能由结构化记录、明确错误、截图和视频定位；原子 Knowledge、单个 Skill、完整 Task 都应能独立运行和检查。人可点选、拖动或输入参数，结果要可保存与复现。
- 一个模块一个模块地改，先复现基线，再做最小修改并交付可验收效果。多步骤修改写计划；简单修正不额外造计划。不要为保持旧结构而复制实现，也不要为未来可能性提前造复杂框架。
- 首选中文注释，关键术语如 gripper、IK、TCP、cuRobo 可保留英文。对坐标、单位、约束和非显然原因写清楚；不要只复述代码。
- 当前项目根目录为 `Gene_Agent`，Python 包直接是 `gene_agent/`，没有额外 `src/`。既有 MuJoCo 路线沿用 `curobo` 环境配置；原生 benchmark 与规划器可分别指定实际解释器，记录依赖来源，不为接入覆盖原环境依赖。真机运动需要单独授权和安全验收。
- 不要过度设计
- 文件结构要清晰，模块之间界限分明， 仅调试时使用的文件和实际推理时使用的功能要分开
- 可以制定很详尽的规划，文档内容可以写的很多，但是最终给我看的版本要精简

## 模块边界与现有目录

| 位置 | 职责 |
|---|---|
| `gene_agent/knowledge/` | Concept、按参数范围及后处理生成的抓取候选，以及旋钮/门操作几何；门轴由用户显式给定；可调用抽象 IK 接口作**可选的纯 pose 可达性**筛选，不依赖 cuRobo，不把碰撞或路径混入该结论 |
| `gene_agent/skills/` | 同步的阶段目标、直线接近、动作执行与失败处理；Pick 是抓取动作的组合，不复制 grasp Knowledge |
| `gene_agent/motion/` | 后端无关的请求/结果契约、碰撞策略；`motion/curobo/` 是当前 cuRobo 实现与配置适配，不把 Pika 专用配置放进通用夹爪模型 |
| `gene_agent/realworld/` | 真机设备、感知、实测状态和执行检查；微波炉专用现场适配集中在 `realworld/microwave/`。现场接线与 Task 策略分开，不复制共享 Knowledge/keypose 算法；共享边界须按实际调用链说明，目录合并不等于三后端已共用完整 Task 执行器 |
| `benchmarks/<name>_task/`、`gene_agent/tasks/policies/<name>_task/` | 独立题目（场景、初态、判据）与任务策略；入口见 `benchmarks/index.md`。Skill 暂沿用现有实现，任务目录统一 `_task` 后缀 |
| `gene_agent/tasks/` | 公共运行、结果语义与确有消费者的共享接线；微波炉目标、规划及仿真适配在其策略目录，显示适配归 `visualization/microwave/`；不恢复 applications 目录 |
| `gene_agent/scene/`、`robots/`、`simulation/` | 场景状态、机器人/夹爪模型与仿真适配；公共 RuntimePort、规划场景与独立诊断分开，MuJoCo 与原生 benchmark 各自实现，避免硬编码一种机器人/引擎 |
| `gene_agent/feedback/`、`visualization/` | 通用记录、失败证据、回放，以及 Viser/MuJoCo 的显示适配；viewer 不重新定义任务成功事实 |
| `tools/`、`experiments/` | 长期 viewer/editor/diagnostic/run 命令放在 `tools/`；可复用研究实验放在 `experiments/` 专题子目录并登记索引，临时实验放 `runs/tmp/`。不要恢复顶层转发脚本，新增代码直接使用实际路径 |

任务目录如 `data/task_1` 可同时持有 Knowledge 和 Skill profile，但参数职责分开，并保存单位、来源和最终生效值。不要修改历史 run 来使新代码通过。当前场景构建仍有 task_1 专用部分，不能称作任意场景的通用导入。

## 坐标、碰撞与规划

- 迁移旧逻辑时，以实际调用代码、配置、资产变换、测试和运行结果交叉核对；旧文档只是线索。Realman URDF、SDK base、flange、夹爪安装和 TCP 容易出错。明确 `T_A_B`、米/弧度、四元数顺序、关节顺序与 pose 应用次数，并用多组关节配置核对 FK/TCP。
- visual、Concept、cuRobo 规划碰撞和 MuJoCo 接触几何是不同用途。visual 尽量接近原资产；collision 可以简化，但须检查尺度、原点、link 映射、可动关系、碰撞球覆盖、self-collision 排除与 attachment。至少验已知净空和已知碰撞样例。
- 每个 Skill 阶段明确起点、目标、路径类型、启用的碰撞对象、允许接触对和 attachment。抓取的 approach 保持 TCP 直线；`pregrasp → grasp` 失败就报错，不暗中从原始状态换一条路径。不要全流程忽略某个物体，也不要靠禁用碰撞或瞬移伪造物理成功。自建/重建场景中，仿真资产摩擦不足而几何抓取已验证时，可在实际双侧指垫接触后显式启用仿真特权 weld；必须记录激活证据、在松爪时释放，并将结果标成“仿真辅助成功”，不能写成纯物理或真机夹持成功。
- IK 目标失败时，同时检查操作 Knowledge、物体摆位、机械臂工作空间、TCP 和求解器。把“纯 IK 无解”“目标碰撞”“起点碰撞”“路径规划失败”“执行/接触失败”分开记录。若失败物体未知，应保留原始证据并标明归因未知。
- **仿真任务的正式机器人运动规划默认使用 cuRobo；当前仿真后端负责状态、物理执行与接触判定，MuJoCo 是其中一个后端。** `MujocoPosePlanner` 等 CPU 求解器只能用于显式标记的诊断/对照，不得替代 cuRobo 产出正式成功率。开始评测前核对实际调用链，运行时记录并校验 `planner_backend=curobo`，保存至少一次真实 cuRobo 规划返回；若任务尚未接入 cuRobo，先完成接入，不能把在 `curobo` conda 环境运行误当作已使用 cuRobo。CPU 诊断须显式启用并在 run/汇总中标明，不得静默 fallback。近似 IK 优先复用已有 cuRobo 阶段策略，不另写 CPU 版本来代替正式评测。
- **正式规划默认开启 CUDA Graph 和规划器复用。** 使用现有 `CuroboPlannerPool`，按任务/Skill或执行桥接层生命周期持有，默认精确同步动态夹爪碰撞球；拓扑、源模型、锁关节、attachment或碰撞策略不兼容时重建，结束后释放资源。不要每个小阶段无条件新建solver，也不要只删缓存key中的哈希而沿用旧几何。模式名保留 `cuda_graph_benchmark` 兼容现有记录；只有明确诊断/对照需要时才显式用 `development` 关闭图，关闭动态夹爪复用须显式 `reuse_gripper_geometry=False` 并记录原因。运行记录保留实际模式、缓存命中及重建原因；不能把该默认设置当作其他任务已通过回归或真机验收。
- **近似 IK 是可配置的阶段能力，不等于严格 IK 成功。** 对推门、pre-contact、pre-grasp 等允许一定末端误差的阶段，可在 Knowledge 中保留未严格收敛的候选，按明确的平移/转角上限和任务语义筛选；不要因严格阈值失败就一律丢弃。每个候选与 Skill/规划结果必须同时传递原始目标 pose、实际 FK TCP pose、关节解、平移/转角误差、严格收敛与约束可行标志、所用阈值及 `approximate` 标记。标记为近似时，不得冒称到达原目标。
- cuRobo 规划近似目标只能由**显式启用的阶段策略**触发：先保留失败 IK 的候选及碰撞/关节约束诊断，拒绝非有限、超误差阈值或已知不安全的候选；用被接受候选的实际 FK pose/关节状态重新规划，仍须通过正常的起点、终点、整条轨迹碰撞与执行检查。不得把失败 IK 的关节角直接当轨迹、放宽关键接触/抓紧/插入终态，或仅把 `success` 改为 true。Skill 应读取真实执行终态，检查本阶段可接受的操作后置条件，并把近似程度传递给下一阶段与 run 记录；若不满足则正常失败或恢复。默认保持严格模式，误差预算按 Skill 阶段配置并在 Knowledge 检查器中可见。
- **编写或接入新Skill时逐阶段声明精度。** pre-grasp、自由移动/中间点、非精准push目标等应评估并显式配置允许误差；grasp、击打接触、插入终态保持其精度要求。误差策略须贯通候选预筛、正式规划、实际FK/执行检查和下一阶段；字段存在但运行请求未消费不算接入。看本次`development_summary.md`与真实请求核对，不能用求解器默认容差替代阶段预算。方法见[仿真迭代](docs/task_development/simulation_iteration.md)。
- 保留可重复的简单同步基线以及 cuRobo 直线规划；不恢复原异步执行链。batch IK、CUDA Graph、缓存和重试是否有益，要在同一场景、seed、约束下比较成功率、冷/热耗时与显存，避免盲目保留复杂加速分支。

## 仿真、反馈与验收

- 下列资产、摆位调整与仿真辅助建议适用于自建/重建场景。严格外部 benchmark 必须保留原初始化、物理、预算、筛选与评分；不得用这些建议改变题目条件。开发诊断可读取原内部状态，但不能把 setting、seed、专家轨迹/抓取点、隐藏评分状态注入策略。

- 仿真服务于操作逻辑开发，不必严格复刻真实材料或每一处摆位。优先保留任务相关的可达性、碰撞、门接触和目标判定；对资产误差可使用可开关、可记录的仿真特权，但不得掩盖规划失败、关键接触缺失或任务终态失败。
- 仿真场景不必逐毫米复刻真实摆位。可显式调整物体和桌面以避免初始穿透、提高可达性，但要记录采集参考、仿真名义、扰动采样和落稳后实际 pose；保持父子 frame 与任务语义。物体扰动必须注明范围、分布、坐标系、seed 和有效初态规则，困难样本不得静默剔除。
- Skill 之间传递**执行后的实际状态**，不静默 reset；完整 Skill 与各阶段使用后端支持的查看器检查同一记录，原生 benchmark 可用原生截图、视频与 3D viewer，不为显示另造 MuJoCo 场景。区分规划轨迹、运动学预览、MuJoCo 实际执行和真机执行；规划 attachment 与物理夹持证据分开显示。
- 终端给简短结论，run 目录保存输入、场景/代码/资产版本、阶段状态、原始 planner 返回、traceback、碰撞清单、相机参数、截图和视频。截图要实际打开检查；相机至少覆盖全景、目标/接触、夹爪侧面与失败位置，并可查看机器人/夹爪坐标轴。
- 自建 MuJoCo 任务可在本地 Linux 交互查看；Viser 适合远程、人机调参和截图；原生 benchmark 优先使用其原生 viewer/媒体。显示适配共享实际状态与回放事实，不能各自实现一套 Skill 逻辑。
- 成功判定分层：Knowledge 几何、纯 IK、碰撞、轨迹连续性、仿真实际接触/物体位移、任务后置条件、真机结果。每项报告说明证据来源和未验证范围。固定开发、回归和最终验收样本，参数修改前后使用同一判定标准；局部恢复成功不代替新 reset 的完整试验。
- **迭代先验证原失败案例，再扩展评测。** 每次修改记录案例/版本、猜想、单一改动、目标阶段与预期观测；规划问题优先冻结请求重规划，接触问题从原reset复现前缀。未到达或未执行要验证的阶段应报“未验证”，一次成功不证明根因。原失败改善后做覆盖不同方向/臂/边界的小回归，再冻结新验收样本；用于调参的旧验收案例转为开发/回归集。现有原生比较入口为`inspect_native_run.py --compare-with`，其他后端沿用同一证据要求，不假定已有统一自动门控。
- 每轮交付写明改动、可运行命令、数值结果、实际检查过的图片或视频及限制。环境 import 成功不等于 GPU 规划成功。当前可用入口见 [tools/README.md](tools/README.md)；Skill 检查见 [用户说明](docs/user/visualization/skill_viser_inspection.md)。

## 当前能力边界与文档

当前硬件与任务的能力必须按具体配置和证据读取，不能由旧快照推断新任务已通过：

- [夹爪实现说明](docs/developer/robots/gripper_implementation.md)：visual、接触代理与模型限制；[硬件迁移](docs/task_development/robot_and_gripper.md)说明跨配置验收。
- [微波炉物理任务](docs/user/microwave/mujoco_task1_physical.md)：历史混合仿真与当前入口，辅助夹持、旋钮更新等条件须明确；不等于全物理、位置泛化或真机成功。
- [Benchmark 入口](benchmarks/index.md)：题目、策略和各任务的证据范围；[用户文档](docs/user/index.md)提供检查方法。
- [原生 benchmark 流程](docs/task_development/native_benchmark.md)：恢复证据与下一步；[embodiment 与仿真接口](docs/developer/architecture/native_benchmark_and_embodiment.md)说明模型/规划/控制边界。新增模型或 adapter 不代表完整 Skill/Task 已验收。

标定文件组织：日常操作以 `gene_agent/perception_modules/camera/calibration/README.md` 的两条路线为准（A：量尺＋板相机；B：手眼＋固定机器人移板）。新采集和失败尝试放 `data/calibration/runs/<用途>/<本轮目录>/`，图片分析等诊断放 `data/calibration/diagnostics/`，开发历史放 `docs/developer/calibration/`；不在 `data/` 根目录、代码目录或全局文档目录散放临时记录。历史原始数据应保留，迁移须核对路径依赖和文件内容。


# 开发代码的准则

- 只保留明确有当前用途的文件，“以后可能有用”不足以保留；其余归档。
- 文件按职责分层；索引只用于帮助选择工作入口，不逐目录、逐源码铺设。代码集中在 `gene_agent/index.md`，复用已有 README；文档、测试、工具和配置按实际导航价值保留。几行就能说明的内容并入上级，缺索引不代表文件无用。
- 区分运行输入与开发产物、长期参考与临时报告。长期内容分类并跟踪；临时文档放 `docs/tmp/`，弃用代码及配置放 `archive/`，均不跟踪。临时 test/check 不进入正式回归。
- 临时实验统一放 `runs/tmp/<专题>/<本轮>/`，不跟踪，表示可删除；有明确保留价值的实验放在该目录之外，并在相应 `index.md` 记录用途和入口。
- 整理时同步更新索引、引用与 Git 状态，验证原功能；不误删原始数据或混入其他 Agent 的改动。
- 生成结果按任务与轮次组织，不把大量临时文件散放在正式代码或永久资产目录中；仅供检查的工具与正式真机运行入口分开。

具体存放、索引及验收规则见 [文件整理说明](docs/developer/maintenance/file_layout.md)。

文档入口按用途区分：`docs/task_development/` 为跨模块的新任务流程，`docs/user/` 为具体操作与验收，`docs/developer/` 为接口、诊断和历史决策。任务专用人工规则、近似 IK 与仿真辅助边界见[专门清单](docs/developer/simulation/task_specific_rules_and_simulation_privileges.md)。阶段性计划与临时报告不写成长效规则；修改功能时同步更新入口和能力边界。

## 用户的开发与交付要求

- 新任务先按[阶段流程](docs/task_development/index.md)选择输入路线：真实场景需重建与合理布置，原生 benchmark 保留原题目。确认成功条件和初态分布后，再标注功能 Concept、复用或补齐 Knowledge/Skill/Task、仿真迭代；需要时进入 sim2real。
- 场景是第一个交付点：Agent 实际检查多个初态截图，并提供查看本次 3D 物体和场景的命令，让用户判断有没有搞错。
- 仿真中自主寻找方案，不把常规调试问题直接抛给用户；较难问题借助 Astra（可用时），重要取舍可寻求用户建议，主要仍由 Agent 推进。真机运动授权不由此替代。
- 每个 Skill / sub-task / Task 应能独立检查；先确认动作成功，再检查完整任务、规划耗时和失败恢复能力。调试既定位直接错误，也检查任务定义、资产、摆位和操作知识等根本原因。
- 策略输入尽量限于真机可获取的信息，可用 GT pose/分割替代对应感知；隐藏接触等内部状态用于诊断/评测不等于允许策略依赖。使用仿真专有输入或辅助必须明确标记及适用边界。
- 任务专用代码放在具体命名的目录，不与通用 Skill 混放。定期整理项目和用户经验，优先改善后续新任务开发效率；经验的具体排查方法维护在 human_experience，不在此重复。

## 视频反馈：统一入口

- 收到“录视频、提供操作视频、给人类视频反馈”等要求时，优先使用 `python tools/viewers/recording/record_skill_run_video.py --run <run> --output <本轮目录>/operation.mp4`；正式模块为 `gene_agent/visualization/video_export.py`。用法见 [视频反馈说明](docs/user/visualization/video_feedback.md)。
- 上述统一导出命令适用于已有回放适配的任务；原生 benchmark 优先交付原 MP4 与 `evaluation_index.json` 的案例/阶段引用，不假定其可由 MuJoCo 视频工具回放。缺适配时明确说明，不为录像转换物理环境或重跑已有动作。
- 默认 actual、EGL、12fps、每视角640×480；从共享实际回放采样离屏导出，复用 Renderer＋FFmpeg管道，不默认在线逐帧写PNG/GIF。换机位不重跑规划；用户要求重新运行时从 fresh reset 执行。微波炉 `run_microwave_task1_physical.py --video` 已接同一模块。
- 规划预览必须显式 `--mode planned`；辅助夹持/运动学更新保留标签，失败前缀不称成功。缺实际采样时补记录适配，不能虚构actual视频。
- 视频反馈默认直接导出 MP4＋JSON，快速检查可解码、时长和阶段覆盖后即交付；不额外批量截图、制作拼图、重复计时或反复导出。仅首次核对新机位、发现异常或用户明确要求时，按需抽取1–3张相关帧。只做解码检查时不得声称已完成画面验收。
- 核对原运行的完整成功条件，报告视频路径、来源、导出耗时和限制；录像反馈不扩展为工具重构或额外评测。

- cuRobo失败先按[诊断流程](docs/user/motion/curobo_diagnostics.md)读摘要，再核对起点、候选FK/约束和路径；球编号必须连同物体/link及快照身份报告。缺证据标未知，不能把有限候选失败当成不可达证明。
