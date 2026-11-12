# 开发新任务

本目录是人和 Agent 共用的阶段流程入口。项目目标见[核心说明](../../核心说明.md)，行为约束见 [AGENTS.md](../../AGENTS.md)，排错思路见[人类经验](../../human_experience.md)。只读当前阶段需要的文档，不必每轮加载全部资料。

## 先选择输入路线

- **已有仿真 benchmark**：按[原生 benchmark 接入流程](native_benchmark.md)使用原环境、机器人和评分器，先复现原生基线并检查场景，不要求重建或转换到 MuJoCo。新 embodiment 单独验收；公共接口和当前 adapter 的[实现与能力边界](../developer/architecture/native_benchmark_and_embodiment.md)单独记录。
- **真实场景 / 新建题目**：按下表准备资产、布置场景和定义判据，再进入共享 Concept、Skill 与任务迭代。

两条路线均先交付实际检查过的多个初态和 3D 查看命令。严格外部 benchmark 的初态、物理与判据不可按自建场景流程自由调整。

## 真实场景路线的阶段与交付

| 阶段 | 阅读入口 | 完成前应交付 |
|---|---|---|
| 1. 从语言 / RGB / 视频理解操作意图 | [场景与成功条件](scene_and_goal.md) | 对象、动作、预期结果、限制与尚不明确的事项 |
| 2. 从 RGB-D 重建并合理布置场景 | [桌面重建实例](reconstructed_tabletop_scene.md) | 来源明确的 mesh、pose、坐标、碰撞与物理配置 |
| 3. 定义成功条件与初态分布 | [场景与成功条件](scene_and_goal.md) | 多个初态的截图，以及能查看 3D 物体和场景的命令；这是第一个交付点 |
| 4. 按真实操作需求标注 part Concept | [Concept 与动作](concept_and_actions.md) | 功能部位、参数和 mesh 叠图 |
| 5. 复用或补充 Knowledge / Skill / Task | [Concept 与动作](concept_and_actions.md) | 可独立检查的动作与子任务、阶段结果 |
| 6. 仿真迭代 | [仿真迭代](simulation_iteration.md) | 单 Skill、sub-task、完整 Task 的结果，耗时、恢复及分布评测 |
| 7. sim2real | [真机迁移](sim2real.md) | 感知与物理差异、授权后的逐段真机证据 |

换机械臂或夹爪时，按需进入[机器人与夹爪迁移](robot_and_gripper.md)。已有合格资产、Concept 或 Skill 可以复用，但须说明依据和当前任务下的验证范围。

编写 Task policy 时参考 [Task 编写约定与迁移方案](../developer/architecture/task_authoring.md)：提供目录职责、状态交接及已有任务的迁移证据，复杂任务才按需拆子任务。五类任务的结构整理已完成；该说明还列出当前 Python 入口、各任务局部调试的差异及逐批验收状态。真机接线与验证范围须单独确认。

## Agent 如何推进

编写前先按[Concept与动作](concept_and_actions.md#写第一版代码之前)选择最小检查，规划接线复用[Motion入口](../../gene_agent/motion/README.md)。恢复时先读本次 `development_summary.md` 或[运行检查命令](../user/motion/curobo_diagnostics.md#计时与独立重规划)，再进入失败现场；缺证据保留未知。

写Skill前逐阶段列出“关键精确交互 / 可配置误差”，并核对原生接线实际传递近似IK预算，不能仅复用阶段名称。迭代前先锁定原失败案例、猜想与预期观测，按[阶段精度与失败回归](simulation_iteration.md#阶段精度与失败回归)验证；原失败未测到目标阶段时，不以其他案例成功或整批成功率替代。

开始或恢复任务时，先读任务现有配置和最近运行记录，说明当前阶段、已验证内容、缺失证据与下一步。每阶段结束报告可运行命令与证据，不凭口头声明通过。阶段失败先定位具体 Skill / pose，再查任务定义、资产、摆位和动作逻辑；重要的语义取舍向用户反馈，常规仿真调试自行推进。

运行输入与题目条件使用现有配置作为单一来源；题目与策略入口见 [benchmarks](../../benchmarks/index.md)。实验记录放 `runs/`，临时探索放 `runs/tmp/<专题>/<本轮>/`；不要把成功条件、场景数值复制到多份进度文档。当前这些是开发约定，**尚不表示已有自动阶段门控或统一 development.json 协议**。

稳定规则、经验和操作步骤分别维护，避免重复：规则写 AGENTS，经验写 human_experience，本目录只写流程及工具入口。修改成功条件、分布或仿真辅助条件时记录新版本，旧结果不能直接算作新条件下的验收。
