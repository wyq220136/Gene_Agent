# Concept 与操作逻辑

## 何时读取与输入

场景和操作目标明确后读取。输入是米制 object-frame mesh、物体 pose、操作意图、机器人配置和已有模块。先查已有实现，缺少需要的 Knowledge / Skill 时再开发。

## 执行

1. 先确定抓哪里、作用于哪里，再标注功能 part。Concept 不用覆盖整个物体，例如杯沿可用 Ring，不必把杯底一起表示。
2. 当前可用的[自动标注](../user/knowledge/concept_agent_annotation.md)入口是 `tools/run/annotate_concept.py`，人工调整入口是 [Concept 编辑器](../user/knowledge/concept_editor.md) `tools/editors/edit_concept.py`。这两个入口当前支持 Cuboid / Cylinder / Ring；自动标注为静态 mesh 功能近似，不是 Articraft 可动结构自动重建。不要把编辑器支持的模板、底层 Concept 类型和抓取 generator 混为同一个列表。
3. 保存 `T_object_part`，检查 `T_world_part = T_world_object @ T_object_part` 与 mesh 叠图；核对尺寸、局部轴、作用面。需要时用实测尺寸校验重建尺度。
4. Knowledge 表达参数化候选、操作几何与适用范围；运动时序、路径约束、碰撞策略、执行与恢复由 Skill / Motion / Task 承担。pre-contact 等目标的几何生成可以复用 Knowledge，但不能据此把整段执行写成开环成功。
5. Skill 按实际状态生成阶段目标，记录路径类型、允许接触对、attachment、夹爪命令与后置条件；Task 传递执行终态并组合 Skill。通用能力与具体任务装配分开，当前新策略位置见 [benchmarks 与 policy](../../benchmarks/index.md)，规划接口见 [Motion](../../gene_agent/motion/README.md)。

## 写第一版代码之前

先读[已验证的常见错误与推荐写法](../../human_experience.md#第一次编写新任务时先避免这些错误)，然后只为本次改变的输入确认下面几项：

| 要确认什么 | 最小检查；已有同配置证据可复用 |
|---|---|
| embodiment、frame、单位与接近轴 | 多组实际关节的 FK/TCP 平移和旋转对照；原生新机器人用 check_embodiment.py，不能只看零位 |
| 抓取区与作用面 | Concept/原资产叠图，同时显示候选夹爪和局部轴；核对 object frame、尺度、夹持宽度 |
| 抓取如何服务后续操作 | 除抓取 pose 外，考虑持物的抬升/转向/操作姿态；先便宜筛选，再为少量候选查关键路径，碰撞检查属于 Motion/Task |
| 每段动作的约束 | 列出真实起点来源、目标、普通/直线路径、允许接触对、attachment、执行后置条件，确认 Skill 衔接 |
| 规划调用方式 | 直接使用持久 session + StageMotion，沿用默认 CUDA Graph/pool，不按物体复制 solver 创建或重试代码 |

第一版先让独立 Skill 在一个已检查初态通过，再看完整任务；接入与性能错误在扩大评测前处理。叠图、IK、规划、实际夹持各自保留证据，不能互相代替。物理操作仍可能需要迭代，这些检查不承诺一次成功。

## 工具与限制

在项目根目录的 cuRobo 环境中，以下是现有特定示例，不是任意新物体的自动接入命令：

```bash
PYTHONPATH=. python tools/editors/view_grasp_knowledge.py --kind ring
PYTHONPATH=. python tools/editors/tune_grasp_profile.py --profile task_1_bowl_0 --output runs/tmp/grasp-tuning/grasp_profiles.json
PYTHONPATH=. python tools/editors/view_microwave_knowledge.py --mode door_push --ik-backend curobo
```

先创建本轮输出父目录。`tune_grasp_profile.py` 的历史默认输出为 `archive/task_1/configs/grasp_profiles.tuned.json`，新试验应显式指定 `runs/tmp/`；检查后再更新正式配置。调参工具的对象与 generator 支持范围见[使用说明](../user/knowledge/knowledge_tuner.md)。微波炉 Knowledge viewer 的纯 pose IK 不代表场景碰撞或路径通过，详见[微波炉调参](../user/microwave/microwave_knowledge_viser.md)。

## 交付与完成条件

交付 Concept 叠图、候选参数与顺序、选中 TCP pose、坐标变换、夹爪开口，以及独立 Skill / sub-task 的运行和查看命令。区分几何、纯 IK、碰撞、规划与执行结果。新对象未接入 viewer 时需补适配并验证，不能只添加 profile 就宣称支持。

## 失败处理

查[人类经验](../../human_experience.md)的 IK 与操作问题：扩大合理候选范围、核对 TCP / frame、摆位及阶段约束。允许近似 IK 时按 AGENTS 的阶段预算执行，返回实际 FK 与误差；关键接触和任务后置条件仍须通过。进入[仿真迭代](simulation_iteration.md)检查真实执行。

候选IK失败时参照[cuRobo诊断](../user/motion/curobo_diagnostics.md)核对实际FK误差、约束与TCP；近似候选展示不等于授权近似执行。
