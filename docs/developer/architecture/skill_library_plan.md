# Skill 能力库与独立调试实施计划

2026-10-31。状态：**Pick 及其余全部现有能力已迁入 library；当前验收见文末第二轮记录**。本文保留分批设计依据，不表示所有物体分布或真机已通过。

执行时使用 `superpowers:executing-plans` 或已明确选定的执行工作流，按下列批次逐项完成；不因本文存在自动启动真机。

**目标：** Agent 能从一个入口找到 Skill 的能力、输入、成功承诺、适用限制及调试方式；完整 Task 与独立调试调用同一实现。
**架构：** 保留现有同步阶段循环、Motion 接口和环境适配；可复用能力集中在 library，执行基础设施与规划事实分开。先完成 Pick 再推广，不创建空目录或新执行框架。
**技术：** Python、现有 Operation/Check 契约、cuRobo、MuJoCo、Viser。
**依据：** 本文前半部分为接口设计；Task 边界沿用 [Task 编写约定](task_authoring.md)，现有能力以 [Skill 入口](../../../gene_agent/skills/index.md) 为准。

## 设计取舍与当前问题

仅移动文件不能改善调用；一次统一全部 Skill 风险过大。选择“先统一一个能力的使用方式，再迁移目录，实际验证后推广”。

- 当前 Pick 同时有 `PickSkill.plan()`、`run_pick_flow()` 和兼容 `run_pick_skill()`；规划辅助、阶段编排、设备装配容易混淆。
- MuJoCo Pick 已在 `simulation/pick_backend.py`，但装配仍需大量回调；不能把这些回调原样推给新 Task 作者。
- `OperationResult.to_dict()` 未保留阶段 details；公共阶段失败通常变成 `stage_failed`，原始原因需额外找报告。
- `door_sequence.compose_task1_door_cycle()` 引用微波炉 Task；`sweep.py` 固定 Pika link 和场景对象名，需要移交合适的责任层。
- Pick 消费者包括微波炉、桌面、清扫、真机、规划预览和 viewer；不能只修改一个仿真例子就删除旧入口。

保留：实际状态连续、unknown 不算通过、接触后显式辅助夹持、默认 cuRobo 与 planner pool、动态夹爪与 attachment 同步。此次不改 Knowledge 算法、动作参数、碰撞策略或 benchmark 判据。

## 目标文件组织

```text
gene_agent/skills/
    index.md                     # 唯一能力导航；按需求选择入口
    contracts.py                 # 共享 Check、状态、阶段和执行结果
    runtime/
        operation.py             # backend 协议与回调适配
        runner.py                # 原有同步循环
        planning.py              # SkillContext／SkillStage／SkillPlan
        profiles.py              # 通用 profile 解析
        poses.py                 # 位姿辅助
        target_selection.py      # 后端候选筛选接线
    library/
        pick/
            __init__.py          # 推荐公开 API
            skill.py             # Pick 阶段、停止范围与检查要求
            model.py             # PickRequest、抓取设置等专用输入
            planning.py          # 现有 Grasp/Pick 规划辅助
        transfer/                # skill.py 执行；planning/ 内放入、放置、取出规划
        door/                    # skill.py 执行；sequence.py 阶段；planning.py 目标
        knob/                    # skill.py 执行；planning.py 目标
        sweep/                   # skill.py 执行；model.py 显式接触角色
```

只创建本批需要的目录。小能力允许一个实现文件，不机械复制上述 Pick 拆分。
专用候选处理归相应能力；多个真实消费者共用的规划工具才保留公共位置。不新增泛用 utils 杂物层。
`SkillPlan/SkillStage` 是规划事实，先保留兼容，之后按消费者移到规划模块；不要与实际执行结果强行合并。
`CheckResult` 是跨 Skill/Task 的检查事实，迁移到 contracts；`resolve_unknown()` 的人工交互仍留 Task 层，旧 tasks.checks 导出同一类型兼容，不能复制两个 CheckResult。

## Pick 的公开接口与责任

推荐调用形态（make_pick_backend 需从已有环境适配获取，不是全局工厂）：

```python
from gene_agent.skills.library import pick

request = pick.PickRequest(object_id="bowl", group="right", settings=settings)
backend = make_pick_backend(session, request=request, candidate=selected_candidate)
result = pick.run(backend, request, stop_after="grasp")
```

`make_pick_backend` 为各环境现有装配的提取，不是新增全局工厂或字符串注册系统。

- `PickRequest` 最小字段：object_id、group、现有 GraspSettings。候选、选定 pregrasp 及已验证交接证据通过明确的后端装配参数绑定；不在 request 里放任意 callback、MuJoCo runtime 或 planner。
- `pick.run(backend, request, *, stop_after="pick") -> OperationResult` 是推荐实际流程入口。backend 在移动前核对请求对象/组与绑定输入一致；配置不得存在两份互相覆盖的默认值。完整请求需满足该配置的 Pick 完成条件。
- 保留原阶段逻辑：open → pregrasp → approach → close → grasp_check → establish_hold → lift → hold_check；旧配置的 lift/hold 选择保持原义，省略阶段时必须记录范围，不能冒称已抬升。
- 独立 grasp 暂作为 stop_after="grasp" 的明确能力，不立即另建重复 Grasp 执行器。接续已抓取状态通过单独、显式的续执行路径，保留现有证据验证，不向普通调用暴露可随意跳检查的开关。
- `.planning` 保留预览与候选规划；旧 `PickSkill.plan()` 等在迁移期间导出兼容。预览输出仍标为规划，不能成为实际持物证据。
- Knowledge 提供候选；Skill 负责动作语义、阶段和完成条件；motion 返回规划事实；backend 获取当前观测、调用设备、提供检查证据。Task 选择目标与恢复策略。
- 首批不承诺一个 backend 支持任意场景/机器人；每个适配必须列出实际支持的配置。

## 结果、信任与失败归因

扩展已有结果，不新造与 OperationResult 并存的 SkillResult：

| 内容 | 要求 |
|---|---|
| 范围 | 记录 skill、requested_scope、已完成阶段；pregrasp 成功不等于 Pick 成功 |
| 执行事实 | 区分预览、未执行、实际执行与部分执行；现有 executed 布尔值不足的情况用明确状态补充，先保持旧字段兼容 |
| 当前状态 | 成功和失败均返回实际终态；无法重新观测时标明最后已知状态及未知，不能填入规划终态 |
| 操作证据 | 接触、持物、释放、辅助夹持、近似 IK 等只保留必要摘要与原报告引用，历史事实注明对象和有效范围 |
| 错误 | 具体 stage、直接错误码/来源、原始报告引用；unknown 与真正失败区分；根因推断另记，不冒充事实 |

在 StageOutcome 增加可选结构化 failure（code、source、message、report_ref）；runner 优先传递原错误，否则使用 stage_failed。序列化保留该 failure 和必要证据引用，不直接倾倒含 numpy/设备句柄的任意 details。
仅在负责设备调用的适配边界捕获执行异常、采集现场并记录；公共层不反复 catch 后重试。程序错误保留 traceback，不变成“任务没成功”。

明确每个检查的生产者与消费者：
- 同一有效证据可复用；下游核对对象、时效、适用配置与中间是否发生使结论失效的事件。
- 搬运可能滑落、松爪、人工干预或状态不明时，需要新检查；不因“相信上游”取消正常路径碰撞或新动作后的持物检查。
- 对重复检查先记录原因与成本再删除；不凭函数同名认定重复，不建立通用证据缓存框架。

## 独立调试交付

1. 完整 Skill 和阶段停止都走同一入口；停止在 pregrasp/approach/grasp 后保存实际现场与范围，不静默 reset/松爪。
2. CLI 启动的新进程不能假装继承持物状态。支持新初态准备前缀及已有会话继续，暂不新增跨进程 checkpoint。
3. 复用 `tools/viewers/view_skill.py --run <run>` 和现有实际回放/视频导出；规划预览和实际采样明确区分。旧模型阶段名与新 flow 阶段名建立显示映射，不改历史 run。
4. 终端一句话说明：操作/停止范围、失败阶段、直接原因、证据位置。目录中能找到输入、实际终态和对应失败视频，不要求用户翻多份 JSON。
5. 参数变化写新 run，不覆盖旧失败。相机优先沿用已验证的全景/接触/夹爪侧面；此次不重构渲染器。
6. 能力目录每项只写用途、前置条件、输入、成功承诺、推荐调用、独立调试命令和已验证范围。细节链接既有文档，不逐 Skill 建索引。

## 实施批次与验收

### 0．冻结当前接口与小型回归窗口

- [x] 读取最新 AGENTS、Skill/Task 索引及 git 状态；保留其他 Agent 修改，核对本计划路径是否仍有效。
- [x] 列全 Pick 调用方及其输入、结果字段、记录路径、来源哈希引用；重点检查 simulation/pick_backend、tasks/tabletop/pick、微波炉 simulation_bowl/pick、清扫与 realworld/microwave/pick、realworld/sweep。
- [x] 运行现有 Skill/Task/回放相关测试，记录原失败；保存至少一个微波炉 Pick 实际基线和一个不同对象配置基线。正式仿真使用 cuRobo，记录真实 planner 返回与模式。
- [x] 在本轮 `runs/tmp/skill-library/<轮次>/` 保存基线与对照。长期回归样例只保留输入/seed/配置/预期类别及证据引用，登记 tests/index；不拷贝整批视频进入测试目录。

窗口至少覆盖：已成功抓取、不可行候选/路径失败、接触不足、抬升后持物失败、过期交接。物理场景难以稳定复现的失败用后端测试验证协议，并明确它不是物理验证。

### 1．公共结果与责任边界先收敛

- [x] 在 `tests/skills/test_operation_runner.py` 增加失败用例：原始 planner code 不被抹掉；部分失败返回实际终态；失败序列化保留报告引用；unknown 拒绝继续；未执行/预览不得计为实际成功。
- [x] 最小扩展 operation 契约与序列化，先不移动文件；现有构造调用增加可选字段保持兼容。失败无法观测时记录 unknown/last_known，不制造 actual。
- [x] 对照 Pick 的 grasp_check、hold_check 与下游持物检查，写明生产者/失效条件；首轮不删安全检查或修改数值阈值。
- [x] 通过现有 runner、Pick、Transfer、Extract/Knob 和相关回放测试，再进入目录迁移。

### 2．Pick 能力库与推荐执行接口

- [x] 将共享契约/循环迁到 contracts/runtime，原导出暂时指向同一实现。检查 import 层级，禁止 skills 反向加载具体 Task。
- [x] 将 Pick/Grasp 规划与阶段编排归入 library/pick；PickRequest 放专用 model，公开 run 与规划辅助分开。纯 pose 辅助从 executor 迁入合适规划位置，解除 simulation 与兼容 executor 的相互导入。
- [x] 从 MuJoCo Pick 提取可绑定的 backend，保持原 candidate/pregrasp、接触分类器、增益与规划设置；原 run_simulation_pick 作为兼容装配调用新入口，不复制阶段实现。
- [x] 真机和预览适配接同一阶段逻辑；纯预检只能报告预检前缀，不因为接口一致而报告已执行。
- [x] 加入请求对象/机器人组不匹配的拒绝测试、已选 pregrasp 不被重新推导测试、stop_after 后无额外动作测试、握持续执行拒绝无效证据测试。
- [x] 顺序迁移微波炉、桌面、清扫、真机和规划消费者。兼容包装只允许转发，不保留第二套算法；完成全仓引用检查后才删除无消费者旧模块。

### 3．把 Pick 调试入口真正接通

- [x] 在现有实际运行入口补齐 Pick 停止范围（先核对 argparse，复用已有选项）；不新建与 view_skill 平行的检查工具，不把物理执行开关混入纯 viewer 默认行为。
- [x] 将新结果接入现有回放：可选阶段、planned/actual、实际持物证据、原始失败和范围；既有历史 run 仍能打开。
- [x] 实测同一配置的独立 Pick 与 Task 内 Pick，比较目标/路径类型/碰撞策略/实际阶段结果；运行完整微波炉连续任务检查下游 Insert/Extract 交接。
- [x] 对杯/勺/刷至少跑受迁移影响的抓取配置；逐个记录已通过/原失败/新失败，不能以微波炉一个例子代替全部消费者。
- [x] 更新 skills/index、用户 Skill 检查文档和已有工具索引，交付实际命令与视频。真机只做离线适配回归，不连接硬件。

### 4．样板验收后再推广

- [x] Transfer：把 insertion/placement/extraction 与对应 flow 归入同一能力目录；保留各动作不同的完成条件，不把“释放撤离”和“保持持物取出”混为一谈。
- [x] Door/Knob：整理单次操作的输入与检查；将 compose_task1_door_cycle 归回微波炉 Task，保留已有阶段与角度容差。目标几何不宣称物理驱动成功。
- [x] Sweep：由适配提供 gripper link、工具/对象角色与允许接触；保留直线接触扫动和实际位置闭环。是否增加目标移动后置条件单独评审，不在目录迁移时暗改成功语义。
- [x] 清理仅转发的旧模块，更新 import、测试 monkeypatch 路径、文档、来源哈希清单和导出接口；不修改旧 run 哈希或历史报告。

每个能力单独回归与交付，不批量迁完再测试。首批默认只执行 0–3，第四批依据 Pick 的实际收益决定。

## 验证命令与通过标准

在根目录、curobo 环境，先运行基础回归：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl \
  /home/ubuntu/miniconda3/envs/curobo/bin/python -m pytest -q \
  tests/skills tests/tasks \
  tests/visualization/test_skill_run_replay.py \
  tests/visualization/test_microwave_actual_replay.py \
  tests/visualization/test_tabletop_actual_replay.py
```

再运行调用清单对应的微波炉/桌面/清扫/真机离线测试；目录可能由其他 Agent 调整，执行时按真实路径选取，不复制过时清单。
基线和迁移后实际运行使用同一配置、seed、判据、planner 模式；命令先按现有入口 --help 固定在本轮记录，再执行。测试不代替 GPU、接触或完整任务验证。

验收关注：有效交接没有重复执行上游；变化后的状态不会错误复用；失败不隐藏也不复位；预览不冒充执行；历史回放可读；控制资源在异常路径正确释放。
记录成功/失败、规划次数、检查次数及其理由、重试/人工介入和冷/热耗时。不设“所有检查越少越好”的指标；预筛选和最终直线规划用途不同，不能当重复调用删除。
交付一份简短对照：改一个 Skill 需要改哪些文件、推荐 API 是否易用、失败能否直接定位、是否产生行为回归及尚未验证范围。

## 本轮实现记录（2026-10-31）

以下为第一轮 Pick 样板的历史验收，后续整理与当前回归见第二轮记录。

- Pick 真正实现位于 library/pick；根目录 pick、grasp、pick_flow、executor 无消费者后已删除。根包导出的规划类和 operation 的旧类型导入保持兼容，指向同一实现。
- contracts 持有同一 CheckResult 与 OperationResult，runtime 管理阶段循环和后端适配。Task 人工确认留在 tasks/checks。
- make_pick_backend 绑定候选、请求和现有运动桥接，pick.run 提供阶段范围；现有场景装配封装内部回调。已选 pregrasp、候选顺序、直线接近与抬升逻辑保留。
- 同一 backend 的 grasp 后接续先复核当前接触；不支持从新进程恢复。本轮没有增加跨进程 checkpoint。
- 原始错误、阶段范围与实际终态进入报告；观测断开记录 unavailable 和 last_known_state_id。独立审查发现的预检误判、执行错误被规划成功覆盖、手臂绑定错误均有先失败再通过的回归用例。

检查责任保持简单：grasp_check 产出闭爪后的接触证据，establish_hold 消费它建立显式辅助；hold_check 检查抬升后的新状态；resume 重新确认持物，因为调用之间可能变化。每段路径仍有正常规划和碰撞检查，没有为减少次数删除它们。

本轮实际运行在 `runs/tmp/skill-library/20261031/`：

| 范围 | 结果与限制 |
|---|---|
| baseline-bowl / verified-full 中的 Pick | 都通过；候选与设置一致，抬升量均约 95.598 mm |
| baseline-spoon / verified-spoon | 都通过；抬升量均约 86.957 mm，辅助夹持显式记录 |
| verified-approach | 只执行 open/pregrasp/approach，通过；没有闭爪或抬升 |
| verified-cup | Pick 通过，辅助夹持；不据此报告倒水 Task 成功 |
| verified-brush/pick | Pick 通过；后续完整清扫候选搜索被主动中断，本轮只验收 Pick |
| verified-full | fresh reset 连续 13 步及原最终判据通过，约 165.73 s；辅助夹持＋运动学旋钮的混合仿真，非真机/全物理成功 |

完整实际 MP4 已解码，抽查了关门末段画面；没有声称逐帧人工观看。源报告、测试日志与独立审查修复记录留在本轮 implementation/。
全量测试为 1228 通过、18 失败：15 项标定缺 cv2/open3d，3 项清扫 Task 接口不匹配。后续消费者回归为 747 通过、8 项清扫接口失败（并发开发又增加了 5 项测试），详细名称见 full-suite.log / final-regression.log；未修改标定依赖或其他 Agent 的 Task 实现。

补充实际调试验收：verified-grasp 接触检查后停止、未抬升、未启用 weld，grasp-contact.mp4 为 3.417 s。独立桌面 Pick 已接入共享回放，修复了原入口只识别完整 Task 名称的问题；视频采用实际前缀的证据标签，未激活辅助时不沿用配置声明。

## 不在本轮范围

不新增全局 Skill 注册器、自动恢复引擎、经验数据库或 LLM 在线调度；不要求所有 Skill 具有相同内部结构；不强迫 Task 使用新的外层框架。
不借重构调抓取参数以“通过验收”。发现原算法缺陷单独记录责任节点，区分原有失败与迁移回归，再修复并回放成功和失败窗口。
用户经验固化到测试、参数说明或既有长期文档，临时诊断留 runs/tmp；AGENTS 只在稳定后增加必要导航，不复制本文。

## 第二轮：全部剩余能力整理（用户已授权）

2026-10-31。沿用上述设计，依次执行：

- [x] 冻结当前测试及实际 load_bowl 基线，不覆盖第一轮记录。
- [x] Transfer/Extract、Knob 迁入 library，规划辅助与执行流程在同一能力目录；增加明确范围的公开调用，保留内部后端模式与接触判据。
- [x] Door 的目标、检查、执行入口集中到 library/door；Task1 开关门组合移到 Task，物理 Session 继续管理设备、接触和 checkpoint。
- [x] Sweep 迁入 library/sweep，移除 Skill 内固定 Pika 和物体名称；适配器显式提供角色与允许接触。移动规划辅助/参数基础设施到 runtime，清理无消费者旧文件。
- [x] 更新消费者、源码版本清单、能力索引与现有调试命令。每个能力说明独立运行、局部停止、已验证范围和实际失败入口。
- [x] 回归现有 Skill/Task/真机离线/回放；fresh reset 连续微波炉及清扫实际验证，保留失败并定位责任，不趁整理调整参数或判据。视频使用同一实际回放导出。

本轮临时记录：runs/tmp/skill-library/20261031-rest/。未运行真机；本轮未增加 checkpoint 或新的恢复框架。

能力结果：Insert／Place 的 release 停止范围覆盖 detach、落稳和释放检查；Knob 的 grasp 停止包含夹持检查。Door Session 用共享能力入口逐阶段推进，保留重观测与原 checkpoint。Sweep 的工具／夹爪角色由 config/tasks/sweep/contacts.json 显式绑定，纯规划预览标记 prediction，未运动不能记为 executed。原始报告中的异常／接触错误由生产者直接传到共享结果，不重新执行验证。

| 本轮实际运行 | 结果与证据边界 |
|---|---|
| baseline-load | 迁移前 load_bowl 通过，约 49.69 s；同配置基线 |
| verified-full | fresh reset 连续 13 步通过，约 152.89 s，仿真时长 87.25 s；原混合仿真判据，辅助夹持＋运动学旋钮 |
| door-prefix | 首个开爪阶段完成后停止，报告 partial；未冒称开门成功。该 CLI 按既有规则返回 2 |
| knob-current | 当前名义场景独立旋钮通过；仍采用声明的运动学更新 |
| sweep-prefix | 固定初态 Pick 后到 lower_to_contact，通过；局部成功，task_success=null |
| sweep-fixed | 固定初态 Pick、预位、下降及扫动通过；撤离触发 execution_forbidden_contact，完整操作失败 |
| sweep-one | 随机 seed 39 在 Pick 的 close_contact 失败：缺双侧指垫接触并存在不允许接触；未执行推扫 |

这两个清扫失败只作本轮实际证据，不据此推断迁移前一定成功或失败，也不调参掩盖。撤离的原始证据为刷子碰簸箕刚性侧壁、穿透约 1.18 mm。失败视频 sweep-fixed/video/operation.mp4 为实际前缀 10.25 s，导出约 1.10 s，已完整解码并抽查末帧；完整任务视频 verified-full/video/task.mp4 为 87.33 s，已完整解码，抽查 40 s 旋钮与 84 s 关门画面，未逐帧验收。
独立源码审查核对了主要几何／规划函数迁移一致性及所有旧导入，发现的 Sweep executed 误标已有回归用例修复。迁移前相关回归 749 项通过，最终 Skill／Task／微波炉／清扫／真机离线／回放回归 **766 项通过**；杯子和 Pika 几何消费者另 15 项、纯规划消费者另 16 项通过，合计 **797 项**。测试和审查日志在 implementation/，没有用这些离线测试代替真机验收。

## 第三轮：执行事实与阶段调试接通（2026-10-31）

用户已授权。先以最小测试复现公共 runner 的 executed 误标，再区分已执行、未执行与证据未知；保留既有阶段顺序。接通微波炉 insert/place/extract-grasp/knob 的 stop_after：仅单项操作允许，准备前缀完整执行，暂停保存实际现场，完整目标未检查不得声称成功。非法范围在构建场景前拒绝。随后验证局部停止与原完整路径、真实 cuRobo/MuJoCo 运行，更新能力说明。临时证据在 runs/tmp/skill-library/20261031-fixes/；不扩展真机运动授权。

第三轮已完成：公共 runner 按阶段证据汇总执行事实，补充 unknown；纯检查用 observation_only 显式区分，声明禁用执行也不能盖掉已发生的执行证据。第二次开门从嵌套 report 提取共享结果。上述问题均先复现再修复；最终相关回归 813 项通过（final-regression.log）。

实际验证：insert-release-final 从 fresh reset 执行必要前缀，约 46.31 s 后停于 release_check；attachment 为空，没有 retreat 或关门，检查阶段均记 not_executed。knob-grasp 在夹持检查后停止，没有旋转／释放。两项均 status=partial、operation_success=true、task_success=null。verified-full 的原 13 步连续通过，约 154.55 s；最后的纯检查／嵌套报告标记修复由单元回归和最终 insert-release-final 实际运行验证，不据此增加物理成功范围。

暂停视频 insert-release-final/video/stop.mp4 来自 actual，时长 3.25 s，导出 0.64 s，已完整解码；抽查过前次同一释放停止场景的末帧。完整任务仍为辅助夹持＋运动学旋钮的混合仿真，没有真机运动；阶段状态 NPZ 仅供检查，不宣称跨进程恢复。
