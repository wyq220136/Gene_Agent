# Hammer：原生抓取与敲击

直接使用外部 `tool_use` 的 `hammer_train / beat_nail_hammer / model_id=0`、SAPIEN/RoboTwin 和 ALOHA。动作顺序为共享 Pick（张开、预抓取、直线接近、闭爪检查、抬升、保持）→共享 Hit（对齐、直线敲击）。原初始化、物理、预算、专家筛选与评分保持不变。

## 输入与代码位置

| 内容 | 位置 |
|---|---|
| 可复用 Hit 作用面几何 | `gene_agent/knowledge/operation/hit.py` |
| 共享阶段逻辑 | `gene_agent/skills/library/pick/`、`hit/` |
| 本 benchmark 的策略、执行和后置检查 | `gene_agent/tasks/policies/novel_tool_task/` |
| 原生状态、控制、场景碰撞 | `gene_agent/simulation/robotwin/` |
| 规划附件和 cuRobo 接线 | `gene_agent/motion/curobo/` |
| 本机实例输入（不追踪） | `data/benchmarks/novel_tool/hammer_train/` |
| 本轮实验、失败记录与验收（不追踪） | `runs/tmp/novel_tool/hammer/2026-11-02/` |

本机 `concept.json` 已标注 handle Cylinder 和 strike-head Cuboid；`grasp_profile.resolved.json` 记录共享 Cylinder Knowledge 的参数和 TCP 基变换，`candidate-concept.json` 是最终有效候选。标注来自原物体碰撞几何的测量与叠图检查，没有使用原专家 grasp/functional point。手柄抓取位置相对 Concept 中心沿轴偏移 −20 mm；闭爪指令 12 mm，属于本实例参数，不应写成其他工具的默认值。

策略只读当前物体/目标 pose 与机器人状态等允许观测，GT pose 替代感知。基于观测选择靠近目标的机械臂。内部接触、seed、专家筛选和评分信息只用于诊断/原评测，不能输入动作策略。规划附件只保护碰撞；没有物理 weld 或修改摩擦。

## 运行

从项目根目录执行，输出目录必须为新目录。解释器与 pinned cuRobo 配置见[原生流程](../../task_development/native_benchmark.md)。实际命令（包含本机完整路径）保存在本轮 `evaluation-*-command.json`。

```bash
<native-python> tools/run/run_novel_tool.py --external-root <tool_use> \
  --family hit --instance hammer_train --mode evaluate --operation hit \
  --candidate data/benchmarks/novel_tool/hammer_train/candidate-concept.json \
  --test-num 3 --seed 1 --python <native-python> \
  --planner-python <planner-python> --planner-pythonpath <planner-path> \
  --output runs/tmp/novel_tool/hammer/<new-run>
```

独立检查 Pick 时改为 `--operation pick --stop-after grasp` 或 `--stop-after pick`；这仍是同一个共享实现。`hit` 自动执行完整 Pick，不受默认 `stop-after=pregrasp` 影响。原生场景查看用同一入口 `--mode scene`，有本机显示时加 `--viewer`。已有运行可用 `tools/diagnostics/inspect_native_run.py --run <run> --stage <stage>` 检查；视频直接打开原 evaluator 的 MP4，不转换到 MuJoCo。

输入目录可被清理：每个 episode 的 `geometry_inputs.json` 保留 Concept/profile 内容与哈希，`candidate.resolved.json` 保存有效候选；恢复时先核对这些输入和源资产版本，不能仅凭此说明重新猜参数。

## 结果解释

完整任务以原 `_result.txt` / `_metadata.json` 和 `native_score.json` 为准，`evaluation_index.json` 连接案例、seed、阶段和原视频。原 evaluator 的专家成功日志不是本策略成功。环境成功提前终止 strike 时，动作层保留 `native_terminated`，不能改写为到达轨迹最后一点。

原 nail 评分是首次物理接触后按受控区域与下压距离计算的原规则，并非持续接触/木材穿透仿真。成功只说明完成这份原 benchmark；不外推其他八个物体、其他任务、任意初态或真机。

最新数值、视频与回归证据见本轮 `acceptance_report.json`；运行文件若已清理，应重新评测而非引用历史成功作为当前证据。


## 本轮已验证结果（2026-11-02）

冻结同一份代码/参数后，原 evaluator 开发集 seed=0：**2/2**（100000、100001）；未参与调参的验收集 seed=1：**3/3**（200000、200001、200002）。两组无专家拒绝、unstable 或 error seed，保留原筛选与完整分母。此前开发失败仍保留，未合并成最终成功率。

25 次正式规划均记录 `backend=curobo`、`planner_mode=cuda_graph_benchmark` 和离散轨迹碰撞检查通过。Pool 保持运行，但本次各阶段的配置/持物/接触策略变化导致重建，实测 cache hit=0；不宣称已有缓存加速收益，也不把离散检查称为连续碰撞证明。

相关回归 **477 passed**；6 项原有失败单列，其中新增发现的历史门候选哈希失败已在旧 worktree 复现。该数字不是全仓库无失败。记录为本轮 `regression-final.log`、`old-worktree-knowledge-failure.log`。

五段原 MP4 均解码通过；已实际查看开发集首例的抓取、抬升、对齐与最终敲击帧。原 head 视角为320×240、10fps，接触细节仍需结合原状态。便捷副本为 `runs/tmp/novel_tool/hammer/2026-11-02/hammer-operation.mp4`；源文件及每个案例的视频对应保留在两组 `evaluation_index.json` 和 `acceptance_report.json`，没有重跑动作或转码。


## 扩大样本后的冻结评测（2026-11-02）

同一 hammer 策略在新 seed=2 的原 evaluator **20例中成功14例，成功率70%**。此前5/5是小样本历史结果，不能替代本次较大样本。95% Wilson 区间约48.1%–85.5%，仍不代表其他物体或任意场景的成功率。

六次失败均为 cuRobo 报告 `goal_ik_no_solution`：align 4例、approach 2例；这表示求解器未找到有效目标解，不等同于已证明目标理论不可达。策略、候选和原benchmark未因失败调整，原筛选和完整分母保留。正式评测墙钟约25.9分钟，运行于共享负载机器；不是纯规划耗时。

证据：`runs/tmp/novel_tool/hit-expansion/2026-11-02/hammer_eval20_report.json`，对应原 `_metadata.json`、每例 `native_score.json`、`evaluation_index.json` 和代码/源benchmark哈希均已核对。后续优先复现这些失败、完善共享候选/规划稳定性，再扩展其他Hit物体；修复后另开新seed组评测，不覆盖本轮70%的记录。

## 抓取与敲击候选的开发入口

原生策略可选读取 `grasp_candidates`（每项为共享 grasp Knowledge 生成的 `T_object_tcp` 和来源 profile）。不提供时沿用单候选行为。`screen_approach_path=true` 在移动前检查到预抓取点的路径及从该规划终态出发的直线接近；实际执行仍从实测状态重新规划，不把预检当成动作成功。`search_other_group=true` 仅在本臂全部候选预检失败、尚未移动机械臂时检查另一臂；实际接近/夹持失败正常终止。

Hit 参数 `alignment_yaw_offsets_rad` 指定保持作用点、法向及净空的对齐朝向。只在规划失败且没有执行、状态未改变时尝试下一个朝向。`screen_yaw_offsets_rad` 用于抓取前的预测对齐 IK 筛查；预测没有已抓住工具的真实相对位姿和 attachment，不能代替抓取后的持物规划及后置检查。

每例 `candidate_screen_<group>.json` 保存原始候选、batch IK 结果和路径预检；`candidate_selected_<group>.json` 保存实际选中的候选。`events.jsonl` 的 `alignment_candidate` 与随后 `planning` 记录解释敲击朝向的选择。实例参数和开发记录放在忽略目录，不在共享 Skill 写入 hammer 专用坐标。上述入口的完整成功率以对应冻结版本的原 evaluator 记录为准。

`allow_finger_tool_contact=true` 显式允许抓取目标 IK、直线 approach 及其路径预检中的**指定手指 link 与当前工具**接触，默认关闭；pregrasp 和预测敲击对齐目标仍严格检查。抓取与对齐 IK 分批使用各自阶段的碰撞策略，结果按候选原顺序保存。手指 link 来自机器人 URDF 的指关节 child，属于整指代理白名单，不能称为仅指垫接触。工具碰撞几何仍留在 world；复用现有局部接触 validator 检查其余外物、掌部/机械臂与工具、自碰撞及路径样本，球半径与余量不变。该规划许可不替代实际闭爪、抬升、保持及任务评分，也不能把执行滑移改判成功。

`screen_held_lift=true` 与路径预检一起启用：从直线接近的规划终态，按候选的理想夹持关系建立原工具碰撞球，检查抬升路径；失败时尚未执行，可继续候选或另一臂。这里保留当前张爪几何，不预测真实夹紧/滑移，不能作为实际持物成功证据。真正闭爪后仍用实际物体关系和实际开口重新规划抬升。
