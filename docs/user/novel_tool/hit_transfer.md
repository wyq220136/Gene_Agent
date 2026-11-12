# Hit实例迁移与复用

本轮迁移 `mallet_train`（木槌）与 `hammer_00003`（RoboDojo hammer/model3）。原环境、ALOHA、初始化、物理、专家筛选与评分均保留；实例成功只按原评分报告。

## 代码和输入放哪里

- 共享抓取候选：`gene_agent/knowledge/`；共享动作：`gene_agent/skills/library/pick/`、`hit/`。
- 原生策略仍为 `gene_agent/tasks/policies/novel_tool_task/`，场景/状态/控制仍为 `gene_agent/simulation/robotwin/`。同类实例不复制 Task 或 Skill。
- 实例输入：`data/benchmarks/novel_tool/<instance>/`，包含 actor frame 的 `concept.json`、有效 grasp profile、candidate；均忽略 Git。
- 场景、标注叠图、调参脚本、冻结请求、试验与评测：`runs/tmp/novel_tool/hit-transfer/2026-11-03/`，忽略 Git，可按轮次删除。

## 本轮必须区别的地方

木槌柄在桌面上方，头部两端几何对称；使用朝下端面避免无必要的翻转。抓取点沿柄向头部移动25mm后，开发样本中的相对滑移从10.7mm降到0.6mm以内，未修改持握验收阈值。

新锤子柄较低且略弯，Concept的椭圆柱只表达功能抓取区，实际规划始终保留原碰撞几何。候选通过沿柄位置、径向深度和腕部角度调节桌面净空，不修改碰撞形状或摩擦。

锤子最终保留6个候选。右臂抓取沿实际细柄向头部移20mm后通过原柄部抓持判定；工作面内6mm偏移后，开发样本原首触、钉入与撤离均通过。该偏移在真实锤面内，不来自原专家作用点。面中心对齐时锤面完全落在钉帽内，接触manifold可能只含面角点；首触不被原工作端判定接受时，后续受阻转动不能直接归因为摩擦不足。此处是本实例的已验证现象，不把6mm偏移写成其他工具默认值。

`hammer_00003/candidate-final.json`引用扩展后的`concept-expanded-handle.json`与冻结的`grasp_profile-final.json`；第0个保留候选来自扩展前同一自主Concept。`candidate_provenance.json`逐候选记录实际生成Concept与哈希，6个候选均独立重建核对，pose矩阵最大误差0。不要仅用当前Concept重新解释历史profile。

`hammer_00003` 的完整判据在父类 `CalibratedNailTool.check_success`，还要求敲击后持物撤离。共享 Hit 因此支持可选 `withdraw`：实例 `hit_parameters.withdraw_distance_m` 显式给正米数时启用，默认0关闭，旧hammer仍为align→strike。撤离从执行后的实际TCP沿当前目标+Z规划直线，保留碰撞检查和实际持握检查；只允许工具/钉子的预期接触。原评分提前中止轨迹时仍记录partial，不伪造到达撤离终点。

新实例开始前应展开完整评分的继承/组合调用，逐项核对抓持、工作端、抬升、敲击与撤离；只看子类名称或Skill执行成功会漏验。策略只用本项目标注的功能几何、当前GT pose和本体状态，不读取隐藏评分/专家点生成动作。

## 证据与命令

木槌已完成原evaluator 20例（seed参数2，从300000起按原专家筛选接纳）：**17/20（85%）**。失败为2例Hit对齐规划、1例抓取预检。参数在本批开始前固定，未中途调整，失败全部计入分母。`runs/tmp/novel_tool/hit-transfer/2026-11-03/mallet_train/eval20-final/`保存原`_metadata.json`、`_result.txt`、20个episode和原生视频；`evaluation_freeze.json`保存实例参数哈希。

该批使用可选撤离接入后的冻结代码。结束后另修复Hit旋转超限分支的numpy布尔值序列化，并新增可选`strike_time_scale`（默认1，不改变旧动作速度）；木槌冻结输入没有启用该参数，不将本批结果描述为修改后的重新20例验收。`source/mallet-eval/`保留本轮涉及的三个源码文件旧版本。

新锤子已完成原evaluator 20例：**10/20（50%）**。冻结输入为6个候选的`candidate-final.json`，预定开发初态0、7、21均通过原完整评分，但分布表现明显不足。原evaluator按seed参数2/3/4分别运行7/7/6例，三个不重叠seed范围完整合并；各原`_metadata.json`分别为4/7、4/7、2/6，参数和三个共享源文件哈希在评测期间保持不变。证据在同轮`hammer_00003/eval20-final/shard_seed_{2,3,4}/`，未将开发成功计入正式分母。

新锤子10例失败：6例`strike/tool_postcondition_failed`、1例`strike/tracking_error`、2例`strike/constraint_violation`、1例`withdraw/goal_ik_no_solution`。6例工具终态失败中原钉入状态均记录`upward_release`取消，工具相对TCP末态转动约25–26°；目前只证明失稳事实，尚未把所有初始原因查明。两例路径复核的TCP横向偏离约12–13mm，大于5mm上限；保留拒绝，不放宽直线标准。跟踪失败的关节终点误差约0.061rad，大于0.02rad上限。逐例状态和原评分见`hammer_00003/final_failure_summary.json`。

后续重点是扩大持稳抓法的覆盖，并在有限候选筛选中核对从对齐到敲击、撤离的可行性。当前筛选通过不能保证这些完整路径和实际持握均成功；不能把上述50%描述成稳定完成或所有Hit工具已迁移。

原环境达到完整成功后会提前结束动作。此时`native_score.success=true`而Hit最后阶段仍可能是`partial/native_terminated`；前者是完整题目的原评分，后者表示撤离轨迹未走到本项目请求的终点，两项都保留，不互相改写。

## 复用和开发成本

与本轮开始时126个相关源码的哈希对照，只有三个共享源文件变化：`skills/library/hit/skill.py`增加可选撤离，`novel_tool_task/execution.py`接入参数，`hit_backend.py`实现撤离及上述失败记录/速度参数。Pick、Knowledge、motion与原生环境adapter没有按工具新增或复制实现。35项相关回归通过；默认Hit仍为两阶段，原有不启用撤离的调用保持。

从开始记录到木槌参数冻结约38分钟，到新锤子参数冻结约1小时58分钟；两个实例交错开发，不能相加当总工时。前者完成4次、后者完成24次fresh-reset开发试验（含失败与回归），另有离线IK/路径筛选。数字来自本轮文件与运行时间，包含仿真和等待，不是纯写代码时间；主要成本是低位抓取、持物碰撞与首触诊断。标注由Sol完成并由主Agent检查，初始标注为数分钟量级，未单独记录精确纯人工时长。正式评测耗时另计。

木槌20例顺序评测约49分钟，新锤子三个批次并行约9.5分钟，均包含原专家筛选、原生物理、渲染和进程启动，不能当作纯规划延迟。`planning_summary.json`的每episode规划总量中位数分别约55.5s/18.6s，候选筛选墙钟约43.3s/17.2s，动作规划等待约10.9s/3.6s；这些统计有包含关系，不能相加。规划总量包含冷初始化、失败与筛选，不能将单次热求解时间冒称整个任务的规划开销。

这轮说明相同Skill/Task可复用，但几何参数迁移仍有明显调试成本，不能据此承诺所有Hit工具只改一份参数即可高成功率。

## 本机运行

### 阶段IK与失败优先迭代

原生Pick现在实际消费`skill_parameters.pregrasp_approximate_ik_budget`，预检与执行规划使用同一预算；Hit支持`hit_parameters.approximate_ik_by_stage`。默认保持严格，按任务语义显式配置，不能只根据求解器默认容差推断已经启用。

| 阶段 | 当前原生接入 | 额外检查 |
|---|---|---|
| pregrasp | 可配置位置/转角预算 | 按实际FK连接严格直线approach；实际TCP仍须在预算内 |
| approach、grasp、strike | 严格 | 抓取/作用点与接触要求保持 |
| align | 可配置预算 | 实际作用点横向≤8mm、法向误差≤0.12rad且位于目标上方 |
| withdraw | 可配置预算 | 位置预算小于请求撤离距离；实际向外位移和表面净空必须通过 |

例如在既有candidate中合并以下可选字段，单位m/rad。此处是配置写法，不能当作所有工具的推荐误差值或成功证据：

```json
{
  "skill_parameters": {
    "pregrasp_approximate_ik_budget": {"max_position_error_m": 0.01, "max_rotation_error_rad": 0.1}
  },
  "hit_parameters": {
    "approximate_ik_by_stage": {
      "align": {"max_position_error_m": 0.008, "max_rotation_error_rad": 0.1},
      "withdraw": {"max_position_error_m": 0.01, "max_rotation_error_rad": 0.04}
    }
  }
}
```

近似IK使用安全候选的实际FK重新规划，保留原目标、替代目标、误差/预算和实际终态。不会放宽碰撞或把失败关节解直接执行。候选预筛保留未严格收敛的对齐候选时，只是推迟到实际规划验证，不宣称已经安全可执行。

每轮先固定原失败案例，记录猜想与单一改动；规划问题先重放保存请求，物理问题再从原reset执行。比较入口与返回值见[阶段精度与失败回归](../../task_development/simulation_iteration.md#阶段精度与失败回归)。原生新run的`development_summary.md`列出实际IK策略，并直接给出比较命令；没进入目标阶段就是未验证。原失败改善后做少量成功案例回归，再冻结新样本评测，旧验收失败用于调参后不再作为独立验收。

本轮固定失败`hammer_00003 / 500039`在100mm撤离时目标IK失败；显式10mm位置预算未修复，候选约束检查仍失败。改为90mm撤离后，严格cuRobo规划及原生评分通过，初始物体pose和双臂关节与旧案例完全一致。两个原成功案例`500034 / 400019`也保持原评分成功；这两个案例用右臂，修复案例用左臂。本轮仅作开发回归，没有新的20例成功率。新参数在本机`data/benchmarks/novel_tool/hammer_00003/candidate-withdraw-90mm.json`，原`candidate-final.json`及历史结果保持冻结；下面完整任务命令可改用该新candidate。记录在`runs/tmp/novel_tool/hit-stage-ik/2026-11-03/`，以其`acceptance_report.json`和原评分引用为准。

原环境可能在成功时抛出`native_terminated`，本轮三例成功都保留该事实；不得据此声称本项目撤离后置检查已完成。木槌的对齐失败仍未修复；近似IK不能自动消除自碰撞或持物失稳。

木槌另用比严格阈值更宽的8mm/0.1rad预算离线复核两个冻结失败请求，也未得到可用规划。这是有限候选/请求的结果，不是不可达证明；下一步应检查安全IK分支与对齐目标，而非继续无依据放宽误差。

从项目根目录运行；实例输入和私有依赖被清理后，需要按记录重新准备，不把本机路径当作开源默认依赖。

```bash
native_python=/root/miniforge3/envs/policy/bin/python
planner_path="$PWD/runs/tmp/native-benchmark/worktree/runs/tmp/native-benchmark/deps/mirror-site:$PWD/runs/tmp/native-benchmark/worktree/runs/tmp/native-benchmark/deps/curobo"

# 原生交互场景；需要可用的DISPLAY，去掉--viewer可离屏保存场景截图。
"$native_python" tools/run/run_novel_tool.py \
  --external-root /inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yixuan/tool_use \
  --family hit --instance hammer_00003 --mode scene --seed 0 --viewer \
  --python "$native_python" --output runs/tmp/hit-check/hammer-scene-01

# 完整任务；mallet_train替换instance及candidate所在目录即可。
CUDA_VISIBLE_DEVICES=0 "$native_python" tools/run/run_novel_tool.py \
  --external-root /inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yixuan/tool_use \
  --family hit --instance hammer_00003 --mode evaluate --operation hit \
  --candidate data/benchmarks/novel_tool/hammer_00003/candidate-final.json \
  --test-num 20 --seed 2 --python "$native_python" --planner-python "$native_python" \
  --planner-pythonpath "$planner_path" --save-motion-requests \
  --output runs/tmp/hit-check/hammer-eval-01
```

输出目录须是新目录。独立Pick检查沿用同一命令的`--operation pick --stop-after pick`；失败只读入口为`tools/diagnostics/inspect_native_run.py --run <run>`，规划耗时加`--performance`，不重新运行环境。正式结果索引在本轮`acceptance_report.json`，包含原评分、失败分类、媒体和完整分母；最终验收检查命令为本轮临时脚本`summarize_migration.py --require-complete`。

本轮原视频检查记录为`media/mallet-media.json`与`media/hammer-media.json`，保存实际MP4路径与采样时间；木槌/锤子原视频分别为13.3s/14.5s、320×240、10fps。已实际打开成功片段的抽样帧和开发末态原生相机图；锤子MP4全帧解码通过。原头部相机在撤离末段可能裁掉部分工具，不能仅凭末帧推断操作全程；任务事实仍读原评分。部分原manifest保留重命名前的`episodeN.mp4`，汇总按原evaluator实际`episodeN_success.mp4`/`_fail.mp4`解析，未改写旧run。
