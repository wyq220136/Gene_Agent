# 微波炉真机子任务入口

代码入口：`gene_agent/realworld/microwave/task.py`；现场装配集中在 `gene_agent/realworld/`，型号驱动位于 `devices/arms/realman/` 和 `devices/grippers/{realman,pika}/adapter.py`，详细职责见[真机代码说明](../../../gene_agent/realworld/README.md)。目录整理没有更改下列命令或执行参数。

当前状态：已有独立子任务和完整 Task 的现场接线；Viser 显式 90° 与 MuJoCo 完整任务已分别重跑，结果及仍存在的视觉差异见动态基线报告（历史材料已本地归档）。合成观测快照已接通真实 GPU cuRobo，验证了简单运动、门把手 `pregrasp` 和抓碗 `pregrasp` 三个独立规划阶段；**真机入口本身尚未完成现场 `plan-only`、标定或设备执行验收**。本入口默认只规划并核对首段，不发送运动命令。

先复制 [站点配置模板](../../../data/task_1/realworld/realworld_site.example.json) 为 `data/task_1/realworld_site.json`，填写两臂 IP、相机序列号、已注册的两个 FoundationPose session、实际注册的微波炉 mesh 路径、相机/机器人/资产坐标变换、桌面 pose 和子任务容差。模板中的桌面与微波炉支撑**尺寸**来自当前规划资产；桌面 pose、外参、mesh 对齐和现场容差必须在现场核对，不能采用空值或直接假设单位矩阵。模板已列出各子任务必填参数名，但故意留空未标定值；请逐项填写，不能把空值或示意容差用于现场动作。实际站点文件被 Git 忽略。

当前 task_1 的碗、旋钮及完整任务只验收右臂；`--check-config` 会在连接设备前拒绝这些任务的左臂配置。门子任务保留左右臂的参数入口，但左臂实际规划与真机动作也尚待单独验收。

模板的 `subtask_options.knob.task_path` 指向真机专用的 `knob_0_task_realworld_tuned.json`。它沿用当前 MuJoCo 完整任务的旋钮抓取方向（方位角 210°、局部 X 倾角 10°），不覆盖原 `knob_0_task.json`。合成场景验证了 grasp IK、`pregrasp` 和直线 `approach`，但静态夹爪侧视及碰撞距离检查发现：该目标下夹爪完全闭合仍距旋钮约 8.65 mm，**尚不具备已验证的夹持效果**。旋钮子任务目前只能作为逐段规划与现场调试入口；实际接触和转动必须另行验收，详见数值诊断（历史材料已本地归档）。需要复现旧六候选/末关节旋转分支时，可以在站点配置中显式指定原任务文件。

放碗与取碗的 `microwave_frame_alignment.T_scene_asset_profile` 也不能直接填单位矩阵：旧 keypose 所在 frame 与当前微波炉 URDF root 不同。一个合成持碗对照（历史材料已本地归档）表明单位矩阵会把 `preplace` 目标放到错误高度并使 IK 失败；旧视觉叠图变换只用于诊断，仍需现场标定后才能在配置中标为 `accepted_for_planning=true`。

`state_file` 指向一个由现场测量程序更新的 JSON 文件，至少包含：

```json
{
  "door": {"angle_rad": 0.0, "observed_at_unix_ms": 0, "source": "现场测角来源"},
  "knob": {"angle_rad": 0.0, "observed_at_unix_ms": 0, "source": "现场测角来源"}
}
```

实际运行前须把两个时间戳更新为真实 Unix 毫秒，并随门/旋钮运动继续更新；过期值会阻止现场场景构建。若 FoundationPose 注册的是会随门运动的把手/门 mesh，门打开后还须在同一文件提供**本次确认静止的柜体根位姿锚点**，否则观察器会拒绝用把手 pose 代表柜体。锚点结构为 `"microwave_root_anchor": {"T_world_scene_asset": 4×4 位姿矩阵, "verified_at_unix_ms": 本次确认的 Unix 毫秒, "source": "现场确认方式"}`；柜体移动后必须重测位姿，不能只刷新时间。文件格式与来源检查见[同帧观测说明](../../developer/perception/realworld_same_frame_tracking.md)。

在项目根目录和 `curobo` 环境中，先只检查静态配置：

```bash
python tools/run/run_microwave_task1_realworld.py \
  --config data/task_1/realworld_site.json --subtask pick_bowl --check-config
```

接着连接相机、Pose 服务与两臂，规划所选子任务的**首段**，并在新目录保存输入与规划证据；此命令不发送运动指令：

```bash
python tools/run/run_microwave_task1_realworld.py \
  --config data/task_1/realworld_site.json --subtask pick_bowl
```

`--subtask` 可选 `open_door`、`close_door`、`pick_bowl`、`insert_bowl`、`rotate_knob`、`extract_bowl`、`place_bowl_on_table` 或 `all`。`all` 调用与独立模式相同的子任务函数，默认也只预检第一段。检查所有任务时，配置中的 `subtask_options` 需填全相应参数。

单独调试时，操作者要先准备该段的真实前置场景；程序不会替你补做上一段，也不会把机械臂移回 home：

| `--subtask` | 人工准备的前置状态 | 执行后的关键检查 |
|---|---|---|
| `open_door` | 门关闭、把手可见，机器人与夹爪处于可测量的安全起点 | 拉把手后重新测门角，内侧推开后确认开门角 |
| `close_door` | 门已打开且内侧推门区域可达，夹爪未持碗 | 确认门已关闭；不会自动先抓把手 |
| `pick_bowl` | 碗在桌上，右臂夹爪空闲 | 确认夹住并抬起，规划 attach 不算真实夹持 |
| `insert_bowl` | 门已打开，右臂已夹稳碗；用 `--prepare-held-bowl` 现场确认，或提供本次复核的持碗记录 | 检查插入路径、松爪及碗留在炉内 |
| `rotate_knob` | 门已关闭，右臂夹爪空闲，旋钮角有新测量 | 检查实际夹持和转后角度；当前 profile 尚未通过接触验收 |
| `extract_bowl` | 门已打开，碗在炉内，右臂夹爪空闲 | 检查夹持、带碗撤离与碗确实离开炉腔 |
| `place_bowl_on_table` | 右臂已夹稳碗；像独立放入一样提供当前持碗证据 | 检查松爪、桌面支撑及撤离后碗仍留桌 |
| `all` | 门关闭、碗在桌上，所有状态和现场标定有效 | 每段从上一段实测终态继续；失败或后置状态未知即停止 |

这张表是现场准备清单，不代表上述动作已经通过真机验收。每次规划使用新的同帧观测、实测两臂关节和当前门/旋钮角；若前置状态与表中不符，应先重新准备现场，再运行对应子任务。

若只想按旧项目的无旋钮流程检查完整 Task，可在 `--subtask all` 后加 `--no-knob`。这只跳过 `rotate_knob`，保留第二次开门前的 Task 过渡；此时不必填写 `subtask_options.knob`。例如，先做静态配置检查：

```bash
python tools/run/run_microwave_task1_realworld.py \
  --config data/task_1/realworld_site.json --subtask all --no-knob --check-config
```

即使跳过旋钮动作，`state_file` 仍须提供带来源和时间的旋钮角，因为微波炉的关节状态仍参与现场碰撞场景构建。该选项仅改变真机入口，不更改原 Viser/MuJoCo 配置或完整任务默认值。

开门和抓碗的第一段都是开爪。默认 `plan-only` 会在开爪预检后停止，因此报告中没有后续手臂轨迹；这不是整项 Skill 已规划成功。要检查 `pregrasp`，需先在现场实际完成开爪并取得新的实测状态，再逐段规划；离线合成场景的单阶段 GPU 检查只用于验证接线，不可代替这一现场步骤。

独立调试 `insert_bowl` 或 `place_bowl_on_table` 时，由人先把碗夹好，再在交互终端加 `--prepare-held-bowl`。入口取得新现场帧并询问是否确实夹稳；回答 `y` 后把同一帧的证据写入本次运行的 `prepared_hold_report.json`，首段规划直接消费这帧。回答 `n`、`?` 或无法确认会停止；这只证明**人工准备的当前持碗状态**，不冒称本程序执行过 Pick。已有有效持碗证据文件也可通过配置中的 `prepared_hold_report` 指定，两种方式不能混用；即使只做首段预检，旧文件遇到新观测帧也会在交互终端重新要求确认持碗。示例预检命令：

```bash
python tools/run/run_microwave_task1_realworld.py \
  --config data/task_1/realworld_site.json --subtask insert_bowl \
  --prepare-held-bowl
```

现场逐段验收后，交互终端可以显式使用 `--execute --yes`；执行模式要求 `scene_drift_limits`，每次发运动或夹爪命令前重新观测并核对场景。自动检查给不出结论时，终端可输入 `y/yes` 表示现场确认通过、`n/no` 表示失败；空输入、`?` 或其他回答仍是未知并停止。自动判定失败不能靠输入改写，仿真证据不能充当真机通过。不会自动补跑前置任务或回 home。此命令**目前尚未在真实设备上验收**，首次使用应按保守接入计划（历史材料已本地归档）从无接触低速阶段开始。

记录默认保存在 `runs/realworld/<子任务>-<时间戳>/`；`site_input.json` 记录配置哈希和生效速度，阶段目录保存规划、碰撞和失败证据。`--check-config` 只检查字段，不连接任何设备，也不证明标定准确或 Pose session 仍有效。


## 放入与取出组合

`--subtask load_bowl` 复用 Pick→Insert，`--subtask unload_bowl` 复用 Extract→桌面 Place。两者从人准备好的当前场景开始，不自动开门、回 home 或执行完整 Task 前缀。要求实际门角处于 90±5° 的已打开范围；每段仍采集新的现场观测，并沿用原后置检查。独立 Pick、Insert、Extract、Place 入口全部保留。

```bash
python tools/run/run_microwave_task1_realworld.py \
  --config data/task_1/realworld_site.json --subtask load_bowl --check-config
python tools/run/run_microwave_task1_realworld.py \
  --config data/task_1/realworld_site.json --subtask unload_bowl --check-config
```

load 需要 `subtask_options.pick/insert`，unload 需要 `extract/table`。组合从本次抓取开始，不能使用 `prepared_hold_report` 或 `--prepare-held-bowl` 注入旧持物证据；人工持碗后仅调 Insert/Place 继续使用原独立入口。去掉 `--check-config` 默认只预检第一段，不把预测终态传给下一段。

组合实际执行仍使用原 `--execute --yes` 和现场检查机制；目前仅完成离线接线/配置/失败路径测试，没有新增真机实测成功证据。`task_result.json` 中 `requested_scope` 与 `operation_success` 标明组合结果，`task_success` 为 null，不代表完整任务完成。
