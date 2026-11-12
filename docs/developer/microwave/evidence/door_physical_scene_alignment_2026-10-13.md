# 开门物理实验：场景版本与执行误差排查

本记录只讨论 MuJoCo 物理执行；Viser 轨迹可规划不等于夹爪已接触把手或门已打开。

## 已定位的问题

`DoorDebugSession` 对已存在的 `scene_run` 直接加载历史 XML。原物理入口默认指向 `runs/pick-bowl-planning-final`，其中的 `source_scene/robot/robot.xml` 是 10 月 6 日生成的旧文件，缺少现行场景生成器写出的机器人本体与安装关节接触排除规则。它不是当前代码重新生成的场景。旧场景在 `(microwave.x, y)=(0.60, 0.35)` m、把手竖直偏移 40 mm、pregrasp 与 grasp 速度比例 0.5 下，pregrasp 的关节终点误差为 0.122 rad，grasp 误差为 0.087 rad，实际 TCP 距目标约 20.7 mm；虽然执行器报告 `completed`，闭爪时只有单侧接触，后续拉门会失去接触。

保持同一配置和执行器，只把 `scene_run` 换为现行生成器构建的新场景后，pregrasp/grasp 的关节终点误差分别为 0.014/0.014 rad，grasp TCP 距目标约 7.7 mm，指垫实际接触把手。因此先核对 XML 和接触规则，再调 Knowledge 抓取深度。隔离工作区中的新场景与主项目旧场景所用的 mesh 路径不同，结构性差异是机器人安装接触排除规则；不能只比较 `scene.xml` 顶层哈希。

物理执行现在要求运动阶段的实际关节终点进入 0.05 rad 容差，最多等待 1.5 s；否则报 `endpoint_tracking_error`，不把“控制指令发完”继续当作“动作到位”。在旧场景中，pregrasp 等待后仍有 0.085 rad 误差，明确失败；在新场景中无需额外等待即通过。此改动是阶段判定，不修改 Skill 目标、规划轨迹或接触策略。

## 可重复的局部结果

新物理配置 `data/task_1/simulation/door_physical_open.json` 指向新的场景目录；首次运行自动构建当前 GT 场景。把手 Knowledge 只把 `vertical_offset_m` 设为 0.04 m；拉门目标沿用已通过的物理入口的 20°，pregrasp、grasp 和三段拉门的执行速度比例是 0.5。与旧视觉配置的差异均写在此独立配置中，旧视觉配置不改。

```bash
conda activate curobo
cd /home/ubuntu/Documents/yixuan/Gene_Agent
python tools/run/run_experiment.py
```

主项目验收记录 `runs/acceptance/door_handle_unified_20261013/report.json`：`handle_pull` 规划、物理执行和阶段目标通过，实际门角 **19.34°**。未采用门关节写值；把手接触来自 `right_gripper_negative_pad ↔ microwave_handle_grip_collision`。同目录 `physical_door_task.mp4`、`physical_door_gripper_side.mp4` 是按保存的实际 MuJoCo 状态录制。这是单个 Skill 的结果，不代表整套开门或完整任务成功。

此前同一配置尝试 `--selection microwave_door_open`，报告 `runs/acceptance/door_open_unified_20261013/report.json`：把手拉门通过；首段内侧推门把实际门角带到 **37.53°**，随后出现 `right_r_inspire_jaw_base_link_collision_1/3 ↔ microwave_door_panel_collision`，按禁止接触规则停在 `push_open/02_push_01`。同目录包含实际视频 `physical_door_task.mp4`、`physical_door_gripper_side.mp4` 和碰撞代理视频 `physical_door_gripper_side_collision.mp4`；侧视中可见指根代理贴近门板上缘。这是接触部位问题，不是该段 IK 失败。统一 Knowledge 当时选出的 `tip_y_deg` 为 0°；已验证的独立开门探针 `tools/diagnostics/run_microwave_door_open.py` 使用 `archive/task_1/configs/door_push_open_candidate.json` 的逐角推点（早期约 −90°）与 40° 后的抬离换点，曾达到 88.50°。

当时统一 Skill 显式读取 `archive/task_1/configs/door_push_open_physical_unified.json`，由 Knowledge 生成逐角目标，40° 后在 Skill 中抬离、换点、重新接触；物理 Runtime 在这三个阶段使用实测门角和 TCP 更新目标。原自动候选生成器仍保留，但当时未放宽指根碰门规则。原独立探针的 50 mm/70 mm 抬离在统一 cuRobo 碰撞场景下会于换点预接触报 `start_collision`；仅把换点抬离改为 **90 mm** 后通过。下方“用户审核后的开门接触规则”更新了默认选择。

主项目运行 `python tools/run/run_experiment.py`，验收记录为 `runs/acceptance/door_open_unified_success_20261013/report.json`：`status=completed`、`planning_success=true`、`execution_success=true`、`goal_success=true`；把手实际拉到 **19.34°**，内侧推门实际到 **92.23°**。两段均满足 0.3 s 稳定窗口、±5° 门角容差和指定指垫接触条件；所有 22 个阶段完成，未记录禁止接触。同行 `physical_door_task.mp4` 与 `physical_door_gripper_side.mp4` 从实际 MuJoCo 逐帧状态生成，用于核对动作。这个验收只证明**开门**，关门、放碗、取碗和旋钮尚需分别验证。

旧实验用 25° 拉门目标时，新场景中 `pull_25` 的目标 IK 未收敛；关闭场景碰撞仍失败，当前没有证据认定为物体阻挡。旧入口的物理成功目标实际是 20°，因此本次局部验收先使用 20°。后续如要恢复 25°，应单独做摆位/候选对照，而不是悄悄扩大 IK 误差预算。

## 连续开门→关门的下一处边界

完整开门后，旧 Task1 会从仍贴门的最后推点直线撤回默认 TCP。原物理判定把**出发瞬间**的任何门接触都当失败；本次实际仅有 4 个指垫接触采样、总法向冲量约 0.095 Ns，最后一次发生在出发后 0.114 s，门角变化 0.44°。现在仅对此过渡阶段允许前 0.15 s、冲量不超过 0.2 Ns、门角变化不超过 1° 的指垫残留接触；阶段记录增加 `departure_pad_contact_accepted`，非指垫、晚碰或明显带动门仍失败。实测 `runs` 前缀成功进入关门阶段；这是物理判定边界修正，没有改轨迹或操作顺序。局部记录：`/tmp/gene_agent_door_cycle_transition_fix_20261013/report.json`。

当前摆位 `(0.60, 0.35)` 下，关门接触 Knowledge 的 rank 39 候选通过 batch IK，但由它生成的 `push_close/01_pre_contact` 在**门碰撞开启**时无解；同一目标仅诊断性移除门板碰撞后可规划，名义轨迹的指根/滑块会穿门。把退让距离从 50 mm 改成 40/30/20/10 mm 均不解决；297 个候选在 50/30/20 mm 下均没有带门碰撞的 pre-contact IK。把微波炉 y 移到 0.335 m 虽可让单独关门预接触规划，却会让已验收的开门在换点后 `push_open/07_push_03` 无 IK，所以没有覆盖开门配置。

显式上方 pre-contact（相对 rank 39 接触点 `[0, -0.05, +0.10]` m）可从真实开门终态规划、执行，且不提前碰门；但从这里直线到**原接触 pose** 时，名义几何已经显示指节、滑块与门板重叠，MuJoCo 同样报 `forbidden_robot_door_contact`，并把门反向推至约 95°。视频在 `/tmp/gene_close_above_far_full_20261013/physical_door_gripper_side.mp4`，逐阶段数值在同行 `stages/push_close_02_contact.json`。对 297 个原接触候选做 batch IK 与 MuJoCo 接触筛选：44 个有 IK，**44 个都发生非指垫门板碰撞**。因此不能靠多试 IK 种子、放宽接触判定或单纯减速宣称关门成功。下一步需要在门外侧重新调关门接触姿态和路径，并用同一候选筛选流程联合检查 IK、实际夹爪接触部位和逐阶段门角；开门配置应保持为已通过的对照组。

另一次**仅诊断**运行临时让关门接触段继续执行，不修改项目碰撞规则，也不算验收：门在 `contact` 被推到 94.95°，首个 push 降到 74.15°，第二个 push 降到 48.10° 后报 `endpoint_tracking_error`；这一段同时出现夹爪指根碰门、指垫撞把手（约 84.6 Ns 累计法向冲量）。因此当前目标路径虽能让门向关方向运动，却伴随不合适的接触和跟踪失败，不能把关闭碰撞判定当作修复。记录在 `/tmp/gene_close_contact_policy_ablation_20261013/report.json`。

## 用户审核后的开门接触规则（2026-10-13）

用户检查了约 37.5° 停机时的侧视视频，确认该次指根碰门属于可接受的推门接触；随后又确认 Viser 默认摆位下 `right_gripper_positive_carriage` 碰门可以接受。此前将所有非指垫接触判为失败、继而换用大幅旋转的逐角物理 profile，是错误的验收方向。现在默认 `door_physical_open.json` 不再加载该 profile；旧结果及参数保存在 `door_physical_profile_probe.json`，仅作对照。

物理推门阶段可显式列出允许碰门的夹爪 geom，本场景允许两处实际出现的指根代理、一处用户确认的 carriage 和原有指垫。报告仍分别记录 `pad_target`、`gripper_target` 和 `forbidden`，pre-contact 仍不允许提前碰门，其他机械臂部位仍保留异常诊断。规划副本可以为 Viser 显示名义门角，但在线 MuJoCo 门关节只由接触动力学推进；阶段边界与规划前后均检查实际门角不被重写。

一次使用**原自动 Knowledge 推门候选**、物理摆位 `(0.60, 0.35)`、拉门目标 20° 的回归运行在 `/tmp/door_original_pose_jaw_allowed_gpu/`：15 个阶段完成，拉门实际 19.34°、推门实际 92.71°，所有相邻阶段门角差 0°。实际接触记录含两处指根及两处指垫，最大指根穿透约 0.24 mm；全景与夹爪侧视视频由保存的 MuJoCo 状态生成。这是对原推门 Knowledge 的物理回归，并非下述 Viser 默认摆位的验收。

随后在 Viser 默认摆位 `(0.52, 0.29)`、拉门目标 25°、无把手 Knowledge 覆盖值下运行。首次于 40.52° 发现 `right_gripper_positive_carriage` 碰门，用户查看侧视视频后确认可接受。只将该 geom 加入推门允许名单，不改推门 pose；门实际到 89.84° 时，原判定把刚结束的运动时段也算进末尾 0.3 s 稳定窗口，报 `angle_not_stable`。在最后一段后使用已有 `execute_hold` 保持 actuator 目标 0.35 s、继续真实物理采样，未放宽 ±5° 与 0.3 s 标准。`/tmp/door_visual_25_hold_run/` 最终记录：拉门实际 **23.06°**，推门实际 **90.24°**，两个后置检查通过，所有阶段边界门角差 **0°**；运行视频含全景、夹爪侧视和碰撞代理侧视。该结果只验收开门，尚不代表关门或完整任务通过。

## 连续关门曾经的失败边界

以新默认开门配置从 fresh reset 执行开→关，记录在 `/tmp/door_cycle_after_open_baseline_20261013/`。开门通过、回默认 TCP 通过，随后 `push_close/01_pre_contact` 由实际 90.25° 出发，夹爪负侧指垫提前碰门，把门推至 110.31°，按 `unexpected_precontact_door_contact` 停止。该接触累计法向冲量约 11.96 Ns，不应因“指垫允许碰门”而放行：它发生在尚未开始关门接触的阶段，且门向错误方向多开约 20°。全景、侧视和碰撞代理视频都在同行目录。

该阶段的名义规划轨迹经 MuJoCo 几何复核无接触，对门板的最小净空约 11.7 mm；实际执行时最大关节跟踪误差为 0.124 rad。仅把该段物理速度比例从 1.0 降到 0.25、0.10 后，最大跟踪误差约 0.043 rad，但门仍被提前推开约 20°。这些记录是下述修复前的失败证据。

## 默认摆位完成一轮 MuJoCo 物理开关门

关门前检查点试验显示，保持原 50 mm pre-contact、原 Knowledge 候选和轨迹不变，仅把该阶段的 MuJoCo arm position `kp` 比例从 2.0 提到 4.0，提前碰门采样从 132 次降为 0 次，关节最大跟踪误差由约 0.124 降到约 0.061 rad。正式关门接触初段会出现另一块指根代理 `right_r_inspire_jaw_base_link_collision_2` 与门板轻微接触；它与用户已接受的指根推门方式同类，仅在推门阶段纳入允许名单，pre-contact 仍禁止任何提前碰门。

未调整接触点时，门从 90.25° 连续关到 47.91°，随后夹爪靠把手过近，在 `push_close/05_push_03` 出现把手接触。沿用原 Knowledge 的 `hinge_fraction_offset`，将关门接触点向门轴侧移动 0.2 个门宽；此前测试的 −0.1 只降低了把手接触冲量，仍未完全避开。−0.2 的配置在 [关门参数](../../../../data/task_1/simulation/door_push_close_physical_tuner_state.json) 与 [物理任务配置](../../../../data/task_1/simulation/door_physical_open.json) 中。没有修改开门 Knowledge、阶段顺序、门角目标、碰撞放宽范围或 cuRobo 算法。

最终从 fresh reset 执行 `python tools/run/run_experiment.py --selection microwave_door_cycle --output runs/door-cycle-check`：25 个阶段全部完成，`handle_pull` 实际 **23.06°**、`push_open` 实际 **90.24°**、`push_close` 实际约 **0°**，三个门角后置检查均通过，阶段边界最大实际门角跳变 **0°**。关门 pre-contact 碰门采样 **0**，没有禁止接触，也没有关门阶段的门把手接触。同一默认配置又在 Gene_Agent 主目录从 fresh reset 独立复跑通过；主目录的原始报告、逐阶段轨迹/接触和双视角视频保存在 `runs/acceptance/microwave_door_cycle_physical_main_20261013/`。这是一轮确定性场景验收，尚未验证扰动分布、放碗/取碗/旋钮的物理串联或真机。
