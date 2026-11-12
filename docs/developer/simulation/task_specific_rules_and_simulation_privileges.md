# 任务专用规则与仿真特权：使用清单

本文记录通用 Knowledge、Skill 和运动规划之外，为当前微波炉任务**显式加入**的操作规则。它们可以帮助快速开发，但必须知道规则来自哪里、在什么阶段生效、解决了什么问题，以及成功结论的边界。当前可复核运行是 `runs/acceptance/microwave_task1_mixed_20261013/`；这里的名义场景成功不代表位置泛化或真机成功。

## 先判断规则属于哪一层

| 层级 | 应放的内容 | 不应混淆的内容 |
|---|---|---|
| Knowledge | 由物体 Concept/部件参数生成把手、推门、抓取、旋钮等候选 pose | 不负责在具体场景里暗中移动物体或伪造接触 |
| Skill | pre-contact、接近、局部转角候选、撤离、阶段碰撞策略、实际终态判定 | 不把每个任务的固定过渡点写成通用几何知识 |
| Task profile | 特定物体实例的微波炉相对 keypose、场景摆位、Skill 间回默认 TCP 等衔接 | 不覆盖 Skill 已验证的阶段顺序与接触语义 |
| 仿真适配 | 接触后辅助固定、场景碰撞代理/支撑高度、视频相机 | 不把仿真辅助成功报告为自然物理成功 |

新增人工规则时至少保存：**来源和原因、坐标系与单位、适用对象/阶段、原始目标与修改后的目标、是否改变碰撞或物理、启用开关、失败时行为、独立验收证据**。修改场景或参数后，同一 seed 的旧运行只作对照，新结论必须来自新 reset。

## 已在当前名义任务中使用的规则

2026-10-27 用户在检查实际失败视频及无遮挡夹爪多视角图后，明确接受夹爪部分参与推门。当前 `data/task_1/simulation/door_physical_open.json` 的 `physics_push_allowed_gripper_contact_geoms` 在已有指根/正侧滑块基础上加入负侧滑块和两侧指杆；两侧指垫仍由原分类器自动识别。只对推开/推关阶段的门板接触生效，pre-contact/避障接近、其他机械臂部位及抓碗接触规则保持不变。原失败记录不重写；新规则下的完整任务需重新运行后才能报告结果。

| 规则 | 当前来源/数值 | 生效边界与必须检查的事 |
|---|---|---|
| 已标定的微波炉相对 TCP keypose | `data/task_1/knowledge/microwave_object_manipulation.json` 中 Bowl_0 的放入四点、取出五点；`gene_agent/applications/microwave/object_keyposes.py` 只做 frame 组合 | 这是人工标定的 task profile。场景微波炉移动后重算世界目标，不把世界坐标写死；仍须重新规划连续轨迹，并验碗口朝上、炉内支撑、取出净空。 |
| 阶段过渡点 | `gene_agent/applications/microwave/door_debug.py` 的推门前 pre-contact；Task1 的回默认 TCP，以及固定放入/取出 keypose | pre-contact 从**实际选中**的推门接触候选沿门面外法向后退，旧 Task1 默认 50 mm；门接触只能从 contact 阶段开始。回默认 TCP 是 Task 连接点，不是通用 Knowledge。不得用过渡点掩盖错误的接触姿态。 |
| 局部夹爪姿态候选 | `data/task_1/simulation/door_physical_open.json` 的 `local_rotation_search_defaults`，当前 x/y 各为 0、±5°、±10° | 为 IK/路径寻找较可达的姿态；只在该阶段配置允许时筛选，并记录选中角度、原目标与实际目标。不能默默改接触方向、直线要求或碰撞白名单。 |
| 场景和操作微调 | `data/task_1/simulation/mujoco_task1_physical.json`：微波炉 XY `(0.52, 0.29) m`、碗 XY `(0.255, 0.12) m`、抓碗深度 15 mm、放入深度偏移 20 mm、撤离局部 Y 旋转 -10°；转盘/炉腔地板分别下移 11/5 mm | 这些是当前**名义实例**的调参，不是新场景默认值。记录真实参考 pose、仿真 pose、有效初态和支撑/碰撞几何；取放轨迹、门关闭和最小净空要一起验。当前放入判定的炉腔边界最小余量仅约 2.3 mm。 |
| 仿真辅助夹持 | `simulation_attachment_pairs.right=bowl`；`gene_agent/simulation/runtime.py` 和抓碗、取碗执行代码 | 仅在真实 MuJoCo 双侧指垫接触通过后启用 weld；放置时释放，报告 `simulation_assisted`。它弥补当前代理/摩擦造成的滑动，不替代接近、闭爪、搬运轨迹或最终落稳判定。无辅助同状态复测能抬高约 68 mm，但保持时相对滑动约 31.6 mm，未通过稳定抓取。证据见 `runs/acceptance/microwave_task1_mixed_20261013/audits/`。 |
| 物理执行速度/刚度 | `data/task_1/simulation/door_physical_open.json` 的分阶段 `physics_speed_scale_by_phase`、`physics_arm_position_kp_scale`；任务 profile 的 `door_arm_kp_scale`、`extract_arm_kp_scale` | 只调 MuJoCo actuator 跟踪与执行速度，不应改变目标 pose、规划碰撞或成功阈值；逐阶段保存规划轨迹与实际 TCP 误差。当前 18 段推门均为 `linear_tcp` 规划，实际接触执行最大侧偏约 16 mm。 |

## 已实现或可配置，但不能误称已经验证

**近似 IK。** `data/task_1/simulation/door_physical_open.json` 对 push、pregrasp、pre-contact、retreat 等阶段配置了最多 **20 mm / 10°** 的预算，代码见 `gene_agent/motion/curobo/approximate.py`、`gene_agent/applications/microwave/door_debug.py`。流程应是：严格 IK 因目标无解时，检查失败种子；只接受数值有限、误差在预算内、关节和碰撞约束可行的候选；用候选的**实际 FK TCP pose**重新做正常路径规划和执行；报告原目标、替代目标、实际终点及误差。起点/目标碰撞、路径失败、关键接触终态失败不能靠“取最近解”放行。这次完整名义运行的门阶段 **没有触发近似 IK**，所以尚不能凭它证明该退让分支在完整任务中可靠。详见 [近似 IK 契约](../knowledge/approximate_ik_execution_design.md)。

**固定 fallback pose。** `door_debug.py` 可在关门 pre-contact 原目标失败后，尝试显式保存的 `close_pre_contact_fallback_pose_world`，并记录两次求解。这次完整运行未设置、未触发。它是 Task/Skill 的人工备用目标，不应伪装成 Knowledge 自动生成或原目标严格成功。

**旋钮运动学提交。** 当前 `physical_knob.py` 在机器人按原 Viser 轨迹执行后，如果真实接触未使旋钮转到目标角，会显式写入关节角，报告 `mixed_kinematic_knob`。本次实测闭爪后指垫离旋钮碰撞体约 8 mm，整个旋转无接触，提交前旋钮约 0°。这能检查任务顺序和视觉轨迹，**不能**证明物理拧动；也不能作为真机流程的隐含动作。

## 修改规则时的最小验收

1. 先看 Knowledge 候选和 Skill 阶段目标：接触方向、局部转角、keypose frame 是否符合操作演示；在 Viser/MuJoCo 看指尖与目标，不能只看 IK 数值。
2. 对失败明确分类：IK 严格无解、近似被拒、起点/目标碰撞、路径无解、轨迹跟踪偏差、接触或任务后置条件失败；保存碰撞对象、实际 FK、截图/短视频。
3. 只改变必要的摆位或参数，从同一初态做前后对照；通过后做 fresh reset 的完整阶段串接，再做事先声明范围的 pose 扰动。失败样本不静默丢弃。
4. 报告证据等级：`visual_plan`、`mujoco_physical`、`simulation_assisted`、`kinematic_follow`、`realworld`。一个 Task 可以混合这些等级，但总成功不能写成全物理成功。

当前完整任务的逐段事实见 [混合仿真验收](../microwave/evidence/mujoco_task1_mixed_acceptance_2026-10-13.md)；无辅助抓碗与旋钮接触数据见 `runs/acceptance/microwave_task1_mixed_20261013/audits/`。

## 清扫任务补充（2026-10-19）

清扫规则位于 `applications/sweep_cube/`、`knowledge/sweep.py`、`skills/sweep.py`，只针对当前刷子/方块/簸箕任务，不改微波炉默认配置。

- 辅助持刷：双侧指垫接触通过后启用原weld，清扫实例的 `solimp[:2]=.9999`；相对激活基准误差预算3 mm/5°。每阶段保存参数和实际误差，超限失败；松爪前释放。
- 碰撞：允许托盘接触不代表允许刷子穿侧壁。理想扫掠、cuRobo轨迹的完整几何复核、实际回放均检查侧壁穿透，阈值1 mm。接触下降只允许当前目标方块，扫动允许与三块相互作用并在每段后重读状态。
- 接触候选：真实mesh最低点至少高于桌面1 mm；实际凸包试探接触法向的水平投影与推进方向余弦至少 .97，另限制每个法向竖直分量绝对值不超过 .5。只做无状态副作用的几何预检，正式IK/轨迹仍由cuRobo完成。
- 操作中间目标：pan frame下侧偏超过15 mm且位于入口外的方块，先扫向 `X=max(当前X,.14), Y=0`；真实侧偏必须进入10 mm走廊后才继续轴向入盘。初始物体分布没有改动；中间移动由接触执行，不是reset或瞬移。
- 入盘目标：入口内保留当前方块旋转后投影半径加16 mm；撤离反向50 mm、上移50 mm。成功判定仍要求旋转后完整方块足迹在托盘内、发生托盘接触、停稳。
- 执行：接触下降/扫动在原轨迹执行成功后追加0.5s真实保持，并验证实际TCP误差不超过5 mm/.05 rad；非接触阶段保留原门槛和显式近似IK策略。不能靠等待挤过未授权接触。

原因、试验及尚未通过的范围见诊断记录（历史材料已本地归档）。
