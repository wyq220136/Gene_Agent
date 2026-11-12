# 长柄刷清扫方块：MuJoCo Task

在仓库根目录、可用 GPU 的 `curobo` 环境运行：

```bash
python tools/run/run_sweep_long.py --output-dir runs/sweep_long/my_nominal
python tools/run/run_sweep_long.py --layout-seed 7 --layout-distribution wide \
  --output-dir runs/sweep_long/my_seed7
```

输出目录必须尚不存在。默认场景从 `data/realworld/my_scene_0/scenes/sweep_long` 生成**运行专用**任务场景；来源场景和 Concept 不被覆盖。长刷仍竖立在原支架内。簸箕开口朝世界 −X，三块方块从远到近排在开口前方；簸箕和方块相对最初尝试沿世界 X 移近机械臂 4 cm。有效名义位置为绿块 `(0.30, 0.10)`、红块 `(0.34, 0.10)`、蓝块 `(0.38, 0.10)`、簸箕 `(0.48, 0.10)` m。运行目录的 `task_scene/scene.xml`、`scene.json`、`source_mapping.json` 记录生效摆位、原场景哈希、对象/Concept 名称映射和参数。刷头使用**来源场景原始 box/cylinder 碰撞尺寸**。

给定 `--layout-seed` 时，默认的 `wide` 分布对三块方块分别采样世界 X `[0.285,0.375] m`、Y `[0.075,0.125] m` 和原朝向增量 `[-60°,60°]`。采样只为避免方块中心小于 30 mm 而重试；不根据规划结果挑选样本。方块的颜色次序和横向错位均可变化，刷子、支架和簸箕固定。[六种分布近景预览](../../../runs/sweep_long/wide_reachable_preview_20261020/gallery_close.png)可先核对摆位；每张图对应 seed 0–5 的同一运行场景。托盘允许横向偏差 45 mm 内的方块直接沿开口方向推入；若纯规划预检失败且现场未动，再尝试归拢。盘外方块的补扫也会检查归拢路线。所有范围、seed、采样 pose 和重采样次数保存在 `task_scene/source_mapping.json`，落稳后实际 pose 与初态有效性保存在 `result.json` 的 `placement`。

`--layout-distribution small` 重现此前的窄分布：每块 `dx∈[-5,5] mm`、`dy∈[-10,10] mm`、yaw 增量 `[-10°,10°]`。`stress` 保留 Y `[0.06,0.14] m` 的侧缘难例，[预览](../../../runs/sweep_long/wide_preview_20261019/gallery_close.png)和失败 run 可用于进一步开发，不计入 `wide` 分布的成功率。省略 seed 使用名义摆位。`--source-layout --layout-shift-x-m 0` 是来源场景对照，不属于已验收分布。

标准 viewer 回放同一 MuJoCo 实际执行记录：

```bash
python tools/viewers/view_skill.py --skill pick_brush_long \
  --run runs/sweep_long/postcheck_nominal_20261019
python tools/viewers/view_skill.py --skill sweep_long_brush \
  --run runs/sweep_long/postcheck_nominal_20261019
python tools/viewers/view_task.py --task sweep_long \
  --run runs/sweep_long/postcheck_nominal_20261019
```

可直接看[名义场景录像](../../../runs/sweep_long/postcheck_nominal_20261019/sweep_actual.mp4)、[初态图](../../../runs/sweep_long/postcheck_nominal_20261019/renders/initial.png)和[终态图](../../../runs/sweep_long/postcheck_nominal_20261019/renders/final.png)。三个小扰动初态：[seed 0](../../../runs/sweep_long/seed0_inset2mm_20261019/renders/initial.png)、[seed 7](../../../runs/sweep_long/seed7_inset2mm_20261019/renders/initial.png)、[seed 21](../../../runs/sweep_long/seed21_inset2mm_20261019/renders/initial.png)。这些图来自本机实际 MuJoCo 场景，便于核对每块方块和簸箕的布局。

任务使用世界 −Z 的直线抓取接近、cuRobo 正式运动规划和 MuJoCo 物理执行。双侧指垫真实接触刷柄后启用显式仿真辅助 weld，拔出 8 cm；松爪时解除。刷头沿窄边推动方块。绿块的一次扫动可顺带推动红、蓝块，但最终分别用真实足迹完全入盘、托盘接触和停稳判定三块；没有直接移动或附着方块。额外深入目标 2 mm 是为覆盖入盘边界误差，任务成功不以期望目标 pose 代替实际终态。

[原小范围参数评测汇总](../../../runs/sweep_long/final_inset2mm_20261019/evaluation.json)：名义场景及 seed 0、7、21 均通过，**4/4**。每条 `result.json` 含 `planner_backend=curobo`、原始阶段计划、实际执行 qpos、接触证据、最终各方块判据。此历史结果只覆盖小范围分布和仿真辅助夹持；扩大范围的结果见下文。

主展示 run 还对实际抓取轨迹执行了从上向下抓取检查：21 帧中最大朝向误差 `0.63°`、最大 XY 偏移 `0.21 mm`，提起后无刷子与支架接触；检查结果在 `result.json` 的 `pick.top_down_and_stand_postcheck`。上述四条固定样本的已保存轨迹也通过同一检查。

## 迁移最初跑通的旧刷摆位

“旧刷成功摆位”应以[旧刷 seed 36 成功 run](../../../runs/sweep_cube/verified_contact_full_36_45_20261019/seed_36/result.json)的实际采样 pose 为准，而非旧场景 `scene.json` 的名义 pose。旧刷成功时，绿、红、蓝块分别在 `(0.2785,-0.0534)`、`(0.4540,-0.1870)`、`(0.3909,-0.0516)` m；簸箕在 `(0.38,0.12)` m，开口朝世界 −Y。原先的[名义 pose 移植失败 run](../../../runs/sweep_long/legacy_short_layout_20261020/result.json)不构成这组成功摆位的对照。

将旧刷 seed 36 的三块方块与簸箕 **XY 和 yaw 原样移植**、保留直立长刷及支架初态的[对照 run](../../../runs/sweep_long/old_success_seed36_transplant_20261020/result.json)：初态有效、从上向下抓取通过，但三块均未进入扫动执行；蓝块的 57 个候选中 26 个预测刷头碰簸箕侧壁，另外还有目标 IK、接触几何不合格。对应[俯视预览](../../../runs/sweep_long/old_success_seed36_transplant_20261020/previews/top.png)。旧刷历史成功 run 与本次长刷 run 的工具几何、持握姿态和部分代码版本均不同，因此不能把差异单独归因于某一个因素。把长刷连同支架移到旧刷的 `(0.354,-0.293)` m 位置，在 CPU 全姿态预抓诊断中也未收敛；正式任务保留长刷原有的可抓支架位置。

可行的[调整组](../../../runs/sweep_long/old_success_seed36_adjusted_20261020/task_scene/source_mapping.json)以旧刷 seed 36 为来源：围绕簸箕将分布旋转 −90°，沿入盘方向压缩到 60%、横向压缩到 30%，簸箕移到 `(0.48,0.10)` m、开口朝 −X。生效位置为绿 `(0.3759,0.1305)`、红 `(0.2958,0.0778)`、蓝 `(0.3770,0.0967)` m，各方块 yaw 同时旋转 −90°。长刷和支架仍保持竖立、原位。[初态俯视](../../../runs/sweep_long/old_success_seed36_adjusted_20261020/previews/top.png)已打开检查，没有初始物体穿透。该组从 fresh reset 使用 cuRobo 逐阶段规划和 MuJoCo 实际执行，**1/1 完整任务成功**；三块的实际入盘判据、阶段结果及仿真辅助夹持证据见 [result.json](../../../runs/sweep_long/old_success_seed36_adjusted_20261020/result.json) 和[终态图](../../../runs/sweep_long/old_success_seed36_adjusted_20261020/renders/final.png)。这是单个调整摆位的成功，不代表整个旧刷摆位分布都可迁移。

查看这条实际执行，或从保存的运行专用场景重新评测：

```bash
python tools/viewers/view_task.py --task sweep_long \
  --run runs/sweep_long/old_success_seed36_adjusted_20261020
python tools/run/run_sweep_cube.py \
  --scene-dir runs/sweep_long/old_success_seed36_adjusted_20261020/task_scene \
  --robot-gravity-compensation \
  --output-dir runs/sweep_long/old_success_seed36_adjusted_repeat
```

## 扩大摆位后的固定样本结果

[wide 分布评测汇总](../../../runs/sweep_long/wide_final_20261020/evaluation.json)保存 seed 0–5 的六条 fresh reset：**5/6 成功**，六条初态有效、代码来源哈希一致，均使用 cuRobo 正式规划和 MuJoCo 实际执行。成功 seed 为 0、1、2、3、5；seed 4 的红块扫动把蓝块推到入口侧缘，后续蓝块没有通过刷头侧壁碰撞、接触几何及 IK 的完整候选预检，见[失败记录](../../../runs/sweep_long/wide_final_seed4_20261020/result.json)。固定改为最近方块先扫的[对照](../../../runs/sweep_long/wide_seed4_nearest_probe_20261020/result.json)也失败，故正式策略没有改成最近先扫。可看 [seed 1 实际执行录像](../../../runs/sweep_long/wide_final_seed1_20261020/sweep_actual.mp4)和[终态图](../../../runs/sweep_long/wide_final_seed1_20261020/renders/final.png)。

`stress` 侧缘范围的先前探索为 4/6；其 seed 1、5 均保留失败 run。把最早短刷的方块和簸箕位姿原样用于长柄刷的[单条对照](../../../runs/sweep_long/legacy_short_layout_20261020/result.json)同样失败，不能将短刷场景的成功率直接迁移到长柄刷。当前 5/6 只支持已声明的 `wide` 范围和仿真辅助夹持，不代表所有摆位、纯物理夹持或真机成功。
