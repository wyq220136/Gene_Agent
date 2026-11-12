# my_scene_0：三类任务、五个单工具场景

输入为 [`my_scene_0/rgb.png`](../../../data/realworld/my_scene_0/rgb.png)。新物体的分割和 SAM3D 网格保存在 `data/realworld/my_scene_0/reconstruction/`。蓝色短刷的网格仍有问题，本轮**跳过**：生成器不再处理该资产，也不再生成 `sweep_small`；仓库中的旧 `sweep_small/` 目录只是历史结果，不作为验收场景。清扫复用旧刷、三色方块与簸箕；倾倒和舀取复用 `data/task_1/objs/Bowl/Bowl_0`。每个有效场景只出现**一个**刷子或勺子。橙色和粉色勺的连续 Skill 与 Task 验收见[盛球倾倒说明](../spoon/spoon_scoop_task.md)；其他场景的能力以各自记录为准。

## 初始场景截图

勺子名义初态为 `(x=0.38, y=-0.23) m`，来源碗为 `(0.43, -0.06) m`，目标碗为 `(0.46, 0.125) m`；这样勺柄抓取、勺头进入源碗和目标碗上方的小范围倾倒都有余量。来源碗有 165 个小球，目标碗为空。长柄刷初始中心在 `(0.20, -0.05) m`，刷头朝下、柄朝上，放在可拔出的低矮支架中；Pika TCP 的局部 +X 作为接近轴，目标姿态严格对准世界 -Z，直线向下抓柄。先看名义初态，再看 seed 0/7/21 的扰动初态。

| 场景 | 名义初态 | 扰动初态全景 |
|---|---|---|
| 旧刷清扫 | [全景](../../../data/realworld/my_scene_0/scenes/sweep_old/renders/overview.png) · [俯视](../../../data/realworld/my_scene_0/scenes/sweep_old/renders/top.png) | [0](../../../data/realworld/my_scene_0/scenes/sweep_old/samples/seed_0/renders/overview.png) · [7](../../../data/realworld/my_scene_0/scenes/sweep_old/samples/seed_7/renders/overview.png) · [21](../../../data/realworld/my_scene_0/scenes/sweep_old/samples/seed_21/renders/overview.png) |
| 白柄长刷清扫 | [全景](../../../data/realworld/my_scene_0/scenes/sweep_long/renders/overview.png) · [俯视](../../../data/realworld/my_scene_0/scenes/sweep_long/renders/top.png) | [0](../../../data/realworld/my_scene_0/scenes/sweep_long/samples/seed_0/renders/overview.png) · [7](../../../data/realworld/my_scene_0/scenes/sweep_long/samples/seed_7/renders/overview.png) · [21](../../../data/realworld/my_scene_0/scenes/sweep_long/samples/seed_21/renders/overview.png) |
| 绿色杯 50° 倾倒 | [全景](../../../data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/renders/overview.png) · [俯视](../../../data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/renders/top.png) | [0](../../../data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/samples/seed_0/renders/top.png) · [2](../../../data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/samples/seed_2/renders/top.png) · [10](../../../data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/samples/seed_10/renders/top.png) · [12](../../../data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/samples/seed_12/renders/top.png) |
| 橙色勺舀取 | [全景](../../../data/realworld/my_scene_0/scenes/scoop_orange/renders/overview.png) · [俯视](../../../data/realworld/my_scene_0/scenes/scoop_orange/renders/top.png) · [近景](../../../data/realworld/my_scene_0/scenes/scoop_orange/renders/objects.png) | [0](../../../data/realworld/my_scene_0/scenes/scoop_orange/samples/seed_0/renders/overview.png) · [7](../../../data/realworld/my_scene_0/scenes/scoop_orange/samples/seed_7/renders/overview.png) · [21](../../../data/realworld/my_scene_0/scenes/scoop_orange/samples/seed_21/renders/overview.png) |
| 粉色勺舀取 | [全景](../../../data/realworld/my_scene_0/scenes/scoop_pink/renders/overview.png) · [俯视](../../../data/realworld/my_scene_0/scenes/scoop_pink/renders/top.png) | [0](../../../data/realworld/my_scene_0/scenes/scoop_pink/samples/seed_0/renders/overview.png) · [7](../../../data/realworld/my_scene_0/scenes/scoop_pink/samples/seed_7/renders/overview.png) · [21](../../../data/realworld/my_scene_0/scenes/scoop_pink/samples/seed_21/renders/overview.png) |

查看可旋转的完整 3D 场景（两种 viewer 读取同一份 MJCF）：

```bash
python tools/viewers/view_reconstructed_tabletop_scene.py \
  --scene-dir data/realworld/my_scene_0/scenes/sweep_long --mode mujoco
python tools/viewers/view_reconstructed_tabletop_scene.py \
  --scene-dir data/realworld/my_scene_0/scenes/scoop_orange/samples/seed_7 --mode viser
```

查看单个 3D mesh 与功能 part，可在浏览器旋转、缩放并编辑：

```bash
python tools/editors/edit_concept.py --object brush_long
python tools/editors/edit_concept.py --object spoon_orange
```

有效场景名为 `sweep_old`、`sweep_long`、`pour`、
`scoop_orange`、`scoop_pink`；将上述 `--scene-dir` 最后一段换成对应名称即可。

## 功能性 Concept

本轮由 Codex 查看 RGB 和规范 mesh 的三视图，按 `gene_agent.perception_modules.concept_agent.annotate_concept` 的 schema 给出操作相关部位，由该 pipeline 验证 JSON 并渲染叠图；没有调用外部 Codex CLI 迭代。灰色为参考 mesh，彩色为每个 part 一个 Cuboid、Cylinder 或 Ring 的近似。旧刷、方块和簸箕的已有 Concept 在 `data/realworld/brush_cube_dustpan_0/concepts/`。

| 物体 | 功能部位 | 三视图叠图 |
|---|---|---|
| 白柄长刷 | 可抓长柄、清扫面 | [查看](../../../data/realworld/my_scene_0/concepts/brush_long/overlay_final.png) |
| 橙色勺 | 勺柄、凹勺头 | [查看](../../../data/realworld/my_scene_0/concepts/spoon_orange/overlay_final.png) |
| 粉色勺 | 勺柄、凹勺头 | [查看](../../../data/realworld/my_scene_0/concepts/spoon_pink/overlay_final.png) |
| 绿色杯 | 杯把、盛放/倾倒杯壁 | [查看](../../../data/realworld/my_scene_0/concepts/cup_green/overlay_final.png) |
| 碗 | 收纳与舀取开口 | [查看](../../../data/realworld/my_scene_0/concepts/bowl/overlay_final.png) |

新增参数和运行时间在 `data/realworld/my_scene_0/concepts/<物体>/annotation_run.json`。例如勺子的细调：

```bash
python tools/editors/edit_concept.py --object spoon_orange
```

## 初态分布与打开方式

所有非球体物体在**机器人基座世界坐标系**以名义 XY/yaw 为中心独立均匀采样；球随来源容器平移。旧刷为 ±2.5 cm、±12°；舀取场景的勺子与两只碗为各轴 ±1 cm、±5°；竖立长刷和支架一起平移 ±1.5 cm、不改变朝向；其他场景的碗为 ±2 cm、±8°；杯 ±2.5 cm、±15°；方块 ±2 cm、±15°；簸箕 ±1.5 cm、±7°。限制在桌面 `x∈[-0.10,0.86] m, y∈[-0.50,0.50] m`，用旋转 AABB 检查至少 5 mm 名义间距。刷子和勺子样本还必须通过**功能抓持部位**的完整姿态 IK、standby 到预抓位姿的关节路径碰撞，以及预抓到抓取位姿的直线 TCP 路径碰撞检查；不合格样本计入 `grasp_reachability` 拒绝次数并继续采样，落稳后再复核一次。每个 `samples/seed_*/scene.json` 保存 seed、范围、拒绝次数、抓取检查与名义/采样/500 步静置 pose。`scene.xml` 的起始 keyframe 是静置后的稳定初态，原始投放 pose 另存于 `placement_object_poses`。

```bash
# 在 curobo 环境，重建五个场景、seed 0/7/21 和离屏截图
python tools/run/build_multi_task_scenes.py

python tools/viewers/view_reconstructed_tabletop_scene.py \
  --scene-dir data/realworld/my_scene_0/scenes/scoop_orange --mode mujoco
python tools/viewers/view_reconstructed_tabletop_scene.py \
  --scene-dir data/realworld/my_scene_0/scenes/scoop_orange/samples/seed_7 --mode viser
```

桌面名义摆位为了操作方便作了调整，不逐毫米复刻照片。新 mesh 是单视角 SAM3D 重建，碰撞为功能代理。长柄刷的圆柄代理按 Concept 设为 8 mm 半径，刷头由低矮支架扶正。之前的长柄刷抓取带 20° 倾角，现已移除。此场景明确启用 MuJoCo 机器人重力补偿，避免位置伺服在重力下留下数度姿态误差；这是仿真控制设置，不是真机验证。名义状态下，目标接近轴严格为世界 -Z，实际预抓位偏差 0.33°、抓取位偏差 0.016°，整个直线下探期间最大偏差 0.32°、横向偏移 0.61 mm、无向上回退：[预抓截图](../../../runs/my_scene_0_grasp_diagnostic/brush_long_upright_pregrasp.png)。闭爪后实际偏差 0.001°，双侧指垫接触成立：[闭爪截图](../../../runs/my_scene_0_grasp_diagnostic/brush_long_upright_close.png)。在**接触核验后仿真辅助 weld** 下，刷子中心抬高 7.76 cm，已无支架接触：[拔出截图](../../../runs/my_scene_0_grasp_diagnostic/brush_long_upright_lift.png)。三个扰动初态 seed [0](../../../runs/my_scene_0_grasp_diagnostic/brush_long_seed_0.json)、[7](../../../runs/my_scene_0_grasp_diagnostic/brush_long_seed_7.json)、[21](../../../runs/my_scene_0_grasp_diagnostic/brush_long_seed_21.json) 也分别通过同一竖直抓取及辅助拔出诊断。这些数值来自显式标记的 `mujoco_cpu_diagnostic`，尚不是 cuRobo 正式任务评测；纯接触夹持仍会滑脱。橙色和粉色勺名义初态已有 cuRobo 正式连续任务成功记录；粉色勺还通过随机 seed 0，见[盛球倾倒说明](../spoon/spoon_scoop_task.md)。绿色杯当前 50° 任务已有 cuRobo 规划、MuJoCo 执行的仿真辅助抓取—少量入碗—放回名义成功记录；旧 105° 版本的三个固定扰动 seed 均未达旧倾倒标准，详见[杯子任务说明](../cup/cup_pour_task.md)。

物理诊断可复现：

```bash
PYTHONPATH=. python experiments/my_scene_0/diagnose_spoon_pickup.py
PYTHONPATH=. python experiments/my_scene_0/diagnose_long_brush_pickup.py
```

长刷诊断结果见[记录](../../../runs/my_scene_0_grasp_diagnostic/brush_long_upright.json)。

长刷已进一步接入[完整 MuJoCo 清扫 Task](../sweep/sweep_long_mujoco_task.md)：在运行专用场景中调整簸箕与方块的可达布局，使用世界 −Z 抓取、cuRobo 规划和实际三块入盘判定。上述原始场景预览及抓取诊断仍是独立证据，不能等同于完整 Task 成绩。
