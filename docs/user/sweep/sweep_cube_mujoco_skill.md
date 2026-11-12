# 三色方块扫入簸箕：MuJoCo Task

场景有一把刷子、绿/红/蓝三个边长约 22.4 mm 的方块和一个簸箕。三块的名义位置、朝向各不相同；任务要求**三块都进入簸箕**。程序从 fresh reset 开始，Pika 抓刷柄，然后根据每块落稳后的实际 pose 依次生成清扫目标。必要时会对未入盘的方块做一次基于实际终态的补扫。每段规划、执行、接触和最终入盘判定分别记录。

默认入口现使用 GPU cuRobo 逐阶段规划完整轨迹，再交给 MuJoCo 物理执行。持刷阶段使用从实际刷子碰撞 mesh 生成的附着球；允许接触的障碍只在相应阶段从 cuRobo world 排除，并在执行前用完整 MuJoCo 几何复核整条轨迹。需要 `curobo` 环境及可用 GPU。在仓库根目录运行，输出目录必须不存在：

```bash
python tools/run/run_sweep_cube.py --output-dir runs/sweep_cube/my_nominal
python tools/run/run_sweep_cube.py --randomize --seed 10 \
  --output-dir runs/sweep_cube/my_random_10
```

批量评测使用明确的种子清单，每个种子都计入分母；无效初态和运行异常也算失败：

```bash
python tools/run/evaluate_sweep_cube.py --seeds 10 11 12 13 14 15 16 17 18 19 \
  --output-dir runs/sweep_cube/my_eval
```

采样配置在 `scene.json` 的 `randomization` 中：桌面 world XY 范围、每件物体的名义 XY/yaw 和独立均匀扰动范围、最小间距、尝试上限都有米/弧度单位。**当前保留用户选定的 B 档扰动幅度**：三块各自的 XY 最大偏移为 9/7 cm、yaw 最大偏移 90°；刷子的 XY 最大偏移为 6/5.5 cm、yaw 最大偏移 35°。簸箕固定在更靠近机械臂的名义位置 (0.38, 0.12) m；刷子及三块的名义位置也作了可达性调整。[新布局六图](../../../runs/sweep_cube/candidate_near_pose_20261017/preview_sheet.png)展示 seed 20–25 的实际采样。采样器先用旋转后的保守平面包围盒检查桌边和物体间距，再让 MuJoCo 落稳 100 步；若位移超过 3 cm 或物体间穿透超过 3 mm，记录无效初态并判失败。`result.json` 保存名义、采样和落稳后的 pose、seed、重采样次数与拒绝原因；批量 `evaluation.json` 保存每个 seed 的成败及耗时。

查看实际执行回放：

```bash
python tools/viewers/view_skill.py --skill sweep_cube \
  --run runs/sweep_cube/my_random_10
python tools/viewers/view_task.py --task sweep_cube_into_dustpan \
  --run runs/sweep_cube/my_random_10
python tools/viewers/recording/record_sweep_cube_video.py runs/sweep_cube/my_random_10
```

`view_skill.py` 可选抓刷子的 `pick_brush` 或三次清扫的 `sweep_cube`；`view_task.py` 显示完整任务。也可不传 `--run`，自动选择与当前 `scene.xml` 哈希匹配的最近一次顶层成功 run。两个 viewer 和录像均读取实际 MuJoCo qpos，不重新规划。输出目录中的 `stages/` 是逐阶段原始记录，`renders/` 保存初态及终态截图，`sweep_actual.mp4` 和 `video_manifest.json` 是附加录像证据。场景改动后旧 run 因哈希不同而无法回放。

成功条件是**每一块**实际进入簸箕托盘范围、发生托盘接触并在结束时停稳。GT pose 仅用于规划输入和判定。抓刷子时，Pika 两侧真实指垫在内侧 site 附近都接触刷柄后启用显式仿真辅助 weld，松爪时解除；报告标记为“仿真辅助夹持”。方块的运动和入盘由 MuJoCo 接触产生，没有把方块固定到刷子或瞬移。簸箕碰撞底面与桌面齐平是任务接触代理的近似。

加 `--diagnostic-cpu-ik` 才运行旧的 MuJoCo CPU 姿态 IK 加离散轨迹碰撞检查；其刷面候选另用 cuRobo **纯 pose IK** 筛选。若 cuRobo 的保守 Pika 球将实际起点判作自碰撞，诊断模式可显式保留 CPU 路径可行的原始候选；该项在 `knowledge/*_contact_candidates.json` 中标为 `cpu_diagnostic_only_curobo_rejected_original`，不计作 cuRobo 可行。默认 cuRobo 模式不会用 CPU 路径替代失败规划。pregrasp、清扫预备位和撤离可使用有记录的小误差预算；接触下降与扫动保持严格目标。扫动阶段将 cuRobo 的原轨迹按 2 倍时间送入 MuJoCo，以减小执行跟踪误差。

固定 seed 36–45 的旧 CPU 对照为 **6/10**；加入接触姿态候选后，同样 seed 的 fresh reset MuJoCo 实际执行为 **8/10**，记录在 `runs/sweep_cube/contact_candidates_final_36_45/evaluation.json`。初版正式 cuRobo 轨迹规划得到 **5/10**；加入 Pika 球精度、非接触阶段有界次优 IK 与扫动降速后的完整复评仍为 **5/10**，记录在 `runs/sweep_cube/curobo_approx_full_36_45/evaluation.json`，成功 seed 39、40、42、44、45。seed 41 的抓刷预位通过次优 IK 并完成抓刷及三次清扫，但后续红块补扫接触下降失败。这里的成功指 MuJoCo 实际执行及接触后的任务成功，持刷仍使用显式仿真辅助 weld。诊断和失败归因见开发记录（历史材料已本地归档）。

## 2026-10-19 调试版本与可查看记录

本轮没有修改 B 档初始分布。新增了整段 cuRobo 预检、真实碰撞网格高度校正、推面法向筛选，以及偏侧方块先在入口外归拢再轴向入盘。辅助持刷误差、刚性侧壁/后壁穿透和真实接触终态均单独检查。**当前仍在回归调试，不能宣称整个分布已解决。**

已验证的完整成功记录（历史开发版本，默认抓刷倾角 -20°）：

```bash
python tools/viewers/view_skill.py --skill sweep_cube \
  --run runs/sweep_cube/verified_contact_full_36_45_20261019/seed_36
python tools/viewers/view_task.py --task sweep_cube_into_dustpan \
  --run runs/sweep_cube/verified_contact_full_36_45_20261019/seed_36
```

[实际执行录像](../../../runs/sweep_cube/verified_contact_full_36_45_20261019/seed_36/sweep_actual.mp4) / [最终截图](../../../runs/sweep_cube/verified_contact_full_36_45_20261019/seed_36/renders/final.png)。该 run 三块全部进入托盘并停稳，持刷最大偏转约0.387°，属于接触核验后的仿真辅助夹持。

`--grasp-tilt-deg` 可显式比较抓取姿态，默认 -20°，不改变已有物体初态。零度对照未获得完整成功，因此没有作为默认设置。新旧策略和检查条件不同，不能混合统计成功率；当前试验、失败类别与尚未验证项见开发诊断（历史材料已本地归档）。


当前带完整接触检查的旧配置固定36–45回归为1/10。已定位并修正Pika底座假自碰撞，同时加入无运动预检失败时的目标延期重排；新配置成功率以独立run为准，不沿用上述旧配置结果。

可用 `--robot-gravity-compensation` 显式测试仅机器人本体的重力补偿；程序会在run内保存有效仿真XML，原场景不修改，操作物体不补偿。它针对仿真位置控制的静态重力误差，完整任务效果仍以该配置对应的run判定，不能将隔离hold诊断当作任务成功。
