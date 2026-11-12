# 绿色杯：抓杯把、向碗倾倒、放回原位

这个任务使用 `my_scene_0` 的绿色杯 mesh、杯把 Concept 和 Bowl_0。机器人是单臂 Realman + Pika；cuRobo 规划每段运动，MuJoCo 执行并判定杯子、小球和碗的实际状态。夹爪从世界上方向下接近杯把上缘；只有验证双侧指垫接触后，`--simulation-weld` 才会启用并记录仿真辅助夹持。该模式不代表纯物理夹持或真机成功。

## 查看初始场景

[名义俯视](../../../data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/renders/top.png)、[名义全景](../../../data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/renders/overview.png)；随机初态：[seed 0](../../../data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/samples/seed_0/renders/top.png)、[seed 2](../../../data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/samples/seed_2/renders/top.png)、[seed 10](../../../data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/samples/seed_10/renders/top.png)、[seed 12](../../../data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/samples/seed_12/renders/top.png)。杯子 XY 各 ±2.5 cm，绕自身向上的局部轴采样 yaw ±180°；目标碗 XY 各 ±2 cm、yaw ±8°。杯内球随杯子的平移和转角一起改变初态。每个候选还要通过杯把 pregrasp 与 grasp 的 MuJoCo CPU 可达性诊断；不合格样本的数量记在 `scene.json`，因此**被接受的 yaw 并非整圈均匀分布**，也不等于正式 cuRobo 抓取成功。名义杯中心约 `(0.35, -0.16) m`，目标碗约 `(0.30, 0.04) m`，杯内 125 颗半径 7 mm 的小球。杯壁和球的仿真摩擦系数为 0.05。

在仓库根目录、`curobo` conda 环境中打开可旋转的 3D 场景：

```bash
python tools/viewers/view_reconstructed_tabletop_scene.py \
  --scene-dir data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour --mode mujoco
python tools/viewers/view_reconstructed_tabletop_scene.py \
  --scene-dir data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/samples/seed_2 --mode viser
```

## 运行与逐段验收

```bash
PYTHONPATH=. python tools/run/run_cup_pour_task.py \
  --simulation-weld --output runs/cup_pour_task/my_run

# 也可从新 reset 运行至某个 Skill 边界
PYTHONPATH=. python tools/run/run_cup_pour_task.py \
  --simulation-weld --until-skill pick \
  --output runs/cup_pour_task/my_pick

# 随机初态
PYTHONPATH=. python tools/run/run_cup_pour_task.py \
  --scene data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/samples/seed_0 \
  --simulation-weld --output runs/cup_pour_task/my_seed_0
```

`--until-skill` 接受 `pick`、`pour`、`place`；每次从新 reset 开始，Skill 之间传递实际 MuJoCo 状态。默认完整 Task 运行到放回。`task_result.json` 保存各阶段的 cuRobo 返回、规划复核、MuJoCo 执行采样、球 ID 归属和最终判定。`result.json` 与 `pick/result.json` 分别保存抓取总结果和 Pick 阶段细节。

场景 `scene.json` 中的 `scene_preview_only` 表示构建脚本本身只检查静态场景；任务入口见同目录 `task.json`，实际是否完成以每次运行的 `task_result.json` 为准。

```bash
python tools/viewers/view_skill.py --skill cup_pick --run runs/cup_pour_task/my_run
python tools/viewers/view_skill.py --skill cup_pour --run runs/cup_pour_task/my_run
python tools/viewers/view_skill.py --skill cup_place --run runs/cup_pour_task/my_run
python tools/viewers/view_task.py --task cup_pour_into_bowl --run runs/cup_pour_task/my_run

python tools/viewers/recording/record_cup_pour_task_video.py \
  runs/cup_pour_task/my_run --segment full
```

视频脚本可用 `--segment pick|pour|place|full`。它从实际执行 qpos 重播，不重新规划，也不把规划轨迹当作物理录像。回放 GUI 和视频默认要求对应 Skill 已成功；失败时直接查看 run 中的 `task_result.json`、`motion_curobo/` 及 `task_observations/`。

## 2026-10-20 实测结果

**抬高搬运与杯身随机朝向：**[新版本完整 MuJoCo 视频](../../../runs/cup_pour_task/yaw360_reachable_nominal_full_shift5mm_20261020/cup_full_actual.mp4)、[搬运关键帧](../../../runs/cup_pour_task/yaw360_reachable_nominal_full_shift5mm_20261020/video_frames/full/carry_above.png)、[倾倒关键帧](../../../runs/cup_pour_task/yaw360_reachable_nominal_full_shift5mm_20261020/video_frames/full/tilt_50.png)、[最终关键帧](../../../runs/cup_pour_task/yaw360_reachable_nominal_full_shift5mm_20261020/video_frames/full/final.png)和[任务记录](../../../runs/cup_pour_task/yaw360_reachable_nominal_full_shift5mm_20261020/task_result.json)。名义初态从 fresh reset 完成抓取、搬运、50° 倾倒与放回，6/125 颗球入碗、0 颗洒落，杯子回位 XY 误差 4.21 mm。搬运时先抬高，再横移到碗上方，之后在碗正上方下降到倾倒高度；在杯子接近碗的实际 MuJoCo 采样中，杯底高于保守估计碗沿的最小距离为 3.05 cm，比旧完整运行的 1.05 cm 高约 2 cm。这是仿真碰撞代理与相机视图下的净空，不是真机安全距离。随机朝向初态另见下方各 seed 的独立运行记录；可达性筛选后的样本不能代表整圈均匀成功率。

**随机 seed 2 完整任务：**[初始图](../../../runs/cup_pour_task/yaw360_reachable_seed2_full_shift5mm_20261020/diagnostic_frames/initial.png)、[抓起后](../../../runs/cup_pour_task/yaw360_reachable_seed2_full_shift5mm_20261020/diagnostic_frames/grasp_lift.png)、[转正并搬到碗上方](../../../runs/cup_pour_task/yaw360_reachable_seed2_full_shift5mm_20261020/diagnostic_frames/carry_above.png)、[完整实际视频](../../../runs/cup_pour_task/yaw360_reachable_seed2_split_retreat_20261020/cup_full_actual.mp4)、[原始记录](../../../runs/cup_pour_task/yaw360_reachable_seed2_split_retreat_20261020/task_result.json)。杯子起始 yaw 约 82°，抓起后在搬运中转到名义倾倒朝向约 0°，放回时恢复原朝向。从 fresh reset 完成全部 Skill，6/125 颗球入碗、0 颗洒落，杯子回位 XY 误差 3.90 mm。松爪后分两段各 3.5 cm 竖直撤离；两段 cuRobo 直线规划、碰撞和 MuJoCo 执行均通过，最大横向偏差分别为 3.79 mm 和 0.13 mm。此前同 seed 的单段 7 cm 撤离轨迹横向偏差 5.55 mm，超过 5 mm 门限而被拒绝；[失败诊断](../../../runs/cup_pour_task/yaw360_reachable_seed2_full_shift5mm_20261020/task_result.json)保留。

**固定四个随机初态评测：**[汇总 JSON](../../../runs/cup_pour_task/yaw360_reachable_fixed4_eval_20261020/summary.json)。同一版代码、`planner_backend=curobo`、MuJoCo 实际执行、双侧指垫接触后仿真辅助夹持，从新 reset 连续运行完整任务，成功 **3/4 = 75%**。所有样本均是场景生成时通过 CPU 抓取可达性筛选后的接受样本；失败样本没有从评测中剔除。

| seed | 杯子初始 yaw | 入碗球数 | 完整结果 |
|---:|---:|---:|---|
| [0](../../../runs/cup_pour_task/yaw360_reachable_seed0_split_retreat_20261020/task_result.json) | 148.6° | 10 | 成功 |
| [2](../../../runs/cup_pour_task/yaw360_reachable_seed2_split_retreat_20261020/task_result.json) | 82.3° | 6 | 成功 |
| [10](../../../runs/cup_pour_task/yaw360_reachable_seed10_split_retreat_20261020/task_result.json) | 88.4° | 5 | 成功 |
| [12](../../../runs/cup_pour_task/yaw360_reachable_seed12_split_retreat_20261020/task_result.json) | 14.9° | — | 抓起并抬杯后，`carry_raise` 目标 cuRobo IK 无解 |

四个接受样本的 yaw 全在正角度一侧，数量也很少；75% 只是这四次观测比例，不能外推为整圈随机朝向、均匀分布或真机的成功率。

**当前 50° 版本：**[完整 MuJoCo 视频](../../../runs/cup_pour_task/tilt50_official_nominal_20261020/cup_full_actual.mp4)、[入碗和回位俯视图](../../../runs/cup_pour_task/tilt50_official_nominal_20261020/video_frames/full/final_bowl_top.png)、[原始任务记录](../../../runs/cup_pour_task/tilt50_official_nominal_20261020/task_result.json)。从 fresh reset 执行全部三个 Skill，`planner_backend=curobo`，实际倾角最大 50°；最终 5/125 颗入碗，0 颗洒落，120 颗留杯。杯子回位 XY 误差 4.2 mm，姿态误差 0.030°，辅助夹持释放。已检查初始预览、50° 阶段、降杯静置、最终回位视频帧和俯视图。抓取使用双侧指垫接触后的显式仿真辅助 weld；该记录不代表纯物理或真机夹持。

以下 105° 版本及其宽分布结果是**历史对照**，使用旧场景 `scenes/pour`、67 颗半径 6 mm 球和旧成功标准，不可与当前 50° 版本混作同一次评测。

名义场景从新 reset 连续通过三个 Skill：[完整实际 MuJoCo 视频](../../../runs/cup_pour_task/nominal_verified_20261020/cup_full_actual.mp4)，也可分别看[抓取](../../../runs/cup_pour_task/nominal_verified_20261020/cup_pick_actual.mp4)、[倾倒](../../../runs/cup_pour_task/nominal_verified_20261020/cup_pour_actual.mp4)、[放回](../../../runs/cup_pour_task/nominal_verified_20261020/cup_place_actual.mp4)。已人工检查[抓取关键帧](../../../runs/cup_pour_task/nominal_verified_20261020/video_frames/full/approach.png)、[倾倒关键帧](../../../runs/cup_pour_task/nominal_verified_20261020/video_frames/full/tilt_105.png)和[最终关键帧](../../../runs/cup_pour_task/nominal_verified_20261020/video_frames/full/final.png)。[任务原始记录](../../../runs/cup_pour_task/nominal_verified_20261020/task_result.json)显示：杯把双侧指垫接触成立，63/67 颗球最终在碗内、4/67 颗在碗外；杯子回位 XY 误差 3.4 mm、姿态误差 0.009°，辅助夹持已释放。33 个阶段均通过，`planner_backend=curobo`。

历史宽分布为杯子 XY 各 ±2.5 cm、yaw ±15°，目标碗 XY 各 ±2 cm、yaw ±8°，独立均匀采样。固定 seed 的旧版完整运行结果如下；失败样本没有被剔除：

| 初态 | 入碗 / 67 | 碗外 / 67 | 结果 |
|---|---:|---:|---|
| 名义 | 63 | 4 | 完整任务通过 |
| [seed 0](../../../runs/cup_pour_task/seed_0_20261020/task_result.json) | 51 | 16 | 倾倒后置条件失败 |
| [seed 7](../../../runs/cup_pour_task/seed_7_20261020/task_result.json) | 61 | 6 | 倾倒后置条件失败 |
| [seed 21](../../../runs/cup_pour_task/seed_21_20261020/task_result.json) | 47 | 20 | 倾倒后置条件失败 |

三个旧版扰动样本均完成了杯把抓取与倾倒规划，但球在 90° 附近从碗近侧洒落。相关诊断 run 保留在 `runs/cup_pour_task/`。新版固定扰动样本需按其独立运行记录判断，名义成功不能推出整个扰动分布成功。

## 成功条件与实现边界

抓取要求双侧内指垫都接触**杯把上缘**，不能用杯壁接触冒充。抬杯后至少 95% 初始球仍在杯内。倾倒最多旋转 50°，可分段降低杯口；球只靠 MuJoCo 接触和重力进入碗。静置后及放回后，至少 5 颗初始球须在目标碗内，碗外不超过 5 颗且不多于入碗数。放回要求释放辅助夹持、杯子 XY 距初始不超过 2 cm、姿态误差不超过 10°，最后再次检查球的归属。仿真物体、杯把薄碰撞条和碗壁均为操作代理，真机抓持及颗粒材料差异尚未验证。
