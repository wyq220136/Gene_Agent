# 橙色勺与粉色勺：抓取、盛球、倾倒

在仓库根目录、`curobo` 环境执行。正式规划使用 cuRobo，机器人轨迹与小球接触由同一个 MuJoCo Runtime 连续执行。已通过名义初态和下文列出的随机样本；夹勺在有效双侧指垫接触后启用**显式仿真辅助 weld**，不代表纯物理夹持或真机成功。

## 当前倾倒方式与随机初态

勺头侧沿碰撞顶端为 8 mm，背沿保留推进接触面。倾倒时先围绕实际勺头中心转 30°，之后最多每段 15°；首段使用显式 cuRobo 原位旋转约束（TCP 最大偏移 2.5 cm、总路程 5 cm），允许旋转所需的小幅回退。每段轨迹还要经过完整 MuJoCo 碰撞与勺头中心偏移复核，实际执行中单段偏移不得超过 2 cm。所有已盛球进入目标碗后停止翻转，静置 1 秒再检查**全部**球仍在碗内。球仅通过 MuJoCo 接触运动，不与勺子 weld。

已从 fresh reset 检查以下连续任务。名义场景是在加入显式 `pivot_tcp` 之前运行；粉色 seed 7、21 与橙色 seed 0、7、21 使用当前约束。各行均为 cuRobo 规划、MuJoCo 实际执行、双指接触后仿真辅助夹勺。

| 勺子与初态 | 转移球数 | 倾倒 TCP 最大偏移 | 视觉验收 |
|---|---:|---:|---|
| 粉色名义 | 3/3 | 2.02 cm | [倾倒视频](../../../runs/spoon_scoop_task/pink_low_lip_early_stop_20261019/spoon_pour_actual.mp4)、[完整任务](../../../runs/spoon_scoop_task/pink_low_lip_early_stop_20261019/spoon_full_actual.mp4)、[碗内俯视](../../../runs/spoon_scoop_task/pink_low_lip_early_stop_20261019/video_frames/pour/target_top_tool_hidden.png) |
| 粉色 seed 0 | 3/3 | 1.98 cm | [完整回放](../../../runs/spoon_scoop_task/pink_low_lip_seed0_20261019/spoon_full_actual.mp4)、[倾倒近景](../../../runs/spoon_scoop_task/pink_low_lip_seed0_20261019/spoon_pour_actual.mp4) |
| 粉色 seed 7 | 2/2 | 1.26 cm | [完整回放](../../../runs/spoon_scoop_task/pink_low_lip_seed7_pivot_20261019/spoon_full_actual.mp4)、[倾倒近景](../../../runs/spoon_scoop_task/pink_low_lip_seed7_pivot_20261019/spoon_pour_actual.mp4) |
| 粉色 seed 21 | 4/4 | 0.86 cm | [完整回放](../../../runs/spoon_scoop_task/pink_low_lip_seed21_pivot_20261019/spoon_full_actual.mp4)、[倾倒近景](../../../runs/spoon_scoop_task/pink_low_lip_seed21_pivot_20261019/spoon_pour_actual.mp4)、[碗内俯视](../../../runs/spoon_scoop_task/pink_low_lip_seed21_pivot_20261019/video_frames/pour/target_top_tool_hidden.png) |
| 橙色名义 | 3/3 | 1.95 cm | [倾倒视频](../../../runs/spoon_scoop_task/orange_low_lip_early_stop_20261019/spoon_pour_actual.mp4) |
| 橙色 seed 0 | 4/4 | 0.86 cm | [完整回放](../../../runs/spoon_scoop_task/orange_low_lip_seed0_pivot_20261019/spoon_full_actual.mp4)、[倾倒近景](../../../runs/spoon_scoop_task/orange_low_lip_seed0_pivot_20261019/spoon_pour_actual.mp4)、[碗内俯视](../../../runs/spoon_scoop_task/orange_low_lip_seed0_pivot_20261019/video_frames/pour/target_top_tool_hidden.png) |
| 橙色 seed 7 | 4/4 | 1.83 cm | [完整回放](../../../runs/spoon_scoop_task/orange_low_lip_seed7_pivot_gpu_20261019/spoon_full_actual.mp4)、[倾倒近景](../../../runs/spoon_scoop_task/orange_low_lip_seed7_pivot_gpu_20261019/spoon_pour_actual.mp4)、[碗内俯视](../../../runs/spoon_scoop_task/orange_low_lip_seed7_pivot_gpu_20261019/video_frames/pour/target_top_tool_hidden.png) |
| 橙色 seed 21 | 4/4 | 1.32 cm | [完整回放](../../../runs/spoon_scoop_task/orange_low_lip_seed21_pivot_gpu_20261019/spoon_full_actual.mp4)、[倾倒近景](../../../runs/spoon_scoop_task/orange_low_lip_seed21_pivot_gpu_20261019/spoon_pour_actual.mp4)、[碗内俯视](../../../runs/spoon_scoop_task/orange_low_lip_seed21_pivot_gpu_20261019/video_frames/pour/target_top_tool_hidden.png) |

粉色名义场景中，小球在勺子约 31–43° 时脱离，并在 45° 停止倾倒；实际 TCP 累计路程约 3.62 cm。俯视诊断图只隐藏勺子的视觉网格，未改变物理运行或结果判定。

勺子和两只碗按世界坐标 XY 独立均匀扰动，各轴 ±1 cm，yaw ±5°；固定 seed 为 0、7、21。采样检查桌边、物体间距、工具抓取可达性和初态穿透。初始图：粉色勺[名义](../../../data/realworld/my_scene_0/scenes/scoop_pink/renders/objects.png)、[seed 0](../../../data/realworld/my_scene_0/scenes/scoop_pink/samples/seed_0/renders/objects.png)、[seed 7](../../../data/realworld/my_scene_0/scenes/scoop_pink/samples/seed_7/renders/objects.png)、[seed 21](../../../data/realworld/my_scene_0/scenes/scoop_pink/samples/seed_21/renders/objects.png)；橙色勺[名义](../../../data/realworld/my_scene_0/scenes/scoop_orange/renders/objects.png)、[seed 0](../../../data/realworld/my_scene_0/scenes/scoop_orange/samples/seed_0/renders/objects.png)、[seed 7](../../../data/realworld/my_scene_0/scenes/scoop_orange/samples/seed_7/renders/objects.png)、[seed 21](../../../data/realworld/my_scene_0/scenes/scoop_orange/samples/seed_21/renders/objects.png)。随机样本的完整 Task 结果须分别查看各自 run，初态预检不代表全流程成功。

重新生成这两种场景和分布图：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/build_multi_task_scenes.py --tasks scoop_orange scoop_pink --seeds 0 7 21
```

运行一个可复现的扰动样本：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/run_spoon_scoop_task.py \
  --scene data/realworld/my_scene_0/scenes/scoop_pink/samples/seed_0 \
  --simulation-weld --output runs/spoon_scoop_task/my_seed0

PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/diagnostics/analyze_spoon_pour.py runs/spoon_scoop_task/my_seed0
```

## 分阶段验收

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/run_spoon_scoop_task.py --simulation-weld --until-skill pick \
  --output runs/spoon_scoop_task/my_pick

PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/run_spoon_scoop_task.py --simulation-weld --until-skill scoop \
  --output runs/spoon_scoop_task/my_scoop

PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/run_spoon_scoop_task.py --simulation-weld --until-skill pour \
  --output runs/spoon_scoop_task/my_full
```

以上默认使用橙色勺。**粉色勺**只需指定场景，其他 Skill 和 Task 命令相同：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/run_spoon_scoop_task.py \
  --scene data/realworld/my_scene_0/scenes/scoop_pink \
  --simulation-weld --until-skill pour \
  --output runs/spoon_scoop_task/my_pink_full
```

`--until-skill` 从 fresh reset 运行至指定 Skill；不在 Skill 之间重置。两种勺子默认过渡高度均为 0.15 m。默认不启用 weld，可用来诊断纯物理夹持；目前纯物理抓勺可完成短距离抬升，但搬运中会滑移，尚未完成全 Task。每次使用新的 `--output` 目录。

打开初始 3D 场景：

```bash
python tools/viewers/view_reconstructed_tabletop_scene.py \
  --scene-dir data/realworld/my_scene_0/scenes/scoop_orange --mode mujoco
python tools/viewers/view_reconstructed_tabletop_scene.py \
  --scene-dir data/realworld/my_scene_0/scenes/scoop_pink --mode mujoco
```

在 Viser 时间轴检查同一次 run 的实际状态，可切换阶段并自由转动相机：

```bash
python tools/viewers/view_skill.py --skill spoon_pick \
  --run runs/spoon_scoop_task/my_full
python tools/viewers/view_skill.py --skill spoon_scoop \
  --run runs/spoon_scoop_task/my_full
python tools/viewers/view_skill.py --skill spoon_pour \
  --run runs/spoon_scoop_task/my_full
python tools/viewers/view_task.py --task spoon_scoop_into_bowl \
  --run runs/spoon_scoop_task/my_full
```

从实际 MuJoCo `qpos` 生成各 Skill 和全流程录像：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/recording/record_spoon_scoop_task_video.py runs/spoon_scoop_task/my_full --segment full
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/recording/record_spoon_scoop_task_video.py runs/spoon_scoop_task/my_full --segment pick
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/recording/record_spoon_scoop_task_video.py runs/spoon_scoop_task/my_full --segment scoop
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/recording/record_spoon_scoop_task_video.py runs/spoon_scoop_task/my_full --segment pour
```

视频和关键帧在 run 目录下。`task_result.json` 记录每个 Skill 的后置条件、球 ID、失败阶段；`motion_curobo/plans/` 保存原始规划返回；`task_observations/` 保存阶段实际状态。抓勺和移动到源碗上方使用 cuRobo，进入、推进和抬升时 TCP 必须直线。舀球判定要求球与勺子实际接触、抬升和保持后球心位于勺槽内；倾倒判定要求同一批球在目标碗内。球只靠 MuJoCo 接触运动，没有被 weld 或手动传送。

## 2026-10-19 旧碰撞代理结果（历史对照）

以下两条旧 run 使用较高侧壁和单段关节倾倒，终态成功但倾倒过程出现较大绕行，不能作为当前动作质量的验收依据。

### 橙色勺

`runs/spoon_scoop_task/full_verified_20261019` 从 fresh reset 连续完成：推进时勺槽内 9 球，抬升后 5 球，水平保持后 4 球，搬运到目标碗上方仍有 4 球；绕世界 X 轴倾斜 90° 并静置 1 秒后，这 4 个球都在目标碗内。真实 cuRobo 轨迹和 MuJoCo 回放均保存在该 run。目标碗内 4 个球心到碗中心的水平距离均小于 2.8 cm；[俯视验收图](../../../runs/spoon_scoop_task/full_verified_20261019/video_frames/pour/target_top.png)能同时看到四球。示例录像：

- [抓勺](../../../runs/spoon_scoop_task/full_verified_20261019/spoon_pick_actual.mp4)
- [盛球](../../../runs/spoon_scoop_task/full_verified_20261019/spoon_scoop_actual.mp4)
- [倾倒](../../../runs/spoon_scoop_task/full_verified_20261019/spoon_pour_actual.mp4)
- [完整任务](../../../runs/spoon_scoop_task/full_verified_20261019/spoon_full_actual.mp4)

### 粉色勺

`runs/spoon_scoop_task/pink_full_verified_20261019` 用**默认过渡高度**从 fresh reset 连续完成：进入源碗时勺槽内 2 球，推进后 6 球，抬升后仍有 6 球，水平保持后 4 球，搬运后仍为 4 球；倾倒并静置 1 秒后，这 4 个球全部在目标碗内。球心到目标碗中心的水平距离为 2.36–2.79 cm，见[俯视验收图](../../../runs/spoon_scoop_task/pink_full_verified_20261019/video_frames/pour/target_top.png)。

- [抓勺](../../../runs/spoon_scoop_task/pink_full_verified_20261019/spoon_pick_actual.mp4)
- [盛球](../../../runs/spoon_scoop_task/pink_full_verified_20261019/spoon_scoop_actual.mp4)
- [倾倒](../../../runs/spoon_scoop_task/pink_full_verified_20261019/spoon_pour_actual.mp4)
- [完整任务](../../../runs/spoon_scoop_task/pink_full_verified_20261019/spoon_full_actual.mp4)

无 weld 全链和真机仍未验收。勺槽的 MuJoCo 碰撞代理使用底面、低侧沿、背沿和前沿；SAM3D mesh 仍作为视觉资产。
