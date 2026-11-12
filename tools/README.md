# 长期使用的命令

快速导航：[本目录索引](index.md)。

完整文件清单与使用前提见 [运行入口索引](run/index.md) 和
[诊断工具索引](diagnostics/index.md)。一次性旧实验脚本归入根目录 `archive/tools/`，
不跟踪；需要复现时恢复原路径和历史输入，不再保留转发脚本。
`runs/` 仍有少量输入依赖，清理前先看 [运行记录清理说明](../runs/index.md)。

**给人类视频反馈的统一入口：** `python tools/viewers/recording/record_skill_run_video.py --run <run> --output <本轮目录>/operation.mp4`。默认 actual/EGL；完整微波炉自动全景＋近景。默认只导出MP4/JSON，快速检查后交付，不额外批量截图。实现、效率与验收规则见 [视频反馈说明](../docs/user/visualization/video_feedback.md)。

刷子真机接入只规划（默认原蓝色旧刷）：`python tools/run/run_sweep_realworld.py --demo --output runs/realworld_sweep/my_demo`；`python tools/viewers/view_skill.py --skill sweep_cube --run runs/realworld_sweep/old_brush_20261023` 或 `python tools/viewers/view_task.py --task sweep_cube_into_dustpan --run runs/realworld_sweep/old_brush_20261023` 查看预测轨迹。长刷显式选择 `--config config/realworld/sweep_long.example.json`。两者复用同一个 Sweep Skill，没有真机运动开关。现场准备、只读采集和验收边界见[刷子真机接入说明](../docs/user/sweep/sweep_realworld_plan_only.md)。

Pika 坐标核对：`python tools/viewers/view_pika_frames.py --report runs/pika_frames/gpu_20261023/report.json` 查看原始 mesh、法兰、模型原点、控制参考点、仿真 TCP 与指垫中心；`python tools/diagnostics/check_pika_frames.py --output runs/pika_frames/my_audit` 生成新的离线报告。可显式增加厂家离线 Algo 与 GPU cuRobo FK；不连接或移动真机。使用和当前差异见 [Pika 坐标说明](../docs/user/robots/pika_frames.md)。

相机标定只需先看[两条路线操作说明](../gene_agent/perception_modules/camera/calibration/README.md)：

**查看最新标定结果：`python -m gene_agent.perception_modules.camera.calibration results`，或打开 `data/calibration/latest/`。**

标定后基座XYZ裁剪与桌面检查：`python -m gene_agent.perception_modules.camera.calibration inspect`，默认打开已保存的最新第三相机点云，六边界交互裁剪、RGB覆盖与桌面平面诊断；[操作说明](../gene_agent/perception_modules/camera/calibration/inspection/README.md)。

- **路线B（手眼＋移板，2＋5）**：`python -m gene_agent.perception_modules.camera.calibration handeye --help`，再运行 `...calibration moving-board --help`；已有有效手眼时跳过第一步。
- **路线A（量尺＋板相机，1＋3）**：`python -m gene_agent.perception_modules.camera.calibration manual --help`；只需独立拍板时用 `...calibration board-camera --help`。
- 统一在 `zehao` 环境运行，相机服务在 `sam3` 环境。新采集集中到 `data/calibration/runs/<用途>/`，诊断放 `data/calibration/diagnostics/`；其他标定方案与旧命令从主说明的扩展链接查看。

在仓库根目录、`curobo` 环境运行：

- `viewers/`：Viser、MuJoCo 查看，截图和录像导出。例：`python tools/viewers/view_skill_run.py`；`python tools/viewers/view_skill_mujoco.py --help`。
- 任意 Concept 与物体 mesh 对照：`python tools/viewers/inspect_concepts.py --help`；用法见 [通用说明](../docs/user/knowledge/concept_mesh_inspection.md)。微波炉 URDF 场景可用 `python tools/viewers/inspect_microwave_concepts.py --help`，它复用同一界面。
- 功能 part 标注的快速正交叠图：`python tools/viewers/capture/render_concept_overlay.py --help`；锤子 Agent 试验与可编辑 Viser 命令见[试验说明](../docs/tmp/history/user/hammer_concept_agent_trial.md)。
- RGB + 米制 mesh + 文字要求的有限轮 Agent 功能标注：`python tools/run/annotate_concept.py --help`；输入、产物和锤子自动验收见[用户指南](../docs/user/knowledge/concept_agent_annotation.md)。
- `editors/`：Knowledge、场景、抓取参数的交互编辑。例：`python tools/editors/view_microwave_knowledge.py --help`。
- Concept 快捷编辑：`python tools/editors/edit_concept.py --object hammer` 或 `--object brush_small` 自动读取最近标注的 mesh/Concept；不带参数显示编号菜单，`--list` 只列清单。新 mesh 仍可用 `--mesh`。详见[编辑器说明](../docs/user/knowledge/concept_editor.md)。
- `diagnostics/`：可重复运行的环境、cuRobo、碰撞和记录一致性检查。例：`python tools/diagnostics/check_motion_linear.py --help`。
- 新Skill的Motion示例：`python tools/diagnostics/motion_skill_example.py --help`；[推荐接口](../gene_agent/motion/README.md)。
- cuRobo 快速模式：主要微波炉/真机清扫入口支持 `--collision-mode robot_only`，或配置 `motion.collision_mode`；跳过场景避障与规划 attachment，保留机器人自碰撞。适用范围和离线计时见[规划模式说明](../docs/user/motion/planning_modes.md)。
- `run/`：当前具体任务的运行入口。例：`python tools/run/run_microwave_task1_physical.py --help`。 真机微波炉子任务的配置检查和首段规划见[入口说明](../docs/user/microwave/realworld_microwave_entry.md)；真实设备仍待验收。

桌面新场景示例：`python tools/run/capture_realsense_frame.py --serial <序列号> --output-dir data/realworld/my_scene_0` 用一台 RealSense 拍一帧对齐 RGB-D；`python tools/run/build_reconstructed_tabletop_scene.py` 从冻结的刷子 RGB-D 自动补齐簸箕、绿方块感知并生成单臂 Pika 场景；`python tools/viewers/view_reconstructed_tabletop_scene.py --mode viser` 或 `--mode mujoco` 打开同一 MJCF。输入、参数和能力边界见[新场景指南](../docs/task_development/reconstructed_tabletop_scene.md)。

新照片的三类任务预览：`python tools/run/build_multi_task_scenes.py` 生成五个单工具 MJCF、seed 0/7/21 的初态和截图（蓝色短刷暂跳过）；`python tools/viewers/view_reconstructed_tabletop_scene.py --scene-dir data/realworld/my_scene_0/scenes/sweep_long --mode viser` 查看竖立长刷。功能性 Concept 和验收图片见[三任务场景说明](../docs/user/scenes/my_scene_0_three_task_previews.md)。

橙色勺抓取 Skill：`python tools/run/run_spoon_pick.py --output runs/spoon_pick/my_run` 从新 reset 用 cuRobo 规划、MuJoCo 实际执行；`python tools/viewers/recording/record_spoon_pick_video.py runs/spoon_pick/my_run` 导出实际录像。默认在双指接触后启用明确记录的仿真辅助夹持；纯物理对照使用 `--no-simulation-weld --grasp-handle-x-offset-m 0.006`，名义场景已通过。命令和验收结果见[抓勺说明](../docs/user/spoon/spoon_pick_skill.md)。

橙色与粉色勺盛球倾倒 Task：`python tools/run/run_spoon_scoop_task.py --simulation-weld --until-skill pour --output runs/spoon_scoop_task/my_run` 默认橙色；粉色添加 `--scene data/realworld/my_scene_0/scenes/scoop_pink`，随机初态可指定其 `samples/seed_0` 等子目录。`--until-skill pick|scoop|pour` 用于分阶段验收。用 `python tools/viewers/view_skill.py --skill spoon_scoop --run runs/spoon_scoop_task/my_run` 或 `python tools/viewers/view_task.py --task spoon_scoop_into_bowl --run runs/spoon_scoop_task/my_run` 在 Viser 检查实际 MuJoCo 时间轴。`python tools/viewers/recording/record_spoon_scoop_task_video.py runs/spoon_scoop_task/my_run --segment full` 导出录像，`--segment pick|scoop|pour` 导出单 Skill；`python tools/diagnostics/analyze_spoon_pour.py runs/spoon_scoop_task/my_run` 量化 TCP 绕行、勺头中心偏移和球脱离角度。当前结果与限制见[盛球倾倒说明](../docs/user/spoon/spoon_scoop_task.md)。

绿色杯把抓取、最多 50° 向碗倾倒几颗球并放回：`python tools/run/run_cup_pour_task.py --simulation-weld --output runs/cup_pour_task/my_run`，`--until-skill pick|pour|place` 可分别在 Skill 边界结束；随机初态指定 `--scene data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour/samples/seed_0`。运行记录可用 `python tools/viewers/view_skill.py --skill cup_pick --run runs/cup_pour_task/my_run`、`--skill cup_pour`、`--skill cup_place`，或 `python tools/viewers/view_task.py --task cup_pour_into_bowl --run runs/cup_pour_task/my_run` 查看实际 MuJoCo 回放。`python tools/viewers/recording/record_cup_pour_task_video.py runs/cup_pour_task/my_run --segment full` 导出视频，单段可用 `--segment pick|pour|place`。结果与验收标准见[杯子任务说明](../docs/user/cup/cup_pour_task.md)。

刷子清扫三色方块：用 `python tools/viewers/view_skill.py --skill sweep_cube` 检查 Skill，用 `python tools/viewers/view_task.py --task sweep_cube_into_dustpan` 检查完整 Task；两者读取最新匹配场景的实际 MuJoCo run。默认运行 `python tools/run/run_sweep_cube.py --randomize --seed 10 --output-dir runs/sweep_cube/my_run` 使用 cuRobo 逐阶段轨迹规划与 MuJoCo 物理执行；批量评测用 `python tools/run/evaluate_sweep_cube.py --seeds 10 11 12 --output-dir runs/sweep_cube/my_eval`。仅诊断旧 CPU 路径时加 `--diagnostic-cpu-ik`，其结果不可当作 cuRobo 正式成功率。录像用 `python tools/viewers/recording/record_sweep_cube_video.py runs/sweep_cube/my_run` 导出。详见[清扫 Skill 说明](../docs/user/sweep/sweep_cube_mujoco_skill.md)。

竖立长柄刷清扫三色方块：`python tools/run/run_sweep_long.py --output-dir runs/sweep_long/my_run`；宽范围扰动添加 `--layout-seed 7`，原窄范围加 `--layout-distribution small`，侧缘难例用 `--layout-distribution stress`。在 `python tools/viewers/view_skill.py --skill sweep_long_brush --run runs/sweep_long/my_run` 和 `python tools/viewers/view_task.py --task sweep_long --run runs/sweep_long/my_run` 查看真实执行。分布预览、入口、评测结果与录像见[长刷 Task 说明](../docs/user/sweep/sweep_long_mujoco_task.md)。

实现只在这里维护。旧的顶层 `experiments/*.py` 转发入口已移除，请直接使用本目录或 `experiments/` 专题子目录中的脚本。
这些目录按现有用途归位，并未改变 Knowledge、Skill、Motion、回放记录或成功判定接口。

## 在线检查开关门 Skill（纯视觉）

在 `curobo` 环境运行 `python tools/viewers/view_skill_run.py --skill door_cycle --mode visual`，打开终端打印的 Viser 地址。可以单独检查三个门 Skill，或检查带 Skill 间过渡的 Task1；拖动微波炉与可编辑过渡点后会废弃旧轨迹。点击“规划完整动作并连续播放”会现场调用 cuRobo，并提供可拖动的真实规划时间轴。无 GUI 的同源入口是 `python tools/run/run_door_visual.py --skill push_open --plan`；连续轨迹视频用 `python tools/viewers/recording/record_door_skill_visual.py --skill push_open --source planned --output runs/door-live-visual/push_open.mp4`。操作细节与证据边界见 [用户说明](../docs/user/microwave/door_skill_live.md)。

## 微波炉物理开门（本地 Linux）

当前 Viser 阶段对应的在线 MuJoCo 实验：`python tools/run/run_experiment.py --selection microwave_door_open`。可用 `--until-stage` 暂停并自动保存实际状态检查点，之后用 `--resume` 继续；`--parameters` 接收只影响未来阶段的 JSON 参数补丁。`--resume <检查点> --probe-next` 仅规划并检查下一段的门接触次序，不执行物理；多组参数从同一检查点顺序对照用 `python tools/run/evaluate_experiment.py --help`。门从 fresh reset 完成一轮物理开关的结果见[调试记录](../docs/developer/microwave/evidence/door_physical_scene_alignment_2026-10-13.md)；早期只到首段推门的记录不再代表当前能力。

## 微波炉完整任务（本地 Linux）

`python tools/run/run_microwave_task1_physical.py` 从 fresh reset 连续执行开门、放碗、关门、旋钮、再次开门、取碗、最终关门。加 `--video` 在结束后用共享模块导出完整视频；已有记录使用上面的统一入口。旧命令 `python tools/viewers/recording/record_physical_task1_video.py <运行目录>` 兼容全景与细节分文件，再用 `python tools/viewers/recording/split_physical_task1_video.py <运行目录>` 裁出七段子任务。结果是已声明的辅助夹持与运动学旋钮混合仿真验收；命令和证据说明见[用户说明](../docs/user/microwave/mujoco_task1_physical.md)。

较早、不同摆位的独立开门诊断 `run_microwave_door_open.py` 已归档到本地 `archive/tools/diagnostics/`；对应旧参数保留在 `archive/task_1/configs/door_push_open_candidate.json`。当前门物理执行用 `run_experiment.py`；运行输入和开发产物的边界见 [task_1 配置清单](../data/task_1/readme.md)。

### 清扫首帧自动感知与三维规划预览

```bash
PYTHONUTF8=1 PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/preview_sweep_from_camera.py --camera-param data/calibration/cam_param.json
```

原蓝色旧刷 + 绿方块 + 簸箕；一帧 RGB-D → SAM3 → FoundationPose → cuRobo 抓刷/清扫 → Viser，无运动执行。模型进程自动启动、顺序退出，无需开多个终端。默认只读实际机械臂初态，`--robot-ip` 可覆盖地址；仅离线诊断用 `--assume-standby`。记录继续用 `tools/viewers/view_task.py --task sweep_cube_into_dustpan --run ...` 和 `view_skill.py --skill sweep_cube --run ...` 查看。详见[首帧预览](../docs/user/sweep/sweep_first_frame_preview.md)。

### 旧刷清扫真机执行

`tools/run/execute_sweep_realworld.py` 复用微波炉共享 Runtime；默认只检查 `--run` 的文件，不触硬件。显式 `--execute` 才使能Pika/执行，默认停在pregrasp，支持 `--stop-after approach|pick|task`。`--camera-param` 可自动重新感知并接执行。现场串口/TCP核验配置与命令见[真机清扫说明](../docs/user/sweep/sweep_realworld_execution.md)。此实现尚未真机运动验收。

共享操作流程：旧刷仿真/首帧预览/真机抓取共用阶段循环；设备驱动已迁至 `realworld/devices/`。离线回归和查看命令见 [操作说明](../docs/user/visualization/shared_operation_workflow.md)。完整微波炉 Task 编排尚未统一。

## 本地网页工作台

`PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python tools/run/workbench.py --open` 启动任务选择、运行和视频、场景/部件编辑及只读助手。当前接入旧刷清扫和微波炉，范围见[操作说明](../docs/user/workbench/usage.md)。
