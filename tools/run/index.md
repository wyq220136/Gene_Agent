# 运行入口索引

按任务选择运行入口；不从本页推断仿真或真机已经通过验收。

| 文件 | 用途 |
|---|---|
| [plan_visual.py](plan_visual.py) | 无 MuJoCo、独立选择场景碰撞的 cuRobo Task / 单 Skill 规划，结果非物理成功 |
| [__init__.py](__init__.py) | 按用途归位的命令入口 |
| [annotate_concept.py](annotate_concept.py) | RGB + 米制 mesh + 用户要求 → 有界 Agent 功能 Concept 标注 |
| [build_multi_task_scenes.py](build_multi_task_scenes.py) | 从 my_scene_0 的有效重建 mesh 生成清扫、倾倒、舀取场景预览 |
| [build_reconstructed_tabletop_scene.py](build_reconstructed_tabletop_scene.py) | 从冻结的 brush_0 RGB-D 自动补齐三物体感知，并生成 Pika 桌面场景 |
| [capture_realsense_frame.py](capture_realsense_frame.py) | 用一台 RealSense 拍一帧 RGB-D，保存为场景构建器的输入 |
| [check_pick_bowl_interface.py](check_pick_bowl_interface.py) | 用现有抓碗候选验证同步执行；可在执行过程中录制 MuJoCo 视频 |
| [evaluate_experiment.py](evaluate_experiment.py) | 从同一个 MuJoCo 检查点顺序比较少量微波炉参数补丁 |
| [evaluate_sweep_cube.py](evaluate_sweep_cube.py) | 在固定 seed 列表上从 fresh reset 评测三方块清扫，不跳过失败样本 |
| [execute_sweep_realworld.py](execute_sweep_realworld.py) | 复用微波炉 Runtime 执行旧刷清扫；默认只检查文件，不连接电机 |
| [preview_sweep_from_camera.py](preview_sweep_from_camera.py) | 一条命令：首帧 RGB-D → SAM3 → FoundationPose → 抓刷/清扫 cuRobo → Viser |
| [run_novel_tool.py](run_novel_tool.py) | 外部 Novel Tool 原生场景、原 evaluator 与共享 Pick 前缀，见[接入流程](../../docs/task_development/native_benchmark.md) |
| [run_cup_pour_task.py](run_cup_pour_task.py) | 连续执行杯把抓取、向碗内倾倒和放回杯子 |
| [run_door_visual.py](run_door_visual.py) | 无 GUI 地运行同一门 Skill 在线纯视觉会话，供 Codex 取结构化反馈 |
| [run_experiment.py](run_experiment.py) | 在线执行一个微波炉物理实验，并保存逐阶段实际状态与诊断；--stop-after 支持抓碗前缀 |
| [run_microwave_task1_physical.py](run_microwave_task1_physical.py) | 完整微波炉或 --subtask 独立/组合仿真；单项 --stop-after 阶段停止、prefix/current 准备与实际视频 |
| [run_microwave_task1_realworld.py](run_microwave_task1_realworld.py) | 微波炉现场独立操作、load/unload 组合与完整 Task；默认仅预检首段 |
| [run_spoon_pick.py](run_spoon_pick.py) | 在 scoop_orange / scoop_pink 场景从新 reset 执行抓勺 Skill；--stop-after 支持阶段前缀 |
| [run_spoon_scoop_task.py](run_spoon_scoop_task.py) | 连续执行抓勺、盛球、倾倒；可在指定 Skill 后停止验收 |
| [run_sweep_cube.py](run_sweep_cube.py) | 完整清扫或 --skill pick/sweep 独立运行；--target/--stop-after 限定本次范围 |
| [run_sweep_long.py](run_sweep_long.py) | 从 fresh reset 运行竖立长刷抓取与三方块入簸箕任务 |
| [run_sweep_realworld.py](run_sweep_realworld.py) | 人工持刷、单方块清扫的现场只规划入口；无运动执行选项 |

- [workbench.py](workbench.py)：启动本机任务开发工作台，复用仿真和查看入口。

- `run_benchmark.py`：显式选择五类 benchmark 与兼容 policy，分别传场景条件和策略参数；`--inspect` 不执行。见 [Benchmark 入口](../../benchmarks/index.md)。
