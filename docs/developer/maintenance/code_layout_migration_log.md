# 代码目录迁移记录（2026-10-07）

本文保留迁移当日实际运行的旧命令作为历史证据；顶层 `experiments/*.py` 入口现已删除。当前命令请查看 [`tools/README.md`](../../../tools/README.md) 与 [`experiments/README.md`](file_layout.md)。

依据 迁移计划（历史材料已本地归档）。本轮只移动实现和入口，保持运行语义。迁移由主 Agent 串行完成；当前协作树内其余 Agent 均已结束，没有同时写入待迁移文件的进程。

## 批次 0：基线

- 工作区并非干净 checkout：大量已有修改、未跟踪文件，以及被 `.gitignore` 忽略的 `data/`、`tests/`。迁移在原工作区进行，避免 worktree 丢失现有未提交输入。**Ruling：** 不创建新 worktree；代价是必须逐文件核对并保护其他改动。
- 完整源备份：`/tmp/gene_agent_layout_pre_20261007.tar.gz`；434 个非缓存文件的 SHA-256 清单：`/tmp/gene_agent_layout_pre_20261007_hashes.json`。包含当前被 ignore 的测试和 task 配置；不能仅靠 Git 恢复。
- 相关基线测试：`test_pick_bowl_skill_feedback.py`、`test_door_skill.py`、`test_rotate_knob_skill.py`、`test_skill_run_replay.py`、`test_skill_inspector.py`、`test_skill_mujoco_viewer.py`、`test_motion_linear.py`、`test_motion_runtime_bridge.py`、`test_motion_gripper.py`、`test_realman_pika_profile.py`：**57 passed**（`curobo`，允许 Viser localhost/GPU）。
- 代表 run：`runs/pick-bowl-synced-20261007` 为 actual/physical pick 成功，可加载 6 阶段；`runs/open-door-dynamic-20261007-01` 为 planned 运动学开门成功，可加载 10 阶段；`runs/rotate-knob-0-calibrated-20261007` 为 planned 运动学旋钮成功，可加载 15 阶段；`runs/pika-pick-bowl-camera-stable` 报告 `success=false` 且既有 ReplayTimeline 不支持其格式；`runs/pick-bowl-interface-v11-failure-bundle` 为失败 run，旧报告缺实例 ID，既有 ReplayTimeline 无法加载。后两项为迁移前事实，不作为新回归。
- 路径依赖：任务装配的三个 runner 用 `Path(__file__).parents[2]` 指项目根；cuRobo `core/gripper/pika` 也有同类定位；脚本有子进程启动与相对 `__file__` 资源路径。每批须在原目标不变的前提下修正层级。

## 预定映射和所有者

- 批次 1：`feedback/{pick_bowl_planning,pick_bowl_skill,door_skill,rotate_knob_skill}.py` → `applications/microwave/`，主 Agent。调用方包括 `experiments/`、`visualization/`、`tests/`；通用 `feedback` 文件不动。
- 批次 2：`motion/{curobo,scene,runtime_bridge,gripper,pika}.py` → `motion/curobo/{core,scene,runtime_bridge,gripper,profiles/pika}.py`，主 Agent。保留 `gene_agent.motion.curobo` 包 API；内部路径直接改调用方。
- 批次 3：长期命令 → `tools/`，研究脚本 → `experiments/{probes,benchmarks,robotwin}/`；旧用户命令保留薄入口，主 Agent。完整脚本清单在动该批前补录。

## 批次 1：任务装配

- 四个任务文件已整文件移入 `gene_agent/applications/microwave/`，仅把项目根层级由 `parents[2]` 改为 `parents[3]`、修正 Python 导入；`applications` 与 `microwave` 补了包声明。
- `gene_agent/`、`experiments/`、`tests/` 的旧 `gene_agent.feedback.{pick_bowl_planning,pick_bowl_skill,door_skill,rotate_knob_skill}` 导入扫描为空。公开业务函数签名、输出报告字段未修改。
- Pick、门、旋钮、Knowledge 和 Viser 相关测试 **51 passed**；三个带 `ROOT` 的 runner 均解析到同一仓库根。首次资源根检查误以为 `pick_bowl_planning.py` 导出 `ROOT`，该模块本来没有该常量；改为检查真正使用它的 `pick_bowl_skill.py` 后通过，不是功能回归。

## 批次 2：cuRobo 后端

- `curobo.py` 整体归入 `curobo/core.py`，`scene.py`、`runtime_bridge.py`、`gripper.py` 和 `pika.py` 分别归入约定目录。`motion/curobo/__init__.py` 显式导出原有调用方实际使用的 13 个公开符号；`motion/__init__.py` 不改变公开名称。原 `motion/scene.py` 等内部路径不留转发文件，调用方已改新路径。
- cuRobo 相关单元/契约测试 **67 passed**。首次 GPU 烟测暴露 `core.py` 的五个延迟导入仍写成 `.collision`，在新包层级下报 `ModuleNotFoundError`；改为 `..collision` 后重跑通过。**Ruling：** 将它视为目录层级修复，不改碰撞算法或参数；代价是需靠真实 GPU 测试覆盖延迟导入。
- `experiments/check_motion.py --warmup-runs 0 --measure-runs 1`：FK、IK、普通规划成功，故意包围的障碍被判 `start_collision`；结果 `/tmp/layout_motion_smoke_fixed_20261007/motion_result.json`。
- `experiments/check_motion_linear.py --seed 123`：真实场景净空 TCP 直线规划成功，具名障碍 `test_linear_blocker` 被判起点碰撞；结果 `/tmp/layout_linear_smoke_20261007/`。
- `experiments/check_realman_pika_motion.py`：Pika 四组 FK/TCP 误差 0，短路径成功，碰撞检查 PASS；结果 `/tmp/layout_pika_motion_smoke_20261007/summary.json`。
- 同时修正四个诊断脚本内参与 `code_sha256` 计算的旧源码文件路径；不改既有报告格式或历史 run。

## 批次 3：命令入口与实验

- 迁移清单：57 个 Python 文件及一个 RoboTwin shell 脚本。`tools/viewers/` 15 个，`tools/editors/` 5 个，`tools/diagnostics/` 20 个，`tools/run/` 6 个；`experiments/probes/` 5 个，`experiments/benchmarks/` 1 个，`experiments/robotwin/` 5 个。每个文件同名归位；完整旧→新映射可由旧脚本模块路径及新目录同名文件直接定位。
- 所有旧顶层 Python 命令保留约 8 行薄入口，只用 `runpy` 转发到唯一实现；旧 RoboTwin shell 命令也转发。**Ruling：** 顶层仍能列出旧脚本名，但没有第二份业务实现；代价是 `experiments/` 文件列表尚有兼容入口，将来删除须另行评估用户使用情况。

| 新目录 | 原顶层同名脚本 |
|---|---|
| `tools/viewers/` | `capture_knowledge_viser.py`, `capture_motion_debug.py`, `capture_pick_bowl_planning.py`, `capture_skill_run.py`, `capture_viser.py`, `export_pick_bowl_sequence_video.py`, `inspect_skill_run.py`, `record_microwave_knowledge_video.py`, `record_skill_run_video.py`, `render_pick_bowl_stage.py`, `render_realman_pika_preview.py`, `view_motion_debug.py`, `view_pick_bowl_planning.py`, `view_skill_mujoco.py`, `view_skill_run.py` |
| `tools/editors/` | `edit_concept.py`, `tune_grasp_profile.py`, `tune_scene_grasp.py`, `view_grasp_knowledge.py`, `view_microwave_knowledge.py` |
| `tools/diagnostics/` | `check_body_collision_audit.py`, `check_collision_overlay.py`, `check_environment.py`, `check_gripper_actuation.py`, `check_gripper_geometry.py`, `check_gripper_visual.py`, `check_knob.py`, `check_knowledge.py`, `check_microwave_viser_interaction.py`, `check_motion.py`, `check_motion_batch.py`, `check_motion_debug.py`, `check_motion_linear.py`, `check_pick_contract.py`, `check_reachability.py`, `check_realman_pika_motion.py`, `check_scene_collision.py`, `check_skill_record_sync.py`, `check_skill_view_consistency.py`, `check_visual_quality.py` |
| `tools/run/` | `check_door_skill.py`, `check_microwave_layout.py`, `check_pick_bowl_interface.py`, `check_pick_bowl_planning.py`, `check_pick_bowl_skill.py`, `check_rotate_knob_skill.py` |
| `experiments/probes/` | `check_bowl_contact.py`, `check_closed_loop_motion.py`, `check_pika_pick_bowl.py`, `check_runtime_bowl_feedback.py`, `replay_closed_loop_motion.py` |
| `experiments/benchmarks/` | `benchmark_pregrasp_speed.py` |
| `experiments/robotwin/` | `render_robotwin_pose_overlay.py`, `replay_robotwin_after_lift_gt.py`, `robotwin_code_as_policy.py`, `robotwin_foundation_pose.py`, `view_robotwin_pose_open3d.py` |

- 路径修复仅调整 `__file__` 目录层数、同伴脚本包导入、截图子进程位置及按路径加载模块的测试。历史运行记录、算法和 CLI 参数未修改。内部动态启动 `capture_viser` 与 motion debug 的脚本已指向新路径。
- 检查 5 对代表 CLI 的 `--help` 与退出码完全一致；旧 `experiments/capture_skill_run.py` 已实际调用新 Viser 子进程，生成 `/tmp/layout_viser_capture/04_lift_actual_0.500.png`，人工查看包含场景、机器人和 Skill 信息。
- 迁移后受影响测试首次为 165 通过、4 失败：RoboTwin 测试仍按旧文件路径加载模块，得到薄入口而非实现。修正该路径后同一组为 **169 passed**。这证明保留旧 CLI 不等于保留旧模块导入；源码及测试使用新模块路径。
- 新 `tools/viewers/capture/capture_knowledge_viser.py --tool lab --kind ring` 真正启动 Knowledge 编辑器并生成 `/tmp/layout_knowledge_ring.png`；人工查看候选 gripper 与 ring 参数控件均显示。
- 当前推荐命令已写到 `tools/README.md`、`experiments/README.md`、根 README 和 `docs/user/`；历史 `docs/developer/` 证据叙述不批量改写。

## 批次 4：整体验收

- 全套 `tests` 加 Knowledge 可视化工具测试：**543 passed, 1 skipped**（`MUJOCO_GL=egl`，`curobo`）。受影响测试集合 169/169；无减少收集来消除失败。
- 旧、新 `view_skill_mujoco.py` 对 `runs/pick-bowl-synced-20261007` 的 `04_lift` actual 0.5 秒、`gripper_side` 视角各输出一帧：两份 JSON 除图像路径外完全相同，PNG 尺寸 960×720，仅 5 像素相差 1 灰度值；实际查看图像有双指和碗。
- 旧截图入口 `experiments/capture_skill_run.py` 真正启动浏览器与新 Viser viewer 子进程；所选阶段 `04_lift`、actual 0.5 秒的截图生成并人工查看。
- 新 `tools/run/check_pick_bowl_interface.py` 以基线候选 `bowl_ring_00_00`、重力补偿和录像重新执行 MuJoCo：`success=true`、`contact_success=true`、`physical_pick_success=true`，阶段顺序仍为 open→pregrasp→approach→close→lift→hold；视频在 `/tmp/layout_pick_physical_20261007/visual/skill_execution.mp4`。抬升高度 0.05098 m（基线 0.05093 m），hold 高度 0.06506 m（基线 0.06494 m）。
- 新 `tools/run/check_microwave_layout.py --only both` 共用摆位 (0.52, 0.32) m：门 10/10 阶段、knob_0 15/15 阶段均规划成功；记录位于 `/tmp/layout_microwave_20261007`。这是运动学规划成功，不宣称门、旋钮物理接触或真机成功。
- Pika 独立抓碗命令在录像器初始化时抛出 `ValueError: gripper_side 录像需要活动夹爪组和 robot_metadata`。迁移前备份中的 Pika 调用与录像器检查代码完全相同，因此这属于基线已有的不兼容；不在目录搬迁中顺手改动接触实验。Pika FK/TCP、小路径和碰撞诊断已在新路径重跑通过（`/tmp/layout_pika_motion_after_tools_20261007`）。**Ruling：** 保留 Pika 实验当前失败边界；代价是该实验仍不能完成物理抓碗，后续需单独修复录像参数。
- 脚本内资源定位和新模块导入扫描无遗留旧路径；RoboTwin 原环境的完整外部任务未复跑，只验了本仓库单元契约、CLI 和 shell `--help`。**Ruling：** 不把目录整理扩展成外部 RoboTwin 实验；代价是该外部环境的视觉/物理任务仍需独立验收。
- 长期未解决：统一 Skill/RunRecord、通用场景导入、sim2real 后端、批量扰动评估、Knowledge 候选整理、门/旋钮接触物理。目录归位本身均不视为解决。

## 最终自审

- 按迁移计划逐项核对：四个微波炉装配文件、五个 cuRobo 文件、57 个 Python 命令及一个 shell 命令均有唯一新实现；旧用户命令仅透传。没有移动资产、历史 runs、数据或环境。
- 复查时未发现本轮新增的重要运行缺陷；发现的 Pika 录像参数错误由迁移前备份证实为基线已有，已按原事实记录。RoboTwin 外部完整任务和本地交互式拖拽没有在本轮重跑，不能从命令和截图推断其验收。
- `git diff --check` 针对本轮相关的已跟踪路径通过；全仓命令只报告已有 `prompts.md` 尾随空格，本轮没有修改该文件。Python AST 解析 371 文件通过，旧顶层 57 个 Python 薄入口最多 9 行。
- 本轮未提交：共享工作区已有大量并行修改、未跟踪和 ignore 的必需文件；避免把他人工作混入一次提交。**Ruling：** 保留可审查的工作区与迁移日志，不进行全仓暂存/提交；代价是跨 checkout 复现需后续按清单分批整理版本管理。
- Final review: self-review；当前任务没有请求额外 Agent 审查。
