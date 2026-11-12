# Bowl 在线 VisualRun 接入（2026-10-12）

## 结构

- `applications/microwave/object_visual_session.py`：已有 `SkillPlan` JSON → `VisualRun`；保存原始轨迹、失败前缀与 attach/detach，不重新生成目标或更改 adapter 策略。
- `visualization/viser_sim/object_app.py`：复用 `ViserVisualScene`、`VisualPlayback`；规划在主线程 tick 同步运行，GUI/tick/seek 以 RLock 串行；编辑显式提交并重建 `prepare_pick_bowl_scene`。新 revision 删除旧 Viser 节点，避免旧 collision/path 留在画面。
- `tools/viewers/_visual_object_cli.py`：统一 Skill CLI 分发，门仍进入原 `DoorVisualApp`。`view_task.py` 与 door 会话不变。
- `data/task_1/visualization/bowl_visual_debug.json`：显式可复现默认场景；单位 m/rad，来源在文件中。

`object_visual_run(report, selection, root, initial_attachments=())` 接受 `stage_contract/stages` schema。成功运动必须提供至少两点等间隔 planner 轨迹；不能把失败候选转为动画。Pick 闭爪只有携带 `visual_attachment` 证据时才改变附件状态。Insert 从同 revision Pick 的实际终态开始，release 后的 detach 保留实际释放位姿。

`hold` 映射为不改变附件的状态事件，不生成零运动伪轨迹。`collision_allowed` 不把 `contact_masks_both_zero` 的 visual-only 几何和活动臂原碰撞球排除记录解释为整体忽略物体；transfer 球代理单独由 attachment 事实表示。

ObjectApp 暴露 `controller.run`、`playback`、`session.config`、`camera`、`time_s`、`_show_frame()`，供现有 `record_visual_run` 复用。规划开始/结束时不把计算耗时推进到回放时间；无轨迹时禁用回放。`planning` 为公开状态；`_start_plan()` 仅准备，`tick()` 每次执行一个同步 Skill。单次 GPU 求解不能中途取消，停止请求在步骤之间处理。

## 测试与真实探针

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/visualization/test_object_visual_session.py -q -p no:cacheprovider
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
```

测试先红后绿，覆盖真实轨迹与附件跟随、失败前缀、释放位姿、场景编辑导致候选/碰撞哈希更新、真实 Viser 旧节点清理、visual-only 排除分类。ROS 自动 pytest 插件依赖的 `lark` 不在此 conda 环境，故禁用无关插件自动加载。

真实 GPU 与 Viser 证据：

- `runs/affordtoolact_migration/object_online_gpu_probe/`：新建场景在线 Pick 前置 + Insert 六段成功。
- `runs/affordtoolact_migration/object_online_viser_verified/`：调用在线界面的规划按钮；Pick 六段成功；碗 X 修改 5 mm，GT/collision 新 revision；再规划 Insert 六段成功，终态附件为空。
- 上述目录 `probe_summary.json`、`pick_success.png`、`insert_after_edit_success.png` 保存事实与图片；截图已实际打开检查。Chromium 第一次可绘制不代表全部 glTF 到位，截图需等待完整模型。
- `/tmp/probe_object_online.py` 与 `/tmp/probe_object_viser_verified.py` 为本轮探针脚本，长期复现副本见 run 目录内 `probe.py`。

GPU 成功只证明此场景与 adapter 所声明策略下生成轨迹。旧 insert 的 `ignore_scene_collision=True` 段仍只检查自身碰撞，不能宣传成全 world 碰撞验收。

## 暂未接入

- Extract adapter 会替换桌面为 `corrected_scene.mjb`，需要用该模型及 `corrected_initial_qpos.json` 建立新共享场景，不能直接沿用 Pick/Insert case。
- Knob adapter 输出格式与 `SkillPlan` 不同，共享 `VisualState` 尚无通用 knob 关节通道；未把只有机械臂运动的片面动画暴露为完整旋钮 Skill。
- Knowledge 浏览不调用纯 IK；Bowl/微波炉/桌面支持 XYZ 位置和旋转数值与拖动，桌面支持 XY 长宽。草稿只改变 visual，应用前隐藏旧 collision 并禁用回放。

最终同步版本追加复验：

```bash
PYTHONPATH=. python tools/diagnostics/verify_object_skill_online.py \
  --output runs/affordtoolact_migration/object_online_sync_final --port 8114
```

`--output` 需使用新目录。探针通过真实界面规划入口，验证拖动碗 X +5 mm、桌面 X 长度 +20 mm 后重建并重新规划；检查闭爪后的实际 `T_tcp_bowl` 与 VisualPlayback 推导偏移最大矩阵误差小于 1e-5，保存该数值、场景版本、碰撞哈希、阶段轨迹和最终 attachment。最终证据以 `object_online_sync_final/` 为准；前面带 `verified` 的目录是后台执行版历史探针，不能作为最终同步行为的证据。

本轮最终验证：`745 passed, 1 skipped`（完整套件，103.30 s）；本模块 11 项行为测试。同步探针 attachment 偏移最大矩阵误差 `4.1285946994484846e-08`，编辑后桌面长宽 `[0.92, 0.9] m`，碗 X `0.29 m`，revision `2`，放入终态 attachments 为空。

## 在线 object Task 与通用 articulation（2026-10-12）

`ObjectTaskApp` 继承现有同步编辑与播放路径，只替换 Task 选择与说明。
`ObjectVisualSession.plan('microwave_object_insert')` 保存最初机器人/碗状态，并把同一次 Pick 的
`VisualRun.stages` 与从其真实终态规划的 Insert 阶段拼接；由 `PlannedTimeline` 校验机器人边界，
由 `VisualPlayback` 验证阶段 attachment。单独 `insert_bowl` 仍从合法 Pick 终态展示自身阶段。

`VisualStage` / `VisualState.object_joint_positions_rad` 保存通用场景 hinge 弧度映射，
`object_state` 事件显式提交状态；失败阶段不提交，向后 seek 从初态重建。
`KinematicSceneState` 拒绝把 free joint、机器人关节或夹爪驱动当作物体 hinge 写入。
既有 `door_angle_rad` 保持兼容；同一关节通过两种字段传入矛盾值时拒绝显示。

真实验收命令：

```bash
PYTHONPATH=. python tools/diagnostics/verify_object_task_online.py \
  --output runs/affordtoolact_migration/object_task_online_final --port 8115
```

命令要求新输出目录。既有 `object_task_online_final/probe_summary.json` 留有两轮真实成功记录：
初始 revision 0，以及 bowl X +5 mm、桌面 X 长度 +20 mm 后的 revision 1。
探针校验 GT Knowledge 目标随 bowl 等量平移、碰撞快照 hash 更新、旧播放失效、
12 阶段时间有序以及 Pick/Insert 边界仍附着。两种机位的截图为 `contact_a.png`、`contact_b.png`，
采用 B 后的实际 GUI 分支截图为 `task_after_edit_contact.png`。

Extract/Place 暂未挂在线 Task：既有 adapter 导出的 `visual_extract/corrected_scene.mjb` 与
`corrected_initial_qpos.json` 修改了桌面/物体基线，不能把结果直接投影到当前 Pick 源模型。
需要先显式加载修正模型、其初态与 collision revision，并使其进入同一编辑重建流程后再接入。

旋钮 `microwave_knob_rotate` 本批未列入在线 choices。最新同场景真实报告
`/tmp/task1_visual_probe_20261012_legacy85_xy_052_029_knob/knob/skill_rotate_knob_0_result.json`
为 grasp IK 0/6，尚无可验收的成功旋转路径；保留这项失败事实。后续 VisualRun adapter
必须区分 cuRobo 轨迹与旧版末关节直接命令，并无损处理末关节周期 wrap、额外 +1 µs 的
`visual_angle_commit` 帧、立即设开口后等待的夹爪事件，不能用均匀插值伪造旋钮联动。

## 逐分区视觉反馈与候选筛选耗时（2026-10-12）

`ObjectVisualSession` 在 Pick 完成后、`Task1VisualSession` 在每个成功分区完成后，
返回 `partial_run` 和 `partial_playback`。Task Viewer 可先播放这一段真实规划轨迹，
再调用下一个同步 `plan_step()`；播放到分区终态才继续规划，避免预览修改
MuJoCo 正运动学状态后使下一段起点错位。该模式只改变 Viser 的展示顺序，
不改变 Skill、Task adapter、cuRobo 请求或候选顺序。一个分区的快速预览
最长约 2 秒；关闭界面开关即可连续计算，尽快得到整条任务轨迹。单个 Skill
仍由原 adapter 同步规划，尚未拆成内部阶段的边规划边播放。

抓碗的 40 个候选已有 batch IK，但 `CuroboPlannerAdapter.evaluate_candidates()`
此前没有传 `diagnose_failures`，导致 batch 中已有有效抓取时，仍对每个失败姿态
调用昂贵的逐物体碰撞归因。现在该路径使用 `when_no_success`：有可用候选时
跳过其余失败候选的逐项归因；整批没有有效候选时保留详细诊断。直接调用
`SynchronousPlanner.solve_ik_batch()` 的默认布尔行为不变。

本机 RTX 4090、同场景 seed 123 的单次诊断探针：修改前热调用 Pick 为约
5.66 秒，其中 7 次逐候选碰撞诊断约 1.96 秒；修改后热调用约 3.69 秒，
诊断调用为 0，选中的抓取候选未变。修改后 Bowl Insert Task 约 7.02 秒，
旋钮 Task 约 11.15 秒；旋钮已记录阶段求解约 7.63 秒，仍需单独分析其运动
规划耗时。这些是单次本机结果，不是跨场景统计，也不能把快速预览模式的
总界面墙钟时间当作规划计算耗时。

在线 Task 的真实 GPU + Viser 探针在 `/tmp/gene-agent-progressive-20261012-230341/`：
先显示 Pick 6 个阶段，再从其终态规划 Insert，最终 12 个阶段成功。
截图 `/tmp/gene-agent-object-task-preview-camera.png` 已实际打开检查；它只证明
初始全景与抓取姿态预览，不证明执行接触或物理夹持。

Task1 第一分区真实 GPU + Viser 探针位于 `/tmp/gene-agent-task1-preview-20261012-230738/`：冷调用约 8.28 秒，生成 10 个阶段，其中包含真实轨迹；界面立即进入该分区快速预览，分区索引变为 1，预览结束后按预期停在下一分区规划边界。该探针随后主动取消，不能作为后续六个分区或完整 Task1 成功证据。
