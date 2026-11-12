# Task1 在线 Viewer 实现边界

`Task1VisualSession` 构造新的 `visual_catalog.build_task1_catalog`，并由每次 `plan_step` 同步运行一个 partition。`Task1VisualApp` 复用 `ObjectVisualApp` 的 RLock、编辑器、VisualPlayback 和 ViserVisualScene，不运行第二套仿真，也不把已有报告回放当作在线规划。

每个门 phase 完成后，通过 catalog 的只读 `on_door_phase_complete` 回调立即复制实际 `PlannedSegment` 与状态通道；回调发生在下一次动态候选筛选之前。即使下一阶段所有候选 IK 无解并取消序列，成功前缀仍保留，不依赖被清空的最终 timeline。partition 前缀避免两轮同名门阶段冲突。原 Task 汇总 JSON 只含门诊断，不能单独恢复完整轨迹；新会话额外保存 `visual_run_samples.json`，包含完整机器人轨迹、状态通道和场景事件。Bowl/Extract 复用已有 VisualRun 转换，Knob 从已有真实 samples/result 转换。所有成功 partition 的回放终态均与 adapter 的关节、开口、门角、旋钮角、物体 pose 和 attachment 逐项核对。

Knob 的零时长开合夹爪按原顺序提交，等待阶段保持其实际时长；最后 `visual_angle_commit` 比轨迹终点晚 1 µs，新共享字段 `state_update_at_end` 保留该离散提交。不会把旋钮插值成跟随机器人持续转动。末关节直接命令单独标注；若出现尚未支持的周期 wrap，拒绝无损转换并保存结构化失败，不造连续轨迹。

`build_task1_catalog(scene_edits=..., scene_revision=...)` 负责同步重建同一 MuJoCo 场景、Knowledge、碰撞和 Task 配置。Task1 固定关门及机器人 home 初态，不允许非零门初角/机器人关节编辑。桌面采用 catalog 当前共享模型的实际 geom pose；不能回读源 XML 的旧桌面 pose 覆盖它。

探针命令（必须使用新目录）：

```bash
PYTHONPATH=. python tools/diagnostics/verify_task1_online.py \
  --output runs/affordtoolact_migration/task1_online_final \
  --port 8116 --edit-replan
```

针对测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest tests/visualization/test_task1_online_visual.py tests/visualization/test_object_task_visual.py \
  tests/visualization/test_viser_sim_model.py tests/visualization/test_viser_sim_state.py \
  -q -p no:cacheprovider
```

默认版本明确为 `legacy_viser` + MW XY [.52,.31] + `tip_x_5_azimuth_240`。它保留两个外部门角事件，与用户最终希望的显式25°→90°推门任务有不同的行为；两者不能混称同一成功结果。录制入口由独立 `record_visual.py` 分发，本界面保持 `controller.run`、`playback`、`planning`、`session.config` 等公开接口。

失败报告的 `final_state` 与失败分区 `end_state` 均取自保留的 VisualPlayback 末态，并标注 `end_state_source=last_planned_visual_state`。它描述最后实际规划状态，不使用 cancel 后恢复的初态，也不代表物理执行。候选失败保留原 `goal_ik_no_solution`、候选检查数量和筛选碰撞快照。CPU 回归覆盖前缀逐点一致、冻结数组不受复用影响，以及机器人/夹爪/门/旋钮/物体/attachment 终态一致。
