# 门 Skill 在线纯视觉调试接口

运行状态在 `gene_agent/applications/microwave/door_debug.py::DoorDebugSession`。`gene_agent/skills/door_sequence.py` 只把同源 Knowledge 路径标上 Skill 阶段的运动类型和接触策略。Viser 的 `live_door_app.py` 与 CLI `run_door_visual.py` 都调用这一个会话。

目标取自 `build_microwave_knowledge(..., ik_scope="none")`：把手使用 `generate_handle_pull_path`，推开/推关使用 `generate_mode_door_push_path` 和 `add_door_push_transitions`。用户拖动物体后重新运行这些函数，不能把旧 world TCP 整体平移，也不能用物理诊断 profile 的另调局部姿态替代。

阶段标识稳定且包含 Skill 名称。`compose_task1_door_cycle()` 只在完整 Task 中插入 `task/open_to_close/return_default_tcp`；目标由当前机器人 home joint 的 MuJoCo FK 读取，不能写死 Realman pose。三个单独 Skill 均不包含这条过渡。Skill 的可编辑非接触点以微波炉坐标保存，Task 过渡以世界坐标保存。

旧界面中“在推门 Skill 起点增加过渡点”的功能已删除。任意 Task 引导点应由 Task 编排声明，不能伪装成 Skill 内部阶段。`set_microwave_pose`、`set_angles` 和阶段覆盖会增加 revision、清空旧规划并恢复编辑初态。目标预览只改独立 `preview_data`，不污染规划起点。

`plan_next` 从上段终态冻结碰撞场景，按 `plan`/`linear_tcp` 调用现有 cuRobo 近似接口。`plan_sequence` 顺序完成选中 Skill/Task，构造通用 `PlannedTimeline`：每段保存真实关节采样、cuRobo `interpolation_dt_s`、门角和夹爪开度。Viser 的播放和时间拖动都读取该时间轴。规划中途失败时保留成功前缀时间轴，便于复查停止位置。

非接触过渡点可显式启用误差预算，推门接触阶段默认严格。替代目标仍需重新规划；失败返回第一阶段及原始报告。活动门只在明确的门接触/撤离阶段从静态障碍列表中排除；纯视觉时门角按 Knowledge 目标做运动学更新，不宣称物理接触。

回归命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  tests/applications/microwave/door/test_door_live.py tests/motion/test_approximate_ik_plan.py tests/applications/microwave/door/test_push_door_skill.py
```

2026-10-09 的当前 GPU 基线：单独 `push_open` 的 8 段全部成功，时间轴 6.50 秒，记录在 `runs/door-live-20261009-205352/result.json`。完整 Task 已完成 handle pull、push open 和返回默认 TCP 共 19 段，随后 `push_close/00_pre_contact` 返回 `goal_ik_no_solution`；24 个当前局部旋转候选均无 IK 解，记录在 `runs/door-live-20261009-205444/result.json`。这说明 Task 过渡本身可规划，关门 Knowledge 候选/摆位仍需对齐；不能通过删除关门碰撞或伪造 IK 继续执行。

## 2026-10-09 验证边界

连续录制入口 `record_door_skill_visual.py --source planned` 会先调用同一 `plan_sequence`，再按时间轴逐帧截图。`runs/door-skill-continuous-20261009/push_open_overview.mp4` 为 1500×762、10 fps、66 帧；人工抽查了 0.5、3.2、6.0 秒画面，机器人、门角和界面阶段同步变化。manifest 明确标为 `planned_kinematic_timeline`、`physical_execution=false`。
