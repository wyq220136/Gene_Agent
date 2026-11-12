# 旋钮在线会话开发说明

`KnobVisualSession` 基于 `ObjectVisualSession` 的 GT 构建、编辑、FK 和输出生命周期。每轮 build 从当前微波炉变换重新调用既有 `generate_grasps`；profile 只写本轮输出副本，不修改 Knowledge 源文件。

`plan_step()` 同步调用 `run_rotate_knob_0_planning(case=当前场景)`，通过 `task1_visual_session.knob_visual_run` 保留实际轨迹、gripper 状态事件与延迟角度提交，再交给共享 `VisualPlayback`。结束后恢复名义 qpos，防止 Knowledge 预览读取规划终态。取消后重启会创建新 revision，避免复用已存在的 adapter 输出目录。

`rotate_knob` 是单 Skill，`microwave_knob_rotate` 是仅含该 Skill 的子任务。两者都由真实合法 home 初态求解，未借用完整 Task1 的已保存结果。运行目录中的 `scene_config.json` 可通过 CLI `--config` 重建。

`KnobVisualApp` / `KnobTaskVisualApp` 共用 `ObjectVisualApp` 的 RLock、主线程同步 tick、编辑器与 `ViserVisualScene`。旋钮 Knowledge 预览使用 `knob_candidates`，不混用 Bowl 候选。零时长事件仍可单独选择，不人为增加播放时间。

原 adapter 的 `scene_revision` 当前固定写 0；在线 `VisualRun.scene_revision` 以当前 `case.revision` 为准。原报告保留原样，避免伪装 planner 内部的阶段编号。连续运动及碰撞快照均来自该次 adapter 调用。

回归：`tests/visualization/test_knob_online_visual.py` 覆盖 GT/Knowledge 重建、默认六候选、取消后重建、失败角度不变、零时长事件及两种入口的真实非线性采样保真。另运行 `tests/visualization/test_task1_online_visual.py` 检查共享 knob 转换器。

2026-10-12 实测证据：`/tmp/gene_agent_knob_online_verified/probe_summary.json`。默认 Skill revision 0 和默认 Task revision 1 都实际 GPU 规划成功，各 18 VisualStage、18 个有效碰撞快照，末端旋钮 45°。MW X 增加 1 mm 后 revision 2 在 `rotate_45` 返回 `path_not_found`（solver 候选触及关节限位）；保留 14 阶段及成功前缀，旋钮保持 0°，没有提交失败目标。10 个 MW 碰撞 geom 全部同向移动 1 mm。切换旧 profile 的 revision 3 为 0/6 grasp IK，失败状态和原始报告可读。

实际打开检查的截图：`knob_overview.png`、`knob_contact.png`、`knob_task_after_edit.png`、`knob_legacy_result.png`。侧视能看清上旋钮和指尖；旧 IK 失败图里的 home 手臂遮挡部分旋钮，但没有伪造接触位姿。关联 CPU 回归 43 passed，日志 `/tmp/knob_online_cpu_final.log`；旋钮专项 7 passed，日志 `/tmp/knob_online_unit_final.log`。
