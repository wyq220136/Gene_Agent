# Pick Skill 执行与证据接口

`gene_agent.skills.executor.run_pick_skill` 是同步编排入口。`GraspSettings` 与任务目录的 `data/task_1/skills/skill_profiles.json` 提供同一组接近距离、开合宽度、抬升距离和保持时间；规划和执行共用 `open → pregrasp → approach → close → lift → hold` 的阶段契约。最终 grasp 的 IK 筛选属于候选尝试，不伪装成执行阶段。`approach` 和 `lift` 要求 TCP 直线运动；每段执行后，下一段读取 MuJoCo 实际终点的 `state_id`。失败立即停止并保存失败阶段。`lift_m=0` 时只验收到双侧指垫接触，不声称物理抬升。

任务装配在 `tools/run/check_pick_bowl_interface.py`。它从当前碗 GT pose 生成候选，读取任务 Skill profile 中的附着球代理，并为 `CuroboRuntimeBridge` 指定 `approach/lift` 的 `linear_tcp`。只有闭爪后观测到有效双侧指垫接触，才在 `lift` 请求中提交附着：碗从静态世界碰撞集合移除，五球代理加入工具碰撞，碰撞快照与附着文件分别保存在 `collision/lift/`、`attachments/lift.json`。代理仅覆盖上半部；物理接触与真实位移由 MuJoCo 单独核验，不能把代理当完整碰撞保证。

`TrajectoryRecorder` 在真实仿真状态上采集全景、活动夹爪侧视和 contact 画面。`open/close/hold` 由执行器接线，运动阶段由 `CuroboRuntimeBridge` 接线；失败的规划阶段不会产生伪造视频。每个阶段的 `replay_samples` 包含起点、低频中间状态和终点，保存完整 qpos、具名关节、TCP、目标物体 pose、仿真时间、场景版本与 `state_id`。视频帧引用同一实际状态。`tools/diagnostics/check_skill_record_sync.py` 检查各帧能否匹配完整样本，并将结果写入 JSON。

`gene_agent.feedback.skill_run` 为规划或执行产物建立统一索引，校验阶段状态链和碰撞快照哈希。`gene_agent.visualization.replay.ReplayTimeline` 按阶段提供 `planned` 或 `actual` 帧。旧 run 缺完整状态时只标为 `partial_actual_sample`，精确 MuJoCo 中间帧回放会拒绝这类数据。Viser 和 MuJoCo viewer 均消费 ReplayTimeline；viewer 不重新规划。

复现命令和人类验收入口见抓碗视频验收（历史材料已本地归档）。同步基线 `runs/pick-bowl-synced-20261007` 的六阶段执行成功，`physical_pick_success=true`，碗抬升约 50.9 mm。`visual/skill_execution.mp4` 为 60 帧、6 秒实际执行录像；`record_sync_check.json` 确认全部 60 帧对应完整实际状态，六阶段起止状态也匹配。此结果只证明当前抓碗基线，不代表十次连续成功或真机可靠性。

回归测试应在 `curobo` 环境设置 `MUJOCO_GL=egl` 和 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，避免系统 ROS pytest 插件干扰；Viser 测试须允许监听本机端口。当前 Skill 仍是抓碗实例的首个基线；门、旋钮、place 和完整任务串联属于后续批次。
