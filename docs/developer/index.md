# 接口与开发参考

按问题选择阅读，不要求逐篇加载。旧计划与临时报告不进入本目录。

| 文档 | 内容 |
|---|---|
| [architecture/native_benchmark_and_embodiment.md](architecture/native_benchmark_and_embodiment.md) | 新 embodiment 与原生仿真 benchmark 接入：机器人/运行/诊断边界、协议与实测能力边界 |
| [architecture/skill_library_plan.md](architecture/skill_library_plan.md) | Skill 能力库与调试：Pick 样板已接入，其他能力待逐项迁移 |
| [architecture/task_authoring.md](architecture/task_authoring.md) | Task 宽松编写约定、五任务整理与 applications 退场的验收范围 |
| [architecture/architecture.md](architecture/architecture.md) | 迁移设计与验收契约 |
| [architecture/shared_operation_runtime.md](architecture/shared_operation_runtime.md) | 仿真与真机的共享操作边界 |
| [architecture/benchmark_policy_skills.md](architecture/benchmark_policy_skills.md) | Benchmark/Policy 分离与 Skill 库审查；重构讨论，未实施 |
| [architecture/incremental_refactor_plan.md](architecture/incremental_refactor_plan.md) | 当前重构总计划与进度：先 benchmark/Task，后 Skill |
| [architecture/task_subtask_review.md](architecture/task_subtask_review.md) | 微波炉子任务：独立调试、状态交接与真机入口；仿真已验证，硬件待验收 |
| [environment/environment.md](environment/environment.md) | 开发环境 |
| [knowledge/approximate_ik_execution_design.md](knowledge/approximate_ik_execution_design.md) | 近似 IK 的规划与执行契约 |
| [knowledge/knowledge_design.md](knowledge/knowledge_design.md) | Knowledge 模块最终设计 |
| [knowledge/knowledge_tuner_feedback.md](knowledge/knowledge_tuner_feedback.md) | Knowledge 调参界面反馈契约 |
| [knowledge/task1_door_knowledge_alignment.md](knowledge/task1_door_knowledge_alignment.md) | Task1 门操作 Knowledge 对齐 |
| [maintenance/code_layout_migration_log.md](maintenance/code_layout_migration_log.md) | 代码目录迁移记录（2026-10-07） |
| [maintenance/file_layout.md](maintenance/file_layout.md) | 文件与索引的维护边界 |
| [microwave/door_skill_live.md](microwave/door_skill_live.md) | 门 Skill 在线纯视觉调试接口 |
| [microwave/evidence/door_physical_scene_alignment_2026-10-13.md](microwave/evidence/door_physical_scene_alignment_2026-10-13.md) | 开门物理实验：场景版本与执行误差排查 |
| [microwave/evidence/mujoco_task1_mixed_acceptance_2026-10-13.md](microwave/evidence/mujoco_task1_mixed_acceptance_2026-10-13.md) | 微波炉 Task1 的连续 MuJoCo 混合仿真验收 |
| [microwave/knob_visual_session.md](microwave/knob_visual_session.md) | 旋钮在线会话开发说明 |
| [microwave/microwave_integration.md](microwave/microwave_integration.md) | 标定微波炉、支撑布局与轴语义 |
| [microwave/microwave_knowledge_alignment.md](microwave/microwave_knowledge_alignment.md) | 微波炉 Knowledge 检查的坐标与验收 |
| [microwave/microwave_object_frame_audit.md](microwave/microwave_object_frame_audit.md) | 微波炉放入/取出 keypose 的资产坐标核对 |
| [microwave/object_visual_session.md](microwave/object_visual_session.md) | Bowl 在线 VisualRun 接入（2026-10-12） |
| [microwave/task1_visual_viewer.md](microwave/task1_visual_viewer.md) | Task1 在线 Viewer 实现边界 |
| [microwave/task1_failure_attribution_human_guidance_and_repairs.md](microwave/task1_failure_attribution_human_guidance_and_repairs.md) | Task1 问题归因、人工有效建议、修复分析与验证边界 |
| [motion/body_collision_audit.md](motion/body_collision_audit.md) | 本体碰撞迁移审计与修复 |
| [motion/curobo_failure_diagnostics.md](motion/curobo_failure_diagnostics.md) | Pick bowl 的 cuRobo 失败诊断 |
| [motion/microwave_planning_performance.md](motion/microwave_planning_performance.md) | 微波炉完整任务规划提速（2026-10-29） |
| [motion/motion_baseline.md](motion/motion_baseline.md) | 同步 cuRobo 基线 |
| [motion/motion_batch.md](motion/motion_batch.md) | 批量 IK 候选评估 |
| [motion/motion_scene.md](motion/motion_scene.md) | MuJoCo GT 静态碰撞快照 |
| [motion/planning_modes.md](motion/planning_modes.md) | robot_only 配置、场景/attachment 分流及性能验收 |
| [motion/reachability_screening.md](motion/reachability_screening.md) | 几何候选的单臂 IK screening |
| [perception/concept_agent_annotation_experience.md](perception/concept_agent_annotation_experience.md) | 功能 Concept 自动标注：方案与实践记录（2026-10-17） |
| [perception/foundationpose踩坑记录.md](perception/foundationpose踩坑记录.md) | 安装流程 |
| [perception/perception_runtime_setup.md](perception/perception_runtime_setup.md) | 感知模型本机资源与环境 |
| [perception/realworld_same_frame_tracking.md](perception/realworld_same_frame_tracking.md) | 真机多物体同帧 FoundationPose 输入 |
| [realworld/realworld_baseline.md](realworld/realworld_baseline.md) | 真机接入的冻结基线与边界 |
| [robots/gripper_implementation.md](robots/gripper_implementation.md) | 可动夹爪实现与验收 |
| [robots/robot_migration.md](robots/robot_migration.md) | Realman MuJoCo 迁移记录 |
| [simulation/asset_preparation.md](simulation/asset_preparation.md) | 场景和碰撞资产准备取舍 |
| [simulation/feedback.md](simulation/feedback.md) | 反馈与渲染接口 |
| [simulation/mujoco_runtime.md](simulation/mujoco_runtime.md) | MuJoCo 同步 Runtime 与失败反馈 |
| [simulation/task_specific_rules_and_simulation_privileges.md](simulation/task_specific_rules_and_simulation_privileges.md) | 任务专用规则与仿真特权：使用清单 |
| [simulation/visual_quality.md](simulation/visual_quality.md) | MuJoCo 视觉质量与相机契约 |
| [visualization/skill_execution_recording.md](visualization/skill_execution_recording.md) | Pick Skill 执行与证据接口 |
| [visualization/skill_viser_inspection_design.md](visualization/skill_viser_inspection_design.md) | Skill 独立检查：运行记录与 Viser 视图设计 |
