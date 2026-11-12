# 操作与验收文档

按问题选择阅读，不要求逐篇加载。旧计划与临时报告不进入本目录。

| 文档 | 内容 |
|---|---|
| [新任务开发流程](../task_development/index.md) | real2sim、Concept、仿真迭代与 sim2real 的阶段入口 |
| [motion/planning_modes.md](motion/planning_modes.md) | 一键关闭场景避障/attachment、速度检查与适用范围 |
| [cup/cup_pour_task.md](cup/cup_pour_task.md) | 绿色杯：抓杯把、向碗倾倒、放回原位 |
| [environment/windows_simulation_environment.md](environment/windows_simulation_environment.md) | Windows 原生 MuJoCo + cuRobo 环境 |
| [knowledge/concept_agent_annotation.md](knowledge/concept_agent_annotation.md) | 功能 Concept 自动标注 |
| [knowledge/concept_editor.md](knowledge/concept_editor.md) | Mesh Concept 标注器 |
| [knowledge/concept_mesh_inspection.md](knowledge/concept_mesh_inspection.md) | 通用 Concept 与物体 mesh 检查 |
| [knowledge/knowledge_tuner.md](knowledge/knowledge_tuner.md) | 两种 Grasp Knowledge 可视化工具 |
| [knowledge/microwave_concept_mesh_inspection.md](knowledge/microwave_concept_mesh_inspection.md) | 检查微波炉 Concept 与 URDF mesh 是否对齐 |
| [microwave/bowl_skill_live.md](microwave/bowl_skill_live.md) | 在线抓碗与放入 Skill |
| [microwave/door_skill_live.md](microwave/door_skill_live.md) | 在线检查门 Skill 与门循环 Task |
| [microwave/knob_live_viewer.md](microwave/knob_live_viewer.md) | 旋钮在线检查 |
| [microwave/microwave_knowledge_viser.md](microwave/microwave_knowledge_viser.md) | 微波炉 Knowledge 的 Viser 调参 |
| [microwave/mujoco_task1_physical.md](microwave/mujoco_task1_physical.md) | 微波炉 Task1 的 MuJoCo 验收 |
| [microwave/realworld_microwave_entry.md](microwave/realworld_microwave_entry.md) | 微波炉真机子任务入口 |
| [microwave/task1_live_viewer.md](microwave/task1_live_viewer.md) | 完整 Task1 在线纯视觉检查 |
| [hammer操作](novel_tool/hammer.md)、[Hit实例迁移](novel_tool/hit_transfer.md) | 原生 RoboTwin：Concept、共享 Pick/Hit、原评分、实例参数和实际视频 |
| [robots/gripper_acceptance.md](robots/gripper_acceptance.md) | 夹爪仿真验收 |
| [robots/pika_frames.md](robots/pika_frames.md) | Pika 模型、控制点与指尖核对 |
| [scenes/my_scene_0_three_task_previews.md](scenes/my_scene_0_three_task_previews.md) | my_scene_0：三类任务、五个单工具场景 |
| [scenes/realworld_single_object_perception.md](scenes/realworld_single_object_perception.md) | 单物体实景采集、分割和重建 |
| [桌面场景重建实例](../task_development/reconstructed_tabletop_scene.md) | 从单帧 RGB-D 构建桌面新场景 |
| [scenes/sweep_scene_concepts.md](scenes/sweep_scene_concepts.md) | 清扫场景的功能 Concept 标注 |
| [spoon/spoon_pick_skill.md](spoon/spoon_pick_skill.md) | 勺子抓取 Skill（橙色勺） |
| [spoon/spoon_scoop_task.md](spoon/spoon_scoop_task.md) | 橙色勺与粉色勺：抓取、盛球、倾倒 |
| [sweep/sweep_cube_mujoco_skill.md](sweep/sweep_cube_mujoco_skill.md) | 三色方块扫入簸箕：MuJoCo Task |
| [sweep/sweep_first_frame_preview.md](sweep/sweep_first_frame_preview.md) | 从相机首帧预览抓刷与清扫轨迹 |
| [sweep/sweep_long_mujoco_task.md](sweep/sweep_long_mujoco_task.md) | 长柄刷清扫方块：MuJoCo Task |
| [sweep/sweep_realworld_execution.md](sweep/sweep_realworld_execution.md) | 旧刷清扫：真机执行入口 |
| [sweep/sweep_realworld_plan_only.md](sweep/sweep_realworld_plan_only.md) | 刷子清扫：真机接入与只规划预览（旧刷优先） |
| [visualization/shared_operation_workflow.md](visualization/shared_operation_workflow.md) | 共享操作流程：当前入口与验收范围 |
| [visualization/skill_viser_inspection.md](visualization/skill_viser_inspection.md) | 在 Viser 中单独检查 Skill |
| [visualization/video_feedback.md](visualization/video_feedback.md) | 给人类提供 Skill / Task 视频反馈 |
| [visualization/viser_skill_task_debugging.md](visualization/viser_skill_task_debugging.md) | 在线视觉检查：Skill 与 Task |
| [workbench/usage.md](workbench/usage.md) | 本地任务开发工作台 |

- [cuRobo失败诊断](motion/curobo_diagnostics.md)：摘要、球/物体定位、候选误差、只读查看与离线复查。
