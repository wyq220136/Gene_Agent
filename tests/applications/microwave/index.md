# 微波炉回归

本模块统一列出测试用途；更深的实现分类不再单独设置 index。
按下面的相对路径选择测试；运行前提见 [测试总入口](../../index.md)。

| 测试文件 | 保护的行为 |
|---|---|
| [bowl/test_bowl_stage_collision.py](bowl/test_bowl_stage_collision.py) | 抓碗阶段碰撞快照与目标接触复核 |
| [bowl/test_microwave_object_keyposes.py](bowl/test_microwave_object_keyposes.py) | 旧 Viser 的微波炉固定 keypose 数值迁移检查 |
| [bowl/test_physical_bowl_insert.py](bowl/test_physical_bowl_insert.py) | 抓碗后放入必须按旧阶段顺序，且消费每段真实状态 |
| [bowl/test_physical_extract.py](bowl/test_physical_extract.py) | 取碗物理验收不能被轨迹或单侧接触替代 |
| [bowl/test_physical_table_place.py](bowl/test_physical_table_place.py) | 桌面放置只接收同一模型、同一实际终态且仍附着的 Bowl |
| [bowl/test_pick_bowl_interface.py](bowl/test_pick_bowl_interface.py) | 临时抓碗探针验证接口编排，不复制 Knowledge 或 planner |
| [door/test_door_knowledge_shared.py](door/test_door_knowledge_shared.py) | 门 Knowledge 参数装配迁移后，原 Viewer 的完整阶段目标保持不变 |
| [door/test_door_live.py](door/test_door_live.py) | 验证 `gene_agent.tasks.policies.microwave_task.planning.door_session` 的接口与失败边界 |
| [door/test_physical_door_experiment.py](door/test_physical_door_experiment.py) | 独立关门 Skill 必须从显式开门初态开始，并以真实闭门角验收 |
| [door/test_push_door_planning_retry.py](door/test_push_door_planning_retry.py) | 同一目标偶发约束失败时，保留每次规划事实并使用成功的那条轨迹 |
| [door/test_push_door_skill.py](door/test_push_door_skill.py) | 推门 Skill 继承已验证任务的阶段与运动约束 |
| [experiment/test_experiment_checkpoint.py](experiment/test_experiment_checkpoint.py) | 检查点必须保存实际积分状态，并拒绝错误模型或 Skill |
| [experiment/test_experiment_session.py](experiment/test_experiment_session.py) | 在线实验必须从实际 MuJoCo 状态开始，并显式区分物理与规划结果 |
| [knob/test_physical_knob_flow.py](knob/test_physical_knob_flow.py) | 离线执行替身验证旋钮共享阶段、部分运动异常落盘、失败终态与运动学许可；不验证物理效果 |
| [knob/test_rotate_knob_skill.py](knob/test_rotate_knob_skill.py) | 旋钮阶段几何与 Skill 回放格式的行为边界 |
| [realworld/bowl/test_realworld_bowl_hold.py](realworld/bowl/test_realworld_bowl_hold.py) | 人工准备的持碗状态有独立来源，不伪造 Pick 执行记录 |
| [realworld/bowl/test_realworld_bowl_skill.py](realworld/bowl/test_realworld_bowl_skill.py) | 现场抓碗逐段执行：预测轨迹不能当实测终态，未知夹持不能算成功 |
| [realworld/door/test_realworld_door_skill.py](realworld/door/test_realworld_door_skill.py) | 门 Skill 现场编排：只推进实测终态，失败保留已执行前缀 |
| [realworld/door/test_realworld_door_task.py](realworld/door/test_realworld_door_task.py) | 开门和关门子任务只组合已存在的单门 Skill |
| [realworld/extract/test_realworld_extract_skill.py](realworld/extract/test_realworld_extract_skill.py) | 真机取碗子任务逐段执行同一 Skill；未知夹持不能继续 |
| [realworld/insert/test_realworld_insert_skill.py](realworld/insert/test_realworld_insert_skill.py) | 现场放碗子任务按旧阶段顺序运行，逐段从实际状态交接 |
| [realworld/knob/test_realworld_knob_skill.py](realworld/knob/test_realworld_knob_skill.py) | 旋钮整段现场执行：实测交接、物理转角和人工检查不能被视觉提交替代 |
| [realworld/table/test_realworld_table_skill.py](realworld/table/test_realworld_table_skill.py) | 桌面 Place 子任务逐段读取现场状态，松爪确认后才撤离 |
| [realworld/task/test_realworld_live_observe.py](realworld/task/test_realworld_live_observe.py) | 现场动态读数必须进入同一份规划碰撞快照 |
| [realworld/task/test_realworld_task1.py](realworld/task/test_realworld_task1.py) | 完整真机与 load/unload 组合复用操作，验证预检/新观测/未知证据停止 |
| [realworld/task/test_realworld_task_transition.py](realworld/task/test_realworld_task_transition.py) | 开门后的默认 TCP 撤离属于 Task 过渡，沿用门阶段规划语义 |
| [task/test_physical_task_shared_flow.py](task/test_physical_task_shared_flow.py) | 完整/独立 MuJoCo 操作、持物状态/模型交接、失败归因及控制恢复 |
| [task/test_task1_visual_parity.py](task/test_task1_visual_parity.py) | 完整微波炉纯视觉任务的顺序和状态交接回归 |
