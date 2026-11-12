# 微波炉运行输入

按消费者选择输入，不是每次运行都读取全部配置。入口与覆盖顺序见 [readme.md](readme.md)。

| 配置 | 消费者与用途 |
|---|---|
| [knowledge/grasp_profiles.json](knowledge/grasp_profiles.json) | Knowledge 及场景抓取编辑器读取的基础抓取候选参数；不属于 Skill 控制参数 |
| [knowledge/grasp_profiles.microwave_calibrated.json](knowledge/grasp_profiles.microwave_calibrated.json) | `knob_0_task.json` 引用的旋钮抓取参数 |
| [knowledge/grasp_profiles.microwave_door.json](knowledge/grasp_profiles.microwave_door.json) | `door_task.json` 引用的把手抓取参数 |
| [knowledge/legacy_extract_visual_table.json](knowledge/legacy_extract_visual_table.json) | 当前视觉取碗与桌面目标对齐使用的旧桌面几何输入，并非可直接删除的历史报告 |
| [knowledge/microwave_knowledge_view.json](knowledge/microwave_knowledge_view.json) | Knowledge viewer 的部件选择、操作参数和预览摆位；门配置无显式位姿时也读取其中的 XY |
| [knowledge/microwave_object_manipulation.json](knowledge/microwave_object_manipulation.json) | `object_keyposes.py` 等读取的放入/取出关键 TCP 相对位姿、标定状态；仿真与真机取放共用 |
| [realworld/grasp_profiles.knob_0_realworld_tuned.json](realworld/grasp_profiles.knob_0_realworld_tuned.json) | 由同目录 knob_0_task_realworld_tuned.json 引用的真机旋钮抓取候选参数 |
| [realworld/knob_0_task_realworld_tuned.json](realworld/knob_0_task_realworld_tuned.json) | `realworld/microwave/knob_planning.py` 的默认旋钮任务，引用下一文件；是运行输入而非未采用的调参输出 |
| [realworld/realworld_site.example.json](realworld/realworld_site.example.json) | 微波炉真机站点配置示例：设备、感知 session 和子任务选项；现场填写副本应放开发目录，保留示例 |
| [simulation/door_physical_open.json](simulation/door_physical_open.json) | `run_experiment.py` 的门物理配置，完整 MuJoCo Task1 的门模板；引用下面的关门参数 |
| [simulation/door_push_close_physical_tuner_state.json](simulation/door_push_close_physical_tuner_state.json) | 由 door_physical_open.json 的 knowledge_states.push_close 引用的已采用关门几何偏移 |
| [simulation/mujoco_task1_physical.json](simulation/mujoco_task1_physical.json) | `run_microwave_task1_physical.py` 默认输入：场景覆盖、Skill 参数、仿真辅助策略；引用 `door_physical_open.json` |
| [simulation/physical_table_place.json](simulation/physical_table_place.json) | `tasks/policies/microwave_task/simulation_bowl/place.py` 读取的桌面放碗高度规则 |
| [skills/door_skill_debug.json](skills/door_skill_debug.json) | `DoorDebugSession`、完整视觉 Task1 和真机门规划的基础参数；名字含 debug，但仍是正式消费者的默认输入 |
| [skills/door_task.json](skills/door_task.json) | 门子任务的门轴、位姿、把手抓取和动作参数；引用 `grasp_profiles.microwave_door.json` |
| [skills/knob_0_task.json](skills/knob_0_task.json) | 仿真旋钮任务参数；引用 `grasp_profiles.microwave_calibrated.json` |
| [skills/skill_profiles.json](skills/skill_profiles.json) | Pick 的 pregrasp/approach/lift、速度和 attachment 代理参数；仿真与真机 Pick 共用 |
| [visualization/bowl_visual_debug.json](visualization/bowl_visual_debug.json) | 碗类在线 Viser Skill/Task 的默认场景覆盖，含碗/微波炉位姿和门初角；批量视觉录像也读取 |
| [visualization/door_visual_open_close_verified_xy_052_031.json](visualization/door_visual_open_close_verified_xy_052_031.json) | 门类视觉验收预设与 `record_visual.py --all` 的门配置；亦被实验评测入口引用 |
| [visualization/knob_visual_tip5_verified.json](visualization/knob_visual_tip5_verified.json) | 旋钮视觉预设与 `record_visual.py --all` 的旋钮配置 |
| [visualization/task1_visual_explicit90_verified.json](visualization/task1_visual_explicit90_verified.json) | 完整视觉 Task1 的显式推门预设与批量录像输入 |

单位以 `_m`、`_rad`、`_deg` 等字段为准。已采用的 debug/tuned 参数按实际读取关系保留。
