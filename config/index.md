# 机器人、场景与现场配置

正式输入按目录分类；现场副本 local/ 不分发，临时试算不放本目录。

| 配置 | 消费者与用途 |
|---|---|
| [realworld/sweep_execution.example.json](realworld/sweep_execution.example.json) | 清扫真机执行限制与设备路径示例；复制到 local 后填写，不含运动授权 |
| [realworld/sweep_long.example.json](realworld/sweep_long.example.json) | 长刷真机只规划示例；demo_source_run 是回放依赖 |
| [realworld/sweep_old.example.json](realworld/sweep_old.example.json) | 旧刷真机只规划示例；demo_source_run 是回放依赖 |
| [robots/gripper_parallel_slide_proxy.json](robots/gripper_parallel_slide_proxy.json) | MuJoCo 双指滑动代理的开口、接触、质量和控制参数 |
| [robots/pika_frames.nominal.json](robots/pika_frames.nominal.json) | Pika 名义安装与控制/TCP 坐标变换 |
| [robots/pika_frames.rm75_6fb.json](robots/pika_frames.rm75_6fb.json) | 指定单臂现场型号的 Pika 坐标配置；核对来源后使用 |
| [robots/realman.json](robots/realman.json) | 倾斜安装 Realman 双臂模型与夹爪配置 |
| [robots/realman_left_default_pose.json](robots/realman_left_default_pose.json) | 左臂默认关节姿态；用于 FK 求默认 TCP |
| [robots/realman_right_default_pose.json](robots/realman_right_default_pose.json) | 右臂默认关节姿态；用于 FK 求默认 TCP |
| [robots/realman_single_pika.json](robots/realman_single_pika.json) | 正装无身体底座 Realman 单臂 + Pika 模型 |
| [robots/standard_gripper.json](robots/standard_gripper.json) | 标准夹爪几何与安装变换 |
| [scenes/task_1.json](scenes/task_1.json) | 微波炉名义场景、桌面、支撑与物理代理 |
| [tasks/sweep/contacts.json](tasks/sweep/contacts.json) | 清扫适配显式绑定工具、支撑、接收器与夹爪 link；Skill 不持有场景默认名 |
| [tasks/sweep/operation.json](tasks/sweep/operation.json) | 共享清扫操作的阶段参数与执行限制 |
| [visualization/studio.json](visualization/studio.json) | 公共显示风格与相机参数，不定义物理成功 |

单位以 `_m`、`_rad`、`_deg` 等字段为准。已采用的 debug/tuned 参数按实际读取关系保留。
