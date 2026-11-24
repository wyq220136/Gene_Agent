# 真机代码

- `devices/arms/realman/adapter.py`：Realman 七轴 SDK、FK/TCP、轨迹预检和急停；`connection.py` 管理显式连接，`realtime_state.py` 接收 UDP，`calibration_replay.py` 提供标定 MoveL。
- `devices/grippers/realman/adapter.py`、`devices/grippers/pika/adapter.py`：两种夹爪的米制控制与状态读取。Pika 只有显式调用 `connect()` 才导入串口 SDK、连接并使能。
- `perception.py`、`pose_tracking.py`、`door_angle.py`：现场观测与门角读数。
- `scene.py`、`planning.py`：实测关节/TCP、坐标对齐和现有 planner 的输入适配。
- `runtime.py`、`checks.py`、`scene_drift.py`：下发动作、检查执行终态及观测是否过期/漂移。
- `microwave/`：当前具体任务的现场接线。`task.py` 提供完整任务和独立子任务；操作模块负责执行，`*_planning.py` 负责现场输入与阶段复核。

Knowledge、固定 keypose、候选选择和已有 Skill 继续从公共模块调用。不要再拆出每个 targets/stage/retreat 的真机专用文件。

通用关节和轨迹契约仍在 `motion/model.py`，设备目录只实现型号行为。`sweep/sdk_worker.py` 保留现场任务的进程协议和 Runtime 装配，调用上述驱动；它不作为通用机械臂驱动。厂商 SDK 延迟导入，加载配置、viewer 和离线预览不会连接设备。原顶层设备模块已移除，Python 调用方应直接导入型号目录；历史 run 中的旧路径只说明当时使用的代码。

设备迁移的离线回归（假 SDK、无真实连接）：

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/ubuntu/miniconda3/envs/curobo/bin/python -m pytest -q \
  tests/realworld/test_device_layout.py tests/realworld/test_robot.py \
  tests/realworld/test_gripper.py tests/robots/kinematics/test_realman_udp_state.py tests/applications/sweep/realworld/test_sweep_execution.py
```

命令和前置条件见 [真机入口说明](../../docs/user/microwave/realworld_microwave_entry.md)。微波炉自动输入由
`microwave/perception_setup.py` 采集、分割、注册并保存，`microwave/configuration.py` 读取标定和设备参数，
`microwave/operation_perception.py` 按当前操作选择刷新对象，`microwave/live_perception.py` 在需要刷新门/旋钮时测角并核对固定柜体参考；默认不感知或操作旋钮，显式 `--with-knob` 才启用旋钮。准备阶段只分割碗和把手；桌面复用已迁移的 AffordToolAct Task1 固定 XY 和厚度，从标定后的场景点云计算高度，不做桌面分割或 pose estimation，不要求手填站点或角度 JSON。
目录合并尚未统一三种后端的完整任务循环，实机效果仍需现场验收。

自动抓碗与仿真使用同一个 `tasks/policies/microwave_task/planning/bowl_candidates.py` 参数入口，
场景查看器的新建/重载和任务可视化接线也复用这一入口，
经 `knowledge.bowl_grasp_candidates` 生成 48 个候选，再由共享 `planning/grasp.py` 筛选和规划。
参数保持 Gene_Agent_new 的 auto 路线：15 mm 深度、六个外侧倾角、实际 FK 指垫支点、
世界 Z 偏移 -12 mm；保留右臂基座侧、外侧接近和碗沿接触筛选。
真机不限制桌面接近角范围，只保留优先 55° 的排序偏好；仿真继续限制 45–70°。
自动模式仅使用 `pick_bowl_legacy_visual`；手动候选探针仍保留其独立的 12 mm 默认。
真机只适配同帧物体观测和实测机器人状态，候选生成不推进物理、不执行设备。
感知 mesh 选择复用 AffordToolAct 迁移的 `pose_est.task_pose_est.pose_mesh_path`，
已知任务资产直接注册，不触发 SAM3D；默认只分割并跟踪碗和门把手。
候选一致性与感知调用范围可离线检查：
`python -m pytest -q tests/realworld/test_bowl_candidate_parity.py tests/realworld/test_automatic_microwave_setup.py`。
候选一致不代表相机标定、夹持物理或真机成功率已经验收。

新场景保存实际候选参数，重载时按记录复现；没有候选参数的历史场景沿用原有候选几何，避免同名候选被重新解释。

按操作感知由 `microwave/operation_perception.py` 管理，自动感知入口默认启用。
对应 AffordToolAct `scripts/run_microwave_task1.py:_estimate_task_poses` 与
`refresh_open_angle_after_post_contact` 的对象刷新边界：碗操作只更新碗，门操作进入及
实际门运动/post-contact 后更新把手，默认 TCP 过渡不触发物体估计；旋钮仅在显式启用的旋钮操作更新。
每段 SDK 关节/夹爪继续实测，发令前场景复核保留相同操作范围。
固定柜体、非操作对象及未刷新的门角沿用操作边界状态，保留原观测时间和来源，
通过 `perception_refresh`、`pose_source_modes`、`all_objects_same_frame` 明示；
它们不是新测量，也不构成独立的柜体移动检测。与参考路线一致，期间必须保持相机、
柜体及非操作对象固定；场景被外力改变时应停止并重新准备，不复用旧参考。
阶段回调不将规划角度写成实测值；门被实际操作后仍须测量并通过原门角和圆轨道检查。

抓把手 Skill 到推门 Skill 的 post-contact 交接，与 AffordToolAct Task1 一样先等待 2 秒让门稳定，再采集测角帧；等待记录在感知来源中，不复用等待前的图像。

Pose tracking session 是服务进程内存状态，perception_setup.json 中的 ID 只是名称，
服务重启或 stop 后名称不代表状态仍存在。新准备记录保存各对象的分割服务、类别和 prompt。
仅在明确的 session 不存在错误时，当前操作对象使用本次冻结 RGB-D 重新分割并以已确认 mesh 注册；
不使用旧准备图像或旧 mask 冒充当前观测，其他网络/模型错误原样停止。
旧记录缺少恢复参数时必须重新运行感知准备并使用新 --perception 目录，不手工修改随机 ID。
重新注册也受原帧龄检查；冷启动过慢导致帧过期时本轮停止，待会话建立后重新预检，
不延长帧龄、不自动执行动作。

完整 Task1 预览：运行入口增加 `--subtask all --plan-only`，可加 `--video`。
`microwave/plan_only.py` 沿真机共享步骤表调用现有阶段规划入口；
`plan_only_world.py` 仅保存预测状态、导出碰撞和规划回放，不持有设备或执行器。
`prediction.py` 区分规划假设与真机证据，真实执行仍逐段重观测；命令和限制见入口说明。

完整 plan_only 的现场几何由 `microwave/observed_geometry.py` 从本次 PERCEPTION 冻结 RGB-D 和 data mesh 构建。
桌面位姿与可见范围来自点云，三维回放显示原始点颜色，视频支持 `--camera observed`；
几何来源、mesh 哈希及坐标检查保存在 `observed_geometry/reconstruction.json`，使用方式见真机操作指南。
