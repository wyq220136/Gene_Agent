# 真机多物体同帧 FoundationPose 输入

这条新接口只补现场观测来源，不修改原来的单物体交互式感知、Viser 或 MuJoCo。`gene_agent/realworld/pose_tracking.py` 提供两个同步函数：

1. `capture_frozen_scene_frame(...)` 从现有 camera server 先取一帧作哨兵，再等待更新的一帧；**只保存后一帧**，使用原有 `save_capture` 保存 `rgb.png`、`depth.png`、`camera.json`，另写 `captured_at_unix_ms`。
2. `track_observed_scene_from_capture(...)` 将同一组冻结文件依次传给每个已注册 FoundationPose session 的 `PoseClient.track_frame()`；请求前后核对 RGB、depth、camera 文件哈希，拒绝相机 ID、session、帧号或时间来源不符的结果。输出兼容现有 `load_observed_scene(require_matched_frame=True)` 的 `camera/camera_parameters_0.json`、每个物体的 `objs/<类别>/<实例>/pose.json`，以及 `capture_provenance.json`。返回 `ObservedScene`，可交给现有 `build_observed_collision_scene`。

示意调用（需要调用方先为各物体注册 tracking session，并提供实测标定）：

```python
from gene_agent.perception_modules.pose_est.pose_client import PoseClient
from gene_agent.realworld.pose_tracking import (
    capture_frozen_scene_frame, track_observed_scene_from_capture,
)

capture = capture_frozen_scene_frame(
    output_dir="runs/live_scene_001/capture",
    endpoint="tcp://127.0.0.1:18082", expected_serial="实际相机序列号",
)
observed = track_observed_scene_from_capture(
    capture_dir=capture,
    pose_client=PoseClient("http://127.0.0.1:18081"),
    sessions={"Bowl/Bowl_0": "已注册的 Bowl session",
              "Microwave/Microwave_0": "已注册的 Microwave session"},
    side="right", camera_id="0", T_sdk_camera=calibrated_T_sdk_camera,
    expected_serial="实际相机序列号", max_age_s=accepted_max_age_s,
    output_dir="runs/live_scene_001/observed",
)
```

**时间来源：** RealSense 原始帧 `timestamp_ms` 可能使用设备时钟，不能直接与 Unix 时间相减；[librealsense 的 timestamp domain 定义](https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/h/rs_frame.h)明确区分硬件、系统与 global time。原值在输出中改名保存为 `sensor_timestamp_ms`；用于帧龄检查和 `ObservedScene.timestamp_ms` 的是采集入口记录的主机 `captured_at_unix_ms`。旧单物体采集文件没有这个主机字段，不能在这里假装成新鲜现场帧。跟踪完成时再检查一次帧龄，慢请求也会被拒绝。

**仍需现场提供：** 同一时刻的双臂关节、夹爪开口、门角、旋钮角、桌面 pose；`T_world_sdk`、相机外参、FoundationPose mesh frame 到场景碰撞资产 frame 的验收变换；各物体的 session、mask/mesh 和现场后置检查器。本接口只证明多个 Pose 请求消费同一冻结 RGB-D，**不证明**位姿准确、物体未在跟踪期间移动、标定正确、cuRobo 规划可行或真机任务成功。真正下发机械臂命令的入口仍未接通。

`applications/microwave/realworld_live_observe.py::build_live_microwave_collision_scene` 现可将 `ObservedScene` 与**新读取**的双臂关节、两夹爪开口、门角和旋钮角交给原有 `build_observed_collision_scene`。门角/旋钮角读数必须各有弧度值、主机 Unix 毫秒时间和来源；可以由现场人工检查或传感器提供，不能从 Microwave 根 pose 推断。调用方另传已经验收的坐标标定、桌面/支撑几何、活动臂及新输出目录。旧帧、旧角度、缺少另一臂或越界关节都将拒绝；结果的 `observed_scene_provenance.json` 保存各来源和采样时间。**物体是同帧，机器人和关节物体只是分别在短有效期内读取**，不声称同时曝光，也不证明设备在规划/执行过程中保持静止。完整 Task 必须在每段重新调用此入口。离线接口测试见 `tests/applications/microwave/realworld/task/test_realworld_live_observe.py`。

**开门后的 Microwave frame 限制：** 旧项目的 Microwave FoundationPose 默认用 `handle_mesh.obj` 注册。关门时可把该 pose 当作关闭状态参考；门打开后，同一 session 跟踪到的是随门移动的把手，不能作为静止微波炉本体 pose。现场组装入口因此要求声明实际注册的 `microwave_pose_mesh_path`，开门超过 1° 时拒绝把手/门 mesh 的 pose 作为本体。这个字段目前由调用方提供并记录，尚未由 Pose server 的注册记录自动核对。现在也可传 `microwave_root_anchor={"T_world_scene_asset": 已确认的世界坐标本体位姿, "verified_at_unix_ms": 本次核对时间, "source": 来源}`：若现场确认柜体没有移动，碰撞场景从当前帧使用 Bowl pose，把 Microwave 本体换成此锚点；把手 pose 被排除出本体来源，门角另由调用方提供。锚点核对过期会拒绝导出，报告以 `pose_source_modes` 区分“当前帧”和“确认静止的锚点”，**不声称全部物体同帧**。若柜体确已移动，必须重新估计本体，不能仅刷新确认时间。旧版测角逻辑见 AffordToolAct 的 `scripts/run_microwave_door_cycle.py::_estimate_post_contact_open_angle`，它也明确不把开门后的把手 pose 写回微波炉根 pose。

**把手测角的旧公式已单独迁入：** `realworld/door_angle.py::estimate_post_handle_open_angle_from_tracked_asset` 接收关闭参考与当前跟踪的 **同一把手 mesh frame** pose、该 mesh frame 下 `handle.pose`、同一 SDK base 下的门轴点/方向，以及当前帧时间和来源。它先按旧 Task1 分别乘 `handle.pose`，再按旧公式求顺时针开门原始角，并取 5° 网格作为规划角；两者分别保存在 `measured_angle_rad` 和 `angle_rad`，可直接作为现场组装的 `angles["door"]`。示例 37° 时旧、新实现均求得 `0.6457718232379019 rad`，规划值是 35°。**门轴、关闭参考、局部把手 pose 与旧网格坐标必须先标定并核对；本函数不从 Microwave 根 pose 猜门角，也不把旧把手 mesh frame 默认等同当前校准 URDF 根 frame。**目前没有现场 FoundationPose/门轴精度验收。

`applications/microwave/realworld_live_observe.py::make_live_microwave_observer` 可生成独立子任务和完整 Task 共用的无参 `observe_collision_scene` 回调。调用方提供 camera/PoseClient/session、两臂与夹爪适配、已核对的静态场景参数，以及从当前 `ObservedScene` 产生门角/旋钮角的 `angle_reader`；必要时提供每次重新确认柜体未移动的 `microwave_root_anchor_reader`。**每次调用**新建序号目录，采集新帧、跟踪、读取实际状态并导出碰撞，不复用上个 Skill 的预测终点或旧角度。构造回调本身不会连接机械臂或发送命令；把它传给 `run_observed_microwave_subtask` 或 `run_observed_microwave_task1` 才进入既有子任务编排。当前没有现成的现场角度传感器、标定值或真实连接配置，因此此函数不是一键实机命令。

**执行前再看一次现场：** 两个任务入口现在都接受可选的 `scene_drift_limits=SceneDriftLimits(...)`。现场调用方应将按相机/机械臂噪声验收过的五个容差传入，入口会在每段规划时冻结当时的物理现场，在发 SDK 命令前由同一观察器再采一帧；物体、桌面、另一臂、夹爪、门或旋钮变化超限即停止。规划的接触放宽或 Bowl 附着阶段仍按其派生碰撞快照 hash 执行，环境漂移则对比其来源物理现场。新的现场帧采集、跟踪或碰撞导出失败也会停止发令。场景来源另外记录 `alignment_input_sha256`，覆盖 `T_world_sdk`、相机外参和各实例 mesh→场景资产变换；执行前复核会拒绝标定版本或物体位姿来源改变，即使 world pose 数值暂时相同。`execute=False` 不因该参数额外拍照。**不传该参数时沿用现有行为，并不自动获得这项保护；当前尚无把已校准的现场参数和服务统一装配起来的用户命令。** 假相机/SDK 和真实场景资产导出已做离线复核，真实噪声容差及耗时必须现场确定。

离线契约测试：

```bash
conda activate curobo
python -m pytest -q -p no:cacheprovider \
  tests/realworld/test_pose_tracking.py tests/realworld/test_scene_shadow.py \
  tests/realworld/test_door_angle.py \
  tests/applications/microwave/realworld/task/test_realworld_live_observe.py
```

这些测试使用假 PoseClient 和真实规划碰撞资产导出，不连接相机、FoundationPose 服务或机器人。
