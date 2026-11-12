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

命令和前置条件见 [真机入口说明](../../docs/user/microwave/realworld_microwave_entry.md)。目录合并尚未统一三种后端的完整任务循环，实机效果仍需现场验收。
