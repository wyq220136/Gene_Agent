# 微波炉 Task1 真机部署与操作指引

本文用于将当前 Gene Agent 微波炉 Task1 从 MuJoCo 验证迁移到 RealMan 真机。
真机入口为 `tools/run/run_microwave_task1_realworld.py`；
`tools/run/run_microwave_task1_physical.py` 仍是 MuJoCo 物理仿真入口，不能用于控制真机。

详细字段、子任务前置条件和能力边界见
[微波炉真机子任务入口](realworld_microwave_entry.md)。本文提供按现场顺序执行的操作清单。

## 当前能力边界

- 真机入口已接入实时观测、cuRobo、RealMan 双臂状态、夹爪控制、场景漂移检查和阶段记录。
- 默认模式只连接现场并预检所选子任务的首段，不发送运动命令。
- 只有同时提供 `--execute --yes` 才允许向设备下发命令。
- 当前入口尚未完成微波炉 Task1 的现场 plan-only、标定和设备执行验收。
- 当前 Task1 的抓碗、放入、旋钮、取出、桌面放置和完整任务只按右臂接线。
- 当前 `connect_realman(...)` 使用 RealMan 原生夹爪适配；如果现场安装 Pika，必须先修改设备装配并重新验收 TCP、开口映射和碰撞模型。
- 旋钮 profile 尚未通过真实接触验收；当前几何检查中，夹爪完全闭合仍距旋钮约 8.65 mm。
- 当前 CLI 的 `--execute` 执行整个所选子任务，没有“只执行到指定阶段”的参数。第一次真机运动前，应先补充阶段停止保护或独立的低速无接触运动检查。

截至 2026-11-12，当前工作站还有两个部署阻塞项：

1. `/data1/yuquan/miniconda3/envs/gene_agent` 尚不能导入 RealMan 厂商包 `Robotic_Arm.rm_robot_interface`。
2. 仓库内只有 `realworld_site.example.json`，尚无填写完成的现场 `realworld_site.json`。

离线真机边界测试通过不等于实际设备验收。不得从仿真成功率推导真机成功率。

## 1. 现场安全准备

在启动任何设备连接前完成以下检查：

- 微波炉、桌面和固定第三视角相机均已机械固定，运行期间不会移动；本流程不使用安装在机械臂手腕上的相机。
- 工作区内无人，线缆不会进入机械臂、夹爪或微波炉门的运动范围。
- 操作者全程握持实体急停；软件停止不能替代实体急停。
- 两臂均已上电、联网并取得真实 IPv4 地址。即使只操作右臂，当前入口也会连接两臂并读取两臂状态。
- 核对真实夹爪型号、最大开口、法兰安装方向和 TCP，确认与代码配置一致。
- 初次现场调试使用 `full` 碰撞模式，不使用 `robot_only`。
- 关闭会控制同一机械臂、夹爪或独占同一 RealSense 的其他进程。
- 记录现场初态：两臂关节、夹爪开口、门角、旋钮角、碗位姿及微波炉柜体位姿。

## 2. 检查 Python 环境和厂商 SDK

在仓库根目录执行：

```bash
cd /data1/yuquan/Gene_Agent_new
PY=/data1/yuquan/miniconda3/envs/gene_agent/bin/python

$PY -c "import torch, curobo, pyrealsense2; print('基础环境正常')"
$PY -c "from Robotic_Arm.rm_robot_interface import RoboticArm; print('RealMan SDK正常')"
```

第二条必须在连接设备前通过。若失败，将 RealMan 官方 Python SDK 安装到运行真机入口的同一个环境，
不要只安装在另一个 Conda 环境中。

确认 CUDA 和 GPU 状态：

```bash
nvidia-smi
```

为 FoundationPose 和真机 cuRobo 选择未被其他实验占用的 GPU。不要与长时间仿真批次共享同一张 GPU。

## 3. 启动和检查固定第三视角 RealSense

查询实际相机序列号：

```bash
rs-enumerate-devices -s
```

确认该序列号对应场景外部固定安装的第三视角相机，而不是手腕相机。相机视野应同时覆盖
微波炉、碗、主要操作区域和标定板可放置区域；完成外参标定后不得移动支架、相机或改变安装角度。

在终端 1 启动相机服务：

```bash
$PY -m gene_agent.perception_modules.camera.camera_server \
  --serial <实际序列号> \
  --endpoint tcp://127.0.0.1:18082 \
  --width 1280 --height 720 --stream-fps 30 --max-fps 10
```

在终端 2 连续读取 10 帧：

```bash
$PY -m gene_agent.perception_modules.camera.camera_client \
  --endpoint tcp://127.0.0.1:18082 --frames 10
```

检查每帧的序列号、RGB、深度、内参、帧号和时间戳。若持续无深度帧：

1. 确认没有其他程序占用相机。
2. 检查 USB 3 线缆、端口和供电。
3. 使用 `rs-enumerate-devices -s` 核对设备支持的流配置。
4. 必要时尝试 `640x480@15`，但不能在未核对设备能力时反复猜测分辨率。

## 4. 启动 FoundationPose 并注册对象

在终端 3 使用空闲 GPU 启动 Pose 服务：

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU编号> \
$PY -m gene_agent.perception_modules.pose_est.pose_server \
  --device cuda:0 \
  --camera-endpoint tcp://127.0.0.1:18082
```

健康检查：

```bash
curl http://127.0.0.1:18081/health
```

准备同一现场的首帧 RGB、深度、相机参数，以及人工检查通过的碗和微波炉 mask。
分别注册两个长期 tracking session：

```bash
$PY -m gene_agent.perception_modules.pose_est.pose_client register \
  --session-id Bowl/Bowl_0 \
  --rgb-path <首帧RGB> \
  --depth-path <首帧深度> \
  --mask-path <碗mask> \
  --mesh-path data/task_1/objs/Bowl/Bowl_0/meshes/obj_mesh.obj \
  --camera-path <相机参数JSON>
```

```bash
$PY -m gene_agent.perception_modules.pose_est.pose_client register \
  --session-id Microwave/Microwave_0 \
  --rgb-path <首帧RGB> \
  --depth-path <首帧深度> \
  --mask-path <微波炉把手mask> \
  --mesh-path data/task_1/objs/Microwave/Microwave_0/meshes/handle_mesh.obj \
  --camera-path <相机参数JSON>
```

注册完成后分别检查新帧跟踪：

```bash
$PY -m gene_agent.perception_modules.pose_est.pose_client track \
  --session-id Bowl/Bowl_0

$PY -m gene_agent.perception_modules.pose_est.pose_client track \
  --session-id Microwave/Microwave_0
```

两个对象必须使用不同 session。配置中的 `microwave_pose_mesh_path` 必须与实际注册的 mesh 一致。

## 5. 完成固定第三视角相机、机器人和资产标定

本指南使用固定第三视角相机，不执行 `handeye` 手腕相机标定，也不把手腕相机外参填入
`camera.T_sdk_camera`。第三视角相机外参必须把该相机 RGB 光学坐标中的点变换到当前右臂
RealMan SDK base。

优先使用不依赖手腕相机的 `manual` 路线。固定第三视角相机服务保持在
`tcp://127.0.0.1:18082`，按现场标定板的实际内角点数量和间距运行：

```bash
CALIB_OUT=data/calibration/runs/manual/microwave_third_view_$(date +%Y%m%d-%H%M%S)

$PY -m gene_agent.perception_modules.camera.calibration manual \
  --data-dir "$CALIB_OUT" \
  --side right \
  --robot-ip <右臂真实IP> \
  --camera-endpoint tcp://127.0.0.1:18082 \
  --pattern-size 11 8 \
  --square-size <相邻内角点实际间距_米>
```

该向导通过人工对齐标定板、只读 RealMan TCP、第三视角相机观测标定板和最终点云检查，
求得固定相机到右臂 SDK base 的外参；不需要手腕相机，也不发送机器人运动命令。
标定板从人工测量到相机采样完成前必须保持不动。只有检查叠加点云并输入 `ACCEPT` 后，
才能使用 `$CALIB_OUT/cam_param.json` 中的 `right_extrinsic_matrix` 作为
`camera.T_sdk_camera`。

如果已经有独立测量且仍然有效的 `T_sdk_board`，也可以只用固定第三视角相机执行：

```bash
$PY -m gene_agent.perception_modules.camera.calibration board-camera \
  --third-endpoint tcp://127.0.0.1:18082 \
  --third-serial <第三视角相机实际序列号> \
  --pattern-size 11 8 \
  --square-size <相邻内角点实际间距_米> \
  --samples 3 \
  --board-pose <包含T_sdk_board的JSON> \
  --board-pose-key T_sdk_board \
  --visualize
```

该路线输出的 `third_view_extrinsic.json` 中，`extrinsic_matrix` 是第三视角相机到 SDK base
的变换，可填入 `camera.T_sdk_camera`。若标定板相对 SDK base 的位置、原点或轴向已变化，
不得复用旧 `T_sdk_board`。

检查已有正式结果：

```bash
$PY -m gene_agent.perception_modules.camera.calibration results
$PY -m gene_agent.perception_modules.camera.calibration inspect
```

需要现场验收的主要变换如下：

| 配置字段 | 含义 |
|---|---|
| `camera.T_sdk_camera` | 固定第三视角相机坐标到右臂 RealMan SDK base |
| `scene_parameters.T_world_sdk` | SDK base 到规划 world |
| `T_perception_object_scene_asset` | FoundationPose 注册 mesh 到规划资产 frame |
| `table_pose_world` | 真实桌面的完整位姿 |
| `microwave_frame_alignment.T_scene_asset_profile` | 微波炉现场资产 frame 到 Insert/Extract profile frame |
| 法兰到 TCP | 真实夹爪安装与规划 TCP 的变换 |

`T_A_B` 表示把 B 坐标系中的点变换到 A。不得用单位矩阵代替未知标定。
Insert/Extract 的 frame 对齐错误会直接产生错误高度或错误方向的 `preplace` 目标。

标定通过后还要在第三视角相机和场景保持固定的条件下核对：

- RealMan SDK FK 的 flange pose。
- 应用法兰到 TCP 后的实际 TCP。
- cuRobo/MuJoCo 中同一关节状态的 TCP。
- 第三视角点云中的标定板、桌面和多个已知 TCP 位置是否落在对应 SDK base 位置。
- 真实夹爪开口与代码中的米制开口映射。

机械臂可以移动到不同核对姿态，但固定第三视角相机不能随机械臂移动。任何相机支架、安装角度、
分辨率或内参变化都要求重新标定并更新 `T_sdk_camera`。

## 6. 创建现场站点配置

复制模板：

```bash
cp data/task_1/realworld/realworld_site.example.json \
   data/task_1/realworld_site.json
```

填写以下内容：

- 左、右臂真实 IP。
- 固定第三视角相机序列号、`camera_id`、外参所属 SDK base 的 `side` 和 `T_sdk_camera`；当前填
  `side: "right"` 表示外参落在右臂 SDK base，不表示相机安装在右腕。
- Bowl 和 Microwave 的两个 FoundationPose session。
- 实际注册的微波炉 mesh 路径。
- `T_world_sdk`、物体资产 frame 对齐、桌面 pose 和桌面尺寸。
- 各子任务的现场位姿、起点和夹爪误差容差。
- Insert/Extract 的 `microwave_frame_alignment`。
- Table Place 的 `table_height_alignment`。
- `scene_drift_limits`。
- `state_file` 路径。

模板中的 `null`、空字符串、示意容差和 `accepted_for_planning=false` 必须由现场结果替换。
只有完成标定和独立检查后，才能把相关 alignment 标为 `accepted_for_planning=true`。

## 7. 维护门角、旋钮角和柜体锚点

`state_file` 至少包含：

```json
{
  "door": {
    "angle_rad": 0.0,
    "observed_at_unix_ms": 0,
    "source": "现场测量方法"
  },
  "knob": {
    "angle_rad": 0.0,
    "observed_at_unix_ms": 0,
    "source": "现场测量方法"
  }
}
```

每次运行前、门运动后和旋钮运动后都必须更新角度、Unix 毫秒时间戳和来源。
可以用以下命令取得当前 Unix 毫秒：

```bash
date +%s%3N
```

若 FoundationPose 注册的是会随门运动的把手或门 mesh，门打开后不能继续用该 pose 代表静止柜体。
此时需在 `state_file` 中提供本次重新确认的柜体根位姿：

```json
{
  "microwave_root_anchor": {
    "T_world_scene_asset": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "verified_at_unix_ms": 0,
    "source": "现场柜体位姿确认方法"
  }
}
```

上面的单位矩阵仅表示 JSON 结构示例，不能作为真实标定值。柜体移动后必须重新测量，不能只刷新时间戳。

## 8. 静态配置检查

静态检查不会连接相机、Pose 服务或机械臂：

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU编号> \
$PY tools/run/run_microwave_task1_realworld.py \
  --config data/task_1/realworld_site.json \
  --subtask pick_bowl \
  --collision-mode full \
  --check-config
```

完整任务在第一次检查时建议先跳过尚未验证的旋钮：

```bash
$PY tools/run/run_microwave_task1_realworld.py \
  --config data/task_1/realworld_site.json \
  --subtask all --no-knob \
  --collision-mode full \
  --check-config
```

`--check-config` 通过只说明字段、路径和基础数值合法，不证明：

- 相机正在出帧。
- Pose session 仍然有效。
- 标定准确。
- 机械臂可连接。
- 轨迹可行或真机安全。

## 9. 连接现场并执行 plan-only

创建新的运行目录。所有结果统一写入仓库 `runs/realworld/`：

```bash
OUT=/data1/yuquan/Gene_Agent_new/runs/realworld/pick_bowl_plan_$(date +%Y%m%d-%H%M%S)

CUDA_VISIBLE_DEVICES=<空闲GPU编号> \
$PY tools/run/run_microwave_task1_realworld.py \
  --config data/task_1/realworld_site.json \
  --subtask pick_bowl \
  --collision-mode full \
  --output "$OUT"
```

该命令会连接两臂、夹爪、相机和 Pose 服务并规划首段，但不会发送运动命令。

检查运行目录中的：

- `site_input.json` 和配置哈希。
- 同帧 RGB-D、FoundationPose 结果及时间来源。
- 双臂关节和两夹爪实际状态。
- 门角、旋钮角和柜体锚点。
- `collision_snapshot.json` 中的微波炉、门、桌面、机器人头部、另一臂和活动臂碰撞模型。
- 规划起点是否等于当前实测关节。
- TCP、目标 pose、attachment 和完整轨迹碰撞检查。

使用 Viser 检查保存的运行证据：

```bash
$PY tools/viewers/view_skill.py --skill pick_bowl --run "$OUT"
```

抓碗和开门的第一段通常是开爪。plan-only 只通过首段不等于 `pregrasp`、`grasp`、
`retreat` 或完整子任务已经可执行。

## 10. 第一次真机运动的放行顺序

当前入口执行的是整个子任务，因此第一次真机运动前应先提供以下保护能力之一：

1. 增加 `--stop-after-stage <stage>`，只执行指定阶段及其必要前缀。
2. 增加独立的低速、无接触、小位移真机 smoke test。

在这些保护入口未实现和验收前，不应直接把完整 Pick 当作第一次真机运动。

推荐逐级验收顺序：

1. 只读连接两臂、读取关节和 SDK FK，不下发命令。
2. 低速无接触小运动；目标与头部、门、桌面和另一臂保持充足净空。
3. 单独开爪、闭爪和夹爪状态读取。
4. 抓碗 `pick_bowl`。
5. 人工稳定持碗条件下的 `insert_bowl --prepare-held-bowl`。
6. 单独开门和关门。
7. `extract_bowl` 与 `place_bowl_on_table`。
8. `load_bowl` 和 `unload_bowl` 组合。
9. 无旋钮完整任务 `all --no-knob`。
10. 旋钮真实接触和转动单独验收后，再启用完整 Task 的旋钮阶段。

每一级都必须满足：

- 从真实观测和实测关节重新规划。
- 执行前场景重新观测通过。
- 实际终点与计划终点一致。
- 物体、门和夹爪的现场后置条件有真实证据。
- 失败后停止，不自动跳过或继续下游阶段。

## 11. 执行单个子任务

在完成上述逐阶段验收后，单个抓碗子任务的命令形式为：

```bash
OUT=/data1/yuquan/Gene_Agent_new/runs/realworld/pick_bowl_exec_$(date +%Y%m%d-%H%M%S)

CUDA_VISIBLE_DEVICES=<空闲GPU编号> \
$PY tools/run/run_microwave_task1_realworld.py \
  --config data/task_1/realworld_site.json \
  --subtask pick_bowl \
  --collision-mode full \
  --output "$OUT" \
  --execute --yes
```

执行模式需要交互终端。自动检查无法判断时，程序可能询问现场操作者：

- 输入 `y` 或 `yes`：确认真实后置条件通过。
- 输入 `n` 或 `no`：确认失败并停止。
- 空输入、`?` 或其他输入：状态未知并停止。

人工回答不能覆盖自动检测到的失败，也不能用仿真证据代替真机后置条件。

独立 Insert 或 Table Place 由人先夹稳碗，再使用：

```bash
$PY tools/run/run_microwave_task1_realworld.py \
  --config data/task_1/realworld_site.json \
  --subtask insert_bowl \
  --collision-mode full \
  --output /data1/yuquan/Gene_Agent_new/runs/realworld/insert_bowl_<时间戳> \
  --prepare-held-bowl \
  --execute --yes
```

`--prepare-held-bowl` 只证明操作者确认当前碗已被夹稳，不代表程序执行过 Pick。

## 12. 完整 Task

所有独立子任务和状态交接均通过后，先执行无旋钮完整任务：

```bash
OUT=/data1/yuquan/Gene_Agent_new/runs/realworld/task1_no_knob_$(date +%Y%m%d-%H%M%S)

CUDA_VISIBLE_DEVICES=<空闲GPU编号> \
$PY tools/run/run_microwave_task1_realworld.py \
  --config data/task_1/realworld_site.json \
  --subtask all --no-knob \
  --collision-mode full \
  --output "$OUT" \
  --execute --yes
```

只有旋钮抓持、接触、转动角和撤离均经过独立真机验收后，才能去掉 `--no-knob`。

完整任务必须从门关闭、碗在桌面、右臂夹爪空闲、两臂处于已测量安全起点开始。
程序不会自动补做缺失的前置任务，也不会自动回 home。

## 13. 异常、停止和恢复

- 感知过期、场景漂移、起点误差、碰撞、轨迹预检或后置条件失败时，停止后续任务。
- `Ctrl+C` 在已经下发轨迹后会尝试调用 RealMan SDK 停止，但实体急停始终是最后安全手段。
- 程序不会回滚部分执行的轨迹，也不会自动恢复到 home。
- 发生异常后先保持现场，不要立即手动移动机器人、门或碗。
- 保存完整运行目录，记录实体急停、软件停止、网络中断或人工干预的时间。
- 读取失败后的实测关节、夹爪状态、门角、碗位姿和微波炉状态。
- 根据当前实际状态制定恢复路径；不得重放基于旧起点和旧场景的轨迹。
- 更新 `state_file` 和现场观测后重新规划。

## 14. 每轮运行必须保存和检查的证据

每轮真机运行目录至少应包含：

- `site_input.json` 和配置哈希。
- 相机、Pose session、标定和 mesh 来源。
- 每阶段的新现场观测及帧号、时间戳。
- 两臂实际关节和夹爪开口。
- 门角、旋钮角和柜体锚点来源。
- cuRobo 请求、目标、轨迹、碰撞快照和 attachment。
- SDK 实际起点、执行终点和执行错误。
- 人工后置检查问题、回答和时间。
- 失败阶段、traceback 和失败后的实际状态。
- 现场视频或至少覆盖全景、夹爪、门和目标物体的同步录像。

规划成功只证明轨迹被求出；SDK 到达只证明关节终点；只有真实物体、门和任务后置条件通过，
才能报告真机操作成功。

## 15. 上机前最终清单

- [ ] RealMan SDK 能在真机运行环境中导入。
- [ ] 两臂 IP 和连接状态已核对。
- [ ] 实际夹爪类型与当前 adapter 一致。
- [ ] 法兰到 TCP、夹爪开口和 FK 已用多姿态核对。
- [ ] RealSense 连续输出有效 RGB-D。
- [ ] Bowl 和 Microwave FoundationPose session 可在新帧跟踪。
- [ ] 相机外参、world/sdk、物体资产 frame 和桌面 pose 已验收。
- [ ] Insert/Extract frame alignment 已验收并标为可规划。
- [ ] `state_file` 角度、时间戳和来源为当前值。
- [ ] 门打开时柜体根锚点来源有效。
- [ ] `scene_drift_limits` 来自现场测量，不是模板示意值。
- [ ] `--check-config` 通过。
- [ ] plan-only 运行及碰撞场景人工检查通过。
- [ ] 无接触低速运动和夹爪单动验收通过。
- [ ] 单个子任务按顺序验收通过。
- [ ] 操作者全程掌握实体急停。
- [ ] 所有结果写入 `/data1/yuquan/Gene_Agent_new/runs/realworld/`。
