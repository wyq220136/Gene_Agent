# 旧刷清扫：真机执行入口

当前为可分阶段验收的执行实现，尚未运动真机验收。复用微波炉的 `RealWorldRuntime` / `RealmanAdapter`，刷子抓取 Knowledge 和原来的 `run_sweep_skill`。不是将预览轨迹直接重放给机器人。

## 已提供的行为

- 每段从 SDK 实测七关节、Pika 实测开口重新调用 cuRobo，保留直线接近/抬升/清扫；规划失败不降级。
- 完整 MuJoCo FK 几何复核、整段 TCP 最低高度检查、SDK 起点一致性检查、0.1 速度比例及 TCP 速度限制，实际终态达标才接续。
- 真实 Pika 串口位置控制：米转毫米，不调用 Realman 内置夹爪接口，不使用会改变控制模式的 `set_velocity`。夹爪状态过期或开口偏离规划时，机械臂逐点检查并停止；关闭进程不自动松爪/disable。
- 闭爪后人工确认夹稳才建立刚性夹持假设；完整清扫前再确认抬升后稳定；任务完成另问方块是否已入簸箕。`--yes` 不替代这些物理结果确认。
- 首帧感知、桌面支撑校正保持现有策略。其后静态障碍与刚性持刷是假设，不声称持续视觉闭环或自动测得接触力。

## 1. 先检查文件，无硬件操作

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/execute_sweep_realworld.py \
  --run runs/realworld_sweep/support_corrected_20261024
```

默认不创建SDK进程、不连接Pika、不运动。文件通过不等于当前真机轨迹预检通过。历史记录仅可用于这项检查，执行要求新鲜首帧及实测起点。

## 2. 现场配置

复制 `config/realworld/sweep_execution.example.json` 到自己的配置文件，例如 `config/realworld/local/sweep_execution.site.json`，填写：

- `gripper_port`：实际Pika串口（参考项目曾用 `/dev/ttyUSB60`，本轮环境未发现该设备，未自动猜测其它串口）。
- TCP配置由已有机器人frame profile自动读取，每次实测状态自动核对模型FK与SDK TCP的一致性，阈值3mm/1°。无需手填 `tcp_verified` 或 `tcp_verification_source`，旧配置中这两项会被忽略。该检查不证明实物指尖的物理TCP位置，未实测的部分如实记录。

`max_observation_age_s` 默认180秒；人工确认之后和SDK连接之后均再次检查帧龄。`max_start_error_rad` 默认0.02，实测TCP目标误差默认5mm/3°。默认速度比例0.1、TCP上限0.1m/s，按共享执行器缩时；最低操作TCP为桌面上方10mm。使用当前真实模型的操作TCP，不套用外部150mm参考控制点。

## 3. 首次执行到 pregrasp

确认实际串口后，由操作者显式运行：

```bash
PYTHONUTF8=1 PYTHONPATH=. PYTHONNOUSERSITE=1 \
/home/ubuntu/miniconda3/envs/curobo/bin/python tools/run/execute_sweep_realworld.py \
  --camera-param data/calibration/latest/third_camera/third_view_extrinsic.json \
  --config config/realworld/local/sweep_execution.site.json \
  --execute --stop-after pregrasp
```

入口自动拍照、分割、首帧估计、完整预览规划，再显示执行确认。确认后才连接/使能Pika并低速执行“打开夹爪→pregrasp”，然后停止。机械臂地址优先使用标定中的 `communication.robot_ip`（此现场192.168.1.18）。SDK与cuRobo分处环境时由入口自动管理子进程，无需另开终端。

`--stop-after approach` 增加直线接近；`--stop-after pick` 增加闭爪、人工夹持确认和抬升。每次命令均从新拍照状态开始，不是从上一条命令后直接续接。因此停在pregrasp/approach后应先人工恢复合适初态或重新检查现场，不能把多次命令拼成一次连续已验收任务。

## 4. 完整清扫

```bash
PYTHONUTF8=1 PYTHONPATH=. PYTHONNOUSERSITE=1 \
/home/ubuntu/miniconda3/envs/curobo/bin/python tools/run/execute_sweep_realworld.py \
  --camera-param data/calibration/latest/third_camera/third_view_extrinsic.json \
  --config config/realworld/local/sweep_execution.site.json \
  --execute --stop-after task
```

不自动松刷，结束时保持当前夹持；退出不意味着机械臂返回待机。Ctrl+C/轨迹异常通过共享SDK执行器停止机械臂，设备状态不明时停止任务并保留失败证据。

## 记录与查看

输出 `runs/realworld_sweep/execution_<时间>/`：`first_frame/` 为首帧与完整预览；`execution_result.json` 为真实执行/失败记录；`executed_frames.json` 为阶段端点实测关节；`stages/` 保存逐段GPU返回、碰撞和前后状态；`device/sdk_worker.log` 保存SDK错误。

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python tools/viewers/view_task.py \
  --task sweep_cube_into_dustpan --run runs/realworld_sweep/execution_<时间>

PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python tools/viewers/view_skill.py \
  --skill sweep_cube --run runs/realworld_sweep/execution_<时间>
```

执行回放显示离散实测端点，不把它们连线当作已测得的连续运动；刷子用已确认的刚性关系显示，方块仍停留首帧位置，真实任务结果看独立人工检查。尚未提供真实执行录像。

## 已验证范围

2026-10-24：默认文件检查不触硬件；假设备测试验证失败停止、实际终态接续、未确认夹持不抬升、帧龄及开口变化拒绝。真实GPU结合明确标注的假设备走通7段重规划、9个执行阶段；这不是实际运动成功。Pika SDK导入与接口已核对参考控制代码；尚未连接/使能串口或下发真机运动。实际Pika串口仍需确认；物理TCP精度尚未实测，不通过手填字段将其标为已验证。

### 2026-10-25：取消人工TCP放行字段

`tcp_verified` 和 `tcp_verification_source` 是早期人工门槛，现已移除。已有frame配置自动加载，执行时已有的SDK/模型FK一致性检查、起点/高度/碰撞检查继续生效。几何不一致时报告具体误差；不要求用户填true，也不将一致性检查冒称物理TCP标定。现场配置文件无需为此改动。

## 2026-10-25：阶段范围与候选日志修正

`--stop-after pregrasp/approach/pick` 现在同时限制首帧预览和实际执行，只要求相应前缀规划通过；不再强制完整清扫规划成功。`task` 仍要求完整任务预览。`planning_complete` 继续只表示完整任务；`requested_scope_complete` 表示本次请求范围通过，不能把前缀通过报告成完整任务成功。

完整预览中的重复 move/lower/sweep 是不同几何候选的离线尝试，不是多次真机动作；每个候选从同一个预测抬升状态重新开始。现已打印候选序号、总数及该候选失败原因。实际执行只使用选中的候选，每个阶段从最新实测状态重规划，不执行这些候选循环。

用户 `execution_20261025_120403` 的抓刷前缀通过，7个清扫候选因簸箕侧壁几何复核失败，另外5个预备姿态IK失败。不能用后者概括成抓刷IK无解。复制该帧离线验证pregrasp范围：`runs/realworld_sweep/pregrasp_scope_fixed_20261025`，只规划open_assumed/pregrasp，实际GPU通过，没有任何设备运动。旧运行记录未改写。

## 2026-10-28：共享抓取流程与设备目录

旧刷抓取现在与 MuJoCo、预览共用 `run_pick_flow`，操作参数集中到 `config/tasks/sweep/operation.json`；真机前提检查仍消费实测/人工证据。Pika 和 Realman 型号驱动已下沉到 `realworld/devices/`。命令及边界见 [共享流程说明](../visualization/shared_operation_workflow.md)。本轮新增验证是离线、假设备与 MuJoCo；用户此前实际 pregrasp 记录保留，不提升为完整抓取成功。
