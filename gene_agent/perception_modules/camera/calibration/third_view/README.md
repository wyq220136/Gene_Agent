# 固定标定板到第三视角相机的标定

若希望**固定机械臂与两台相机，只移动板**，并使用已有手眼标定求第三相机到基座外参，请使用[固定机器人移板模式](moving_board.md)。它与下面的纯板相机流程及固定板回放模式是不同入口。

**板已经移动过，需要复用旧手眼标定并回放旧机械臂姿态、重新求当前板位和完整外参时，直接使用[固定板完整向导](fixed_board_replay.md)。** 它会完成双相机新采样、矩阵组合和 Open3D 叠加；以下纯相机流程适合只求板到相机，或已有当前板到基座矩阵的情况。

默认向导只需一台第三视角相机：板与相机全程固定，直接通过 PnP 求 `T_camera_board`。
无需手腕相机或机器人连接，也无需改变机械臂姿态。`T_A_B` 表示把 B 中的坐标转换到 A，平移单位米。

先在一个终端启动固定第三视角 D435 的服务（已有服务时直接复用）：

```bash
conda activate sam3
python -m gene_agent.perception_modules.camera.camera_server \
  --serial 337122070986 --endpoint tcp://127.0.0.1:18083 \
  --width 1280 --height 720 --stream-fps 30 --max-fps 10
```

另一个终端运行向导：

```bash
conda activate zehao
python -m gene_agent.perception_modules.camera.calibration.guided_third_view_crosscheck \
  --third-endpoint tcp://127.0.0.1:18083 \
  --third-serial 337122070986 \
  --pattern-size 11 8 --square-size 0.015 \
  --samples 3 --visualize
```

也可用名称更直接的 `python -m gene_agent.perception_modules.camera.calibration.guided_board_camera`，参数相同。
序列号需与实际固定相机一致；`--pattern-size` 是内角点列数/行数，`--square-size` 是相邻角点实际间距。
可选 `--handeye <已有handeye.json>` 仅读取棋盘格尺寸，命令中显式给出的尺寸优先；不会读取旧板位估计。

1. 固定棋盘格与相机，调整画面到整块棋盘格清晰可见。红色 0 号内角点应是你定义的板坐标原点，角点排列决定板的 X/Y 轴。
2. 在实时窗口按一次空格取样，松开脚踏板约 1 秒，再按空格确认；N 重拍，Q 退出。
3. 保持板与相机固定，重复采样 3 次。可用 `--samples 1` 只求解单帧；多帧仅检查重复性，不代表绝对精度。
4. 默认拒绝单帧 PnP 误差超过 2 px、相对首帧板位变化超过 5 mm / 2° 的观测。采样期间无需移动板或机械臂。
5. `--visualize` 在 Open3D 中显示相机坐标下的点云、板平面与板轴；检查贴合后在终端输入 `ACCEPT` 保存。

输出目录自动命名为 `data/calibration/runs/board_camera/calibration_board_camera_*`。`camera_to_board.json` 同时保存：

- `T_camera_board` / `board_to_camera`：**标定板到相机**的变换。
- `T_board_camera` / `camera_to_board`：上述矩阵的逆。
- `session.json`、`sample_*.npz` 和角点 PNG：采集记录、原始 RGB-D、内参和预览。
- `quality_report.json`：静止重复性与候选矩阵。只有最终输入 `ACCEPT` 才写正式变换。

中断后指定 `--data-dir <实际目录> --resume`，保持板位、相机安装、分辨率、serial 与尺寸参数一致。
旧双相机 session 不能用于这个固定板模式。没有图形桌面时加 `--no-preview`、去掉 `--visualize`，终端回车取样、y 接受帧。

## 使用已有板到机械臂基座的变换（可选）

只求板到相机时不需要提供机械臂矩阵。如需进一步求固定相机到 SDK base，在上面的命令中增加：

```bash
  --board-pose <已有板位矩阵.json>
```

JSON 可为纯 4×4 数组，或包含 `T_sdk_board` 字段的对象。
对于原单臂流程输出的 `board_to_robot.json`，加 `--board-pose-key board_to_robot`；输入必须确实是该臂 SDK base 下的板位，不能混用旧安装/规划基座坐标系。
向导要求输入 `BOARD_UNCHANGED` 确认当前板位置、原点和轴向与该矩阵一致，再计算：

```text
T_sdk_camera = T_sdk_board @ inverse(T_camera_board)
```

最终确认后额外保存 `third_view_extrinsic.json`，其中 `extrinsic_matrix` 是固定相机到 SDK base，`T_third_sdk` 是其逆。
**如果板在得到已有矩阵后移动过，必须恢复相同板位或重新测量 T_sdk_board。** 不会自动采用旧 `handeye.json` 的 `T_sdk_board_estimate`。

当前 PnP 使用 RGB 内参 K，假定 RGB 已去畸变或畸变可忽略；相机协议未传递畸变系数。静止重复性不能代替绝对精度或双相机交叉验证。

## 旧双相机移板交叉检查（显式选择）

以下流程仅用于需要验证手腕标定与第三视角整体一致性的场景。须在命令中加 `--crosscheck`，才启用移板和手腕旋转要求。

本向导读取已经完成的手腕相机 `handeye.json`，给**固定在场景中**的第三视角相机求 `T_sdk_third`（把第三视角 RGB 光学坐标中的点变换到机械臂 SDK base）。它只读 RealMan 七轴与法兰 FK，不发送运动命令。

### 为什么要移动标定板两次

两台相机都观察同一块棋盘格。第一个板位 A 的观测给出：

```text
T_sdk_board_A = T_sdk_flange @ T_flange_wrist @ T_wrist_board_A
T_sdk_third   = T_sdk_board_A @ inverse(T_third_board_A)
```

其中 `T_flange_wrist` 来自手腕的 `handeye.json`。同一板位 A 上板坐标系重合是上述公式的直接结果，**不能证明整体标定正确**。所以须把板移到第二个板位 B：

```text
T_sdk_board_B_from_wrist = T_sdk_flange_B @ T_flange_wrist @ T_wrist_board_B
T_sdk_board_B_from_third = T_sdk_third @ T_third_board_B
```

第二板位两条路径的平移、旋转偏差才是独立交叉检查。它检验整体一致性，仍不是相对于外部真值的绝对精度。检查用的棋盘格需与手腕标定时具有相同的内角点列数、行数和真实角点间距；可在手腕标定后移动。

### 准备相机服务

两台 RealSense 需有**不同 serial 和 endpoint**；下例把手腕服务放在 `18082`，第三视角放在 `18083`。两个服务均须输出深度对齐至 RGB 的帧。先查询当前实际连接的相机序列号，并根据相机所在位置确认哪台是第三视角相机；不要沿用旧文档中的序列号。

```bash
conda activate sam3
python -c 'import pyrealsense2 as rs; c=rs.context(); print([(d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.name)) for d in c.query_devices()])'
```

```bash
conda activate sam3
python -m gene_agent.perception_modules.camera.camera_server \
  --serial 412622271769 --endpoint tcp://127.0.0.1:18082 \
  --width 1280 --height 720 --stream-fps 30 --max-fps 10
```

另开终端启动第三视角服务：

```bash
conda activate sam3
THIRD_SERIAL=337122070986  # 示例：只在确认第三视角相机确为该 D435 时使用
python -m gene_agent.perception_modules.camera.camera_server \
  --serial "$THIRD_SERIAL" --endpoint tcp://127.0.0.1:18083 \
  --width 1280 --height 720 --stream-fps 30 --max-fps 10
```

### 运行向导

```bash
conda activate zehao
THIRD_SERIAL=337122070986  # 与上面启动第三视角服务时使用的 serial 保持一致
python -m gene_agent.perception_modules.camera.calibration.guided_third_view_crosscheck \
  --crosscheck \
  --handeye data/calibration/runs/handeye/calibration_wrist_right_001/handeye.json \
  --ip 192.168.1.18 --group right \
  --wrist-endpoint tcp://127.0.0.1:18082 \
  --third-endpoint tcp://127.0.0.1:18083 \
  --third-serial "$THIRD_SERIAL" \
  --validation-poses 3 --visualize
```

省略 `--data-dir` 时自动创建带时间戳的 `data/calibration/runs/crosscheck/calibration_third_crosscheck_*`，避免覆盖已有标定。中途退出后，从终端输出中找到实际目录，重跑时指定 `--data-dir <该目录> --resume`。手腕 serial、棋盘格内角点数和格宽直接从 `handeye.json` 读取。若旧手腕标定的 `samples.json` 还在，向导会比对机器人 IP。

启动向导前可分别检查两台服务：

```bash
conda activate zehao
python -m gene_agent.perception_modules.camera.camera_client --endpoint tcp://127.0.0.1:18082 --frames 0
python -m gene_agent.perception_modules.camera.camera_client --endpoint tcp://127.0.0.1:18083 --frames 0
```

两条命令都应打印 `{'ok': True, ...}`，且 `18082` 返回手腕相机 serial、`18083` 返回第三视角相机 serial。若 `health` 超时，检查服务终端是否显示“camera server 已启动”、endpoint 是否一致、设备是否被其他进程占用。向导会短暂重试并在错误中指出具体相机和 endpoint。连接失败时输出目录尚未创建，启动服务后直接重跑原命令，无需 `--resume`。

1. **板位 A**：让棋盘格同时进入两台相机画面；两台相机须看到**同一面**。两画面的红色 0 号角点必须落在同一个物理内角点。调整手腕姿态后停稳，按空格取样；松开脚踏板，等待约 1 秒，再按空格确认。按 N 可重拍，Q 可退出。
2. **移动板**：把板移动到新的位置或方向，并固定；建议移动 10 cm 左右，并适当改变角度。脚本至少要求移动 3 cm 或旋转 5°。确保两台相机仍能看完整棋盘格，且 0 号角点物理身份未变。
3. **板位 B**：同样采集并确认 3 组配对样本。手腕相对板位 A 的方向至少改变 15°，几组之间需绕至少两个不同轴改变方向；只平移或维持原姿态无法检查手眼误差。每次采帧前手腕和板都要停稳，板位 B 期间板保持不动。脚本打印每组两条链推得的板位姿偏差。
4. **检查**：默认门槛为最大双链偏差 10 mm / 3°，并检查板位 B 的稳定性；超阈值不输出正式外参。加 `--visualize` 会在 Open3D 中显示板位 B 的两台相机点云；大板轴来自手腕链，小板轴来自第三视角链。检查棋盘格/桌面有无明显双层、错位，再在终端输入 `ACCEPT` 保存。

数值门槛仅是初筛。两相机目前顺序取帧，没有硬件同步；程序默认拒绝主机接收时间相差超过 250 ms 的帧对（可通过 `--max-pair-gap-ms` 调整），仍须保证棋盘格在整次采帧期间静止。向导使用 OpenCV SB 角点检测提高精度，并尽量与旧手腕标定的角点顺序对齐；仍要目视确认两画面的红色 0 号角点是同一个物理点。角点重投影小，并不保证深度点云或末端任务误差小。深度点云偏差还受深度噪声、RGB 深度对齐和相机安装影响。当前 PnP 使用 RGB 内参 K，并假定 RGB 已去畸变或畸变可忽略；相机协议目前没有传递畸变系数。若第二板位误差大，先检查两画面的 0 号角点、板/相机是否松动、是否在采帧期间移动，再考虑重新标定手腕相机。

### 输出

- `session.json`：参数、首个板位与第二板位的配对采样；支持 `--resume`。
- `board_a_*`、`board_b_*`：两台相机的原始 RGB-D、内参、检测到的板位姿、机器人法兰位姿和红圈角点预览。
- `quality_report.json`：候选 `T_sdk_third`、每组第二板位的双链误差、最大误差与板位稳定性；即使数值不通过也保存。
- `third_view_extrinsic.json`：数值通过且输入 `ACCEPT` 后才写入。`T_third_board_stage_a` 是你要的**板位 A 到第三视角相机**变换，`T_board_third_stage_a` 是其逆；板移动后这两个矩阵必须重新测量。`extrinsic_matrix` 是固定相机到 SDK base 的外参，`T_third_sdk` 是其逆。它不同于随关节变化的手腕 `T_sdk_camera(q)`。

没有图形桌面时可用 `--no-preview`，在终端回车采样并打开保存的候选 PNG 检查角点；该模式下不要加 `--visualize`。真机与两台相机的实际精度尚未在本仓库的离线测试中验收。
