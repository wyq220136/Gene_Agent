# 单臂相机外参标定

本流程适用于只有一只 Realman 机械臂的安装；以下以**右臂**为例。输出 `T_Sright_Camera`，把相机坐标中的点变换到右臂 **SDK base**。整个标定计算不使用旧双臂间的相对位姿，也不额外旋转 45°。若实际使用左臂，把命令中的 `--side right` 改成 `--side left`。

优先使用下面的**终端向导**，它按摆板、自动读取机械臂 TCP、相机采集与 Open3D 检查、量尺、外参复核依次提示。后面的分步命令用于离线重跑单个环节。

## 坐标和前提

- 长度单位为**米**。在线向导会自动读取这只机械臂的七轴关节角，并通过 RealMan SDK FK 和当前夹爪安装配置求 SDK base 中的 TCP 位姿；用户无需抄示教器数值。不能把规划器、MuJoCo 或另一个机器人的 base 坐标当作本次标定数据。
- 棋盘格原点是有物理标记的左上角**内角点**；板 +X 沿 SDK base +X，板 +Y 沿 SDK base +Y，板 +Z 向上。若提供完整 TCP 位姿做额外检查，其目标姿态为 +X 沿 SDK +X、+Y 沿 SDK −Y、+Z 向下，允许角度误差最多 2°。此处的 180° 翻转是**向下 TCP 对向上板轴**的关系，和旧安装的 45° 无关。
- 在 TCP 测量后、相机采集期间，板必须保持固定。相机窗口中确认红圈角点 0 落在物理标记上。单点代回误差为零不能证明标定准确，最后要用点云中的夹爪位置做独立检查。
- “带字的一面朝相机、黑色方格在画面左上角”可用于摆板，但不足以唯一确定 OpenCV 的角点编号；仍要确认**带物理标记的内角点**就是红圈 0。量尺需要给出 TCP 相对板原点的 `x y z` 三个带符号分量，单个直线距离不够。
- 这种方法由一个点确定平移、由人工对轴确定旋转。若对轴误差为 1°，离板 0.5 m 的目标可出现约 9 mm 横向误差；精度要求更高时，应增加第二个独立已知点检查，或用至少三个不共线的板点和 SDK TCP 测量来估计板姿态。
- 当板轴已与 SDK base 同向，`T_SDK_BOARD` 的旋转就是单位阵，平移为 `p_SDK_TCP - p_BOARD_TCP`；再与相机采集的 `T_BOARD_CAMERA` 相乘即可。完整 TCP 朝向不是该计算的必需量，也不会在只输入位置时被伪造为实测值。
- 旧 `board_to_robot.py` 所需的 `RealmanArm` 后端仍缺失；新向导改用 `gene_agent.realworld` 中现有 SDK FK/TCP 适配。它只读关节和计算 FK，不配置夹爪，也不发送运动命令。默认夹爪 `T_flange_tcp` 取 `config/robots/standard_gripper.json`；若实物 TCP 安装与该配置不同，须先校准配置，否则自动读取的 TCP 位置仍会有系统误差。真机连接与外参效果尚未由此仓库的离线测试验收。

## 1. 新建一次标定目录，摆好机械臂和板

建议每次使用新的目录，例如 `data/calibration/runs/manual/calibration_single_right`，避免旧 `mounted` 安装的外参混入。将夹爪 TCP 移到棋盘格已知点后启动向导；它会在你确认摆板后**自动从右臂 API 读取 TCP**。无需准备 4×4 文件，也无需人工抄示教器坐标。

```bash
mkdir -p data/calibration/runs/manual/calibration_single_right
```

用格尺测量 TCP 在板坐标系中的位置，例如对准板原点上方 0.15 m 时为 `0 0 0.15`。虽然 API 会自动读取完整 TCP 矩阵，位置法仍**不能独立测得板的旋转**，所以必须认真确认板 +X/+Y/+Z 与 SDK base 同向。向导会把本次 API 读取的关节角和 TCP 矩阵保存为 `sdk_tcp_reading.json`。

## 一条命令运行向导

启动下面的 camera server 后，在另一终端运行：

```bash
conda activate sam3
python -m gene_agent.perception_modules.camera.calibration.guided_single_arm_calibration \
  --data-dir data/calibration/runs/manual/calibration_single_right \
  --side right \
  --robot-ip 169.254.128.19 \
  --camera-endpoint tcp://127.0.0.1:18082 \
  --pattern-size 11 8 --square-size 0.015
```

终端依次要求 `BOARD_ALIGNED`，随后自动读取并显示 TCP；相机窗口采集及 Open3D 确认 `CAMERA_OK`；输入格尺量得的板坐标 `x y z`；确认 `BOARD_UNCHANGED`；让夹爪回到记录位置后确认 `TCP_VISIBLE`；最后在第二次 Open3D 检查后输入 `ACCEPT`。相机检查可以输入 `RETRY` 重新拍。未输入最终 `ACCEPT` 时，候选保留在标定目录，但不会更新任务相机参数。若需要确认后自动写入任务参数，加 `--task-camera-parameters data/task_1/camera/camera_parameters_0.json`；目标文件须先存在。`--tcp-pose-json` 和 `--sdk-tcp-position` 仅供不连接机械臂时离线复现。

## 2. 板固定后，采集相机到板

相机所在机器先启动服务：

```bash
conda activate sam3
python -m gene_agent.perception_modules.camera.camera_server \
  --serial 323522063015 --endpoint tcp://127.0.0.1:18082 \
  --width 1280 --height 720 --stream-fps 30 --max-fps 10
```

在另一个终端采集。保持板不动；必要时把机械臂移出相机视野。按 SPACE 前确认红圈 0 是物理标记角点：

```bash
conda activate sam3
python -m gene_agent.perception_modules.camera.calibration.camera_to_board \
  --camera_endpoint tcp://127.0.0.1:18082 \
  --pattern_size 11 8 --square_size 0.015 \
  --data_dir data/calibration/runs/manual/calibration_single_right
```

此步生成 `camera_to_board.json` 和含内参的 `cam_param.json`。棋盘格尺寸应改成你实际使用的规格。`--headless` 无法确认物理角点 0，不适用于这个流程。

## 3. 合并单臂外参

板保持与步骤 1 相同的放置，确认后执行：

```bash
python -m gene_agent.perception_modules.camera.calibration.single_arm_calibration \
  --data-dir data/calibration/runs/manual/calibration_single_right \
  --side right \
  --sdk-tcp-position 0.5 0.2 0.3 \
  --board-tcp-position 0 0 0.15 \
  --confirm-board-aligned-and-unchanged
```

程序拒绝未确认板、缺失相机角点确认或缺失采集标识。只有使用 `--tcp-pose-json` 时才检查 TCP 朝向。它保存 `board_to_robot.json`，并在 `cam_param.json` 写入 `extrinsic_matrix` 和 `right_extrinsic_matrix`；两者都表示 `T_Sright_Camera`。不会生成或使用 `cam_param_right.json`，也不运行 `convert_to_right_arm.py`。若需要给现有任务相机参数文件写入外参，可在上述命令后加：

```text
--task-camera-parameters data/task_1/camera/camera_parameters_0.json
```

该选项要求目标文件已存在，只更新其中的 `right_extrinsic_matrix`，保留帧号、时间戳和内参。使用左臂时相应写入 `left_extrinsic_matrix`。

## 4. 独立检查

可用已有可视化命令把当前 RGB-D 点云显示到这只机械臂的 SDK base 中：

```bash
python -m gene_agent.perception_modules.camera.calibration.compute_extrinsic \
  --data_dir data/calibration/runs/manual/calibration_single_right \
  --visualize --camera_endpoint tcp://127.0.0.1:18082
```

该命令会重新计算 `cam_param.json` 中的通用 `extrinsic_matrix`；如果后来修改了板上 TCP 位置，还须重新运行第 3 步来更新对应的 `right_extrinsic_matrix`。检查点云桌面、标定板和实际 TCP 是否位于正确位置；最好在第二个已知 TCP 位置重复检查。矩阵正交、行列式为 1 或同一个 TCP 点代回一致，都不足以证明实际外参准确。

相机或机械臂安装、SDK TCP 定义、板位姿、相机内参改变后，需要重新标定。规划或 MuJoCo 若仍使用旧 `mounted` 的 45° `sdk_to_curobo`，也应按实际单臂安装单独核对其机器人配置；相机标定不会自动修改该配置。
