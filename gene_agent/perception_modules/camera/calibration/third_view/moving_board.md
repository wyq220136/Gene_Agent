# 固定机械臂与两台相机，只移动棋盘格

适用于已有 `T_flange_camera` 手眼标定、希望避免机械臂运动的第三相机基座外参标定。相机与法兰的安装必须与已有标定一致。

## 启动

复用已运行的相机服务：手腕 `tcp://127.0.0.1:18082`、第三视角 `tcp://127.0.0.1:18083`。实际相机序列号必须与命令匹配，每台设备只由一个相机服务打开。

```bash
conda activate zehao
python -m gene_agent.perception_modules.camera.calibration moving-board \
  --handeye data/calibration/runs/handeye/calibration_wrist_right_001/handeye.json \
  --ip 192.168.1.18 --group right \
  --wrist-endpoint tcp://127.0.0.1:18082 \
  --third-endpoint tcp://127.0.0.1:18083 \
  --third-serial 323522063939 \
  --poses 8 --validation-poses 2 --visualize
```

本入口没有 `--execute`、`--pose-indices` 或运动目标，也不发送机械臂运动指令。机器人只用于读取当前关节和SDK FK。默认使用UDP遥测（8888端口、10ms周期、0.2秒最大帧龄），连接会设置控制器推送目标；退出注销回调但不恢复推送配置。不要同时运行其他机器人控制程序。可显式 `--state-source tcp` 作对照。
已有handeye需携带 `robot_sdk_ip`，或保留其 `source_samples` 指向的同目录 `samples.json`，以核对机器人来源，避免组合到错误基座。

## 操作

1. 选一个让两台相机共同视野足够大的手腕姿态。**从启动向导起，机械臂和两台相机全程不动。**
2. 连接成功后自动打开双相机实时预览。板由人移动，每个板位停稳后，两台相机都须检测到完整棋盘格。角点原点必须对应同一物理内角点。
3. 点击图像窗口，按空格取样；松开脚踏板约1秒，再按空格确认。N重拍、Q退出。`BLOCKED`的具体原因会显示在终端。
4. 每次将板相对所有已采板位移动至少3cm或旋转5°；采用不同位置、距离、倾角，避免仅重复同一画面。板须平整，避免反光、遮挡和贴近画面边缘。
5. 默认前6个板位用于估计两相机相对变换，最后2个板位只验证，不回拟合。每个板位相对初始机器人状态须保持在1mm/0.2°法兰和0.2°最大关节差以内；超出终止采集。
6. 向导打印逐板位毫米/角度偏差，训练与验证样本都必须在默认10mm/3°内，不自动丢弃坏样本。通过只代表内部一致性。
7. Open3D只叠加**第一板位同一组配对点云**，手腕和第三相机均保留原始RGB颜色，全部在SDK base下；检查同一板和静态桌面有无双层。不会把不同板位点云叠在一起要求它们重合。
8. 数值通过且人工检查后，终端输入 `ACCEPT` 发布正式外参；失败只保留候选与原始数据。

## 计算与输出

约定 `T_A_B` 把B中的点转换到A，平移单位米。

```text
T_wrist_third_i = T_wrist_board_i @ inverse(T_third_board_i)
T_wrist_third   = mean(前若干训练板位的相对相机变换)
T_sdk_wrist     = T_sdk_flange_initial @ T_flange_wrist_existing
T_sdk_third     = T_sdk_wrist @ T_wrist_third
```

均值为等权平移均值与旋转矩阵投影均值，属于多观测变换平均，不是像素级联合优化。留出板位比较 `T_wrist_board_i` 与 `T_wrist_third @ T_third_board_i`。

新目录 `data/calibration/runs/moving_board/calibration_moving_board_*` 保存：

- `session.json`：初始实测法兰/关节、已有手眼变换、参数、实际通信状态、每个板位的图像/观测来源。
- `board_*_wrist.npz`、`board_*_third.npz`、`board_*_corners.png`：各组RGB-D、内参及角点预览。
- `quality_report.json`：候选外参、逐板位误差、估计/独立验证划分。
- `third_view_extrinsic.json`：通过且ACCEPT后保存 `extrinsic_matrix` 与逆 `T_third_sdk`，兼容现有外参矩阵方向。
- `board_poses.json`：通过且ACCEPT后保存**每个板位各自**的板到基座、板到两相机变换和相应逆矩阵。移动板没有唯一的固定 `board_to_robot.json`。

原来的板到基座估计和旧照片不参与求解。当前RGB协议未提供畸变系数，PnP沿用零畸变假设；实际流应已去畸变或确认畸变可忽略。

**本模式无法独立检验已有手眼变换的绝对准确性。** 手眼变换中的固定偏差会传入基座外参，多板位检查仍可能全部通过。摄像头安装被人碰动不能仅靠机器人状态检查发现。

## 续采与离线预览

中断后保持机器人和两相机不动，在原命令后追加：

```bash
--data-dir data/calibration/runs/moving_board/calibration_moving_board_<实际时间> --resume
```

板允许移动；恢复后先核对机器人仍在初始姿态，再从下一个板位继续。不能复用旧固定板回放模式目录。

采完后离线预览，不连接任何设备：

预览默认仅显示各相机光轴深度小于 **1.2米** 的点，隐藏远处背景；原始数据和标定计算不受影响。可在下面命令后加 `--max-depth 1.0` 调整范围，单位米。

```bash
python -m gene_agent.perception_modules.camera.calibration moving-board \
  --preview-only --data-dir data/calibration/runs/moving_board/calibration_moving_board_<实际时间>
```

无图形桌面可用 `--no-preview` 并去掉 `--visualize`：终端回车取帧、y确认，之后在桌面离线预览。

本版已做离线几何和假设备流程验证，未做本方案的真实双相机采集验收。
