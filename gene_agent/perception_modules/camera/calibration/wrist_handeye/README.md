# 手腕 RealSense 单臂手眼标定（终端向导）

适用于相机**固定在机械臂手腕/法兰上**、棋盘格固定在桌面上的 eye-in-hand 标定。程序求 `T_flange_camera`，不需要用尺测 `T_sdk_board`，也不需要旧安装的 45° 旋转。机械臂由人手动移动；程序只连接 SDK 读取七轴和法兰 FK，不发送运动或夹爪指令。

## 准备

1. 把棋盘格牢固固定在工作台。记下**内角点列数、行数和相邻角点实际距离（米）**。棋盘格在整次采集中不能移动。
2. 在棋盘格一个物理内角点附近做明显标记。每次采样都检查 `candidate.png` 的红圈 **0** 落在同一个物理内角点；棋盘格对称可能使角点序列翻转。
3. 确认手腕 RealSense 的 RGB 能完整看到棋盘格，曝光清晰；相机安装和分辨率保持不变。深度对齐到 RGB 后一并保存，但求解使用 RGB 棋盘格角点。
4. 在有 RealMan SDK、OpenCV、NumPy、pyzmq 的环境运行向导；本机 `zehao` 环境已能导入这些包。相机服务可在 `sam3` 环境启动。两进程可在同机或经网络连接。

## 运行

先启动**手腕相机**服务，把 serial 改成真实设备编号：

```bash
conda activate sam3
python -m gene_agent.perception_modules.camera.camera_server \
  --serial <手腕相机序列号> --endpoint tcp://127.0.0.1:18082 \
  --width 1280 --height 720 --stream-fps 30 --max-fps 10
```

另开终端运行；每次用新的输出目录：

```bash
conda activate zehao
python -m gene_agent.perception_modules.camera.calibration handeye \
  --ip 192.168.1.18 --group right \
  --camera-endpoint tcp://127.0.0.1:18082 \
  --serial 412622271769 \
  --pattern-size 11 8 --square-size 0.015 \
  --poses 20 --visualize --data-dir data/calibration/runs/handeye/calibration_wrist_right_001
```

上述 IP 和相机序列号是本次设备的示例，其他设备请替换。`--ip` 推荐填写纯 IPv4。若误填 `http://192.168.1.18/`，向导会先显示规范化后的纯 IP，再传给 SDK；URL 带其他路径、端口或参数会被拒绝。连接失败默认最多尝试 30 次，每次间隔 0.5 秒；每次失败都会销毁该次 SDK 实例，再建新实例。若要持续等待机器人上线，增加 `--connect-retries 0`，按 Ctrl+C 停止。校准连接保持只读，不调用清错、夹爪配置或运动 API。

只有机器人和相机都连接成功后，才会在新目录写 `samples.json`。旧版本连接失败后留下的**仅含零样本 `samples.json`** 可直接用相同命令继续；已有采样记录的目录须加 `--resume`，不会自动覆盖。若纯 IP 连续连接失败，先核对机械臂电源、网卡地址与到 `192.168.1.18:8080` 的网络可达性；重试无法修复网络或端口配置错误。

连接后会打开 **OpenCV 实时画面**。人工移动手腕到不同位置和方向时，窗口持续显示 RGB、绿色棋盘格角点、红色 **0 号角点**、帧号和 PnP 重投影误差；看见 `BOARD FOUND` 且视角合适后停稳机械臂，**第一次踩脚踏板发送空格以采样**。程序会重新取新帧并冻结候选画面，检查红圈 0 后，**松开脚踏板，再踩一次发送空格以确认收录**。仍可按 N 重拍、Q 退出，也保留 Y 确认。程序会过滤切换画面时排队的空格事件；脚踏板请短踩并松开，避免系统键盘自动重复。红圈 0 必须一直对应同一物理角点；`EDGE CLOSE` 提醒角点靠近画面边缘，宜调整位置。窗口文字使用英文是因为 OpenCV 内置字体不显示中文，终端同时保留中文提示。

建议 15–25 个姿态，绕至少两个不同轴旋转，同时让棋盘格覆盖图像中心和边缘、近处和较远处。不要只平移，也不要在采帧期间移动机器人。`--no-preview` 可用于没有图形桌面的环境，恢复原有“终端回车采样、查看 `candidate.png`、输入 y/n/q”的方式。按 Q 退出时，此前样本仍保存在 `samples.json` 中。

程序采样时先丢弃一帧，保证取得新的相机帧；读取采帧前后关节并拒绝运动中的样本；计算 PnP 重投影误差并提示确认角点。采完后用训练姿态求解 `T_flange_camera`，每第 5 个姿态留作验证。默认留出样本最大棋盘格位姿残差必须不超过 **10 mm / 3°**；否则保存 `quality_report.json` 和原始样本，不生成 `handeye.json`。可核对报告中的异常样本，再用同一目录加 `--resume --poses 25` 续采（板不能移动），或新建目录重采。`--visualize` 会在 SDK base 下叠放 3 个姿态的点云并显示棋盘格坐标轴；目视确认静态桌面和板重合后输入 `ACCEPT` 才保存结果。数值阈值仍只是初筛，最终要在任务工作距离下检查点云。

## 输出与使用

- `samples.json`：每组 `T_sdk_flange`、`T_camera_board`、七轴关节弧度、帧号、相机 serial、RGB 内参、采帧时间与重投影误差。
- `sample_NNN.npz`、`sample_NNN_corners.png`：原始 RGB-D（深度米）和红圈 0 角点预览。
- `quality_report.json`：求解候选与每个样本的棋盘格位姿残差；即使质量未通过也会保存。
- `handeye.json`：`T_flange_camera`、固定棋盘格的 `T_sdk_board_estimate`、训练和留出样本残差。

坐标约定 `T_A_B` 将 B 中坐标变换到 A；长度米。运行时先读**同一时刻**的法兰 FK，再计算：

```python
from gene_agent.perception_modules.camera.calibration.wrist_handeye import camera_pose_from_flange
T_sdk_camera = camera_pose_from_flange(T_sdk_flange, T_flange_camera)
```

手腕相机的 `T_sdk_camera` 随关节变化，**不能**将其写进固定相机的 `right_extrinsic_matrix`。此向导只标定 RGB 光学坐标系到法兰；若后续使用深度点云，必须保持 RealSense 深度对齐至 RGB，并核对点云与固定棋盘格在多个关节姿态下重合。当前相机协议只传 K，PnP 假定输入 RGB 已经去畸变或畸变可忽略；若重投影误差较大，应先扩展协议并传真实畸变系数。真机和现场点云精度尚未由离线测试验收。
