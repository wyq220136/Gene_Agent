# 板移动后：MoveL 回到旧位姿、重新拍板、组合完整外参

此向导复用 `handeye.json` 的 `T_flange_camera`。棋盘格可以已经移动过；从这次开始，固定棋盘格和第三视角相机，整次采集保持不动。
机械臂/手腕相机的安装和 SDK base 必须与手眼标定时一致。无需重做 20 姿态手眼标定，也不使用旧板位或旧照片。

## 启动相机服务

若已有相同服务在运行，直接复用；每台设备只能由一个采集进程打开。

手腕服务，在一个终端运行：

```bash
conda activate sam3
python -m gene_agent.perception_modules.camera.camera_server \
  --serial 412622271769 --endpoint tcp://127.0.0.1:18082 \
  --width 1280 --height 720 --stream-fps 30 --max-fps 10
```

固定第三视角 D435，在另一个终端运行：

```bash
conda activate sam3
python -m gene_agent.perception_modules.camera.camera_server \
  --serial 337122070986 --endpoint tcp://127.0.0.1:18083 \
  --width 1280 --height 720 --stream-fps 30 --max-fps 10
```

## 一条命令完成向导

```bash
conda activate zehao
python -m gene_agent.perception_modules.camera.calibration.guided_fixed_board_extrinsic \
  --handeye data/calibration/runs/handeye/calibration_wrist_right_001/handeye.json \
  --replay-samples data/calibration/runs/handeye/calibration_wrist_right_001/samples.json \
  --pose-indices 1 11 20 \
  --ip 192.168.1.18 --group right \
  --wrist-endpoint tcp://127.0.0.1:18082 \
  --third-endpoint tcp://127.0.0.1:18083 \
  --third-serial 337122070986 \
  --execute --speed-percent 5 --visualize
```

`--pose-indices` 是旧样本编号，可按实际视角选择其他编号，不要求这三个旧目标都适合移动后的板位。
只想查看目标而不连接任何设备，在同一命令后加 `--list-poses`。
省略 `--execute` 则只读机器人，由人摆手腕，直接回车采样。

1. 固定当前板和第三相机，确保两相机能看到同一面棋盘格。两个画面红色 0 号角点必须是同一个物理内角点。
2. 每个姿态先显示旧法兰目标和原关节记录与当前读数差（关节差只供参考）。检查通路后，在终端直接按回车发送一次 MoveL 位姿目标；`c` 加回车使用当前姿态采样；`q` 加回车退出。此时等待终端输入，不打开相机窗口。
3. 机械臂到位后，程序用实际 FK 的 TCP 位姿确认目标（2 mm / 0.5°），才打开双相机实时窗口；选择 `c` 加回车也会直接打开窗口。完成该组采样后窗口关闭，再返回终端选择下一目标。若板不在旧视角内，可通过示教器调整手腕；采样时读取的是调整后的实际位姿。
4. 停稳后在窗口按空格取样，松开脚踏板约 1 秒，再按空格确认；N 重拍、Q 退出。板和第三相机始终保持固定。
   实时预览中需两相机都检测到合格棋盘格才能取样；窗口显示 `READY` 时可按空格或回车。显示 `BLOCKED` 时查看终端的具体原因，按采样键也会打印原因，不会静默忽略。请先点击图像窗口使其获得键盘焦点；检测失败时检查整板可见、内角点数量和图像清晰度，保持板与第三相机固定。
5. 第一组新样本求候选外参，之后几组仅做独立检查。每组重新通过 SDK 读法兰 FK，拍摄当前板；不会用旧目标位姿代替实际读数。
6. Open3D 中，手腕和固定第三视角点云均保留原始 RGB 颜色，全部变换到 SDK base。检查静态棋盘格和桌面是否重合；运动机械臂的点云不要求重合。只显示基座和当前棋盘格两组坐标轴。
7. 数值通过并检查后，在终端输入 `ACCEPT` 保存正式外参。

回放使用厂家 SDK **`rm_movel`**，默认速度参数为 5%，限制 1–10%。从旧记录读取 `T_sdk_flange`，自动查询当前工具和工作坐标系，将法兰目标换算为工作坐标系下的当前工具 TCP 目标。**当前工具 TCP 走直线，末端到位后关节解可能与旧记录不同**；原关节读数只作参考。直线失败即停止报错，绝不降级为 MoveJ。没有当时手掰的运动路径或现场障碍物碰撞规划；逐点确认前需检查当前起点到目标的直线通路。回放失败、超时或 Ctrl+C 会尝试停止并清除队列，不自动重发或回 home。本功能只做了离线验证，未执行真机运动验收。

工具/工作坐标系查询遇到 SDK `-1/-2/-3` 时最多读取 3 次、间隔 0.3 秒。发令前仍读不到时不运动，记录 `not_sent` 并回到确认界面，可重新检查连接后回车重试，或选 `c` 加回车人工采样；不会默认坐标系为单位阵。运动后复查失败则停止并退出，不自动重发运动。输入支持全角英文与粘贴边界格式字符；未识别的命令会显示原始内容，回车使用当前模式的默认动作，`c` / `q` 加回车可选择当前姿态或退出；原 `MOVE` / `CURRENT` / `Q` 完整输入仍兼容。

### 通信与回执异常

参考 `umi_data_train` 的真机实现，本向导默认通过 **UDP 推送读取关节状态**，通过同一个 SDK 连接发送控制指令。默认端口 8888、推送周期 10 ms，自动按机器人网段选择本机回传 IP；SDK 接收超时在连接前设为 2000 ms。启动必须收到匹配机器人 IP 的有效七轴状态，才进入操作选择。

UDP 状态最大帧龄为 0.2 秒，每次读取必须是未消费过的新帧；等待新帧超过 0.5 秒会报错。运动中持续读不到状态会尝试停止，拒绝用旧数据宣称到位。不会静默切换 TCP。可显式使用 `--state-source tcp` 做对照；TCP 模式的关节读取对 SDK `-1/-2/-3` 最多重读 3 次、间隔 0.1 秒。

**MoveL 返回 `-1/-2/-3` 表示通信回执不确定，不能据此认定机器人没有运动。** 程序只发令一次，保留原连接，继续按新鲜实测关节与 FK 判断到位；记录原始回执和实测结果。控制器明确拒绝、未到位超时或持续失联则停止退出，绝不重发 MoveL 或降级 MoveJ。上文“直线失败停止”指这些明确失败；回执异常本身不会立即判失败。

需要指定网络时使用 `--udp-target-ip <本机有线网卡IP> --udp-port 8888`；可用 `--sdk-timeout-ms 2000` 和 `--udp-timeout 0.5` 调整接收超时。不要同时运行 UMI 或另一份机器人控制程序：控制器的 UDP 推送目标是共享配置。向导启动会设置推送目标；退出解除本进程回调，但不恢复控制器先前的推送配置。下一次由实际使用该机器人的程序重新配置。连接参数与实际状态源写入 `session.json` 和最终外参记录。

## 矩阵与数值检查

```text
T_sdk_board = T_sdk_flange_current @ T_flange_wrist_existing @ T_wrist_board_current
T_sdk_third = T_sdk_board @ inverse(T_third_board_current)
```

`T_A_B` 把 B 中的点转换到 A，平移单位米。第一组新观测求解，后续新观测验证；不是用同一组观测拟合并宣称验证通过。
默认要求双链偏差和 SDK 下的板位稳定性均不超过 10 mm / 3°；固定相机所见板位变化超过 5 mm / 2° 时当场拒收，提示检查板移动、相机移动或原点翻转。
正式保存至少需要一组相对首帧手腕**实际旋转不少于 10°**的新观测，重复拍摄同一实际姿态不计为独立验证。若只选择一个旧姿态，仍可求解和预览候选，但不会生成正式外参。
这些是内部一致性检查，不是相对于外部真值的绝对精度。PnP 继续使用当前协议的 RGB 内参 K，并假定畸变已去除或可忽略。

## 保存、续采与重新预览

自动生成新目录 `data/calibration/runs/fixed_board/calibration_fixed_board_*`，不覆盖原手眼目录。

- `session.json`：旧目标来源、每次实测 flange/关节、相机内参、当前板观测、运动事件与状态。
- `pose_*_wrist.npz`、`pose_*_third.npz`、角点 PNG：本轮新拍摄的 RGB-D 和预览。
- `quality_report.json`：候选矩阵、各次独立检查、板位稳定性；数值失败也保留。
- `board_to_robot.json`：**当前**板到 SDK base 的 `T_sdk_board`。
- `camera_to_board.json`：当前板到第三相机的 `T_camera_board` 和逆矩阵。
- `third_view_extrinsic.json`：固定第三相机到 SDK base 的 `extrinsic_matrix`，以及逆矩阵 `T_third_sdk`。

最后三个正式矩阵文件只在数值通过且输入 `ACCEPT` 后生成。
正常中断后，在原命令中增加 `--data-dir <实际目录> --resume`，保持当前板位和相机安装不变。已采好的姿态不会重复运动或覆盖。
旧双相机移板 session 和纯板相机 session 不能用于此模式。

重新查看已保存的候选或正式结果，不需要连接相机和机械臂：

```bash
conda activate zehao
python -m gene_agent.perception_modules.camera.calibration.guided_fixed_board_extrinsic \
  --preview-only --data-dir <实际输出目录>
```

没有图形桌面时，加 `--no-preview`、去掉 `--visualize`；终端回车取帧、y 确认，事后在桌面使用 `--preview-only` 检查。

已判失败的目录不允许续采。`data/calibration/runs/fixed_board/calibration_fixed_board_20261023_164942` 已按用户要求标记失败，保留原记录；下一次请省略 `--data-dir` 和 `--resume`，自动创建新目录。MoveJ 旧模式目录也不能恢复为本版 MoveL 流程。
