# Camera server

默认端口：`tcp://127.0.0.1:18082`。该端口由 camera server 使用，供其他感知模块取 RGB-D 帧。

使用已安装 `pyrealsense2` 的环境启动；本机 `robotwin_full` 已安装，`foundationpose` 环境当前没有该包：

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/robotwin_full/bin/python -m gene_agent.perception_modules.camera.camera_server \
  --serial 323522063015 \
  --endpoint tcp://127.0.0.1:18082 \
  --width 1280 --height 720 --stream-fps 30 --max-fps 10
```

可用设备：

```text
D415斜: 323522063237
D415上: 323522063015
```

server 启动后独占指定序列号的 RealSense；其他模块通过客户端取帧：

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/robotwin_full/bin/python -m gene_agent.perception_modules.camera.camera_client \
  --endpoint tcp://127.0.0.1:18082 --frames 10
```

客户端接口：`health()`、`info()`、`get_frame()`。`get_frame()` 返回 RGB、
米制 float32 depth、帧号、时间戳、序列号和内参矩阵。
单物体拍一帧、反复试 prompt、重建并查看 FoundationPose 对齐的入口见 [实景单物体流程](../../../docs/user/scenes/realworld_single_object_perception.md)。
任务采集脚本会按项目规范保存到 `rgbs/rgb_0.png`、`depths/depth_0.png` 和
`camera/camera_parameters_0.json`；深度 PNG 使用毫米单位。相机参数中的
`left_extrinsic_matrix` 约定为 `T_Sleft_C0`。

如果启动时报 `Frame didn't arrive within 5000`，当前代码会重试短暂超时；
预热期间累计 6 次无帧仍会停止并打印序列号、请求的深度/彩色流和 USB type。
此时运行 `rs-enumerate-devices -s` 核对设备型号、USB 类型和支持的流组合，
再检查 USB 连接与其他进程是否占用相机。可按设备实际支持情况尝试较低带宽的配置，
例如在启动命令后加 `--width 640 --height 480 --stream-fps 15`。
`pipeline.start()` 成功只说明流配置被接受，不保证设备已经产出帧。

### D435 337122070986 深度无帧（2026-10-08）

本机这台 D435 在 `640x480@15` 下持续无 RGB-D 帧。分别启动流后，彩色和
IR 能收到帧，单独启动 Z16 深度流在 `15/30 fps` 下均超时；更换本机已有的
librealsense 2.55/2.58 Python 环境也未恢复深度流。同机 D405 的深度流正常。
内核日志反复出现该 D435 对应 USB 设备 `2-8` 的
`uvcvideo ... Non-zero status (-71) in video completion handler`。这些证据指向
D435 的深度传输或设备链路，当前不能靠增加应用层等待时间恢复。

请在现场先关闭所有 RealSense 程序，重新插紧 D435 两端的 USB 线，并换一根
确定支持 USB 3 数据传输的线，尽量接主板直连 USB 3 端口。临时断开其他相机可排除
共享控制器或供电影响。重新运行上述 camera server 命令并查看系统日志：

```bash
journalctl -k -n 100 --no-pager | rg 'uvcvideo|usb 2-8'
```

若换线、换端口后仍无深度帧，请用 RealSense Viewer 单独开启该 D435 的 Depth 流；
若 Viewer 也失败，应继续排查相机硬件或固件，勿把这次失败归因于 RGB-D 对齐代码。

## 录制多相机 RGB-D 帧

`record_dual_frames` 直接为每个序列号创建独立的 `RealSenseStream`，适合
连续录制，不依赖 camera server 的“只保留最新帧”缓存：

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/robotwin_full/bin/python -m gene_agent.perception_modules.camera.record_dual_frames \
  --serials 323522063237 323522063015 \
  --out-dir data/chailiang/video \
  --fps 5 --duration 10
```

不指定 `--duration` 时按 Ctrl+C 停止。输出为：

```text
data/chailiang/video/<serial>/rgb/<timestamp>.png
data/chailiang/video/<serial>/depth/<timestamp>.npy
```

其中 RGB 是 RGB 顺序的 `uint8` PNG，深度是已经对齐到 RGB、单位为米的
`float32` NumPy 数组。同一时间戳的 PNG 和 NPY 属于同一帧。
