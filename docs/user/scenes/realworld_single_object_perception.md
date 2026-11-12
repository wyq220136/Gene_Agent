# 单物体实景采集、分割和重建

受众：在本机用一台 RealSense 检查一个实物的用户。状态：交互入口已实现；2026-10-13 用已保存的 `brush_0` RGB-D 和 mask 实测了 SAM3D 重建、FoundationPose 位姿估计及 Open3D 文件导出。新拍照、重新试 prompt 和交互窗口仍需现场检查。

一次运行只选**一台相机、拍一帧、处理一个物体**。锤子和刷子分别运行两次。采集后可在同一帧上反复换 prompt；确认 mask 后才运行 SAM3D，随后用带实际尺度的简化 mesh 运行 FoundationPose，并在 Open3D 中查看 RGB-D 点云与估计位姿下的 mesh。不会调用机械臂。

## 启动四个服务

从项目根目录分别在四个终端运行。下面使用你当前指定的 `337122070986`；更换相机时同时修改相机服务和交互入口的序列号。相机服务会独占选定设备。

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/robotwin_full/bin/python -m gene_agent.perception_modules.camera.camera_server --serial 337122070986 --endpoint tcp://127.0.0.1:18082
```

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/sam3/bin/python -m gene_agent.perception_modules.segmentation.seg_server --device cuda
```

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate sam3d
PYTHONPATH= PYTHONNOUSERSITE=1 python -m gene_agent.perception_modules.recon_3d.recon_server --device cuda
```

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/foundationpose/bin/python -m gene_agent.perception_modules.pose_est.pose_server --device cuda:0
```

其中相机入口使用本机已有 `pyrealsense2`、`pyzmq` 的 `robotwin_full` 环境；`foundationpose` 环境本身没有 `pyrealsense2`，不用于启动相机服务。SAM3、SAM3D 和 FoundationPose 在对应请求到来时加载模型；交互入口在确认 mask 后卸载 SAM3，重建完成后卸载 SAM3D，位姿 session 停止后卸载 FoundationPose。服务进程仍可能保留 CUDA context 和非模型显存；需要彻底释放时，正常结束对应服务进程。

如果相机启动时持续提示无帧，先运行 `rs-enumerate-devices -s` 查看 `337122070986` 的型号、USB type 和支持的流配置。服务会对短暂的 5 秒超时重试；仍无帧时会打印实际请求的 `1280x720@30` 流和诊断信息。核对设备支持后，可尝试在相机服务命令末尾加 `--width 640 --height 480 --stream-fps 15`；也需检查 USB 线、端口和是否有其他程序占用该相机。不要在未确认设备支持该组合时反复修改分辨率。

## 处理锤子或刷子

在第五个终端，从项目根目录运行：

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/robotwin_full/bin/python -m gene_agent.perception_modules.interactive_realworld_perception --object hammer_0 --serial 337122070986 --prompt "a hammer"
```

换成刷子时，先将物体放入画面，再另运行一次：

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/robotwin_full/bin/python -m gene_agent.perception_modules.interactive_realworld_perception --object brush_0 --serial 337122070986 --prompt "a brush"
```

如果换相机，先停止旧相机服务，以新序列号重启相机服务，并将交互命令的 `--serial` 改为相同值。交互入口会核对服务及采集帧的序列号，避免把另一台相机的内参与深度混用。

每次 SAM3 分割后，入口保存 `prompt_trials/overlay_001.png` 等红色叠加图，尝试用桌面默认图片查看器打开，并在终端显示路径和前景像素数。查看后输入 `y` 接受；不满意则直接输入下一个 prompt。只有接受的 mask 会复制到 `mask.png`。输入 `q` 可退出，已拍的一帧与之前的 prompt 尝试仍留在目录。若没有图形桌面，手动打开终端打印的叠加图即可。

想对**同一帧**重新试 prompt，使用 `--reuse-capture`，其余参数保持不变；这样不会重新拍照。默认拒绝覆盖已有采集帧，以免 prompt、RGB、深度和内参来自不同时刻。要拍新帧，请改用新的物体目录名。

如果分割已确认，但 SAM3D 因 OOM 失败，先关闭其他 GPU 程序；若 SAM3D 服务状态异常则重启它。使用已保存的输入继续，不需要相机或 SAM3 服务：

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/robotwin_full/bin/python -m gene_agent.perception_modules.interactive_realworld_perception \
  --object brush_0 --serial 337122070986 --reuse-mask
```

`--reuse-mask` 核对相机序列号、RGB/深度/mask 尺寸、有效深度、最终 mask 与已确认候选 mask，并在新记录中核对采集文件哈希。旧版 prompt 记录没有采集哈希，只能做文件和尺寸校验。

如果 mesh 已保存且 `meshes/reconstruction_inputs.json` 存在，可跳过 SAM3D，仅启动 FoundationPose 服务并加 `--reuse-mesh`；该选项自动复用采集帧和 mask，核对重建输入及 mesh 哈希，不需要 SAM3D 服务：

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/robotwin_full/bin/python -m gene_agent.perception_modules.interactive_realworld_perception \
  --object brush_0 --serial 337122070986 --reuse-mesh
```

OOM 时终端给出续跑提示，服务端仍保留完整 traceback。不要使用 `torch.cuda.empty_cache()` 代替卸载仍被服务引用的模型；阶段结束后的 `/health` 可检查 `model_loaded`，但该字段不代表 CUDA context 的全部显存都已归还。

## 输出和检查

以锤子为例，主要结果在 `data/realworld/hammer_0/`：

```text
rgb.png                         本次采集的 RGB
depth.png                       对齐 RGB 的 uint16 毫米深度
camera.json                     内参、相机序列号、帧号、时间戳、深度单位
prompt_trials.json              每次 prompt、mask 路径、前景像素数和接受状态
prompt_trials/mask_*.png        各次候选 mask
prompt_trials/overlay_*.png     各次分割预览
mask.png                        人工确认的 mask
meshes/obj_mesh.obj             SAM3D 原始 mesh
meshes/obj_mesh_simplified.obj  已烘焙 SAM3D scale 的 mesh
meshes/sam3d_pose.npz           SAM3D 原始姿态和尺度
meshes/foundationpose_pose.npy  FoundationPose 的 T_camera_object，单位米
visualization/scene_rgbd.ply    采集帧的彩色点云
visualization/sam3d_mesh_camera.obj  按 FoundationPose pose 放入相机坐标系的 mesh
```

最后一个文件沿用现有可视化器的历史文件名；这次流程的实际变换来自 `foundationpose_pose.npy`。Open3D 会显示彩色场景点云、红色 mesh、相机和物体坐标轴。检查物体的轮廓、尺度、旋转以及与有效深度点的重合情况。一个视角下看起来对齐，并不证明背面几何准确。

如果暂时只想检查 SAM3D，可以加 `--skip-pose`；仍会保存 mesh，但不会输出 FoundationPose pose 或打开 Open3D。需要先排查环境时参见 [感知模型本机环境](../../developer/perception/perception_runtime_setup.md)。
