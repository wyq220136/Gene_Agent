# Perception modules client

受众：需要单独运行相机、分割、位姿估计和重建服务的开发者。状态：已完成包导入与路径迁移；
未在本目录启动真实相机、SAM3、FoundationPose 或 SAM3D 推理。默认感知工作区是
`data/perception_test/`，使用前需放入对象目录和必要 mesh。输出 `pose.json` 保留
`T_camera_object` 语义，不能直接作为仿真 world pose。

本机 SAM3、SAM3D、FoundationPose 的环境版本、权重目录、安装经验和 FoundationPose 原始踩坑记录见 [感知环境配置](../../docs/developer/perception/perception_runtime_setup.md)。
实景中一次用一台 RealSense 为一个物体拍一帧、反复试 prompt、重建并检查 FoundationPose 位姿，见 [单物体实景流程](../../docs/user/scenes/realworld_single_object_perception.md)。

四个模块都支持从自己的 `__init__.py` 直接导入主要接口：

```python
from gene_agent.perception_modules.camera import CameraClient
from gene_agent.perception_modules.segmentation import segment_image
from gene_agent.perception_modules.pose_est import PoseClient
from gene_agent.perception_modules.recon_3d import reconstruct_object
```

先启动对应的 server，再从仓库根目录运行 Python。默认端口如下：

| 功能 | 默认地址 | client |
| --- | --- | --- |
| camera RGB-D | `tcp://127.0.0.1:18082` | `camera.camera_client` |
| segmentation | `http://127.0.0.1:18080` | `segmentation.seg_client` |
| pose estimation/tracking | `http://127.0.0.1:18081` | `pose_est.pose_client` |
| SAM3D reconstruction | `http://127.0.0.1:18083` | `recon_3d.recon_client` |

## 1. Camera：读取 RGB-D

```python
from gene_agent.perception_modules.camera import CameraClient

camera = CameraClient("tcp://127.0.0.1:18082")
try:
    packet = camera.get_frame()
    rgb = packet.rgb                  # HxWx3 uint8
    depth = packet.depth              # HxW float32，单位：米
    intrinsic = packet.intrinsic_matrix
finally:
    camera.close()
```

也可以调用 `camera.health()` 和 `camera.info()` 检查服务状态。

## 2. Segmentation：按 prompt 分割

```python
from gene_agent.perception_modules.segmentation import segment_image

status, result = segment_image(
    "http://127.0.0.1:18080",
    "data/perception_test/rgbs/rgb_0.png",
    [{
        "category": "Bottle",
        "prompt": "a bottle",
        "mask_path": "data/perception_test/objs/Bottle/Bottle_0/masks/mask_0.png",
    }],
)
if status != 200 or not result.get("ok"):
    raise RuntimeError(result)
```

`items` 可以包含多个物体；每项至少需要 `category`、`prompt`、`mask_path`。

## 3. Pose：首帧估计和实时 tracking

```python
from gene_agent.perception_modules.pose_est import PoseClient

client = PoseClient("http://127.0.0.1:18081")
session_id = "Bottle/Bottle_0"
registered = False
try:
    first = client.register(
        session_id=session_id,
        rgb_path="data/perception_test/rgbs/rgb_0.png",
        depth_path="data/perception_test/depths/depth_0.png",
        mask_path="data/perception_test/objs/Bottle/Bottle_0/masks/mask_0.png",
        mesh_path="data/perception_test/objs/Bottle/Bottle_0/meshes/obj_mesh.obj",
        camera_path="data/perception_test/camera/camera_parameters_0.json",
    )
    registered = True
    pose = first["T_camera_object"]       # 4x4 矩阵
    current = client.track(session_id)    # server 从 camera server 取最新帧
finally:
    if registered:
        client.stop(session_id)
```

`track()` 可以循环调用；结束后务必调用 `stop()` 释放 session。

## 4. SAM3D：重建 mesh

```python
from gene_agent.perception_modules.recon_3d import reconstruct_object

result = reconstruct_object(
    "data/perception_test/objs/Bottle/Bottle_0",
    server="http://127.0.0.1:18083",
    simplify_mesh=True,
)
print(result["output_path"])
```

函数会按项目数据结构自动推断 RGB、depth、mask、相机内参和 mesh 输出路径；
也可以传入 `rgb`、`depth`、`mask`、`intrinsics`、`output` 覆写路径。

## 5. 一键运行感知测试目录

先启动 camera、segmentation 和 pose server，然后在仓库根目录执行：

```bash
python -m gene_agent.perception_modules.run_task_perception
```

脚本会自动取任务初始 RGB-D、执行分割，并为 `data/perception_test/objs` 下配置中的全部物体估计 pose。
也可以指定类别：

```bash
python -m gene_agent.perception_modules.run_task_perception --categories Microwave Bottle
```

只处理一个 Bottle 实例，不经过 task level pose 批处理：

```bash
python -m gene_agent.perception_modules.run_single_object_perception --category Bottle --visualize
```

如果该类别有多个实例，使用 `--object-dir` 指定目录。
