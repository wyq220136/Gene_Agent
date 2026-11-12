# SAM3D RGB-D 重建服务

默认端口：`http://127.0.0.1:18083`。

本模块使用项目内的 SAM3D，对项目数据结构中的一个物体执行 RGB-D 3D 重建，输出 OBJ mesh。

## 目录约定

输入目录遵循项目 `readme.md`：

```text
data/perception_test/
├── rgbs/rgb_0.png
├── depths/depth_0.png
├── camera/camera_parameters_0.json
└── objs/
    └── hammer_0/
        └── hammer_0_0/
            ├── masks/mask_0.png
            └── meshes/
```

客户端既可以接收类别目录 `hammer_0`，也可以直接接收 instance 目录 `hammer_0_0`。
类别目录下必须只有一个包含 `masks/` 的 instance；多个 instance 时使用 `--instance`。

## 启动服务

在仓库根目录执行：

```bash
conda activate sam3d-objects
python -m gene_agent.perception_modules.recon_3d.recon_server \
    --host 127.0.0.1 \
    --port 18083 \
    --device cuda
```

检查服务：

```bash
curl http://127.0.0.1:18083/health
```

服务启动后先监听 HTTP，默认不加载 SAM3D 模型；首次重建请求或显式
`--load-model` 时加载权重，请求会串行使用 GPU。`/health` 的
`model_loaded` 指示模型是否驻留。加载失败会作为请求错误返回，服务可继续接收控制请求。

如果需要暂时释放 SAM3D 的 GPU 显存，可以手动卸载和重新加载模型：

```bash
python -m gene_agent.perception_modules.recon_3d.recon_client \
    --server http://127.0.0.1:18083 \
    --unload-model

python -m gene_agent.perception_modules.recon_3d.recon_client \
    --server http://127.0.0.1:18083 \
    --load-model
```

两个控制命令都可以重复执行。普通重建请求发现模型未加载时，会自动加载模型后再推理。

## 最简调用

```bash
python -m gene_agent.perception_modules.recon_3d.recon_client \
    --object-dir data/perception_test/objs/hammer_0
```

外部 Python 模块可以直接调用 client 封装函数：

```python
from gene_agent.perception_modules.recon_3d.recon_client import reconstruct_object

result = reconstruct_object("data/perception_test/objs/hammer_0")
print(result["output_path"])
```

该函数和命令行使用相同的默认路径推断规则，也支持 `instance`、`rgb`、`depth`、
`mask`、`intrinsics`、`output`、`server`、`depth_scale`、`seed`、`timeout`、
`simplify_mesh`、`visualize`、`use_foundationpose` 和 `pose_server` 参数。

客户端会自动使用：

```text
RGB：        data/perception_test/rgbs/rgb_0.png
Depth：      data/perception_test/depths/depth_0.png
相机内参：    data/perception_test/camera/camera_parameters_0.json
Mask：       data/perception_test/objs/hammer_0/hammer_0_0/masks/mask_0.png
输出 mesh：   data/perception_test/objs/hammer_0/hammer_0_0/meshes/obj_mesh.obj
```

## 简化 mesh 和可视化

输出简化 mesh：

```bash
python -m gene_agent.perception_modules.recon_3d.recon_client \
    --object-dir data/perception_test/objs/hammer_0 \
    --simplify-mesh
```

会额外生成：

```text
meshes/obj_mesh_simplified.obj
meshes/sam3d_pose.npz
```

简化 mesh 会把 SAM3D 预测的 `scale` 直接烘焙到顶点中；原始 mesh 和原始 pose
文件不修改。

同时打开场景点云、简化 mesh、相机坐标系和 SAM3D mesh 局部坐标系：

```bash
python -m gene_agent.perception_modules.recon_3d.recon_client \
    --object-dir data/perception_test/objs/hammer_0 \
    --simplify-mesh \
    --visualize
```

`--visualize` 会自动生成简化 mesh，并在对象目录旁边保存：

```text
visualization/
├── scene_rgbd.ply
├── sam3d_mesh_camera.obj
└── sam3d_axes.ply
```

可视化使用 SAM3D 返回的 `rotation`、`translation`、`scale`，并将其从
PyTorch3D 坐标系转换到 RGB-D/OpenCV 相机坐标系。Open3D 窗口关闭后，本次请求才会返回。

## 使用 FoundationPose 对齐

如果已经启动 FoundationPose server，可以让它使用当前 mesh、RGB-D 和 mask
重新注册物体，并使用返回的 `T_camera_object` 做可视化：

```bash
python -m gene_agent.perception_modules.recon_3d.recon_client \
    --object-dir data/perception_test/objs/hammer_0 \
    --simplify-mesh \
    --visualize \
    --use-foundationpose \
    --pose-server http://127.0.0.1:18081
```

此模式会：

- 调用 FoundationPose server 的 `/pose/register`；
- 自动使用带有 SAM3D scale 的简化 mesh 进行注册；
- 将返回的 `T_camera_object` 保存为 `meshes/foundationpose_pose.npy`；
- 可视化时使用 FoundationPose pose，而不是 SAM3D pose；
- 请求结束后调用 `/pose/stop` 释放 tracking session。

启用该模式时，SAM3D 完成 mesh、pose 和带 scale 的简化 mesh 写盘后，会自动执行：

```text
卸载 SAM3D 模型和编译缓存
调用 FoundationPose
无论成功或失败，重新加载 SAM3D 模型
```

因此 FoundationPose 推理时不会保留 SAM3D 模型权重。

`meshes/sam3d_pose.npz` 仍会保留，用于对比两种 pose。

## 覆盖路径

显式参数优先于自动推断：

```bash
python -m gene_agent.perception_modules.recon_3d.recon_client \
    --object-dir data/perception_test/objs/hammer_0 \
    --instance hammer_0_1 \
    --mask data/perception_test/objs/hammer_0/hammer_0_1/masks/mask_1.png \
    --output ./tmp/hammer.obj
```

可覆盖参数：

```text
--instance
--rgb
--depth
--mask
--intrinsics
--output
--server
--depth-scale
--seed
--simplify-mesh
--visualize
--use-foundationpose
--pose-server
--load-model
--unload-model
```

## 输入格式

- RGB 支持 PNG/JPG，读取为 `H×W×3` RGB 图像。
- Depth 支持 PNG 和 NPY，默认按 uint16 毫米读取，`depth_scale=0.001` 转换为米。
- Mask 使用灰度图，非零像素为前景。
- 相机内参支持项目格式 `{"intrinsic_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]}`，也支持 `3×3` NPY。
- RGB、depth、mask 必须具有相同的宽高。

真实 depth 会根据相机内参反投影为 pointmap，并传入 SAM3D；不会运行 MoGe 估计深度。
