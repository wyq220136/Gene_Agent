# Pose estimation 与 tracking

默认端口：pose server 使用 `http://127.0.0.1:18081`；它通过
`tcp://127.0.0.1:18082` 连接 camera server。

## 启动服务

先启动 camera server，再在 `foundationpose` 环境启动 pose server：

```bash
conda activate foundationpose
python -m gene_agent.perception_modules.pose_est.pose_server \
  --host 127.0.0.1 --port 18081 \
  --camera-endpoint tcp://127.0.0.1:18082 --device cuda:0
```

服务接口：

```text
POST /pose/register  首帧 register，保存 session
POST /pose/track     读取最新相机帧并执行一次 track_one
POST /pose/stop      释放 session
POST /model/unload   无活动 session 时释放 FoundationPose 权重
GET  /health         服务状态、session_count、model_loaded
```

## 批量验证任务数据

```bash
python -m gene_agent.perception_modules.pose_est.task_pose_est \
  --task-dir data/perception_test --categories all
```

指定类别并在结束后显示场景：

```bash
python -m gene_agent.perception_modules.pose_est.task_pose_est \
  --task-dir data/perception_test --categories Bottle Microwave --visualize
```

输入遵循仓库 `readme.md`：任务级 `rgbs/rgb_0.png`、`depths/depth_0.png`、
`camera/camera_parameters_0.json`，物体级 `masks/mask_0.png` 和 mesh。
普通物体使用 `meshes/obj_mesh.obj`；Microwave 只分割 handle，使用
`meshes/handle_mesh.obj` 估计 pose。结果保存为物体目录下的 `pose.json`，
该 pose 作为整个 Microwave concept 的父 pose。

## Python client

```python
from gene_agent.perception_modules.pose_est.pose_client import PoseClient

client = PoseClient("http://127.0.0.1:18081")
result = client.register(session_id="Bottle/Bottle_0", rgb_path=rgb_path,
                         depth_path=depth_path, mask_path=mask_path,
                         mesh_path=mesh_path, camera_path=camera_path,
                         camera_id="0")
result = client.track("Bottle/Bottle_0")
client.stop("Bottle/Bottle_0")
client.unload_model()  # 单物体流程结束后释放 GPU 权重
```

server 返回相机坐标系 `T_camera_object`。任务脚本在返回后执行 Microwave 修正，
修正时使用 `left_extrinsic_matrix=T_Sleft_C0`。`pose.json` 格式为：
`{"camera_parameters_0": [[4x4 pose]]}`。

实时可视化入口为 `visualization/live_tracking.py`，它只通过 PoseClient 请求
register/track/stop，不在可视化进程中加载 FoundationPose。
