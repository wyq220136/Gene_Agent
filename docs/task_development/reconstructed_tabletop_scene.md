# 从单帧 RGB-D 构建桌面场景：刷子实例

本页是具体任务实例，不是任意 mesh / URDF 的通用导入协议。流程与场景验收见[场景与成功条件](scene_and_goal.md)。

本例使用 `data/realworld/brush_0` 的冻结拍摄结果，生成刷子、绿方块、簸箕和一台 Realman 单臂 Pika 夹爪的场景；另复制绿方块的几何资产、赋予红蓝颜色，得到红、蓝方块。这两个方块是**仿真增加的目标**，并非照片中额外重建出的物体。名义布局经过任务需要重新排布，三块位置与朝向不同，不声称复现拍摄时的机械臂相对位置。完整 MuJoCo 任务与随机评测见[清扫 Skill 用户说明](../user/sweep/sweep_cube_mujoco_skill.md)。

另一张 `my_scene_0` 照片中的两刷、两勺、一杯，以及清扫/倾倒/舀取三个场景草案，见[三任务场景预览](../user/scenes/my_scene_0_three_task_previews.md)。

## 用一台 RealSense 拍摄一帧

在仓库根目录运行下面的命令；将 `--serial` 改为现场选用的 RealSense 序列号，将输出目录改为新场景名称。此命令直接打开相机，拍摄一帧后退出，**不需要启动 camera server**：

```bash
PYTHONPATH=. PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/robotwin_full/bin/python \
  tools/run/capture_realsense_frame.py \
  --serial 337122070986 \
  --output-dir data/realworld/my_scene_0 \
  --width 640 --height 480 --stream-fps 15
```

`robotwin_full` 环境提供 `pyrealsense2`。运行前可用 `rs-enumerate-devices -s` 查看序列号，并关闭其他占用同一相机的程序。默认采用 `640×480@15`；其他分辨率和帧率必须是该相机同时支持的彩色、深度流组合。命令会生成 `rgb.png`、与彩色图对齐的 `depth.png`（uint16 毫米）和 `camera.json`（内参、序列号、帧号、时间戳）。已有这三个文件中的任意一个时会拒绝覆盖；重新拍摄请换新目录。此前序列号 `337122070986` 的 D435 曾出现深度流持续无帧，排查方法见[相机说明](../../gene_agent/perception_modules/camera/readme.md)。

拍照只提供原始 RGB-D。下面这个具体场景构建入口还需要同一采集目录中的刷子 `mask.png` 和 `meshes/obj_mesh_simplified.obj`；先按[单物体感知流程](../user/scenes/realworld_single_object_perception.md)取得这两个产物，再将新目录传给 `--capture-dir`。簸箕和绿方块由场景构建命令继续处理。

## 输入与生成

在仓库根目录使用 `curobo` 环境：

```bash
PYTHONPATH=. PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/build_reconstructed_tabletop_scene.py \
  --capture-dir data/realworld/brush_0 \
  --scene-dir data/realworld/brush_cube_dustpan_0
```

输入目录需要 `rgb.png`、毫米深度的 `depth.png`、`camera.json`，以及已有刷子的 `mask.png` 和 `meshes/obj_mesh_simplified.obj`。命令自动对簸箕、绿方块运行 SAM3，簸箕再运行 SAM3D；绿方块由 mask 和深度生成等边 box。模型服务按阶段启动、卸载、停止，避免 SAM3 与 SAM3D 长驻显存。已有且与输入一致的产物会复用。首次运行需要本机 `sam3`、`sam3d` conda 环境及其权重。

修改 prompt 时可传 `--dustpan-prompt "..." --cube-prompt "..."`；修改输入照片、prompt 或已有 mask 会触发重新推理。要强制重跑可传 `--refresh-perception`。只有想从现有感知产物重建 MJCF 时传 `--no-inference`。本例 `--cube-scale` 默认 0.5；新场景若无需该尺寸修正可设为 1.0。不同拍摄结果建议使用新的 `--capture-dir` 和 `--scene-dir`，以便保留各次资产与记录。服务日志在场景目录的 `logs/`。

## 检查场景

```bash
PYTHONPATH=. PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/view_reconstructed_tabletop_scene.py --mode render

PYTHONPATH=. PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/view_reconstructed_tabletop_scene.py --mode viser

PYTHONPATH=. PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/view_reconstructed_tabletop_scene.py --mode mujoco
```

离屏图片在 `data/realworld/brush_cube_dustpan_0/renders/`；临时渲染建议加 `--output-dir runs/tmp/scene-preview/<本轮>`，避免混入正式场景资产。Viser 默认地址为 `http://127.0.0.1:8098`。Viser 和 MuJoCo 读取同一份 `scene.xml`，在模型有 keyframe 时都从第一个 keyframe 开始（本例为 `scene_start`）。可用 `--scene-dir` 打开别的输出目录。

场景记录 `scene.json` 包括每个物体的来源和 SHA256、源 mesh 到规范物体 frame 的变换、`T_world_object`、尺寸、碰撞代理、单帧相机观测中心及机械臂的纯位置 IK 检查。名义世界 XY：刷子 `(0.45,-0.22)`，绿块 `(0.40,-0.01)`，红块 `(0.49,-0.05)`，蓝块 `(0.445,0.06)`，簸箕 `(0.45,0.22)`，单位米。三块 yaw 分别为 −20°、+30°、−36°。`randomization` 给出用户选定的 B 档独立均匀 XY/yaw 扰动、桌面边界和最小间距；方块最多偏移 9/7 cm 与 90°，刷子最多偏移 6/5.5 cm 与 35°。随机初态的采样与落稳 pose 保存在每个 run 中。`perception/manifest.json` 保留原图像哈希、mask、prompt 和模型输出路径。`task.json` 要求三块都入盘。`scene.xml` 及 `assets/` 是可直接加载的 MuJoCo 文件。

## 开发其他新场景

1. 用一台相机拍一帧 RGB-D，保存内参、深度单位与原图；冻结输入，避免重建中途换图。
2. 为每个对象确定 `tool`、`movable_target`、`receptacle` 等角色，并用 prompt 逐个检查 mask。薄小物体若单视角模型不稳定，可用深度约束的几何代理，必须在记录里标明来源。
3. 保存原始 mesh、mask、prompt、相机坐标观测和哈希；将 mesh 规范到物体局部 frame，明确规范化变换。单帧观测中心不能充当六自由度 pose。
4. 根据桌面尺寸、机器人工作区与任务语义选宽松摆位；检查物体不穿透、支撑稳定、纯位置可达。接着验证抓取姿态、碰撞、规划、接触和任务终态，不能把纯位置 IK 当作动作成功。
5. 把任务目标、初态有效规则和成功判定补进任务配置，分别记录 Viser 观察、MuJoCo 执行和真机验证证据。

本例中簸箕固定在桌面，铲口朝向工作区（世界 -Y）；刷子和三块方块是 freejoint。刷子的长柄沿桌面延伸，刷毛面朝下并平行于桌面，资产变换记录在 `T_object_source_mesh`。刷子 collision 使用 VHACD 分解的 4 个凸 mesh，文件在 `assets/brush_collision_*.obj`，参数、哈希和每块体积见 `scene.json`；原总质量 0.12 kg 保持不变。三块 collision 都是同尺寸 box；用户根据实物大小把绿方块场景边长从 44.8152 mm 调整为 22.4076 mm，红蓝块沿用该尺寸与 Concept。簸箕是随物体转向的开口托盘代理，均与可视 mesh 分开。簸箕托盘碰撞面与桌面齐平，保证方块能从铲口滑入。单帧 SAM3D 背面和刷毛细节可能不准确；桌面摆位和簸箕朝向是人为指定；Pika 的实际 TCP 与扫动时的接触动力学尚未实物验收。

红色碰撞凸块与刷子 visual 的对照见[刷子碰撞预览](../../data/realworld/brush_cube_dustpan_0/renders/brush_collision.png)。若使用新的刷子 mesh 重建场景，`curobo` 环境需有 `vhacdx`，场景生成器会重新分解并写入 `scene.json`；无法分解时直接报错。

刷子、方块和簸箕的功能 part Concept 已另存于 `concepts/`，查看命令和尺寸见[清扫场景标注](../user/scenes/sweep_scene_concepts.md)。这些标注使用场景 object frame，与 `scene.json` 的摆位通过 `T_world_object @ T_object_part` 组合。

## 本阶段验收与失败处理

这是静态场景预览，MuJoCo 查看窗口不执行 Skill，也不证明物体已通过落稳或抓取试验。先交付不同初态的截图及实际场景目录的查看命令，再按[仿真迭代](simulation_iteration.md)验证动作。已有输出与缓存必须核对输入来源；无法重建时保留错误与原始采集，不用默认示例替代新输入。
