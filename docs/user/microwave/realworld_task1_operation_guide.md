# 微波炉 Task1 真机操作流程

本流程对应 `tools/run/run_microwave_task1_realworld.py`：
**启动服务 → 提供相机标定 → 自动准备感知 → plan-only → 确认后执行**。
不再创建或手工填写 `realworld_site.json`，也不需要手写门角、旋钮角状态 JSON。
程序生成的 JSON 用于保存感知结果和运行证据，操作者无需编辑。
准备感知和运行任务都默认 `no_knob`：不分割、不注册、不跟踪旋钮，也不执行旋钮动作。

`run_microwave_task1_physical.py` 是 MuJoCo 仿真入口。真机概览见[真机微波炉任务入口](realworld_microwave_entry.md)。

## 1. 使用前提

- 操作 Skill、Knowledge、阶段顺序及操作 profile 沿用原实现，本流程只调整配置与感知接线。
- 默认只预检首段；完整任务预览需显式 `--subtask all --plan-only`，可加 `--video`。实际运动需要显式 `--execute --yes`；预测规划通过不等于真机已跑通。
- 当前使用右臂 SDK base 作为场景参考，柜体和相机在本次运行期间必须固定。
- 使用任务对应的微波炉、碗和标准夹爪资产；标准夹爪 TCP、开口范围必须与规划模型一致。
- 正式使用 `--collision-mode full`，不能用 `robot_only` 掩盖场景碰撞问题。当前没有任意阶段停止的 `--stop-after-stage` 参数。
- 当前流程只分割和跟踪碗、把手，实时测量门角；桌面按已有矩形范围和场景深度直接计算，不分割、不做 pose estimation。完整任务跳过旋钮，保留第二次开门前的过渡动作。

## 2. 环境与解释器

以下为 Linux 真机的 Bash 命令。按实际安装位置设置解释器和两臂 IP：

```bash
cd /home/ubuntu/yuquanwang/Gene_Agent
PY=/home/ubuntu/anaconda3/envs/curobo/bin/python
PY_FP=/home/ubuntu/anaconda3/envs/foundationpose/bin/python
PY_SAM3=/home/ubuntu/anaconda3/envs/sam3/bin/python
LEFT_IP=169.254.128.18
RIGHT_IP=169.254.128.19

"$PY" -c "import torch, curobo, pyrealsense2, trimesh; print('运行依赖可导入')"
"$PY" -c "from Robotic_Arm.rm_robot_interface import RoboticArm; print('RealMan SDK 可导入')"
"$PY_FP" --version
FP_ROOT=gene_agent/perception_modules/pose_est/fp_pose_est/FoundationPose
PYTHONPATH="$FP_ROOT" "$PY_FP" -c "import mycpp.build.mycpp as mycpp; print('FoundationPose 扩展', hasattr(mycpp, 'cluster_poses'))"
"$PY_SAM3" -c "import sam3; print('SAM3 可导入')"
nvidia-smi
```

Gene_Agent 自有代码以 Python 3.9 为最低版本；FoundationPose 可继续使用原有 Python 3.9 环境。
无需合并 SAM3、cuRobo 和 FoundationPose 环境，也不要把 Windows Python 3.11 的
`environment/requirements.windows.lock.txt` 安装到 FoundationPose 环境。
从仓库根目录运行可直接导入本项目；若需安装包，在已准备好的环境中使用
`"$PY_FP" -m pip install -e . --no-deps`，保留原 CUDA、PyTorch 和 FoundationPose 依赖。
规划仍使用原有 cuRobo 环境及其 MuJoCo 依赖，不用 FoundationPose 环境替换规划环境。
代码兼容 Python 3.9 不代表任意旧依赖都兼容；例如规划碰撞导出需要 `mujoco.mj_copyData`，
本地 Python 3.9 测试用的 MuJoCo 3.3.7 未提供此接口。

2026-11-12 的 Linux 运行记录中，旧 dev2 cuRobo 曾出现 `Tensor has no attribute robot_spheres`。
沿用已跑通 AffordToolAct 的 vendored dev42 时，确认实际导入路径：

```bash
CUROBO_SRC=/home/ubuntu/yixuanjiang/AffordToolAct/third_party/curobo
export PYTHONPATH="$CUROBO_SRC${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "import curobo; print(curobo.__file__, curobo.__version__)"
```

## 3. 启动相机、Pose 与分割服务

查看相机序列号：

```bash
rs-enumerate-devices -s
```

分别在独立终端启动服务。相机分辨率必须与标定一致：

```bash
SERIAL=323522063015
"$PY" -m gene_agent.perception_modules.camera.camera_server \
  --serial "$SERIAL" --endpoint tcp://127.0.0.1:18082 \
  --width 1280 --height 720 --stream-fps 30 --max-fps 10
```

```bash
"$PY" -m gene_agent.perception_modules.camera.camera_client \
  --endpoint tcp://127.0.0.1:18082 --frames 10
```

若相机无法出帧，先检查 USB 3 连接、设备占用和序列号，不继续使用旧帧建立感知记录。

```bash
GPU_FP=0
CUDA_VISIBLE_DEVICES="$GPU_FP" "$PY_FP" -m gene_agent.perception_modules.pose_est.pose_server \
  --device cuda:0 --camera-endpoint tcp://127.0.0.1:18082
```

```bash
GPU_SAM3=0
CUDA_VISIBLE_DEVICES="$GPU_SAM3" "$PY_SAM3" -m gene_agent.perception_modules.segmentation.seg_server \
  --host 127.0.0.1 --port 18080 --device cuda \
  --checkpoint /home/ubuntu/yuquanwang/Gene_Agent/ckpts/SAM3/sam3.pt
```

```bash
curl http://127.0.0.1:18081/health
curl http://127.0.0.1:18080/health
```

GPU 编号按现场资源选择。准备流程先运行 SAM3，再卸载其模型并注册 FoundationPose。
Pose session 保存在服务内存中；重启 Pose 服务后需要重新准备感知。

## 4. 提供已有相机标定

提供固定相机到**右臂 RealMan SDK base** 的外参，以及相机内参和序列号。
可直接使用现有标定工具生成的 `cam_param.json` 或符合该坐标约定的第三视角标定输出，
不用复制矩阵到另一个站点文件。没有有效标定时，使用现有手动采样向导：

```bash
CALIB_OUT=data/calibration/runs/manual/microwave_third_view_$(date +%Y%m%d-%H%M%S)
"$PY" -m gene_agent.perception_modules.camera.calibration manual \
  --data-dir "$CALIB_OUT" --side right --robot-ip "$RIGHT_IP" \
  --camera-endpoint tcp://127.0.0.1:18082 \
  --pattern-size 11 8 --square-size 0.015
CAMERA_PARAMS="$CALIB_OUT/cam_param.json"
```

向导读取 SDK TCP 并指导采样；先完成标定流程和结果检查，再进入下一步。
已有标定时只需将 `CAMERA_PARAMS` 指向输出文件。移动或更换相机、改变分辨率后重新标定。

### 4.1 本轮绕 X 半转修正版（保留原始测量偏移）

2026-11-25 按操作者最新要求，已撤销此前对测量 Y/Z 数值符号的修改，
保留原始板中 TCP 测量值 **`[0,+0.17,-0.125] m`**，仅保留板到 SDK 的轴向关系
**`R_sdk_board=diag(1,-1,-1)`**。现有修正版路径已同步更新，无需更换文件名：

```bash
cd /data1/yuquan/Gene_agent
PY=/data1/yuquan/miniconda3/envs/gene_agent/bin/python
CAMERA_PARAMS="$PWD/data/calibration/runs/manual/microwave_third_view_20261125_yz_rx180/cam_param.json"
```

使用原 SDK TCP 实测点重算
`t_sdk_board = p_sdk_tcp - R_sdk_board @ p_board_tcp`，再计算
`T_sdk_camera = T_sdk_board @ T_board_camera`。
“只保留旋转修正”表示不改测量数值；外参中的平移仍须按上述锚点公式一致地重算，
不能只改旋转块而把属于另一板轴定义的平移块原样拼接。

当前板中 TCP 输入为 `[0,0.17,-0.125] m`，相机在右臂 SDK 中的位置约为
`[0.023422,1.071217,0.584066] m`。`extrinsic_matrix` 和 `right_extrinsic_matrix` 同步更新，
重新合并也保留显式板轴关系。原始 11-12 标定目录未改动；
前一版“测量符号也修改”的内容备份在修正版目录的 `revisions/before_restore_original_measurement/`。
来源、锚点检查和点云/mesh 核对结果见当前目录 `correction_report.json`。

物体 mesh 与点云一起使用同一外参变换，其相对贴合不变，不能单凭碗/把手贴合证明外参平移正确。
应同时查看实测关节状态得到的机器人 mesh 与真实点云是否重合。
同一冻结帧的检查结果：右臂 mesh 朝相机的采样点到点云最近距离中位数，
本版约 `31 mm`，上一版同时修改偏移符号时约 `108 mm`；包含遮挡和 CAD 差异，不能当作标定精度。
碗/把手约 `4.4/4.9 mm`，消歧后的门板约 `9.5 mm`。整体改善，但机器人仍有残差，不能认定完全对齐。
尚未做新的现场采集或真机动作验证。

随后按 §5 在**新目录**生成 `$PERCEPTION`，完成后按 §7 **重新定义 `COMMON` 数组**。
已展开的 Bash 数组不会因修改 `$PERCEPTION` 自动更新；不能复用旧数组、旧感知记录或旧 `$OUT`。

### 4.2 只调整 base 到棋盘格的 SDK X/Z 平移

可以保持板轴旋转、SDK Y、相机内参、机器人安装和原始测量输入不变，单独修正 SDK X/Z 平移。
本轮自动拟合各连杆给出的量不一致，尚不能确定唯一可靠修正；不把拟合结果覆盖到正式外参。
可使用下面的专用 Viser 面板在冻结数据上核对（不连接设备、不重新规划）：

```bash
# OUT 为已有日志；不要求重新跑真机。CAMERA_PARAMS 使用 §4.1 当前文件。
CANDIDATE="data/calibration/diagnostics/xz_candidate_$(date +%Y%m%d_%H%M%S)"
"$PY" "$PWD/tools/diagnostics/tune_camera_translation.py" \
  --run "$OUT" --camera-params "$CAMERA_PARAMS" \
  --output "$CANDIDATE" --host 127.0.0.1 --port 8121
```

本机建立隧道：`ssh -N -L 8121:127.0.0.1:8121 -p 23400 yuquan@58.220.16.98`，
浏览器打开 `http://127.0.0.1:8121`。
机器人 mesh 来自日志里的实测关节，保持固定。仅拖动 `SDK ΔX (mm)`、`SDK ΔZ (mm)`；
正值表示给 `T_sdk_camera` 和 `T_sdk_board` 的对应 SDK 平移分量加上该值。
同时检查腕部、上臂和固定安装部位；不能只贴合夹爪或将图像横/竖方向当作 SDK X/Z。
面板只显示机器人和原始点云，不显示旧日志里可能过时的操作物体和桌面代理。

点击“另存平移候选”才写文件，且输出目录必须不存在。保存为 `$CANDIDATE/cam_param.json`，
同步更新两份相机外参和板到 SDK 变换，原始测量值及旧文件保持不变。
修正量单独保存在 `translation_refinement_sdk_m`；重新合并标定时只应用一次，不反复累加。
候选仍需以独立 SDK 参考点或多组姿态检查，不能仅凭单帧最近点拟合当作精确标定。
确认候选后设置 `CAMERA_PARAMS="$CANDIDATE/cam_param.json"`，按 §5 生成新 PERCEPTION，再重建 COMMON 和 OUT。

## 5. 自动准备感知

将碗、微波炉和机械臂摆到本次初始位置，把门关紧；无需旋钮置零。
让相机看清碗、把手，并在已有桌面范围内采到足够的桌面深度；不要求完整桌面边界入镜。此命令只调用感知服务，不连接机械臂：

```bash
PERCEPTION=runs/realworld/perception_$(date +%Y%m%d-%H%M%S)
"$PY" tools/run/prepare_microwave_realworld.py \
  --camera-params "$CAMERA_PARAMS" \
  --output "$PERCEPTION" \
  --bowl-prompt "blue bowl" \
  --handle-prompt "pink microwave door handle"
```

prompt 按实际外观调整。非默认地址用 `--camera-endpoint`、`--seg-server`、`--pose-server` 指定。

1. 输入 `CLOSED`，确认门关紧、柜体和相机固定，不要求旋钮零位。
2. 程序采集同一帧 RGB-D，核对相机标定，只请求 SAM3 分割碗、把手两个目标。
3. 程序保留 AffordToolAct 的固定矩形 XY 范围，从本帧 RGB-D 拟合桌面 3D 平面，保留倾角；桌面上法向朝向可见相机一侧，盒体厚度沿法向向下延伸。不请求桌面分割或桌面 FoundationPose。结果在 `table_measurement.json`；若法向反指 SDK −Z，报 `camera_sdk_axis_inconsistent` 并停止，先核对标定轴向，不能翻转机器人或抬高碗绕过。
4. 打开 `masks_preview.png` 检查两个物体 mask，核对终端显示的桌面顶高、固定中心和尺寸与现场相符后输入 `ACCEPT`。若物体 mask 错误，调整 prompt；若桌面高度异常，检查深度、标定和场景位置，然后使用新目录重跑。
5. 程序卸载 SAM3，使用已有 mesh 自动注册碗和门把手两个 FoundationPose session；不导出旋钮 mesh，不注册旋钮。把手 mesh 存在局部 Z 半转的等价姿态，代码先验证实际 mesh 的对称性，再按实测桌面上方向选择柜体直立分支；原始返回保存在 `registrations/`，选择依据在 `microwave_pose_disambiguation.json`，后续跟踪采用同一归一化规则。
6. 成功后生成 `perception_setup.json`，保存感知范围、session、关门参考、桌面结果及标定和资产来源哈希，无需编辑。

准备完成后保持 Pose 服务运行。运行时读取新帧；初始关门姿态只作为几何参考，不充当当前门角。
改变标定、资产或准备输入后，需以新目录重新准备。
新版准备记录使用 `geometry_contract_version=2`；旧记录没有真实桌面法向和把手消歧来源，真机入口会明确拒绝。
旧日志仍可用只读 Viser 查看，不能手改版本号或覆盖原记录来继续运行。
切换到本流程时，按上述命令在新目录重新准备。旧桌面分割流程的记录因桌面来源已改变，需重新生成；不要手改记录或继续使用失败目录。

## 5.1 可视化区分标定、位姿和桌面问题

本检查只读取保存的真实 RGB-D、原始 pose 和配置，不连接机器人/相机/Pose 服务，
不调用 cuRobo，不启动仿真，也不修改任何标定、桌面或运行记录。
在运行真机命令的同一仓库根目录执行；本次修改位于远程 `/data1/yuquan/Gene_agent`。
沿用已有 `$PY`，需有 `numpy`、`Pillow`、`trimesh`、`plotly`；缺 Plotly 时可在显示用解释器安装，勿替换规划环境。

先使用最新运行的 `$OUT`，生成自包含交互 HTML：

```bash
CHECK=runs/diagnostics/realworld_alignment_$(date +%Y%m%d-%H%M%S)
"$PY" "$PWD/tools/diagnostics/view_realworld_alignment.py" \
  --run "$OUT" --output "$CHECK/overview"
```

`--run` 也可直接指向已搬迁的 `initial_scene/`。它使用第一帧 RGB-D 和原始 camera pose，
不混用 `PERCEPTION` 的旧参考帧。若原始帧、mesh 或 pose 缺失，会明确拒绝，而不找其他帧替代。
也可检查准备记录，或新版输入一致性检查拒绝的准备目录：

```bash
"$PY" "$PWD/tools/diagnostics/view_realworld_alignment.py" \
  --perception "$PERCEPTION" --camera-params "$CAMERA_PARAMS" \
  --output "$CHECK/preparation"
```

准备成功时直接读取 setup 的标定；准备失败时使用所给原标定文件及保存的原始注册返回。
旧准备记录未保存原始 camera pose 时只能由 SDK pose 反算，报告会标明它不能独立排除转换错误。
这时优先检查 `--run` 中的原始 tracking pose，或使用新目录重新准备。

打开生成的 `alignment.html`（浏览器可直接打开，无需联网），按以下顺序看：

1. **RGB 图**：青色 mesh/碗沿是否贴着实物，黄色箭头是否指向真实碗口。橙色为配置桌面顶面在图像上的投影。悬停可读 `u,v` 像素。
2. 点击图中工具栏的矩形绘制图标，在 RGB 中框选一块**干净、可见的桌面上表面**，排除碗、柜体、桌沿和背景。页面底部显示 `--table-roi U0 V0 U1 V1`；复制该参数加到原命令，换新输出目录。例如以下像素仅展示格式，必须替换为你框选的结果：

```bash
"$PY" "$PWD/tools/diagnostics/view_realworld_alignment.py" \
  --run "$OUT" --table-roi 120 360 480 650 --output "$CHECK/table_roi"
```

3. 重新打开 `table_roi/alignment.html`。绿色是 ROI 在**原相机坐标**独立拟合的平面，不使用外参或 SDK 水平假设；橙色是当前配置桌面。下方两张 3D 图可旋转/缩放，并用图例隐藏遮挡图层。
4. 绿色法向箭头朝向相机，选择的必须是桌面上表面。橙色箭头是配置顶面 +Z；同时比较法向和高度，不能只看两张平面是否平行。

`alignment.json` 给出相机坐标下的碗沿到实测平面的有符号距离、碗口与该平面的朝向、
实测平面在 SDK 中的法向、配置/实测法向夹角和高度差、输入帧号及来源。
`configured_normal_angle_deg` 是不区分正反的平面夹角；
`configured_top_normal_alignment` 接近 -1 则表示配置顶面方向与实测上表面相反。
**SDK +Z 不一定是真实竖直方向；输入一致性报错也不等于已经证明实物倒扣。**

仅靠碗和桌面不能唯一判定外参正确。进一步用同帧可见、已有独立 SDK 测量记录的固定标记验证标定：
重复添加 `--anchor u v sdk_x sdk_y sdk_z`，后 3 项为右臂 SDK base 中的米。
至少使用 3 个不共线且分散在工作区域的点，不用正在怀疑的碗 pose 或桌面配置当真值。
以下仅为参数格式示例，数值必须替换成真实标记像素和独立测量值：

```bash
"$PY" "$PWD/tools/diagnostics/view_realworld_alignment.py" \
  --run "$OUT" --table-roi 120 360 480 650 \
  --anchor 320 400 0.10 0.20 0.00 \
  --anchor 500 400 0.20 0.20 0.00 \
  --anchor 320 550 0.10 0.30 0.00 \
  --output "$CHECK/with_references"
```

SDK 图中的紫色线是参考点残差；JSON 保存逐点残差，单位米，不自动设定合格阈值。
参考点必须来自当前冻结画面，不能拿另一次移动后的 TCP 位置配旧图。

| 观察结果 | 优先排查 |
|---|---|
| 原相机图中 mesh 就与实物不贴合 | FoundationPose、注册 mesh frame、内参、深度/RGB 配准；先不改机器人外参 |
| 独立 SDK 参考点一致，而橙/绿桌面高度、范围或方向不一致 | 桌面高度拟合、固定 XY 范围、厚度和“SDK 中水平”的设定 |
| 多个独立 SDK 参考点存在明显且一致的残差 | 外参方向/左右臂 SDK 选择、单位、坐标变换链及 RGB-D 配准 |
| 原相机中的碗沿与实测平面也矛盾 | 感知 pose、mesh/Concept 或 ROI 选择；全局外参不能改变它们的真实相对几何 |
| 没有独立 SDK 参考点 | 只能定位矛盾所在，不能仅凭图唯一宣判“标定错”或“桌面错” |

远程查看可下载这个 HTML 单文件后打开；也可只在远程回环地址提供目录：

```bash
"$PY" -m http.server 8083 --bind 127.0.0.1 --directory "$CHECK"
```

在自己的电脑另开终端建立 SSH 隧道（按实际 SSH 配置/端口登录）：

```bash
ssh -N -L 8083:127.0.0.1:8083 58.220.16.98
```

然后打开 `http://127.0.0.1:8083/overview/alignment.html`，或对应的 `table_roi/` 页面。

## 5.2 真机统一 Viser：六视图查看初始对齐、轨迹和逐帧感知

一个入口、一个浏览器页面。默认界面提供六个按钮，切换同一 3D 画布的显示内容：

| 按钮 | 显示内容 | 背景与时间轴 |
|---|---|---|
| 1 · 初始标定 / 点云对齐 | 冻结初始 RGB-D、机器人模型（原 URDF 转换的冻结模型）、SDK/相机/物体/桌面坐标轴、原始注册碗 mesh、规划障碍；有保存 SDK FK TCP 时与模型 FK TCP 对照 | 初始感知静态结果 |
| 2 · 规划轨迹 | 所有已保存阶段的机器人关节轨迹及蓝色 TCP FK 路径，不要求存在抓碗 selection | 初始点云固定；独立规划时间轴 |
| 3 · 感知点云逐帧 | 每次采集的原始 RGB-D 点云 | 观测序号、上一帧/下一帧、播放/暂停、帧率 |
| 4 · 点云 + 夹爪状态 | 本帧两臂/夹爪 FK，开口数值与来源；有记录时显示 SDK TCP 轴 | 与视图3共用观测帧点云 |
| 5 · 点云 + 物体 axis / bbox | 当帧原始 FoundationPose 估计轴及注册 mesh 局部包围盒，经估计 pose 转到 world；右侧 RGB 叠加当帧已保存的 SAM3 mask | 与视图3共用观测帧点云 |
| 6 · 点云 + 夹爪 + 物体 | 同时显示4和5 | 与视图3共用观测帧点云 |

### 一条命令启动

在 **58.220.16.98** 上，从项目根目录执行：

```bash
cd /data1/yuquan/Gene_agent
/data1/yuquan/miniconda3/envs/gene_agent/bin/python tools/viewers/view_realworld_scene.py \
  --run runs/realworld/你的运行目录 --host 127.0.0.1 --port 8120
```

`--run` 指向具体一轮运行，包含 `initial_scene/manifest.json` 和 `observations/`。例如现有只规划记录为 `runs/realworld/task1_plan_20261124_200548`；该记录只有一帧且尚未生成轨迹，因此可以检查对齐，无法凭空播放后续动作。

运行尚未结束时，在另一个终端启动同一条命令并增加 `--follow`：

```bash
/data1/yuquan/miniconda3/envs/gene_agent/bin/python tools/viewers/view_realworld_scene.py \
  --run runs/realworld/正在运行的目录 --follow --host 127.0.0.1 --port 8120
```

`--follow` 每两秒刷新新增的完整 RGB-D、观测状态及报告；滑块可查看已落盘帧。它不额外请求相机、SAM3、FoundationPose 或机器人，不改变任务的采集频率。需要 `initial_scene` 已保存。默认每帧最多显示60000点，可用 `--max-points 100000` 调整；仅减少显示采样，不改变深度或原日志。

在自己电脑另开终端，转发端口：

```bash
ssh -N -L 8120:127.0.0.1:8120 -p 23400 yuquan@58.220.16.98
```

浏览器打开 **http://127.0.0.1:8120**。服务器上的显示命令保持运行，`Ctrl+C` 退出。若在真机主机本地启动，只需浏览器打开对应本地端口。

也支持 `--run /实际路径/initial_scene`、`manifest.json` 或含唯一 `initial_scene` 的完整日志 `.tar.gz`。完整日志压缩包会把该轮的观测和报告一起安全提取到临时目录，退出时清理。只有 `initial_scene` 的包只提供静态部分，界面明确报告缺少动态帧；需要多帧播放时一并复制 `observations/` 和阶段报告。显示解释器需要 `viser`、`mujoco`、`numpy`、`scipy`、`trimesh`、`Pillow`。

### 初始标定与规划核对

先看视图1的活动臂、frame 和 snapshot，确认运行身份；切换图层可分别检查机器人、点云、原始注册 mesh、配置桌面和规划障碍。所有图层共用保存的 world：`T_world_camera = T_world_sdk × T_sdk_camera`，不再额外乘一次机器人 base 变换。X红、Y绿、Z蓝；浏览器上方向设置仅改变观察方式。

有保存 `tcp_poses_world` 时，同时画 SDK FK TCP 与冻结模型 TCP，并报告平移/转角差；旧日志缺少 SDK TCP 时直接标缺失，不用当前 SDK 补算成历史证据。青色原始注册碗 mesh 使用原 FoundationPose 相机 pose；可对照点云及任务实际消费的碗 pose，检查是否有重复坐标变换。配置桌面顶面为橙色，需要独立 ROI 平面拟合时继续使用 §5.1 的 HTML 检查。

视图2选择“轨迹段”，拖动“轨迹采样帧”或播放。蓝线是已保存关节轨迹的 TCP FK；它是规划预览，不是实际连续执行轨迹。初始物体与障碍保持初始状态，动态碰撞球不冒充逐帧验证。没有保存轨迹则显示“无保存轨迹”。

原抓碗候选、首个拒绝阶段、底层失败原因及桌面抬升修正仍可在旧版专用界面检查：

```bash
/data1/yuquan/miniconda3/envs/gene_agent/bin/python tools/viewers/view_realworld_scene.py \
  --run runs/realworld/你的运行目录 --legacy-view --host 127.0.0.1 --port 8120
```

旧版仍支持 `--selection /实际路径/selection.json` 和 `--no-pick-debug`；selection 与初始快照不匹配时会提示，不能当作同一轮规划证据。

### 点云、夹爪与物体逐帧播放

视图3～6共用一个观测序号。每帧展示 frame、采集 Unix 时间和相对目录；支持上一帧、下一帧、播放/暂停和帧率调整。各帧使用同一保存的 world 和各自记录的相机标定；不做 ICP、自动居中或重新拟合外参。如果运行中的 `T_world_sdk` 改变，会标明无法统一显示，不悄悄对齐。

勾选“叠加第一帧点云”后，第一帧以青色固定显示，当前帧保持RGB颜色。相机内外参和分辨率一致时，同时显示相同采样像素的深度差中位数和绝对差P95。运动物体、遮挡、反光也会造成差异，因此这些数值用于辅助检查，不能单独判定深度传感器漂移；优先观察静止桌面/柜体/背景区域。

夹爪显示**程序消费的开口**和保存关节做出的 FK，不额外读取夹爪。`sdk_live_read` 表示本次直接读取，`confirmed_execution_endpoint` 表示已确认动作终态的复用，`retained_initial_observation` 表示首次观测复用。来源未知的旧日志明确注明。未测量期间保持离散状态，不插值成实际运动；开口变化影响模型两侧指爪间距。

新日志的 SDK 原始反馈还保存检测时间。在视图4或6勾选“播放夹爪原始检测样本”，可用“夹爪检测样本”滑块和播放按钮逐个检查命令完成确认期间的开口。背景用该检测之前最近一帧点云，显示时间差；初始化早于第一帧时显示首个后续点云并注明不同步。旧日志没有检测时间戳时禁用此功能，仍能按观测帧播放程序使用的开口。此功能只补充记录已有读取的时间，不增加 SDK 查询。

物体使用 `tracked/objs/.../pose.json` 中原始 `T_camera_object`，通过 `T_world_camera` 放到同一 world，显示 axis+bbox；不把门角约束、Concept 修正或沿用状态重新标为当前 FoundationPose。bbox 使用原注册 mesh 的局部范围；资产缺失时只画 axis，不造包围盒。当帧进行了 SAM3 分割/注册且保存 mask 时，RGB 显示分割叠加；纯 tracking 帧没有新 mask，会显示数量0，不贴旧帧 mask。未刷新对象的来源和原观测帧在右侧列出。

新运行每次 capture 额外保存 `viewer_context.json` 坐标链，后续感知失败也能检查原始点云；缺少关节/位姿时只显示已有部分，不拿上一帧数据冒充本帧。旧日志缺少原始RGB-D、外参或位姿时会提示具体缺失。

仅核对加载数据、不启动服务：

```bash
/data1/yuquan/miniconda3/envs/gene_agent/bin/python tools/viewers/view_realworld_scene.py \
  --run runs/realworld/你的运行目录 --check-only
```

输出包含静态快照身份、观测帧数量、轨迹数量、带时间戳夹爪反馈数量和缺失提示。查看器完全只读，不连接设备、不重新运行推理或规划，不修改历史运行记录。

## 5.3 模型翻转与桌面穿插的判别

`task1_plan_20261124_200548` 的原始 RGB-D 分析表明：桌面上法向在该标定输出中约为
`[0.0360, -0.0868, -0.9956]`，指向 SDK −Z；碗口与它的点积约 `0.99987`。
因此不能把这轮画面中的碗称为真实倒扣。原单点标定把 `T_sdk_board` 的旋转设为 I，
没有测量板朝向；若 PnP 板轴与 SDK 板轴约定不同，点云会整体反向。
该轮机器人安装与 AffordToolAct 相同，原关节的 URDF FK 与保存的右臂 SDK TCP 误差约
`0.0022 mm / 0.0116°`；不应通过旋转机器人来迎合错误外参。

微波炉另有独立歧义：注册把手 mesh 绕其中心局部 Z 转 180° 后完全重合，
不能从把手完整 pose 唯一确定整台柜体。新版使用实测桌面法向选等价分支，保持把手位置和几何不变，
同时纠正柜体方向；动态门测角继续使用固定柜体和把手中心圆轨道。
桌面原先强制水平，在该轮约 `5.39°` 的 SDK 平面倾角下，碗附近偏差约 `22 mm`；
新版构造保留实测平面，候选净空也使用各候选 XY 处的顶面高度。
这些数值是冻结帧几何诊断，不是真机动作验证。

修正标定时，需独立确认 PnP 板 +X/+Y/+Z 与 SDK 的对应；优先用多个不共线的已知 SDK 点确定板姿态，
不能只用一个点或 `RᵀR=I` 判断正确，也不能简单给旧外参乘一个未核对的 180° 旋转。
本轮用户已确认的轴关系修正版见 §4.1；使用修正版后，重新执行 §5 的 prepare 命令生成新 `$PERCEPTION`，再运行原只规划命令与 §5.2 的只读查看器。
历史附件不被重写，新旧快照分别保留。

## 6. 配置和状态来源

| 参数或状态 | 来源 |
|---|---|
| 两臂 IP、端口 | 命令行；端口默认 8080 |
| 相机外参、内参、序列号 | 已有标定输出 |
| 碗、把手 session | 准备命令自动注册，两路跟踪 |
| 门角 | 当前帧把手 pose 相对固定门轴的圆周投影 |
| 旋钮 | 不感知、不操作；碰撞模型保留资产零角的静态代理，标为未观测，不代表实际旋钮角 |
| 柜体 pose | 关闭参考与资产 frame 配对；每帧核对把手中心的门轴圆轨道一致性 |
| 桌面高度 | 同帧深度经标定转换到右臂 SDK base，在固定矩形范围内计算最密集水平高度带 |
| 桌面 XY、厚度 | 已迁移的 AffordToolAct Task1 桌面几何，自动读取；不依赖桌面 mask 或可见边界 |
| 资产 frame、关节轴、限位 | 已有场景配置和 URDF，自动读取 |
| TCP、开口范围 | 标准夹爪 profile，并与规划模型校验 |
| 操作 Skill 参数 | 原有 profile；现场误差检查由配置层统一装配 |

柜体参考依赖“柜体固定”的前提，几何一致性检查不能替代独立柜体测量。
门角与 AffordToolAct 一样由把手中心的圆周投影计算；中心偏离预期圆轨道超过 `25 mm` 时拒绝该帧。
把手 mesh 的完整朝向差异只作诊断，不再单独触发“柜体移动”拒绝，也不会用它更新柜体朝向。
每次测角通过或圆轨道校验失败，均自动保存 `observations/<观测序号>/door_handle_diagnostics.json`，
其中包含中心误差、半径差、轴向差、朝向差及参考/当前 pose；这是运行输出，不需手填。
桌面默认中心 XY 为 `[0.1, 0.3] m`、尺寸为 `0.6 × 0.6 m`、碰撞厚度为 `0.02 m`，位于右臂 SDK base；来自已有 `data/task_1/knowledge/legacy_extract_visual_table.json` 的 Task1 几何，不需另填文件。
只复用该文件的 XY 和厚度，旧 Z 高度不参与计算；本次桌面顶高来自深度，碰撞盒中心放在顶高下方 `0.01 m`。
这与 AffordToolAct 的微波炉桌面处理一致：假设桌面水平，范围固定，高度由场景点云计算。桌面 XY、厚度和微波炉支撑代理均不是本次自动测得的尺寸。
plan-only 时核对碰撞模型与现场一致，更换支撑尺寸时不能直接沿用。

角度和 pose 使用同一采集帧的真实时间戳，不会再因忘记更新门角 JSON 而过期。
但采集、tracking 或场景导出超过 `--max-age-s`（默认 5 秒）时，仍会拒绝过期观测；
应检查服务延迟或相机状态，不能伪造新时间戳。默认 `no_knob` 只关闭旋钮感知。物体按下文“关键帧感知与遮挡状态估计”调度。热会话复用 tracking；冷注册若已耗尽帧有效期，先保存注册证据，再重新采帧、跟踪和同步采样 SDK，最多重试一次，旧帧不交给执行器；不改写采集时间。

## 7. 配置检查、首段预检与完整规划预览

先在本终端设置前文的 `PY`、`PERCEPTION`、`LEFT_IP`、`RIGHT_IP`，并确认位于工程根目录：

```bash
COMMON=(--perception "$PERCEPTION" --left-ip "$LEFT_IP" --right-ip "$RIGHT_IP")
"$PY" "$PWD/tools/run/run_microwave_task1_realworld.py" \
  "${COMMON[@]}" --subtask all --collision-mode full --check-config
```

`--check-config` 只检查本地输入和配置，不连接设备，也不验证远端 session 存活。
正式 CLI 已不接收 `--config realworld_site.json`。

### 7.1 单个子任务首段预检

不加 `--execute` 或 `--plan-only` 时，只预检首个状态改变阶段：

```bash
OUT="runs/realworld/pick_bowl_precheck_$(date +%Y%m%d_%H%M%S)"
"$PY" "$PWD/tools/run/run_microwave_task1_realworld.py" \
  "${COMMON[@]}" --subtask pick_bowl --collision-mode full --output "$OUT"
```

终端会按 AffordToolAct 粒度打印：运行模式和输出目录、冻结观测 frame/snapshot/障碍数、
候选生成数量、主要失败原因、终态阶段、轨迹数量、
证据文件路径及可直接复制的 Viser 命令。最后一行仍保留机器可读 JSON，便于脚本解析。

首段通过后返回 `validated_prefix`，不发送运动命令。
仅使用 `--subtask all` 而不加执行或完整预览参数时，也只预检完整任务的首段，
不表示已规划整个抓碗子任务或整条 Task。

真机抓碗候选对齐 AffordToolAct：160个候选，SDK -X优先、-Y次优先，只排序不按仿真的
近基座侧、外侧接近、碗沿接触或45–70°桌面夹角条件硬删候选；仿真筛选保持不变。
真机使用 `data/task_1/realworld/pick_bowl.json` 中的4档深度、5档倾角和SDK Z偏移 -15 mm，
TCP限高至实测桌面上方50–70 mm后再派生pregrasp。IK、碰撞、关节行程和发令前复核仍保留。
若之后无可行解，应区分候选生成、IK、碰撞和轨迹阶段；生成160个候选不代表实际抓取已经成功。

### 抓碗筛选的停止条件与请求复用

真机 `pick_bowl` 默认使用 `first_successful_batch`：最终 grasp IK 与 pregrasp IK 都按最多 128 个逻辑请求分批求解，发现首个包含可用候选的成功批次就停止后续批次。pregrasp 的关节行程门槛继续有效；整批没有通过现有门槛的候选才尝试下一批。后续 pregrasp 请求按需生成，不提前穷举。已命中去重缓存的请求不会再次进入 cuRobo。

截止后仅在已经检查的候选中做排序和轨迹规划，pregrasp、原位对齐、直线 approach、碰撞和预测 lift 仍须通过原有检查，才可交给执行流程。若首个可用批次内不能生成合格前缀，正常失败，不继续求解后续 IK 批次，也不另启水平入口搜索。只有原批筛完全没有可用 pregrasp 时才保留既有水平 fallback；fallback 复用原最终 IK 证据。是否真机发令仍由 `--execute --yes` 控制。

4 档深度（15、25、28、30 mm）与 8 方位、5 倾角仍生成 160 个原始候选，作为备选范围。找到成功批次后未检查的候选会记为未到达，不记作失败，也不宣称已全部求解。

运行命令不变，使用新的 `OUT`。`selection.json` 中 `pregrasp_search`、`selection_stop_condition`、`candidate_ranking_scope` 和 `ik_batch_screening` 记录实际策略、检查数、剩余数/提前截止及批次数；`pregrasp_search_summary` 记录实际后端计算/复用数。完整相同上下文的成功和失败结果仍可复用，复用耗时为 0，原始求解耗时另行保留。

### 7.2 完整 Task1 plan_only 与预览视频

实际执行之前，使用以下命令连续规划整个任务并导出视频：

```bash
OUT="runs/realworld/task1_plan_$(date +%Y%m%d_%H%M%S)"

MUJOCO_GL=egl "$PY" "$PWD/tools/run/run_microwave_task1_realworld.py" \
  --perception "$PERCEPTION" \
  --left-ip "$LEFT_IP" --right-ip "$RIGHT_IP" \
  --subtask all --plan-only --video \
  --output "$OUT"
```

`--plan-only` 兼容写法 `--plan_only`，当前只用于 `--subtask all`，
与 `--execute`、`--prepare-held-bowl` 互斥。
碰撞模式默认 `full`，可显式追加 `--collision-mode full`；
两臂端口默认 8080，非默认端口追加 `--left-port`、`--right-port`。
输出目录须为空，每次重跑使用新目录。

该模式开始时取得一次新的现场感知及双臂、夹爪状态，
不设置夹爪行程，不发送机械臂或夹爪命令。
后续在私有运动学影子中，按 cuRobo 返回的真实规划轨迹末帧衔接下一段，
更新预测门角、夹爪开口、碗位姿和规划附着，并重新导出阶段碰撞场景；
不再采集新相机帧或将预测状态写成 SDK 实测。

完整 `plan_only` 预览会读取本次 `PERCEPTION/capture/rgb.png`、`depth.png`、
`camera.json` 与 `perception_setup.json` 的相机标定，重建带原始 RGB 颜色的米制点云。
坐标只按 `T_world_camera = T_world_sdk @ T_sdk_camera` 转换一次，
不使用历史仿真相机位姿、名义桌面中心或名义物体摆位替代本次观测。
桌面的中心、倾角和可见范围由点云平面拟合；厚度沿用已确认值，遮挡和视野外边界仍属未知。
碗、微波炉机身、门板和把手显示读取 `data/task_1/objs/.../meshes` 下的实际 mesh，
按本次感知 pose 与原资产 frame 对齐，门板和把手随原门轴运动，不重新分割或生成 mesh。

预览的桌面碰撞与拟合位姿同步；其他机器人、碗、微波炉及支撑的碰撞保护代理保留，
RGB-D 背景表面本身不作为已验收的完整碰撞模型。
以上重建用于完整 `plan_only`；真实执行仍沿原现场逐段观测、规划和验收流程，
不能把预览几何或预测轨迹直接作为已验收的执行输入。
背景隐藏原位置的待操作物体及机器人几何区域，避免运动回放重影；
深度缺失和遮挡处保留空洞，不跨越深度跳变补面。

默认视频第一视角及三维回放初始视角使用本次实测相机，第二视角保留夹爪侧视图。
若点云桌面朝上法向不符合当前规划世界的 `+Z` 约定，
程序保存重建场景并报告 `observed_world_up_mismatch`，可先打开三维回放检查；
应核对相机外参和 SDK 坐标，程序不自动翻转、平移点云来消除差异。

预览包含两轮开关门、抓碗、放入并释放撤离、取碗、放回桌面并释放撤离，
以及全部默认 TCP 过渡；默认跳过旋钮。
沿用真机各阶段的目标、候选和碰撞策略，原有接触规划层的停用障碍保留在记录中。

`--video` 可选，生成 `"$OUT/plan_preview.mp4"`；省略时仍保存规划及回放数据。
视频导出需要规划环境具备可用的 MuJoCo EGL 和 FFmpeg。
查看预览时核对桌面、支撑、门轴、尺度、机械臂、TCP 和接近/撤离路径与现场是否一致。

| 输出或字段 | 含义 |
|---|---|
| `task_result.json` | 全部步骤的状态、完成数量及首个失败原因 |
| `status=planned` 且 `planning_success=true` | 所有选中步骤完成预测规划 |
| `status=planning_failed`、`first_failure` | 规划在该处停止；保留成功前缀和原规划器证据，后续步骤未规划 |
| `plan_preview.json` | 真实规划轨迹及预测状态采样，用于三维回放和重新导出视频 |
| `plan_preview.mp4` | 添加 `--video` 后导出的规划预览；失败时可能只有前缀 |
| `assumptions` | 连续预览所采用的夹持、门/旋钮随动、释放后保持位姿等假设 |

完整预览始终记录 `execute=false`、`task_success=null`。
规划成功不证明真实夹持、接触、释放落稳或真机任务成功；
实际执行仍须逐段重观测、规划和检查后置条件。
失败后根据 `first_failure` 及对应碰撞/轨迹记录处理问题，再以新目录重跑。
首阶段前失败时可能尚无可回放的轨迹；先查看报告，视频导出失败另存 `video_result.json`。

若以后启用旋钮，准备感知与运行时均须加 `--with-knob`；
若 profile 含未经过轨迹规划的直接末关节命令，
预览会报告 `unplanned_direct_joint_command` 并停止，不能算完整规划通过。

### 7.3 浏览器三维回放

保留上一步的 `OUT`，运行：

```bash
"$PY" "$PWD/tools/viewers/view_skill_run.py" \
  --run "$OUT" --mode planned \
  --host 0.0.0.0 --port 8092
```

浏览器打开 `http://<运行机器IP>:8092`，可选择阶段、拖动时间轴、播放/暂停和显示规划碰撞层。
若使用另一终端，先把 `OUT` 设置为本次规划输出目录。
回放只读取已有记录，不重新规划或控制真机。

### 7.4 从已有规划记录重新导出视频

无需重新采集、规划或连接机器人：

```bash
MUJOCO_GL=egl "$PY" "$PWD/tools/viewers/recording/record_skill_run_video.py" \
  --run "$OUT" --mode planned \
  --output "$OUT/plan_preview.mp4" \
  --camera observed --camera gripper_side --profile microwave_task1
```

`observed` 使用点云重建保存的实测相机；旧版未重建点云的预览记录仍使用 `--camera task`。
`--mode planned` 表示规划运动学预览。失败前缀也可导出，不能据此认为整条任务已通过。

## 8. 真机执行

所有带 `--execute --yes` 的任务入口，统一先将右夹爪设为 SDK 开口值 **800**，阻塞命令返回后直接构建现场观测并进入任务，不再追加actpos读取或800±10到位复核。该动作不依赖发令前的夹爪开口读取，每次命令启动只执行一次；`all`、`load_bowl`、`unload_bowl` 内部子阶段不重复重置。SDK命令失败立即停止；命令成功不代表已测量到位，记录见 `right_gripper_preparation.json`。仅预检或 `--plan-only` 不发送该命令。独立持物任务也先初始化，持物确认以初始化后的现场为准。


现场确认急停、通信、初态和完整碰撞模型后，先测试单个子任务：

```bash
OUT=runs/realworld/pick_bowl_exec_$(date +%Y%m%d-%H%M%S)
"$PY" tools/run/run_microwave_task1_realworld.py \
  "${COMMON[@]}" --subtask pick_bowl --collision-mode full --output "$OUT" \
  --execute --yes
```

人工确认时 `y/yes` 表示通过，`n/no` 表示失败；不要把未观察到的状态确认通过。
独立 Insert 或 Table Place 需要确认现场已夹持的碗，程序自动生成持物记录：

```bash
OUT=runs/realworld/insert_bowl_$(date +%Y%m%d-%H%M%S)
"$PY" tools/run/run_microwave_task1_realworld.py \
  "${COMMON[@]}" --subtask insert_bowl --prepare-held-bowl \
  --collision-mode full --output "$OUT" --execute --yes
```

独立桌面放碗用 `--subtask place_bowl_on_table --prepare-held-bowl`。
`load_bowl`、`unload_bowl` 组合从上一阶段传递持物状态，不需要另填记录。
先按第 7 节检查完整任务的规划效果，各子任务通过现场验收后，再测试完整任务：

```bash
OUT=runs/realworld/task1_no_knob_$(date +%Y%m%d-%H%M%S)
"$PY" tools/run/run_microwave_task1_realworld.py \
  "${COMMON[@]}" --subtask all --collision-mode full --output "$OUT" \
  --execute --yes
```

省略旋钮参数或显式加 `--no-knob` 的效果相同，单个子任务也使用该默认配置。
默认模式会直接拒绝 `--subtask rotate_knob`。现有旋钮实现保留为可选能力；以后需要启用时，必须在准备感知和运行命令中都显式加 `--with-knob`，并另行完成旋钮接触验收，不属于本文当前流程。
结束或中断后按实物状态检查和复位，不能假定机器人已经回 home。

## 9. 中断与恢复

| 现象 | 处理 |
|---|---|
| 碗或把手 mask 错误 | 调整物体视角或 prompt，以新目录重新准备 |
| 固定桌面范围内有效深度点太少、计算高度异常 | 检查深度图、标定和场景位置；无需桌面 mask，不手填高度 |
| 相机序列号、内参或标定不一致 | 修正相机配置或重新标定，再准备感知 |
| tracking session 不存在 | 检查 Pose 服务是否重启，重新准备感知 |
| 把手中心偏离固定门轴圆轨道 | 查看报错的毫米误差和同帧 `door_handle_diagnostics.json`，结合 `capture/`、`tracked/` 检查关门参考、mask、标定和 tracking；若柜体或相机已移动，关门后用新目录重新准备，不手改角度或阈值 |
| 门角或观测过期 | 检查采集与计算延迟，取得新帧，不修改记录时间 |
| 阶段失败或夹持不可靠 | 停止后按实物状态复位，保留记录 |

`Ctrl+C` 不能替代机械急停。异常时按现场停机流程处理，确认机械臂和夹爪状态后再恢复。

## 10. 运行证据

### 请求与碰撞资产复用

真机当前 `first_successful_batch` 筛选启用 `deduplicate_pregrasp=True`，保留已检查批次内部的关节排序、显式关节目标、原位对齐和 lift 可行性检查。pregrasp IK/路径请求只在全部求解字段相同（追踪 ID 除外）时复用；成功和失败均缓存，水平入口 fallback 共享本次缓存，复用耗时为 0。新现场、起点、目标、seed、策略、关节目标变化会形成不同请求。运行命令不变。

collision mesh 在本任务内以内容地址复用 OBJ；strict/contact/预测场景分别保存自己的位姿，不重复导出同一局部几何。文件哈希结合 dev/inode/size/mtime_ns/ctime_ns 与 inode 变更事件核对；同大小改写、恢复 mtime、替换和缺失都会使复用失效。监测不可用（例如 inotify 配额耗尽）时安全重读，不只靠时间戳命中，不要求修改主机参数；重读确认内容相同后仍跳过重复 OBJ 序列化和写盘。读取中变化会拒绝复用。所有缓存随任务释放。baseline 入口也有任务内资产/哈希缓存。

共享同一 `world.data` 的多个 solver 只做一次同步，未变化的 cuboid/mesh 子部分不重复加载，各 wrapper 及 graph 缓冲仍更新。独立 world 保持独立同步；真实场景变化后的实测重规划和完整检查继续执行。

`selection.json` 的 `deduplicate_pregrasp` / `pregrasp_search_summary` 可核对请求复用；`planning_optimization.json`（single_reuse 模式）的 `mesh_export_cache_hits/misses`、`mesh_file_hash_cache`、`world_sync_requests` / `world_data_syncs` 可核对重复工作是否减少。剩余的 `Mesh already in cache` 提示可能来自不同独立 world 的有效复用，不能当作重新构建 mesh 或求解次数。

### 时间统计（schema_version=2）

原命令不变，使用新的 `OUT`。`runtime_timing.json` 的 `total_duration_s` 使用统一单调墙钟，从脚本首条可测语句开始，覆盖导入/配置、设备连接、感知/场景、候选与路径规划、执行与人工等待、诊断写盘、资源清理、可选视频和最终终端输出。Python 启动到首条计时语句之前、计时报告自身最终写盘及进程退出不在此范围。终端 JSON 是最终输出前的快照，文件为输出完成后的正式终态。

| 字段 | 含义与核对方式 |
|---|---|
| `accounting.exclusive_components_s` | 互斥组件墙钟；各项加总等于 `total_duration_s`，用于核对耗时去向 |
| `accounting.exclusive_stages_s` | 互斥阶段墙钟；各项加总也等于总时间，`outside_stages` 为启动、清理等动作阶段外时间 |
| `accounting.unattributed_s` | 未被任何计时区间覆盖的真实时间，显式列出，不隐去或冒称 solver 时间 |
| `accounting.reconciliation_error_s` | 互斥分项与总时间的核算差，应为 0 |
| `stages[]` | 每次阶段的起止偏移、状态、包含子步骤的时间和独占时间；重复阶段保留独立身份 |
| `components.*.total_duration_s` | 同组件实测区间的并集，避免 adapter/StageMotion 嵌套重复；不同组件间仍可能包含关系，不能直接跨组件相加 |
| `components.*.reported_duration_s` | 没有起止位置的后端报告耗时，仅供参考，不参与墙钟核算 |
| `events[]` | 含父子身份、起止偏移、组件和阶段关联的时间线，用于定位具体慢调用 |

`MotionResult.duration_s` 仍表示后端方法耗时，批 IK 通常是批耗时均摊；不是完整阶段时间。其 `metrics.runtime_timing_event_id` 指向完整 adapter 调用区间，同一批不能逐项重复累加该区间。SDK 的 `execution_duration_s` 是原驱动窗口；`stage_wall_duration_s` 才是相应完整阶段调用（包括前后检查/观测）的墙钟时间。selection 的完整阶段时间通过 `runtime_timing_event_id` 查询，避免漏掉输出成本或重复嵌入计时详情。


真机入口默认按**操作终态**输出 cuRobo 详情：候选筛选中部分 IK 失败但最终找到轨迹，不触发详细输出；感知、夹爪、SDK 执行或独立姿态/接触检查失败也不触发。只有操作最终因 cuRobo IK/路径求解失败而停止，才写详细求解证据。运行命令不变。

- `diagnostic_manifest.json` 的 `curobo_logging` 保存策略、总批次/请求数、候选失败数和详情是否输出；没有求解详情不表示没有做规划。
- 普通 `selection.json` 保留场景锚点、选中候选及必要阶段轨迹，`plan.attempts` 不再反复嵌入；`candidate_selection.json` 仅引用它。候选清单保留状态与简短原因，成功前缀不展开全部求解明细。
- cuRobo 终态失败时，`initial_scene/motion_requests/` 保存原批次的全部输入目标和一次公共上下文；成功项不额外写结果，失败项的唯一完整结果保存在 `motion_results/`。`events.jsonl` 保留原批次大小/索引及结果引用，`diagnostics/` 仅放摘要与引用。初始场景归档直接使用这份批次输入，不再复制另一份。
- `failure_analysis.json` 引用事件和候选索引，Markdown 按阶段/原因分组计数；不重复嵌入全部候选和原始诊断。终端只显示终态失败的汇总，逐请求细节通过索引查看。
- 原始场景与执行证据仍保留。详细记录失败会写入 `diagnostic_errors.json`，不会改变原规划/操作结果；旧日志的读取方式继续兼容。


- 感知目录：原始 RGB-D、两个物体 mask、桌面高度计算及固定几何来源、关门参考、两个 session 和来源哈希。
- `runtime_input.json`：自动装配的有效配置、输入来源、profile 和运行参数。
- `observations/`：碗与把手的新帧 tracking、门角、场景与碰撞导出。旋钮仅记录静态碰撞代理及未观测标记，没有实测角或观测时间戳。
- `observed_geometry/reconstruction.json`：冻结帧、相机变换、mesh 路径及哈希、桌面拟合质量与坐标检查；`observed_pointcloud.npz` 保存实测点和原始颜色，`scene/` 保存重建模型与纹理。
- 完整 `plan_only` 目录：`task_result.json`、`plan_preview.json`、各阶段规划与碰撞快照，以及可选的 `plan_preview.mp4`；视频导出失败时另存 `video_result.json`。预测状态及其假设与真机执行证据分别记录。
- 各阶段计划、SDK 执行结果、人工确认和异常 traceback。

反馈问题时提供实际命令、解释器和上述输出目录，不要只提交最后一行异常。


### 候选筛选与 AffordToolAct 对齐（2026-11-26）

真机先生成8方位×5倾角×4深度=160个Bowl候选，按SDK -X优先、-Y其次排序；保留AffordToolAct的桌面净空50～70mm限高。候选进入规划前不再调用仿真base_side_bowl_proposals，不以45～70度桌面夹角、近基座侧、仿真接触支点半径/高度/切向容差或双指跨沿条件硬删候选。仿真自己的筛选不变。

这只对齐候选输入；后续仍检查最终抓取IK、pregrasp IK与关节状态排序、碰撞路径、接近段以及抬升可行性。碰撞模式沿用原运行配置，未改变任何碰撞开关。若仍无可执行候选，查看candidate_selection.json与candidate_diagnostics.json区分生成数、几何过滤后数量及规划阶段失败，不能把生成有160候选当作抓取成功。

离线同一历史有效场景：旧筛选160→0，其中112个候选角度在45～70度之外；新入口160→160。该证据不代表本次尚未同步的现场失败已完成真机回归。


## 2026-11-26 性能优化

P0/P1/P2 已接入真机入口，原运行命令不变。批次容量、精确规划复用、相机连接复用、并行状态采集及检查结果见 [性能优化与验证](performance_optimization.md)。


### 所有 sub-task 的夹爪终态交接

`all` 与独立子任务统一使用公共真机 Runtime，按 AffordToolAct 的命令时序更新夹爪：每条命令仅下发一次。位置开爪等待SDK阻塞返回；RealMan力控抓取等待命令接收确认（非阻塞、timeout=3），由下述实测运动状态决定完成，避免控制器阻塞到位事件等待超时误报。力控抓取保留配置的稳定期（默认1.5秒），随后读取 `rm_get_gripper_state`；若 mode=4/5 仍在运动，则按50ms间隔等待，普通开爪/位置控制默认上限2秒；所有力控夹取（command_mode=grasp，包括开门抓把手、Pick/Extract抓碗、可选旋钮夹取）统一默认上限20秒，以覆盖闭合、接触阻力及力控建力。该20秒从稳定等待（默认1.5秒）结束后计时，检测到完成即继续，不固定等满20秒。SDK命令只下发一次，仍须等到完成状态才进入后续动作。只有反馈运动完成后，才核对位置目标、写入 `measured_end_width_m` 并缓存终态；不要求连续三次开口稳定。运动中的读数不会因目标误差大而提前失败。等待超时、SDK失败、无效反馈或停止后超出原到位容差仍会停止阶段。力控开口以实际反馈为准，不等于规划闭合宽度。日志 `endpoint_confirmation` 保存首尾读数、运动完成标志、轮询次数和等待耗时，避免重复日志膨胀。右夹爪800初始化仅等待阻塞命令返回，不做额外反馈检查；后续首次场景读取实际开口。初始化报告中status=command_completed、endpoint_verified=false，不会将800作为已确认实测终态。

夹爪 TCP 位姿继续由实际机械臂关节和 FK 更新；开口仅在初次观测以及新夹爪命令后读取。两条命令之间复用实测终态，来源为 `confirmed_execution_endpoint`；未发令的另一侧保留首次实测值，标为 `retained_initial_observation`。新命令前撤销旧终态，失败后不沿用；重连重新读取。保留命令起点核对、关节连续性及物体操作后置检查。

### 800初始化后的任务开口与990余量

初始化800不是所有动作通用的工作开口。Pick/Extract 随后设置约513（0.041m），用于既有候选的接近规划；抓把手和旋钮的开爪、以及抓取后的释放均与后续动作有关，保留这些阶段。关门起始需要闭爪推门，放碗/桌面Place在释放前保持夹持；没有额外无关的满开重置。

所有 RealMan 位置开爪命令最大为SDK **990**，满量程换算仍为1000对应0.08m，所以990对应 **0.0792m**；800及中间开口的换算不变。释放操作使用阻塞位置开爪到990，替代没有位置目标、会直达满开的SDK release接口。门/旋钮及放置的真机目标同步使用有限开口，共享仿真配置不变。阶段日志 `hardware_command.target_width_m` 记录实际命令目标，`measured_end_width_m` 记录反馈；不会用命令目标覆盖真实actpos，也不改变现有反馈有效范围。此余量用于减少端点过冲，仍需真机确认实际反馈。

### 闭门接触后的柜体小平移

小平移检查的默认轨道误差门限为25mm。显式取得闭门接触关键帧时，仍可使用累计30mm以内的平移约束；不从把手朝向推断机身转动。普通闭爪后为避免遮挡不再强制重测，门运动过程中不重新设闭门零位。相机、柜体被外部移动后须重新准备场景。

### 关键帧感知与遮挡状态估计（2026-11-28）

Gene_Agent_new仿真没有固定“感知Hz”：候选/规划检查、执行前后和后置检查直接读GT，多个读取之间可能没有物体运动。这些读取不能逐次接成SAM3分割和FoundationPose注册。已有日志的单次视觉处理约4.7～5.8秒（约0.17～0.21次/秒，仅代表该两轮耗时，非硬件极限）。

默认真机命令不变，新建OUT后正常运行。实际物体刷新规则：

| 位置 | 碗/门状态来源 |
| --- | --- |
| 首次使用可见物体 | 实际RGB-D感知；门操作进入时重新测门 |
| Pick接近、闭爪、重复持碗检查 | 复用碗位姿；闭爪完成后建立假定夹持关系 |
| Lift完成并稳定后 | 实测碗；与当前SDK TCP及此前夹持关系交叉检查 |
| 持碗转向、preplace、insert | 当前SDK TCP × 已保存TCP→碗关系；明确标注运动学估计 |
| 碗在炉内、释放、炉内落稳、插入后撤离 | 保存释放时估计位置，不因相机遮挡强制重分割；不是接触/落稳实测 |
| Extract炉内抓取、开始撤离 | 上次放入的释放位置和新的假定夹持关系 |
| post_extract_2完全退出炉腔 | 实测碗并校验；随后继续桌面放置 |
| 桌面释放、落稳 | 先保存释放位置；退开retreat后再实测碗是否留在提交位置 |
| 门pull/swing/push中间路点 | 已成功执行阶段的角度假设，来源不是observed_door_angle |
| 门Skill最终撤离 | 等2秒稳定后实际测门；相邻post_contact检查复用这次结果，避免重复注册 |

相机帧和机械臂SDK状态仍按阶段读取，夹爪终点反馈保留既有完成确认。只有成功执行且确认完成的SDK命令才提交夹持/释放事件；仅预检、规划失败或执行失败不能创建成功的夹持估计。关键帧真实测不到或发现碗没有跟随移动时，继续保留失败/未知，不把丢失物体自动判成成功。

同一完整all运行可跨子任务沿用炉内释放位置。独立extract若没有此前的释放状态、也看不到碗，仍需要显式提供/建立可靠初态，不能从空场景编造碗的位置。独立insert/table的人工持碗准备通过当前帧确认后，才恢复夹持估计。

日志在每个scene provenance.perception_refresh中分别记录refreshed_object_ids、retained_object_ids、estimated_object_ids、object_observations、state_estimates、keyframe_counts及effective_sensor_tolerances。缓存保留测量时间；估计记录SDK TCP和原始测量来源。open_loop_object_state只表示用户选择的开环策略允许继续，physical_verification=false；它不通过严格的真实成功证据检查。Task的task_success/operation_success表示执行流程完成，verified_task_success/physical_verification另行说明是否存在未验证假设。

默认感知容差：碗位置/夹持关系20mm、开口轴20°；碗绕局部+Y的圆对称转角先消歧，不作为滑落；柜体目标平移25mm、转角15°；门角到位5°，传感测角在资产限位附近允许5°噪声后截回合法区间，轨道残差25mm。大偏差仍拒绝，原始数值保留。关节限位、SDK起点误差、轨迹/自身碰撞、最低TCP高度和规划净空没有因感知噪声而放宽。

### 门板辅助把手方向与门角

复用AffordToolAct的门板ROI、RANSAC法向及yaw校正函数，但ROI坐标来自当前microwave_calibrated URDF，而不是旧模型的轴。先用已保存关闭参考RGB-D修正参考把手方向；运行关键帧再用门板校正把手yaw，位置不被平面校正改变。修正15°以内、内点至少60%、平面残差6mm以内且近竖直时采纳；门板遮挡或点数不足则记录原因，使用把手轨道测角。

两次平面都有效时，门板法向相对关闭参考的旋转与把手绕轴测角交叉校验（相差超过10°拒绝），一致时取二者平均作为门角。这里的“90°”包括抓取方向与门板正交和开门目标；实际开门角仍来自测量，绝不把指令90°当成实测90°。door_handle_diagnostics.json和perception_refresh.door_plane保存原始/修正pose、ROI点数、法向、内点比及修正量，辅助识别抓歪。

此复用策略以程序之外没有人工/其他进程操作夹爪为前提；现场外部改动后应重新建立任务连接。开口仍按0～1000满量程换算，实际位置命令上限990；反馈接受范围仍为 -10～1500。SDK 字段说明：https://develop.realman-robotics.com/robot/apipython/struct/gripperState/


## 操作策略与状态来源

除用户定制的 pick_bowl 外，放入、炉内取碗、取出撤离、桌面放置和第二次开门前过渡，
按 Gene_Agent_new 当前物理仿真的策略迁移。目标和净空几何在
gene_agent/tasks/policies/microwave_task/operations/transfer_scene.py 中按同源规则迁移；
真机适配负责实际状态来源、设备执行和现场后置证据。

- 放入：必要预倾斜、炉顶净空和持碗转向候选；采纳首个同时通过持碗进入、松爪包络、
  开爪撤离的完整序列。松爪落稳后用提交的释放位置估计重新筛撤离，增加 retreat_lift；炉内遮挡不强制重感知。
- 取碗：按当前碗缘和实际夹持关系生成目标；撤出用完整碗碰撞包络与炉口5mm净空。
- 桌面放置：直立碗的目标由显式工作区和真实桌面顶高决定，按当前夹持关系反算TCP；
  下降保持TCP直线，不自动改成绕行。
- 第二次开门前：保持姿态上抬20mm，再回配置的home关节分支。
- 物体更新：按上述关键帧调度；热session跟踪，缺失session时才分割注册并在需要时重采新帧。
  遮挡时沿用AffordToolAct式已保存物体/attachment状态，不持续重估整个场景。
- 当前TCP：同AffordToolAct，从实时关节读数经厂家SDK FK和已标定的flange→TCP得到；
  不用感知识别夹爪，也不用MuJoCo当前site冒充真机TCP。
- MuJoCo仅承载已观测状态的静态几何和预测副本，不执行物理step。预测接触、规划attachment
  不作为真实夹持或释放证据；允许继续的开环检查与实际视觉验证分开报告。

继续使用现有真机命令。新生成配置默认使用
data/task_1/realworld/door_strategy.json 与 operation_strategy.json；
后者的桌面XY工作区继承仿真的world参数，须与现场任务坐标下的桌面区域对应。
自定义旧配置中的显式参数仍需检查，历史run不会被修改。
代码级几何/mock检查不等于真机成功；最终需在右臂现场逐段验证。

本次只更新真机调用链及必要共享接线，Gene_Agent_new 和本仓库仿真执行器不作修改。
