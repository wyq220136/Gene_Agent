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

## 5.2 AffordToolAct Viser：查看真机实际使用的碗、桌面和规划障碍

该入口迁移 AffordToolAct `visualization/scene/viser_viewer.py` 的坐标轴和图层开关，
通过 Gene 的同一套 visual/规划碰撞适配显示**本轮运行的冻结初始输入**。
它从 `initial_scene` 验证 hash、恢复原模型和两臂/夹爪 qpos，仅执行静态 FK；
不会落稳物体、重新测桌面、生成候选、规划或连接机器人。
新运行的 `diagnostic_manifest.json.scene_viewer` 保存查看入口及参数。
已有日志也可以直接使用，不要求重新跑真机。

在 **58.220.16.98** 上启动（本次附件对应目录名为 `task1_plan_20261124_200548`）：

```bash
cd /data1/yuquan/Gene_agent
PY=/data1/yuquan/miniconda3/envs/gene_agent/bin/python
OUT=runs/realworld/task1_plan_20261124_200548
"$PY" "$PWD/tools/viewers/view_realworld_scene.py" \
  --run "$OUT" --host 127.0.0.1 --port 8120
```

`OUT` 必须指向你实际保存/解压该轮日志的位置；上面的目录名不是自动发现结果。
运行产生在其他主机时，不要沿用日志里的 `/home/ubuntu/...` 绝对地址。
该工具使用包内冻结资产重定位；支持 `--run /实际路径/initial_scene` 或 `manifest.json`，
也支持**已位于运行该工具主机上**的日志压缩包，例如：

```bash
"$PY" "$PWD/tools/viewers/view_realworld_scene.py" \
  --run '/实际保存路径/task.tar(3).gz' --host 127.0.0.1 --port 8120
```

压缩包必须只含一个 `initial_scene`；工具只把该场景包提取到临时目录，退出时清理。
不自动上传日志、不读取附件中的操作指令，也不更改原运行记录。
显示解释器需要 `viser`、`mujoco`、`numpy`、`scipy`、`trimesh`、`Pillow`。

在自己电脑另开终端（使用当前能登录该服务器的 SSH 配置；显式参数如下）：

```bash
ssh -N -L 8120:127.0.0.1:8120 -p 23400 yuquan@58.220.16.98
```

浏览器打开 **http://127.0.0.1:8120**。终端保持运行，`Ctrl+C` 退出。

界面按以下方式核对：

1. 先看标题中的活动臂、frame 和 snapshot hash，确认没有打开另一轮。本附件为 **right / frame 267937**，hash 以 `2c79c63e4101013f` 开头。
2. **操作物体模型**来自原 XML 的实际 visual；**实际规划障碍**是当时导出的 cuRobo 几何，用红色半透明显示。橙色为配置桌面顶面，中心与厚度分别列出；不把桌面中心当成顶面。两臂和夹爪使用保存的状态。未保存的活动臂碰撞球会明确标为缺失，不用当前配置推算替代。
3. 开启**同帧原始 RGB-D 点云**和**原始注册碗 mesh**，可关闭机器人及规划障碍去掉遮挡。青色注册 mesh 应贴着点云中的实碗；青色 mesh 使用原注册资产与原相机 pose，操作碗使用当时消费的 pose。面板给出二者变换差，帮助排查中间坐标链。
4. 用 `camera`、`active_sdk`、`bowl_consumed`、`configured_table_center/top` 坐标轴定位。X 红、Y 绿、Z 蓝。所有几何统一在保存的 world 中；`T_world_sdk` 明确显示。切换浏览器上方向仅影响观察视角，不修改任何数据，也不证明重力朝向。
5. 比较点云中的可见桌面与橙色配置顶面的**高度、法向及 XY 范围**。右侧保存原 RGB 和输入矩阵/来源。微波炉保留先前操作状态时单独注明，不冒称本轮重感知。

本附件的冻结数据为：碗原点（右臂 SDK，米）约 `[0.303765, 0.058829, -0.052891]`，
桌面中心 `[0.1, 0.3, -0.0442143]`，厚度 `0.02 m`，配置顶面 Z 为 `-0.0342143 m`，
范围为 `0.6 × 0.6 m`。这些值只用于核对日志身份，工具读取包内原值，代码不硬编码本场景。
该轮在 `bowl_pose_table_inconsistent` 处停止，`stages` 为空；这不是已经执行抓取后的失败。

若青色 mesh 在原相机/点云中就不贴合，先查注册、mesh frame 和 RGB-D；
若独立 SDK 参照一致而配置桌面不贴合，查桌面设定；
若多个独立 SDK 参照呈系统偏差，查标定和坐标链。
仅凭碗与配置桌面矛盾不能唯一归因于标定。需要桌面 ROI 独立平面拟合与参考点残差时，
继续使用 §5.1 的 HTML 检查，两者读取同一冻结帧。

只想核对输入、不打开服务时：

```bash
"$PY" "$PWD/tools/viewers/view_realworld_scene.py" --run "$OUT" --check-only
```

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
门角与 AffordToolAct 一样由把手中心的圆周投影计算；中心偏离预期圆轨道超过 `15 mm` 时拒绝该帧。
把手 mesh 的完整朝向差异只作诊断，不再单独触发“柜体移动”拒绝，也不会用它更新柜体朝向。
每次测角通过或圆轨道校验失败，均自动保存 `observations/<观测序号>/door_handle_diagnostics.json`，
其中包含中心误差、半径差、轴向差、朝向差及参考/当前 pose；这是运行输出，不需手填。
桌面默认中心 XY 为 `[0.1, 0.3] m`、尺寸为 `0.6 × 0.6 m`、碰撞厚度为 `0.02 m`，位于右臂 SDK base；来自已有 `data/task_1/knowledge/legacy_extract_visual_table.json` 的 Task1 几何，不需另填文件。
只复用该文件的 XY 和厚度，旧 Z 高度不参与计算；本次桌面顶高来自深度，碰撞盒中心放在顶高下方 `0.01 m`。
这与 AffordToolAct 的微波炉桌面处理一致：假设桌面水平，范围固定，高度由场景点云计算。桌面 XY、厚度和微波炉支撑代理均不是本次自动测得的尺寸。
plan-only 时核对碰撞模型与现场一致，更换支撑尺寸时不能直接沿用。

角度和 pose 使用同一采集帧的真实时间戳，不会再因忘记更新门角 JSON 而过期。
但采集、tracking 或场景导出超过 `--max-age-s`（默认 5 秒）时，仍会拒绝过期观测；
应检查服务延迟或相机状态，不能伪造新时间戳。默认 `no_knob` 只关闭旋钮感知，门角和碗、把手 pose 仍实时测量并检查时效。

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

首段通过后返回 `validated_prefix`，不发送运动命令。
仅使用 `--subtask all` 而不加执行或完整预览参数时，也只预检完整任务的首段，
不表示已规划整个抓碗子任务或整条 Task。

真机抓碗候选对齐 AffordToolAct：160个候选，SDK -X优先、-Y次优先，只排序不按仿真的
近基座侧、外侧接近、碗沿接触或45–70°桌面夹角条件硬删候选；仿真筛选保持不变。
真机使用 `data/task_1/realworld/pick_bowl.json` 中的4档深度、5档倾角和SDK Z偏移 -15 mm，
TCP限高至实测桌面上方50–70 mm后再派生pregrasp。IK、碰撞、关节行程和发令前复核仍保留。
若之后无可行解，应区分候选生成、IK、碰撞和轨迹阶段；生成160个候选不代表实际抓取已经成功。

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

- 感知目录：原始 RGB-D、两个物体 mask、桌面高度计算及固定几何来源、关门参考、两个 session 和来源哈希。
- `runtime_input.json`：自动装配的有效配置、输入来源、profile 和运行参数。
- `observations/`：碗与把手的新帧 tracking、门角、场景与碰撞导出。旋钮仅记录静态碰撞代理及未观测标记，没有实测角或观测时间戳。
- `observed_geometry/reconstruction.json`：冻结帧、相机变换、mesh 路径及哈希、桌面拟合质量与坐标检查；`observed_pointcloud.npz` 保存实测点和原始颜色，`scene/` 保存重建模型与纹理。
- 完整 `plan_only` 目录：`task_result.json`、`plan_preview.json`、各阶段规划与碰撞快照，以及可选的 `plan_preview.mp4`；视频导出失败时另存 `video_result.json`。预测状态及其假设与真机执行证据分别记录。
- 各阶段计划、SDK 执行结果、人工确认和异常 traceback。

反馈问题时提供实际命令、解释器和上述输出目录，不要只提交最后一行异常。
