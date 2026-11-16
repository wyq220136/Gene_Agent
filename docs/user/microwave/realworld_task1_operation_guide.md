# 微波炉 Task1 真机操作流程

本流程对应 `tools/run/run_microwave_task1_realworld.py`：
**启动服务 → 提供相机标定 → 自动准备感知 → plan-only → 确认后执行**。
不再创建或手工填写 `realworld_site.json`，也不需要手写门角、旋钮角状态 JSON。
程序生成的 JSON 用于保存感知结果和运行证据，操作者无需编辑。
准备感知和运行任务都默认 `no_knob`：不分割、不注册、不跟踪旋钮，也不执行旋钮动作。

`run_microwave_task1_physical.py` 是 MuJoCo 仿真入口。真机概览见[真机微波炉任务入口](realworld_microwave_entry.md)。

## 1. 使用前提

- 操作 Skill、Knowledge、阶段顺序及操作 profile 沿用原实现，本流程只调整配置与感知接线。
- 默认只预检；实际运动需要显式 `--execute --yes`。离线检查通过不等于真机已跑通。
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
3. 程序按 AffordToolAct 的方式，在已有矩形桌面范围内从标定后的点云计算水平桌面高度；不请求桌面分割，也不做桌面 pose estimation。计算记录保存在 `table_measurement.json`，无需编辑。
4. 打开 `masks_preview.png` 检查两个物体 mask，核对终端显示的桌面顶高、固定中心和尺寸与现场相符后输入 `ACCEPT`。若物体 mask 错误，调整 prompt；若桌面高度异常，检查深度、标定和场景位置，然后使用新目录重跑。
5. 程序卸载 SAM3，使用已有 mesh 自动注册碗和门把手两个 FoundationPose session；不导出旋钮 mesh，不注册旋钮。
6. 成功后生成 `perception_setup.json`，保存感知范围、session、关门参考、桌面结果及标定和资产来源哈希，无需编辑。

准备完成后保持 Pose 服务运行。运行时读取新帧；初始关门姿态只作为几何参考，不充当当前门角。
改变标定、资产或准备输入后，需以新目录重新准备。
切换到本流程时，按上述命令在新目录重新准备。旧桌面分割流程的记录因桌面来源已改变，需重新生成；不要手改记录或继续使用失败目录。

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

## 7. 配置检查与 plan-only

在运行终端设置以上变量，再使用统一参数：

```bash
COMMON=(--perception "$PERCEPTION" --left-ip "$LEFT_IP" --right-ip "$RIGHT_IP")
"$PY" tools/run/run_microwave_task1_realworld.py \
  "${COMMON[@]}" --subtask pick_bowl --collision-mode full --check-config
```

`--check-config` 只检查本地输入和配置，不连接机械臂，也不证明远端 session 存活。
正式 CLI 已不接收 `--config realworld_site.json`。

```bash
OUT=runs/realworld/pick_bowl_plan_$(date +%Y%m%d-%H%M%S)
"$PY" tools/run/run_microwave_task1_realworld.py \
  "${COMMON[@]}" --subtask pick_bowl --collision-mode full --output "$OUT"
"$PY" tools/viewers/view_skill.py --skill pick_bowl --run "$OUT"
```

plan-only 会连接并读取两臂和夹爪状态，取得新感知后检查首段规划，不发送运动命令。
它不能证明后续真实接触或整条 Task 成功。检查桌面、支撑、门轴、尺度、机械臂和 TCP，
两臂停在已知净空位置后再开始。

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
各子任务通过现场验收后，再测试完整任务：

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
- 各阶段计划、SDK 执行结果、人工确认和异常 traceback。

反馈问题时提供实际命令、解释器和上述输出目录，不要只提交最后一行异常。
