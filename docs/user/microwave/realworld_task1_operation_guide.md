# 微波炉 Task1 真机操作流程

本流程对应 `tools/run/run_microwave_task1_realworld.py`：
**启动服务 → 提供相机标定 → 自动准备感知 → plan-only → 确认后执行**。
不再创建或手工填写 `realworld_site.json`，也不需要手写门角、旋钮角状态 JSON。
程序生成的 JSON 用于保存感知结果和运行证据，操作者无需编辑。

`run_microwave_task1_physical.py` 是 MuJoCo 仿真入口。真机概览见[真机微波炉任务入口](realworld_microwave_entry.md)。

## 1. 使用前提

- 操作 Skill、Knowledge、阶段顺序及操作 profile 沿用原实现，本流程只调整配置与感知接线。
- 默认只预检；实际运动需要显式 `--execute --yes`。离线检查通过不等于真机已跑通。
- 当前使用右臂 SDK base 作为场景参考，柜体和相机在本次运行期间必须固定。
- 使用任务对应的微波炉、碗和标准夹爪资产；标准夹爪 TCP、开口范围必须与规划模型一致。
- 正式使用 `--collision-mode full`，不能用 `robot_only` 掩盖场景碰撞问题。当前没有任意阶段停止的 `--stop-after-stage` 参数。
- 旋钮接触需现场单独验收。整条任务可先用 `--no-knob` 跳过旋钮动作；仍跟踪旋钮以核对柜体参考。

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

将碗、微波炉和机械臂摆到本次初始位置，把门关紧，上方旋钮置零。
让相机看清碗、把手、上方旋钮及完整桌面边界。此命令只调用感知服务，不连接机械臂：

```bash
PERCEPTION=runs/realworld/perception_$(date +%Y%m%d-%H%M%S)
"$PY" tools/run/prepare_microwave_realworld.py \
  --camera-params "$CAMERA_PARAMS" \
  --output "$PERCEPTION" \
  --bowl-prompt "blue bowl" \
  --handle-prompt "pink microwave door handle" \
  --knob-prompt "upper microwave knob including pointer" \
  --table-prompt "table top"
```

prompt 按实际外观调整。非默认地址用 `--camera-endpoint`、`--seg-server`、`--pose-server` 指定。

1. 输入 `CLOSED_ZERO`，确认门关紧、旋钮零位、柜体和相机固定。
2. 程序采集同一帧 RGB-D，核对相机标定，自动请求 SAM3 分割四个目标。
3. 打开输出目录中的 `masks_preview.png`，检查四个 mask；核对终端显示的桌面中心和 XY 尺寸。拟合诊断保存在 `table_measurement.json`，无需编辑。
4. 确认桌面边界完整、mask 和标定正确后输入 `ACCEPT`。若边界裁剪或 mask 错误，停止并调整视角或 prompt，使用新目录重跑。
5. 程序卸载 SAM3，自动注册碗、门把手和旋钮三个 FoundationPose session。碗和把手使用已有 mesh；旋钮 mesh 从当前 URDF 的 visual 自动导出。
6. 成功后生成 `perception_setup.json`，保存 session、零位参考、桌面结果及标定和资产来源哈希，无需编辑。

准备完成后保持 Pose 服务运行。运行时读取新帧；初始零位姿态只作为几何参考，不充当当前门角。
改变标定、资产或准备输入后，需以新目录重新准备。

## 6. 配置和状态来源

| 参数或状态 | 来源 |
|---|---|
| 两臂 IP、端口 | 命令行；端口默认 8080 |
| 相机外参、内参、序列号 | 已有标定输出 |
| 碗、把手、旋钮 session | 准备命令自动注册 |
| 门角 | 当前帧把手 pose 相对固定门轴的圆周投影 |
| 旋钮角 | 当前帧旋钮方向相对零位的转角，支持跨越 180° |
| 柜体 pose | 关闭参考与资产 frame 配对；每帧核对把手、旋钮的关节几何一致性 |
| 桌面高度、可见 XY 范围 | RGB-D 与桌面 mask 拟合，并由操作者确认 |
| 资产 frame、关节轴、限位 | 已有场景配置和 URDF，自动读取 |
| TCP、开口范围 | 标准夹爪 profile，并与规划模型校验 |
| 操作 Skill 参数 | 原有 profile；现场误差检查由配置层统一装配 |

柜体参考依赖“柜体固定”的前提，几何一致性检查不能替代独立柜体测量。
桌面 XY 是可见范围；桌板厚度和微波炉支撑代理仍来自任务资产，未自动测得。
plan-only 时核对碰撞模型与现场一致，更换支撑尺寸时不能直接沿用。

角度和 pose 使用同一采集帧的真实时间戳，不会再因忘记更新门角 JSON 而过期。
但采集、tracking 或场景导出超过 `--max-age-s`（默认 5 秒）时，仍会拒绝过期观测；
应检查服务延迟或相机状态，不能伪造新时间戳。`--no-knob` 不关闭状态测量。

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
  "${COMMON[@]}" --subtask all --no-knob --collision-mode full --output "$OUT" \
  --execute --yes
```

旋钮单独验证后再去掉 `--no-knob`。结束或中断后按实物状态检查和复位，不能假定机器人已经回 home。

## 9. 中断与恢复

| 现象 | 处理 |
|---|---|
| mask 错误、桌面边界不全 | 调整视角或 prompt，以新目录重新准备 |
| 相机序列号、内参或标定不一致 | 修正相机配置或重新标定，再准备感知 |
| tracking session 不存在 | 检查 Pose 服务是否重启，重新准备感知 |
| 把手或旋钮不符合固定关节轴 | 检查柜体移动、mask、标定及 tracking 漂移，不手改角度 |
| 门角或观测过期 | 检查采集与计算延迟，取得新帧，不修改记录时间 |
| 阶段失败或夹持不可靠 | 停止后按实物状态复位，保留记录 |

`Ctrl+C` 不能替代机械急停。异常时按现场停机流程处理，确认机械臂和夹爪状态后再恢复。

## 10. 运行证据

- 感知目录：原始 RGB-D、mask、桌面拟合、零位参考、session 和来源哈希。
- `runtime_input.json`：自动装配的有效配置、输入来源、profile 和运行参数。
- `observations/`：每次新帧 tracking、门角和旋钮角、场景与碰撞导出。
- 各阶段计划、SDK 执行结果、人工确认和异常 traceback。

反馈问题时提供实际命令、解释器和上述输出目录，不要只提交最后一行异常。
