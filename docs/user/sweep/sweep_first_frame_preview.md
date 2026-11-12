# 从相机首帧预览抓刷与清扫轨迹

适用于原蓝色旧刷、一个绿方块和簸箕，固定相机，Realman 单臂 + Pika。使用现有物体 mesh 与 Concept，自动分割和首帧 FoundationPose 注册，不需要重新 SAM3D 重建。

## 一个终端运行

在项目根目录执行：

```bash
PYTHONUTF8=1 PYTHONPATH=. PYTHONNOUSERSITE=1 \
/home/ubuntu/miniconda3/envs/curobo/bin/python tools/run/preview_sweep_from_camera.py \
  --camera-param data/calibration/cam_param.json
```

无需预先启动相机、SAM3 或 FoundationPose 服务。入口直接采一帧后释放相机，临时启动自己的模型服务，依次完成 SAM3、FoundationPose，退出模型进程释放显存后再运行 cuRobo。不会卸载或结束其他程序的模型服务；其他程序占满显存仍可能导致 OOM。全过程没有机器人运动执行选项。

默认相机序列号来自标定文件；当前 `cam_param.json` 为 **323522063237**，不是早先的 337122070986。固定相机的 `extrinsic_matrix` 必须是米制 `T_sdk_camera`，即将相机点映射到 SDK base。已经合成的最终外参只应用一次，不再乘 board 变换或翻轴。采集分辨率、序列号和内参不匹配时停止，不猜测外参。

默认只读真实机械臂的关节与法兰，地址优先采用标定文件的 `communication.robot_ip`，其次现场配置，均未配置时使用 **169.254.128.19**。连接或读取失败时停止，不用模板姿态替代。也可显式指定地址：

```bash
PYTHONUTF8=1 PYTHONPATH=. PYTHONNOUSERSITE=1 \
/home/ubuntu/miniconda3/envs/curobo/bin/python tools/run/preview_sweep_from_camera.py \
  --camera-param data/calibration/cam_param.json \
  --robot-ip 169.254.128.19 --gripper-width 0.04
```

入口只读关节与法兰，采集前后检查机器人静止，不发运动命令。界面初态为拍照时的实测状态，不是持续实时遥测。仅做离线诊断时可显式加 `--assume-standby`，界面会标为假定待机。开口由用户填写（米），不连接夹爪电机。SDK Python 默认使用本机 `pika` 环境，可用 `--sdk-python` 指定。

## 预览内容

规划顺序为：假定打开夹爪 → pregrasp → TCP 直线接近 → 假定闭爪 → 直线抬升 → 原 Sweep Skill 的预备、下降、扫动、撤离。使用 cuRobo，不启用 CPU IK fallback。抓取几何来自现有旧刷 Knowledge；清扫复用 `run_sweep_skill`。完整 MuJoCo FK 几何复核每段轨迹，未运行物理仿真。

默认打开 Viser：**http://127.0.0.1:8093**。橙线是规划轨迹的实际 FK TCP 路径；时间轴可检查机械臂与刷子；`Show first-frame RGB-D cloud` 叠加首帧点云核对 mesh 与姿态。`Show target TCP frames` 查看抓取/清扫目标。

抓取前刷子固定在观测位置；预测闭爪后才假定刷子相对 gripper 固定。没有真实夹持、接触或任务成功证据，方块不按预测扫动。感知残差偏大时界面提示核对，不能把“规划完整”当作感知正确或真实任务成功。当前 Pika profile 是名义离线核对配置，也不能据此直接执行。

方块 mesh 会依据场景记录恢复用户曾为仿真缩小的尺度，Concept 和碰撞几何同步调整；不修改历史场景。桌面高度由排除物体后的水平背景深度估计，尺寸仍为配置的障碍物代理。无法确认水平桌面时停止，可用 `--table-top-z` 显式指定 SDK base 下的米制桌面高度，并仍核对背景深度支持。

## 换 prompt、复用同一张图和重新查看

每轮输出新目录 `runs/realworld_sweep/camera_<时间>/`，或用 `--output` 指定不存在的目录。保存 capture、三个 mask 与 overlay、mesh、原始模型返回、位姿残差、桌面证据、碰撞快照、原始 GPU 返回、轨迹与失败 traceback。

分割不合适时，复用上一轮 `capture/`，换 prompt，不重新拍照：

```bash
PYTHONUTF8=1 PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/preview_sweep_from_camera.py \
  --camera-param data/calibration/cam_param.json \
  --capture-dir runs/realworld_sweep/camera_<上一轮时间>/capture --assume-standby \
  --brush-prompt "a brush" --cube-prompt "a green block" \
  --dustpan-prompt "a white plastic dustpan"
```

离线帧不能混用当前机器人状态，需要显式 `--assume-standby`；它不会显示拍照时未知的实际姿态。用 `--no-viewer` 可只生成记录；之后沿用原入口：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python tools/viewers/view_task.py \
  --task sweep_cube_into_dustpan --run runs/realworld_sweep/camera_<时间>

PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python tools/viewers/view_skill.py \
  --skill sweep_cube --run runs/realworld_sweep/camera_<时间>
```

规划失败时也保留已完成的抓取前缀与初态可供检查；不换成仿真 GT。仅按下面的支撑策略校正初始 Z，其余摆位不自动改变。若感知本身失败，查看对应 overlay、服务日志和 traceback，修正 prompt 或采集条件后重跑。

## 已验证范围（2026-10-24）

- 旧图 `data/realworld/brush_0`：三个物体的真实 SAM3 分割与 FoundationPose 注册均运行完成，模型按顺序退出。首次分割阶段约 8.48 秒，三物体位姿阶段约 7.11 秒，包含服务启动/退出，不代表新机器或新帧固定耗时。
- 历史旧刷 fresh-reset 初态：实际 GPU 的 pregrasp、直线接近、抬升、四段清扫均规划通过；不是新相机闭环或物理任务成功。记录 `runs/realworld_sweep/table_preview_gpu_20261024`，已检查初态与下降接触预览截图。
- 新相机拍照、现场标定精度、新摆位可达性与真实物体 mesh 对齐仍需运行上述命令现场检查。旧图测试中刷子、簸箕存在明显深度残差，已提供残差提示及三维点云核对，不冒称精确估计。

### 2026-10-24 实景首轮诊断

`camera_20261024_222241` 停在 `open_assumed` 的初态几何复核，刷子与桌面最大穿透约 16.4mm，尚未调用 cuRobo，不是 IK 失败。可见深度拟合残差较小也不能证明完整 mesh 与桌面一致；当轮旧版本尚未做支撑面校正（现已按下节改进）。界面现在显示具体停止原因与红色碰撞点。新标定 `latest/third_camera/third_view_extrinsic.json` 记录右臂 `192.168.1.18`，只读 SDK 关节/法兰已实际验证可用；旧记录的模板姿态不会被改写成实测。

此设备的 connected SDK 实测为 RM75/6FB，四组纯 FK 与 URDF link7 对齐；现场配置按 IP 选择独立的 `pika_frames.rm75_6fb.json`，避免套用标准末端离线 profile 的 −17.2mm 偏移。Pika 物理 TCP 仍未因此获得现场标定。可用 `--frame-profile` 显式覆盖其他设备的配置。

## 初始感知穿桌的处理（当前默认）

默认只对声明在桌上的 brush/cube/dustpan 做有界支撑校正：沿 world Z 向上移动，使碰撞几何最低点比桌面高1mm；XY和旋转不变。单物体默认最大校正30mm，超限仍报错。不会将初始感知穿透直接当作机器人运动失败，也不放宽机器人、夹爪或后续轨迹的碰撞阈值。

原始 observation/点云保留；`scene/support_adjustment.json` 保存每个物体的原始/规划pose、校正量和上限。界面 `Show raw perception meshes` 显示紫色半透明原始mesh，可与校正后的规划场景、首帧点云对照。校正后的初态明确标记为规划假设，不冒称实测pose已改变或物理任务成功。

可用 `--no-support-correction` 关闭，或 `--max-support-correction 0.02` 改为20mm预算。此校正只适用于首帧只规划入口，不改变已有人工作业持刷接口或真机执行门禁。

在保存的同帧实机输入上，刷子/方块/簸箕分别校正约16.4/25.9/11.8mm后，七段实际cuRobo规划全部通过。记录：`runs/realworld_sweep/support_corrected_20261024`。这证明该规划场景可规划，不证明校正后的mesh精确符合实物，也不代表清扫任务已执行成功。

真机执行使用独立薄入口 `tools/run/execute_sweep_realworld.py`，仍复用原Skill与微波炉Runtime；本页相机入口保持只规划。阶段执行及现场配置见[真机清扫说明](sweep_realworld_execution.md)。
