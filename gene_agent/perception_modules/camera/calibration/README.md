# 相机标定：两条使用路线

| 路线 | 怎么做 | 适用情况 |
|---|---|---|
| **B：手眼＋移板（2＋5）** | 先固定板、移动机械臂求手眼；再固定机器人与双相机、只移动板 | 已有手腕相机，减少量尺操作；已有有效手眼时直接做第二步 |
| **A：量尺＋板相机（1＋3）** | 人工对齐板与基座，API读TCP并量尺；相机拍板后组合外参 | 不依赖手腕手眼变换，保留最初的人工方法 |

**统一命令：** `python -m gene_agent.perception_modules.camera.calibration --help`。
以下在仓库根目录运行。长度单位米，`T_A_B` 把B中的点变换到A。

## 只想找最新结果

```bash
python -m gene_agent.perception_modules.camera.calibration results
```

**第三相机外参固定入口：`data/calibration/latest/third_camera/third_view_extrinsic.json`。**
手眼固定入口：`data/calibration/latest/handeye/handeye.json`。
它们指向最近确认保存的完整采集目录；标定通过并ACCEPT后自动更新，失败和取消不会替换。终端会突出打印固定入口和本轮目录。仓库外自定义输出目录不自动登记到本仓库latest。

正式第三相机结果采用旧 `cam_param.json` 的公共字段：`width`、`height`、`intrinsic_matrix`、`serial`、`extrinsic_matrix`。`extrinsic_matrix` 是相机→SDK base，平移米；原同义字段 `T_sdk_third`、`camera_serial` 在正式输出中分别改为 `extrinsic_matrix`、`serial`，不重复保存。求解过程和质量报告仍使用数学变量 `T_sdk_third`。内参与尺寸来自本次第三相机原始RGB，不能沿用另一台相机的旧参数。其他程序可将该JSON直接作为相机参数文件读取。

## 标定后检查桌面与基座XYZ裁剪

```bash
python -m gene_agent.perception_modules.camera.calibration inspect
```

默认直接读取最新正式结果和已保存图像，显示基座XYZ、六个裁剪边界、裁剪前后点云及RGB覆盖。`Analyze table`给出平面候选倾角与高度，保存记录集中在`data/calibration/diagnostics/`。`--live`可用当前相机单帧检查；`--snapshot`无界面保存检查图。详见[裁剪检查说明](inspection/README.md)。桌面完整/平整只是一项诊断，不能独立证明全部外参准确。

## 路线B：下次使用从这里开始

### 0. 相机服务

已有服务时直接复用，不要重复打开同一设备。没有服务时在两个终端分别运行：

```bash
conda activate sam3
python -m gene_agent.perception_modules.camera.camera_server \
  --serial 412622271769 --endpoint tcp://127.0.0.1:18082 \
  --width 1280 --height 720 --stream-fps 30 --max-fps 10
```

```bash
conda activate sam3
python -m gene_agent.perception_modules.camera.camera_server \
  --serial 323522063939 --endpoint tcp://127.0.0.1:18083 \
  --width 1280 --height 720 --stream-fps 30 --max-fps 10
```

相机序列号须与实际设备匹配。两步中都应确认红圈0号角点对应棋盘格同一物理内角点。

### 1. 手眼标定：安装未变、有有效结果时跳过

板固定不动，人工改变机械臂位置及至少两个旋转方向。默认采20姿态，空格取样、松开约1秒后再按空格确认；Open3D检查后输入ACCEPT保存。

```bash
conda activate zehao
python -m gene_agent.perception_modules.camera.calibration handeye \
  --ip 192.168.1.18 --group right \
  --camera-endpoint tcp://127.0.0.1:18082 --serial 412622271769 \
  --pattern-size 11 8 --square-size 0.015 --poses 20 --visualize
```

新结果自动放在 `data/calibration/runs/handeye/<本轮目录>/handeye.json`，同时保留来源 `samples.json` 和原图。
你已有的手眼结果已整体归入：

```text
data/calibration/runs/handeye/calibration_wrist_right_001/handeye.json
```

它给出 `T_flange_camera`（手腕相机→法兰）。手腕相机安装变化后需重做；仅移动板不需要重做。

### 2. 第三相机外参：固定机器人，只移动板

先将手腕停在两相机都能看见板的姿态，**启动后机械臂和两相机全程固定**。以下使用最近正式手眼结果；重做第一步并确认保存后，该固定入口会自动更新。若要使用指定历史结果，再将 `--handeye` 换为对应历史路径。

```bash
conda activate zehao
python -m gene_agent.perception_modules.camera.calibration moving-board \
  --handeye data/calibration/latest/handeye/handeye.json \
  --ip 192.168.1.18 --group right \
  --wrist-endpoint tcp://127.0.0.1:18082 \
  --third-endpoint tcp://127.0.0.1:18083 --third-serial 323522063939 \
  --poses 8 --validation-poses 2 --visualize
```

每次只移动板，停稳并确保两相机看全板，空格取样、松开后空格确认。前6板位估计、后2板位独立验证。检查同板位双相机点云，数值通过且输入 `ACCEPT` 后保存：

- `third_view_extrinsic.json`：第三相机→基座 `extrinsic_matrix`，及逆矩阵 `T_third_sdk`。
- `board_poses.json`：每个板位各自的板→基座、板↔相机变换。

默认目录为 `data/calibration/runs/moving_board/<本轮目录>/`。本步骤不发送运动指令；状态默认UDP，初始化会设置遥测目标，不要同时运行其他机器人控制程序。多板位一致性无法独立验证已有手眼的绝对精度。

详情：[手眼采集](wrist_handeye/README.md)、[移板采集与验证](third_view/moving_board.md)。

## 路线A：量尺＋板相机

```bash
conda activate zehao
python -m gene_agent.perception_modules.camera.calibration manual \
  --side right --robot-ip 192.168.1.18 \
  --camera-endpoint tcp://127.0.0.1:18083 \
  --pattern-size 11 8 --square-size 0.015
```

向导已包含人工对齐、API读取TCP、量尺、相机拍板和外参组合，不必重复运行一次板相机标定。板在测量到拍照之间必须不动；默认结果在 `data/calibration/runs/manual/`，最终外参在 `cam_param.json`。

只需单独拍板时用 `python -m gene_agent.perception_modules.camera.calibration board-camera --help`；额外提供**当前有效的**板→基座矩阵，可组合固定相机基座外参。详情：[人工量尺流程](single_arm/README.md)、[板相机流程](third_view/README.md)。

## 数据、续采和离线检查

```text
data/calibration/
  latest/             最近确认的正式结果固定入口，指向原始会话
  runs/handeye/        手眼原图、样本、质量报告与结果
  runs/moving_board/   固定机器人移板采集与结果
  runs/manual/         人工量尺采集与结果
  runs/board_camera/   单相机板观测
  runs/fixed_board/    保留的固定板回放方案记录
  runs/crosscheck/     保留的旧交叉检查记录
  diagnostics/        图片分析等诊断记录
  archive/            目录迁移清单与校验记录
```

原 `data/calibration/` 下的 `cam_param*.json` 等人工结果保留兼容；新会话不继续堆在该层。

- 输出目录每次自动新建，也可显式 `--data-dir <本轮目录>`。
- 中断后原命令增加 `--data-dir <同一目录> --resume`。手眼步骤要求板不动；移板步骤允许板移动，但机器人和两相机必须不动。
- 移板数据采齐后离线预览：`python -m gene_agent.perception_modules.camera.calibration moving-board --preview-only --data-dir <本轮目录>`，不连接设备。
- 现有19个旧目录已整体归档，原JSON和原图未改写；常用向导能读取迁移前旧路径，但新命令请使用新路径。外部工具、旧报告中的原样路径不会自动更新。
- 原始采集、诊断图片和试验输出只放本专题数据目录，不放代码或说明文档目录；开发历史集中在 `docs/developer/calibration/`。

## 扩展方案

两条路线之外的功能仍保留：[固定板、MoveL回放旧姿态](third_view/fixed_board_replay.md)、[旧双相机交叉检查](third_view/README.md#旧双相机移板交叉检查显式选择)。旧 `guided_*` 命令仍可使用，数值和运动策略不变。旧45°双臂安装说明仅作[历史参考](../../../../docs/tmp/history/developer/calibration/legacy_dual_arm.md)，不用于当前单臂标定。
