# 标定数据

操作命令见[标定主说明](../../gene_agent/perception_modules/camera/calibration/README.md)。

**找最新结果直接打开 `latest/`。** 第三相机外参：`latest/third_camera/third_view_extrinsic.json`；手眼：`latest/handeye/handeye.json`。成功且确认保存后自动更新，原始会话保留。终端也可运行 `python -m gene_agent.perception_modules.camera.calibration results`。

- `runs/handeye/`：路线B第一步，手眼采集和结果。
- `runs/moving_board/`：路线B第二步，固定机器人移板采集和结果。
- `runs/manual/`、`runs/board_camera/`：路线A量尺和板相机数据。
- `runs/fixed_board/`、`runs/crosscheck/`：保留的其他方案采集记录。
- `diagnostics/`：图片分析和诊断，避免与正式代码、永久资产混放。
- `archive/`：数据目录迁移清单，保留旧新路径和原文件SHA256。

2026-10-24已整体迁移19个原data根目录的标定目录，校验127个文件内容不变。没有删除失败样本，也没有将试验结果自动提升为正式标定。

已有手眼：`runs/handeye/calibration_wrist_right_001/handeye.json`。
已有移板会话：`runs/moving_board/calibration_moving_board_20261024_202400/`。

本层旧 `cam_param*.json`、板位JSON和预览图是原人工流程输出，为兼容现有调用保留；新采集须在runs中单独建本轮目录。
