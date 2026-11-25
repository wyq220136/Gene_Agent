# 2026-11-25 当前外参：原始测量偏移 + Rx180 板轴关系

按用户最新要求，已撤销测量符号修正。板中 TCP 测量恢复为原始 [0,+0.17,-0.125] 米。
只保留用户确认的板到 SDK 旋转 diag(1,-1,-1)。

T_sdk_board 平移按 p_sdk_tcp - R_sdk_board @ p_board_tcp 求得，保持原 SDK TCP 锚点。
当前相机 SDK 位置约 [0.023422,1.071217,0.584066] 米。
extrinsic_matrix 与 right_extrinsic_matrix 同步更新；完整板内 TCP 旋转使之后重新合并不退回单位旋转。
未更改原 PnP 相机到板矩阵、11-12 原始目录、机器人安装或历史运行。

上一版符号修正的所有文件已备份在 revisions/before_restore_original_measurement/，该目录不再用于新感知。
当前结果与点云/mesh 对照见 correction_report.json。物体与点云的共同变换不会验证标定平移，需机器人独立几何核对。
使用本目录 cam_param.json 在新目录 prepare，再重建 COMMON 数组和新 OUT；旧冻结日志保持原样。
