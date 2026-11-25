# 2026-11-25 标定修正版

本目录复用 microwave_third_view_20261112-161501 的原相机/PnP 采样及 .19 右臂 SDK TCP 读数。
依据用户明确确认进行两项修正：测量板坐标 [0,+.17,-.125] 改为 [0,-.17,+.125] 米；板到 SDK 轴关系为 Rx(180°)。

T_sdk_board 的平移按 p_sdk_tcp - R_sdk_board @ p_board_tcp 重算，不直接围绕 SDK 原点翻转旧外参。
本例两项修正同时应用后，T_sdk_board 平移与原值相同，旋转变为 diag(1,-1,-1)。
board_axes_aligned_to_sdk=false；完整 TCP 记录和显式板内 TCP 旋转确保之后重新合并不会还原成单位旋转。
board_tcp_rotation 是依据保存的 SDK TCP 姿态与用户指定板轴关系推导，未冒称新增实测。

cam_param.json 的 extrinsic_matrix / right_extrinsic_matrix 均为 T_Sright_Camera，两字段一致。
原目录、机器人安装和历史运行均不修改；来源 hash、锚点误差及修正矩阵见 correction_report.json。
同一冻结帧几何检查恢复桌面 +Z 和碗口方向一致；尚未采集新现场、验证真机操作或进行新规划。

使用：将 CAMERA_PARAMS 指向本目录 cam_param.json，在新目录 prepare 感知；不能复用旧 PERCEPTION。
随后重建 COMMON 数组，并按操作指南运行只规划与 Viser 检查。再次标定板姿态须独立确认实际轴向，不能盲用本次半转。
