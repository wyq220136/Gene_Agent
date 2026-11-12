# 标定索引

含 cv2/open3d 的固定板、移板、手眼测试在 `zehao` 环境用 unittest 运行；该环境目前没有 pytest。不连接或移动真机。

```bash
python -m unittest discover -s tests/perception/calibration -p "test_*extrinsic.py"
python -m unittest discover -s tests/perception/calibration -p "test_wrist_handeye.py"
```

单臂输入和存储回归可在 cuRobo 环境使用 pytest。

按主题列出用途与入口；少量子目录内容直接合并到此处。

| 文件 | 用途 |
|---|---|
| [test_calibration_storage.py](test_calibration_storage.py) | 标定数据归档与统一入口的离线回归 |
| [test_fixed_board_extrinsic.py](test_fixed_board_extrinsic.py) | 固定板完整外参与关节回放的离线回归；不连接硬件 |
| [test_moving_board_extrinsic.py](test_moving_board_extrinsic.py) | 固定机器人移板标定：几何与流程回归，不连接设备 |
| [test_single_arm_calibration.py](test_single_arm_calibration.py) | 单臂 SDK base 标定的结果与拒绝条件 |
| [test_wrist_handeye.py](test_wrist_handeye.py) | 手腕相机手眼标定的离线几何回归；无需真机 |
