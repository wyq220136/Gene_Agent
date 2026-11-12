# 标定后检查：基座XYZ裁剪与桌面

参考 `pika_usage` 的实际 `bash pika.sh guide data/sweep_cube/hand_single` 裁剪流程：RGB-D反投影 → 用外参变换到机器人SDK base → 按X/Y/Z闭区间裁剪。裁剪前后点云和RGB覆盖使用同一完整像素掩码，抽稀仅用于3D显示。运行不依赖旧项目。

## 直接检查最近正式标定

```bash
conda activate zehao
python -m gene_agent.perception_modules.camera.calibration inspect
```

默认读取 `data/calibration/latest/third_camera/` 的正式外参与第一板位第三相机RGB-D；不连接机器人或相机。指定其他保存板位：

```bash
python -m gene_agent.perception_modules.camera.calibration inspect \
  --data-dir data/calibration/runs/moving_board/calibration_moving_board_20261024_202400 \
  --sample 3
```

窗口左侧为裁剪前点云（删除点灰色），右侧为保留点；底部为原始RGB和裁剪覆盖。调整右侧六个 `X/Y/Z min/max` 滑块或数值框，橙框为基座下裁剪范围，基座轴红X、绿Y、蓝Z。`Top XY`、`Front XZ`、`Side YZ`切换视角，`3D overview`恢复斜视。

RGB中暗色是XYZ裁剪排除，紫色是无效深度或超过相机深度上限；紫色区域不能误认为是外参裁剪错误。默认 `--max-depth 1.2` 米，限定相机光轴深度；它与基座下的XYZ边界是两种不同过滤。

初始范围只是操作起点：X[-0.5,0.9]、Y[-0.6,0.6]、Z[-0.05,0.10]米，**不是已测得的桌面范围**。根据现场调整。保留点比例是全场景有效点的比例，不是语义桌面覆盖率；工具没有自动识别“真正桌面”的像素标注。

## 怎样检验标定

1. 先通过独立测量，确定桌面相对SDK base的预期高度、边界或几个已知点，再用固定边界检查点云。只把裁剪框调到“看起来合适”，不能验证外参正确。
2. 看RGB覆盖图：桌面是否出现不合理的斜切或缺失，桌上高物体是否随Z上界正确变化。
3. 看基座XYZ和俯视/侧视，比较桌面法线与基座Z。若实际桌面并不平行于基座XY，不能把真实安装倾角归咎于标定。
4. 点击 `Analyze table`，显示水平主平面**候选**的倾角、基座Z中位数、平面拟合残差。它可能选到其他水平表面，须结合图像确认。
5. 若有**独立测得**的桌面基座Z高度，启动时加 `--table-z 0.0`（这里0.0须换为实际值），才报告高度偏差。

单平面无法检验XY平移、绕Z旋转，也不能证明深度尺度/绝对外参正确。不要通过调整框或放宽范围把错误标定“验收通过”；此工具不改标定、不生成标定通过标志。

## 保存与复查

点击 `Save config + report + image`，本轮输出统一放在 `data/calibration/diagnostics/base_crop_<时间>/`：

- `inspection.png`：裁剪前后点云、原始RGB和完整像素覆盖图。
- `inspection_report.json`：全部有效点统计、范围、来源、外参哈希和桌面候选诊断。
- `point_cloud_processing.json`：与PIKA相同的XYZ裁剪配置格式，绑定相机serial、基座和外参SHA256；2048采样字段只供下游使用，本检查不以2048点估计覆盖率。
- 实时采集额外保存 `capture.npz` 供复现；已有会话原图不重复拷贝。

无界面直接保存快照：

```bash
python -m gene_agent.perception_modules.camera.calibration inspect --snapshot
```

固定范围复查：

```bash
python -m gene_agent.perception_modules.camera.calibration inspect \
  --crop-config data/calibration/diagnostics/<实际目录>/point_cloud_processing.json
```

重新打开同一次检查：

```bash
python -m gene_agent.perception_modules.camera.calibration inspect \
  --data-dir data/calibration/diagnostics/<实际目录> \
  --crop-config data/calibration/diagnostics/<实际目录>/point_cloud_processing.json
```

相机serial、K、分辨率不符时拒绝加载；配置与当前外参哈希不符时拒绝复用。更换外参后需重新检查并保存范围；软件不会静默将旧框套到新标定。默认按已去畸变/畸变可忽略的对齐RGB-D处理；显式给出非零畸变时拒绝忽略。

## 用当前相机画面检查（也适用于量尺方案）

```bash
python -m gene_agent.perception_modules.camera.calibration inspect \
  --live --camera-endpoint tcp://127.0.0.1:18083 \
  --camera-params data/calibration/latest/third_camera/third_view_extrinsic.json
```

此处参数文件必须属于当前相机。量尺方案可将参数路径换为 `data/calibration/latest/manual/cam_param.json` 或明确指定的有效人工标定结果。`--live`只读相机服务的一帧，之后冻结供检查；不连接机器人、不发送运动。省略 `--camera-params` 使用最近第三相机正式结果。旧人工参数若属于另一台相机，会被serial检查拒绝。
