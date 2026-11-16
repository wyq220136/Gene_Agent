# 真机微波炉任务入口

完整流程见[真机操作指南](realworld_task1_operation_guide.md)。现场提供标定结果、感知服务和设备地址；程序自动装配配置，**不需要手写站点或角度状态 JSON**。操作 Skill、Knowledge 和 keypose 算法沿用原实现。

## 准备感知

先启动相机、SAM3 和 FoundationPose 服务，并完成右臂 SDK base 下的相机标定：

```bash
python tools/run/prepare_microwave_realworld.py \
  --camera-params data/calibration/latest/manual/cam_param.json \
  --output runs/realworld/perception_current
```

确认关门、旋钮零位、柜体和相机固定，再按提示核对 mask 和桌面结果。程序生成 `perception_setup.json`，该文件用于复用感知结果，无需编辑。正式 CLI 不读取旧 `realworld_site.json`，也不再提供 `--config` 参数。

## 检查和运行

```bash
python tools/run/run_microwave_task1_realworld.py \
  --perception runs/realworld/perception_current \
  --left-ip 169.254.128.18 --right-ip 169.254.128.19 \
  --subtask pick_bowl --check-config
```

按现场填写 IP。端口默认 8080，可用 `--left-port`、`--right-port` 覆盖。
去掉 `--check-config` 后是读取真实状态的 plan-only；显式添加 `--execute --yes` 才允许运动。运行时用 `--output` 指定证据目录。

| 子任务 | 说明 |
|---|---|
| `open_door` / `close_door` | 已有拉门、推门 Skill，使用新帧 tracking 测角 |
| `pick_bowl` | 从桌面抓碗 |
| `insert_bowl` | 独立运行时用 `--prepare-held-bowl` 确认已持物 |
| `rotate_knob` | 需单独完成旋钮接触验收 |
| `extract_bowl` | 从炉内取碗 |
| `place_bowl_on_table` | 独立运行时用 `--prepare-held-bowl` |
| `load_bowl` | Pick → Insert，内部传递持物状态 |
| `unload_bowl` | Extract → Table Place |
| `all` | 完整 Task1，可用 `--no-knob` 跳过旋钮动作 |

`--no-knob` 用于 `all`，仍保留旋钮感知以核对柜体参考。当前使用右臂 SDK base、标准夹爪和对应任务资产；两臂 IP 不能相同。

## 实时状态与边界

每次观测使用同一新 RGB-D 帧跟踪碗、把手和旋钮，再计算当前角度。不再依赖人工更新门角文件；真实过期帧仍会被默认 5 秒时效检查拒绝。
柜体必须固定，几何一致性检查不等于独立柜体测量。Pose 服务重启、输入改变、柜体或相机移动后需重新准备，必要时重新标定。

桌面 pose 与可见 XY 范围来自 RGB-D；桌板厚度和支撑尺寸仍来自任务资产，需要现场核对。输入校验和 plan-only 通过均不代表真实接触、整条任务通过。
FoundationPose 可继续使用 Python 3.9；依赖环境保持独立，详见操作指南。

执行目录中的 `runtime_input.json` 保存自动配置及来源。任务接线位于 `gene_agent/realworld/microwave/task.py`；模块边界见[真机模块说明](../../../gene_agent/realworld/README.md)。
