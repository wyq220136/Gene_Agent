# 快速规划模式

在任务/现场配置 JSON 中加入：

```json
{
  "motion": {
    "backend": "curobo",
    "collision_mode": "robot_only"
  }
}
```

`full` 为默认值。`robot_only` 保留机器人/夹爪自碰撞、关节约束与 TCP 路径要求，跳过桌面和物体等场景障碍，以及抓持物体的规划 attachment。不导出普通场景 mesh 或生成物体附着球；仍保存小型 JSON，记录真实夹爪状态、机器人保护与生效模式。CUDA Graph 和规划器复用继续默认开启。

以下入口支持配置，也支持命令行覆盖 `--collision-mode full|robot_only`：

- `tools/run/preview_sweep_from_camera.py`
- `tools/run/execute_sweep_realworld.py`（自动首帧预览也使用同一模式）
- `tools/run/run_microwave_task1_realworld.py`
- `tools/run/run_microwave_task1_physical.py`

例如，沿用原有清扫预览命令并加上开关：

```bash
python tools/run/preview_sweep_from_camera.py \
  --camera-param data/calibration/latest/third_camera/third_view_extrinsic.json \
  --collision-mode robot_only
```

配置传递只覆盖这次同步调用，结束后恢复默认。使用已有首帧记录执行清扫时，未显式配置模式则继承原预览模式；执行配置和 CLI 可以覆盖它。感知、物体刚性抓持关系和成功判断仍然需要正常输入。

双臂模型会保留本体和冻结另一臂的保护代理。其非 box 接触几何用保守包围盒，可能比原 mesh 更早拒绝目标；实际范围见 `collision_snapshot.json`。当前单臂 Pika 模型无需构建这些额外 world 障碍。

`robot_only` 适合已确认无需场景避障的运动。MuJoCo 接触和任务判定仍正常运行：2026-10-30 的完整微波炉尝试在拉门后的 pre-contact 提前碰门，按原规则失败；该任务应继续使用 `full`。同轮 `full` 从 fresh reset 的 13 个阶段及最终条件通过。不能把无场景规划通过当成整条轨迹没有物体碰撞。

查看当前机器离线规划速度，不连接设备：

```bash
python tools/diagnostics/check_motion.py --collision-mode robot_only \
  --output runs/tmp/motion_check/my_run
```

该诊断独立于完整任务，检查 IK、短轨迹、FK、自碰撞负例及场景忽略正例。报告分别保存首次调用和热态测量。RTX 4090 上本轮固定样本：热 IK 约 7 ms、热短轨迹约 53 ms；Pika 内部工作姿态的热 IK/轨迹/1 cm 直线约 7/58/67 ms。这些是简单样本基线，详见[实现与验收](../../developer/motion/planning_modes.md)。
