# 仿真与真机的共享操作边界

## 当前实际调用关系

2026-10-30：Task1 流程现位于 `tasks/policies/microwave_task/policy.py`，物理装配位于同目录 `simulation.py`；旧路径保留兼容。其他任务与 benchmark 入口见 [总览](../../../benchmarks/index.md)。以下共享操作语义仍有效。

Knowledge 生成几何目标；Skill 决定阶段顺序、前置与后置检查；Task 决定子任务顺序。
执行后端提供实际状态和动作，观测后端提供 GT 或现场感知。它们不重新定义操作顺序。

| 共享入口 | 实际适配 |
|---|---|
| `skills/pick_flow.py` | MuJoCo 抓刷/抓碗、现场抓刷/抓碗、首帧预测预览 |
| `skills/transfer_flow.py` | 微波炉放入与桌面放置的 MuJoCo / 现场适配 |
| `skills/extract_flow.py` | 炉内抓取前缀、检查后接续撤离、现场完整取出 |
| `skills/knob_flow.py` | 旋钮现场物理模式 / MuJoCo 显式混合模式 |
| `skills/door_sequence.py` | 门 Skill 次序、实际门角重估与门角检查 |
| `tasks/policies/microwave_task/policy.py` | MuJoCo `physical_task1` 与现场 `microwave/task` 的完整 Task1 循环 |

这些入口调用 `skills/operation_runner.py::run_operation`（门目前共享的是次序/几何与检查，门适配仍有各自的逐阶段执行会话）。
实际后端通过 `CallbackOperationBackend` 提供 `observe/perform`，返回 `StageOutcome`。
`OperationState` 必须标明版本与来源；不要求不同传感器具有相同物理信息。
现场 Task 交接下层报告引用和终态来源，子任务继续取得新观测并核对 SDK 终态，不能把报告路径当实测关节。

完整 Task 的 Pick 与 Insert 分别由共享 Task 调用，不把放入藏在 Pick 的接续回调里。
独立子任务使用同一组现场函数，`tools/run/` 负责 CLI 装配；正式 MuJoCo 抓碗实现在
`tasks/policies/microwave_task/simulation_bowl/pick.py`，不从诊断脚本导入正式实现。

## 参数与环境策略

共享逻辑不意味着所有参数完全相同。机器人 TCP、场景物体属性、执行速度、接触获取方式、
允许的阶段误差、稳定等待时间，分别由机器人/任务 profile、环境适配与现场配置提供。
已有数值参数本轮保持原值，没有为通过迁移测试调整摩擦、刚度、位姿或碰撞范围。

已知明确差异：

- `MicrowaveTaskPolicy.before_reopen_transition`：现场默认 True；已验收 MuJoCo 混合基线为 False，
  在 Task 报告中记录原因。故当前仿真不能验证现场这条专有的回默认 TCP 轨迹。
- 刷子仿真默认世界 Z 抬升，现场/预览沿 pregrasp 方向抬升；记录最终参数及来源。
- MuJoCo 碗仅在有效双侧指垫接触后启用显式辅助 weld，松爪后释放；现场不启用。
- MuJoCo 旋钮物理夹持检查保持 unknown，另有仅混合模式可用的运动学许可。
  现场物理模式必须通过真实夹持检查，不能使用该许可。

## 检查与异常

`CheckResult` 只有 pass/fail/unknown，并保留 source、evidence 与 reason。
实际阶段的 required check 缺失或 unknown 即停止，不能把仿真/预测来源用作现场成功证据。
现场视觉检查绑定当前观测帧；新观测与 SDK 终态必须连续，过期帧不能让松爪或下一动作通过。

每次现场发令和稳定等待前先把终态标为未知；只有 SDK 测量或成功新观测才能覆盖它。
部分 SDK 数据（例如只有夹爪宽度）保留原数据，并标 `joint_state_unknown`。
Task 不因 terminal 字典存在就把 unavailable 或 partial 状态标为完整已知状态。

阶段中途异常需保存失败阶段、traceback 与仍可取得的实际终态。
MuJoCo 独立旋钮/取出等 Skill 同样保存这些记录；不能只靠外层 Task 捕获异常。
预测预览不向设备发令；现场仅预检首段，不把预测终点传给后续子任务。

## 验收范围

2026-10-29 剩余迁移的固定 seed=123 fresh-reset 完整 Task1 已通过真实 cuRobo 规划及 MuJoCo 执行。
13 个共享 Task 阶段全部通过，旧 11 个物理子记录继续保留；最终门关闭、旋钮45°、碗在桌面、无残余附着。
与上一轮基准的抓碗候选和 pregrasp/approach/lift 三段 `(41,7)` 关节轨迹一致，最大差为0 rad。
这验证该固定样本的迁移回归，不是随机成功率或真机验收，也不是全物理任务成功。
报告与对比：`runs/shared_operation_refactor/20261029/remaining/`。

本轮没有连接相机、SDK、Pika 或机械臂。后续真机试验仍逐项验收感知、标定、夹持和任务后置条件。
