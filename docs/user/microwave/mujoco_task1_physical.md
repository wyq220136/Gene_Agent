# 微波炉 Task1 的 MuJoCo 验收

2026-10-27 用户确认推门时夹爪部分接触门板可以接受。当前 `door_physical_open.json` 因此允许两侧指垫、指杆、滑块及已有指根代理在推开/推关阶段接触门板；pre-contact 和避障接近仍禁止提前碰门，机械臂其他部位及抓碗接触规则不变。该规则更新本身不代表完整任务已经重新验收。

先进入 `curobo` 环境，并在 `Gene_Agent` 项目根目录运行：

```bash
python tools/run/run_microwave_task1_physical.py
```

脚本从新场景依次执行：开门、抓碗并放入、关门、旋转 knob_0、再次开门、取碗放到桌面、最终关门。终端只输出摘要；详细结果在打印的 `summary.json` 及各子任务目录。失败会停止并保留已完成前缀。默认配置在 `data/task_1/simulation/mujoco_task1_physical.json`。

开发碗操作时：共同目标规则在 `gene_agent/tasks/policies/microwave_task/operations/bowl.py`，MuJoCo 抓取/放入/取出/放置在同任务的 `simulation_bowl/`，现场适配在 `gene_agent/realworld/microwave/`。`simulation.py` 管理 Task 会话及交接。失败先看 `summary.json` 的 `failure`，按 `report_ref` 找原操作/阶段，再查看 `traceback_ref`；录像收尾报错也会保留已落盘的原操作失败。旧 run 的记录目录和读取方式保持兼容。

MuJoCo 的自动抓碗使用从碗沿外侧斜向进入的候选，绕实际指垫中心补偿 TCP 位置，并上移接触位置给指杆留净空，也给后续腕部入炉留空间。闭爪逐步减小开口，确认双侧指垫接触后停止继续收紧；此处反馈来自 MuJoCo 实际接触，真机需对应接触/力反馈。Insert 按实际 `T_tcp_bowl` 从直立的碗目标反算 TCP，目标中心位于转盘前半部；preplace 留有炉口高度净空，插入时保持朝向，松爪后先退出手指，再返回已到达过的炉外准备位置。该策略只接入此处 MuJoCo 路线，不改变真机旧标定 keypose。

`pick_insert/physical_insert/targets.json` 保存目标来源、实际夹持变换、抓取轴夹角及新旧 TCP 对照。`plans/*.json` 的 `pre_execution_check` 检查持碗倾角和实际碰撞几何；`executions/*.json` 的 `failure_evidence.forbidden_contacts` 给出接触双方及侵入深度。炉体和门始终参与 cuRobo 避碰，退出手指时仅允许指垫接触碗；15° 倾角、最终炉腔占用、转盘支撑及撤离后碗位移检查仍保留。局部回归通过不等于完整 Task 或新位置分布已经验收。

## 独立运行子任务

以下命令从 fresh reset 开始，执行原 Task 中必要的前缀，完成所选子任务后停止。`--subtask` 支持 `open_door`、`close_door`、`rotate_knob`、`pick_bowl`、`insert_bowl`、`extract_bowl`、`place_bowl_on_table`、`load_bowl`、`unload_bowl`；省略则执行完整任务。

单项 Skill 可进一步指定 `--stop-after`：例如 `--subtask insert_bowl --stop-after release` 在释放、解除辅助固定和落稳检查后停止，不执行撤离；`--subtask rotate_knob --preparation current --stop-after grasp` 停在夹持检查后。支持范围见 [Skill 入口](../../../gene_agent/skills/index.md)。正常暂停为 `status=partial`，只表示所选阶段范围成功；保存 actual_final 与 debug_stop_state.npz，完整目标未评分。该选项不能配合 all、load_bowl 等组合，避免不完整结果继续传给下一操作。

```bash
python tools/run/run_microwave_task1_physical.py --subtask load_bowl \
  --output runs/tmp/microwave-review/load-01 --video
python tools/run/run_microwave_task1_physical.py --subtask unload_bowl \
  --output runs/tmp/microwave-review/unload-01 --video
python tools/run/run_microwave_task1_physical.py --subtask rotate_knob --preparation current \
  --output runs/tmp/microwave-review/knob-01 --video
```

`load_bowl` 是 Pick→Insert（含释放/撤离），`unload_bowl` 是 Extract→桌面 Place。开关门及原 Task 过渡保持原顺序。旋钮 `current` 示例从初始关门状态直接运行，不先放碗；仍是原有运动学提交旋钮角的混合仿真。

`prefix` 是默认准备模式；`current` 只执行所选步骤，不补前序动作。CLI 每次新建场景，因此用它直接 `current insert_bowl` 会缺少持物证据而拒绝。若要在同一进程中单独调试 Insert，可复用会话：

```python
from pathlib import Path
from gene_agent.tasks.policies.microwave_task.simulation import MicrowaveSimulationSession

session = MicrowaveSimulationSession(Path("runs/tmp/microwave-review/insert-01"),
    config=Path("data/task_1/simulation/mujoco_task1_physical.json"))
try:
    picked = session.run(selection="pick_bowl")  # 先执行开门和原过渡
    if picked["operation_success"]:
        inserted = session.run(selection="insert_bowl", preparation="current")
finally:
    session.close()
```

不 reset、不读取任意旧 run 冒充当前持物。后续操作会核对实际状态及模型指纹，并沿用原操作的接触检查。同一会话禁止重复覆盖同名步骤；失败可新建场景复现，本轮未扩展 Task checkpoint。

`summary.json` 是统一摘要：`requested_scope`、`operation_success` 表示本次范围；局部运行的 `goal_success/task_success` 为 null。`failure` 给出失败步骤、原阶段/分类、原始报告与 traceback 路径；`actual_final` 保留真实终态。每次会话调用另存 `calls/<序号_选择>/subtask.json`。

`timing` 分开记录前缀/目标的墙钟耗时和实际仿真时长；它们不是纯规划/执行墙钟。门阶段可查看原 `planning.timing`，其他操作尚未全量采集纯规划墙钟，因此摘要不捏造该数字。

`--video` 使用同一实际回放导出 `video/task.mp4`，局部运行默认只显示所选操作；前缀保留在原始记录中。检查放入/取出的实际结果：

```bash
python tools/viewers/view_task.py --task microwave_object_insert --run <放入运行目录> --mode actual
python tools/viewers/view_task.py --task microwave_object_extract --run <取出运行目录> --mode actual
```

独立真机入口及人工准备方式见 [真机子任务](realworld_microwave_entry.md)。仿真成功不代替硬件验收。

## 完整任务对照与证据

2026-10-29 性能对照：同一 seed=123 完整混合仿真从 **343.91 秒降至139.06秒**，13个共享Task阶段及最终状态检查均通过。门配置已启用 `reuse_gripper_geometry`：夹爪开口变化时精确更新碰撞球、复用求解器，不改原候选顺序、碰撞策略或直线要求。单次对照及失败方案见 [性能记录](../../developer/motion/microwave_planning_performance.md)。此处耗时包含规划和仿真执行，不含视频导出。

推荐在运行命令后加 `--video`，结束后统一导出 `video/task.mp4`；已有记录使用 [统一视频入口](../visualization/video_feedback.md)。

需要旧版全景/近景分文件及七段子任务时：

```bash
python tools/viewers/recording/record_physical_task1_video.py <运行目录>
python tools/viewers/recording/split_physical_task1_video.py <运行目录>
```

全景、接触细节分别为 `task_overview.mp4`、`task_detail.mp4`；短片在 `clips/`。视频使用 MuJoCo 实际状态采样，不使用规划动画。`push_motion_audit` 分别记录规划直线度和实际 TCP 偏离。

检查某次已保存的实际 Task1，不重新运行任务：

```bash
python tools/viewers/view_task.py --task microwave_task1 --run <运行目录> --mode actual
python tools/viewers/view_task.py --task microwave_task1 --run <运行目录> --mode actual --play
python tools/viewers/view_skill.py --skill insert_bowl --run <运行目录> --mode actual
python tools/viewers/view_skill.py --skill place_bowl --run <运行目录>/table_place --mode actual
```

前两条分别打开 Viser 和本地 MuJoCo，只读同一组实际采样；`--stage`、`--time`、`--output` 可用于阶段检查和截图。没有保存完整中间采样时只显示已有实际边界；混合仿真证据等级保持原报告的声明。参数与限制见 [Skill 检查说明](../visualization/skill_viser_inspection.md)。

如需在线修改摆位、重规划并检查原有视觉 Skill/Task，仍用成熟的 Viser 入口：

```bash
python tools/viewers/view_task.py --task microwave_task1
```

当前 MuJoCo 完整任务是**混合仿真验收**：门由实际接触带动；碗仅在有效双侧指垫接触后启用仿真辅助夹持，并在松手时释放；旋钮机器人轨迹实际执行，但旋钮关节沿用原 Viser 的运动学更新。不能把这次结果称为全物理夹持或真机成功。
