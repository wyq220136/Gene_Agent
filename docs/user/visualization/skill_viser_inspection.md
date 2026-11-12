# 在 Viser 中单独检查 Skill

Pick 可在实际执行后停在指定阶段，再用本页的同一回放入口检查：

```bash
python tools/run/run_spoon_pick.py --stop-after grasp --output runs/tmp/skill-review/spoon-grasp
python tools/viewers/view_skill.py --run runs/tmp/skill-review/spoon-grasp
```

`--stop-after` 支持 pregrasp、approach、grasp；省略为完整 Pick。grasp 停在闭爪检查，不激活辅助夹持、不抬升。它是局部结果，现场会保留，不自动松爪；实际成功与失败均查看 `pick/result.json` 的 requested_scope、shared_flow 和 actual_final。更多场景装配与能力边界见 [Skill 入口](../../../gene_agent/skills/index.md)。

其他能力也从同一入口文档选择：Transfer 支持独立放入／取出／放置，Door 沿用 `--until-stage`，Knob 可从当前场景单独运行；Sweep 的 `--skill sweep --target cube --stop-after lower_to_contact` 会先抓刷，再停于指定清扫阶段。准备前缀仍须实际执行；这不是跨进程恢复，也不报告完整 Task 成功。各入口均写新 run，保持原记录不变。

微波炉也支持阶段停止，例如 `python tools/run/run_microwave_task1_physical.py --subtask insert_bowl --stop-after release --output runs/tmp/skill-review/insert-release`。正常暂停显示 partial，表示所选前缀完成，不是执行失败；完整 Skill 判据尚未完成，task_success 保持 null。保存的 actual_final 和 debug_stop_state.npz 对应暂停现场，回放仍用本页相同命令。

已有微波炉实际运行记录可直接用统一入口检查：

```bash
python tools/viewers/view_task.py --task microwave_task1 --run runs/microwave-task1-rerun-gpu-20261027 --mode actual
python tools/viewers/view_skill.py --skill insert_bowl --run runs/microwave-task1-rerun-gpu-20261027 --mode actual
python tools/viewers/view_skill.py --skill place_bowl --run runs/microwave-task1-rerun-gpu-20261027/table_place --play
```

带 `--run` 的微波炉入口只读现有记录，在 Viser 中选择阶段、播放或逐采样检查；`--play` 打开本地 MuJoCo 窗口。截图可加 `--output /tmp/task.png --camera task --time 20`，同时保存来源 JSON；无显示器时可设 `MUJOCO_GL=osmesa` 使用软件渲染。`--help` 列出全部参数。实际记录默认 `actual`；独立规划记录默认 `planned`，也可显式指定 `--mode`。

实际回放使用 Runtime 保存的完整 `qpos` 和仿真时间；部分旧 release/settle 只有 NPZ 实际状态边界，界面明确标注并停在最近的已有样本，不能据此查看未保存的中间接触。失败前缀也能检查。实际记录关闭场景编辑与重新规划；不带 `--run` 的入口保留在线调参。混合仿真中的 weld 和旋钮运动学提交仍以原报告为准，回放画面不提高证据等级。旧原始记录尚未关联的逐阶段规划碰撞快照会显示“未记录”。

真机微波炉报告沿用相同命令，例如 `python tools/viewers/view_skill.py --skill insert_bowl --run <现场运行目录>`。界面自动切为“静态观测 + 实测端点/unknown 证据”：显示保存的报告、检查来源和规划目标，保留 SDK 实测端点及未知终态；不连接设备，不插值成执行轨迹。若同时保存同源 `observed_scene_provenance.json`、`collision/collision_snapshot.json` 中的完整 `qpos_snapshot` 和 `scene/scene.xml`，可切换静态观测帧；缺少这些文件时仍可打开报告面板，并列出缺项。`--output` 仅对已有完整静态场景截图；真机报告不支持 `--mode planned`、`--play`、`--stage` 或 `--time`，不会用旧观测伪造后续执行状态。

受众：人工验收者。独立 Viser Skill 检查界面已可用，已用同步抓碗 run 验证。后续顺序见下一批计划（历史材料已本地归档）。

这个界面读取已保存的 Skill run，可以看完整 Skill，也可以只看一个阶段的目标 pose、路径、实际执行和规划碰撞资产。碗、微波炉和桌面支持拖动或输入 pose 作**画面预览**；画面里的 visual 和碰撞代理会一起移动，但机器人和轨迹仍来自原 Skill，旧规划快照文件不变。点击“应用位姿并重新规划 Skill”后，界面会以当前 run 的首帧为基线生成新场景和碰撞快照；抓碗重新规划 `open→pregrasp→approach→close→lift→hold`，放碗先重新规划抓碗，再规划 `preplace→insert→release→detach→retreat`，随后用**同一套页面控件**重新打开新 run。旧 run 保留；新 run 只有规划，尚未进行 MuJoCo 物理执行。

重新打开后，仍用原来的“检查范围”“查看数据”和时间轴逐阶段查看新轨迹、碰撞快照与失败诊断；规划未成功的阶段也会如实显示。需要复查旧运行时，可按终端打印的旧目录另开检查器；当前浏览器不叠加第二套控件或特殊模式。新 run 的 `lift/hold` 需要实际闭爪后的状态，所以规划报告会标为未执行，不能据此判断物理抓取成功。

阶段摘要中的“无需路径规划”用于开爪、闭爪和保持等命令；“规划成功”表示该阶段已有 cuRobo 轨迹；“已执行”表示 MuJoCo 运行过该阶段。规划和执行分别显示，失败阶段仍能单独查看。

“查看数据”里的“规划轨迹”是 cuRobo 保存的目标运动，“仿真实际执行”是 MuJoCo 采集的状态。回放时间从所选 Skill 或阶段的起点算起；“仿真时刻”是 MuJoCo 场景重置后的累计时间，因此两者数值通常不同。夹爪开口显示为毫米，时间保留三位小数；完整 state ID 供精确检索日志。

“显示 visual 外观”和“显示规划碰撞层”可分别开关；碰撞层可按“碗、桌面、微波炉”等对象组单独查看。界面上方的“当前场景约束”常显本阶段碰撞对象组、活动机械臂碰撞球数量与附着代理。碰撞层取自当前阶段的规划快照。`open/close/hold` 没有独立规划快照：从 `open` 打开碰撞开关时，界面会跳到第一个有快照的 `pregrasp` 并标明来源。附着代理表示规划器随 TCP 搬运的物体；物理夹持是否成功仍看 MuJoCo 结果。碰撞代理是规划用的几何，不等同于物理接触几何。

只看碰撞代理可在界面中取消“显示 visual 外观”，或运行 `python tools/viewers/view_skill_run.py --show-collision --hide-visual`。已查看的[纯碰撞层截图](../../../runs/skill-inspector-ui-fix-20261007/collision_only/01_pregrasp_actual_0.000.png)对应 `pregrasp` 规划快照，旁边的 [JSON 清单](../../../runs/skill-inspector-ui-fix-20261007/collision_only/capture_manifest.json)注明了实际显示阶段与图层开关。

**现在可直接检查**：在 `curobo` 环境执行：

```bash
python tools/viewers/view_skill_run.py
```

默认自动打开最近一次完整、物理抓取成功且含模型的抓碗 Skill；终端会打印实际选中的目录。打开本地网址，选择“完整 Skill”或阶段，拖动时间条、播放或逐采样前进。物体编辑区初始折叠；在“拖动预览物体”中选物体后出现三维拖动控件，也可展开物体输入 xyz 与 roll/pitch/yaw，点击“清除位姿预览”恢复记录画面。未执行阶段会自动切到规划视图。旧 run 的中间点可能只有部分反馈，界面会明确标注；同步 run 使用完整 `qpos` 和物体位姿。

检查放碗 Skill 可运行 `python tools/viewers/view_skill_run.py --skill insert`；它自动选最近一次成功的放碗**规划记录**。当前可复查的 X 方向偏移 15 mm 样本位于 `runs/affordtoolact_migration/insert_bowl_x_plus_15mm_probe/`，同目录的 `insert_bowl_contact_side.mp4` 是炉口侧视的规划运动学视频。该视频不证明 MuJoCo 接触或物理放置成功。

固定阶段与时间可用如下命令直接打开：

```bash
python tools/viewers/view_skill_run.py --stage 04_lift --mode actual --time 0.5 --show-collision
```

自动截图和 JSON 清单可用 `tools/viewers/capture/capture_skill_run.py` 的相同 `--run/--stage/--mode/--time` 参数，加上 `--output` 目录。清单包含截图哈希、stage ID、采样时间、state ID、`qpos` 哈希、物体 pose、attachment 和碰撞快照哈希。本次已查看的 [完整 Skill 截图](../../../runs/skill-inspector-evidence-20261007/synced_full/all_actual_3.500.png)、[闭爪截图](../../../runs/skill-inspector-evidence-20261007/synced_close/03_close_actual_0.500.png)、[抬升与碰撞截图](../../../runs/skill-inspector-evidence-20261007/synced_lift/04_lift_actual_0.500.png) 和 [后端不支持阶段截图](../../../runs/skill-inspector-evidence-20261007/failed/02_approach_planned_0.000.png) 可直接复查。

本次重新验收把碗沿 x 移动 0.04 m；[重新打开后的原版页面截图](../../../runs/skill-inspector-reload-verified-20261007/capture/01_pregrasp_planned_0.400.png)与[新 Skill 规划报告](../../../runs/skill-inspector-reload-verified-20261007/revision_002/skill_pick_bowl_result.json)对应。新 run 的 `pregrasp` 与直线 `approach` 均规划成功；它们的目标 TCP x 约为 0.3661 m。此截图仅验证规划与显示，不验证物理抓取。

同帧数值验收命令：

```bash
conda run -n curobo python tools/diagnostics/check_skill_view_consistency.py --run runs/pick-bowl-synced-20261007 --stage 04_lift --time 0.5
```

它分别让 Viser 和 MuJoCo viewer 恢复该帧，比较完整 `qpos`、碗 pose、attachment 和碰撞快照引用；本次闭爪、抬升和完整时间轴的比较均通过，数值结果在 `runs/skill-inspector-evidence-20261007/synced_*_compare.json`。回放只读现有 run；只有显式点击“应用位姿并重新规划 Skill”才会新建规划 revision。

开发接口、记录格式和自动验收规则见 [详细设计](../../developer/visualization/skill_viser_inspection_design.md)。

旧刷共享操作记录的参数来源与检查证据说明见 [共享操作流程](shared_operation_workflow.md)。`view_skill.py`/`view_task.py` 入口保持原样，新 run 可显示共享抓取阶段。
