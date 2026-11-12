# 在线检查门 Skill 与门循环 Task

先进入 `curobo` 环境和项目根目录。

单独检查一个 Skill：

```bash
python tools/viewers/view_door_skill.py --skill push_close
```

`--skill` 还可使用 `handle_pull` 或 `push_open`，只决定初始选项。页面顶部的“选择 Skill”可以在三个门 Skill 之间切换，不必重启 Viser；切换后保留场景和任务参数，清除上一 Skill 的轨迹。这个页面只包含选中 Skill 的 Knowledge 目标和执行阶段，不显示 Task 过渡点。

门 Skill/Task 默认启用 CUDA Graph，页面会显示实际 planner 模式。首次遇到一种碰撞策略或候选 batch shape 时需要捕获图，因此第一次规划可能更慢；同一 Viewer 内再次规划会复用图和 planner。诊断 CUDA Graph 本身时可临时运行 `python tools/viewers/view_door_skill.py --planner-mode development`。

检查门循环 Task：

```bash
python tools/viewers/view_door_task.py
```

Task 页面包含拉门、推开、Skill 间过渡和推关。两个入口共享同一个业务会话和 cuRobo 后端，但检查范围固定，避免阶段滑块在单 Skill 与完整 Task 之间错位。

黄色夹爪表示 Knowledge 目标；“规划并播放”生成真实 cuRobo 连续关节轨迹。完整规划按阶段推进，页面会持续显示当前阶段、完成数量和阶段墙钟耗时。规划失败后，顶部会保留：

- 起点严格检查是否通过；
- IK seed 与局部旋转候选的成功数；
- 最近候选的位置和姿态误差；
- 场景碰撞、自碰撞、关节限位和碰撞对象；
- 关闭场景碰撞后是否仍失败；
- 完整记录目录。

“显示场景碰撞几何”在 Knowledge 预览中显示当前目标的碰撞层；规划轨迹播放时读取该阶段实际保存的 `collision_snapshot.json`。页面同时说明 door 是否参与避障以及当前 attachment。门是场景关节物体，不是夹爪 attachment。

拖动微波炉或修改数值会重建 Knowledge、保持当前 Skill/Task 检查范围，并废弃旧轨迹。只有 pre-grasp、pre-contact、撤离和 Task 过渡点可在这里覆盖；接触目标应在 Knowledge 编辑器中调整。保存按钮写入 `data/task_1/skills/door_skill_debug.json`。

推门默认复用旧 Task1 的规则：pre-contact 沿所选门面法向退出 50 mm。首次 contact 保持严格；连续 `push_*` 目标和非接触过渡点允许显式的 20 mm / 10° 有界近似 IK。近似结果仍需通过关节限位、自碰撞和场景碰撞复核，界面会标出实际终点误差。

本页面是运动学检查器：它检查目标、IK、cuRobo 路径和规划碰撞，不步进 MuJoCo 物理，也不证明门已经被接触力带动。

Codex 使用同一逻辑获取结构化反馈：

```bash
python tools/run/run_door_visual.py --skill push_close --plan
python tools/diagnostics/check_door_inspector.py --run runs/door-live-YYYYMMDD-HHMMSS
```

终端只打印简要结论，完整 seed、碰撞和耗时证据保存在 run 内。录制 Knowledge 或成功规划的连续轨迹：

```bash
python tools/viewers/recording/record_door_skill_visual.py \
  --skill push_open --source planned --camera overview \
  --output runs/door-live-visual/push_open.mp4
```
