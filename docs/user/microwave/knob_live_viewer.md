# 旋钮在线检查

在 Gene_Agent 目录、curobo 环境运行：

```bash
python tools/viewers/view_skill.py --skill rotate_knob --port 8092
python tools/viewers/view_task.py --task microwave_knob_rotate --port 8093
```

打开终端中的本地网址。默认使用 `tip_x_5_azimuth_240` 配置，从机器人 home、关门、旋钮零角初态重新规划。黄色夹爪是当前 GT 的 Knowledge 目标，蓝线来自实际规划轨迹。

移动或旋转微波炉后，点击「应用摆位并重建 GT / Knowledge / collision」，再点击规划。数值输入和拖动均可用；旧轨迹会失效。时间轴可查看每个阶段的碰撞范围、快照、耗时、夹爪状态及旋钮角。

「旋钮候选配置」可切换 `default`，保留原始六候选配置。当前摆位的成功与失败均重新求解；失败诊断保留 IK 可达数、候选总数和原始报告路径。

旋钮角遵循 adapter 的离散提交：机器人旋转期间不伪造旋钮连续转动；最后的 `visual_angle_commit` 明确标出。这是纯视觉规划检查，未验证指尖接触动力学。

真实 GPU 与浏览器验收脚本：

```bash
python tools/diagnostics/verify_knob_online.py --output /tmp/knob_online_new --port 8130
```

必须使用新目录；脚本依次实际规划单 Skill、编辑后的子任务和原六候选基线，并保存截图与 `probe_summary.json`。

已验证的默认预设保存在 `data/task_1/visualization/knob_visual_tip5_verified.json`，可显式传入 `--config` 复现。该预设与不传配置时的默认值相同。
