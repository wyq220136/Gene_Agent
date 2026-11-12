# 在线抓碗与放入 Skill

在 `curobo` 环境、Gene_Agent 项目根目录运行：

```bash
conda activate curobo
PYTHONPATH=. python tools/viewers/view_skill.py --skill pick_bowl --port 8092
PYTHONPATH=. python tools/viewers/view_skill.py --skill insert_bowl --port 8092
```

打开终端打印的本地网址。默认使用 `data/task_1/visualization/bowl_visual_debug.json`，门初态为 85°，碗/微波炉摆位来自已验证的纯视觉配置。

1. 在“场景摆位草稿”输入碗、微波炉、桌面的位置和 XYZ 姿态，可修改桌面长宽或开启拖动与旋转控件，点击“应用摆位并重建”。这会更新 GT、Knowledge 候选及碰撞快照，并使旧轨迹失效。
2. 点击“规划所选 Skill”。同步规划按 Pick/Insert 两个步骤推进，单次 cuRobo 调用结束后才会响应取消；放入会先重新规划同场景 Pick，使用其实际规划终态与 attachment 作为合法初态；前置失败时不会伪造放入动作。
3. 使用时间轴、阶段列表和相机查看真实 cuRobo 轨迹。界面显示 attachment、各段规划耗时与碰撞范围；失败信息独立于回放。
4. 保存后的配置可通过 `--config runs/<本次目录>/scene_config.json` 重开。重开只重建场景，点击规划才使用 GPU。

黄色夹爪为 Knowledge 几何目标，不代表可达或无碰撞。这里没有 MuJoCo 动力学验证；放入保留旧适配器的分阶段碰撞策略，部分直线阶段停用 world 障碍，界面会明确标出，不能把该规划成功解释成全程严格避障或物理放置成功。

当前此入口只接入 `pick_bowl`、`insert_bowl` 和原有门 Skill；extract/place/knob 尚未接入在线菜单。

## 完整抓碗 → 放入 Task

```bash
conda activate curobo
cd /home/ubuntu/Documents/yixuan/Gene_Agent
python tools/viewers/view_task.py --task microwave_object_insert --port 8093
```

该入口一次规划并展示 Pick → Insert 的全部 12 个阶段，阶段名保留所属 Skill。
拖动 bowl、microwave、table 或编辑数值后，点击“应用摆位并重建 GT / Knowledge / collision”，再规划完整 Task。
同一条时间轴包含抓取、附着、携带、释放和撤离；失败时保留已完成的前缀及真实失败诊断。
“夹爪侧面”在放入阶段改为从炉口看接触，避免打开的门板遮住碗与指尖。

实测截图：`runs/affordtoolact_migration/object_task_online_final/task_after_edit_contact.png`。
本 Task 仍是纯视觉规划，Insert 延续原 `ignore_scene_collision=True` 策略，界面会显示该阶段实际排除项。
