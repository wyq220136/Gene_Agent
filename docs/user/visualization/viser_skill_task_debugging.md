# 在线视觉检查：Skill 与 Task

在 `curobo` 环境、项目根目录运行以下命令，打开终端打印的本地网址：

```bash
python tools/viewers/view_skill.py --skill pick_bowl
python tools/viewers/view_task.py --task microwave_object_insert
python tools/viewers/view_task.py --task microwave_task1 \
  --config data/task_1/visualization/task1_visual_explicit90_verified.json
```

两个入口顶部都有“全局选择”下拉框。Skill 可在把手拉门、内侧推开、外侧推关、抓碗、放入、取出、桌面放置和旋钮之间切换；Task 可在开门、关门、开关门、放入、取出、旋钮和完整 Task1 之间切换。同类切换保留**已应用**的场景，跨类别切换加载目标类别的默认配置；位姿草稿未应用时不能切换。`--config` 只作用于启动时选择的类别。需要留存摆位时，先在当前界面保存配置。两种界面都直接规划当前场景，`view_skill_run.py` 才是历史记录回放器。

物体类界面默认显示碗、微波炉和桌面的鼠标拖动/旋转轴，也可在折叠区输入位置和角度；门界面可拖动微波炉。先点击“应用摆位并更新抓取目标与碰撞”，再规划所选 Skill/Task；未应用的拖动只是草稿。Bowl Task 默认随机预览 5 个抓取姿态，可调整数量或换一组；预览不改变实际规划的候选全集。完整 Task 固定从关门初态开始，推门配置可在“门操作目标角”里调整目标角。Task 页面默认开启“逐分区快速预览”：每完成一个分区就播放其实际规划轨迹，再规划下一个分区；单个 Skill 仍由其原有规划器一次规划。快速预览让首个动作更早可见，但会增加完整任务的总墙钟时间，可关闭以尽快得到全套轨迹。页面显示累计规划计算耗时，以及各阶段已记录的求解耗时；两者差额包含候选筛选、场景准备和结果记录。还可检查连续轨迹、碰撞物体组、允许接触、attach 状态和失败原因。“夹爪侧面”适合检查抓取与接触。

开到 90° 的门子任务使用 `data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json`；旋钮单项使用 `data/task_1/visualization/knob_visual_tip5_verified.json`。`archive/task_1/configs/task1_visual_legacy85_verified.json` 在拉门后直接更新门角到 85°，没有内侧推门轨迹；需要检查推门时请加载 `data/task_1/visualization/task1_visual_explicit90_verified.json`。

录制当前在线规划结果：

```bash
python tools/viewers/recording/record_visual.py --kind skill --name rotate_knob --camera gripper_side
python tools/viewers/recording/record_visual.py --all --fps 6
```

视频与同名 JSON 清单位于 `artifacts/viser_acceptance/`。它们证明的是 GT 场景中的运动学规划与视觉状态，不证明 MuJoCo 门接触、夹持、旋钮物理作用或真机成功。规划失败时保留原始诊断和已完成的真实阶段，不会用关键 pose 插值补出假轨迹。

批量录像中断后可加 `--resume`。程序会核对有效配置、运行代码与 Task JSON/资产摘要，并实际解码既有 MP4；不匹配的项会重录。完成后运行 `python tools/diagnostics/check_visual_acceptance.py`，核对 23 个视频的实际解码帧数与清单。
