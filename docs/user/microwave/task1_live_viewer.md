# 完整 Task1 在线纯视觉检查

```bash
conda activate curobo
cd /home/ubuntu/Documents/yixuan/Gene_Agent
python tools/viewers/view_task.py --task microwave_task1 --port 8093
```

默认配置会先把手拉门到 25°，再把场景中的门角直接设为 85°，中间**没有机器人推门轨迹**。旋钮使用 tip X 5°、azimuth 240° 的配置。要检查机器人继续从内侧推门到 90°，启动时添加 `--config data/task_1/visualization/task1_visual_explicit90_verified.json`。

点击“规划所选完整 Task”会从当前场景实际规划七个连续分区：开门、抓碗放入、关门、转旋钮、再次开门、取出放桌面、再次关门。每个分区完成后刷新进度；完成后使用“Task 分区”、阶段选择器和时间轴检查真实轨迹、collision、attachment、门角及旋钮角。失败时保留成功前缀和结构化失败信息。

场景摆位栏支持 bowl、microwave、table 的 XYZ 平移和旋转，以及桌面长宽；也可显示拖动控件。修改后点击“应用摆位并更新抓取目标与碰撞”，再规划完整 Task。Bowl 抓取目标默认随机预览 5 个，可调整数量或换一组；预览不改变规划使用的全部候选。完整 Task 固定从关门初态开始，因此门初角控件不可编辑。

保存的 `scene_config.json` 可用 `--config <路径>` 重新加载。加载完整推门配置后，界面会显示实际规划出的连续轨迹及失败阶段；直接更新门角的配置不能作为推门成功的证据。

这是纯视觉规划，不验证物理接触、夹持或桌面支撑。各 Skill 延续其原有碰撞策略，特别是 Insert 的 world 障碍停用和显式外部门状态更新，均可在阶段信息中检查。
