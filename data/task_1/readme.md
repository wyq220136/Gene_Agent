# task_1：运行配置怎么选

分类与逐文件消费者见 [index.md](index.md)。21 份 JSON 是不同入口共享或专用的运行输入，不是 21 份重复任务配置。

| 要做的事 | 主输入 |
|---|---|
| 完整 MuJoCo Task1 | [simulation/mujoco_task1_physical.json](simulation/mujoco_task1_physical.json) |
| MuJoCo 门 Skill/循环 | [simulation/door_physical_open.json](simulation/door_physical_open.json) |
| Viser 碗 Skill/子任务 | [visualization/bowl_visual_debug.json](visualization/bowl_visual_debug.json) |
| Viser 完整 Task1 | [visualization/task1_visual_explicit90_verified.json](visualization/task1_visual_explicit90_verified.json) |
| Knowledge 调参 | [knowledge/microwave_knowledge_view.json](knowledge/microwave_knowledge_view.json) 与同层抓取 profile |
| 真机站点配置 | [realworld/realworld_site.example.json](realworld/realworld_site.example.json)，另填本机副本；运动需单独授权 |

在仓库根目录的 cuRobo 环境运行：

```bash
python tools/run/run_microwave_task1_physical.py
python tools/viewers/view_skill.py --skill pick_bowl
python tools/viewers/view_task.py --task microwave_task1
```

默认入口已读取新的子目录，无需在顶层保留转发配置。显式指定 --config 时使用分类后的路径。

## 生效顺序与证据边界

- `config/scenes/task_1.json` 和 assets 提供名义模型与布局；选中的运行配置再覆盖场景和阶段参数。
- Knowledge 给操作几何/候选；skills 保存阶段控制基础参数；simulation/visualization/realworld 分别装配后端专用输入，仍有跨分类共享，不宣称参数已完全解耦。
- MuJoCo Task1 配置引用门模板，生成本轮专用副本；不回写源文件。Viser 编辑后的实际配置和覆盖值保存在本轮 run。
- 单位以 `_m`、`_rad`、`_deg` 等字段为准。原始 pose/相机/RGB-D 不会被当前仿真入口自动重估。
- 当前完整 MuJoCo 任务为已声明的辅助夹持与运动学旋钮混合仿真；Viser 规划成功不等于物理或真机成功。
- `source_reference` 等来源说明不等于运行依赖；历史运行记录保持原样，不为新目录批量改写。

临时实验统一放 `runs/tmp/`；已淘汰的旧配置保存在本机 `archive/task_1/configs/`，不跟踪。已采用参数必须登记实际消费者后才提升为正式输入。
