# 清扫场景的功能 Concept 标注

场景：`data/realworld/brush_cube_dustpan_0`。参数在各物体的 **object frame** 中，长度单位为米；参考 mesh 是场景 `assets/` 中已经规范化的版本。每个 part 只用一个基础几何，供抓取、推扫和接收关系使用，不是 visual 或碰撞网格的精确替代。

| 物体 | Concept 文件 | 功能 part |
|---|---|---|
| 刷子 | [brush.json](../../../data/realworld/brush_cube_dustpan_0/concepts/brush.json) | `grip_handle`: Cylinder；`bristle_face`: Cuboid |
| 绿方块 | [cube.json](../../../data/realworld/brush_cube_dustpan_0/concepts/cube.json) | `body`: 与场景 box 同尺寸的 Cuboid |
| 红、蓝方块 | 同一份 [cube.json](../../../data/realworld/brush_cube_dustpan_0/concepts/cube.json) | 共享绿方块的功能几何，场景中分别有自己的 pose 与颜色 |
| 簸箕 | [dustpan.json](../../../data/realworld/brush_cube_dustpan_0/concepts/dustpan.json) | `grip_handle`: Cuboid；`receiving_tray`: 薄 Cuboid |

主要尺寸和中心如下，数字为毫米，`size` 对 Cylinder 是局部截面两个半径与轴向长度，对 Cuboid 是 XYZ 全长：

| part | size | 中心 `T_object_part` 平移 |
|---|---:|---:|
| 刷子抓取柄 | 8, 7, 75 | −62.5, 22, 69；轴沿物体 X |
| 刷毛接触面 | 90, 42, 20 | 68, 22, 10 |
| 绿方块 | 22.4076, 22.4076, 22.4076 | 0, 0, 11.2038 |
| 簸箕抓取柄 | 70, 26, 12 | −46, 0, 9 |
| 簸箕承接托盘 | 90, 140, 8 | 37, 0, 4 |

刷毛接触面的法向为物体 `−Z`；簸箕铲口朝物体 `+X`，在当前场景摆位中对应世界 `−Y`，朝向三块的工作区。要把 part 放进场景，使用 `T_world_part = T_world_object @ T_object_part`；每块各自的 `T_world_object` 已在 `scene.json`。来源 mesh、哈希和 part 名称见 [manifest.json](../../../data/realworld/brush_cube_dustpan_0/concepts/manifest.json)。

## 检查和修改

在项目根目录，将 `object` 设为 `brush`、`cube` 或 `dustpan`，每次打开一个物体：

```bash
object=dustpan
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python tools/editors/edit_concept.py \
  --mesh "data/realworld/brush_cube_dustpan_0/assets/${object}.obj" \
  --concept "data/realworld/brush_cube_dustpan_0/concepts/${object}.json" \
  --output "data/realworld/brush_cube_dustpan_0/concepts/${object}.tuned.json"
```

正交叠图保存在 `runs/concept_annotation/sweep_scene_20261017/` 的 `brush_overlay.png`、`cube_overlay.png` 和 `dustpan_overlay.png`。参考 mesh 为灰色，功能 part 为彩色。

这次没有调用自动 Agent 后端。自动审批拒绝了将真实 RGB 和 mesh 派生图上传给外部 Codex 后端，因为没有明确的数据上传授权；因此这里根据本地场景 mesh、已查看的 RGB 与正交视图直接做了几何标注。它们只说明功能几何大致位置，刷毛细节与簸箕侧壁未精确拟合；固定名义场景的 MuJoCo 实际任务结果见[清扫 Skill 说明](../sweep/sweep_cube_mujoco_skill.md)。

基于刷子、方块、簸箕三种 Concept 开发抓取、清扫 Skill 与完整仿真 Task 的步骤见开发方案（历史材料已本地归档）。红蓝方块只共享几何 Concept，不新增感知标注。
