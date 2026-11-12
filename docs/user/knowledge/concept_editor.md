# Mesh Concept 标注器

受众：需要对照物体 mesh 人工创建或调节 Concept part 的用户。

状态：已实现首版，只支持 `Cuboid`、`Cylinder` 和 `Ring`。输入 mesh 必须已经位于
object frame，长度单位为米；工具不会自动缩放。

## 启动

在仓库根目录的 `curobo` 环境中，已有标注直接输入物体名：

```bash
python tools/editors/edit_concept.py --object hammer
python tools/editors/edit_concept.py --object spoon_orange
```

省略参数会列出已标注物体并提示输入编号；`--list` 只列清单。物体名来自
`annotation_run.json`，会自动读取其 mesh 与最终 Concept；同名的多轮标注默认选最近
完成的一轮，也可以传完整运行目录名，例如
`--object robotwin_hammer_agent_20261017_v2`。如果已有 `concept.tuned.json`，
会优先加载它；默认仍输出到该文件，保存覆盖须在界面内显式确认。

尚未标注的新 mesh 继续用显式路径：

```bash
python tools/editors/edit_concept.py \
  --mesh assets/objects/bowl_visual.obj \
  --concept assets/objects/bowl_concept.json
```

`--concept` 可省略，此时从一个默认 Cuboid part 开始。`--output` 可指定输出文件；
否则在输入 Concept 旁生成 `.tuned.json`，没有输入 Concept 时在 mesh 旁生成
`*_concept.tuned.json`。

浏览器界面支持：

- 新增、选择、重命名和显式确认后删除 part；
- 用 transform gizmo 直接平移、旋转 `T_object_part`；
- 用数值框精确设置 XYZ、RPY 和几何尺寸；
- 调节 Ring 的 0–360 度 `exist_angle`；
- 按 mesh 的 object-frame 轴对齐包围盒给出简单初值；
- 撤销、重做、聚焦 mesh/part 和安全另存。

已有 JSON 中暂不支持的 part 与额外 metadata 会原样保留，但界面将其标为只读。
默认拒绝覆盖已有输出；需要覆盖时必须在界面中显式勾选。

## 当前限制

- 自动初始化只是轴对齐包围盒初值，不是几何拟合结果。
- 不支持点云、mesh 分割、抓取 profile、IK、碰撞或 MuJoCo。
- RPY 只用于数值编辑显示；保存的事实仍是 4×4 `pose_matrix`。

首版 bowl 实际画面见
[`runs/knowledge/concept_editor_smoke/views/bowl_editor.png`](../../../runs/knowledge/concept_editor_smoke/views/bowl_editor.png)。
