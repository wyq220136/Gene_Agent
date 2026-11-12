# 检查微波炉 Concept 与 URDF mesh 是否对齐

此页是[通用 Concept ↔ mesh 检查器](concept_mesh_inspection.md)的微波炉场景示例；微波炉命令只提取 URDF visual 和多套 Concept，显示界面与其他物体共用。

在项目根目录、`curobo` 环境启动只读 Viser 页面：

```bash
cd /home/ubuntu/Documents/yixuan/Gene_Agent
conda run --no-capture-output -n curobo env PYTHONPATH=. \
  python -u tools/viewers/inspect_microwave_concepts.py --port 8084
```

打开 `http://127.0.0.1:8084`。侧栏可切换把手、门板、炉腔、上旋钮和下旋钮；各图层可以单独开关，并有正面、侧面、俯视及“只聚焦实物”视角。**灰色**是校准 URDF 导入 MuJoCo 后的实际 visual mesh；**红色**是原始旧 `microwave_concept.json`；**绿色**是校准 Concept JSON；**蓝色**是当前 Knowledge 运行时实际用的几何或门板接触代理。所有 mesh 和 Concept 均在微波炉根部的 object frame、关闭门及零旋钮角度下显示。页面不会修改任何资产。

当前数值检查结果如下。数字是 **AABB 中心的距离**，仅供快速发现坐标或尺度问题，不是表面精确拟合误差：

| 部件 | 旧 JSON 原样叠加 | 校准 JSON | 运行时几何 |
| --- | ---: | ---: | ---: |
| 把手 | 231.9 mm | 0.2 mm | 5.8 mm |
| 门板 | 261.1 mm | 无独立条目 | 2.2 mm |
| `knob_0` | 329.6 mm | 2.7 mm | 2.7 mm |
| `knob_1` | 266.3 mm | 2.7 mm | 当前任务未使用 |
| 炉腔 | 不计算 | 无独立条目 | 当前任务未使用 |

这说明**旧 JSON 的原始 frame 不能直接与当前校准 URDF 相乘**，但不等于旧版参数本身错了：旧 JSON 来自早期重建资产，尚未给出可信的旧物体 frame 到当前校准 frame 的配准。校准版把手、旋钮的中心与 visual 接近；尺寸差仍需结合实际图形判断，校准旋钮来源是 collision envelope，本就不要求完全贴合 visual。当前运行时把手沿用旧 Trifold 的尺寸、将其 pose 重锚定到 URDF 把手 visual，因此与校准 JSON 不完全相同。推门不直接使用旧门 Concept，而使用校准 URDF 的门板碰撞几何和显式门轴；蓝色门板是接触代理，不是 visual 的复制。炉腔没有一组独立可对应的 URDF visual，故不计算无意义的全机壳包围盒差。

已检查的截图：[把手](../../../runs/knowledge_tuner/microwave_concept_alignment/views/handle.png)、[门板](../../../runs/knowledge_tuner/microwave_concept_alignment/views/door.png)、[炉腔](../../../runs/knowledge_tuner/microwave_concept_alignment/views/cavity.png)、[上旋钮](../../../runs/knowledge_tuner/microwave_concept_alignment/views/knob_0.png)、[下旋钮](../../../runs/knowledge_tuner/microwave_concept_alignment/views/knob_1.png)。[JSON 数值报告](../../../runs/knowledge_tuner/microwave_concept_alignment/alignment.json)保存每个部件的 AABB 中心、尺寸和差值；[截图清单](../../../runs/knowledge_tuner/microwave_concept_alignment/views/capture_manifest.json)保存视角、场景快照哈希及 PNG 哈希。

只需要数字或想重新生成五张截图时：

```bash
conda run --no-capture-output -n curobo env PYTHONPATH=. \
  python -u tools/viewers/inspect_microwave_concepts.py --inspect-only

conda run --no-capture-output -n curobo env PYTHONPATH=. \
  python -u tools/viewers/inspect_microwave_concepts.py \
  --report runs/knowledge_tuner/microwave_concept_alignment/alignment.json \
  --capture-dir runs/knowledge_tuner/microwave_concept_alignment/views
```

现有的 `tools/editors/edit_concept.py` 适合**修改**单个 Cuboid/Cylinder/Ring Concept；它目前不支持 Trifold 把手或完整 URDF 装配。本检查器专门解决“现在使用的微波炉几何与 mesh 差多少、差在哪里”，保持只读，避免误保存旧 frame 数据。
