# 通用 Concept 与物体 mesh 检查

在项目根目录运行，参考 mesh 与 Concept 的 `pose_matrix` 必须使用**同一个物体局部坐标系**，单位为米：

```bash
conda run --no-capture-output -n curobo env PYTHONPATH=. \
  python -u tools/viewers/inspect_concepts.py \
  --mesh assets/objects/bowl_visual.obj \
  --concept assets/objects/bowl_concept.json --port 8084
```

打开 `http://127.0.0.1:8084`。灰色为物体 mesh，彩色线框为 Concept；可选部件、开关图层、切换正面/侧面/俯视。支持当前 Knowledge 的五种 typed Concept：Cuboid、Cylinder、Ring、Trifold 和 Container。界面只读，不会改动 JSON。

截图和机器可读记录：

```bash
conda run --no-capture-output -n curobo env PYTHONPATH=. \
  python -u tools/viewers/inspect_concepts.py \
  --mesh assets/objects/bowl_visual.obj \
  --concept assets/objects/bowl_concept.json \
  --report /tmp/bowl_concept_report.json \
  --capture-dir /tmp/bowl_concept_views
```

`capture_manifest.json` 记录每个部件的 PNG 哈希、视角、图层及来源路径。仅给出**整个物体 mesh** 时，局部 Concept 与完整 mesh 不是一对一部件，不计算误导性的中心差；重点看叠加图。若有逐部件参考 mesh，可由场景适配器构建 `ComparisonData`，这种情况下才报告同一 object frame 的 AABB 中心与尺寸差。AABB 差不能代表精确表面配准。

已验证的非微波炉示例：[bowl Ring 截图](../../../runs/knowledge_tuner/concept_comparison/bowl_views/part_00.png)、[截图清单](../../../runs/knowledge_tuner/concept_comparison/bowl_views/capture_manifest.json)和[检查报告](../../../runs/knowledge_tuner/concept_comparison/bowl_report.json)。

微波炉示例沿用独立命令，它只负责从场景提取 URDF visual 和旧/校准/运行时 Concept 数据，调用的是同一个通用界面：

```bash
conda run --no-capture-output -n curobo env PYTHONPATH=. \
  python -u tools/viewers/inspect_microwave_concepts.py --port 8084
```

微波炉的已知 frame 差异和数值解释见[专项检查说明](microwave_concept_mesh_inspection.md)。
