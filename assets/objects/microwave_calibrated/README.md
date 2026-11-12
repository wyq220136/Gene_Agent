# 校准后的微波炉 URDF

入口为 `microwave_calibrated.urdf`。复制时保留整个目录，特别是相对路径引用的 `meshes/`。

沿用现有 RGB-D 约束估计尺度：闭合外形约宽 448 × 深 307 × 高 236 mm，含把手。网格已缩放，URDF 中 mesh scale 为 1，导入时不要重复缩放。坐标为模型局部坐标，+X 向右、−Y 朝前、+Z 向上；相机位姿另见 `../calibration/calibration.json`。

包含 4 个 link、3 个 revolute joint、57 个视觉元素和 16 个碰撞元素。碰撞体保留 MuJoCo 版本的分墙炉腔和独立把手，不使用填满炉腔的整机凸包。

已在 MuJoCo 3.14.0 成功导入，并核对四组关节姿态、质量、质心、主惯量与 MJCF 一致；导出网格实测包围盒与校准尺寸一致。结果见 `validation.json`。URDF 不完整表达 MJCF 的接触、求解器、armature 和 site 设置；MuJoCo 仿真仍优先使用原生 MJCF。

尺寸继续保留原标定的不确定性：宽度部分外推，深度依赖先验。本次仅导出格式，没有重新拟合或修正把手等局部形状。

在 Gene_Agent 根目录重新导出并验证：

```bash
third_partys/articraft/.venv/bin/python data/microwave_mujoco/export_urdf.py
```
