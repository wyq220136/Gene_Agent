# 场景和碰撞资产准备取舍

受众：开发 Agent。用于本次 calibrated microwave 接入；不是完整 FoundationPose 感知流水线。

## 参考与采用

- [RoboCasa 资产导入指南](https://github.com/robocasa/robocasa/blob/main/robocasa/scripts/asset_scripts/README.md)
  将 raw、visual、collision 分开，并通过独立 viewer 检查包围盒、碰撞体、关节和 site。
  本项目保留用户原数据只读，冻结使用副本与哈希；显示和接触几何独立验证。
- [MuJoCo Menagerie 的 Panda 场景](https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/franka_emika_panda/scene.xml)
  将机器人与场景显示配置分开，用 skybox、headlight 和灯光建立可辨认的背景与轮廓。
  本项目把这些设置集中在 visualization，不向业务碰撞场景添加装饰障碍。
- [MuJoCo 碰撞建模说明](https://mujoco.readthedocs.io/en/3.8.0/computation.html)
  常规 mesh 接触基于凸几何；非凸物体需组合多个凸几何，必要时用 CoACD 等外部工具分解。
  对碗和微波炉优先使用有结构意义的底/墙/门分块；不将单个整机或整碗 mesh 当作可用内腔。

## 当前流水线

1. 检查来源资产的单位、缩放是否已经烘焙、root frame、关节轴/范围与 inertial。
2. visual 保留原颜色/材质/网格；collision 独立记录对应 link、origin、形状与近似。
3. 对 URDF 与原生 MJCF 比较多组关节姿态，而非仅看零位；补回 URDF 无法表达的接触/求解参数。
4. 采集世界 pose 只作参考；选择满足支撑、间隙、可视与后续可达性检查的仿真名义摆放。
5. 动态物体给予小初始间隙并重力落稳，记录采集/名义/采样/落稳四层 pose；固定家具底部对齐支撑面。
6. 碰撞正负例：外壁阻挡、内腔净空、开门与关门的同路径对照、薄缝/碗底、模型间初始穿透。
7. GT 作为唯一运行事实来源，再转换到 planner frame。必须保留显式接触规则/接触 margin 的语义，
   无法表达时明确拒绝或报告经测试的保守代理，不能只复制坐标与 mesh 就宣称等价。

对复杂扫描外形可后续采用凸分解，但不额外安装完整场景生成框架作为首批任务前置条件。
视觉好看不证明碰撞正确；碰撞代理简单也不要求牺牲 visual 质量，两者以同一 frame 对齐检查。
