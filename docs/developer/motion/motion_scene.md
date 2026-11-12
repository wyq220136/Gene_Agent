# MuJoCo GT 静态碰撞快照

> 历史记录：本体固定误报现已通过补回运行期安装接触策略修复；默认场景也已换为标定微波炉。最新结论见 [审计记录](body_collision_audit.md)。

受众：开发 Agent。状态：2026-10-04 已实现、CPU测试和GPU碰撞探针已通过；完整快照下候选IK全部被拒绝，原因证据如下。

## 接口

`gene_agent.motion.scene.export_collision_scene(model, data, robot_metadata, active_group, output_dir)`
返回原生 cuRobo `scene_model` 字典、逐geom来源/变换/近似、排除清单、`snapshot_sha256`、
`active_group`、`other_arm_frozen`、`frozen_groups`及当前qpos。

导出时先 `mj_copyData`，再在副本执行 `mj_forward`，保证积分后qpos和geom pose一致。
调用方缓存与warmstart不变，不要求调用方预先forward。
根body从 `robot_metadata.group_root_bodies[group]`读取，不假定七轴、right或特定根名称。
Realman metadata由机器人构建器填left_mount/right_mount，但导出器可以接受任意命名。

几何统一使用 `T_base_geom = inv(T_world_base) @ T_world_geom`，长度米、四元数wxyz。
按contype/conaffinity两者皆零排除visual/禁用接触几何，不看alpha或viewer group。
活动组根body子树交由原cuRobo机器人球与self_collision_ignore负责；其他组当前pose冻结成障碍。
桌面、碗底圆柱、128块碗壁、微波炉proxy、机器人本体和另一臂都保留。

cuboid直接使用MuJoCo半尺寸的2倍。mesh使用编译后的model.mesh_vert和当前geom世界pose，
不重复应用mesh主轴/质心、scale或源STL坐标。碰撞mesh导出凸包，匹配本轮MuJoCo默认凸接触。
当前cuRobo检查器原生支持cuboid/mesh，圆柱采用64段外接代理，最大半径增加约0.12%，高度不变；
球/胶囊以保守外接多面体导出。碗壁仍为分块box，不合并成封口凸包。

未知启用形状报错。显式contact pair可覆盖双零mask，因此npair>0直接NotImplementedError；
非零margin/gap也直接NotImplementedError，禁止静默遗漏。当前场景npair=0且margin/gap全0。
逐几何pair位运算过滤没有映射为cuRobo link过滤；导出障碍对活动组保守生效，必须审查结构接触语义。
原始场景已关闭knob接触，所以该geom在排除列表，不能宣称旋钮已有碰撞保护。

快照hash覆盖接触mask、源几何、变换、mesh哈希、qpos和世界映射，不受输出目录影响。
SynchronousPlanner结果保存结构化scene_snapshot_sha256，输入验证错误、后端异常与solver拒绝分开。
快照不自动跟踪另一臂/物体后续运动；不能把静态快照称为双臂协同规划。

## 已执行检查

```bash
PYTHONUTF8=1 PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  -m pytest tests/robots/kinematics/test_robot_model.py tests/motion/test_motion_contract.py tests/motion/test_motion_scene.py -q
PYTHONUTF8=1 PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/diagnostics/check_scene_collision.py --output runs/scene-collision
```

25项本组测试通过，包含：非透明/透明接触选择、active/other臂划分、任意根名称、
frame/尺寸转换、快照hash变化、未知类型拒绝、圆柱代理、mesh偏移只应用一次、
step后无需forward且不改调用方状态、显式pair与非零margin/gap拒绝。
另一Agent还独立检查了mesh非均匀scale和两级旋转平移，世界顶点误差9.35e-8m。

nominal与建议微波炉world X−0.10m各导出137 cuboid+13 mesh。
真实GPU sphere查询均通过以下四项：

- 桌面内部：碰撞代价>0。
- world [5,5,5]远方：代价0。
- 碗腔内部：代价0，说明没有用封口凸包替代凹腔。
- alpha=0的碗壁内部：代价>0，说明没有按视觉透明度漏掉接触。

查询返回的是碰撞代价，不当作精确SDF或MuJoCo接触深度。

## 全快照IK结果与固定冲突

两组均 bowl 0/40、knob 0/6，都是solver拒绝，无Python异常。目标、起点与静态snapshot均保存。
此前独立screening中微波炉X−0.10m有6/6 IK通过，加入完整当前接触快照后全部失败；
因此之前的摆放建议仅具有局部IK证据，不能据此执行。

当前home有87个启用机器人球，8个与外部快照相交，最大累计代价0.10561。
逐障碍独立探针诊断（仅用于贡献定位，不用于规划）得到：

| 静态源geom | 机器人link / sphere index | 代价 |
|---|---|---|
| robot_body_body | base_link / 0–6 | 0.03791–0.10423 |
| robot_body_body | link_1 / 13 | 0.005336 |
| robot_body_back | base_link / 1 | 0.001381 |

`base_link`是固定基座，关节变化不能移走这些球；当前本体proxy与基座球之间存在固定约束冲突。
这能解释为什么只调整目标/微波炉位置不能解决完整场景拒绝。
尚不能据此认定哪份标定/球/本体proxy错误，也不能直接推导应该忽略哪些接触。
后续需要核对真实安装的结构接触语义及本体proxy/球覆盖，建立具体验证后再改几何或碰撞规则。
本轮没有删球、禁障碍、放宽碰撞或移动机器人基座来制造通过。

[完整结果](../../../runs/scene-collision/result.json)保留每个相交球的link、base/world中心、半径和source geom id/name，
也保留所有92个候选的失败结果、目标pose和snapshot ID。
[运行日志](../../../runs/scene-collision/run.log)。
每组`collision/collision_snapshot.json`保存150障碍及全部排除来源，OBJ为冻结另一臂和圆柱代理。
不宣称物理抓取、旋钮轨迹或真实机器人验收成功。
