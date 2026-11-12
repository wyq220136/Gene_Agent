# 标定微波炉、支撑布局与轴语义

受众：开发Agent。状态：已接入用户提供的calibrated URDF；旧简化外壳已移除。
本页优先于scene_migration.md里的旧proxy描述。原data目录只读，未修改原文件。

## 资产来源和格式选择

读取 `data/microwave_mujoco/{README.md,urdf/README.md,urdf/validation.json,calibration/calibration.json}`、
原生 `model/microwave.xml`、`build_mujoco.py` 与已有验证报告后，冻结必要URDF、相对mesh及native MJCF。
位置：`assets/objects/microwave_calibrated/`，`provenance.json`记录来源与SHA256。

`gene_agent/scene/microwave.py`是针对该资产的窄转换器，不泛化成URDF框架：
URDF提供4个link、3关节、57 visual、16 collision及惯量；已缩放的mesh保持scale1。
原生MJCF补充URDF不能表达的armature=.0001、geom摩擦/solref/solimp/margin和handle/cavity sites。
显式60组机身与门/旋钮碰撞pair保留native的父子碰撞检测，避免把filterparent全局关闭影响机器人。
惯量在origin旋转后以fullinertia写入，关节限位单位弧度；objects.xml显式compiler angle=radian。
源native全局implicitfast/80 solver iterations属于整体场景求解策略，未用资产强行覆盖总场景；
物体独立测试使用MuJoCo默认2ms积分，整体仿真应记录实际solver配置。

闭合外形约448×307×236 mm；宽度部分外推、深度依赖先验，不能称为实测全尺寸。
机身9.5kg、门1.4kg、旋钮.04kg及摩擦/惯量仍为源模型假设，不是真机辨识。

## Frame与业务命名

新model根frame：+X电器右方、-Y前方、+Z上方，底脚最低Z=0。
它不同于旧task_1微波炉object frame；不能把旧Concept局部pose直接套在新root。
标定 `T_camera_from_scaled_model` 左乘已核验方向的 `T_world_camera=right_extrinsic_matrix`，
得到新的captured_pose。旧pose另存legacy_captured_pose，不重复scale。

| 源关节 | 业务joint/body | link局部轴 | 范围 |
|---|---|---|---|
| door_hinge | microwave_door_joint / microwave_door | [0,0,-1] | 0..1.91986 rad |
| power_knob_rotation | knob_0_joint / knob_0 | [0,-1,0] | 0..4.71239 rad |
| timer_knob_rotation | knob_1_joint / knob_1 | [0,-1,0] | 0..5.23599 rad |

旧knob_0位于上方，因此将上方power映射为业务knob_0；这是一项记录过的语义对应假设，
不是用户确认“knob_0一定是功率”或由名字猜轴。

业务knob_0_site采用Concept frame：局部+Z沿URDF旋转轴朝外，+X沿电器右，+Y向上。
其原点是旋钮collision包络中心，不是父joint原点，但两点位于同一旋转轴。
`microwave_joint_mapping`保留原joint origin及真实axis/range。
新Concept从collision网格实测：外椭圆半径约30.796mm/26.829mm，轴向厚9.756mm；
该包络不是按钮前盖实测尺寸或精确接触形状。

Knowledge必须显式传 `concept=scene['knob_concept']`；不传仍用旧Concept进行历史回归。
新Concept +Z朝外，grasp_direction='top'使TCP接近轴朝内；旧Concept则仍bottom。
保留原azimuth集合和TCP局部-15mm预偏移，其物理抓取效果尚未验证。
`T_microwave_knob_0`/`T_world_knob_0`是新Conceptframe，不能等同knob_0 body frame。

## 仿真摆放

新用户允许仿真布局调整，故区分 `captured_poses` 与 `simulation_nominal_poses`；
兼容键nominal_poses指后者，sample_poses由后者加扰动。

- 桌面：0.9×0.9m，中心[.5,.4]，顶高-.12m。它是仿真支撑设置，不是实测桌面。
- 微波炉：root [.65,.52,.04]，yaw=-90°；下方固定支撑高.16m，顶面位于.04m。
- 支撑底面严格采用机身结构碰撞包络 .4479437482×.2682940816m，并随微波炉XY/yaw扰动；
  不是任意放大的桌台。支撑visual为棕色，独立collision为group 3。
- 碗：中心[.30,.20,-.078]，object+Y竖直向上，保留采集投影yaw；底部留2mm重力落稳间隙。
- 碗移至门扫掠区域外侧。名义门全程0..110°检查无初始实体穿透。

桌面覆盖支撑及碗；支撑托住关门机身/脚，打开门的扫掠体仍不碰碗。微波炉与支撑固定，
未模拟倾覆。
扰动仍world XY±15mm/yaw±5°，不变桌高、不做IK过滤，不隐式丢弃不可达样本。
可达性由后续Motion单独验收；微波炉x=.65并不保证机械臂可达。

## 碰撞与显示修复

炉腔以独立左壁、控制仓、底、顶、后壁等碰撞体构造，不用整机凸包。
门板、握把及连接件独立；旋钮与转盘使用凸mesh是局部实体合理近似。
collision group3 alpha0、visual group2 contype=conaffinity=0，碰撞不随视觉修复变化。
原生contact margin=.0001m、gap=0表示接触激活壳，不是100µm实体厚度。
pair只在微波炉内部，不允许Motion为兼容而删pair或静默丢障碍。

近景实际发现两旋钮rim visual有大面积三角伪影。源上旋钮mesh非watertight、19个连通分量。
只替换这两个visual为测量外包络/内盖半径一致的闭合椭圆环，保留原mesh和其余55个视觉元素。
`visual_repairs.json`记录原/新bounds及理由，`*_repaired.obj`单独命名；轴、质量和collision保持原样。
已重新打开after.png确认大块三角闪烁消失，源其他细节仍保留。

## 实际验证与产物

- 8项scene测试通过，其中支撑测试检查高度不少于15cm、底面与微波炉包络完全一致及顶面承托关系。
- 10个扰动scene × 3组关节姿态与native MJCF比较：link位置/旋转/惯量atol<=1e-9。
- 名义初态无穿透；25个门角0..110°无穿透；20个扰动seed均初态无穿透并在3秒内碗落稳，
  最大碗速度范数.004715（详见runs/microwave_validation/validation.json）。
- 8mm球真实gravity drop接触turntable；同50点入口路径闭门受阻、100°开门全部净空。
- 上旋钮45°运动学41帧，矩阵终点误差5.55e-17rad，父pose固定；不计入物理操作成功。

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python -m unittest discover -s tests -p test_scene_objects.py
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python -m unittest discover -s tests -p test_knowledge.py
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python tools/diagnostics/check_knob.py --output-dir runs/knob_calibrated --rotation-deg 45
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python tools/diagnostics/check_knowledge.py --output-dir runs/knowledge_calibrated --seed 17 --perturb
```

已实际查看 `runs/knob_calibrated/{scene_open.png,after.png}`：碗在桌上、微波炉脚有支撑、门打开后
炉腔与转盘可见、旋钮刻度可辨。该目录另有scene_closed.png、before.png、rotation.gif与完整JSON。
整体机械臂/桌面初态碰撞及GPU可达性仍需总场景审核，物体独立测试不能替代。

## 参考项目采用原则

只读参考 `third_partys/EmbodiedSWE/real_to_sim/examples/home_desk_franka/calibrate_scene.py`：
它明确区分度量尺度、支撑平面world frame、台面extent合理性检查。本项目采用显式support/尺度元数据，
不运行其重建依赖，不把允许调整后的桌高冒充重建测量。
只读参考 `third_partys/ASPIRE/aspire/sim/cap/integrations/motion/pyroki_snippets/_solve_ik_with_collision.py`：
它分开self/world collision并逐一加入world几何；本项目保留独立碰撞表示及明确margin，不借其权重或整套优化器。
另参考ASPIRE scene_snapshot.py同观察多视角记录原则，具体总场景视觉由本项目feedback入口负责。
