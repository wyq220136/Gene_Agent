# 微波炉 Knowledge 检查的坐标与验收

`microwave_scene.build_microwave_knowledge` 是 Viser 无关、但依赖 MuJoCo 场景的适配入口：加载冻结场景、task 抓取 profile、typed Concept，返回把手／旋钮抓取和绕轴路径以及逐个纯 IK 结果。真正不依赖 MuJoCo 的推门 Knowledge 位于 `gene_agent.knowledge.operation.door_push`。`microwave_app.MicrowaveKnowledgeApp` 只负责 visual、交互和状态清单；不复制 grasp 公式，不调用 cuRobo。

旧 `assets/objects/microwave_concept.json` 的 part pose 源自早期重建，不等于当前校准 URDF 的 part pose。直接把它与当前 `T_world_microwave` 相乘，旋钮 TCP 会落在约 `[0.69, 0.572, 0.03]` m，而当前 URDF 旋钮中心约为 `[0.507, 0.335, 0.187]` m。**不能通过移动夹爪 visual 掩盖此 frame 错误。** 展示入口从冻结场景的 `metadata.objects.knob_concept` 构造 CylinderConcept；把手中段中心从 `microwave_door_handle_visual` 取得，局部 +Y 沿竖直把手、局部 +Z 从门板指向用户，再以 `grip_segment.pose_in_trifold` 反算 `T_microwave_trifold`。展示配置只覆盖旋钮 `top` 抓法。把手使用旧 `microwave_handle` 迁移值：`vertical_offset_m=0.05`、`tip_x_deg=-60`、`tip_y_deg=0`、`depth_m=0.013`、TCP 局部 Z 偏移 `-0.012`。可达性不得通过暗改这些值取得。

门轴取 URDF 的 `microwave_door_joint`，是显式资产标注；旋钮轴取校准 CylinderConcept。旧 `gene_knowledge.operation.door` 中的 `door_push_mode_state → door_push_preference_for_mode → generate_door_push_candidates` 已作为纯几何 `generate_mode_door_push_path` 迁入新 Knowledge：每个门角度先用显式轴更新校准门板，再重算模式参数、三个门宽候选及局部 X 倾角偏移 `(0,-10,+10)°`；`legacy_viser` 按旧界面的逐角动态倾角和候选顺序选取，`top_down` 保留先前新展示器的选择方式。**不能把初始 TCP 刚性旋转到终点**，那会丢掉门宽候选切换、退让距离及末段 25° 门板倾角。四种旧模式的公式均保留；当前右臂 Viser 只开放 `right_open/right_close`。右臂开门沿用旧阶段二 `30°→85°`、每 10° 一点，关门 `85°→0°`；旋钮目标角仍为 0.45 rad。对同一校准门板，逐点调用旧 `make_door_push_pose` 和旧候选生成器，7 个开门 TCP 矩阵最大逐元素差约 `4.4e-16`。这验证了几何公式继承，**不验证 IK、碰撞或真机效果**。

关门不能把“基础模式参数相同”理解为“最终夹爪姿态与旧界面相同”。2026-10-07 审计确认：旧 `right_close` 模式状态的外侧接触、门宽 `(0.75,0.875,1.0)`、退让、顶部距离及基础 X 倾角 `40°` 与新实现的逐角公式一致；旧 `viser_microwave_door_push.py` 的 `right_close` 默认关闭 `prefer_top_down`，按当前 TCP 相对 SDK 原点的方向计算 X 倾角。当前完整场景适配器以 active group 的 base link 为参考原点，显式调用 `selection_style="legacy_viser"`，状态 JSON 也记录该选择策略。旧 `task_1` 场景 85° 的选中值为门宽 `0.725`、X≈`−36.9°`；新校准场景为门宽 `0.725`、X≈`−39.1°`。场景资产和摆位不同，不能比较 world TCP 绝对坐标。旧长程 `build_door_cycle_plan` 在此基础上还采样 X/Y 倾角并用 IK 逐角筛选，默认角步长 `20°`；当前展示仍不是旧 Task 的最终关门路径。旧截图为 `runs/knowledge_tuner/legacy_right_close_85.png`，新截图为 `runs/knowledge_tuner/microwave_door_close_legacy_preview.frames/frame_000.png`。

Knowledge Viser 默认预览摆位由 `preview_microwave_xy_m=[0.52,0.29]` 给出，冻结 run 仍为 `[0.65,0.52]`；修改只作用于内存中的 MuJoCo 模型和状态。该摆位初始无物体穿透，把手纯 IK 可解；`legacy_viser` 规则下右臂开门为 4/7、关门为 7/10 waypoint 纯 IK 可解，旋钮抓取当前也未求得。2026-10-07 受控扫描表明：固定摆位仅将局部 Y 倾角偏移设为 `−20°`，开门和关门可分别达到 7/7、10/10；固定原 Knowledge 数值，改到 `[0.48,0.19]` 也得到 7/7、10/10。两组都只证明纯位姿 IK，前者尚未验收指尖接触和碰撞，后者尚未验收与碗、垫台的空间关系，不能直接作为默认场景。更早 `[0.42,0.18]` 的搜索基于先前 `top_down` 规则，且发现垫台与碗严重穿透。需联合检查摆位、碗位置、抓门、推门和旋钮；当前没有碰撞或运动规划验收。

交互时 `mj_copyData` 从冻结快照构造预览数据，设置门／旋钮关节及当前 IK 关节解，再调用 `mj_forward`；`PickBowlScene.update_visuals` 同步机器人、活动臂原生夹爪和微波炉 mesh。不可达时显示求解器的最佳尝试关节角，并以红线标出实际 TCP 与期望 TCP 的差距；此姿态不代表 IK 可解。目标夹爪是独立的半透明 ghost，默认隐藏，仅供检查期望姿态。冻结源数据与正式 task profile 均不变。滑块只求当前姿态的 IK，避免每次拖动都计算所有 waypoint；显式按钮才运行完整 IK。滑块参数保存在截图 JSON 的 `tuner_values`，未检查的 waypoint 以 `null` 记录。无效输入会退回上次有效滑块和场景，`last_rejected_error` 记录原因；Viser 节点增删在同一锁内完成，避免并发回调删除新建路径。URDF 门上限 `1.91986 rad` 和滑块 `110°` 的约 `2.18e-6 rad` 舍入差按精确 URDF 上限夹紧，不再误报越界。

`tools/viewers/recording/record_microwave_knowledge_video.py` 在 Viser 本体逐帧切换已求解的推门 waypoint，由同一 Chromium 相机截图并编码 MP4。每帧 PNG、目标 TCP、逐角参数、纯 IK 与截图 SHA256 写入视频清单；默认打开目标夹爪 ghost，保持机器人真实姿态与目标姿态可区分。录制内容仍是离散 Knowledge 预览，不是碰撞可行轨迹。

回归检查：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/ubuntu/miniconda3/envs/curobo/bin/python -m pytest -q -p no:cacheprovider \
  tests/knowledge/test_microwave_scene_knowledge.py tests/knowledge/test_door_push_pose.py
```

截图应人工检查真实夹爪相对于把手／旋钮的位置，以及门终点是否能同时看到当前夹爪和已转动门板；不能仅以 PNG 存在或 `ik_reachable=true` 代替视觉验收。
