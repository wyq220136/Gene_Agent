# 微波炉放入/取出 keypose 的资产坐标核对

本记录是迁移诊断，**不是**当前 URDF 上已通过规划或接触验收的声明。旧 Viser 成功基线使用 `Plate_0`；`Bowl_0` 的固定 keypose 虽已记录，仍需单独规划和视频审核。旧操作配置已原样复制到 `data/task_1/knowledge/microwave_object_manipulation.json`，SHA256 为 `8a4f205b874b76e37d8f21f15c45b842fc82f2991d03583399be15c9d5a8d1c5`。

旧微波炉网格 `data/task_1/objs/Microwave/Microwave_0/meshes/obj_mesh.obj` 的局部尺寸为 `[0.350, 0.230, 0.260] m`：X 是门宽、Y 是高度、Z 是深度。当前校准 URDF 的柜体局部轴为 X 门宽、Y 深度、Z 高度。旧 profile 中的 `microwave_from_*_tcp` 是前一种 frame；直接与当前 `T_world_microwave` 相乘会给出错误目标。

为了只做几何排查，用旧门网格和新门框 visual 的包围盒中心，以及轴对应关系，得到一个**尚未用于执行**的刚体对齐候选：

```text
R_new_from_old = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
t_new_from_old = [0.10833834, -0.15496415, 0.12520390] m
```

在该候选下，旧/新门 visual 中心按定义重合；旧/新把手 visual 质心相差约 **3.7 mm**。这两个特征说明它适合作为 Viser 叠图初值，但还不足以证明整机几何一致，尤其不能拿它定位旋钮。旧 `knob_0` Concept 与当前 URDF 旋钮约有厘米级局部偏差，旋钮应单独使用当前资产的 Concept 和关节轴。

把旧 `Bowl_0` 固定目标经这个候选投到新 URDF root，可得到用于叠图的局部位置：

| 目标 | 当前 URDF root 中的候选位置，m |
|---|---|
| pre-place TCP | `[-0.0073, -0.2644, 0.1293]` |
| place TCP | `[-0.0233, -0.1318, 0.0965]` |
| post-place 1 TCP | `[-0.0307, -0.1457, 0.1222]` |
| post-place 2 TCP | `[-0.0073, -0.2944, 0.1293]` |
| 旧 profile 的炉内碗中心 | `[-0.0467, -0.0646, 0.1115]` |

旧 `Bowl_0` 网格和新 `bowl_visual.obj` 数值尺寸相同。按旧固定的炉内物体 pose，碗的最高/最低 Z 约为 `0.174/0.062 m`，口朝上但倾斜约 **21°**。新炉腔地板 visual 约在 `Z=0.033 m`，所以投影后的碗底高出地板约 **29 mm**；碗的前缘约到 `Y=-0.147 m`，跨过当前门框前平面。旧资产中同一 bowl pose 的深度范围为 `Z=0.0075–0.1569 m`，而旧门厚度处于 `Z=0–0.027 m`，也存在门板区域的空间重叠。旧纯视觉 Task 的成功因此不能被说成已经验证关门时的物理无碰撞。

下一步先用旧资产场景验证矩阵目标与旧 trace；再在当前 URDF 的 Viser 里同时显示门、炉腔、碗和上述 TCP，人工/截图核对入口、碗口方向与夹爪姿态。若为当前资产调整相对 pose，须保存新值和旧值的差异，不能覆盖原始 profile。只有叠图通过后才启动新版连续规划、Skill/Task 组合和视频审核。

## 已有的纯视觉检查入口

在 `curobo` 环境、Gene_Agent 根目录运行 `python tools/viewers/inspect_microwave_object_keyposes.py`。它显示当前机器人、碗和微波炉视觉资产，并把旧配置的放入/取出 TCP 关键位姿投到当前场景。下拉框可切换 4 个放入点与 5 个取出点，选择机位和门角。此界面只检验几何；黄色夹爪是目标 pose 示意，紫色碗是旧记录的炉内目标，**没有 IK、规划或物理成功证据**。

可复核截图：`runs/affordtoolact_migration/object_frame/place_front.png` 与 `extract_grasp_side.png`，同名 JSON 含映射残差。对应命令示例：

```bash
python tools/viewers/inspect_microwave_object_keyposes.py \
  --target '取出 / grasp' --view '夹爪侧面' \
  --capture /tmp/extract.png --report /tmp/extract.json
```

## Pick 数值迁移核对

旧 `build_object_ring_grasps` 与新 `bowl_grasp_candidates` 在同一冻结碗 pose 上生成的 **40 个 4×4 TCP pose 数值相同**，但旧版还按世界方向优先级重排：先偏好 -X，再偏好 -Y。新版原本直接使用 azimuth 枚举顺序，会选错第一个尝试的候选；在线视觉 Pick 入口现已恢复旧排序，测试对全部 40 个有序矩阵做哈希核对。

旧 `MicrowaveObjectInsertTask` 对 Bowl 使用 50 mm pre-grasp 距离，候选自带 41 mm 预抓取开口、7 mm 抓取闭合宽度；旧 `PickSkill` 的 post-pick 默认沿 `pregrasp - grasp` 方向撤离 100 mm。此前新 Skill profile 是 30/80 mm，夹爪开口 80/0 mm，均不等价。`data/task_1/skills/skill_profiles.json` 现把这些旧值保存在单独的 `pick_bowl_legacy_visual` 配置；既有 `pick_bowl` 保留给 MuJoCo 物理探针，避免视觉迁移改变物理实验。

在当前冻结场景的真实 cuRobo 探针中，采用旧距离后已规划 `open/pregrasp/approach/close/lift/hold`；详见 `runs/affordtoolact_migration/pick_bowl_old_profile_probe/skill_pick_bowl_result.json`。最初未接旧版候选筛选前，对同一场景、同一 seed 重启进程曾分别选中 `bowl_ring_03_04` 与 `bowl_ring_03_02`；接入旧 IK 筛选后两次检查都选中 `bowl_ring_03_02`，但这还不是严格的跨进程可复现性证明。旧 Viser adapter 虽在 InsertTask 构造时传入 `use_single_pose_planning=True`，但其 `pick_kwargs` 未传此参数到 Pick；因此 Pick 走默认的 grasp/pregrasp 候选筛选路径，不能误把那个单点模式搬到 Pick。

后续增量已把旧 pregrasp 筛选顺序接入独立的 `legacy_ik` 模式：先检查最终 grasp，再按候选顺序生成原姿态、局部 X、局部 Y 的 17 个 pregrasp 旋转姿态；每批最多 128 个纯 IK 请求，首个成功批次的首个可达姿态确定候选，之后才做路径规划。**路径失败时不换另一个 grasp**，与旧版计划/执行的边界一致。冻结场景一次运行记录了 40 个最终 IK、85 个 pregrasp IK，随后各一条 pregrasp 普通轨迹和 approach 直线轨迹，选中 `bowl_ring_03_02`。

旧 `tamp/curobo/attachment_spheres.py` 用 Bowl 原始局部 mesh 清理断开组件，固定种子采样 32 个表面球（半径 2 mm），然后通过真实 `link_7` 和物体 pose 转到附着 link。新 `assets/objects/bowl_visual.obj` 与旧 `obj_mesh.obj` **文件 SHA 相同**；新旧 32 球中心按 1 nm 舍入后的哈希一致，见 `tests/test_legacy_attachment_spheres.py`。现在闭爪后用实际 approach 终点 FK 建立附着快照，将 Bowl 从 world collision 删除，并把这 32 球加入 `link_7`。该冻结场景的 lift 有 41 帧 cuRobo `linear_tcp` 轨迹，规划耗时约 0.45 s；pregrasp 约 1.01 s，approach 约 0.46 s。`hold` 只是视觉保持事件；这些数值只证明纯视觉规划，**不证明碗与指尖发生了真实物理夹持**。

旧版 post-pick 方向来自已选 `pregrasp - grasp`，目标顺序为原姿态、局部 X ±5/10/15/20°、局部 Y ±5/10/15/20°。新 Skill 保留顺序，先以普通完整轨迹筛选首个可行候选，再对它做直线 TCP 规划；如果直线失败，不换另一候选。当前抽象没有旧 `plan_tcp_sequences_batch` 的完整路径批接口，故筛选是逐候选调用，这一点仍是性能差异，不能声称 Pick 速度与旧版等价。

同一保存的规划记录可在 Viser 中逐阶段检查，或生成纯运动学视频：

```bash
python tools/viewers/view_skill_run.py --run runs/affordtoolact_migration/pick_bowl_old_profile_probe --mode planned
python tools/viewers/recording/record_skill_run_video.py --run runs/affordtoolact_migration/pick_bowl_old_profile_probe \
  --output runs/affordtoolact_migration/pick_bowl_visual_skill.mp4 --mode planned --camera task
```

视频 `runs/affordtoolact_migration/pick_bowl_visual_skill.mp4` 的同名 JSON 记录六个阶段、105 帧、4.375 s；开爪、闭爪、保持等零时长状态事件在录制时各停留 0.35 s，运动帧仍取原 planner 轨迹。另有 `runs/affordtoolact_migration/pick_bowl_visual_gripper_side.mp4`，用于检查指尖与 Bowl rim；已抽查闭爪和 lift 中段关键帧，画面中碗与夹爪共同撤离。Viser 截图 `runs/affordtoolact_migration/pick_bowl_visual_lift_capture/04_lift_planned_0.500.png` 显示 lift 阶段的 Bowl 与 gripper。碗跟随 TCP 是附着后的特权视觉状态，不调用 MuJoCo 物理步进。全量回归为 `677 passed, 1 skipped`。
