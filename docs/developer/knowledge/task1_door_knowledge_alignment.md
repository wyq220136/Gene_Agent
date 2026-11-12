# Task1 门操作 Knowledge 对齐

本文记录 Gene_Agent 门操作 Knowledge 与旧项目 Task1 实际同步调用链的接口边界。唯一行为参考是旧项目 `scripts/run_microwave_task1.py --door-only ... --reestimate-open-angle-after-handle` 的实际调用代码；独立 viewer 和历史说明不作为等价基准。

## 源行为对照

| 阶段 | 旧 Task1 的候选与排序 | pre-contact / 撤离 | 当前对齐要求 |
|---|---|---|---|
| 抓把手并拉门 | `microwave_handle` profile 先生成 grasp；每个 grasp 再生成局部 Y `0,+5,-5,+10,-10,+15,-15°` 与局部 X `0,+10,-10,+20,-20,+30,-30°` 的组合。组合按总转角、是否保持 X=0 等稳定排序；使 TCP 局部 +Z 更指向门板的 X 旋转会被剔除。默认完整路线先检查，失败后选择可达 waypoint 数最多且顺序最靠前的候选。选中 grasp 后把拉门角等分为 3 个门轴旋转点；每点还右乘 TCP 局部 X 的负门增量角补偿。源码：旧 `visualization/knowledge/microwave_door_cycle_visualizer.py:863-1078` 与 `gene_knowledge/operation/door.py:412-462`。 | 把手 grasp 的接近由 Grasp Skill 处理；拉门后的 clearance 属于 Task 阶段编排。 | Knowledge 保留完整 grasp 候选和同源三点路径；viewer 只手动选候选，不冒充 Task 的自动 IK fallback。 |
| 内侧推开 | 每个门角独立生成集合。`right_open` 使用 inside 面；门宽比例前半程 `(0.75,0.875,1.0)`，后半程 `(0.5,0.625,0.75)`；standoff 从 40 mm 递减到 0，距顶部从 30 mm 增到 50 mm，后半程最多向门板倾 25°。先以第一个门宽位置、tip X=0 求“远离 SDK base”的中心 X 倾角，再按 X 半径 50°、Y 半径 70°、10°网格生成搜索序列；每个 tip 组合按三个门宽位置生成候选。默认当前值最优先，其余按与当前值的 L1 角距离、只改 X/只改 Y/同时改两轴、正负方向稳定排序。源码：旧 `_push_stage_candidate_sets()`、`tip_rotation_xy_search_order()` 与 `generate_door_push_candidates()`。 | 必须先由外部 IK/约束筛选选中第一个 contact candidate，再沿该 candidate 的 inside 门面朝外法向移动 50 mm；姿态不变。pre-contact 不允许门/把手碰撞，contact 起才由 Skill/Motion 放开接触。 | Knowledge 返回完整有序集合与法向，不做碰撞或路径规划；viewer 可选择集合中的 candidate 并从它生成 pre-contact。 |
| 外侧推关 | 每角使用 outside 面；`right_close` 门宽比例为 `(0.75,0.875,1.0)`，同样使用 standoff/顶部距离逐角公式。tip 中心按 SDK base 方向求得，X/Y 搜索半径均为 50°。候选顺序与开门相同。源码同上。 | 从实际选中的第一个 close candidate 沿 outside 门面朝外法向移动 50 mm。旧循环阶段还在它前面加入“上方回撤参考 pose”，这属于阶段衔接，不属于 contact Knowledge。 | Knowledge 只交付 close 候选、所选 contact 与 pre-contact；Task 的上方回撤、默认 TCP 和路径碰撞策略不进入本接口。 |

旧 Task1 每个门角先检查 16 个候选，失败时再扩展剩余候选；这是外部可选 IK 筛选策略。候选生成本身不依赖 cuRobo。`select_reachable_door_push_points()` 默认不把相邻 waypoint 的距离当作连续轨迹成功。

## 公共接口

本轮接口位于 `gene_agent.knowledge.operation.door_push`：

- `generate_mode_door_push_candidate_sets(...) -> tuple[DoorPushCandidateSet, ...]`：按起止角生成有序集合。每个 `DoorPushCandidate` 提供稳定 `candidate_id`、world TCP pose、world 门面朝外单位法向、门角、几何参数、候选顺序与来源。
- `select_door_push_path(candidate_sets, selected_candidate_ids=None)`：把每角选中的候选组装成接触 `OperationPath`。未提供选择时确定性地取每组第一个候选；这只表示名义预览，不表示 IK、碰撞或路径通过。
- `make_door_push_precontact(selected_candidate, distance_m=0.05)`：保持 selected candidate 姿态，位置增加 `surface_normal_world * distance_m`。

把手路线候选位于 `gene_agent.knowledge.operation.handle_pull`：

- `generate_handle_pull_candidates(grasp_tcp_pose_world, toward_panel_direction_world)` 返回  Task1 同源的有序 `HandlePullCandidate` 集合。每项包含稳定 ID、world grasp TCP、局部 X/Y 旋转参数、偏好序号与来源；只改姿态，不改位置。
- 候选排序与旧 `_handle_grasp_pose_candidates()` 相同；让 TCP 局部 +Z 更指向门板的局部 X 旋转会被剪除。`generate_handle_pull_path(..., rotation_steps=3, tip_rotation_x_scale=1)` 消费**已选** grasp pose，不内置 IK 选择。默认 25° 时三个拉门点为 `8.333°、16.667°、25°`，并分别右乘 TCP 局部 X `-8.333°、-16.667°、-25°`；pregrasp、grasp/close 和 release/retreat 是 viewer 的检查点，不属于旧函数的三个旋转点。

公共量均为 world frame、米和弧度。`closed_panel_pose_world` 是 **0 rad 闭门参考**；函数内部只按每个 waypoint 的绝对门角应用一次旋转。`axis.direction` 的正向就是调用场景的门关节正方向。若从旧 SDK base 输入，调用者必须先用 `T_world_sdk_base @ T_sdk_base_tcp` 转成 world；不得把已按门角更新的门板 pose 再作为 closed pose 传入。

最小调用示例：

```python
candidate_sets = generate_mode_door_push_candidate_sets(
    closed_panel_pose_world,
    panel_width_m=0.36,
    panel_height_m=0.22,
    panel_thickness_m=0.03,
    axis=door_axis_world,
    mode="right_open",
    start_angle_rad=np.deg2rad(30),
    target_angle_rad=np.deg2rad(85),
    step_rad=np.deg2rad(20),
    reference_origin_world=robot_base_world,
)
selected = candidate_sets[0].candidates[0]  # 外部可改为 IK/约束筛选结果
precontact_pose = make_door_push_precontact(selected, distance_m=0.05)
contact_path, parameters = select_door_push_path(candidate_sets)
```

`precontact_pose` 与 `selected.tcp_pose_world` 的平移差必须正好是 0.05 m，方向等于 `selected.surface_normal_world`，旋转完全相同。碰撞白名单、连续轨迹、真实执行和任务成功不属于以上函数返回值。

## Task1 的运行时覆盖

`--reestimate-open-angle-after-handle` 会把阶段二起始门角替换为把手阶段执行后的观测角；Knowledge viewer 只接收一个明确的起始角，不复制感知。`return_to_initial_tcp_after_stage_two=True` 会在 Task 层把推开后的撤离改为 `load_default_joints(side)` 经 FK 得到的初始 TCP。Knowledge 不读取默认关节，也不在 viewer 伪造该撤离点。

## 数值验收

规范化门板使用相同 closed pose、门轴、门板尺寸和 reference origin，分别直接调用旧项目 `_push_stage_candidate_sets()` 与新接口。对比了每个门角的**全部候选**，而非只比首个 pose：

| 模式 | 角度 | 新旧候选数 | 全部 TCP 矩阵最大绝对差 | 全部几何参数最大绝对差 |
|---|---|---|---:|---:|
| `right_open` | `30, 50, 70, 85°` | 均为 `495, 405, 405, 450` | `3.33e-16` | `7.11e-15` |
| `right_close` | `85, 65, 45, 25, 5, 0°` | 均为 `330, 330, 330, 330, 363, 363` | `5.55e-16` | `1.07e-14` |

候选数会随门板和机器人 base 的相对方向变化，因为中心 X 倾角和被 `[-60°, 60°]` 截断的搜索网格都取决于 reference origin。当前 Knowledge 页面的默认预览摆位实际生成：

- `right_open`：`495, 495, 450, 405`；
- `right_close`：`297, 297, 363, 363, 297, 297`。

自动测试另外覆盖了 inside/outside 法向、显式 candidate ID 选择、50 mm pre-contact 位移与姿态不变性，以及微波炉平移/旋转后的刚体等变性。当前场景中开门首选的 pre-contact/contact 距离为 `0.0500000000000000 m`，方向为 world `[0.8660254, -0.5, 0]`。

把手规范化输入也直接调用旧 `_handle_grasp_pose_candidates()` 对比：新旧均生成 30 个有序候选，全部 TCP 矩阵最大绝对差为 `0.0`。另用同一任意 grasp、门轴和 25° 目标，按坐标约定把旧 SDK 门轴换成新场景的相反关节正向轴后，三个拉门 TCP 的完整矩阵最大绝对差为 `0.0`；这项比较同时覆盖门轴位置、姿态以及局部 X 补偿。测试另保存把手和开/关门完整候选集合的几何/顺序哈希，避免只检查前几项而放过后续漂移。

## Viewer 与状态迁移

`MicrowaveKnowledge` 现在同时保留 `handle_pull_candidates`/所选把手候选，以及 `door_candidate_sets`、`door_selected_candidate_ids` 和由已选候选组成的 contact path。Viewer 可切换把手路线候选；紫色为当前门角的前 50 个候选位置/方向，红色球为已选 contact，绿色球与线段为 pre-contact 和门面朝外法向。完整集合仍保留在 Knowledge 对象中；画面限制数量只为避免几百条方向线遮挡门板。状态 JSON 记录候选数量、所选 ID/顺序/参数、门面法向和预览 ID。

旧 schema version 1 状态仍可读取：缺少 `handle_pull_selection` 或 `door_push_selection` 时默认选顺序 0；原状态中的 `retract_up_m` 和 `retract_local_z_m` 字段不再进入 Knowledge viewer 默认路径。为了不打断现有 live door 装配，`build_microwave_knowledge()` 仍可在**同时显式传入**两个旧撤离距离时追加兼容点；默认值为 `None`，不代表 Task1 撤离。纯几何、MuJoCo IK 和 `ik_scope="none"` 不会在模块导入时加载 cuRobo；只有显式选择 `ik_backend="curobo"` 才延迟加载该适配。

真实 Viser 截图已实际打开检查：

- 全景：`runs/knowledge_tuner/task1_door_candidate_precontact.png`；
- 门板侧向近景：`runs/knowledge_tuner/task1_door_candidate_precontact_closeup.png`；
- 把手候选与 25° 路线：`runs/knowledge_tuner/task1_handle_pull_candidates.png`；
- 同名 `.json` 保存图片哈希、相机、完整当前状态和场景快照哈希。

可直接运行：

```bash
conda run --no-capture-output -n curobo env PYTHONPATH=. MUJOCO_GL=egl \
  python -u tools/viewers/capture/capture_knowledge_viser.py \
  --tool microwave --mode door_push --waypoint 0 \
  --output runs/knowledge_tuner/task1_door_candidate_precontact.png

conda run --no-capture-output -n curobo env PYTHONPATH=. MUJOCO_GL=egl \
  python -u tools/viewers/capture/capture_knowledge_viser.py \
  --tool microwave --mode door_push --waypoint 0 \
  --restore-state runs/knowledge_tuner/task1_door_candidate_precontact_closeup.restore.json \
  --output runs/knowledge_tuner/task1_door_candidate_precontact_closeup.png
```

## 仍由主线处理的差异

- Viewer 默认确定性地选每组第一个候选，并只提供起始门角的人工候选切换。它没有复制旧 Task 的“先检查 16 个，失败后扩展”外部 IK 筛选；主线应依次消费每个 `DoorPushCandidateSet`。
- 把手 Viewer 可手动切换全部同源候选，但不代替旧 Task 的“默认整路先检查，失败后按可达 waypoint 数与原始顺序选 fallback”。主线应对每个 `HandlePullCandidate` 生成整条路线并实施该选择策略。
- 本轮没有检查场景碰撞、候选间连续轨迹、门接触或动力学。严格/近似 IK 状态只是 Knowledge 页面的位姿可达性证据。
- `--reestimate-open-angle-after-handle` 的观测更新、contact 阶段的允许接触、返回默认关节 FK 的 TCP 和执行后置条件仍由 Skill/Task 实现。
