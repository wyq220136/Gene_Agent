# 真机接入的冻结基线与边界

日期：2026-10-13。实施计划见 真机增量接入计划（历史材料已本地归档）。

本次只新增 `gene_agent/realworld/`、离线诊断和测试；现有仿真、Viser、Skill、cuRobo 源码不因真机接入而改动。开始前在 `curobo` 环境运行 `tests/motion/test_motion_model.py` 与 `tests/applications/microwave/task/test_task1_visual_parity.py`，结果为 **34 passed**。这组回归保护公共规划契约及 Task1 视觉顺序，不等于完整 MuJoCo 物理验收。开始时仓库已有其他未提交改动，不能把它们归因于真机接入。

记录时的 Git HEAD 为 `379941b52379fb59d86305b1abe624a20563a3e9`。三个直接相关的 SHA256：`task1_visual.py` 为 `7aed7ec47f5526a5dd6300bb2e6c90ea832db21814800b075d0e8f0638baf135`；`task1_visual_session.py` 为 `1c1d509b2fb64fd8c758766872c8dad045e07b0ea7e663da4fd44b40d8a5cb7e`；`data/task_1/visualization/task1_visual_explicit90_verified.json` 为 `146efa30769963944a6af10896aa26e269c78cdac31df7896a0b1cfaf454e4c3`。这些指纹只用于发现无意修改，不能代替轨迹与视频验收。

当前纯视觉 Task1 的七段顺序为：首次开门 → 放碗 → 首次关门 → 旋钮 → 再次开门 → 取碗并桌面放置 → 再次关门。该顺序由 `gene_agent/applications/microwave/task1_visual.py::TASK1_STEP_ORDER` 定义，视觉会话 `Task1VisualSession` 使用 MuJoCo 场景对象，不能直接用作真机运行时。当前 `CuroboRuntimeBridge` 从 MuJoCo `model/data` 导出碰撞快照，同样不能直接接收真机观测。真机先复用 Knowledge/Skill 目标和 motion 请求契约，再独立建立现场碰撞快照。

旧项目真机行为以 `AffordToolAct/运行指南.md` 的 `scripts/run_microwave_task1.py` 实际调用链为参照。门的动作顺序是抓把手拉至约 25°、松开后从内侧推至约 90°，关门从外侧推；把手阶段之后可重新测门角。RealMan 原驱动 `robots/realman/realman_arm.py` 使用 SDK 关节角单位度、公共代码单位弧度，轨迹逐点调用 `rm_movej_canfd`。旧代码在 TCP 高度阈值处可能截断轨迹并继续后续 Task，也可能抬高 TCP 目标；新真机执行边界改为整段预检失败并停止，不静默改变目标。

新增的 `realworld/perception.py` 只读取现有 FoundationPose 输出，按 `T_sdk_object = T_sdk_camera @ T_camera_object` 转换。`pose.json` 的历史格式没有保存采集帧号，因此 `pose_frame_verified=false`；只有带 `_capture` 且与相机帧号和时间戳匹配时，才可设置 `require_matched_frame=True` 作为实时快照输入。历史 task_1 文件可用于离线坐标检查，不能据此宣称实时感知已通过。相机外参是对应机械臂 SDK base 下的 `T_sdk_camera`，双臂/仿真世界转换需另行标定。

新增的 `realworld/robot.py` 在 SDK 注入后可读取具名七轴状态，执行接口需要显式调用。完整轨迹必须匹配关节顺序、实测起点和 TCP 高度；SDK 失败会停止，不自动回 home。`realworld/runtime.py` 默认仅预检单个已规划阶段，`execute=True` 才调用运动接口。该离线边界尚未验收 SDK 实机连接、真机碰撞快照、夹爪控制、接触后置条件或完整 Task。

离线复核命令：

```bash
conda activate curobo
cd /home/ubuntu/Documents/yixuan/Gene_Agent
python -m pytest tests/realworld tests/motion/test_motion_model.py tests/applications/microwave/task/test_task1_visual_parity.py -q -p no:cacheprovider
python tools/diagnostics/inspect_realworld_snapshot.py --task-dir data/task_1 --side right --objects Bowl/Bowl_0 Microwave/Microwave_0 --output /tmp/task1_observed.json
```

`realworld/scene.py` 已提供纯几何公式 `T_world_scene_asset = T_world_sdk @ T_sdk_perception_object @ T_perception_object_scene_asset`，并拒绝未经同帧验证或缺少逐实例对齐矩阵的输入。它尚未给 task_1 标定实际对齐矩阵，也没有生成 cuRobo 碰撞快照。下一批应先让姿态估计输出携带同帧来源，标定 mesh 与碰撞资产的相对 frame 及桌面/支撑位置，再用实测 pose 生成可审计的 cuRobo 快照；完成已知碰撞/净空、FK/TCP 和单段 plan-only 检查后，才逐个 Skill 进行实机运动验收。真机专属抓取深度、夹持力、门接触偏移和速度应放在 `data/task_1/realworld` 配置并记录最终生效值，不修改冻结的仿真 profile。

特别注意：现有 `gene_agent/applications/microwave/object_frame_alignment.py` 的旧微波炉 frame 叠图结果明确标记 `accepted_for_planning=False`，只能作 visual 初值，不能直接填入真机资产对齐配置。

## 共享 Task 整理开始时的复核（2026-10-13）

本节对应共享 Task 与真机增量接入计划（历史材料已本地归档） A 批。工作树 HEAD 是 `9ee2b74e4478d669e63b792c2717e457d347ce3b`；开始时已有其他未提交修改，未 reset/stash。当前 Python 3.11.17、torch 2.5.1+cu121、MuJoCo 3.14.0、Viser 1.1.1。`curobo` 环境的 53 项公共规划、纯视觉 Task1 和真机离线定向测试通过。

已有完整运行 `runs/acceptance/microwave_task1_mixed_20261013/summary.json` 报告 `goal_success=true`、`fully_physical_success=false`、18 段规划推门直线轨迹；旋钮依赖运动学提交，碗有仿真辅助固定。该记录只证明当时单一名义场景的混合仿真。重新运行当前入口到 `/tmp/gene_agent_shared_task_baseline_20261013`，在 `open_first/handle_pull/01_pregrasp` 的 cuRobo 初始化处停止：`RuntimeError: No CUDA GPUs are available`。PyTorch 显示 `cuda_available=False`、设备数 0，`nvidia-smi` 无法连接驱动。失败发生在规划前，不能据此判断门轨迹有回归，也不能宣称当前 GPU 动态验收通过。

本轮通用 Task 运行器及 RealMan 夹爪离线适配加入后，定向测试为 **68 passed**。尚需在有 GPU 的环境重跑 Viser 默认与显式 90° 两种配置、完整 MuJoCo 任务，保存同视角视频和阶段/性能对照。任何进一步修改 MuJoCo 执行路径前，须先取得这组可运行基线。

真机夹爪阶段已补充只验证/显式执行接口；关节运动中途失败时记录即时实测关节。该离线改动后定向测试为 **68 passed**，没有下发真实 SDK 指令。

真机连接对象仅在显式调用时创建 SDK 连接并配置夹爪行程；注入假 SDK 的握手/断开与 Ctrl+C 关节流急停均已通过离线测试。现场连接、规划碰撞资产与任务成功仍待验证。

## 现场碰撞影子离线切片（2026-10-13）

新增 `gene_agent/applications/microwave/realworld_scene.py::build_observed_collision_scene`。它先要求 FoundationPose 物体 pose 带同帧来源，并逐实例使用显式的感知 mesh → 场景资产 frame 对齐；桌面完整 pose、尺寸、厚度以及微波炉支撑尺寸与局部中心由调用方确认。两臂七轴弧度状态、两侧夹爪宽度、门角、旋钮角也必须显式提供。函数只构建现有 MuJoCo 资产并调用 `mj_forward` 计算几何，再复用原 `export_collision_scene`；**不调用 `mj_step`，不把仿真落稳、接触或物体名义摆位当成现场事实**。真机影子暂拒绝仿真专用转盘/炉底高度补偿。

测试中的合成同帧样本生成 `/tmp/gene_agent_observed_collision_demo_20261013/`：`collision/collision_snapshot.json` 含 168 个导出障碍、106 条逐 geom 排除记录；包含碗、微波炉/门、桌面、支撑、本体及冻结的左臂和左夹爪。`observed_scene_provenance.json` 记录原观测来源、两臂状态、资产配置哈希与碰撞快照哈希。合成样本把 frame 对齐矩阵设为单位阵，只检验数据流，**不证明真实资产对齐或现场碰撞代理准确**。

定向测试命令：`conda run --no-capture-output -n curobo python -m pytest tests/tasks/test_shared_tasks.py tests/applications/microwave/task/test_task1_visual_parity.py tests/realworld/test_gripper.py tests/realworld/test_scene_shadow.py tests/motion/test_motion_model.py -q -p no:cacheprovider`；本轮 **53 passed**。测试覆盖实际导出的碗底、桌面和两臂/夹爪 qpos，以及缺少同帧来源、遗漏另一臂、未经核对的支撑尺寸和门角越界。下一步仍须用真实标定、已知碰撞/净空样例和 plan-only 验证现场快照，再接单个 Skill；这一步不是实机成功。当前环境没有 CUDA，Viser/MuJoCo 动态等价仍未重验。

## 现场快照到 Skill/Motion 的 plan-only 接线（2026-10-13）

`gene_agent/realworld/planning.py::prepare_plan_only_context` 接受上述现场快照和长生命周期 `CuroboPlannerPool`，返回现有 `CuroboPlannerAdapter`、`PlanningSceneRef`、活动臂实测 `JointState` 与 planner 缓存/初始化耗时。它复用 `load_realman_motion_config` 的原始 URDF、TCP、夹爪球和自碰撞配置；未新建运动求解算法，也不生成 Skill 目标或发送 SDK 指令。接线前核对同帧来源、碰撞快照 hash、另一臂冻结、活动夹爪快照与 world frame。场景或实测状态变化后必须重新导出快照。

无 GPU 的真实资产导出＋替身 planner 集成测试表明，`SkillContext` 可直接消费返回的 scene/start，且旧配置的 gripper 快照传到 planner 创建入口；同一快照复用 pool 不重复创建 planner。本轮 56 项相关测试通过。当前 CUDA 设备数为 0；尚未完成 cuRobo 的实际 IK/路径、门子任务或真机动作。

## 门 Skill 目标和阶段碰撞的共享切片（2026-10-13）

将原 `DoorDebugSession._make_knowledge` 的参数装配提到 `applications/microwave/door_knowledge_factory.py`，Viewer 现在调用同一函数；Knowledge/Skill 的候选、角度、局部旋转和顺序未改。`door_phases` 可接单个门 Skill，也保持原三段组合。`door_stage_scene.py` 提取原 Viewer 的阶段碰撞范围：把手阶段关闭微波炉及支撑 world，推门接触阶段将可动门板移出静态规划 world，其他障碍继续保留。

`realworld_door.py` 可从同帧现场快照生成独立的 `handle_pull`、`push_open`、`push_close` 目标；它用私有零角门几何生成 Knowledge，但让实际碰撞快照保持观测门角，避免对门板角度应用两次。逐阶段碰撞快照另存原观测与父快照 hash。默认 Viewer 三段完整阶段目标的冻结数值签名、现场 0°/25°/90° 目标和阶段碰撞清单测试均通过。物理门 profile 的纯目标构建得到 9/6/9 个阶段；尚未运行其 cuRobo 或物理执行。合成样本可查 `/tmp/gene_agent_door_shadow_demo_20261013/push_open_targets.json` 与 `contact_stage/collision/collision_snapshot.json`：六个推开阶段，接触段 164 个导出障碍、110 条排除记录，其中运动门板被明确排除。该样本使用合成同帧观测和单位 frame 对齐，不能代表真机标定通过。

本轮相关门、Task、真机适配测试合计 **97 passed**。现场门入口目前只交付目标与碰撞阶段事实，尚未交付真实 cuRobo 轨迹或 SDK 命令。显式逐角推门 profile 的 40° 后换点、用户覆盖的中间目标尚未进入现场入口；遇到这些配置会明确拒绝，不静默改动作。下一步在共享阶段规则下接 approximate IK、局部旋转候选和计划输出，再做 GPU 动态对照。

## 现场门阶段 plan-only 接口（2026-10-13）

`applications/microwave/door_stage_policy.py` 现在同时给门 Viewer 和现场阶段规划选择原有近似 IK 预算、局部旋转范围。`realworld_door_planning.py::plan_observed_door_stage` 只接受与当前碰撞快照、配置 hash 一致的门目标，导出阶段碰撞并从实测关节起点调用同一个 cuRobo 搜索函数，输出 `plan.json`、`result.json`。返回的成功只表示规划器报告成功且轨迹采样与时间间隔合法；尚不表示机器人已执行。成功时另存 `skill_stage.json`，可由 `load_planned_door_skill_stage` 还原并交 `RealWorldRuntime.run_motion_stage(execute=False)` 做现场起点预检。无效轨迹不生成可执行契约。门相关 35 项离线测试通过，扩大定向回归 **102 passed**；合成现场快照使用替身 planner 的贯通检查得到 `push_open/00_pre_contact`、168 个障碍，记录在 `/tmp/gene_agent_door_stage_work/integration_run/`。当前 CUDA 不可用，真实 cuRobo 规划及 Viser/MuJoCo 动态等价尚未验收。

## 单门 Skill 现场状态交接（2026-10-13）

同一 Skill 内复用初始 Knowledge 目标，通过 `rebase_observed_door_skill_targets` 绑定新实测碰撞快照；微波炉本体 pose 与资产锚点超过显式容差时停止。`run_observed_door_skill` 逐段调用现有规划和真机 runtime，执行后必须读回机器人/夹爪和门角，失败保留现场状态；默认只验证首段，绝不把未执行终点继续当下一段起点。把手拉门后的内侧推门仍是新 Skill，必须重新观测门角并生成目标。定向回归 **106 passed**；合成资产+替身 planner/robot 的记录在 `/tmp/gene_agent_door_stage_work/rebase_integration/` 和 `/tmp/gene_agent_door_stage_work/skill_preview_run/`。没有真实 GPU cuRobo、动态 Viser/MuJoCo 对照或真机运动证据。

## 旧逐角候选与开关门子任务（2026-10-14）

`door_candidate_selection.py` 只抽取 `visual_catalog` 原来两批 IK 筛选与选中候选写回阶段目标的纯逻辑，原 Viewer 和现场入口共用。现场候选选择的规划 world 暂移出活动门板；之后真正的 pre-contact 仍打开门碰撞。`run_observed_door_subtask` 只组合 `handle_pull`/`push_open` 或单独 `push_close`；执行与否由显式 `execute` 控制，未执行的终点不能衔接下一 Skill。扩大的定向回归 **110 passed**；合成场景+替身 planner 的候选记录在 `/tmp/gene_agent_door_stage_work/skill_preview_run/`。当前真实 cuRobo、动态 Viser/MuJoCo、真机门接触仍未验证；带 40° 后换点的旧 profile 现场版仍不支持。
