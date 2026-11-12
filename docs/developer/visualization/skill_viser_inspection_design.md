# Skill 独立检查：运行记录与 Viser 视图设计

受众：后续实现的 Codex。状态：2026-10-06 调研与实施设计，本文不表示功能已完成。第一目标是 `pick_bowl`，格式不能依赖 Realman、左右臂命名或 cuRobo。

## 已核对的代码和证据

`skills/model.py` 的 `SkillPlan` / `SkillStage` 可序列化目标、`MotionRequest`、`MotionResult` 和候选尝试。`skills/grasp.py` 先批量筛选最终 grasp IK，再规划 `home→pregrasp`，接着请求 `LINEAR_TCP`；`skills/pick.py` 追加 `lift`，但没有执行。当前 cuRobo adapter 对 `LINEAR_TCP` 和接触放宽不提供执行能力，因此不能把新 `PickSkill` 标为完整成功。

`visualization/pick_bowl/app.py` 的 `PickBowlPlanningApp` 有 visual/collision 开关、语义碰撞组、候选抓取预览、`抓取目标` 和 `Skill pregrasp` 两种轨迹来源、场景编辑和固定相机。它的 UI 与碗候选、当前场景 revision 耦合，尚不能选择任意 Skill 的任意阶段；`抓取目标` 是从 home 独立规划到 grasp，不接在 pregrasp 后。`visualization/pick_bowl/scene.py` 可以复用 MuJoCo visual 更新和碰撞快照显示，但其节点路径、分组推断含 pick bowl 特例。碰撞球目前只代表起点，播放后续帧时隐藏，不应当误说它随机器人运动。

`motion/model.py` 已定义 `ik / plan / linear_tcp`、显式起点、scene 引用、接触允许项、轨迹、失败类别和后端状态。`motion/runtime_bridge.py` 的旧顺序执行每段从当前 MuJoCo 状态查询起点、导出新碰撞快照、规划、执行、写入 `plans/<stage>.json` 和 `executions/<stage>.json`。但此旧桥的 `final` 是普通关节空间 `plan`，没有 TCP 直线保证。

可用实物证据：`runs/pick-bowl-live-recording-two-view-20261006/` 含 `result.json`、各阶段 `collision/*/collision_snapshot.json`、规划、执行、观测、阶段状态、录像帧及 MP4。`pregrasp` 的执行终点 state ID 与 `final` 的起点 state ID 一致。录像 manifest 的每帧有仿真时间和 state ID。执行 JSON 的机械臂采样有参考与实际关节角；开合爪采样有目标与实际开口。目前未保存**每个执行采样点的完整 MuJoCo qpos/qvel 和物体位姿**；只有阶段关键帧 state.npz 和图像，因此不能从这些日志精确重建所有执行中间帧的三维物体状态。第一版可播放录制图像与机器人关节近似回放，必须把近似标清；新记录应补充完整状态的低频采样或可回放状态文件。

## 数据边界

增加轻量只读 `SkillRunView`，作为反馈模块输出和 Viser 输入。不要让 Viser 解析 `PickSkill` 的内部 Python 对象，也不要将 Viser 的 GUI 状态写回 Skill 结果。`run.json` 只存索引和小字段，轨迹、执行采样、碰撞快照、截图继续放独立文件；使用相对路径和文件哈希。建议最小结构：

```json
{
  "schema_version": 1,
  "skill": "pick_bowl",
  "run_id": "...",
  "robot_group": "right",
  "scene_revision": 1,
  "asset_refs": {"model": "model/scene.xml", "robot_config": "..."},
  "selected_candidate_id": "bowl_ring_00_00",
  "stages": [
    {
      "id": "02_pregrasp",
      "kind": "motion",
      "planning_status": "planned",
      "execution_status": "completed",
      "motion_operation": "plan",
      "start_state_id": "...",
      "end_state_id": "...",
      "target_pose_ref": "...",
      "motion_request_ref": "...",
      "motion_result_ref": "plans/pregrasp.json",
      "execution_ref": "executions/pregrasp.json",
      "collision_snapshot_ref": "collision/pregrasp/collision_snapshot.json",
      "video_frames_ref": "visual/manifest.json"
    }
  ]
}
```

这是结构示例，不是承诺上述旧 run 已有所有字段。对规划但未执行的阶段，`execution_status` 必须为 `not_executed`，引用为空；对夹爪命令，`motion_operation` 应为 `gripper_command` 且不能伪造 cuRobo 规划请求；对纯 IK，不能显示轨迹。`SkillPlan.success` 只说明阶段已规划，不代表物理成功。场景编辑后要创建新 revision 和快照，绝不将旧轨迹套在新场景上。

阶段内记录这些可核查事实：

1. 输入：机器人组、实际起点的关节名称与数值及 state ID、目标 pose/frame、候选 ID、Skill 参数与 seed。
2. 规划：明确的 `operation`，结果状态、轨迹、耗时、backend 状态、原始错误引用；对 `LINEAR_TCP` 保存 TCP 轨迹的直线偏差与姿态误差，不能只信请求名称。
3. 碰撞：当前阶段快照哈希；启用和排除的物体/geom、导出类型、mesh/cuboid 路径、机器人球配置、gripper 开口快照、允许接触的 link 对及原因、attachment 状态。`CollisionPolicy` 是请求，实际后端是否执行及近似情况需单独记录。
4. 执行：轨迹参考与实际采样、阶段起终 state ID、物理接触/夹持检查、图像及完整状态样本的引用；保存 `planned` 与 `actual` 的差异指标。

碰撞清单从**该阶段原始快照**读取，不能由 Viser 当前 MuJoCo 状态重新导出。先按对象组显示启用数量、代理种类、排除数量，再展开到具体碰撞文件和 hash。用半透明面显示 mesh、线框显示简单 box/球，保持 visual 与 collision 的独立开关；允许接触和实际碰撞点另用颜色/连线标注。不要把 MuJoCo 物理接触几何与 cuRobo 规划代理当作同一层。

## Viser 交互与模块分工

新增只读入口，例如 `tools/viewers/view_skill_run.py --run <目录>`。`feedback` 负责把新旧报告规范化为 `SkillRunView`、校验引用和状态链；`visualization` 只做加载、绘制、时间轴和截图；`simulation` 负责实际状态采样与物理判定；`motion` 返回规划与碰撞诊断；`skills` 决定阶段和接触策略。首版为旧 run 写一个小适配函数即可，避免迁移整个旧桥或在界面里铺大量 `if skill == pick_bowl`。

界面按一次 Skill 展示：阶段列表及总状态；当前阶段的起点、目标、轨迹类型、结果；三维场景与本阶段碰撞快照；局部时间轴；诊断卡片。时间轴默认显示规划点，提供实际采样和叠加模式；两个模式缺数据时禁用并注明原因。阶段切换时一并切换碰撞快照、目标 frame、轨迹、统计和相机推荐位置。`open / close / attach` 这类无规划轨迹的阶段显示命令和前后状态；`IK` 只显示候选目标与解姿态。提供 overview、gripper 掌心、目标近景、碰撞视角，按当前阶段目标自动聚焦。保留现有单候选 grasp 预览，但让颜色/标签区别“候选 ghost”“规划目标”“实际机器人”。

失败诊断先显示稳定分类（起点碰撞、目标 IK 无解、目标碰撞、轨迹失败、路径碰撞、后端不支持），再显示后端原始状态和证据。若碰撞证据可定位到帧，则时间轴跳至首个无效帧，突出相关 link、物体组和接触点；若无法归因，显示“未知：缺少哪项证据”，不伪造碰撞原因。分别报告规划耗时、执行时间和 Viser 加载耗时。

录像无需成为 Viser 的真相来源：当前 `TrajectoryRecorder` 已有逐阶段图像与 MP4。第一版在 Viser 中可按阶段帧索引显示已有图像/视频引用，并与三维时间轴对齐；后续完整状态采样到位后再做精确三维执行回放。截图工具复用现有 Chrome CDP 等待 ready、固定相机、PNG/hash/manifest 的流程，只增加 `stage_id`、模式、帧索引、scene hash 和碰撞快照 hash。保持按需启动 Viser，离线报告与失败统计不依赖浏览器。

## 最小实施与自动验收

1. **统一记录，不重做规划。** 为当前 `SkillPlan` 和旧实际执行 probe 各写适配器，输出一个有序阶段清单。校验每个引用存在、场景 hash 匹配、阶段 ID 唯一、`request.operation` 与显示标签一致、成功轨迹的关节顺序与时间合法。旧 probe 的 `final` 必须标为普通 `plan`。运行一个只读 CLI 打印阶段摘要，便于 Codex 不开 Viser 即检查。
2. **Viser 阶段查看。** 基于现有场景渲染器添加阶段选择、目标 ghost、局部规划时间轴、逐阶段碰撞快照和可折叠碰撞组。接受现有 `PickSkill` 规划 run 时能依次看到 `final_grasp_ik / pregrasp / approach / close / attach / lift`，其中 `approach=unsupported`、后续未执行；接受旧 probe 时看见它自己的四个真实阶段。固定视角截每个关键阶段的 normal/collision 两图。
3. **实际执行查看。** 对旧 probe 关联执行样本与录像 manifest，先做参考/实际关节曲线或误差和图像同步；新记录时加入低频完整 MuJoCo 状态，三维回放才标“精确”。自动核对相邻阶段 `end_state_id == next.start_state_id`、录像帧时间单调、视频帧数与 manifest 相符。阶段起点不连续时拒绝“连续执行”标签。
4. **真正的 PickSkill 闭环。** 后端实现 `LINEAR_TCP` 和阶段接触约束后，再执行 `pregrasp→grasp→close→lift`，保存每阶段快照；测试 TCP 轨迹最大直线偏差、夹爪开口/物体抬升的实际状态和物理成功。直线接近失败必须报错，不悄悄退化成关节空间路径。只有整段通过才称 `pick_bowl` Skill 执行成功。

第一轮人工验收的核心动作：打开同一个 run，点击 `pregrasp` 与 `approach/final`，在 Viser 中看到不同的起点/目标、明确的 `plan` 与 `linear_tcp`（或旧 run 的 `plan`）标签、各自碰撞清单和路径；拖到执行帧时看到实测误差或清楚的“无实测数据”。此轮不以抓碗物理成功作为 UI 功能完成条件。
