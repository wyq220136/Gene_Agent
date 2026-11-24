# 真机微波炉任务入口

完整流程见[真机操作指南](realworld_task1_operation_guide.md)。现场提供标定结果、感知服务和设备地址；程序自动装配配置，**不需要手写站点或角度状态 JSON**。操作 Skill、Knowledge 和 keypose 算法沿用原实现。

## 准备感知

先启动相机、SAM3 和 FoundationPose 服务，并完成右臂 SDK base 下的相机标定：

```bash
python tools/run/prepare_microwave_realworld.py \
  --camera-params data/calibration/latest/manual/cam_param.json \
  --output runs/realworld/perception_current
```

默认 `no_knob`。输入 `CLOSED` 确认关门、柜体和相机固定，无需旋钮置零；核对碗、把手两个 mask 和计算出的桌面结果。桌面使用已有矩形范围和标定后的场景深度计算，不分割、不做 pose estimation。程序只注册碗、把手两个 session，并生成 `perception_setup.json`，该文件用于复用感知结果，无需编辑。正式 CLI 不读取旧 `realworld_site.json`，也不再提供 `--config` 参数。

## 检查和运行

```bash
python tools/run/run_microwave_task1_realworld.py \
  --perception runs/realworld/perception_current \
  --left-ip 169.254.128.18 --right-ip 169.254.128.19 \
  --subtask pick_bowl --check-config
```

按现场填写 IP。端口默认 8080，可用 `--left-port`、`--right-port` 覆盖。
去掉 `--check-config` 后默认只做首段预检。完整任务预览使用 `--subtask all --plan-only`；显式添加 `--execute --yes` 才允许运动。运行时用 `--output` 指定证据目录。

| 子任务 | 说明 |
|---|---|
| `open_door` / `close_door` | 已有拉门、推门 Skill，使用新帧 tracking 测角 |
| `pick_bowl` | 从桌面抓碗 |
| `insert_bowl` | 独立运行时用 `--prepare-held-bowl` 确认已持物 |
| `rotate_knob` | 当前默认模式禁用 |
| `extract_bowl` | 从炉内取碗 |
| `place_bowl_on_table` | 独立运行时用 `--prepare-held-bowl` |
| `load_bowl` | Pick → Insert，内部传递持物状态 |
| `unload_bowl` | Extract → Table Place |
| `all` | 默认跳过旋钮，保留第二次开门前的过渡动作 |

准备和运行均默认不感知、不操作旋钮，可省略 `--no-knob`，对单个子任务也生效。以后若启用旋钮，需在准备和运行两处显式使用 `--with-knob`，并单独完成接触验收。当前使用右臂 SDK base、标准夹爪和对应任务资产；两臂 IP 不能相同。

## 力控闭爪与状态交接

真机抓取沿用 AffordToolAct 的 `pick_on` 连续力控闭爪。命令返回后先等待配置的
闭爪稳定时间（默认 1.5 秒），再读取终点开口并进入新帧观测和夹持检查。
取碗与旋钮复用各自阶段的等待配置，已在闭爪阶段完成的等待不会重复执行。
力控夹持的实际开口由物体决定，不要求等于规划中的几何闭合宽度；正常闭合期间
的宽度变化也不作为意外漂移。稳定后的异常开口变化、机械臂漂移、过期观测和
无效读数仍会阻止继续，等待本身不表示夹持成功。只规划模式不下发命令或等待。

## 实时状态与边界

每次观测使用同一新 RGB-D 帧跟踪碗与把手，再计算门角；不注册或跟踪旋钮。碰撞中的旋钮保留资产零角的静态代理，标为未观测，不代表实测状态。不再依赖人工更新门角文件；真实过期帧仍会被默认 5 秒时效检查拒绝。
柜体必须固定，几何一致性检查不等于独立柜体测量。Pose 服务重启、输入改变、柜体或相机移动后需重新准备，必要时重新标定。

桌面沿用 AffordToolAct Task1 的固定范围：右臂 SDK base 下中心 `[0.1, 0.3] m`、尺寸 `0.6 × 0.6 m`、碰撞厚度 `0.02 m`；仅高度从本帧深度计算，无需桌面 mask、完整边界入镜或新增手填 JSON。切换后在新目录重新准备感知。固定尺寸和支撑代理需与现场核对；输入校验和 plan-only 通过均不代表真实接触、整条任务通过。
FoundationPose 可继续使用 Python 3.9；依赖环境保持独立，详见操作指南。

执行目录中的 `runtime_input.json` 保存自动配置及来源。任务接线位于 `gene_agent/realworld/microwave/task.py`；模块边界见[真机模块说明](../../../gene_agent/realworld/README.md)。

## 完整任务规划预览

默认不加执行参数仍是“首段预检”，会返回 `validated_prefix`。完整预览需显式使用
`--subtask all --plan-only`（兼容 `--plan_only`）；它与 `--execute` 互斥。
只读取一次新的现场感知及双臂/夹爪状态，之后在私有运动学影子里沿 cuRobo 返回的
真实轨迹末帧继续下一段，不再调用相机或 SDK，不设置夹爪行程、不发送命令。
两轮开关门、Pick、Insert、Extract、桌面 Place 及全部默认 TCP 过渡沿用真机步骤表、
目标生成和各阶段碰撞策略。默认仍跳过旋钮；原有仅自碰撞/机器人本体等接触规划层保持
原有范围，记录中的停用障碍仍需在实际执行前按原流程审核。

```bash
MUJOCO_GL=egl "$PY" tools/run/run_microwave_task1_realworld.py \
  --perception "$PERCEPTION" \
  --left-ip "$LEFT_IP" --right-ip "$RIGHT_IP" \
  --subtask all --plan-only --video \
  --output runs/realworld/task1_full_plan
```

`--video` 可选，生成 `plan_preview.mp4`；失败前缀也可导出。输出目录须为空。
`task_result.json` 中 `status=planned` 和 `planning_success=true` 才表示所有选中步骤
规划完成；`planning_failed` 会保留 `first_failure`、全部成功前缀及原规划器证据。
始终 `execute=false`、`task_success=null`；不把规划完成写成真机操作成功。
夹持、门/旋钮随动、释放留在预测位姿均为显式 `assumptions`，没有动力学落稳或真实接触证据。
预测不修改初始现场帧号/时间，另用 `prediction_step` 表示规划状态变化。

浏览器三维回放（可拖时间轴、选择阶段和显示碰撞）：

```bash
"$PY" tools/viewers/view_skill_run.py \
  --run runs/realworld/task1_full_plan --mode planned \
  --host 0.0.0.0 --port 8092
```

重新导出视频无需重新规划，也无需连接真机：

```bash
MUJOCO_GL=egl "$PY" tools/viewers/recording/record_skill_run_video.py \
  --run runs/realworld/task1_full_plan --mode planned \
  --output runs/realworld/task1_full_plan/plan_preview.mp4 \
  --camera task --camera gripper_side --profile microwave_task1
```

旋钮仅在准备及运行均显式 `--with-knob` 时纳入；若所用 profile 包含未经轨迹规划的
直接末关节 SDK 命令，预览会报告 `unplanned_direct_joint_command` 并停在该处，
不能把未规划路径说成完整碰撞规划通过。实际执行仍逐段读取现场、重规划和验证接触，
不能将预览预测状态、夹持假设或历史轨迹直接作为执行证据。


## 真机失败定位和现场重放

原运行命令默认启用失败诊断，仍须给新 `$OUT`，不加 `--execute --yes` 即保持原来的只规划范围。
可用 `--diagnostic-budget-s 5` 设置每次失败的软预算；`--diagnostic-mode off`
只关闭补充 FK/约束诊断，原始请求、返回值、候选清单和初始场景仍保存。
日志接线复用 `Gene_Agent_new` 仿真中的通用 Motion 诊断和冻结请求机制。

- `failure_analysis.md/json`：首个异常、每条实际规划请求的分类、误差和详细证据引用。
- `events.jsonl`、`motion_results/`、`diagnostics/`：逐请求追加；原始错误和诊断错误分开。
- `grasp/selection/candidate_diagnostics.json/md`：全部生成/排序候选、参数和 TCP、每个 pregrasp 分支的实际结果、拒绝数值及未到达阶段。某分支失败后另一分支成功，不把整个候选误报为拒绝。
- `motion_requests/`：完整原 batch、起点、目标、模型、夹爪、碰撞策略、seed、约束和依赖签名。
- `initial_scene/`：自包含可移动目录，保存第一帧原感知与 RGB-D、标定/pose 来源文件、两臂 q/夹爪、桌面、代理模型、资产哈希、项目源码版本；后续候选筛选场景另存，不覆盖最初现场。`manifest.json` 的 `file_map` 对照原路径和包内资产。

```bash
# 只读分析；无需 GPU、相机、Pose 服务或 SDK。--details 显示全部请求。
"$PY" "$PWD/tools/diagnostics/inspect_realworld_run.py" --run "$OUT" --details
# 精确筛候选或阶段，也可 --format json 供 Agent 读取。
"$PY" "$PWD/tools/diagnostics/inspect_realworld_run.py" --run "$OUT" --candidate real_d00_bowl_ring_03_00 --stage final_grasp_ik

# 把 initial_scene 整目录拷至仿真机后，验证文件并恢复原 qpos；--render 输出静态代理图。
"$PY" "$PWD/tools/diagnostics/replay_realworld_scene.py" \
  --bundle "$OUT/initial_scene" --render --output "$OUT-scene-import"
# 重跑原生成候选与原 profile，使用该次 selection 对应的现场。
"$PY" "$PWD/tools/diagnostics/replay_realworld_scene.py" \
  --bundle "$OUT/initial_scene" --action pick_bowl --selection 0 --output "$OUT-pick-replan"
# 严格重跑任一冻结请求；下标对应 manifest.requests，也可给其中的文件名。
"$PY" "$PWD/tools/diagnostics/replay_realworld_scene.py" \
  --bundle "$OUT/initial_scene" --action request --request 0 --output "$OUT-request-replan"
```

模型与状态也可直接导入 Python：

```python
from gene_agent.realworld.scene_archive import load_scene_archive
scene, model, data = load_scene_archive("<initial_scene>")
# model/data 是 MuJoCo 对象；data.qpos 为原实测状态，尚未 mj_step。
```

候选重跑保留原候选、顺序、profile 和现场；代码改变时须显式 `--allow-code-change`，
输出记录变化列表。严格请求重跑还验证 Python/cuRobo 完整源码配置签名，资产只允许
经过验签的地址重定位，不能用该选项替换模型或依赖。`--action request` 保留整个原 batch。
导入场景不要求启动原感知服务；它还原的是感知结果和规划代理，并未验证真实摩擦、接触或抓碗成功。
实际仿真 `mj_step` 会产生新的物理状态，应另开实验目录，不得覆盖此冻结证据。

每个失败候选记录实际返回 q、独立 FK、位置/转角误差、收敛阈值和可用约束证据。
只返回一个最优失败解时明确记 `returned_candidate_count=1`，不冒充已保存全部 optimizer seeds。
未生成路径会标 `not_returned`/未到达路径阶段；失败 seed 上的碰撞不等于目标位姿碰撞。
预算不足、缺帧、感知早退和写盘异常分别记录，`initial_scene_unavailable` 不能导入成假现场。
旧 run 的检查可提炼已有 attempts；缺失的原始图像、模型、候选参数和请求无法追补。

迁移完成与验收分开：最终完整性以真机实际运行的 `failure_analysis`、候选清单、冻结请求和 `initial_scene/manifest.json` 为准。离线导入或规划对照不作为真机验收。后端误差与独立 FK 误差分别保留；二者不一致时不相互覆盖。
