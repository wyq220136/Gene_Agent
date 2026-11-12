# 微波炉 Task1 的连续 MuJoCo 混合仿真验收

本次只串接已经在 Viser 中形成的动作：把手拉门约 25°、内侧推门到 90°、抓碗并按微波炉相对 keypose 放入、外侧推门关闭、旋钮、再次开门、取碗放到桌面、最终关门。没有重写门、抓取、旋钮或取放碗的 Knowledge/Skill 轨迹算法；只在任务场景与旋钮抓取 profile 中保存可复现的摆位/姿态参数。

## 入口与状态边界

- 用户入口：`python tools/run/run_microwave_task1_physical.py`。配置：`data/task_1/simulation/mujoco_task1_physical.json`。
- 任务装配：`gene_agent/applications/microwave/physical_task1.py`。场景只编译一次；每段共用同一个 `MujocoRuntime`，传递积分状态，不从名义 pose 重置机械臂、门或碗。`summary.json` 为每段保存起止 state ID，末端额外保存 `final_state.npz`。
- 门：继续使用 `ExperimentSession`/`DoorDebugSession` 与原 Knowledge；抓碗和放入使用现有 `check_pick_bowl_interface.run_experiment`、`physical_insert`；取碗使用现有 `physical_extract`、`physical_extract_retreat`、`physical_table_place`。Skill 间回默认 TCP 的动作保留为 Task 过渡。
- 旋钮：`rotate_knob_skill.run_rotate_knob_0_planning` 仍生成原阶段与 cuRobo 轨迹；`physical_knob.run_planned_knob_execution` 用 actuator 执行机器人动作。若接触物理没有转动旋钮，只在原 Viser 的最终提交位置更新旋钮关节，并记录提交前实际角度与 `kinematic_commit_applied=true`。该段属于混合仿真，不能计入物理旋钮成功。
- 失败即停，保留已完成前缀；`summary.json` 分别记录子任务和完整后置条件。门关节在物理开关阶段没有动画赋值或瞬移。仿真 weld 只在双侧指垫接触证据通过后启用，放置时释放。

## 名义场景与此次调参

- 微波炉世界 XY 为 `(0.52, 0.29) m`，碗中心初始 XY 为 `(0.255, 0.12) m`；微波炉朝向、碗朝向沿已有配置。门使用已验证的 close Knowledge `hinge_fraction_offset=-0.2`，不改变阶段顺序。
- 转盘和炉腔地板在编译前分别下移 `11 mm`、`5 mm`。这保持碗由转盘接触支撑，同时给取碗撤离留下净空；两者的 visual/collision 同步改变。配置及原因见 `docs/tmp/history/developer/mujoco_bowl_scene_probe_2026-10-13.md`。
- 旋钮只调整抓取姿态候选到 `azimuth=210°`、`tip X=10°`。在 `(0.52, 0.29) m` 摆位下，原单一 `240°/5°` 候选最终 grasp IK 不可达；原六方位中加入 tip X 候选后，`210°/10°` 按原旋钮阶段顺序规划到 45°。另外两个尝试的失败证据留在调试运行目录：`240°/10°` 最后 5° 碰关节限位，`300°/10°` 在约 25° 违反直线约束。没有为成功静默关闭碰撞或直线检查。

## 验收事实与限制

从 fresh reset 的连续记录保存在 `runs/acceptance/microwave_task1_mixed_20261013/`，包括编译场景、有效配置、阶段报告、终态、视频及子任务短片。此前工作树探针 `/tmp/gene_full_task1_20261013_try2/` 的 11 个子任务均通过，相邻交接的 state ID 无不一致；首次/再次开门约 `90.24°/90.23°`，两次关门回到约 `0°`，终态旋钮约 `45°`，碗在桌面且无残留 weld。旋钮提交前的实际物理角约 `0°`，因此完整任务明确标为 `assisted_pick_extract_and_kinematic_knob`，`fully_physical_success=false`。

逐段数值审查：18 段推门请求与实际规划操作均为 `linear_tcp`，没有 pose fallback；规划路径最大侧向偏差约 `1.80 mm`。MuJoCo 执行期间受接触与 actuator 跟踪影响，实测 TCP 最大侧向偏差约 `16.18 mm`、最大单步反向约 `1.75 mm`。这说明**规划轨迹受直线约束**，并不意味着接触中的实际末端轨迹严格是直线；两类指标分别写在 `push_motion_audit`，不能混用。

`record_physical_task1_video.py` 从 1616 个实际状态样本生成 12 fps 的全景和接触细节视频；每帧选择最近实际 qpos，不插值构造门角或碗位置。`split_physical_task1_video.py` 在同一视频上裁出开门、放碗、关门、旋钮、再次开门、取碗与最终关门七段。相机经过实际抽帧审核：全景能看见机器人/微波炉/碗，细节镜头分别对准把手、炉内碗和旋钮。录像和播放使用独立 MuJoCo data，不改在线执行状态。

当前验收只覆盖这一个 seed/名义摆位。位置扰动范围、批量成功率、无需辅助 weld 的夹持、物理驱动旋钮、在线物理 viewer、感知与真机仍未验证。后续首先用同一入口做有效初态与小范围摆位采样，保留全部失败样本；不能从本次 nominal 成功推断位置泛化。
