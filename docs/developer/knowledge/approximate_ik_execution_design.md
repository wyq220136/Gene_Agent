# 近似 IK 的规划与执行契约

读者：实现 Knowledge、Motion、Skill 的开发 Agent。状态：最小接口已实现，真实 cuRobo 近似失败目标的 GPU 回归和 MuJoCo 完整 Skill 验收仍待完成。默认严格；以下目标与负例也是后续验收清单。

## 目标

推门、pre-contact、pre-grasp 等阶段不必一律要求目标 TCP 精确到严格 IK 阈值。阶段显式启用近似策略后，可接受误差预算内的替代 TCP 姿态，并用正常碰撞/轨迹规划生成动作。结果须让下一阶段知道实际到达姿态与原目标的偏差。接触抓紧、插入和任务终态仍可保持严格要求；不同阶段的误差预算分开配置，默认严格。

## 当前基础

`gene_agent/motion/curobo/core.py::summarize_ik_failure` 已从失败的 cuRobo IK 返回各 seed 的 `solution`、`feasible`、平移/转角误差。`gene_agent/motion/curobo/approximate.py` 在严格 `plan_pose` 因目标 IK 无成功种子失败后，才筛选候选、复核碰撞、用实际 FK pose 二次规划。`MotionRequest.approximate_ik_budget`、`GraspSettings.pregrasp_approximate_ik_budget` 和 `CuroboRuntimeBridge.approximate_ik_by_stage` 提供显式启用入口。`gene_agent/visualization/grasp_tuner/pose_ik.py` 的黄色目标夹爪和最近姿态是另一套 MuJoCo 运动学预览，不能用其关节角直接替代 cuRobo 规划结果；Knowledge 页面仅用 `gene_agent/knowledge/reachability.py` 标记纯位姿近似候选。

## 接口与判定

1. Motion 请求携带可选的近似预算：平移上限（m）、转角上限（rad）。阶段名称由 Skill/Runtime Bridge 记录；后置条件仍属于 Skill，尚未纳入统一预算对象。未设置策略时行为与当前严格路径完全一致。Knowledge 的调参页面用 mm/度显示，保存时换算为 m/度；它不直接依赖 cuRobo。
2. 严格规划若因目标 IK 无成功种子失败，才考虑近似候选；路径碰撞、起点碰撞、请求错误、后处理碰撞失败不能触发此退让。候选须数值有限、满足阶段误差预算，关节限位和碰撞检查通过。若现有选中种子不满足，检查其他返回种子，而不只看第一个。
3. 候选的**实际 FK TCP pose**成为新的规划目标。正常 cuRobo 规划仍须成功，起点、终点和整条轨迹仍须通过相同碰撞策略和后处理。不能把失败 IK 关节解当作轨迹、修改原始目标、放松碰撞或将失败结果的 `success` 强行改为 true。
4. 规划结果同时保存 `requested_tcp_pose_world`、`planned_tcp_pose_world`、`planned_joint_state`、`position_error_m`、`rotation_error_rad`、`strict_ik_success`、`constraint_feasible`、`approximate`、阈值、候选 ID、原始失败及二次规划诊断。`success=true` 只表示替代目标的轨迹规划成功；原目标未精确达成必须独立标记。
5. Skill 对每个阶段选择策略。Runtime Bridge 已在近似路径执行后查询**实际** TCP，计算相对原目标的误差，超预算则阶段失败；下一阶段从实际状态重新规划。物体/门操作后置条件仍由对应 Skill 负责，尚未统一接入全部 Skill。回放同时显示原目标、替代目标和实际终点三个姿态及误差来源仍待实现。

## 最小验收

- 同一场景、同一 seed 下，严格失败而误差预算内且无碰撞的 pre-contact 能返回带 `approximate=true` 的**完整可执行轨迹**；禁用策略仍按旧逻辑失败。
- 超出预算、关节/场景碰撞、原规划是路径碰撞的负例都不能被退让成成功。
- Skill 运行记录同时显示原目标与实际终点；后置条件失败不得算阶段成功。Knowledge 检查器能查看同一预算与误差标记。
- 最终用新 reset 的 MuJoCo 完整 Skill 回归，而不只用失败快照重试。GPU 规划和 MuJoCo 接触结果须分别报告。

## 2026-10-09 实测边界

`tools/diagnostics/check_motion.py --warmup-runs 0 --measure-runs 1` 在本机 GPU 上通过 FK、严格 IK、严格规划与起点碰撞负例，记录于 `runs/approximate-ik-motion-baseline/`。随后用当前微波炉右臂开门的 pre-contact（30° 门角）作一次真实 cuRobo 近似规划探针，预算为 20 mm/15°，记录于 `runs/approximate-ik-precontact-probe/result.json`。该探针仍返回 `goal_ik_no_solution`：16 个 cuRobo IK seed 的 `feasible` 全为 false，最优 seed 的位置误差约 60 mm；因此当前实现正确拒绝了退让，没有生成轨迹。这不能作为近似规划成功或任务成功证据。下一步需在明确无碰撞、误差预算内但严格 IK 不收敛的场景上做正例 GPU 回归，再进行 MuJoCo 完整 Skill 验收。
