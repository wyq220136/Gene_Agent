# Pick bowl 的 cuRobo 失败诊断

一次规划请求分两层：先检查起始关节姿态，再执行目标 IK 或轨迹规划。报告中每个 `ik_attempts[].result.failure` 的 `kind` 是稳定分类；`details.solver_diagnostic` 保存 cuRobo 当次求解给出的数值证据。不要仅凭 `feasible=False` 断言碰撞，因为它还包括关节限位等约束。

起点先运行优化器使用的宽泛 mounting mask，再用 `MountingCollisionValidator.check` 复核完整自碰撞与外物碰撞。只有显式列出的安装 link 与本体障碍物配对可接触；近基座 link 碰到碗、桌面等外物不能因优化期 mask 而放行。相同场景、相同关节起点缓存一次严格结果，40 个候选不重复初始化诊断。起点失败时 `collision_verification.start_check` 保存严格各组成本和逐物体名称；`allowed_mounting_contacts` 另列允许接触，不放进碰撞失败原因。

用名义起点在 `link_1` 碰撞球中心构造过一个边长 2 cm 的外部 cuboid：优化期 mask 返回 `optimizer_valid=True`，严格复核返回 `external=0.06789`、`valid=False`，逐物体定位为 `external_test_near_link_1`。原始结果保存在 `runs/pick-bowl-start-collision-diagnostic.json`。名义场景不加这个物体时严格检查通过，仍得到 30/40 个 IK 成功和成功轨迹。起点错误的短消息会显示自身碰撞、外物名称或关节限位；Viser 的「起点碰撞预检」直接使用该消息。

IK 失败时记录最优候选的位置、姿态误差及当次求解器阈值，并分别标记 `pose_converged`、`constraint_feasible`。`pose_not_converged` 表示 **这次种子与优化设置没有收敛**，不是数学上证明不可达。若候选姿态已收敛但约束不满足，再用同一场景与 mounting mask 复查该关节；只有复查实际检出场景或自身碰撞时才标为 `goal_collision`。场景碰撞逐个启用导出的障碍物，给出 `colliding_obstacles`；关节限位和自身碰撞独立记录。诊断异常写入 `collision_check_error`，不会伪装成无碰撞。

轨迹失败与目标 IK 失败分开。cuRobo 返回 `None` 时，当前 `max_attempts=1` 的实现表示没有找到成功的目标 IK 种子；诊断代码会再独立求一次 IK，记录所有返回种子的位姿误差与可行性。若依旧没有成功种子，可再做一次**只用于诊断**的空环境 IK，并将所得姿态在原场景严格复核；该结果绝不作为可执行路径。返回失败轨迹时报告 `path_not_found`，并检查求解器返回的离散关节样本。若样本确实碰撞，则改报 `path_collision` 与首次无效 waypoint。没有失败轨迹样本时，不能推断是哪件物体挡路。离散检查也不能证明连续轨迹绝对无碰撞。不可达的 10 m 目标示例保存于 `runs/pick-bowl-unreachable-plan-diagnostic.json`，其中规划内部 IK 无成功种子，空环境试算同样无解。

汇总字段 `ik_diagnostics.failure_counts` 按稳定类别统计，`reason_counts` 按原始证据统计，`colliding_obstacle_counts` 只统计经过几何复核的物体。当前名义场景 40 个候选中 30 个 IK 成功；10 个失败均是 `pose_not_converged`、`constraint_feasible=True`，所以没有已证实的碰撞。典型候选 `bowl_ring_00_03` 的最优位置误差为 7.88 mm，超过 5 mm 阈值。

复现：

```bash
cd /home/ubuntu/Documents/yixuan/Gene_Agent
/home/ubuntu/miniconda3/envs/curobo/bin/python tools/run/check_pick_bowl_planning.py --output runs/pick-bowl-diagnostics-new
```
