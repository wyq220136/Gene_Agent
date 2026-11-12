# Failure attribution

## Owner taxonomy

| Owner | Typical evidence | Meaning |
|---|---|---|
| `target_generation` | actual pose, target pose, TCP/object transform, candidate geometry | The request passed to planning does not fit the executed state or task geometry. |
| `planner_ik` | cuRobo or another IK result | No acceptable target joint candidate was produced under recorded search and constraints. |
| `planner_trajectory` | path optimizer result | A target candidate exists, but no complete trajectory was returned. |
| `collision_postcheck` | independent waypoint validation | A returned candidate path was rejected before execution. |
| `execution` | commanded versus actual joints, execution contact | The plan existed, but actual motion did not track it or encountered a forbidden physical contact. |
| `task_postcondition` | grasp, angle, support, placement, stability, release checks | Motion completed, but the intended task state was not achieved. |
| `wrapper` | operation exception or orchestration failure | The task stopped; follow references to find the inner cause. |
| `infrastructure` | CUDA, process, renderer, FFmpeg, missing artifact, diagnostic traceback | Runtime or evidence collection failed independently of task semantics. |

MuJoCo normally reports state and contacts; Gene Agent applies the allowed-contact or task condition. Phrase this as `MuJoCo contact evidence caused the Gene Agent execution/postcondition check to reject the phase`, not `MuJoCo decided the task failed`.

## Required causal form

Write each failure in this order:

```text
问题直接表现是 <human-observable task symptom>；
其触发了 <owner and rejected condition>；
详细情况为 <reason code, threshold, error, collision pair, candidate counts, or traceback>。
问题产生于 <upstream design or state-propagation cause>，
对应代码位于 <source file and symbol>；
修改该位置能够通过 <causal mechanism> 消除根因。
```

Example:

```text
问题直接表现是夹爪已经夹住物体，但机械臂没有将物体抬起；其触发了
post-pick lift候选全部淘汰条件；详细情况为17个候选中11个在有限搜索内
未得到目标IK，6个目标与机器人本体规划代理碰撞。问题产生于抓取后构型、
attachment与lift目标集合不匹配；直接修复对象是attachment条件下的lift目标覆盖。
```

## Deepening the analysis

After locating the rejecting condition, continue until the analysis reaches the upstream design cause and its code owner:

- IK failure -> audit target frame, target geometry, actual object state, gripper state, constraints, and the Knowledge function that generated the target. Do not start by increasing solve time.
- Path failure -> establish whether target IK exists, then inspect start state, corridor geometry, orientation continuity, attachment, first invalid waypoint, and the transition code that constructed the route.
- Execution tracking failure -> verify planning success, then inspect trajectory difficulty, discontinuity, joint-limit proximity, speed, duration, settling, controller state, and the phase-local execution request. Do not start by raising the error threshold.
- Contact/postcondition failure -> inspect video pose and actual contacts, then audit target contact geometry, allowed-contact semantics, condition meaning, and the code that classifies the contact.
- Placement failure -> inspect actual held-object transform, release state, support, footprint, tilt, settling, retreat displacement, and the target constructor that maps the held object to its final pose.

Do not list every possible cause. Select the cause supported by cross-layer evidence and name the source function that produces it. If a helper failed, state the failure and stop attribution below that point. This is precise, not vague.

## Root-cause adequacy test

Before accepting a cause, ask:

1. Does it explain the visible behavior and the rejecting condition?
2. Does it explain why this target, state, or transition differs from successful cases?
3. Does the proposed code change prevent recurrence without weakening a valid condition?
4. Can the cause be falsified by a same-state comparison?

If the answer to question 3 is no, the proposed change is a workaround rather than a root-cause repair.

## Hypothesis discipline

For each proposed cause, record:

- predicted observation if true;
- existing supporting evidence;
- existing contradicting evidence;
- single-factor comparison that can falsify it;
- whether the comparison tests a local phase or the complete task.

Examples of useful controls:

- actual object pose versus stale nominal profile;
- position-only target update versus full rigid target regeneration;
- robot-only collision checking versus full scene plus attachment;
- planner success versus actual execution and endpoint error;
- expected pad contact versus forbidden palm/wrist/arm contact;
- identical initial state before and after a target-generation change.

## Repair guidance boundary

Name the responsible input, contract, and code owner, not an arbitrary parameter value. Implement `regenerate the grasp target from actual object geometry` rather than `always offset 6 mm`; implement `reorient outside the narrow passage` rather than `always move 40 mm`; change phase-local execution settling only for a demonstrated tracking failure rather than globally weakening the endpoint threshold.

Treat solver budget as a repair only after the target is proven correct and feasible under the original strict constraints. Treat a collision or postcondition rule as editable only after its physical and task meaning is analyzed; preserve the rule when it correctly protects safety or success.

Do not call a modification effective until the original failed state improves and a fresh-reset downstream regression passes its unchanged success conditions.
