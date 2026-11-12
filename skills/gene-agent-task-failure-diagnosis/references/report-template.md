# Diagnostic report template

## Run identity

- Task and run:
- Seed or initial-state fingerprint:
- Canonical stage map:
- Stage -> operation/sub-task -> Skill -> phase:
- Complete task, resumed phase, or frozen-request replay:

## Failure statement

> 问题直接表现是 **[observable symptom]**；其触发了 **[owner and rejected condition]**；详细情况为 **[reason code and numeric/tool evidence]**。问题产生于 **[upstream design or state-propagation cause]**，对应代码位于 **[source file and symbol]**；修改该位置能够通过 **[causal mechanism]** 消除根因。

## Evidence table

| Layer | Finding | Exact evidence |
|---|---|---|
| Task symptom | What a human sees | video path, timestamp/frame, view |
| Stop location | First failed stage and phase | summary/report field |
| Target/state | Actual versus requested state | JSON/NPZ field |
| Planner | IK, constraint, or path result | plan/result field |
| Postcheck | First invalid waypoint or rejection | diagnostic field |
| Execution | Executed or not; max/endpoint error | execution field |
| Contact/postcondition | Actual contact, angle, pose, support, stability | result field |
| Infrastructure | Independent helper or capture error | traceback |
| Source owner | Function that creates the target, transition, policy, or condition | source path and symbol |

## Cause analysis

- Failure owner:
- Direct rejecting condition:
- Immediate technical mechanism:
- Upstream design/state-propagation cause:
- Exact code owner and live consumer:
- Condition meaning and whether it is reasonable:
- Explanations rejected by evidence:
- Diagnostic artifact gaps:

Write declarative conclusions. If a deeper attribution is unavailable, write the concrete reason: for example, `the collision helper failed before emitting an object pair, so the recorded failure is trajectory optimization rather than confirmed collision.`

## Root-cause modification and validation

- Smallest responsible source layer:
- Intended code change:
- Why this removes the cause instead of hiding the error:
- Valid conditions and prior fixes preserved:
- Predicted observation after a correct change:
- Minimal single-factor validation:
- Preserved controls:
- Required downstream/full-task regression:
- Actual validation result:

When the user requests a fix, include the exact changed file/function and validation result. Do not stop at a recommendation.

## Batch summary table

| Run | Stage / phase | Visible symptom | Owner | Rejected condition | Numeric/tool detail | Root cause | Code owner / modification | Evidence |
|---|---|---|---|---|---|---|---|---|

For successful runs, use `-` for failure cause. Do not invent a problem for a successful run.
