---
name: gene-agent-task-failure-diagnosis
description: Evidence-driven root-cause diagnosis and repair of completed Gene Agent task runs using a canonical task-stage SVG, structured run artifacts, actual execution video, source-code tracing, and same-state regression. Use when an agent must locate the exact failed stage and phase; distinguish target generation, cuRobo IK or trajectory, path postcheck, execution tracking, MuJoCo contact, task postcondition, wrapper, and infrastructure failures; judge whether a simulation condition is semantically valid; extract failure-time frames; prevent stage-name drift; build a stable task-stage SVG; identify and modify the responsible code; or verify that a root-cause fix makes the original failed run and complete task succeed.
---

# Gene Agent task failure diagnosis

Diagnose after an experiment has produced logs and run artifacts. When the user asks for a fix, continue from diagnosis into the responsible source code and validate the repair. Preserve historical evidence and valid safety/task conditions.

## Single principle

Correct the upstream code that produces the failure so the next equivalent run can succeed. Do not hide the failure by weakening a valid condition, extending a budget before checking the target, or suppressing the error.

## Load the references

1. Read [stage-map-and-anchoring.md](references/stage-map-and-anchoring.md) before naming a stage. For microwave Task1, always use `docs/01-task-skills.svg` as the fixed canonical map.
2. Read [evidence-routing.md](references/evidence-routing.md) before opening run artifacts.
3. Read [visual-diagnosis.md](references/visual-diagnosis.md) before claiming an observed pose, collision, missed grasp, or abnormal motion.
4. Read [failure-attribution.md](references/failure-attribution.md) before assigning an owner or cause.
5. Use [report-template.md](references/report-template.md) for the delivered diagnosis.
6. Read [representative-cases.md](references/representative-cases.md) when diagnosing grasp, narrow-passage insertion, door-transition, contact, or execution-tracking failures.
7. Read [root-cause-repair.md](references/root-cause-repair.md) before changing runtime code or a simulation condition.
8. If the evidence reaches cuRobo IK, constraint, collision, or trajectory internals, also read `skills/curobo-failure-diagnosis/SKILL.md` for read-only classification, then return to this Skill for authorized root-cause repair.

## Mandatory workflow

Follow this order. Do not start from the loudest exception or the latest image.

1. **Freeze the vocabulary.** Select the canonical stage map and record its path. Map the run's operation, sub-task, Skill, phase, transition, and pass condition to that map. Never rename a phase for narrative convenience.
2. **Locate the first stop.** Read the batch row and `summary.json`, then follow `failure.report_ref`, `traceback_ref`, operation reports, phase plans, execution records, and postcondition results. Treat wrapper exceptions as stop markers.
3. **State the visible task symptom.** Inspect actual execution video near the failure and describe what a human can observe: the gripper missed the object, the wrist rotated near an opening, the arm stopped before a target, the door moved but did not remain open, or the object slipped after release. Use structured state to corroborate the observation.
4. **Build the failure stack.** Write the chain from task symptom to rejected condition to numeric or tool evidence. Use: `问题直接表现是...；其触发了...条件；详细情况为...。`
5. **Audit the requested target and actual state.** Check actual object pose, target pose, frames, units, TCP convention, gripper state, attachment, active arm, start joints, collision-world revision, and whether the target used the executed state or a stale nominal profile.
6. **Separate responsible layers.** Distinguish target generation, planner IK, planner path, independent path postcheck, physical execution, MuJoCo contact facts, task postconditions, wrapper errors, and infrastructure. A later failure can be caused by an earlier bad target, but the evidence must show that chain.
7. **Reach the upstream design cause.** Do not stop at `IK did not converge`, `path not found`, `tracking error`, or `postcondition failed`. Inspect the target generator, actual-state handoff, transition strategy, collision semantics, controller request, or task-condition implementation that caused the rejection.
8. **Audit condition meaning.** State what the failed condition protects and whether that meaning is correct for this phase. Expected finger-pad contact with the manipulated object is task-required; palm, wrist, or unintended-link contact remains forbidden. Preserve reasonable reachability, collision, stability, and task-success conditions.
9. **Test competing explanations against evidence.** Use same-state comparisons, positive and negative controls, frozen planning requests, or fresh-reset replays. Reject explanations contradicted by video, state drift, contact logs, successful planning, unchanged physical parameters, or a known feasible target.
10. **Trace the code owner.** Follow the live call path from Task stage to operation, Skill, target generation, planner request, execution, and postcondition. Identify the exact source file, function, configuration, and data contract that creates the bad behavior.
11. **Implement the smallest root-cause fix when authorized.** Change the responsible target, state propagation, transition, phase-scoped contact rule, execution strategy, or diagnostic implementation. Do not globally loosen thresholds, disable collision, delete obstacles, or add retries that leave the bad target unchanged.
12. **Validate causally.** Preserve the original initial state, target distribution, planner mode, collision world, and success criteria. First reproduce and repair the original failed case; then run the complete operation and full task from fresh reset. Confirm that prior successful stages and fixes remain intact.

## Writing rules

- Lead with observable behavior, then the rejecting condition, then parameters and cause analysis.
- Write evidence-backed causes as declarative statements. Do not fill the report with vague alternatives.
- When evidence stops at a layer, state that fact positively: `碰撞诊断子程序抛出 IndexError 且未产生碰撞对象；本轮归类为轨迹优化失败，不归类为碰撞。`
- Keep tool output and Gene Agent judgment separate. MuJoCo provides state and contacts; Gene Agent usually applies the threshold or task condition.
- Name both the canonical task stage and the internal phase. A Task transition is not silently promoted to the next stage.
- Preserve the distinction between local operation success and complete task success.
- Name the exact upstream code owner and explain why changing it removes the cause rather than only making the current error disappear.

## Hard rules

- Never diagnose from `summary.json` alone when it contains a report or traceback reference.
- Never use a trajectory PNG or PLY as proof of actual execution, contact, or task success.
- Never call cuRobo planning collision, independent postcheck collision, and MuJoCo execution contact the same event.
- Never call `endpoint_tracking_error` a cuRobo planning failure.
- Never call a completed motion a successful operation before checking contact and task postconditions.
- Never call a fixed sub-stage a complete-task fix without a fresh-reset full-task replay.
- Never treat more seeds, longer solve time, slower motion, a relaxed tolerance, or a larger retry count as the first response to a target or strategy defect.
- Never weaken a reasonable task or safety condition merely because it rejects the current behavior.
- Never preserve an unreasonable condition that rejects task-required contact; scope the correction to the intended phase and exact contact pair while retaining all unrelated collision checks.
- Never overwrite the microwave Task1 stage map. Its fixed anchor is `docs/01-task-skills.svg`.

## Stage-map creation boundary

For a task without a canonical map, create a stable stage specification from the orchestration source and representative run artifacts, then render an SVG with `scripts/render_stage_map.py`. Store the specification and SVG under the task's documentation directory. The map is a diagnostic artifact; it must describe actual execution order and checks, not a desired future design.

## Completion criteria

A diagnosis is complete only when it records the canonical stage-map path, exact run and phase, visible symptom, first rejecting owner, rejected condition, numeric/tool evidence, upstream design cause, condition-meaning audit, exact code owner, and causal validation plan. If the user requests a fix, completion additionally requires the responsible code change, reproduction of the original failed state, and a fresh-reset downstream or full-task regression with unchanged success conditions. If actual video or structured evidence is missing, record the missing artifact and stop at the deepest supported layer rather than inventing a lower-level cause.
