---
name: curobo-failure-diagnosis
description: Read-only, evidence-driven diagnosis of cuRobo and downstream robot-operation failures. Use when an agent must explain why a run or planning phase failed; distinguish goal IK non-convergence, finite-search unreachability, target or path collision, joint limits, trajectory optimization failure, postcheck rejection, execution tracking, MuJoCo contact, task postconditions, and infrastructure errors; locate exact JSON fields and the first invalid waypoint; compare runs; or propose minimal validation and repair hypotheses without modifying code.
---

# cuRobo failure diagnosis

Diagnose from saved evidence. Do not modify source, historical runs, thresholds, collision policy, or task state. Provide repair hypotheses and minimal validation steps only. Treat later implementation authorization as a separate task.

## Load the references

1. Read references/evidence-routing.md before inspecting a run.
2. Read references/failure-taxonomy.md before assigning an owner, reason code, or evidence strength.
3. Use references/report-template.md for every delivered diagnosis.
4. Read references/representative-cases.md when evidence is mixed or an auxiliary diagnostic also failed.

## Mandatory workflow

Follow this order. Do not skip from an outer exception to a root-cause claim.

1. Locate the stop. Read summary, report reference, traceback reference, operation report, and failed phase. Treat operation_exception and wrapper exceptions as stop markers only.
2. Audit the target. Check actual object pose, transform composition, TCP convention, units, frames, active arm, gripper opening, held-object transform, attachment, start joints, and scene revision. Determine whether the target came from current executed state or a stale profile.
3. Separate goal IK from trajectory planning. Establish whether a goal joint candidate exists. If none exists, classify IK convergence and constraints. If a candidate exists but no trajectory is returned, classify trajectory optimization.
4. Attribute collision precisely. Separate goal from path, self from scene, failed-seed observations from converged-candidate checks, cuRobo evidence from independent postcheck, and planned collision from MuJoCo execution contact. Report obstacle, robot link or sphere, evidence source, and first invalid waypoint when recorded.
5. Continue after planner success. Check independent path validation, whether execution occurred, maximum and endpoint tracking error, stale-state checks, and forbidden execution contact.
6. Check operation postconditions. Inspect bilateral pad contact, held-object displacement, door-angle stability, support, placement footprint, settling, final tilt, and attachment release as applicable.
7. Separate infrastructure. Record CUDA, process, EGL, renderer, FFmpeg, missing-file, and diagnostic-helper failures without replacing the robot or planner root cause.
8. Report evidence and uncertainty. Give owner, raw category, stable reason code, exact source fields, evidence strength, unknowns, disallowed conclusions, and one minimal single-factor validation. Suggestions are not validated fixes.

## Hard rules

- Never call finite seed or bounded joint search proof of global unreachability.
- Never translate path_not_found into collision without confirmed collision evidence.
- Never use collision on a far, non-converged failed IK seed to prove that the requested goal collides.
- Never count endpoint_tracking_error as a cuRobo planning failure.
- Never merge cuRobo planning collision, independent path postcheck, and MuJoCo execution contact.
- Never call planner success, a completed animation, or simulation attachment complete physical task success.
- Never disable self collision, delete obstacles, switch to robot_only, loosen postconditions, or edit historical evidence to make a case pass.
- Never claim visual confirmation from trajectory PNG or PLY. Use actual execution video for motion and pose observations, then corroborate contact using structured state evidence.
- Preserve unknown when evidence is missing or an auxiliary diagnostic failed.

## Repair-advice boundary

Offer a repair hypothesis only after locating the failing layer. Tie every suggestion to a predicted observation and a same-state control. Prefer frozen-request replanning for planner questions and fresh-reset replay for execution or contact questions. Do not prescribe task-specific offsets, angles, gains, seed counts, or collision exclusions as universal defaults.

## Completion criteria

A diagnosis is complete only when it identifies the failed task stage and phase, separates the responsible layer from wrappers, cites exact files and fields, states what remains unknown, and supplies a minimal validation that could falsify the proposed explanation. If evidence cannot support a finer cause, return insufficient_evidence rather than guessing.
