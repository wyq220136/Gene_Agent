# Evidence routing

## Read in this order

1. run/summary.json
2. failure.report_ref and failure.traceback_ref
3. Referenced operation report.json or result.json and its failed stage file
4. Nearby plan.json, result.json, stages files, candidate-selection records, and attachment records
5. failure.json, diagnostics.json, collision_snapshot.json, state.npz, and saved motion requests
6. Actual execution video at the failed time when geometry or motion needs visual checking
7. trajectory PNG, collision frames, or PLY only for planning geometry and orientation support

Layouts differ by task and version. Search by stage ID, request ID, reason code, report reference, or result filename rather than assuming one nesting depth. Report missing artifacts as evidence missing; never reconstruct values from a neighboring run.

## Establish the failed stage

Record run, seed, task/subtask, failed step, stage, code, category, message, report and traceback references, actual final state, and whether execution was completed, partial, not executed, or unknown. The outer operation_exception or wrapper traceback shows where orchestration stopped, not whether IK, collision, execution, or a postcondition caused it.

## Audit target provenance

Check T_world_tcp composition, actual versus nominal object pose, active group and start joints, TCP convention and tool frame, units, gripper opening, synchronized collision snapshot, held-object T_tcp_object, attachment state, scene revision, collision mode, source hashes, and request identity.

If the target is stale or incompatible with actual state, classify target_generation as primary and downstream IK or path failure as a consequence.

## Inspect planning evidence

For goal IK inspect failure_category, failure_detail.reason_code, pose convergence and errors, seed counts, selected seed, constraint flags, world_obstruction_probe, evidence source, and finite-search marker.

For trajectory planning inspect whether a goal state exists, backend status, complete trajectory presence, first_invalid_waypoint, collision checks, obstacles, independent postcheck, and collision_check_error or other auxiliary exceptions.

Map obstacle proxy names through collision_snapshot.json to body, geom, and source asset. State that this is robot collision-sphere versus planning-proxy evidence, not visual geometry or MuJoCo contact.

## Inspect execution and postconditions

When planning succeeded, record whether execution started and completed, maximum and endpoint tracking errors, stale-state failures, MuJoCo forbidden contacts, actual TCP/object displacement, grasp checks, door checks, placement checks, and attachment release. Keep valid-plan execution failure and executed-motion postcondition failure separate.

## Use visual evidence correctly

Use actual video to inspect arm configuration, gripper orientation, object motion, slipping, or contact loss. Extract frames shortly before, at, and after the structured failure time. Use contact logs for contact truth because frames may be occluded and contain no force history.

Use trajectory PNG and PLY only for planned path and proxy geometry. They cannot prove execution, contact, or task success. Rendering or video export failure means incomplete evidence, not automatically task failure.

## Compare repairs

Require the same initial pose, seed, resolved settings, assets, acceptance criteria, source hashes or explicit code delta, and evidence mode. Use the same frozen request for planner comparison and a fresh reset for execution/contact comparison. Report requested, completed, reached-stage, stage-success, and whole-task-success counts separately.
