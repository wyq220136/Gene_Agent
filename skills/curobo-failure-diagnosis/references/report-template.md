# Diagnosis report template

## Conclusion

Give one sentence naming the failed layer and most specific supported cause. State explicitly when failure is not IK, not confirmed collision, or not cuRobo.

## Classification

| Field | Value |
|---|---|
| run / seed | |
| task stage / operation / phase | |
| execution status | executed / partial / not_executed / unknown |
| failure_owner | target_generation / curobo_ik / curobo_trajectory / collision_postcheck / execution / task_postcondition / infrastructure |
| raw category and status | |
| stable reason code | |
| evidence strength | confirmed / finite_search / solver_result / insufficient_evidence |

## Evidence chain

List exact file links and JSON field paths with observed values.

- Summary stop and report reference:
- Target provenance and actual start state:
- IK convergence and seed statistics:
- Goal or path constraints:
- Collision type, robot entity, obstacle, evidence source, and first invalid waypoint:
- Trajectory returned and independent postcheck:
- Execution tracking and contacts:
- Task postconditions:
- Infrastructure or auxiliary diagnostic errors:

## Root-cause boundary

- Supported conclusion:
- Unknown or missing evidence:
- Conclusions not supported by this evidence:
- Primary failure versus secondary or auxiliary failures:

## Repair hypotheses

For each hypothesis give its evidence, smallest proposed change category, predicted observation, same-state or fresh-reset falsifier, and unchanged acceptance checks. Never present a hypothesis as a validated fix.

## Minimal next validation

Prefer frozen-request replanning for IK, collision, or trajectory questions; strict full-world checking of a converged candidate; first-invalid-waypoint inspection; fresh-reset replay for execution, contact, or postconditions; or paired comparison with identical state and thresholds.

## Overall task status

If task success is requested, separately report stages reached, stages passed, final postconditions, simulation assistance, and whether whole-task success is established.
