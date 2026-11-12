# Root-cause repair

## Repair depth gate

Do not edit code until the analysis reaches all five levels:

1. visible task symptom;
2. first rejecting condition;
3. immediate technical mechanism;
4. upstream design or state-propagation cause;
5. exact code owner that creates the cause.

`IK did not converge`, `path not found`, `endpoint tracking error`, and `postcondition failed` are levels 2 or 3. They are not sufficient root causes by themselves.

## Trace the live code path

Trace the actual runtime path, not a similarly named unused implementation:

```text
Task stage
-> operation/sub-task
-> Skill phase
-> Knowledge or target generator
-> planner request and collision policy
-> execution bridge
-> contact/task postcondition
```

For every layer, record the source file, function, input state, produced value, and downstream consumer. Confirm that the proposed edit affects the path used by the failing run.

## Target-first rule for IK and path failures

Before changing solver time, seeds, tolerance, retry count, or motion speed, establish that the target is semantically and geometrically valid:

- uses the current executed object and robot state;
- uses the correct frame, units, TCP convention, active arm, gripper opening, and attachment;
- represents a task-useful pose rather than only a numerically convenient pose;
- leaves reasonable joint-limit and singularity margin;
- respects self and scene collision except explicitly intended contact;
- provides orientation continuity through narrow passages;
- is reachable from the actual start state under the same strict constraints.

If the target violates one of these properties, repair target generation or the transition. More search only spends longer on a bad request.

Solver coverage becomes a legitimate root cause only after the same strict target is shown feasible, for example through a known valid joint solution, a neighboring-state success, or a diagnostic solve that is strictly rechecked in the original world. If only isolated scenes fail after target validity is established, a bounded phase-local seed, time, or retry adjustment is valid. Do not globally relax pose accuracy or collision checks.

Execution speed cannot repair an IK failure because no executable trajectory exists yet.

## Condition-meaning audit

For every condition considered for modification, answer:

1. What physical, safety, or task invariant does it protect?
2. Is that invariant required in this phase?
3. Does the implementation measure the intended event?
4. Does it distinguish expected task contact from unintended collision?
5. Can the correction be scoped to one phase, link pair, object pair, or state?

### Expected contact

Contact required to perform the task must not be rejected as a generic collision. During grasp or handle contact, finger-pad contact with the manipulated object is expected. Allow only the intended pad-object pair in the intended phase. Continue to reject palm, wrist, forearm, robot-body, and unrelated scene contact.

Do not remove the manipulated object from all collision checking. Approach, non-contact motion, held-object clearance, and unintended-link collisions still require protection. Separate planning-time contact allowances, independent path postcheck, and MuJoCo execution-contact semantics.

### Reasonable conditions

Preserve conditions that correctly express task success or safety, such as bilateral grasp contact, stable articulated angle, placement support, object tilt, attachment release, joint limits, and collision clearance. If these conditions reject a run, fix the behavior that violates them.

Change the condition only when its meaning or implementation is wrong, such as treating required finger-pad contact as forbidden, applying a post-contact rule during a pre-contact phase, checking the wrong object/link pair, or using a stale frame.

## Root-cause repair patterns

| Root cause | Correct code owner | Symptom-level workaround to reject |
|---|---|---|
| Bad grasp or contact target | Knowledge/target generator | Add seeds or loosen contact postcondition |
| Stale object-to-TCP transform | State handoff and downstream target construction | Hard-code a new nominal TCP |
| Reorientation inside narrow geometry | Transition/waypoint strategy | Disable obstacle collision |
| Expected pad contact classified as collision | Phase-scoped contact policy | Ignore all robot-object contacts |
| Valid target missed in rare bounded search | Phase-local solver coverage | Globally relax pose tolerance |
| Trackable plan not settled before check | Phase-local execution/settling | Raise endpoint-error threshold |
| Diagnostic helper overwrites root result | Diagnostic/logging implementation | Change task behavior |

## Implementation requirements

When authorized to fix:

1. inspect the exact target and current code path;
2. state the predicted behavioral change before editing;
3. make the smallest coherent source change at the root owner;
4. preserve valid safety checks and prior fixes;
5. add or update focused tests for the corrected contract;
6. replay the original failed state or frozen request;
7. run a fresh-reset complete operation and full task;
8. report successes, remaining failures, and any simulation assistance.

A fix is effective when the original cause no longer occurs and downstream acceptance passes without weakening unrelated conditions. A different later failure is progress but not complete task success.
