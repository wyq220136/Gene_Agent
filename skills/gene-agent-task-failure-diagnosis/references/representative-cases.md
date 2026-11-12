# Representative diagnostic cases

These cases preserve the reasoning that led to three successful microwave Task1 modifications while extracting task-independent lessons. The exact distances, angles, gains, and openings are case parameters, not universal defaults.

## 1. Pick target geometry caused poor grasp posture

### Evidence chain

- Canonical location: Stage 2 `load_bowl -> Pick -> pregrasp/approach/close` in `docs/01-task-skills.svg`.
- Actual video showed the gripper approaching too far inside the bowl rim with an inward tilt.
- Contact and hold evidence showed that target geometry, rather than merely the number of IK seeds, controlled whether both finger pads straddled the rim and whether the bowl followed the gripper.
- The repair changed grasp-candidate geometry while preserving collision and grasp postconditions. A ten-run check reached 9/10 Pick success; the remaining run stopped in planning before the physical grasp, so it belonged to a different failure class.

### Correct diagnosis

The visible symptom was an unstable or missed bowl grasp. The rejecting condition was invalid grasp/contact or inability to execute the selected candidate. The upstream cause was a grasp target placed too far inside the feature with an unfavorable gripper axis. The repair target was grasp Knowledge and target geometry, not global collision removal or blind seed expansion.

### Transferable lesson

When grasp failures cluster around the same visible TCP posture, inspect feature-relative target geometry, finger-pad support points, approach direction, and object motion before tuning solver search. Validate with video and bilateral contact/hold evidence.

## 2. Insert reused a stale held-object relationship in a narrow passage

### Evidence chain

- Canonical location: Stage 2 `load_bowl -> Insert -> preplace/insert/release/retreat`.
- After Pick geometry changed, the object-to-TCP transform also changed. The old nominal placement TCP no longer represented the actual held object.
- Video showed wrist reorientation near the microwave opening; structured evidence separated preplace/insert planning failures from later tilt and placement postcondition failures.
- The effective modification generated insertion targets from the actual closed-gripper `T_tcp_object` and maintained orientation continuity through the narrow passage. In the combined verification, all five runs that entered Insert passed it; the later six-case full regression passed Stage 2 in all six cases.

### Correct diagnosis

The visible symptom was a held object approaching the opening with an unfavorable wrist orientation, followed by path failure, collision rejection, or invalid placement. The immediate rejection varied by run, but the shared upstream cause was target generation that did not consume the executed grasp relationship and allowed large reorientation inside constrained geometry. The repair target was actual-state target generation and narrow-passage orientation continuity.

### Transferable lesson

After any grasp change, recompute downstream placement and extraction targets from the actual attachment transform. Separate failures before motion, during path validation, during execution, and after release; do not collapse them into one `insert failed` category.

## 3. Open-second transition reconfigured the arm too close to the door

### Evidence chain

- Canonical location: the Task-owned transition after Stage 4 `rotate_knob` and before Stage 5 `open_door`, followed by Stage 5 handle pull.
- Failure-time video showed the arm taking a short/direct reconfiguration near the door and entering an abnormal redundant-joint posture.
- Planner and postcheck evidence showed that the transition, not the later handle-pull semantics, was the first unsafe layer in affected runs.
- The effective modification first created clearance while preserving the actual TCP orientation, then adjusted the arm in safer free space and used a C-space target for the final return. Handle-contact retention was diagnosed separately from the transition. The six restored historical initial states later completed the full seven-stage task 6/6 under the recorded simulation-assisted conditions.

### Correct diagnosis

The visible symptom was abnormal joint twisting near the articulated object before the next operation began. The rejecting condition was path or self/scene collision validation in the Task transition. The upstream cause was a transition strategy that attempted major orientation and redundant-joint changes before establishing clearance. The repair target was transition decomposition and waypoint semantics, not disabling door collision.

### Transferable lesson

For articulated objects and narrow workspaces, create clearance before major reorientation. Treat Task transitions as first-class phases with their own planning and evidence. Verify the repaired transition on the original failed states and continue through the complete task.

## 4. Door motion succeeded but the task postcondition failed

### Evidence chain

- The planner returned and executed a pull path.
- Video showed the door moving, but contact weakened before the target angle was stably reached.
- The operation failed `angle_not_stable`, not IK or path planning.
- Single-factor comparisons supported insufficient handle-contact margin; unrelated changes such as more path points or a generic speed change did not reproduce the same improvement.

### Transferable lesson

Separate motion generation from contact retention and articulated-state postconditions. A moving door is not a successful door operation. Diagnose expected finger-pad contact separately from forbidden palm, wrist, or arm contact.

## 5. Execution tracking was mislabeled as planning failure

### Evidence chain

- cuRobo returned a complete pregrasp, approach, or lift path.
- The arm stopped short or had not settled when the phase ended.
- Gene Agent rejected the phase with `endpoint_tracking_error`; actual endpoint error exceeded the unchanged threshold.

### Transferable lesson

Once planning success is established, inspect whether the trajectory itself is difficult to track before changing execution parameters: joint discontinuity, excessive step size, limit proximity, abrupt direction change, and phase-local controller saturation. Change speed, duration, or settling only when the trajectory and target are valid. Do not raise the endpoint threshold to make the failure disappear.

## 6. A diagnostic helper failed alongside the planner

### Evidence chain

- The target IK converged, but the trajectory optimizer returned no complete path.
- An auxiliary collision diagnostic then raised a tensor-mask `IndexError` before emitting a collision object.

### Correct diagnosis

The main failure is trajectory optimization without confirmed collision. The helper failure is an infrastructure/diagnostic defect. The report must not promote the helper exception into collision evidence or allow it to overwrite the original planner result.

## 7. Expected grasp contact was treated as forbidden collision

### Evidence chain

- The phase required finger pads to touch the manipulated object.
- The collision or contact classifier rejected the intended pad-object pair before the task could establish a grasp.
- Palm, wrist, forearm, and unrelated robot-object contacts were not required by the task.

### Correct diagnosis

The rejecting condition was semantically too broad for the contact phase. The root cause was a contact policy that did not distinguish task-required pad contact from unintended robot collision. The correct code owner was the phase-scoped contact/collision policy, not the grasp target or global scene model.

### Transferable lesson

Allow the exact intended link-object pair only in the phase that requires contact. Preserve all unrelated self, scene, palm, wrist, arm, and held-object clearance checks. Never solve this problem by removing the manipulated object from collision checking everywhere.

## 8. Rare IK non-convergence after target validity was proven

### Evidence chain

- The target used the current actual state, correct frames, TCP, gripper state, and attachment.
- A known joint solution or neighboring identical-state solve reached the same target and passed strict self, scene, and joint-limit checks.
- Only isolated runs failed within the configured bounded search.

### Correct diagnosis

The target and task condition were valid. The remaining cause was insufficient phase-local solver coverage for a rare scene, so a bounded seed, time, or retry adjustment was a root-cause repair for that search configuration. Pose accuracy and collision rules remained unchanged.

### Transferable lesson

Solver-budget changes are justified only after target validity and strict feasibility are demonstrated. Before that proof, longer solve time merely hides a target-generation defect.
