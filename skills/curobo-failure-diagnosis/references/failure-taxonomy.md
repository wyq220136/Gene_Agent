# Failure taxonomy

Use reason codes already emitted by Gene Agent. Do not create a parallel vocabulary. The owner identifies the layer that rejected the operation; it does not claim who authored the policy.

## Evidence strengths

| Strength | Meaning |
|---|---|
| confirmed | A converged state, returned path, independent check, executed state, or task measurement directly violates a named constraint. |
| finite_search | Bounded seeds, limits, or search budget failed. This is not a global proof. |
| solver_result | The backend returned failure, but no independent observation proves a more specific physical cause. |
| insufficient_evidence | Required fields are absent, inconsistent, or an auxiliary diagnostic failed. |

## Goal and IK failures

| Reason code | Owner | Supported conclusion | Do not conclude |
|---|---|---|---|
| goal_pose_not_converged | curobo_ik | Recorded IK search did not converge. finite_search. | Collision or absolute unreachability without an additional probe. |
| goal_pose_unreachable_within_search | curobo_ik | World-disabled probe still failed with self collision and limits retained. finite_search. | Global mathematical unreachability. |
| goal_self_collision | curobo_ik | Checked converged or world-relaxed goal state self-collides. confirmed. | Every IK branch self-collides. |
| goal_scene_collision | curobo_ik | Checked goal state collides with named scene proxies. confirmed. | Visual-mesh collision or MuJoCo contact unless separately observed. |
| goal_joint_limit_violation | curobo_ik | Checked goal candidate violates configured limits. confirmed. | Global unreachability under other configurations. |
| goal_multiple_constraints | curobo_ik | Checked candidate violates all listed constraints. confirmed for that candidate. | A unique cause among those constraints. |
| goal_ik_failure_unresolved | curobo_ik | Goal IK failed but evidence cannot resolve why. insufficient_evidence. | Collision, limit, or unreachability. |

Collision on a failed seed is only an observed candidate constraint if that seed did not reach the goal. Promote it to goal collision only when evidence comes from a converged candidate or a world-relaxed solution placed back into the strict scene.

## Trajectory failures

| Reason code | Owner | Supported conclusion | Do not conclude |
|---|---|---|---|
| path_optimization_failed_no_confirmed_collision | curobo_trajectory | Goal candidate exists but optimization returned no complete path. solver_result. | Collision, bad goal IK, or absolute impossibility. |
| path_self_collision | curobo_trajectory or collision_postcheck | Named path waypoint self-collides. confirmed. | Goal itself self-collides unless separately checked. |
| path_scene_collision | curobo_trajectory or collision_postcheck | Named path waypoint collides with a scene proxy. confirmed. | MuJoCo contact or visual-mesh penetration. |
| path_joint_limit_violation | curobo_trajectory or collision_postcheck | Named waypoint violates configured limits. confirmed. | Goal IK failure. |
| path_multiple_constraints | curobo_trajectory or collision_postcheck | Path waypoint violates all listed checks. confirmed. | Which constraint first caused optimizer failure. |
| path_postcheck_rejected | collision_postcheck | cuRobo returned a path but independent validation rejected it. | cuRobo reported the same reason unless raw evidence says so. |

Always report first_invalid_waypoint when present. If collision diagnostics throw after the primary solver failure, keep the primary solver result and mark collision attribution unknown.

## Downstream failures

| Code or condition | Owner | Supported conclusion | Do not conclude |
|---|---|---|---|
| endpoint_tracking_error | execution | Planning completed, but actual execution missed endpoint tolerance. | cuRobo IK or trajectory failure. |
| forbidden contact | execution | MuJoCo observed contact and Gene Agent policy rejected it. | cuRobo planning collision unless the planned path was separately rejected. |
| invalid_pad_contact | task_postcondition | Grasp failed bilateral or allowed-contact requirements. | IK or path failure when motion completed. |
| angle_not_stable | task_postcondition | Action executed but measured state failed its stability contract. | Planner failure. |
| placement footprint, support, tilt, settling, or release failure | task_postcondition | Named final-state contract failed. | Planner cause unless planning separately failed. |
| operation_exception or wrapper RuntimeError | task wrapper | Task stopped in the referenced operation. | Underlying cause; follow report_ref. |
| CUDA, process, EGL, renderer, FFmpeg, or missing evidence | infrastructure | Runtime setup or evidence capture failed. | Reachability, collision, or task failure unless independently recorded. |

Use exactly one primary owner: target_generation, curobo_ik, curobo_trajectory, collision_postcheck, execution, task_postcondition, or infrastructure. Name the earliest causal failure as primary and list auxiliary failures separately.
