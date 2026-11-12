# Representative cases

These archetypes are self-contained. Local paths identify original evidence when available; classification does not depend on those paths being present in another clone.

## Finite-search IK failure

Evidence: no strict goal solution; a probe disables scene collision but preserves self collision and limits; bounded seeds still do not converge.

Classify as curobo_ik, goal_pose_unreachable_within_search, finite_search. Say unreachable within recorded search, not globally impossible. Freeze target and start, then vary only the search budget or compare a nearby target.

## Goal scene collision

Evidence: a converged candidate or world-relaxed solution fails strict full-world checking and names obstacles.

Classify as curobo_ik, goal_scene_collision, confirmed for that candidate. Map proxies through collision_snapshot.json. Do not call it MuJoCo contact or prove all IK branches collide. Evaluate alternate branches under the same target and scene.

## Goal IK exists but trajectory optimization returns no path

Evidence: pose_converged is true with small error; no complete trajectory; no confirmed collision or limit.

Classify as curobo_trajectory, path_optimization_failed_no_confirmed_collision, solver_result. It is not IK failure and not confirmed collision. An auxiliary collision-check exception makes collision attribution unknown. Replay the frozen request before varying one planner parameter.

Local example: runs/microwave_task1_varied_bowl_front_curobo_logging20_20261109-004250/run_013/before_reopen_transition.

## Returned path rejected by postcheck

Evidence: backend returns a trajectory but independent waypoint validation finds self, scene, or limit violation.

Classify as collision_postcheck, using a specific path collision or limit reason when available; otherwise path_postcheck_rejected. State that Gene Agent rejected the path, and inspect the first invalid waypoint and proxy identity.

## Endpoint tracking failure

Evidence: plan and execution samples exist; endpoint error exceeds configured threshold.

Classify as execution, endpoint_tracking_error, confirmed. Planning succeeded; do not count it as a cuRobo planning failure. Validate using a fresh reset and change only a stage-local timing or controller parameter.

Local example: runs/microwave_task1_varied_bowl_front_curobo_logging20_20261109-004250/run_005.

## Invalid bilateral pad contact

Evidence: open, approach, and close complete, but bilateral_pad_contact or valid_pad_contact is false.

Classify as task_postcondition, invalid_pad_contact, confirmed by MuJoCo contact classification. Do not call it IK or path failure. Restore the same pre-close state and vary one target component while retaining contact criteria.

Local example: runs/microwave_task1_open_second_full_replay6_20261107-182847/run_006/extract_grasp/result.json.

## Door angle not stable

Evidence: planning and execution succeed, but measured door angle misses the band or stability duration.

Classify as task_postcondition, angle_not_stable, confirmed. This is not planner failure. Contact loss remains a hypothesis unless contact evidence supports it. Replay from the same pre-pull state with one changed grasp parameter.

Local example: runs/microwave_task1_open_second_full_replay6_20261107-182847/run_001/open_second/report.json.

## Auxiliary collision diagnostic also throws

Evidence: the primary result is path_not_found, while an additional collision diagnostic records IndexError or a shape mismatch.

Keep the primary solver owner and reason. Add the helper failure as secondary and set collision cause to unknown unless another strict check completed. Never replace the primary result with the helper exception or treat it as collision evidence.

## Mixed candidate screen

When candidate_screen_failed aggregates different candidates, report counts by reason. Some may be finite-search IK failures and others confirmed scene collisions. Retain every candidate's curobo_ik reason rather than assigning one cause to the whole screen.

Local example: runs/microwave_task1_varied_bowl_front_curobo_logging20_20261109-004250/run_016/pick_insert/candidate_selection.json.
