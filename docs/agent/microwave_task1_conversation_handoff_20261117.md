# Microwave Task1 Conversation Handoff

Updated: 2026-11-17  
Repository: /data1/yuquan/Gene_Agent_new

This note preserves the key experiment context, user requirements, real-world runtime explanations, and evidence caveats from the conversation. Treat current source and the user guide as authoritative when they differ from historical discussion.

## Current Checkout and Source of Truth

- The current checkout has no visible data/ directory. data/task_1/realworld_site.json, data/task_1/realworld_state.json, and simulation configs are local/ignored inputs and were not present in this checkout on 2026-11-17. Do not infer their present contents from old chat messages.
- The current real-world guides were updated on 2026-11-16: docs/user/microwave/realworld_task1_operation_guide.md and docs/user/microwave/realworld_microwave_entry.md. Read those before proposing a current command or claiming hardware readiness.
- Current guides state that the real-world entry is wired to live observation, cuRobo, RealMan arm state, grippers, scene-drift checks, and per-stage records, but the entry has not completed on-site plan-only, calibration, or device execution acceptance. Simulation success is not real-robot evidence.
- Run outputs visible in this checkout are under runs_log/. The user previously required experiment results under /data1/yuquan/Gene_Agent_new/runs and explicitly said not to write them to other directories. Honor the latest explicit output-root instruction for any new run, and check the actual path before launch. Some prior run artifacts were subsequently organized under runs_log/; do not move or overwrite historical data without authorization.
- The bowl placement visualization was reported as generated at runs/visualizations/bowl_placement_range_comparison_20261113/, containing 10 images, README, manifest, and rerender script. That directory was not present in the 2026-11-17 checkout inspection. Treat the previous generation as conversational history, not as a currently available artifact; recreate only if requested.

## Experiment History and Results

### Local right-lower placement, 20 runs

- Batch: runs_log/microwave_task1_varied_bowl_front_curobo_logging20_20261109-004250/.
- Seeds: 153-172; bowl placement was a local randomized range near the microwave handle and right-arm base; the table had not been expanded.
- experiment_results.md reports 20/20 completed, 12/20 strict full-task successes (60%), versus 0/20 for the paired old-code baseline on the same seeds and initial placements.
- Failure distribution: six Pick execution-tracking failures (four pregrasp, two approach), one before_reopen_transition path optimization failure, and one post-grasp lift candidate-screen failure. The remaining 12 passed seven task stages and terminal checks.
- Evidence standard: distinguish planner result from MuJoCo execution and task postconditions. endpoint_tracking_error is an execution-check failure, not a cuRobo planning failure. summary.json is an index; follow report/traceback references to phase-level evidence.

### Expanded table and high-variance lower-table placement, 50 runs

- Batch: runs_log/microwave_task1_left_lower_high_variance50_dynamic5mm_direct_20261112-140746/.
- Seeds: 12001-12050; completed 50/50. The recorded batch status reports 24 successes, 26 task failures, and zero infrastructure failures; all 50 have summary, terminal log, placement, and video artifacts.
- Table in the experiment config: size_xy_m=[1.2, 1.0], center [0.35, 0.35], top z=-0.12 m, bounds x=[-0.25, 0.95], y=[-0.15, 0.85]. The table was extended to cover the left arm base region.
- Bowl proposal area was the microwave-left / lower-table region, not an IK-filtered subset. The preflight for this exact batch reports 50 valid placements, no IK/task filtering, x=[0.04142, 0.29876] m, y=[0.02370, 0.33517] m, and standard deviation [0.07214, 0.06435] m.
- Failure distribution: pick_bowl/candidate_selection 12; insert_bowl/insert_sequence_screen 8; insert_bowl/retreat_1 3; extract_bowl/post_extract_2 2; pick_bowl/lift 1.
- Batch status: runs_log/microwave_task1_left_lower_high_variance50_dynamic5mm_direct_20261112-140746/live_status.md. Per-run evidence is linked there: summary, result report, traceback where present, actual task video, terminal log, and placement.

### Latest 50-run report available in checkout

- runs_log/microwave_task1_left_lower_high_variance50_latest_20261116-0938/experiment_results.md reports 50/50 complete, 28 successes, 22 failures. It is the newest final experiment report found on 2026-11-17.
- Failure distribution in that report: candidate_selection 9; insert_sequence_screen 8; retreat_1 2; post_extract_2 2; lift 1.
- A separate layered_pick_full_bowl_staged_insert batch directory has a stale live_status.md showing 15/50 at its last refresh on 2026-11-12. Do not call that directory complete based on its per-run files or a later, differently named batch. Check the precise batch report and completion records.
- None of these simulation batches establishes real-world success. Historical batches can include simulation-assisted bowl attachment or kinematic knob updates; retain their stated evidence mode.

## Bowl Placement Visuals

- The requested comparison was five top views from each of two distributions:
  1. Expanded table, bowl positions across the lower table and microwave-left reachable region, from the 50-run batch.
  2. Original table, local lower-right perturbation near the microwave handle/right-arm base, from the 20-run batch.
- The previous visualization selection policy was min_x, max_x, min_y, max_y, and nearest-to-centroid actual run per batch.
- The intended image combined a top-down render from the run's saved model_0.mjb and first frames.jsonl qpos with a placement map. All batch points were blue and the selected bowl red; both groups shared the same world-coordinate axes and camera scale.
- The reported earlier verification was 10 nonblank images at 900x554; however, the output directory is absent from the current checkout and must be verified/recreated before linking it to a future report.

## User's Grasp and Motion Requirements

- Bowl grasp should be on the rim, from the outside. The fingers must not both enter the bowl interior; reject visually invalid inside grasps based on the actual fingertip/contact geometry, not a literal text-only heuristic.
- The requested gripper axis angle is approximately 45-70 degrees relative to the tabletop / bowl-bottom plane. This is not a 90-degree angle to the bowl's center axis. Avoid near-parallel-to-table candidates and avoid approaches that contact the bowl from inside.
- Approach the grasp from the side near the right-arm base, not an abstract whole-robot base. This is to reduce self-collision with the head and joints. Avoid reverse-handed/large wrist-twist solutions; approximately obtuse but moderate grasp geometry is preferred over near-150-degree wrist reversal.
- Bowl insertion may need a flatter arm posture than grasping. If upright insertion risks microwave-ceiling collision, adapt the insertion knowledge/waypoints while preserving bowl tilt and collision constraints.
- For every repair, preserve downstream compatibility: pass actual measured post-stage state, TCP/frame transforms, attachment evidence, and task postconditions. Do not relax justified safety limits, disable collision, remove obstacles, or merely extend solver time to hide a bad target.

## Real-World Perception and Execution Semantics

- Current runtime is stage-wise re-observation, not continuous visual servoing.
- Each live observer call captures a new RGB-D frame, tracks Bowl and Microwave from that same frozen frame with separate FoundationPose sessions, samples both arms' joint state and gripper widths, reads effective door/knob state, then rebuilds the collision scene. The frame, poses, source timestamps, and scene provenance are saved per observation.
- In execution mode, before sending a planned stage, the drift guard obtains a fresh observation and compares object/table poses, both-arm joints, gripper widths, articulated angles, collision roster, source/calibration identity, and frame freshness. If the current scene no longer satisfies the planned envelope, it rejects the command; it does not automatically retarget or replan that same segment.
- A dispatched stage is a precomputed joint trajectory. There is no continuous camera-driven TCP correction or online cuRobo replanning while that trajectory is moving.
- After a completed stage, the Task observes again and plans the next phase from measured state. Failures also attempt a recovery observation when configured. Predicted endpoints are not passed as if they were measured.
- Plan-only still connects to the live devices and reads physical state; it is not offline planning. --check-config is the static no-device check. The current guide says default plan-only checks only the selected task's first executable prefix/stage; it does not prove the entire Skill or Task is executable.
- The current guide says the --execute path runs the selected whole subtask and does not provide a stage stop flag. Before first real motion, add/verify a bounded stop-after-stage or low-speed no-contact smoke path; keep the physical emergency stop available. Do not directly run the full Pick as the first real movement.
- --subtask all --no-knob skips knob motion only. Door/knob state is still needed for scene construction; no-knob does not remove perception or collision requirements.

## RealMan SDK Import Error

- The reported traceback ended at importing Robotic_Arm.rm_robot_interface in gene_agent/realworld/devices/arms/realman/connection.py. It occurs before any SDK network connection attempt; it is an import/environment error, not a robot IP or cuRobo failure.
- The entry's run_site() constructs RealMan connections before the observer and before planning. It connects both arms, even for a right-arm-only task, because both arm states are used in collision planning.
- In the earlier check, /data1/yuquan/miniconda3/envs/gene_agent/bin/python and the base interpreter could not import Robotic_Arm. The 2026-11-16 guide later states that the workstation's curobo environment imports the vendor package. These may be different hosts/interpreters or a later environment update; always run the import test using the exact PY used to launch the entry.
- Install/use the official RealMan SDK in that same Python environment; do not guess that a public package with the same name is the vendor SDK. The guide's check is: PY -c "from Robotic_Arm.rm_robot_interface import RoboticArm; print('RealMan SDK正常')".

## Site Configuration and the Microwave Root Anchor

- The actual realworld_site.json is local/ignored and absent from this checkout, so this handoff cannot certify its current values. The conversation previously saw a minimal right-arm pick_bowl plan-only config with only subtask_options.pick and no scene_drift_limits; that old snapshot could not be used for --subtask all --execute --yes.
- For a scope-specific plan-only check, the current guide allows minimal fields required by that subtask. For execution, scene_drift_limits must be measured and present, and all --no-knob still requires the relevant door, pick, insert, extract, and table options. Never copy the plan-only config into execution unchanged.
- state_file needs fresh door and knob angles with real Unix-millisecond timestamps and source. If the door angle is derived from an observed handle pose using configured calibration, the legacy door field is compatibility input; knob state remains time-sensitive.
- When FoundationPose tracks a door-moving handle/door mesh, its pose cannot stand in for the static microwave cabinet after the door opens. microwave_root_anchor.T_world_scene_asset must be the actual current cabinet-root pose and verified_at_unix_ms must be a fresh real timestamp from an actual verification. The guide's identity matrix and timestamp 0 are schema examples only, invalid as deployed calibration/evidence. Do not paste them as real values or refresh only the timestamp after the cabinet moved.
- Table pose is the collision-box center pose, not its top surface. The table XY footprint and location must be measured, checked against both robot bases and collision geometry; the Nov-16 guide records a prior plan-only start_collision involving the temporary table proxy overlapping base_link and gripper. Do not switch to robot_only, remove the table, or loosen collision to get around it.
- Insert/Extract microwave_frame_alignment is not safely defaulted to identity: historical synthetic comparison found the wrong frame could put preplace at wrong height and lead to IK failure. It needs valid asset/frame calibration before being accepted for planning.

## Evidence and Workflow Expectations

- For a Task1 failure, use the canonical stage graph docs/01-task-skills.svg and repository-local diagnosis instructions (AGENTS.md, then skills/gene-agent-task-failure-diagnosis/SKILL.md; read cuRobo Skill if entering solver/collision/path internals).
- Navigate from summary.json through report_ref and traceback_ref. Inspect the actual task video at the failure timestamp and select an unobstructed view; planning PNGs alone are not evidence of actual behavior.
- Report in order: observable behavior, exact rejection/postcondition, numeric/config evidence, upstream design cause, then source function/file and actual call chain. Do not stop at IK failed, path_not_found, endpoint_tracking_error, or post-hoc status labels.
- Separate visual geometry, MuJoCo contact geometry, cuRobo world proxies, robot self-collision spheres, and attached-object collision proxies.
- For experiment batches: use fresh reset per trial, large declared bowl placement variance in the requested accessible region, no success/IK filtering in the initial placement sampler, complete diagnostic logs and videos, persistent auto-monitoring linked to a Markdown status file, and report startup only after workers are confirmed running. For hypothesis testing, the user explicitly prefers testing independent hypotheses concurrently when GPU resources permit, rather than serial one-at-a-time trials.
- Preserve original failed initial states for regression, then run a fresh-reset full-task regression. A local prefix success does not count as downstream/full-task success.
- Do not overclaim incomplete batches. live_status.md, batch_status.jsonl, completion records, per-run summaries, videos, and final result report must agree before reporting the final denominator or rate.

## Useful Current References

- Real-world operation guide: docs/user/microwave/realworld_task1_operation_guide.md
- Real-world subtask entry: docs/user/microwave/realworld_microwave_entry.md
- 20-run local-placement results: runs_log/microwave_task1_varied_bowl_front_curobo_logging20_20261109-004250/experiment_results.md
- Completed dynamic 50-run status: runs_log/microwave_task1_left_lower_high_variance50_dynamic5mm_direct_20261112-140746/live_status.md
- Latest 50-run report found in checkout: runs_log/microwave_task1_left_lower_high_variance50_latest_20261116-0938/experiment_results.md
- Extended-table placement preflight: runs_log/microwave_task1_left_lower_high_variance50_dynamic5mm_direct_20261112-140746/placement_distribution_preflight.json

