# Visual diagnosis from actual execution video

## Principle

Use actual execution video to establish visible behavior. Use structured logs to establish the rejecting condition and numeric cause. A planner PNG is never a substitute for actual video.

## Locate the failure interval

1. Read the failed stage and phase from structured artifacts.
2. Inspect `video/task.json`, `live/manifest.json`, evaluation indexes, or phase timestamps to map the phase to video time.
3. Extract at least three frames: immediately before the phase, at the visible failure, and immediately after or at the last valid frame.
4. If several camera views exist, select the view that exposes the relevant geometry with least occlusion. Record the selected view and timestamp.
5. Inspect a short clip around the frames when the issue involves motion, slip, contact retention, or joint reconfiguration.

Resolve a usable FFmpeg dynamically and verify it before extraction:

```bash
ffmpeg_bin="${GENE_AGENT_FFMPEG:-$(command -v ffmpeg)}"
"$ffmpeg_bin" -version
"$ffmpeg_bin" -ss 12.4 -i run/video/task.mp4 -frames:v 1 /tmp/failure_12_4.png
```

If the video is a multi-view mosaic, crop only after saving an uncropped reference frame. Do not choose a close-up that hides the arm configuration when diagnosing a path or wrist problem.

## What to inspect

### Grasp and contact

- TCP position relative to the object rim, handle, surface normal, or grasp axis;
- finger-pad placement and whether both fingers straddle the intended feature;
- gripper tilt relative to the object's axis;
- object displacement during approach and close;
- whether the object follows the gripper during lift.

### Narrow passages and insertion

- wrist orientation before entering the passage;
- whether large reorientation happens inside the constrained region;
- clearance between held object, gripper, wrist, and surrounding geometry;
- continuity from preplace through insert and retreat;
- object tilt and support after release.

### Door, drawer, and articulated-object manipulation

- clearance before reorienting the arm;
- whether redundant joints twist near the articulated object or robot base;
- contact retention between gripper and handle;
- articulated angle before release and during retreat;
- expected contact versus palm, wrist, or arm collision.

### Execution tracking

- whether the arm stops short, oscillates, overshoots, or is still moving at phase termination;
- whether the visible error agrees with endpoint and maximum tracking error;
- whether the failure occurred before any grasp/contact action, preventing misclassification as a physical grasp failure.

## Cross-check rules

- A visible near-contact is not a collision classification. Confirm the contact pair or collision evidence.
- A stationary final frame is not endpoint convergence. Read actual joint error and settling window.
- A door that moved is not an accepted door operation. Check target angle and stability duration.
- A closed gripper is not a valid grasp. Check bilateral pads, object motion, and hold postconditions.
- A successful-looking release is not a successful placement. Check support, footprint, tilt, settling, retreat displacement, and attachment removal.

## Output

For every visual claim, record:

```text
video path | timestamp or frame | selected view | visible observation | corroborating log field
```

Keep extracted frames in the run's diagnostic/report directory only when the user asks to preserve them. Temporary inspection frames belong outside formal run evidence.
