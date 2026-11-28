import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import pytest

from gene_agent.visualization.realworld_scene import scene_bundle
from gene_agent.visualization.realworld_timeline import RealworldTimeline, bbox_segments


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def rig(tmp_path):
    run = tmp_path / "run"
    bundle = run / "initial_scene"
    sdk = np.eye(4); sdk[:3, 3] = [1, 0, 0]
    camera_pose = np.eye(4); camera_pose[:3, 3] = [0, 2, 0]
    setup = dict(camera=dict(serial="cam", T_sdk_camera=camera_pose.tolist()),
                 meshes={"Bowl/Bowl_0": "/old/bowl.obj"})
    write(bundle / "setup.json", setup)
    (bundle / "bowl.obj").write_text("v 0 0 0\nv .1 0 0\nv 0 .2 .3\nf 1 2 3\n")
    write(bundle / "manifest.json", {"file_map": {
        "/old/perception_setup.json": "setup.json", "/old/bowl.obj": "bowl.obj"}})
    scene = dict(snapshot={"world_frame": "world"}, provenance={"T_world_sdk": sdk.tolist()})
    for i in range(2):
        folder = run / f"observations/{i:04d}"
        write(folder / "capture/camera.json", dict(frame_number=100+i, captured_at_unix_ms=1000+i*10,
            intrinsic_matrix=[[100., 0, 0], [0, 100., 0], [0, 0, 1]], serial="cam", depth_unit="millimeter"))
        Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(folder / "capture/rgb.png")
        Image.fromarray(np.full((12, 12), 1000+i*10, dtype=np.uint16)).save(folder / "capture/depth.png")
        write(folder / "scene/observed_scene_provenance.json", dict(T_world_sdk=sdk.tolist(),
            gripper_widths_m={"right": .08-i*.04}, joint_positions_by_group_rad={"right": [0]*7}))
        object_pose = np.eye(4); object_pose[2, 3] = 1
        write(folder / "tracked/objs/Bowl/Bowl_0/pose.json", {
            "object_id": "Bowl/Bowl_0", "_capture": {"frame_number": 100+i},
            "camera_parameters_0": [object_pose.tolist()]})
    return run, bundle, scene


def test_fixed_world_cloud_and_raw_foundationpose_chain(tmp_path):
    _, bundle, scene = rig(tmp_path)
    replay = RealworldTimeline(bundle, scene, max_points=144)
    assert len(replay.frames) == 2
    a, b = replay.cloud(0), replay.cloud(1)
    np.testing.assert_allclose(a["points"][0], [1, 2, 1])
    assert (b["points"][0]-a["points"][0])[2] == pytest.approx(.01)
    obj = replay.objects(0)[0]
    np.testing.assert_allclose(obj["pose"][:3, 3], [1, 2, 1])
    assert bbox_segments(obj["bounds"], obj["pose"]).shape == (12, 2, 3)
    assert replay.frames[1]["provenance"]["gripper_widths_m"]["right"] == .04


def test_sam_mask_only_on_its_capture_and_no_old_mask_on_tracking_frame(tmp_path):
    run, bundle, scene = rig(tmp_path)
    mask = run / "observations/0000/capture/session_recovery/Bowl/Bowl_0/mask.png"
    mask.parent.mkdir(parents=True)
    Image.fromarray(np.full((12, 12), 255, dtype=np.uint8)).save(mask)
    replay = RealworldTimeline(bundle, scene)
    assert len(replay.cloud(0)["masks"]) == 1
    assert not replay.cloud(1)["masks"]
    assert replay.cloud(0)["overlay"][0, 0, 0] > 0


def test_mismatched_pose_is_not_overlaid_on_another_cloud(tmp_path):
    run, bundle, scene = rig(tmp_path)
    path = run / "observations/0001/tracked/objs/Bowl/Bowl_0/pose.json"
    value = json.loads(path.read_text()); value["_capture"]["frame_number"] = 100
    write(path, value)
    replay = RealworldTimeline(bundle, scene)
    with pytest.raises(ValueError, match="帧不一致"):
        replay.objects(1)


def test_world_change_is_reported_instead_of_automatically_aligned(tmp_path):
    run, bundle, scene = rig(tmp_path)
    write(run / "observations/0001/scene/observed_scene_provenance.json", {"T_world_sdk": np.eye(4).tolist()})
    replay = RealworldTimeline(bundle, scene)
    assert len(replay.frames) == 1
    assert "T_world_sdk 改变" in replay.warnings[0]


def test_unknown_depth_unit_cannot_change_cloud_scale_silently(tmp_path):
    run, bundle, scene = rig(tmp_path)
    path = run / "observations/0000/capture/camera.json"
    value = json.loads(path.read_text()); value["depth_unit"] = "unknown"
    write(path, value)
    with pytest.raises(ValueError, match="深度单位"):
        RealworldTimeline(bundle, scene).cloud(0)


def test_archive_contains_observation_sequence_and_relocates_assets(tmp_path):
    run, _, scene = rig(tmp_path)
    archive = tmp_path / "run.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        out.add(run, arcname="task")
    with scene_bundle(archive, include_run=True) as bundle:
        replay = RealworldTimeline(bundle, scene)
        assert len(replay.frames) == 2
        assert replay.objects(1)[0]["bounds"] is not None


def test_no_selection_still_loads_door_stage_trajectory(tmp_path):
    run, bundle, scene = rig(tmp_path)
    write(run / "door/planning/stage.json", {"stage": "door", "result": {"trajectory": {
        "joint_names": ["right_joint_1"], "times_s": [0, 1], "positions_rad": [[0], [.1]]}}})
    replay = RealworldTimeline(bundle, scene)
    assert len(replay.trajectories) == 1


def test_partial_capture_is_not_visible_until_rgbd_is_complete(tmp_path):
    run, bundle, scene = rig(tmp_path)
    path = run / "observations/0001/capture/depth.png"
    content = path.read_bytes(); path.unlink()
    replay = RealworldTimeline(bundle, scene)
    assert len(replay.frames) == 1
    path.write_bytes(content)
    assert replay.refresh() == 2


def test_failed_observation_uses_own_recorded_camera_context(tmp_path):
    run, bundle, scene = rig(tmp_path)
    path = run / "observations/0001/scene/observed_scene_provenance.json"
    path.unlink()
    pose = np.eye(4); pose[2, 3] = .5
    write(run / "observations/0001/capture/viewer_context.json", {
        "T_sdk_camera": pose.tolist(), "T_world_sdk": scene["provenance"]["T_world_sdk"], "expected_serial": "cam"})
    replay = RealworldTimeline(bundle, scene, max_points=144)
    np.testing.assert_allclose(replay.cloud(1)["points"][0], [1, 0, 1.51])
    assert not replay.frames[1]["provenance"]


def test_gripper_polling_samples_are_deduplicated_and_sorted_without_new_reads(tmp_path):
    run, bundle, scene = rig(tmp_path)
    execution = {"source_format": "realworld_gripper_stage", "group": "right",
                 "endpoint_confirmation": {"readings": [
        {"width_m": .04, "sdk_reading": {"sampled_at_unix_ms": 1005, "width_m": .04, "actpos": 500}},
        {"width_m": .05, "sdk_reading": {"sampled_at_unix_ms": 1002, "width_m": .05, "actpos": 625}},
    ]}}
    write(run / "pick_result.json", {"execution": execution})
    write(run / "pick/close/stage.json", {"execution": execution})
    replay = RealworldTimeline(bundle, scene)
    assert [r["width_m"] for r in replay.feedback] == [.05, .04]
    assert len(replay.feedback) == 2
    assert replay.refresh_reports() is False


def test_old_feedback_without_timestamp_is_not_given_invented_capture_time(tmp_path):
    run, bundle, scene = rig(tmp_path)
    write(run / "pick_result.json", {"source_format": "realworld_gripper_stage", "group": "right",
        "endpoint_confirmation": {"readings": [{"width_m": .04, "sdk_reading": {"width_m": .04}}]}})
    assert not RealworldTimeline(bundle, scene).feedback


def test_follow_refreshes_same_frame_when_segmentation_arrives_later(tmp_path):
    run, bundle, scene = rig(tmp_path)
    replay = RealworldTimeline(bundle, scene)
    assert not replay.cloud(0)["masks"]
    mask = run / "observations/0000/capture/session_recovery/Bowl/Bowl_0/mask.png"
    mask.parent.mkdir(parents=True)
    Image.fromarray(np.full((12, 12), 255, dtype=np.uint8)).save(mask)
    replay.refresh()
    assert len(replay.cloud(0)["masks"]) == 1


def test_sdk_feedback_timestamp_adds_no_device_query():
    from gene_agent.realworld.devices.grippers.realman.adapter import RealmanGripperAdapter
    class SDK:
        calls = 0
        def rm_get_gripper_state(self):
            self.calls += 1
            return 0, {"actpos": 800, "mode": 3}
    sdk = SDK()
    adapter = RealmanGripperAdapter(sdk, max_width_m=.08)
    assert adapter.read_width_m() == .064
    assert adapter.last_reading["sampled_at_unix_ms"] > 0
    assert sdk.calls == 1
