"""无需站点/角度 JSON 的配置和感知回归；设备与服务均为离线替身。"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from gene_agent.realworld.microwave import perception_setup as setup
from gene_agent.realworld.microwave.configuration import runtime_configuration
from gene_agent.realworld.microwave.live_perception import LiveArticulationReader, collision_objects, _axis_rotation
from gene_agent.realworld.perception import ObservedScene
from tools.run import run_microwave_task1_realworld as cli


@pytest.fixture
def prepared(tmp_path):
    calibration_path = tmp_path / "cam_param.json"
    setup.save_json(calibration_path, {"serial": "camera-1", "robot_group": "right",
        "extrinsic_matrix": np.eye(4).tolist(), "intrinsic_matrix": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        "width": 100, "height": 100})
    assets = setup.asset_geometry()
    root = np.eye(4)
    root[:3, 3] = [.5, .3, .05]
    reference = {setup.MICROWAVE: (root @ np.linalg.inv(assets["alignment"])).tolist(),
                 setup.BOWL: np.eye(4).tolist(),
                 setup.KNOB: (root @ np.array(assets["joints"]["knob"]["T_asset_joint"])).tolist()}
    mesh = tmp_path / "handle.obj"
    mesh.write_text("v 0 0 0\n", encoding="utf-8")
    record = {"format": setup.FORMAT, "accepted": True,
              "reference_confirmation": "CLOSED_ZERO", "mask_confirmation": "ACCEPT",
              "camera": setup.load_camera_calibration(calibration_path),
              "assets": assets, "camera_endpoint": "tcp://camera", "pose_server": "http://pose",
              "sessions": {name: name for name in reference},
              "closed_reference_poses_sdk": reference,
              "meshes": {name: str(mesh) for name in reference},
              "table": {"table_pose_world": np.eye(4).tolist(), "table_size_xy_m": [.9, .6],
                        "accepted_table_thickness_m": assets["table_thickness_m"]},
              "input_sha256": {**assets["sources"], str(calibration_path): setup.sha256(calibration_path), str(mesh): setup.sha256(mesh)}}
    path = tmp_path / "perception_setup.json"
    setup.save_json(path, record)
    return path, record


@pytest.fixture
def prepared_no_knob(prepared):
    path, record = prepared
    record["include_knob"] = False
    record["reference_confirmation"] = "CLOSED"
    for field in ("sessions", "closed_reference_poses_sdk", "meshes"):
        record[field].pop(setup.KNOB)
    setup.save_json(path, record)
    return path, record


def observed(record, *, frame=1, stamp=1000., door=0., knob=0.):
    refs = {k: np.array(v) for k, v in record["closed_reference_poses_sdk"].items()}
    root = refs[setup.MICROWAVE] @ np.array(record["assets"]["alignment"])
    for name, key, angle in (("door", setup.MICROWAVE, door), ("knob", setup.KNOB, knob)):
        if key not in refs:
            continue
        joint = record["assets"]["joints"][name]
        point = (root @ np.array(joint["T_asset_joint"]))[:3, 3]
        axis = root[:3, :3] @ np.array(joint["axis_asset"])
        refs[key] = _axis_rotation(axis, angle, point) @ refs[key]
    return ObservedScene("right", "0", frame, stamp, np.eye(4), refs,
                         {k: k for k in refs}, True)


def test_configuration_requires_no_site_or_state_file(prepared):
    path, _ = prepared
    config = runtime_configuration(perception=path.parent, left_ip="192.168.1.1", right_ip="192.168.1.2")
    assert "state_file" not in config
    assert config["include_knob"] is False
    assert set(config["pose_sessions"]) == {setup.BOWL, setup.MICROWAVE}
    assert "knob" not in config["subtask_options"]
    assert config["robot_ports"] == {"left": 8080, "right": 8080}
    assert config["scene_drift_limits"] is not None
    for kind, options in config["subtask_options"].items():
        cli._required_options(kind, options)
    alignment = config["scene_parameters"]["T_perception_object_scene_asset"][setup.MICROWAVE]
    inverse = config["subtask_options"]["insert"]["microwave_frame_alignment"]["T_scene_asset_profile"]
    np.testing.assert_allclose(alignment @ inverse, np.eye(4), atol=1e-12)


@pytest.mark.parametrize("flags", [[], ["--no-knob"]])
def test_static_cli_never_connects_devices(prepared_no_knob, monkeypatch, capsys, flags):
    path, _ = prepared_no_knob
    monkeypatch.setattr(cli, "run_site", lambda *a, **k: pytest.fail("static check connected devices"))
    assert cli.main(["--perception", str(path.parent), "--left-ip", "192.168.1.1", "--right-ip", "192.168.1.2",
                     "--subtask", "all", "--check-config", *flags]) == 0
    assert "自动装配" in capsys.readouterr().out
    assert not (path.parent / "realworld_state.json").exists()


@pytest.mark.parametrize("door_deg", [0, 20, 85, 110])
def test_door_angle_and_fixed_cabinet_from_current_frame(prepared_no_knob, door_deg):
    _, record = prepared_no_knob
    reader = LiveArticulationReader(record)
    first = observed(record)
    reader(first)
    frame = observed(record, frame=2, stamp=90000., door=math.radians(door_deg))
    reading = reader(frame)
    assert set(reading) == {"door"}
    assert reading["door"]["angle_rad"] == pytest.approx(math.radians(door_deg), abs=1e-5)
    assert reading["door"]["observed_at_unix_ms"] == 90000.
    assert reader.anchor(frame)["verified_at_unix_ms"] == 90000.
    assert reader.anchor(frame)["checked_parts"] == ["door"]
    np.testing.assert_allclose(reader.anchor(frame)["T_world_scene_asset"], reader.root)
    assert set(collision_objects(frame).object_poses_sdk) == {setup.BOWL, setup.MICROWAVE}


def test_knob_wrap_uses_real_orientation_not_commanded_angle(prepared):
    _, record = prepared
    reader = LiveArticulationReader(record, include_knob=True)
    for i, angle in enumerate((0, 90, 179, 185, 260, 270)):
        frame = observed(record, frame=i, stamp=1000 + i * 100, knob=math.radians(angle))
        assert reader(frame)["knob"]["angle_rad"] == pytest.approx(math.radians(angle), abs=1e-5)


def test_rejects_repeated_frames_missing_knob_and_cabinet_motion(prepared):
    _, record = prepared
    reader = LiveArticulationReader(record, include_knob=True)
    frame = observed(record)
    reader(frame)
    with pytest.raises(ValueError, match="同一新帧"):
        reader(frame)
    with pytest.raises(ValueError, match="同一新帧"):
        reader(collision_objects(observed(record, frame=2)))
    moved = observed(record, frame=2)
    moved.object_poses_sdk[setup.MICROWAVE][2, 3] += .1
    with pytest.raises(ValueError, match="柜体可能移动"):
        reader(moved)


def test_reference_input_tampering_rejected(prepared):
    path, record = prepared
    Path(record["camera"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="输入已改变"):
        runtime_configuration(perception=path, left_ip="192.168.1.1", right_ip="192.168.1.2")


def test_with_knob_requires_perception_prepared_for_knob(prepared_no_knob):
    path, _ = prepared_no_knob
    with pytest.raises(ValueError, match="--with-knob"):
        runtime_configuration(perception=path, left_ip="192.168.1.1", right_ip="192.168.1.2", no_knob=False)


@pytest.mark.parametrize("flags, enabled", [([], False), (["--no-knob"], False), (["--with-knob"], True)])
def test_prepare_cli_defaults_to_no_knob(tmp_path, monkeypatch, flags, enabled):
    from tools.run import prepare_microwave_realworld as prepare_cli
    monkeypatch.setattr(prepare_cli.sys.stdin, "isatty", lambda: True)
    calls = []
    def prepare(**kwargs):
        calls.append(kwargs)
        return tmp_path / "perception_setup.json"
    monkeypatch.setattr(prepare_cli, "prepare_perception", prepare)
    assert prepare_cli.main(["--camera-params", str(tmp_path / "camera.json"),
                             "--output", str(tmp_path / "new"), *flags]) == 0
    assert len(calls) == 1
    assert calls[0]["include_knob"] is enabled


def test_with_knob_explicitly_reuses_legacy_three_sessions(prepared):
    path, _ = prepared
    config = runtime_configuration(perception=path, left_ip="192.168.1.1", right_ip="192.168.1.2", no_knob=False)
    assert config["include_knob"] is True
    assert setup.KNOB in config["pose_sessions"]
    assert "knob" in config["subtask_options"]
    assert cli.main(["--perception", str(path), "--left-ip", "192.168.1.1", "--right-ip", "192.168.1.2",
                     "--subtask", "rotate_knob", "--with-knob", "--check-config"]) == 0


def test_rotate_knob_is_blocked_before_devices(prepared_no_knob, monkeypatch, tmp_path):
    path, _ = prepared_no_knob
    config = runtime_configuration(perception=path, left_ip="192.168.1.1", right_ip="192.168.1.2")
    monkeypatch.setattr("gene_agent.realworld.devices.arms.realman.connection.connect_realman",
                        lambda *a, **k: pytest.fail("disabled knob connected robot"))
    with pytest.raises(ValueError, match="no_knob"):
        cli.main(["--perception", str(path), "--left-ip", "192.168.1.1", "--right-ip", "192.168.1.2",
                  "--subtask", "rotate_knob", "--check-config"])
    with pytest.raises(ValueError, match="no_knob"):
        cli.run_site(config, subtask="rotate_knob", output_dir=tmp_path / "disabled", execute=False)
    assert not (tmp_path / "disabled").exists()


def test_no_knob_reader_still_rejects_reused_frame_and_moved_handle(prepared_no_knob):
    _, record = prepared_no_knob
    reader = LiveArticulationReader(record)
    frame = observed(record)
    reader(frame)
    with pytest.raises(ValueError, match="同一新帧"):
        reader(frame)
    moved = observed(record, frame=2)
    moved.object_poses_sdk[setup.MICROWAVE][2, 3] += .1
    with pytest.raises(ValueError, match="柜体可能移动"):
        reader(moved)


def test_table_uses_fixed_geometry_and_current_depth_in_sdk_frame():
    depth = np.ones((100, 100))
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
    assets = setup.asset_geometry()
    assert assets["table_geometry"] == {"center_xy_m": [.1, .3], "size_xy_m": [.6, .6]}
    assert assets["table_thickness_m"] == .02
    transform = np.eye(4)
    transform[2, 3] = -1.2
    # 只看到部分桌面仍使用已有完整矩形；不要求分割或观察全部边界。
    depth[:70] = 0.
    table = setup.table_from_scene_geometry(depth, K, transform, assets["table_geometry"], thickness_m=.02)
    np.testing.assert_allclose(np.array(table["table_pose_world"])[:3, 3], [.1, .3, -.21], atol=1e-7)
    assert table["table_top_z_m"] == pytest.approx(-.2)
    assert table["table_size_xy_m"] == [.6, .6]
    assert table["footprint_is_visible_extent"] is False
    assert table["source"] == "registered_rgbd_fixed_table_geometry"


def test_table_height_ignores_obstacles_without_requiring_seventy_percent_plane():
    depth = np.ones((100, 100))
    depth[:40] = .5
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
    table = setup.table_from_scene_geometry(depth, K, np.eye(4),
        {"center_xy_m": [0., 0.], "size_xy_m": [2., 2.]}, thickness_m=.02)
    assert table["table_top_z_m"] == pytest.approx(1.)
    assert table["point_counts"]["inliers"] == 6000
    assert table["point_counts"]["inside_geometry"] == 10000


@pytest.mark.parametrize("depth", [np.zeros((10, 10)), np.full((10, 10), np.nan), np.full((10, 10), 4.)])
def test_table_missing_depth_does_not_invent_height(depth):
    with pytest.raises(ValueError, match="有效深度点太少"):
        setup.table_from_scene_geometry(depth, np.eye(3), np.eye(4),
            {"center_xy_m": [0., 0.], "size_xy_m": [.6, .6]}, thickness_m=.02)


def test_generated_knob_mesh_keeps_pointer(tmp_path):
    import trimesh
    path = setup.export_knob_mesh(tmp_path)
    mesh = trimesh.load(path, force="mesh", process=False)
    assert len(mesh.vertices) > 100
    assert np.isfinite(mesh.vertices).all()
    assert mesh.extents.max() < .1


@pytest.mark.parametrize("include_knob", [False, True])
def test_prepare_pipeline_writes_every_input_without_manual_json(prepared, tmp_path, monkeypatch, include_knob):
    _, record = prepared
    calibration = record["camera"]
    registered = []
    def capture(**kwargs):
        directory = kwargs["output_dir"]
        directory.mkdir(parents=True)
        Image.fromarray(np.zeros((100, 100, 3), np.uint8)).save(directory / "rgb.png")
        Image.fromarray(np.full((100, 100), 1000, np.uint16)).save(directory / "depth.png")
        setup.save_json(directory / "camera.json", {"serial": "camera-1", "intrinsic_matrix": calibration["intrinsic_matrix"],
                    "frame_number": 1, "timestamp_ms": 12., "captured_at_unix_ms": 1000.})
        return directory
    def segment(server, image_path, items):
        expected = {"bowl", "handle"} | ({"knob"} if include_knob else set())
        assert {Path(item["mask_path"]).stem for item in items} == expected
        for item in items:
            Image.fromarray(np.full((100, 100), 255, np.uint8)).save(item["mask_path"])
        return 200, {"ok": True}
    class Pose:
        def __init__(self, server): pass
        def register(self, **kwargs):
            registered.append(kwargs)
            name = kwargs["session_id"].rsplit("-", 1)[0]
            return {"ok": True, "session_id": kwargs["session_id"], "T_camera_object": record["closed_reference_poses_sdk"][name]}
    monkeypatch.setattr(setup, "capture_frozen_scene_frame", capture)
    monkeypatch.setattr("gene_agent.perception_modules.segmentation.seg_client.segment_image", segment)
    monkeypatch.setattr("gene_agent.perception_modules.segmentation.seg_client.unload_model", lambda *a: {})
    monkeypatch.setattr("gene_agent.perception_modules.pose_est.pose_client.PoseClient", Pose)
    if not include_knob:
        monkeypatch.setattr(setup, "export_knob_mesh", lambda *a, **k: pytest.fail("default exported knob mesh"))
    answers = iter(("CLOSED_ZERO" if include_knob else "CLOSED", "ACCEPT"))
    flags = {"include_knob": True} if include_knob else {}
    path = setup.prepare_perception(camera_params=calibration["path"], output=tmp_path / "generated", ask=lambda _: next(answers), **flags)
    _, result = setup.load_perception_setup(path.parent)
    assert len(registered) == (3 if include_knob else 2)
    assert result["include_knob"] is include_knob
    assert (setup.KNOB in result["sessions"]) is include_knob
    assert len({str(row["rgb_path"]) for row in registered}) == 1
    assert result["accepted"] is True
    assert not (path.parent / "masks/table.png").exists()
    assert result["table"]["table_size_xy_m"] == [.6, .6]
    assert result["table"]["accepted_table_thickness_m"] == .02
    config = runtime_configuration(perception=path, left_ip="192.168.1.1", right_ip="192.168.1.2")
    assert config["scene_parameters"]["accepted_table_thickness_m"] == .02
    assert config["scene_parameters"]["table_size_xy_m"] == [.6, .6]
    assert config["scene_parameters"]["table_pose_world"][2, 3] == pytest.approx(.99)
    assert config["runtime"]["min_tcp_z_m"] == pytest.approx(1.02)
    assert not list((tmp_path / "generated").rglob("*site*.json"))
    assert not list((tmp_path / "generated").rglob("*state*.json"))


def test_new_run_entry_binds_live_reader_and_never_reads_angle_file(prepared_no_knob, tmp_path, monkeypatch):
    path, record = prepared_no_knob
    config = runtime_configuration(perception=path, left_ip="192.168.1.1", right_ip="192.168.1.2")
    connections, closed = [], []
    class Connection:
        robot = object()
        gripper = object()
        def __init__(self, side): self.side = side
        def __enter__(self): return self
        def __exit__(self, *args): closed.append(self.side)
    def connect(ip, **kwargs):
        connections.append((ip, kwargs))
        return Connection(kwargs["group"])
    def make_observer(**kwargs):
        assert kwargs["include_knob"] is False
        assert set(kwargs["sessions"]) == {setup.BOWL, setup.MICROWAVE}
        frames = iter([observed(record), observed(record, frame=2, stamp=90000., door=math.radians(85))])
        def observe():
            frame = next(frames)
            angles = kwargs["angle_reader"](frame)
            assert set(angles) == {"door"}
            anchor = kwargs["microwave_root_anchor_reader"](frame)
            filtered = kwargs["observation_filter"](frame)
            assert set(filtered.object_poses_sdk) == {setup.BOWL, setup.MICROWAVE}
            assert anchor["verified_at_unix_ms"] == frame.timestamp_ms
            return angles
        return observe
    def task(**kwargs):
        first = kwargs["observe_collision_scene"]()
        second = kwargs["observe_collision_scene"]()
        assert first["door"]["angle_rad"] == pytest.approx(0)
        assert second["door"]["angle_rad"] == pytest.approx(math.radians(85))
        assert second["door"]["observed_at_unix_ms"] == 90000.
        return {"status": "validated_prefix", "success": False}
    monkeypatch.setattr("gene_agent.realworld.devices.arms.realman.connection.connect_realman", connect)
    monkeypatch.setattr("gene_agent.realworld.microwave.scene.make_live_microwave_observer", make_observer)
    monkeypatch.setattr(cli, "run_observed_microwave_subtask", task)
    result = cli.run_site(config, subtask="pick_bowl", output_dir=tmp_path / "run", execute=False)
    assert result["status"] == "validated_prefix"
    assert closed == ["right", "left"]
    assert all(row[1]["port"] == 8080 for row in connections)
    evidence = json.loads((tmp_path / "run/runtime_input.json").read_text(encoding="utf-8"))
    assert evidence["angle_source"] == "same_frame_foundationpose"
    assert not (tmp_path / "run/site_input.json").exists()


@pytest.mark.parametrize("kwargs", [{"left_port": 70000}, {"speed_scale": 2}, {"max_age_s": float('nan')}])
def test_invalid_device_parameters_fail_offline(prepared, kwargs):
    path, _ = prepared
    with pytest.raises(ValueError):
        runtime_configuration(perception=path, left_ip="192.168.1.1", right_ip="192.168.1.2", **kwargs)


def test_gripper_tcp_mismatch_is_rejected(prepared, tmp_path):
    path, _ = prepared
    profile = json.loads((setup.ROOT / "config/robots/standard_gripper.json").read_text(encoding="utf-8"))
    profile["flange_to_tcp_z_m"] += .1
    changed = tmp_path / "gripper.json"
    setup.save_json(changed, profile)
    with pytest.raises(ValueError, match="规划模型不一致"):
        runtime_configuration(perception=path, left_ip="192.168.1.1", right_ip="192.168.1.2", gripper_config=changed)


@pytest.mark.parametrize("include_knob", [False, True])
def test_live_observer_checks_camera_then_measures_enabled_objects(prepared, tmp_path, monkeypatch, include_knob):
    from gene_agent.realworld.microwave import scene
    _, record = prepared
    if not include_knob:
        for field in ("sessions", "closed_reference_poses_sdk", "meshes"):
            record[field].pop(setup.KNOB)
    frames = iter([observed(record), observed(record, frame=2, stamp=90000., door=math.radians(85))])
    serial = [record["camera"]["serial"]]
    tracked = []

    def capture(**kwargs):
        folder = kwargs["output_dir"]
        folder.mkdir(parents=True)
        Image.fromarray(np.full((100, 100), 1000, np.uint16)).save(folder / "depth.png")
        setup.save_json(folder / "camera.json", {"serial": serial[0],
                        "intrinsic_matrix": record["camera"]["intrinsic_matrix"]})
        return folder

    def track(**kwargs):
        tracked.append(kwargs["sessions"])
        return next(frames)

    def build(**kwargs):
        assert kwargs["include_knob"] is include_knob
        assert set(kwargs["angles"]) == ({"door", "knob"} if include_knob else {"door"})
        assert set(kwargs["observed"].object_poses_sdk) == {setup.BOWL, setup.MICROWAVE}
        assert kwargs["microwave_root_anchor"]["verified_at_unix_ms"] == kwargs["observed"].timestamp_ms
        return kwargs["angles"]

    monkeypatch.setattr(scene, "capture_frozen_scene_frame", capture)
    monkeypatch.setattr(scene, "track_observed_scene_from_capture", track)
    monkeypatch.setattr(scene, "build_live_microwave_collision_scene", build)
    reader = LiveArticulationReader(record, include_knob=include_knob)
    observer = scene.make_live_microwave_observer(
        output_dir=tmp_path / "observe", camera_endpoint="tcp://camera", expected_serial=serial[0],
        pose_client=object(), sessions=record["sessions"], side="right", camera_id="0",
        T_sdk_camera=np.eye(4), robots={}, grippers={}, angle_reader=reader,
        microwave_root_anchor_reader=reader.anchor, observation_filter=collision_objects,
        camera_calibration=record["camera"], scene_parameters={}, max_age_s=5.,
        include_knob=include_knob,
        microwave_pose_mesh_path=record["meshes"][setup.MICROWAVE])
    assert observer()["door"]["angle_rad"] == pytest.approx(0.)
    assert observer()["door"]["angle_rad"] == pytest.approx(math.radians(85))
    expected = {setup.BOWL, setup.MICROWAVE} | ({setup.KNOB} if include_knob else set())
    assert all(set(sessions) == expected for sessions in tracked)
    serial[0] = "different-camera"
    with pytest.raises(ValueError, match="serial"):
        observer()
    assert len(tracked) == 2


def test_existing_manual_calibration_output_is_accepted_only_for_right_arm(prepared):
    _, record = prepared
    path = Path(record["camera"]["path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("robot_group")
    data["calibrated_side"] = "right"
    data["right_extrinsic_matrix"] = data["extrinsic_matrix"]
    setup.save_json(path, data)
    assert setup.load_camera_calibration(path)["T_sdk_camera"] == np.eye(4).tolist()
    data["calibrated_side"] = "left"
    setup.save_json(path, data)
    with pytest.raises(ValueError, match="右臂"):
        setup.load_camera_calibration(path)
