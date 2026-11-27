"""无需站点/角度 JSON 的配置和感知回归；设备与服务均为离线替身。"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from gene_agent.realworld.microwave import perception_setup as setup
from gene_agent.realworld.microwave.configuration import runtime_configuration
from gene_agent.realworld.microwave.live_perception import HandleOrbitError, LiveArticulationReader, collision_objects, _axis_rotation
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
    halfturn = np.eye(4); halfturn[:3, :3] = np.diag([-1., -1., 1.])
    record = {"format": setup.FORMAT, "geometry_contract_version": 2, "accepted": True,
              "microwave_pose_disambiguation": {"T_mesh_equivalent_halfturn": halfturn.tolist(),
                  "T_mesh_asset": assets["alignment"], "observed_up_sdk": [0., 0., 1.],
                  "mesh_sha256": setup.sha256(mesh)},
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


@pytest.mark.parametrize("door_deg", [0, 40, 85])
@pytest.mark.parametrize("orientation_deg", [12, 180])
def test_handle_orientation_change_at_same_center_is_not_cabinet_motion(
    prepared_no_knob, door_deg, orientation_deg,
):
    """中心轨迹不变的朝向估计变化曾被完整 6D pose 校验误拒绝。"""
    _, record = prepared_no_knob
    reader = LiveArticulationReader(record)
    frame = observed(record, door=math.radians(door_deg))
    mesh = frame.object_poses_sdk[setup.MICROWAVE]
    center = (mesh @ reader.handle_local)[:3, 3].copy()
    changed = _axis_rotation([0, 0, 1], math.radians(orientation_deg), center) @ mesh
    frame.object_poses_sdk[setup.MICROWAVE] = changed

    reading = reader(frame)["door"]

    assert reading["angle_rad"] == pytest.approx(math.radians(door_deg), abs=1e-8)
    assert reading["diagnostics"]["handle_center_error_m"] == pytest.approx(0., abs=1e-12)
    assert reading["diagnostics"]["mesh_rotation_error_rad"] == pytest.approx(math.radians(orientation_deg))
    assert reading["diagnostics"]["mesh_rotation_is_diagnostic_only"] is True
    np.testing.assert_allclose(reader.anchor(frame)["T_world_scene_asset"], reader.root)
    # 测角约束不改写 FoundationPose 原始观测；不能伪装为更精确的感知。
    np.testing.assert_allclose(frame.object_poses_sdk[setup.MICROWAVE], changed)


@pytest.mark.parametrize("radial, axial", [(0., .02), (.02, 0.), (-.02, 0.), (.011, .011)])
def test_handle_center_off_orbit_still_rejected_with_numeric_evidence(prepared_no_knob, radial, axial):
    _, record = prepared_no_knob
    reader = LiveArticulationReader(record)
    reader(observed(record))
    frame = observed(record, frame=2, door=math.radians(40))
    point, axis, _ = reader._joint("door")
    center = (frame.object_poses_sdk[setup.MICROWAVE] @ reader.handle_local)[:3, 3]
    radius = center - point
    radius -= axis * (radius @ axis)
    frame.object_poses_sdk[setup.MICROWAVE][:3, 3] += radial * radius / np.linalg.norm(radius) + axial * axis

    with pytest.raises(HandleOrbitError, match="中心误差.*15.00 mm.*帧 2") as failure:
        reader(frame)

    evidence = failure.value.diagnostics
    assert evidence["status"] == "rejected"
    assert evidence["handle_center_error_m"] == pytest.approx(math.hypot(radial, axial))
    assert evidence["radius_error_m"] == pytest.approx(radial)
    assert evidence["axial_error_m"] == pytest.approx(axial)
    assert evidence["max_translation_m"] == .015
    with pytest.raises(ValueError, match="尚未通过当前帧几何核对"):
        reader.anchor(frame)


def test_orbit_check_does_not_depend_on_mesh_coordinate_origin(prepared_no_knob):
    """同一几何换 mesh 坐标原点后，测角和验收结果应不变。"""
    _, record = prepared_no_knob
    reader = LiveArticulationReader(record)
    frame = observed(record, door=math.radians(40))
    center = (frame.object_poses_sdk[setup.MICROWAVE] @ reader.handle_local)[:3, 3]
    frame.object_poses_sdk[setup.MICROWAVE] = (
        _axis_rotation([0, 0, 1], .1, center) @ frame.object_poses_sdk[setup.MICROWAVE])
    first = reader(frame)["door"]
    rebased = json.loads(json.dumps(record))
    change = _axis_rotation([1, 0, 0], .3, [0, 0, 0])
    change[:3, 3] = [.5, .2, -.3]
    rebased["assets"]["alignment"] = (np.linalg.inv(change) @ reader.alignment).tolist()
    rebased["closed_reference_poses_sdk"][setup.MICROWAVE] = (reader.reference[setup.MICROWAVE] @ change).tolist()
    frame.object_poses_sdk[setup.MICROWAVE] = frame.object_poses_sdk[setup.MICROWAVE] @ change

    second = LiveArticulationReader(rebased)(frame)["door"]

    assert second["angle_rad"] == pytest.approx(first["angle_rad"])
    assert second["diagnostics"]["handle_center_error_m"] == pytest.approx(first["diagnostics"]["handle_center_error_m"], abs=1e-12)
    assert second["diagnostics"]["mesh_origin_error_m"] > .015


@pytest.mark.parametrize("angle_deg", [-5, 115])
def test_orbit_change_preserves_door_joint_limits(prepared_no_knob, angle_deg):
    _, record = prepared_no_knob
    with pytest.raises(ValueError, match="超出资产关节范围"):
        LiveArticulationReader(record)(observed(record, door=math.radians(angle_deg)))


def test_knob_wrap_uses_real_orientation_not_commanded_angle(prepared):
    _, record = prepared
    reader = LiveArticulationReader(record, include_knob=True)
    for i, angle in enumerate((0, 90, 179, 185, 260, 270)):
        frame = observed(record, frame=i, stamp=1000 + i * 100, knob=math.radians(angle))
        assert reader(frame)["knob"]["angle_rad"] == pytest.approx(math.radians(angle), abs=1e-5)


def test_handle_orbit_change_does_not_disable_knob_orientation_check(prepared):
    _, record = prepared
    reader = LiveArticulationReader(record, include_knob=True)
    frame = observed(record)
    point, axis, _ = reader._joint("knob")
    column = int(np.argmin(np.abs(frame.object_poses_sdk[setup.KNOB][:3, :3].T @ axis)))
    tilt_axis = frame.object_poses_sdk[setup.KNOB][:3, column]
    frame.object_poses_sdk[setup.KNOB] = _axis_rotation(tilt_axis, .3, point) @ frame.object_poses_sdk[setup.KNOB]
    with pytest.raises(ValueError, match="旋钮观测不符合固定柜体关节轴"):
        reader(frame)


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
    table_pose = np.array(table["table_pose_world"])
    np.testing.assert_allclose(table_pose[:3,3]+table_pose[:3,2]*.01, [.1,.3,-.2], atol=1e-7)
    assert table["sdk_up_consistent"] is False  # 该测试相机在平面 SDK -Z 侧。
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
    service_calls = []
    def capture(**kwargs):
        service_calls.append("capture")
        directory = kwargs["output_dir"]
        directory.mkdir(parents=True)
        Image.fromarray(np.zeros((100, 100, 3), np.uint8)).save(directory / "rgb.png")
        Image.fromarray(np.full((100, 100), 1000, np.uint16)).save(directory / "depth.png")
        setup.save_json(directory / "camera.json", {"serial": "camera-1", "intrinsic_matrix": calibration["intrinsic_matrix"],
                    "frame_number": 1, "timestamp_ms": 12., "captured_at_unix_ms": 1000.})
        return directory
    def segment(server, image_path, items):
        service_calls.append("segment")
        expected = {"bowl", "handle"} | ({"knob"} if include_knob else set())
        assert {Path(item["mask_path"]).stem for item in items} == expected
        for item in items:
            Image.fromarray(np.full((100, 100), 255, np.uint8)).save(item["mask_path"])
        return 200, {"ok": True}
    class Pose:
        def __init__(self, server): pass
        def register(self, **kwargs):
            service_calls.append("register")
            registered.append(kwargs)
            name = kwargs["session_id"].rsplit("-", 1)[0]
            return {"ok": True, "session_id": kwargs["session_id"], "T_camera_object": record["closed_reference_poses_sdk"][name]}
    # 已有资产路线禁止额外重建；桌面和旋钮不能混入默认分割请求。
    monkeypatch.setattr("gene_agent.perception_modules.interactive_realworld_perception.post_reconstruction",
                        lambda *a, **k: pytest.fail("known task assets triggered SAM3D"))
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
    assert service_calls == ["capture", "segment"] + ["register"] * (3 if include_knob else 2)
    assert len(registered) == (3 if include_knob else 2)
    from gene_agent.perception_modules.pose_est.task_pose_est import pose_mesh_path
    for name in (setup.BOWL, setup.MICROWAVE):
        assert result["meshes"][name] == str(pose_mesh_path(name.split("/")[0], setup.ROOT / "data/task_1/objs" / name))
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
    evidence = json.loads((tmp_path / "observe/0001/door_handle_diagnostics.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "accepted"
    assert evidence["frame_number"] == 2
    assert evidence["angle_rad"] == pytest.approx(math.radians(85))
    expected = {setup.BOWL, setup.MICROWAVE} | ({setup.KNOB} if include_knob else set())
    assert all(set(sessions) == expected for sessions in tracked)
    serial[0] = "different-camera"
    with pytest.raises(ValueError, match="serial"):
        observer()
    assert len(tracked) == 2


@pytest.mark.parametrize("write_failure", [False, True])
def test_live_observer_saves_rejected_orbit_before_robot_read_or_scene_export(
    prepared_no_knob, tmp_path, monkeypatch, write_failure,
):
    from gene_agent.realworld.microwave import scene
    _, record = prepared_no_knob
    frame = observed(record)
    frame.object_poses_sdk[setup.MICROWAVE][2, 3] += .1

    def capture(**kwargs):
        folder = kwargs["output_dir"]
        folder.mkdir(parents=True)
        return folder

    monkeypatch.setattr(scene, "capture_frozen_scene_frame", capture)
    monkeypatch.setattr(scene, "track_observed_scene_from_capture", lambda **kwargs: frame)
    monkeypatch.setattr(scene, "build_live_microwave_collision_scene",
                        lambda **kwargs: pytest.fail("rejected perception reached robot read / scene export"))
    reader = LiveArticulationReader(record)
    observer = scene.make_live_microwave_observer(
        output_dir=tmp_path / "observe", camera_endpoint="tcp://camera", expected_serial="camera-1",
        pose_client=object(), sessions=record["sessions"], side="right", camera_id="0",
        T_sdk_camera=np.eye(4), robots={}, grippers={}, angle_reader=reader,
        microwave_root_anchor_reader=reader.anchor, observation_filter=collision_objects,
        scene_parameters={}, max_age_s=5., include_knob=False,
        microwave_pose_mesh_path=record["meshes"][setup.MICROWAVE])

    if write_failure:
        def fail_write(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(HandleOrbitError) as failure:
        observer()

    assert not (tmp_path / "observe/0000/scene").exists()
    if write_failure:
        assert isinstance(failure.value.__cause__, OSError)
        assert failure.value.diagnostics["handle_center_error_m"] == pytest.approx(.1)
        return
    evidence = json.loads((tmp_path / "observe/0000/door_handle_diagnostics.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "rejected"
    assert evidence["handle_center_error_m"] == pytest.approx(.1)
    assert evidence["frame_number"] == frame.frame_number
    np.testing.assert_allclose(evidence["observed_mesh_pose_sdk"], frame.object_poses_sdk[setup.MICROWAVE])


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

# 按操作刷新感知；设备、模型服务使用替身，实际碰撞导出仍走生产代码。
def _partial_observation(record, keys, *, frame=1, door=0., knob=0.):
    from dataclasses import replace
    value = observed(record, frame=frame, stamp=1000. + frame, door=door, knob=knob)
    return replace(value, object_poses_sdk={k:v for k,v in value.object_poses_sdk.items() if k in keys},
                   sources={k:v for k,v in value.sources.items() if k in keys})


def test_pick_refreshes_bowl_without_handle_orbit_and_keeps_original_times(prepared_no_knob, monkeypatch):
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    _, record = prepared_no_knob
    scope = OperationPerception(record)
    scope.begin_operation("pick_bowl")
    assert set(scope.selected_sessions(record["sessions"])) == {setup.BOWL}
    monkeypatch.setattr(LiveArticulationReader, "__call__",
                        lambda *a: pytest.fail("bowl observation invoked handle orbit"))
    for frame in (1, 2):
        if frame == 2:
            scope.before_interaction("approach", "motion")
        value, angles, anchor = scope.consume(_partial_observation(record, {setup.BOWL}, frame=frame))
        refresh = value.perception_refresh
        assert refresh["refreshed_object_ids"] == [setup.BOWL]
        assert refresh["retained_object_ids"] == [setup.MICROWAVE]
        assert refresh["object_observations"][setup.MICROWAVE]["observed_at_unix_ms"] == 0.
        assert angles["door"]["observed_at_unix_ms"] == 0.
        assert anchor["verified_at_unix_ms"] == 0.
        assert refresh["retained_cabinet_anchor"] is True


def test_door_only_refreshes_at_entry_contact_motion_and_post_contact(prepared_no_knob):
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    _, record = prepared_no_knob
    scope = OperationPerception(record)
    scope.begin_operation("open_door")
    assert set(scope.selected_sessions(record["sessions"])) == {setup.MICROWAVE}
    _, angles, _ = scope.consume(_partial_observation(record, {setup.MICROWAVE}, door=.2))
    assert angles["door"]["angle_rad"] == pytest.approx(.2)
    assert scope.selected_sessions(record["sessions"]) == {}
    _, retained_angles, _ = scope.consume(_partial_observation(record, set(), frame=2))
    assert retained_angles["door"]["angle_rad"] == pytest.approx(.2)
    assert retained_angles["door"]["observed_at_unix_ms"] == angles["door"]["observed_at_unix_ms"]
    scope.after_door_phase("pregrasp")
    assert scope.selected_sessions(record["sessions"]) == {}
    for frame, phase in enumerate(("pull_0", "post_contact", "push_1"), 3):
        scope.after_door_phase(phase)
        assert set(scope.selected_sessions(record["sessions"])) == {setup.MICROWAVE}
        value, angles, _ = scope.consume(_partial_observation(record, {setup.MICROWAVE}, frame=frame, door=.3))
        assert angles["door"]["observed_at_unix_ms"] == 1000. + frame
        assert value.perception_refresh["retained_angle_keys"] == []
        assert scope.selected_sessions(record["sessions"]) == {}
    scope.begin_operation("pick_bowl")
    _, held_angles, anchor = scope.consume(_partial_observation(record, {setup.BOWL}, frame=9))
    assert held_angles["door"]["observed_at_unix_ms"] == 1000. + frame
    assert anchor["verified_at_unix_ms"] == 1000. + frame


def test_scope_preserves_real_bad_door_rejection_in_door_phase(prepared_no_knob):
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    _, record = prepared_no_knob
    scope = OperationPerception(record)
    scope.begin_operation("open_door")
    value = _partial_observation(record, {setup.MICROWAVE}, door=math.radians(.63))
    point, axis, _ = scope.reader._joint("door")
    center = (value.object_poses_sdk[setup.MICROWAVE] @ scope.reader.handle_local)[:3,3]
    radial = center - point
    radial -= axis * (axis @ radial)
    value.object_poses_sdk[setup.MICROWAVE][:3,3] -= .02718 * radial / np.linalg.norm(radial)
    with pytest.raises(HandleOrbitError, match="27.18 mm"):
        scope.consume(value)
    assert set(scope.selected_sessions(record["sessions"])) == {setup.MICROWAVE}


def test_knob_scope_does_not_refresh_handle_or_bowl(prepared):
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    _, record = prepared
    scope = OperationPerception(record, include_knob=True)
    scope.begin_operation("rotate_knob")
    assert set(scope.selected_sessions(record["sessions"])) == {setup.KNOB}
    _, angles, anchor = scope.consume(_partial_observation(record, {setup.KNOB}, knob=.4))
    assert angles["knob"]["angle_rad"] == pytest.approx(.4)
    assert angles["door"]["observed_at_unix_ms"] == anchor["verified_at_unix_ms"] == 0.


def test_transition_runs_no_object_estimation_and_rejects_unknown_scope(prepared_no_knob):
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    _, record = prepared_no_knob
    scope = OperationPerception(record)
    with pytest.raises(ValueError, match="设置当前操作"):
        scope.selected_sessions(record["sessions"])
    scope.begin_operation("transition")
    assert scope.selected_sessions(record["sessions"]) == {}
    value, _, _ = scope.consume(_partial_observation(record, set()))
    assert value.perception_refresh["refreshed_object_ids"] == []
    with pytest.raises(ValueError, match="未定义"):
        scope.begin_operation("unknown")


@pytest.mark.parametrize("operation", ["pick_bowl", "open_door"])
def test_scoped_observer_exports_real_scene_with_retained_microwave(prepared_no_knob, tmp_path, monkeypatch, operation):
    import time
    from gene_agent.motion.model import JointState
    from gene_agent.realworld.microwave import scene
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    _, record = prepared_no_knob
    scope = OperationPerception(record)
    ticks = []
    movement = np.zeros(3)
    def capture(**kw):
        folder = kw["output_dir"]
        folder.mkdir(parents=True)
        ticks.append(len(ticks)+1)
        setup.save_json(folder/"camera.json", {"frame_number":ticks[-1],
                        "captured_at_unix_ms":time.time()*1000.})
        return folder
    def track(**kw):
        assert set(kw["sessions"]) == ({setup.BOWL} if operation == "pick_bowl" else {setup.MICROWAVE})
        from dataclasses import replace
        value = _partial_observation(record,set(kw["sessions"]),frame=ticks[-1])
        if setup.MICROWAVE in value.object_poses_sdk:
            value.object_poses_sdk[setup.MICROWAVE][:3,3] += movement
        return replace(value, timestamp_ms=time.time()*1000.)
    class Robot:
        def __init__(self, side): self.side=side
        def read_joint_state(self):
            path=setup.ROOT/f"config/robots/realman_{self.side}_default_pose.json"
            return JointState(tuple(f"{self.side}_joint_{i}" for i in range(1,8)),
                              tuple(json.loads(path.read_text())["joints_rad"]))
    class Gripper:
        def read_width_m(self): return .04
    assets=record["assets"]
    monkeypatch.setattr(scene,"capture_frozen_scene_frame",capture)
    monkeypatch.setattr(scene,"track_observed_scene_from_capture",track)
    observer=scene.make_live_microwave_observer(
        output_dir=tmp_path/"scoped",camera_endpoint="tcp://camera",expected_serial="camera-1",
        pose_client=object(),sessions=record["sessions"],side="right",camera_id="0",
        T_sdk_camera=np.eye(4),robots={s:Robot(s) for s in ("left","right")},
        grippers={s:Gripper() for s in ("left","right")},angle_reader=lambda *a:pytest.fail("global angles"),
        observation_filter=collision_objects,microwave_pose_mesh_path=record["meshes"][setup.MICROWAVE],
        include_knob=False,max_age_s=5.,operation_perception=scope,
        scene_parameters={"T_world_sdk":np.eye(4),"T_perception_object_scene_asset":{
            setup.BOWL:np.eye(4),setup.MICROWAVE:np.array(assets["alignment"])},
            "table_pose_world":np.eye(4),"table_size_xy_m":[.9,.6],
            "accepted_support_size_xyz_m":[*assets["support"]["size_xy_m"],assets["support"]["height_m"]],
            "accepted_support_center_xy_local_m":assets["support"]["center_xy_local_m"],
            "accepted_table_thickness_m":assets["table_thickness_m"],"active_group":"right"})
    observer.begin_operation(operation)
    first,second=observer(),observer()
    for value in (first,second):
        provenance=value["provenance"]
        assert provenance["all_objects_same_frame"] is False
        assert provenance["live_state"]["door"]["observed_at_unix_ms"] == first["provenance"]["live_state"]["door"]["observed_at_unix_ms"]
        assert provenance["live_state"]["door"].get("observed", True) is (operation == "open_door" and value is first)
        assert provenance["pose_source_modes"][setup.MICROWAVE] == "verified_stationary_anchor"
        assert any(row["body_name"].startswith("microwave") for row in value["snapshot"]["records"])
        assert any(row["body_name"]=="bowl" for row in value["snapshot"]["records"])
    assert second["provenance"]["source_frame_number"] > first["provenance"]["source_frame_number"]

    from gene_agent.realworld.scene_drift import compare_scene_drift, SceneDriftLimits
    checked = compare_scene_drift(first, second, limits=SceneDriftLimits(.005,.15,.02,.005,.05))
    assert checked.status == "pass", checked.reason
    assert second["provenance"]["perception_refresh"]["retained_cabinet_anchor"] is True

    if operation == "open_door":
        reader = scope.reader
        point, axis, _ = reader._joint("door")
        center = (reader.reference[setup.MICROWAVE] @ reader.handle_local)[:3,3]
        radial = center - point
        radial -= axis * (radial @ axis)
        movement[:] = .021 * radial / np.linalg.norm(radial)
        observer.before_interaction("handle_pull/03_close", "grasp")
        third = observer()
        old_support = np.array(first["scene_metadata"]["objects"]["microwave_support"]["T_world_support"])
        new_support = np.array(third["scene_metadata"]["objects"]["microwave_support"]["T_world_support"])
        np.testing.assert_allclose(old_support, new_support, atol=1e-12)
        old_microwave = np.array(first["provenance"]["aligned_object_poses_world"][setup.MICROWAVE])
        new_microwave = np.array(third["provenance"]["aligned_object_poses_world"][setup.MICROWAVE])
        np.testing.assert_allclose(new_microwave[:3,3]-old_microwave[:3,3], movement, atol=1e-12)
        # 再加载实际导出的XML，保证viewer/planner不会读回移动后的支撑箱。
        import mujoco
        model = mujoco.MjModel.from_xml_path(third["scene_xml_path"])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.body("microwave_support").xpos, old_support[:3,3], atol=1e-12)


def test_guard_forwards_operation_and_door_phase_hooks():
    from gene_agent.realworld.scene_drift import bind_live_scene_guard,SceneDriftLimits
    events=[]
    def observe(): return {}
    observe.begin_operation=lambda name:events.append(("operation",name))
    observe.after_door_phase=lambda name:events.append(("phase",name))
    observe.before_interaction=lambda name,kind:events.append(("interaction",name,kind))
    wrapped,_=bind_live_scene_guard(observe,lambda _:None,limits=SceneDriftLimits(.005,.15,.02,.005,.05))
    wrapped.begin_operation("pick_bowl")
    wrapped.after_door_phase("post_contact")
    wrapped.before_interaction("lift", "motion")
    assert events[-1] == ("interaction", "lift", "motion")
    events.pop()
    assert events == [("operation","pick_bowl"),("phase","post_contact")]


@pytest.mark.parametrize("name", ["pick_bowl", "open_door", "close_door", "rotate_knob", "extract_bowl"])
def test_subtask_entry_routes_operation_before_runner(tmp_path, monkeypatch, name):
    from gene_agent.realworld.microwave import task
    events=[]
    def observer(): pytest.fail("entry should not add extra perception")
    observer.begin_operation=lambda operation:events.append(operation)
    def run(**kwargs):
        assert events == [name]
        return {"status":"validated_prefix"}
    attribute={"pick_bowl":"run_observed_bowl_pick", "open_door":"run_observed_door_subtask",
               "close_door":"run_observed_door_subtask", "rotate_knob":"run_observed_knob_skill",
               "extract_bowl":"run_observed_bowl_extract"}[name]
    monkeypatch.setattr(task,attribute,run)
    result=task.run_observed_microwave_subtask(
        name=name,observe_collision_scene=observer,make_runtime=lambda _:None,
        planner_pool=object(),output_dir=tmp_path,options={})
    assert result["status"] == "validated_prefix"


def test_full_task_routes_first_operation_through_guard(tmp_path, monkeypatch):
    from gene_agent.realworld.microwave import task
    from gene_agent.realworld.scene_drift import SceneDriftLimits
    events=[]
    def observer(): pytest.fail("task wrapper must not add perception")
    observer.begin_operation=lambda operation:events.append(operation)
    def door(**kwargs):
        assert events == ["open_door"]
        assert hasattr(kwargs["observe_collision_scene"],"begin_operation")
        return {"status":"validated_prefix","success":False}
    monkeypatch.setattr(task,"run_observed_door_subtask",door)
    result=task.run_observed_microwave_task1(
        observe_collision_scene=observer,make_runtime=lambda _:None,planner_pool=object(),
        output_dir=tmp_path/"task",subtask_options={"door":{}},execute=False,
        scene_drift_limits=SceneDriftLimits(.005,.15,.02,.005,.05))
    assert result["status"] == "validated_prefix"


@pytest.mark.parametrize("bad", ["duplicate", "unverified", "extrinsic", "old_time"])
def test_bowl_only_scope_preserves_new_frame_and_calibration_checks(prepared_no_knob, bad):
    from dataclasses import replace
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    _, record=prepared_no_knob
    scope=OperationPerception(record)
    scope.begin_operation("pick_bowl")
    first=_partial_observation(record,{setup.BOWL},frame=1)
    scope.consume(first)
    current=_partial_observation(record,{setup.BOWL},frame=2)
    if bad=="duplicate": current=first
    elif bad=="unverified": current=replace(current,pose_frame_verified=False)
    elif bad=="old_time": current=replace(current,timestamp_ms=0.)
    else:
        transform=np.eye(4);transform[0,3]=.1
        current=replace(current,T_sdk_camera=transform)
    with pytest.raises(ValueError,match="新帧|外参"):
        scope.consume(current)


def test_post_contact_wait_happens_before_next_capture_once(prepared_no_knob):
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    _,record=prepared_no_knob
    waits=[]
    scope=OperationPerception(record,sleep=waits.append)
    scope.begin_operation("open_door")
    scope.after_door_phase("post_contact")
    scope.prepare_capture()
    assert waits == [2.]
    assert scope.last_settle_s == 2.
    scope.prepare_capture()
    assert waits == [2.]
    assert scope.last_settle_s == 0.


def test_handle_to_push_handoff_requests_post_contact_refresh(tmp_path, monkeypatch):
    from gene_agent.realworld.microwave import door
    events=[]
    def observer(): pytest.fail("handoff must schedule, not duplicate capture")
    observer.after_door_phase=lambda phase:events.append(phase)
    def skill(**kw):
        if kw["skill_id"] == "push_open":
            assert events == ["post_contact"]
        return {"success":True,"status":"executed","postcondition":{}}
    monkeypatch.setattr(door,"run_observed_door_skill",skill)
    monkeypatch.setattr(door,"has_realworld_success_evidence",lambda _:True)
    result=door.run_observed_door_subtask(
        part="open",observe_collision_scene=observer,make_runtime=lambda _:None,
        planner_pool=object(),output_dir=tmp_path/"door",config_path="unused",execute=True,
        max_microwave_translation_m=.005,max_microwave_rotation_rad=.15,
        max_start_error_rad=.02,max_gripper_width_error_m=.005,max_door_angle_error_rad=.05)
    assert result["success"]
    assert events == ["post_contact"]


def test_operation_normalizes_symmetric_handle_before_fixed_cabinet_check(prepared_no_knob):
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    from dataclasses import replace
    _, record = prepared_no_knob
    operation = OperationPerception(record)
    operation.begin_operation("plan_only")
    frame = observed(record)
    turn = np.array(record["microwave_pose_disambiguation"]["T_mesh_equivalent_halfturn"])
    raw = frame.object_poses_sdk[setup.MICROWAVE] @ turn
    frame = replace(frame, object_poses_sdk={**frame.object_poses_sdk, setup.MICROWAVE: raw})
    result, angles, anchor = operation.consume(frame)
    np.testing.assert_allclose(result.object_poses_sdk[setup.MICROWAVE], record["closed_reference_poses_sdk"][setup.MICROWAVE], atol=1e-12)
    assert result.perception_refresh["pose_frame_disambiguation"]["selected_equivalent_branch"] == 1
    assert angles["door"]["angle_rad"] == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(np.array(anchor["T_world_scene_asset"])[:3,:3],np.eye(3),atol=1e-12)


def test_old_geometry_record_requires_new_preparation(prepared_no_knob):
    path, record = prepared_no_knob
    record.pop("geometry_contract_version")
    setup.save_json(path, record)
    with pytest.raises(ValueError, match="旧准备记录"):
        setup.load_perception_setup(path)


@pytest.mark.parametrize("operation,events,object_id", [
    ("pick_bowl", [("approach", "motion"), ("close", "grasp"), ("lift", "motion")], setup.BOWL),
    ("extract_bowl", [("grasp", "motion"), ("close", "grasp"), ("post_extract_1", "motion"),
                      ("post_extract_2", "motion")], setup.BOWL),
    ("insert_bowl", [("preplace", "motion"), ("insert", "motion"), ("release", "release"),
                     ("settle", "release_settled")], setup.BOWL),
    ("place_bowl_on_table", [("preplace", "motion"), ("place", "motion"), ("release", "release"),
                             ("settle", "release_settled")], setup.BOWL),
    ("open_door", [("handle_pull/02_pull_0", "motion"), ("push_open/03_push_1", "motion"),
                   ("handle_pull/03_release", "release")], setup.MICROWAVE),
])
def test_interaction_events_refresh_once_and_keep_pose_between_commands(
        prepared_no_knob, operation, events, object_id):
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    _, record = prepared_no_knob
    scope = OperationPerception(record)
    scope.begin_operation(operation)
    scope.consume(_partial_observation(record, {object_id}, frame=1))
    retained = scope.records[object_id].copy()
    for name, kind in [("open", "position"), ("pregrasp", "motion"),
                       ("pre_extract_1", "motion"), ("retreat_1", "motion")]:
        scope.before_interaction(name, kind)
        assert scope.selected_sessions(record["sessions"]) == {}
    scope.after_door_phase("door_complete")
    scope.after_door_phase("failure")
    value, _, _ = scope.consume(_partial_observation(record, set(), frame=2))
    assert value.perception_refresh["object_observations"][object_id] == retained
    scope.begin_operation(operation)
    assert scope.selected_sessions(record["sessions"]) == {}
    for frame, (name, kind) in enumerate(events, 3):
        scope.before_interaction(name, kind)
        assert set(scope.selected_sessions(record["sessions"])) == {object_id}
        scope.consume(_partial_observation(record, {object_id}, frame=frame))
        assert scope.selected_sessions(record["sessions"]) == {}


def test_scope_keeps_unconsumed_interaction_across_operation_switch(prepared_no_knob):
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    _, record = prepared_no_knob
    scope = OperationPerception(record)
    scope.begin_operation("pick_bowl")
    scope.consume(_partial_observation(record, {setup.BOWL}))
    scope.before_interaction("lift", "motion")
    scope.begin_operation("transition")
    assert set(scope.selected_sessions(record["sessions"])) == {setup.BOWL}


def test_knob_events_retain_other_objects_and_refresh_only_on_contact(prepared):
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    _, record = prepared
    scope = OperationPerception(record, include_knob=True)
    scope.begin_operation("rotate_knob")
    scope.consume(_partial_observation(record, {setup.KNOB}))
    for name, kind in [("pregrasp", "motion"), ("open", "position")]:
        scope.before_interaction(name, kind)
        assert scope.selected_sessions(record["sessions"]) == {}
    for frame, (name, kind) in enumerate([("grasp", "motion"), ("close", "grasp"),
                                         ("rotate_0", "last_joint"), ("release", "release")], 2):
        scope.before_interaction(name, kind)
        assert set(scope.selected_sessions(record["sessions"])) == {setup.KNOB}
        scope.consume(_partial_observation(record, {setup.KNOB}, frame=frame))


def test_closed_contact_small_translation_reanchors_root_and_keeps_raw_pose(prepared_no_knob):
    from gene_agent.realworld.microwave.operation_perception import OperationPerception
    _, record = prepared_no_knob
    scope = OperationPerception(record)
    scope.begin_operation("open_door")
    scope.consume(_partial_observation(record, {setup.MICROWAVE}, frame=1))
    root = scope.reader.root.copy()
    scope.before_interaction("handle_pull/03_close", "grasp")
    frame = _partial_observation(record, {setup.MICROWAVE}, frame=2, door=math.radians(-.875))
    point, axis, _ = scope.reader._joint("door")
    center = (frame.object_poses_sdk[setup.MICROWAVE] @ scope.reader.handle_local)[:3, 3]
    radial = center - point
    radial -= axis * (radial @ axis)
    # 重现本轮约20.24mm径向、-1.07mm轴向偏差；闭门接触假设显式记录。
    frame.object_poses_sdk[setup.MICROWAVE][:3, 3] += .0202435 * radial / np.linalg.norm(radial) - .0010707 * axis
    raw = frame.object_poses_sdk[setup.MICROWAVE].copy()
    result, angles, anchor = scope.consume(frame)
    motion = anchor["contact_reanchor"]
    assert .02 < np.linalg.norm(motion["translation_sdk_m"]) < .03
    assert motion["independent_cabinet_measurement"] is False
    assert angles["door"]["angle_rad"] == pytest.approx(0., abs=1e-10)
    np.testing.assert_allclose(scope.reader.root[:3, 3], root[:3, 3] + motion["translation_sdk_m"])
    np.testing.assert_allclose(scope.reader.root[:3, :3], root[:3, :3])
    np.testing.assert_allclose(result.object_poses_sdk[setup.MICROWAVE], raw)
    assert scope.reader.closed_contact_pending is False


@pytest.mark.parametrize("shift,door_deg", [(.031, 0.), (.021, 4.)])
def test_contact_reanchor_rejects_large_or_nonclosed_motion(prepared_no_knob, shift, door_deg):
    _, record = prepared_no_knob
    reader = LiveArticulationReader(record)
    root = reader.root.copy()
    reader.closed_contact_pending = True
    frame = observed(record, door=math.radians(door_deg))
    point, axis, _ = reader._joint("door")
    center = (frame.object_poses_sdk[setup.MICROWAVE] @ reader.handle_local)[:3,3]
    radial = center - point
    radial -= axis * (radial @ axis)
    frame.object_poses_sdk[setup.MICROWAVE][:3,3] += shift * radial / np.linalg.norm(radial)
    with pytest.raises(HandleOrbitError):
        reader(frame)
    np.testing.assert_allclose(reader.root, root)


def test_contact_reanchor_is_bounded_against_original_reference(prepared_no_knob):
    _, record = prepared_no_knob
    reader = LiveArticulationReader(record)
    point, axis, _ = reader._joint("door")
    center = (reader.reference[setup.MICROWAVE] @ reader.handle_local)[:3,3]
    direction = center - point
    direction -= axis * (direction @ axis)
    direction /= np.linalg.norm(direction)
    first = observed(record, frame=1)
    first.object_poses_sdk[setup.MICROWAVE][:3,3] += .021 * direction
    reader.closed_contact_pending = True
    reader(first)
    root = reader.root.copy()
    second = observed(record, frame=2)
    second.object_poses_sdk[setup.MICROWAVE][:3,3] += .042 * direction
    reader.closed_contact_pending = True
    with pytest.raises(HandleOrbitError):
        reader(second)
    np.testing.assert_allclose(reader.root, root)
