"""现场入口先核对配置，再允许创建任何相机或机器人连接。"""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.run.run_microwave_task1_realworld import load_site_config, main, run_site
from tools.run import run_microwave_task1_realworld as cli


def test_documented_script_runs_from_project_root_without_pythonpath():
    """用户按文档直接运行脚本时，不应依赖 shell 预设 PYTHONPATH。"""
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "tools/run/run_microwave_task1_realworld.py", "--help"],
        cwd=root, env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--check-config" in completed.stdout


def _site_config(tmp_path: Path) -> Path:
    identity = np.eye(4).tolist()
    mesh = tmp_path / "microwave_handle.obj"
    mesh.write_text("o handle\n", encoding="utf-8")
    angles = tmp_path / "angles.json"
    angles.write_text(json.dumps({
        "door": {"angle_rad": 0., "observed_at_unix_ms": 1., "source": "operator"},
        "knob": {"angle_rad": 0., "observed_at_unix_ms": 1., "source": "operator"},
    }), encoding="utf-8")
    config = {
        "schema_version": 1,
        "active_group": "right",
        "robot_ips": {"left": "169.254.1.1", "right": "169.254.1.2"},
        "camera": {
            "endpoint": "tcp://127.0.0.1:18082", "serial": "camera-1",
            "camera_id": "0", "side": "right", "T_sdk_camera": identity,
            "max_age_s": 5.,
        },
        "pose_server": "http://127.0.0.1:18081",
        "pose_sessions": {
            "Bowl/Bowl_0": "bowl-session",
            "Microwave/Microwave_0": "microwave-session",
        },
        "microwave_pose_mesh_path": str(mesh),
        "state_file": str(angles),
        "scene_parameters": {
            "T_world_sdk": identity,
            "T_perception_object_scene_asset": {
                "Bowl/Bowl_0": identity,
                "Microwave/Microwave_0": identity,
            },
            "table_pose_world": identity,
            "table_size_xy_m": [1., .6],
            "accepted_support_size_xyz_m": [.3, .2, .1],
            "accepted_support_center_xy_local_m": [0., 0.],
            "accepted_table_thickness_m": .04,
            "active_group": "right",
        },
        "runtime": {
            "max_start_error_rad": .02, "min_tcp_z_m": .08,
            "motion_speed_scale": .1, "max_tcp_speed_mps": .1,
            "canfd_frequency_hz": 100.,
        },
        "scene_drift_limits": {
            "max_object_translation_m": .01,
            "max_object_rotation_rad": .1,
            "max_robot_joint_error_rad": .02,
            "max_gripper_width_error_m": .005,
            "max_articulated_angle_error_rad": .05,
        },
        "subtask_options": {
            "pick": {
                "max_bowl_translation_m": .02,
                "max_bowl_rotation_rad": .1,
                "max_start_error_rad": .02,
                "max_gripper_width_error_m": .005,
            },
        },
    }
    path = tmp_path / "site.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_check_config_does_not_connect_robot_or_camera(tmp_path, capsys, monkeypatch):
    """配置检查不能因导入或入口调用而建立任何硬件连接。"""
    path = _site_config(tmp_path)
    monkeypatch.setattr(
        "gene_agent.realworld.devices.arms.realman.connection.connect_realman",
        lambda *a, **k: pytest.fail("检查配置时不能连接机器人"),
    )

    code = main([
        "--config", str(path), "--subtask", "pick_bowl",
        "--output", str(tmp_path / "run"), "--check-config",
    ])

    assert code == 0
    assert "配置检查通过" in capsys.readouterr().out
    assert not (tmp_path / "run").exists()


def _full_site_config(tmp_path):
    """构造能通过全部子任务静态检查的合成站点配置。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    template = json.loads((root / "data/task_1/realworld/realworld_site.example.json").read_text(
        encoding="utf-8"))
    identity = np.eye(4).tolist()
    options = template["subtask_options"]
    for values in options.values():
        for key, value in tuple(values.items()):
            if value is None:
                values[key] = .02
    options["door"]["config_path"] = str(root / "data/task_1/skills/door_skill_debug.json")
    # 测试把模板内容复制进临时目录；相对路径须指向原 task 资产。
    options["knob"]["task_path"] = str(
        root / "data/task_1/realworld/knob_0_task_realworld_tuned.json")
    for kind in ("insert", "extract"):
        options[kind]["microwave_frame_alignment"] = {
            "accepted_for_planning": True, "source": "仅用于离线格式测试",
            "T_scene_asset_profile": identity,
        }
    options["table"]["table_height_alignment"] = {
        "accepted_for_planning": True, "source": "仅用于离线格式测试",
        "height_bias_m": 0.,
    }
    data["subtask_options"] = options
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, options


def test_complete_synthetic_site_config_covers_every_full_task_subtask(tmp_path):
    """完整 Task 的静态参数契约能一次检查全部子任务，且不接设备。"""
    path, options = _full_site_config(tmp_path)

    loaded = load_site_config(path, subtask="all", execute=True)

    assert set(loaded["subtask_options"]) == {
        "door", "pick", "insert", "knob", "extract", "table"}
    assert loaded["scene_drift_limits"] is not None
    assert loaded["subtask_options"]["knob"]["task_path"] == options["knob"]["task_path"]


def test_no_knob_full_task_allows_missing_knob_profile_and_checks_cli(tmp_path):
    """旧版 `--no-knob` 对应的完整任务不能强迫填写未启用的旋钮 Skill 参数。"""
    path, _ = _full_site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["subtask_options"]["knob"]
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_site_config(path, subtask="all", execute=False, no_knob=True)

    assert loaded["include_knob"] is False
    assert "knob" not in loaded["subtask_options"]
    assert main(["--config", str(path), "--subtask", "all",
                 "--no-knob", "--check-config"]) == 0


def test_no_knob_flag_rejected_for_single_subtask(tmp_path):
    """跳过旋钮只影响完整任务，不能改变单个子任务的含义。"""
    path = _site_config(tmp_path)
    with pytest.raises(ValueError, match="--no-knob.*all"):
        main(["--config", str(path), "--subtask", "pick_bowl",
              "--no-knob", "--check-config"])


def test_missing_camera_extrinsic_rejected_before_connection(tmp_path):
    """缺少坐标变换时不能用仿真名义值填补。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["camera"]["T_sdk_camera"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="T_sdk_camera"):
        load_site_config(path, subtask="pick_bowl", execute=False)


@pytest.mark.parametrize("subtask", (
    "pick_bowl", "insert_bowl", "rotate_knob", "extract_bowl",
    "place_bowl_on_table", "all",
))
def test_task1_right_only_subtask_rejects_left_before_connection(tmp_path, subtask):
    """下层已明确限右臂的 task_1 动作，配置检查就应指出边界。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["active_group"] = "left"
    data["scene_parameters"]["active_group"] = "left"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="task_1.*右臂"):
        load_site_config(path, subtask=subtask, execute=False)


def test_execute_requires_drift_limits_and_explicit_yes(tmp_path):
    """执行模式不能因配置文件本身存在而自动授权 SDK 运动。"""
    path = _site_config(tmp_path)
    with pytest.raises(ValueError, match="--yes"):
        main(["--config", str(path), "--subtask", "pick_bowl",
              "--output", str(tmp_path / "run"), "--execute"])
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["scene_drift_limits"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="scene_drift_limits"):
        load_site_config(path, subtask="pick_bowl", execute=True)


def test_missing_scene_transform_names_the_missing_field(tmp_path):
    """现场 frame 缺失要给出字段名，而不是只抛 KeyError。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["scene_parameters"]["T_world_sdk"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="T_world_sdk"):
        load_site_config(path, subtask="pick_bowl", execute=False)


@pytest.mark.parametrize("group", ("right", "left"))
def test_door_profile_path_is_relative_to_site_config(tmp_path, group):
    """站点文件在 task 目录时，相对 profile 路径不应依赖启动 cwd。"""
    path = _site_config(tmp_path)
    door_profile = tmp_path / "door_config.json"
    door_profile.write_text("{}", encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["active_group"] = group
    data["scene_parameters"]["active_group"] = group
    data["camera"]["side"] = group
    data["subtask_options"]["door"] = {
        "config_path": "door_config.json",
        "max_microwave_translation_m": .02,
        "max_microwave_rotation_rad": .1,
        "max_start_error_rad": .02,
        "max_gripper_width_error_m": .005,
        "max_door_angle_error_rad": .05,
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_site_config(path, subtask="open_door", execute=False)

    assert loaded["active_group"] == group
    assert loaded["subtask_options"]["door"]["config_path"] == str(door_profile)


def test_check_config_rejects_missing_angle_source_before_device_connection(tmp_path):
    """角度来源缺失时，静态检查就应给出明确错误。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    state_file = Path(data["state_file"])
    state_file.write_text(json.dumps({"door": {"angle_rad": 0.}}), encoding="utf-8")
    with pytest.raises(ValueError, match="knob"):
        load_site_config(path, subtask="pick_bowl", execute=False)


def test_optional_measured_angle_must_match_planning_angle_before_connection(tmp_path):
    """实测角与规划角冲突不能留到已连接设备后才暴露。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    state_file = Path(data["state_file"])
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["door"]["measured_angle_rad"] = .4
    state_file.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="state_file.door.measured_angle_rad"):
        load_site_config(path, subtask="pick_bowl", execute=False)


def test_malformed_stationary_microwave_anchor_rejected_before_connection(tmp_path):
    """开门后使用的柜体锚点若缺 pose，静态检查就要指出字段。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    state_file = Path(data["state_file"])
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["microwave_root_anchor"] = {
        "T_world_scene_asset": None,
        "verified_at_unix_ms": 1., "source": "现场柜体核对",
    }
    state_file.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="microwave_root_anchor.T_world_scene_asset"):
        load_site_config(path, subtask="pick_bowl", execute=False)


def test_valid_anchor_and_consistent_measured_angle_pass_static_check(tmp_path):
    """合法字段可通过静态检查；现场帧龄仍由实时观察器判断。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    state_file = Path(data["state_file"])
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["door"]["measured_angle_rad"] = .02
    state["microwave_root_anchor"] = {
        "T_world_scene_asset": np.eye(4).tolist(),
        "verified_at_unix_ms": 1., "source": "现场柜体核对",
    }
    state_file.write_text(json.dumps(state), encoding="utf-8")

    loaded = load_site_config(path, subtask="pick_bowl", execute=False)

    assert loaded["source_path"] == str(path)


def test_knob_angle_above_180_degrees_is_allowed_within_asset_joint_limit(tmp_path):
    """knob_0 资产允许约 270°，不能沿用门角的 180° 上限。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    state_file = Path(data["state_file"])
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["knob"].update(angle_rad=3.5, measured_angle_rad=3.5)
    state_file.write_text(json.dumps(state), encoding="utf-8")

    loaded = load_site_config(path, subtask="pick_bowl", execute=False)

    assert loaded["state_file"] == state_file


def test_door_angle_outside_asset_joint_limit_fails_before_connection(tmp_path):
    """门资产只允许约 110°，越界读数要在设备连接前拒绝。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    state_file = Path(data["state_file"])
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["door"]["angle_rad"] = 2.0
    state_file.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="state_file.door.angle_rad.*限位"):
        load_site_config(path, subtask="pick_bowl", execute=False)


def test_site_config_keeps_exact_input_hash_for_run_provenance(tmp_path):
    """调试报告须能追溯这次运行所用的完整现场配置文件。"""
    import hashlib

    path = _site_config(tmp_path)
    loaded = load_site_config(path, subtask="pick_bowl", execute=False)

    assert loaded["source_path"] == str(path)
    assert loaded["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_site_config_change_after_check_stops_before_device_connection(tmp_path):
    """检查和执行之间换了标定文件，必须重新检查，不能用旧解析值下发。"""
    path = _site_config(tmp_path)
    loaded = load_site_config(path, subtask="pick_bowl", execute=False)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["runtime"]["motion_speed_scale"] = .2
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="配置文件.*改变"):
        run_site(loaded, subtask="pick_bowl", output_dir=tmp_path / "run",
                 execute=False)


def test_connection_failure_keeps_traceable_entry_report(tmp_path, monkeypatch):
    """现场 SDK 连接失败也要留下可复查的配置与失败原因。"""
    path = _site_config(tmp_path)
    loaded = load_site_config(path, subtask="pick_bowl", execute=False)

    def disconnected(*_args, **_kwargs):
        raise RuntimeError("SDK offline")

    monkeypatch.setattr("gene_agent.realworld.devices.arms.realman.connection.connect_realman", disconnected)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="SDK offline"):
        run_site(loaded, subtask="pick_bowl", output_dir=output, execute=False)

    record = json.loads((output / "entry_result.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["failure"]["code"] == "connection_failed"
    assert record["site_config_sha256"] == loaded["source_sha256"]


def test_required_subtask_tolerance_cannot_be_null(tmp_path):
    """模板中的未标定容差不能通过静态检查并留到连接后才失败。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["subtask_options"]["pick"]["max_bowl_translation_m"] = None
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="max_bowl_translation_m"):
        load_site_config(path, subtask="pick_bowl", execute=False)


def test_unknown_subtask_option_is_caught_before_connection(tmp_path):
    """参数拼写错误应在配置检查时定位，不应等机器人连接后才报 TypeError。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["subtask_options"]["pick"]["max_bowl_translaton_m"] = .02
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="max_bowl_translaton_m"):
        load_site_config(path, subtask="pick_bowl", execute=False)


def test_check_config_has_short_command_without_output_dir(tmp_path, capsys):
    """只检查配置时用户不需要手动指定一次运行的记录目录。"""
    path = _site_config(tmp_path)
    assert main(["--config", str(path), "--subtask", "pick_bowl",
                 "--check-config"]) == 0
    assert "配置检查通过" in capsys.readouterr().out


def test_invalid_site_config_cli_reports_field_without_traceback():
    """命令行遇到待标定模板时，应直接指出字段，不用 Python traceback 淹没错误。"""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "tools/run/run_microwave_task1_realworld.py"),
         "--config", str(root / "data/task_1/realworld/realworld_site.example.json"),
         "--subtask", "pick_bowl", "--check-config"],
        cwd=root, text=True, capture_output=True, check=False,
    )

    assert result.returncode == 2
    assert "robot_ips.left" in result.stderr
    assert "Traceback" not in result.stderr


def test_observer_setup_failure_records_reason_and_closes_both_arms(tmp_path, monkeypatch):
    """连接后相机/观察器初始化失败也必须有结果文件，且关闭两臂连接。"""
    path = _site_config(tmp_path)
    loaded = load_site_config(path, subtask="pick_bowl", execute=False)
    closed = []

    class Connection:
        robot = object()
        gripper = object()

        def __init__(self, group):
            self.group = group

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            closed.append(self.group)

    def unavailable_observer(**_kwargs):
        raise RuntimeError("camera unavailable")

    monkeypatch.setattr(
        "gene_agent.realworld.devices.arms.realman.connection.connect_realman",
        lambda _ip, *, group: Connection(group),
    )
    monkeypatch.setattr(
        "gene_agent.realworld.microwave.scene.make_live_microwave_observer",
        unavailable_observer,
    )
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="camera unavailable"):
        run_site(loaded, subtask="pick_bowl", output_dir=output, execute=False)

    assert sorted(closed) == ["left", "right"]
    record = json.loads((output / "entry_result.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["failure"]["code"] == "entry_failed"
    assert "camera unavailable" in record["failure"]["message"]
    assert record["site_config_sha256"] == loaded["source_sha256"]


@pytest.mark.parametrize("include_knob", (True, False))
def test_full_task_failure_closes_both_arms_and_planner_pool(
    tmp_path, monkeypatch, include_knob,
):
    """完整 Task 入口异常时，资源和失败报告应在同一次运行内收束。"""
    path = _site_config(tmp_path)
    loaded = load_site_config(path, subtask="pick_bowl", execute=False)
    loaded["include_knob"] = include_knob
    events = []

    class Connection:
        robot = object()
        gripper = object()

        def __init__(self, group):
            self.group = group

        def __enter__(self):
            events.append(("connect", self.group))
            return self

        def __exit__(self, *_args):
            events.append(("close", self.group))

    class Pool:
        def __init__(self, *, planner_factory, max_entries):
            assert callable(planner_factory) and max_entries == 6

        def clear(self):
            events.append(("pool", "clear"))

    monkeypatch.setattr(
        "gene_agent.realworld.devices.arms.realman.connection.connect_realman",
        lambda _ip, *, group: Connection(group),
    )
    monkeypatch.setattr(
        "gene_agent.realworld.microwave.scene.make_live_microwave_observer",
        lambda **_kwargs: lambda: {"source": "fresh-observation"},
    )
    monkeypatch.setattr(
        "gene_agent.perception_modules.pose_est.pose_client.PoseClient",
        lambda _server: object(),
    )
    monkeypatch.setattr("gene_agent.motion.curobo.session.CuroboPlannerPool", Pool)

    def task(**kwargs):
        assert kwargs["observe_collision_scene"]() == {
            "source": "fresh-observation"}
        assert kwargs["subtask_options"]["pick"] == loaded["subtask_options"]["pick"]
        assert kwargs["execute"] is False
        assert kwargs["include_knob"] is include_knob
        events.append(("task", "all"))
        raise RuntimeError("synthetic planning failure")

    monkeypatch.setattr(cli, "run_observed_microwave_task1", task)
    output = tmp_path / "full_run"
    with pytest.raises(RuntimeError, match="synthetic planning failure"):
        run_site(loaded, subtask="all", output_dir=output, execute=False)

    assert events == [
        ("connect", "left"), ("connect", "right"), ("task", "all"),
        ("pool", "clear"), ("close", "right"), ("close", "left"),
    ]
    entry = json.loads((output / "entry_result.json").read_text(encoding="utf-8"))
    site_input = json.loads((output / "site_input.json").read_text(encoding="utf-8"))
    assert site_input["include_knob"] is include_knob
    assert entry["failure"]["code"] == "entry_failed"
    assert "synthetic planning failure" in entry["failure"]["message"]
    assert entry["site_config_sha256"] == loaded["source_sha256"]


def test_insert_requires_accepted_microwave_alignment_before_connecting(tmp_path):
    """frame 对齐显式未验收时，不能让独立放碗或完整 Task 运行到接触阶段。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["subtask_options"]["insert"] = {
        "microwave_frame_alignment": {
            "accepted_for_planning": False, "source": "",
            "T_scene_asset_profile": None,
        },
        "max_microwave_translation_m": .01,
        "max_microwave_rotation_rad": .1,
        "max_hold_translation_m": .01,
        "max_hold_rotation_rad": .1,
        "max_start_error_rad": .02,
        "max_gripper_width_error_m": .005,
    }
    hold = tmp_path / "held.json"
    hold.write_text("{}", encoding="utf-8")
    data["prepared_hold_report"] = str(hold)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="microwave_frame_alignment.*未验收"):
        load_site_config(path, subtask="insert_bowl", execute=False)
    data["subtask_options"]["insert"]["microwave_frame_alignment"].update(
        accepted_for_planning=True, source="现场 frame 标定",
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="T_scene_asset_profile"):
        load_site_config(path, subtask="insert_bowl", execute=False)


def test_table_place_requires_accepted_height_alignment_before_connecting(tmp_path):
    """桌面高度对齐仍是占位值时，不能等接触阶段才发现。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["subtask_options"]["table"] = {
        "table_height_alignment": {
            "accepted_for_planning": False, "source": "",
            "height_bias_m": None,
        },
        "max_table_height_delta_m": .01,
        "max_table_translation_m": .01,
        "max_table_rotation_rad": .1,
        "max_bowl_translation_m": .01,
        "max_bowl_rotation_rad": .1,
        "max_hold_translation_m": .01,
        "max_hold_rotation_rad": .1,
        "max_start_error_rad": .02,
        "max_gripper_width_error_m": .005,
    }
    hold = tmp_path / "held.json"
    hold.write_text("{}", encoding="utf-8")
    data["prepared_hold_report"] = str(hold)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="table_height_alignment.*未验收"):
        load_site_config(path, subtask="place_bowl_on_table", execute=False)
    data["subtask_options"]["table"]["table_height_alignment"].update(
        accepted_for_planning=True, source="现场桌面测量",
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="height_bias_m"):
        load_site_config(path, subtask="place_bowl_on_table", execute=False)


def test_site_config_rejects_json_value_for_live_check_callback(tmp_path):
    """现场检查器是代码回调；JSON 中填布尔值不能等运动后才因不可调用失败。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["subtask_options"]["pick"]["check_pick"] = True
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="check_pick.*不能写入 JSON"):
        load_site_config(path, subtask="pick_bowl", execute=False)


def test_unset_runtime_limit_names_field_in_configuration_error(tmp_path):
    """现场速度或安全阈值尚未填写时，配置检查应指出字段而非抛 TypeError。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["runtime"]["min_tcp_z_m"] = None
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime.min_tcp_z_m"):
        load_site_config(path, subtask="pick_bowl", execute=False)


def test_unset_door_angle_names_field_in_configuration_error(tmp_path):
    """门角读数缺失必须在连接前定位到 state_file 的具体字段。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    state = Path(data["state_file"])
    readings = json.loads(state.read_text(encoding="utf-8"))
    readings["door"]["angle_rad"] = None
    state.write_text(json.dumps(readings), encoding="utf-8")

    with pytest.raises(ValueError, match="state_file.door.angle_rad"):
        load_site_config(path, subtask="pick_bowl", execute=False)


def test_unset_drift_limit_names_field_before_execute(tmp_path):
    """执行保护容差留空时，必须在任何现场连接前指出是哪一个容差。"""
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["scene_drift_limits"]["max_object_translation_m"] = None
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="scene_drift_limits.max_object_translation_m"):
        load_site_config(path, subtask="pick_bowl", execute=True)


def _insert_site_without_hold_file(tmp_path):
    path = _site_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["subtask_options"]["insert"] = {
        "microwave_frame_alignment": {
            "accepted_for_planning": True, "source": "测试用标定",
            "T_scene_asset_profile": np.eye(4).tolist(),
        },
        "max_microwave_translation_m": .01,
        "max_microwave_rotation_rad": .1,
        "max_hold_translation_m": .01,
        "max_hold_rotation_rad": .1,
        "max_start_error_rad": .02,
        "max_gripper_width_error_m": .005,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_standalone_insert_can_prepare_hold_from_current_observation(tmp_path, monkeypatch):
    """独立放碗的人工作业证据应绑定当前帧，规划首段复用该帧。"""
    path = _insert_site_without_hold_file(tmp_path)
    loaded = load_site_config(path, subtask="insert_bowl", execute=False,
                              prepare_held_bowl=True)
    events = []
    scene = {
        "snapshot": {"snapshot_sha256": "same-frame", "active_group": "right"},
        "provenance": {
            "physics_stepped": False, "pose_frame_verified": True,
            "collision_snapshot_sha256": "same-frame", "source_frame_number": 23,
            "source_timestamp_ms": 1000.,
            "aligned_object_poses_world": {"Bowl/Bowl_0": np.eye(4).tolist()},
        },
    }

    class Connection:
        robot = object()
        gripper = object()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            events.append("closed")

    class Pool:
        def __init__(self, **_kwargs):
            pass

        def clear(self):
            events.append("pool_closed")

    def observer(**_kwargs):
        def observe():
            events.append("observe")
            return scene
        return observe

    def subtask(**kwargs):
        assert kwargs["observe_collision_scene"]() is scene
        prepared = kwargs["prepared_hold_report"]
        assert prepared["source_format"] == "manual_prepared_held_bowl"
        assert prepared["source_collision_snapshot_sha256"] == "same-frame"
        assert prepared["postcondition"]["status"] == "pass"
        assert events == ["observe"]
        return {"status": "validated_prefix", "success": False}

    monkeypatch.setattr("gene_agent.realworld.devices.arms.realman.connection.connect_realman",
                        lambda *_args, **_kwargs: Connection())
    monkeypatch.setattr("gene_agent.realworld.microwave.scene.make_live_microwave_observer",
                        observer)
    monkeypatch.setattr("gene_agent.perception_modules.pose_est.pose_client.PoseClient",
                        lambda _server: object())
    monkeypatch.setattr("gene_agent.motion.curobo.session.CuroboPlannerPool", Pool)
    monkeypatch.setattr(cli, "run_observed_microwave_subtask", subtask)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    output = tmp_path / "run"
    result = run_site(loaded, subtask="insert_bowl", output_dir=output,
                      execute=False, prepare_held_bowl=True)

    assert result["status"] == "validated_prefix"
    saved = json.loads((output / "prepared_hold_report.json").read_text(encoding="utf-8"))
    assert saved["source_frame_number"] == 23
    assert events == ["observe", "pool_closed", "closed", "closed"]


def test_prepare_hold_rejects_failed_operator_confirmation_before_planning(tmp_path, monkeypatch):
    """未确认夹住碗时，独立放碗不能把人工准备写成成功证据。"""
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    scene = {
        "snapshot": {"snapshot_sha256": "same-frame", "active_group": "right"},
        "provenance": {
            "physics_stepped": False, "pose_frame_verified": True,
            "collision_snapshot_sha256": "same-frame", "source_frame_number": 23,
            "source_timestamp_ms": 1000.,
            "aligned_object_poses_world": {"Bowl/Bowl_0": np.eye(4).tolist()},
        },
    }
    with pytest.raises(ValueError, match="未确认.*夹持"):
        cli._prepare_current_held_bowl_report(
            scene, output=tmp_path, ask=input,
        )
    assert not (tmp_path / "prepared_hold_report.json").exists()


def test_prepare_hold_only_for_standalone_hold_subtasks(tmp_path):
    path = _site_config(tmp_path)
    with pytest.raises(ValueError, match="只适用于独立放碗或桌面 Place"):
        load_site_config(path, subtask="pick_bowl", execute=False,
                         prepare_held_bowl=True)


def test_prepared_hold_file_is_checked_before_connecting_devices(tmp_path):
    """已有持碗文件不能只查存在：格式错误会在连机器前暴露。"""
    path = _insert_site_without_hold_file(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    hold = tmp_path / "held.json"
    hold.write_text("{}", encoding="utf-8")
    data["prepared_hold_report"] = str(hold)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="prepared_hold_report.*source_format"):
        load_site_config(path, subtask="insert_bowl", execute=False)


def test_valid_manual_hold_file_passes_static_check(tmp_path):
    """静态契约接受现场生成的报告；帧时效仍留给现场观测检查。"""
    path = _insert_site_without_hold_file(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    hold = tmp_path / "held.json"
    hold.write_text(json.dumps({
        "schema_version": 1,
        "source_format": "manual_prepared_held_bowl",
        "object_id": "Bowl/Bowl_0",
        "source_collision_snapshot_sha256": "snapshot-1",
        "source_frame_number": 17,
        "source_timestamp_ms": 1000.,
        "postcondition": {"status": "pass", "source": "manual_terminal"},
    }), encoding="utf-8")
    data["prepared_hold_report"] = str(hold)
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_site_config(path, subtask="insert_bowl", execute=False)
    assert loaded["prepared_hold_report"] == hold


def test_prepared_hold_file_rejects_conflicting_inner_frame_before_connection(tmp_path):
    """持碗文件内部证据若指向旧帧，静态检查就应指出矛盾。"""
    path = _insert_site_without_hold_file(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    hold = tmp_path / "stale-held.json"
    hold.write_text(json.dumps({
        "schema_version": 1,
        "source_format": "manual_prepared_held_bowl",
        "object_id": "Bowl/Bowl_0",
        "source_collision_snapshot_sha256": "snapshot-17",
        "source_frame_number": 17,
        "source_timestamp_ms": 17000.,
        "postcondition": {
            "status": "pass", "source": "manual_terminal",
            "evidence": {"source_frame_number": 16},
        },
    }), encoding="utf-8")
    data["prepared_hold_report"] = str(hold)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="prepared_hold_report.*来源不一致"):
        load_site_config(path, subtask="insert_bowl", execute=False)


def test_plan_only_with_saved_hold_file_can_request_fresh_manual_recheck(
    tmp_path, monkeypatch,
):
    """历史持碗文件配合新相机帧时，预检也必须能请求当前持碗复核。"""
    path = _insert_site_without_hold_file(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    hold = tmp_path / "held.json"
    hold.write_text(json.dumps({
        "schema_version": 1, "source_format": "manual_prepared_held_bowl",
        "object_id": "Bowl/Bowl_0",
        "source_collision_snapshot_sha256": "old-snapshot",
        "source_frame_number": 1, "source_timestamp_ms": 100.,
        "postcondition": {"status": "pass", "source": "manual_terminal"},
    }), encoding="utf-8")
    data["prepared_hold_report"] = str(hold)
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_site_config(path, subtask="insert_bowl", execute=False)
    events = []

    class Connection:
        robot = object()
        gripper = object()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            events.append("closed")

    class Pool:
        def __init__(self, **_kwargs):
            pass

        def clear(self):
            events.append("pool_closed")

    monkeypatch.setattr("gene_agent.realworld.devices.arms.realman.connection.connect_realman",
                        lambda *_args, **_kwargs: Connection())
    monkeypatch.setattr("gene_agent.realworld.microwave.scene.make_live_microwave_observer",
                        lambda **_kwargs: lambda: {"source": "new-frame"})
    monkeypatch.setattr("gene_agent.perception_modules.pose_est.pose_client.PoseClient",
                        lambda _server: object())
    monkeypatch.setattr("gene_agent.motion.curobo.session.CuroboPlannerPool", Pool)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    def subtask(**kwargs):
        assert kwargs["execute"] is False
        assert kwargs["prepared_hold_report"]["source_frame_number"] == 1
        assert kwargs["options"]["ask_manual"]("当前仍夹持？") == "y"
        events.append("subtask")
        return {"status": "validated_prefix", "success": False}

    monkeypatch.setattr(cli, "run_observed_microwave_subtask", subtask)
    run_site(loaded, subtask="insert_bowl", output_dir=tmp_path / "run",
             execute=False)
    assert events == ["subtask", "pool_closed", "closed", "closed"]


@pytest.mark.parametrize('subtask,kinds', [('load_bowl',('pick','insert')), ('unload_bowl',('extract','table'))])
def test_combo_site_config_requires_both_operations_and_no_prepared_hold(tmp_path, subtask, kinds):
    path, _ = _full_site_config(tmp_path)
    data = json.loads(path.read_text())
    data['subtask_options'] = {kind:data['subtask_options'][kind] for kind in kinds}
    path.write_text(json.dumps(data))
    assert load_site_config(path, subtask=subtask, execute=False)['subtask_options'].keys() == set(kinds)
    data['subtask_options'].pop(kinds[-1])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=kinds[-1]):
        load_site_config(path, subtask=subtask, execute=False)


@pytest.mark.parametrize('subtask', ['load_bowl', 'unload_bowl'])
def test_combo_rejects_old_hold_report_before_device_connection(tmp_path, subtask):
    path, _ = _full_site_config(tmp_path)
    data = json.loads(path.read_text())
    data['prepared_hold_report'] = 'old-held-bowl.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='组合.*持碗'):
        load_site_config(path, subtask=subtask, execute=False)
