"""独立/完整真机任务共用的现场复核接线，不得让旧场景继续发令。"""

from copy import deepcopy

import numpy as np

from gene_agent.motion.model import JointState
from gene_agent.realworld.runtime import RealWorldRuntime
from gene_agent.realworld.scene_drift import SceneDriftLimits, bind_live_scene_guard


NAMES = tuple(f"right_joint_{index}" for index in range(1, 8))
START = JointState(NAMES, (0.,) * 7)
LIMITS = SceneDriftLimits(.01, .05, .02, .005, .03)


class Robot:
    group = "right"

    def read_joint_state(self):
        return START


class Gripper:
    max_width_m = .08

    def __init__(self):
        self.commands = []

    def read_width_m(self):
        return .08

    def command_width_m(self, width):
        self.commands.append(width)


def _scene(frame, bowl_x=0.):
    pose = np.eye(4).tolist()
    bowl = deepcopy(pose)
    bowl[0][3] = bowl_x
    snapshot_hash = f"base-{frame}"
    return {
        "snapshot": {
            "active_group": "right", "world_frame": "world",
            "snapshot_sha256": snapshot_hash,
            "records": [{"geom_name": "bowl_bottom", "body_name": "bowl",
                         "export_kind": "mesh", "mesh_sha256": "shape-v1",
                         "geom_size": [.1, .1, .1], "contact_margin_m": 0.,
                         "contype": 1, "conaffinity": 1}],
        },
        "provenance": {
            "physics_stepped": False, "pose_frame_verified": True,
            "collision_snapshot_sha256": snapshot_hash,
            "source_frame_number": frame, "source_timestamp_ms": 1000. + frame,
            "source_camera_id": "0", "scene_config_sha256": "scene-v1",
            "aligned_object_poses_world": {"Bowl/Bowl_0": bowl},
            "table_pose_world": pose, "table_size_xy_m": [1., .7],
            "accepted_table_thickness_m": .04,
            "accepted_support_size_xyz_m": [.4, .3, .2],
            "accepted_support_center_xy_local_m": [0., 0.],
            "joint_positions_by_group_rad": {"left": [0.] * 7,
                                              "right": [0.] * 7},
            "gripper_widths_m": {"left": .08, "right": .08},
            "door_angle_rad": .3, "knob_angle_rad": .1,
        },
    }


def _inputs(bowl_x):
    scenes = iter((_scene(10), _scene(11, bowl_x)))
    gripper = Gripper()

    def make_runtime(snapshot_hash):
        return RealWorldRuntime(
            Robot(), scene_snapshot_sha256=snapshot_hash,
            max_start_error_rad=.02, tcp_height_fn=lambda _q: .2,
            min_tcp_z_m=.08, gripper=gripper,
        )

    observe, runtime_factory = bind_live_scene_guard(
        lambda: next(scenes), make_runtime, limits=LIMITS,
    )
    return observe, runtime_factory, gripper


def _close(runtime):
    return runtime.run_gripper_stage(
        "close", group="right", target_width_m=.04,
        planned_start=START, planned_start_width_m=.08,
        max_width_error_m=.005,
        planned_scene_snapshot_sha256="derived-stage-hash", execute=True,
    )


def test_derived_collision_stage_still_rechecks_physical_source_scene():
    observe, make_runtime, gripper = _inputs(.05)
    assert observe()["snapshot"]["snapshot_sha256"] == "base-10"
    result = _close(make_runtime("derived-stage-hash"))
    assert result["failure"]["code"] == "scene_recheck_failed"
    assert "Bowl/Bowl_0" in result["failure"]["details"]["reason"]
    assert gripper.commands == []


def test_small_pose_noise_allows_command_and_records_new_frame():
    observe, make_runtime, gripper = _inputs(.003)
    observe()
    result = _close(make_runtime("derived-stage-hash"))
    assert result["status"] == "executed"
    assert result["scene_recheck"]["evidence"]["current_frame_number"] == 11
    assert gripper.commands == [.04]


def test_single_subtask_entry_binds_recheck_without_changing_runner(
    tmp_path, monkeypatch,
):
    from gene_agent.realworld.microwave import task as realworld_subtasks

    scenes = iter((_scene(10), _scene(11, .05)))
    gripper = Gripper()

    def make_runtime(snapshot_hash):
        return RealWorldRuntime(
            Robot(), scene_snapshot_sha256=snapshot_hash,
            max_start_error_rad=.02, tcp_height_fn=lambda _q: .2,
            min_tcp_z_m=.08, gripper=gripper,
        )

    def fake_runner(**kwargs):
        kwargs["observe_collision_scene"]()
        execution = _close(kwargs["make_runtime"]("derived-stage-hash"))
        return {"status": "failed", "success": False, "execution": execution}

    monkeypatch.setattr(realworld_subtasks, "run_observed_bowl_pick", fake_runner)
    result = realworld_subtasks.run_observed_microwave_subtask(
        name="pick_bowl", observe_collision_scene=lambda: next(scenes),
        make_runtime=make_runtime, planner_pool=object(),
        output_dir=tmp_path, options={}, execute=True,
        scene_drift_limits=LIMITS,
    )
    assert result["execution"]["failure"]["code"] == "scene_recheck_failed"
    assert gripper.commands == []


def test_full_task_entry_binds_same_recheck_and_stops_at_first_failure(
    tmp_path, monkeypatch,
):
    from gene_agent.realworld.microwave import task as realworld_task1

    scenes = iter((_scene(10), _scene(11, .05)))
    gripper = Gripper()

    def make_runtime(snapshot_hash):
        return RealWorldRuntime(
            Robot(), scene_snapshot_sha256=snapshot_hash,
            max_start_error_rad=.02, tcp_height_fn=lambda _q: .2,
            min_tcp_z_m=.08, gripper=gripper,
        )

    def fake_runner(**kwargs):
        kwargs["observe_collision_scene"]()
        execution = _close(kwargs["make_runtime"]("derived-stage-hash"))
        return {"status": "failed", "success": False,
                "first_failure": execution["failure"]}

    monkeypatch.setattr(realworld_task1, "run_observed_door_subtask", fake_runner)
    result = realworld_task1.run_observed_microwave_task1(
        observe_collision_scene=lambda: next(scenes),
        make_runtime=make_runtime, planner_pool=object(),
        output_dir=tmp_path, subtask_options={}, execute=True,
        scene_drift_limits=LIMITS,
    )
    assert result["first_failure"]["code"] == "scene_recheck_failed"
    assert len(result["steps"]) == 1
    assert gripper.commands == []
