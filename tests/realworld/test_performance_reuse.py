"""性能优化必须保持精确状态边界，并在异常时收齐后台设备读取。"""
from dataclasses import replace
from threading import Barrier, Event, Lock
import os
import time
import numpy as np
import pytest

from gene_agent.motion.model import (
    JointState, JointTrajectory, MotionRequest, MotionResult, MotionOperation,
    PlanningSceneRef, PoseTarget, CollisionPolicy,
)
from gene_agent.realworld.microwave.planning_reuse import reuse_selected_motion, summarize_selection
from gene_agent.realworld.state_sampling import RobotStateSampler
from gene_agent.realworld.runtime import RealWorldRuntime
from gene_agent.skills.library.pick.pregrasp_search import PregraspRequestCache
from gene_agent.tasks.checks import CheckResult


def request(name="candidate:pregrasp"):
    state = JointState(tuple(f"right_joint_{i}" for i in range(1, 8)), (0.,) * 7)
    return MotionRequest(name, MotionOperation.PLAN, "right", state,
                         PoseTarget("world_tcp", np.eye(4)), PlanningSceneRef(0, "scene", "full"),
                         seed=123, joint_target=state)


def result(req):
    return MotionResult(req.request_id, req.operation, True, "curobo", req.planner_mode, 2.,
                        goal_state=req.start, trajectory=JointTrajectory(
                            req.start.joint_names, (0., .1), (req.start.positions_rad,) * 2))


def selection(req):
    return {"reuse_process_id": os.getpid(), "planning_reuse_identity": "identity",
            "stage_collision_snapshot_sha256": {"pregrasp": "scene"},
            "plan": {"stages": [{"name": "pregrasp", "status": "planned",
                                  "request": req.to_dict(), "result": result(req).to_dict()}]}}


def test_selection_reuses_only_exact_state_and_rebinds_request_id():
    old = request()
    new = replace(old, request_id="execution", scene=replace(old.scene, revision=1))
    reused, _, reason = reuse_selected_motion(selection(old), "pregrasp", new,
        identity="identity", strict_snapshot_sha256="scene")
    assert reason == "exact_request_reused"
    assert reused.request_id == "execution" and reused.duration_s == 0
    assert reused.trajectory == result(old).trajectory


@pytest.mark.parametrize("change", ["start", "snapshot", "seed", "policy", "target", "joint_target", "identity", "strict", "process"])
def test_selection_never_reuses_changed_contract(change):
    old = request()
    selected, new, identity, strict = selection(old), old, "identity", "scene"
    if change == "start":
        new = replace(old, start=replace(old.start, positions_rad=(1e-12,) + (0.,) * 6))
    elif change == "snapshot":
        new = replace(old, scene=replace(old.scene, snapshot_sha256="changed"))
    elif change == "seed":
        new = replace(old, seed=124)
    elif change == "policy":
        new = replace(old, collision_policy=CollisionPolicy("changed"))
    elif change == "target":
        pose = np.eye(4); pose[0, 3] = 1e-12
        new = replace(old, target=PoseTarget("world_tcp", pose))
    elif change == "joint_target":
        new = replace(old, joint_target=None)
    elif change == "identity":
        identity = "changed"
    elif change == "strict":
        strict = "changed"
    elif change == "process":
        selected["reuse_process_id"] = -1
    reused, _, _ = reuse_selected_motion(selected, "pregrasp", new, identity=identity, strict_snapshot_sha256=strict)
    assert reused is None


def test_global_dedup_packs_batches_and_preserves_all_ids():
    batches = []
    class Planner:
        def evaluate_candidates(self, requests):
            batches.append(len(requests))
            return tuple(replace(result(r), trajectory=None) for r in requests)
    base = replace(request(), operation=MotionOperation.IK, joint_target=None)
    requests = []
    for i in range(260):
        pose = np.eye(4); pose[0, 3] = float(i % 130)
        requests.append(replace(base, request_id=str(i), target=PoseTarget("world_tcp", pose)))
    rows = PregraspRequestCache(Planner()).evaluate_candidates(tuple(requests), batch_size=128)
    assert batches == [128, 2]
    assert [r.request_id for r in rows] == [str(i) for i in range(260)]
    assert sum(bool(r.metrics.get("request_reuse")) for r in rows) == 130
    assert rows[131].goal_state == rows[1].goal_state


def test_signed_zero_normalization_is_exact_not_pose_rounding():
    a = request(); pose = np.eye(4); pose[0, 3] = -0.
    b = replace(a, target=PoseTarget("world_tcp", pose))
    assert PregraspRequestCache.key(a) == PregraspRequestCache.key(b)
    pose[0, 3] = 1e-15
    c = replace(a, target=PoseTarget("world_tcp", pose))
    assert PregraspRequestCache.key(a) != PregraspRequestCache.key(c)


def test_independent_arms_overlap_and_width_override_avoids_read():
    barrier = Barrier(3, timeout=3)
    reads = []
    class Robot:
        def __init__(self, group): self.group, self.sdk = group, object()
        def read_joint_state(self):
            barrier.wait()
            return JointState(tuple(f"{self.group}_joint_{i}" for i in range(1,8)), (0.,)*7)
    class Gripper:
        def __init__(self, robot): self.sdk, self.group = robot.sdk, robot.group
        def read_width_m(self): reads.append(self.group); return .04
    robots = {s: Robot(s) for s in ("left", "right")}
    sampler = RobotStateSampler(robots, {s: Gripper(r) for s, r in robots.items()})
    try:
        with sampler.sample({"right": .05}) as pending:
            barrier.wait()  # 两臂读取可以与本线程视觉处理同时进行。
            samples = pending.result()
        assert reads == ["left"]
        assert samples["right"]["width_m"] == .05
        assert samples["right"]["width_source"] == "confirmed_execution_endpoint"
    finally:
        sampler.close()
    with pytest.raises(RuntimeError):
        with sampler.sample(): pass


def test_shared_sdk_is_serialized():
    active, maximum = [0], [0]
    lock = Lock(); sdk = object()
    class Device:
        def __init__(self): self.sdk = sdk
        def read_joint_state(self):
            with lock:
                active[0] += 1; maximum[0] = max(maximum[0], active[0])
            time.sleep(.01)
            with lock: active[0] -= 1
            return request().start
        def read_width_m(self): return .04
    devices = {s: Device() for s in ("left", "right")}
    sampler = RobotStateSampler(devices, devices)
    try:
        with sampler.sample() as pending: pending.result()
        assert maximum[0] == 1
    finally: sampler.close()


def test_visual_failure_joins_readers_before_return():
    done = []
    class Device:
        def read_joint_state(self): time.sleep(.01); done.append(True); return request().start
        def read_width_m(self): return .04
    devices = {s: Device() for s in ("left", "right")}
    sampler = RobotStateSampler(devices, devices)
    try:
        with pytest.raises(ValueError, match="vision"):
            with sampler.sample(): raise ValueError("vision")
        assert len(done) == 2 and not sampler.busy
    finally: sampler.close()


@pytest.mark.parametrize("age,expected_reads", [(0., 1), (1., 2)])
def test_guard_sample_avoids_initial_read_but_keeps_final_command_check(age, expected_reads):
    req = request(); reads = []; widths = []; commands = []
    class Robot:
        group = "right"
        def read_joint_state(self): reads.append(True); return req.start
    class Gripper:
        max_width_m = .08
        def read_width_m(self): widths.append(True); return .04
        def command_width_m(self, width, *, blocking=True): commands.append(width)
    calls = []
    def check():
        calls.append(True)
        return CheckResult("pass", "guard", {"robot_sample": {
            "group": "right", "started_monotonic_s": time.monotonic() - age,
            "joint_positions_rad": [0.] * 7, "width_m": .04, "width_source": "sdk_live_read"}})
    check.provides_robot_sample = True
    runtime = RealWorldRuntime(Robot(), gripper=Gripper(), scene_snapshot_sha256="scene",
        max_start_error_rad=.02, tcp_height_fn=lambda _: .2, min_tcp_z_m=.08,
        pre_execute_scene_check=check)
    out = runtime.run_gripper_stage("open", group="right", target_width_m=.04,
        planned_start=req.start, planned_start_width_m=.04, max_width_error_m=.005,
        planned_scene_snapshot_sha256="scene", execute=True)
    assert out["success"] and commands == [.04] and len(calls) == 1
    assert len(reads) == expected_reads
    assert len(widths) == expected_reads + 2  # 终态稳定确认读取三次。


def test_selection_summary_separates_backend_work_and_reuse():
    req = request(); first = result(req)
    second = PregraspRequestCache.reuse(replace(req, request_id="other"), first)
    report = summarize_selection({"attempts": [
        {"stage": "pregrasp", "request": req.to_dict(), "result": first.to_dict()},
        {"stage": "pregrasp", "request": replace(req, request_id="other").to_dict(), "result": second.to_dict()}]})
    assert report["unique_request_count"] == 1
    assert report["stages"]["pregrasp"]["cache_hits"] == 1
    assert report["stages"]["pregrasp"]["backend_calls_or_batch_items"] == 1


def test_bowl_stage_reuses_selected_joint_target_through_production_entry(tmp_path, monkeypatch):
    from gene_agent.realworld.microwave import pick_planning as module
    old = request()
    selected = selection(old)
    selected.update(grasp_prefix_planned=True, selected_candidate_id="candidate")
    scene = {"snapshot": {"snapshot_sha256": "scene"},
             "provenance": {"source_frame_number": 2}}
    snapshot = {"snapshot_sha256": "scene"}
    monkeypatch.setattr(module, "_check_anchor", lambda *a, **k: (.04, {}))
    monkeypatch.setattr(module, "build_observed_bowl_grasp_stage_snapshots", lambda *a, **k: (snapshot, snapshot))
    monkeypatch.setattr(module, "_planning_reuse_identity", lambda *a: "identity")
    class Planner:
        def plan(self, req): pytest.fail("相同完整请求不应再次进入求解器")
    monkeypatch.setattr(module, "prepare_plan_only_context", lambda *a, **k: {
        "start": old.start, "scene": replace(old.scene, revision=2), "planner": Planner(),
        "planner_cache_hit": True, "planner_setup_duration_s": 0.})
    report = module.plan_observed_bowl_motion_stage(scene, selected, stage_name="pregrasp",
        planner_pool=object(), output_dir=tmp_path, max_bowl_translation_m=.005,
        max_bowl_rotation_rad=.02, max_gripper_width_error_m=.002)
    assert report["success"]
    assert report["selection_plan_reuse"] == "exact_request_reused"
    assert report["stage"].request.joint_target == old.joint_target


def test_checked_path_cache_keeps_geometry_result_and_rechecks_changed_request():
    from gene_agent.tasks.policies.microwave_task.planning.grasp import _BowlGeometryCheckedAdapter
    adapter = object.__new__(_BowlGeometryCheckedAdapter)
    calls = []
    def checked(req):
        calls.append(req)
        return replace(result(req), metrics={"actual_geometry_postcheck": {"success": True}})
    adapter._plan_checked = checked
    first = request()
    adapter.plan(first)
    reused = adapter.plan(replace(first, request_id="same-path"))
    assert reused.metrics["actual_geometry_postcheck"]["success"]
    assert reused.duration_s == 0 and len(calls) == 1
    adapter.plan(replace(first, start=replace(first.start, positions_rad=(1e-12,) + (0.,)*6)))
    assert len(calls) == 2
