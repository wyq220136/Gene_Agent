"""直线约束的纯几何负例；实际求解另由 GPU 检查入口验收。"""
import numpy as np

from gene_agent.motion.path_metrics import measure_linear_tcp_path, measure_pivot_tcp_path
from gene_agent.motion.curobo import add_tcp_tool_frame, add_attached_collision_spheres, CuroboPlannerAdapter
from gene_agent.motion.curobo.core import SynchronousPlanner
from gene_agent.motion import (
    CollisionAllowance, CollisionPolicy, JointState, MotionFailureKind, MotionOperation, MotionRequest,
    PlanningSceneRef, PoseTarget, PlannerMode,
)


def test_arbitrary_direction_and_backward_detection():
    start = np.array([.2, -.3, .7])
    goal = np.array([.5, .1, .8])
    points = start + np.linspace(0, 1, 11)[:, None] * (goal - start)
    good = measure_linear_tcp_path(points, start, goal)
    assert good.satisfies(lateral_m=1e-9, backward_step_m=0.,
                          cumulative_backward_m=0., overshoot_m=0., endpoint_m=1e-9)
    detour = points.copy()
    detour[5, 2] += .03
    assert measure_linear_tcp_path(detour, start, goal).max_lateral_deviation_m > .01
    backwards = points.copy()
    backwards[6] = points[3]
    assert measure_linear_tcp_path(backwards, start, goal).max_backward_step_m > .01


def test_endpoint_error_is_not_hidden_by_line_alignment():
    start = np.zeros(3)
    goal = np.array([.1, 0, 0])
    points = np.array([[.01, 0, 0], [.09, 0, 0]])
    result = measure_linear_tcp_path(points, start, goal)
    assert not result.satisfies(lateral_m=.001, backward_step_m=.001,
                                cumulative_backward_m=.001, overshoot_m=.001, endpoint_m=.005)


def test_rotation_pivot_allows_small_tcp_return_but_rejects_large_detour():
    start = np.array([.35, .12, .16])
    goal = start + [.004, 0., 0.]
    points = np.asarray([start, start + [-.006, .002, 0.],
                         start + [-.004, .003, 0.], goal])
    result = measure_pivot_tcp_path(points, start, goal)
    assert result.satisfies(max_excursion_m=.025, max_path_length_m=.05,
                            endpoint_m=.005)
    detour = points.copy()
    detour[2] += [0., .06, 0.]
    assert not measure_pivot_tcp_path(detour, start, goal).satisfies(
        max_excursion_m=.025, max_path_length_m=.05, endpoint_m=.005)


def test_tcp_frame_is_registered_without_mutating_collision_config():
    source = {"kinematics": {"tool_frames": ["link_7"],
                             "collision_link_names": ["link_7"],
                             "collision_spheres": {"link_7": [{"center": [0, 0, 0], "radius": .02}]}}}
    transform = np.eye(4)
    transform[:3, 3] = [.02, -.01, .04]
    configured = add_tcp_tool_frame(source, "link_7", transform)
    assert configured["kinematics"]["extra_links"]["gene_tcp"]["parent_link_name"] == "link_7"
    assert configured["kinematics"]["extra_links"]["gene_tcp"]["fixed_transform"][:3] == [.02, -.01, .04]
    assert configured["kinematics"]["collision_link_names"] == ["link_7"]
    assert source["kinematics"].get("extra_links") is None


def test_attached_spheres_extend_only_copy_of_tool_collision():
    source = {"kinematics": {"collision_spheres": {
        "tool": [{"center": [0, 0, 0], "radius": .02}],
    }}}
    result = add_attached_collision_spheres(source, "tool", [[.1, 0, 0, .03]])
    assert len(result["kinematics"]["collision_spheres"]["tool"]) == 2
    assert len(source["kinematics"]["collision_spheres"]["tool"]) == 1
    assert result["kinematics"]["collision_spheres"]["tool"][-1]["radius"] == .03


def test_adapter_routes_linear_request_and_preserves_constraint_failure():
    class Backend:
        joint_names = ["a"]
        scene_snapshot_sha256 = "hash"
        scene_scope = "scene"
        planner_mode = PlannerMode.CUDA_GRAPH_BENCHMARK
        random_seed = 3

        def plan_linear_tcp(self, names, q, target, *, target_frame):
            assert target_frame == "world_tcp"
            return {"success": False, "status": "linear_path_rejected",
                    "failure_category": "constraint_violation",
                    "message": "TCP路径横向偏差过大", "duration_s": .2,
                    "linear_path_metrics": {"max_lateral_deviation_m": .02}}

    result = CuroboPlannerAdapter(Backend(), group="arm", scene_revision=1).plan(
        MotionRequest("line", MotionOperation.LINEAR_TCP, "arm", JointState(("a",), (0.,)),
                      PoseTarget("world_tcp", np.eye(4)), PlanningSceneRef(1, "hash", "scene"),
                      CollisionPolicy("finger_object_contact", (
                          CollisionAllowance("gripper_fingers", "bowl", "抓取指尖接触"),
                      )), seed=3)
    )
    assert result.failure.kind is MotionFailureKind.CONSTRAINT_VIOLATION
    assert result.metrics["linear_path_metrics"]["max_lateral_deviation_m"] == .02
    assert result.metrics["effective_collision_policy"] == "strict_conservative"


def test_linear_failure_diagnostic_uses_physical_goal_instead_of_virtual_line_frame():
    """直线辅助 frame 只约束轨迹，不能拿去做失败后的普通 IK 诊断。"""
    planner = object.__new__(SynchronousPlanner)
    physical_goal = object()
    virtual_goal = object()
    planner._goal = lambda target, target_frame: physical_goal

    selected = planner._followup_ik_goal(
        "linear_tcp", np.eye(4), "world_tcp", virtual_goal,
    )

    assert selected is physical_goal
    assert planner._followup_ik_goal("plan", np.eye(4), "world_tcp", virtual_goal) is virtual_goal


def test_linear_solver_strengthens_non_terminal_constraint_by_default():
    planner = object.__new__(SynchronousPlanner)
    captured = {}

    def solve(*args, **kwargs):
        captured.update(kwargs)
        return {"success": False}

    planner._solve = solve
    planner.plan_linear_tcp(["j"], [0.], np.eye(4))
    assert captured["non_terminal_scale"] == 5.0


def test_linear_report_distinguishes_direct_distance_from_travelled_distance():
    points=np.array([[0.,0.,0.],[.05,.03,0.],[.1,0.,0.]])
    metrics=measure_linear_tcp_path(points,points[0],points[-1]).to_dict()
    assert np.isclose(metrics['direct_distance_m'],.1)
    assert np.isclose(metrics['actual_path_length_m'],2*np.hypot(.05,.03))
    assert metrics['path_length_m']==metrics['direct_distance_m'] # 旧字段数值兼容
