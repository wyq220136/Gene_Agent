"""两个 Knowledge 界面的纯逻辑验收，不占用 Viser 端口。"""
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gene_agent.visualization.grasp_tuner.controls import GraspParameterControls, numeric_spec
from gene_agent.visualization.grasp_tuner.controls import _limits
from gene_agent.visualization.grasp_tuner.pose_ik import MuJoCoPoseIK
from gene_agent.tasks.policies.microwave_task.planning.bowl import load_pick_bowl_case
from gene_agent.visualization.grasp_tuner.lab_app import annular_sector, KnowledgeGraspLabApp, GEOMETRY_DEFAULTS, DEFAULT_PARAMETERS
from gene_agent.knowledge.grasp import generate_grasps, parse_grasp_profiles
from gene_agent.visualization.grasp_tuner.scene_app import SceneGraspTunerApp
from gene_agent.visualization.grasp_tuner.selection import closest_candidate_index, representative_candidate_indices
from gene_agent.visualization.grasp_tuner.feedback import read_tuner_state
from gene_agent.visualization.grasp_tuner.session import build_session_state
from tools.viewers.capture.capture_knowledge_viser import _scene_material_ready


ROOT = Path(__file__).resolve().parents[3]


def test_scene_tuner_applies_gt_object_pose_exactly_once():
    local = build_session_state(ROOT, "task_1_bowl_0")
    world = np.eye(4)
    world[:3, 3] = [.3, .2, -.078]
    scene = build_session_state(ROOT, "task_1_bowl_0", object_pose_world=world)
    np.testing.assert_allclose(scene.candidates[0].tcp_pose_world, world @ local.candidates[0].tcp_pose_world)


def test_slider_modes_preserve_fixed_and_range_specs():
    assert numeric_spec({"values": [55]}) == ("固定", [55.])
    assert numeric_spec({"values": [0, 90]}) == ("列表", [0., 90.])
    assert numeric_spec({"start": 0, "stop": 315, "step": 45}) == ("范围", [0., 315., 45.])


def test_tip_sliders_use_fine_ranges_without_clipping_existing_values():
    assert _limits("tip_y_deg", [0.]) == (-30., 30., .5)
    assert _limits("tip_x_deg", [-60.]) == (-90., 90., .5)
    assert _limits("tip_y_deg", [-45., 55.]) == (-45., 55., .5)


def test_ring_profile_exposes_only_one_axial_offset():
    payload = json.loads((ROOT / "data/task_1/knowledge/grasp_profiles.json").read_text())
    params = payload["profiles"]["task_1_bowl_0"]["parameters"]
    assert "vertical_offset_m" in params and "depth_m" not in params
    params["depth_m"] = {"values": [.001]}
    with pytest.raises(ValueError, match="必须且只能"):
        build_session_state(ROOT, "task_1_bowl_0", profile_document=payload)


def test_pose_ik_solves_nearby_tcp_without_collision_queries():
    case = load_pick_bowl_case(ROOT / "runs/pick-bowl-planning-final")
    robot = case.metadata["robot"]
    group = case.snapshot.get("active_group", "right")
    site = case.data.site(robot["tcp_sites"][group])
    goal = np.eye(4)
    goal[:3, :3] = site.xmat.reshape(3, 3)
    goal[:3, 3] = site.xpos + [.01, 0., 0.]
    solver = MuJoCoPoseIK(case.model, case.data, tuple(robot["joint_groups"][group]), robot["tcp_sites"][group])
    solved = solver.solve(goal)
    assert solved.reachable
    assert solved.position_error_m < .003
    assert solved.rotation_error_deg < 4.


def test_gripper_focus_keeps_camera_above_table_and_shows_both_fingers():
    case = load_pick_bowl_case(ROOT / "runs/pick-bowl-planning-final")
    state = build_session_state(ROOT, "task_1_bowl_0", object_pose_world=case.bowl_pose)
    app = SceneGraspTunerApp.__new__(SceneGraspTunerApp)
    app.case, app.state = case, state
    app.candidate = SimpleNamespace(value=0)
    app.gripper_preview = SimpleNamespace(data=case.data, available=True)
    table = case.model.geom("scene_table")
    table_top = case.data.geom("scene_table").xpos[2] + table.size[2]

    robot = case.metadata["robot"]
    group = case.snapshot["active_group"]
    names = robot["grippers"][group]["fingers"]
    source_tcp = case.data.site(robot["tcp_sites"][group])
    opening_source = (case.data.site(names["positive"]["inner_site_name"]).xpos
                      - case.data.site(names["negative"]["inner_site_name"]).xpos)
    opening_local = source_tcp.xmat.reshape(3, 3).T @ opening_source
    for index, proposal in enumerate(state.candidates):
        app.candidate.value = index
        camera = app._gripper_camera()
        position = np.asarray(camera["position"])
        look_at = np.asarray(camera["look_at"])
        assert position[2] > table_top + .03
        assert look_at[2] > table_top
        opening_world = proposal.tcp_pose_world[:3, :3] @ opening_local
        viewing = look_at - position
        assert abs(np.dot(opening_world, viewing)) / (np.linalg.norm(opening_world) * np.linalg.norm(viewing)) < .4


def test_browser_camera_update_is_saved_as_actual_view():
    app = SceneGraspTunerApp.__new__(SceneGraspTunerApp)
    app.camera = {"position": [0, 0, 1], "look_at": [0, 0, 0], "up": [0, 0, 1], "fov": .8}
    saved = []
    app._write_state = lambda: saved.append(dict(app.camera))
    camera = SimpleNamespace(position=np.array([.4, .2, .3]), look_at=np.array([.3, .2, 0.]),
                             up_direction=np.array([0., 0., 1.]), fov=.65)
    app._on_browser_camera_update(camera)
    assert saved[-1]["position"] == [.4, .2, .3]
    assert saved[-1]["fov"] == .65


def test_gripper_snapshot_material_check_accepts_side_view_without_fixed_region(tmp_path):
    pixels = np.full((500, 1000, 3), 232, dtype=np.uint8)
    pixels[100:150, 50:100] = (60, 170, 180)
    pixels[200:250, 100:150] = (220, 170, 55)
    image = tmp_path / "side.png"
    Image.fromarray(pixels).save(image)
    assert _scene_material_ready(image, require_gripper=True, view="gripper")


def test_ring_visual_respects_empty_partial_and_full_arc():
    assert annular_sector(.04, .05, .01, 0.) is None
    partial = annular_sector(.04, .05, .01, np.pi / 2)
    assert partial is not None
    assert np.all(partial.vertices[:, 0] >= -1e-12)
    assert np.all(partial.vertices[:, 1] >= -1e-12)
    full = annular_sector(.04, .05, .01, 2*np.pi)
    assert full is not None and full.vertices[:, 0].min() < 0


@pytest.mark.parametrize("kind", ("ring", "cylinder_axial", "cylinder", "cuboidal_block", "trifold_handle"))
def test_each_lab_kind_has_usable_own_geometry_defaults(kind):
    app = KnowledgeGraspLabApp.__new__(KnowledgeGraspLabApp)
    app.kind = SimpleNamespace(value=kind)
    app.dimensions = {name: SimpleNamespace(value=value) for name, value in GEOMETRY_DEFAULTS[kind].items()}
    concept = app._concept()
    profile = parse_grasp_profiles({"profiles": {"lab": {
        "generator": kind, "part": "lab_object", "parameters": DEFAULT_PARAMETERS[kind],
        "closing_width": {"mode": "geometry", "clearance_m": .001},
    }}})["lab"]
    result = generate_grasps(concept, np.eye(4), profile)
    assert len(result.proposals) == 1
    assert not result.diagnostics
    if kind == "cylinder_axial":
        assert concept.height_m == pytest.approx(.01)
        np.testing.assert_allclose(concept.radius_xy_m, [.0275, .0275])
    if kind == "cuboidal_block":
        np.testing.assert_allclose(concept.size_xyz_m, [.03, .03, .03])


def test_tuner_rejects_excessive_cartesian_product_before_generation():
    payload = json.loads((ROOT / "data/task_1/knowledge/grasp_profiles.json").read_text())
    parameters = payload["profiles"]["task_1_bowl_0"]["parameters"]
    parameters["azimuth_deg"] = {"values": list(range(201))}
    parameters["tip_y_deg"] = {"values": list(range(101))}
    with pytest.raises(ValueError, match="超过 Viser 调参上限"):
        build_session_state(ROOT, "task_1_bowl_0", profile_document=payload)


class _Control:
    def __init__(self, value):
        self.value = value
        self.visible = True
        self._callback = None

    def on_update(self, callback):
        self._callback = callback

    def remove(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _Gui:
    def add_folder(self, name, **kwargs):
        return _Control(name)

    def add_dropdown(self, name, *, initial_value, **kwargs):
        return _Control(initial_value)

    def add_slider(self, name, *, initial_value, **kwargs):
        return _Control(initial_value)

    def add_text(self, name, *, initial_value, **kwargs):
        return _Control(initial_value)


def test_range_order_survives_slider_read_and_list_conversion():
    controls = GraspParameterControls(_Gui(), {"azimuth_deg": {"start": 10, "stop": 40, "step": 10, "order": "center_out"}}, lambda: None)
    row = controls.rows["azimuth_deg"]
    assert controls.read()["azimuth_deg"]["order"] == "center_out"
    row["mode"].value = "列表"
    controls._changed("azimuth_deg")
    assert controls.read()["azimuth_deg"] == {"values": [20., 30., 10., 40.]}


def test_irregular_list_cannot_silently_become_a_range():
    errors = []
    controls = GraspParameterControls(_Gui(), {"azimuth_deg": {"values": [0, 45, 120]}}, lambda: None, errors.append)
    row = controls.rows["azimuth_deg"]
    row["mode"].value = "范围"
    controls._changed("azimuth_deg")
    assert row["mode"].value == "列表"
    assert controls.read()["azimuth_deg"] == {"values": [0., 45., 120.]}
    assert "等差" in errors[0]


def test_descending_range_keeps_signed_step_and_order():
    controls = GraspParameterControls(_Gui(), {"azimuth_deg": {"start": 180, "stop": 0, "step": -45, "order": "center_out"}}, lambda: None)
    assert controls.read()["azimuth_deg"] == {"start": 180., "stop": 0., "step": -45., "order": "center_out"}
    row = controls.rows["azimuth_deg"]
    row["mode"].value = "列表"
    controls._changed("azimuth_deg")
    assert controls.read()["azimuth_deg"] == {"values": [0., 45., 90., 135., 180.]}


def test_candidate_selection_follows_geometry_when_indices_change():
    payload = json.loads((ROOT / "data/task_1/knowledge/grasp_profiles.json").read_text())
    before = build_session_state(ROOT, "task_1_bowl_0", profile_document=payload)
    original = before.candidates[-1]
    profile = payload["profiles"]["task_1_bowl_0"]
    profile["parameters"]["azimuth_deg"] = {"values": [0, 45, 90]}
    after = build_session_state(ROOT, "task_1_bowl_0", profile_document=payload)
    selected = closest_candidate_index(after.candidates, original)
    assert 0 <= selected < len(after.candidates)
    assert selected == min(range(len(after.candidates)), key=lambda i: (
        np.linalg.norm(after.candidates[i].tcp_pose_world[:3, 3] - original.tcp_pose_world[:3, 3])
        + .02 * np.linalg.norm(after.candidates[i].tcp_pose_world[:3, :3] - original.tcp_pose_world[:3, :3]), i,
    ))


def test_representative_grippers_are_few_spread_and_deterministic():
    proposals = build_session_state(ROOT, "task_1_bowl_0").candidates
    first = representative_candidate_indices(proposals, selected=0, count=3)
    assert first == representative_candidate_indices(proposals, selected=0, count=3)
    assert len(first) == len(set(first)) == 3
    assert 0 not in first
    assert representative_candidate_indices(proposals, selected=0, count=0) == ()
    positions = [proposals[index].tcp_pose_world[:3, 3] for index in first]
    assert max(np.linalg.norm(left - right) for left in positions for right in positions) > .05


def test_representative_gripper_count_can_exceed_ten_and_clamps_to_candidates():
    proposals = build_session_state(ROOT, "task_1_bowl_0").candidates
    assert len(proposals) > 12
    indices = representative_candidate_indices(proposals, selected=0, count=12)
    assert len(indices) == len(set(indices)) == 12
    assert 0 not in indices
    all_others = representative_candidate_indices(proposals, selected=0, count=len(proposals) + 10)
    assert len(all_others) == len(proposals) - 1


def test_advanced_postprocess_synchronizes_simple_sliders():
    app = SceneGraspTunerApp.__new__(SceneGraspTunerApp)
    app._syncing = False
    app.shift_frame = _Control("object")
    app.shift_xyz = [_Control(0.) for _ in range(3)]
    app._sync_postprocess_controls([{"type": "rotate", "frame": "tcp", "rpy_deg": [0, 0, 10]}, {"type": "translate", "frame": "tcp", "xyz_m": [.01, -.015, .02]}])
    assert app.shift_frame.value == "tcp"
    assert [item.value for item in app.shift_xyz] == [.01, -.015, .02]
    assert app._syncing is False


def test_feedback_reader_accepts_manifest_and_rejects_stale_invalid_input(tmp_path):
    path = tmp_path / "capture.json"
    state = {"schema_version": 1, "tool": "lab", "input_valid": True, "parameters": {"azimuth_deg": {"values": [90]}}}
    path.write_text(json.dumps({"state": state}), encoding="utf-8")
    assert read_tuner_state(path, tool="lab") == state
    with pytest.raises(ValueError, match="不是 scene"):
        read_tuner_state(path, tool="scene")
    state["input_valid"] = False
    path.write_text(json.dumps({"state": state}), encoding="utf-8")
    with pytest.raises(ValueError, match="输入无效"):
        read_tuner_state(path, tool="lab")
