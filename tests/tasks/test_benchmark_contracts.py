"""题目判据独立于 policy，并保留迁移前的边界行为。"""
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.catalog import load_definition
from benchmarks.cup_pour_task.evaluation import pour_postcondition, ball_membership, placement_postcondition
from benchmarks.microwave_task.evaluation import evaluate_final, table_place_postcondition
from benchmarks.spoon_scoop_task.evaluation import balls_in_spoon, pour_postcondition as scoop_pour


@pytest.mark.parametrize("name", ["microwave_task", "sweep_cube_task", "sweep_long_task",
                                 "cup_pour_task", "spoon_scoop_task"])
def test_task_definitions_identify_environment_without_loading_policy(name):
    definition = load_definition(name)
    assert definition["benchmark_id"] == name
    assert definition["version"] == 1
    assert definition["scene"] and definition["evaluation"]
    assert "policy" not in definition


def test_definition_rejects_paths_outside_catalog():
    with pytest.raises(ValueError):
        load_definition("../microwave_task")


@pytest.mark.parametrize("transferred,spilled,expected", [(4, 0, False), (5, 0, True),
                                                        (5, 5, True), (5, 6, False),
                                                        (8, 6, False)])
def test_cup_goal_preserves_minimum_transfer_and_spill_boundaries(transferred, spilled, expected):
    assert pour_postcondition(set(range(transferred)), set(range(spilled))) is expected


def test_cup_membership_uses_actual_container_frame_and_source_precedence():
    cup = SimpleNamespace(xpos=np.array([1., 0., 0.]), xmat=np.eye(3).ravel())
    target = SimpleNamespace(xpos=np.array([2., 0., 0.]), xmat=np.eye(3).ravel())
    bodies = {"cup": cup, "target": target,
              "a": SimpleNamespace(xpos=np.array([1., 0., .03])),
              "b": SimpleNamespace(xpos=np.array([2., 0., .03])),
              "c": SimpleNamespace(xpos=np.array([0., 0., .03]))}
    row = ball_membership(SimpleNamespace(body=bodies.__getitem__), "cup", "target", ("a", "b", "c"))
    assert row == {"in_cup": {"a"}, "in_target": {"b"}, "outside": {"c"}}


def test_spoon_requires_nonempty_cargo_and_all_original_cargo_transferred():
    assert not scoop_pour(True, set(), set())
    assert not scoop_pour(False, {"a"}, {"a"})
    assert not scoop_pour(True, {"a", "b"}, {"a"})
    assert scoop_pour(True, {"a", "b"}, {"a", "b"})
    assert balls_in_spoon({"a": [.029, .016, .006], "b": [.028, 0., .02]}) == {"a"}


def test_microwave_preserves_angular_checks_and_attachment_requirement():
    row = evaluate_final(0., np.pi / 4, np.pi / 4, True, [])
    assert all(row[k] for k in ("door_closed", "knob_at_target", "bowl_on_table", "no_simulation_attachment"))
    assert not evaluate_final(np.deg2rad(6), np.pi / 4, np.pi / 4, True, ["weld"])["door_closed"]
    assert not evaluate_final(0., np.pi / 4, np.pi / 4, True, ["weld"])["no_simulation_attachment"]
    assert not evaluate_final(0., np.nan, np.pi / 4, True, [])["knob_at_target"]


def test_placement_needs_real_support_and_released_attachment():
    assert placement_postcondition(.02, np.deg2rad(10), .015, [])
    assert not placement_postcondition(.0201, 0., .005, [])
    assert not placement_postcondition(0., 0., .005, ["weld"])
    checks = {"settled_table_support": {"supported": True},
              "final_table_support": {"supported": False},
              "settle_translation_m": 0., "retreat_translation_m": 0.,
              "maximum_allowed_translation_m": .005, "final_bowl_tilt_deg": 0.,
              "maximum_allowed_tilt_deg": 15., "simulation_attachment_released": True}
    assert not table_place_postcondition(checks)
    checks["final_table_support"]["supported"] = True
    assert table_place_postcondition(checks)
