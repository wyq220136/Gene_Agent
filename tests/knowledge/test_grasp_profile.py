"""Task-owned grasp profile 严格解析与 resolved 输出测试。"""

import json
from pathlib import Path

import numpy as np
import pytest

from gene_agent.knowledge.grasp import ClosingWidthSpec
from gene_agent.knowledge.grasp.profile import load_grasp_profiles, parse_grasp_profiles


PROFILE_PATH = Path(__file__).parents[2] / "data" / "task_1" / "knowledge/grasp_profiles.json"


def _minimal_profile(**updates):
    profile = {
        "generator": "ring",
        "part": "rim",
        "parameters": {
            "surface": ["bottom"],
            "azimuth_deg": {"values": [0.0]},
            "tip_y_deg": {"values": [0.0]},
            "depth_m": {"values": [0.01]},
        },
        "closing_width": {"mode": "geometry", "clearance_m": 0.0},
    }
    profile.update(updates)
    return {"profiles": {"demo": profile}}


def test_current_task_profiles_resolve_units_order_metadata_and_defaults() -> None:
    profiles = load_grasp_profiles(PROFILE_PATH)

    assert list(profiles) == [
        "task_1_bowl_0",
        "task_1_knob_0",
        "cylinder",
        "cuboidal_block",
        "trifold_handle",
    ]
    assert {profile.generator for profile in profiles.values()} == {
        "ring",
        "cylinder_axial",
        "cylinder",
        "cuboidal_block",
        "trifold_handle",
    }

    bowl = profiles["task_1_bowl_0"]
    assert bowl.part_name == "bowl_rim"
    assert bowl.parameters["surface"] == ("bottom",)
    assert bowl.parameters["azimuth_rad"] == pytest.approx(
        tuple(np.deg2rad(value) for value in range(0, 316, 45))
    )
    assert bowl.metadata == {
        "status": "tuning_required",
        "acceptance_baseline": False,
        "status_reason": (
            "旧 bowl 参数仅作为待调初值；尚未通过 Viser 调参与新基线验收。"
            "旧 world -Z 15 mm 经验偏移不作为默认正确结论。"
        ),
    }
    assert bowl.sorting == ()

    cylinder = profiles["cylinder"]
    assert cylinder.part_name is None
    assert len(cylinder.parameters["cylinder_rotation_rad"]) == 32
    assert cylinder.parameters["cylinder_rotation_rad"][0] == 0.0
    assert cylinder.parameters["cylinder_rotation_rad"][-1] == pytest.approx(
        np.deg2rad(348.75)
    )
    assert cylinder.parameters["tip_x_rad"] == pytest.approx(
        tuple(np.deg2rad(value) for value in [0, -10, 10, -20, 20, -30, 30])
    )
    assert [rule.to_dict() for rule in cylinder.sorting] == [
        {
            "order": "ascending",
            "field": "tip_x_rad",
            "transform": "absolute",
        },
        {
            "order": "descending",
            "metric": "top_down",
            "frame": "world",
        },
    ]

    payload = {name: profile.to_dict() for name, profile in profiles.items()}
    assert payload["cuboidal_block"]["parameters"] == {}
    assert payload["cuboidal_block"]["postprocess"] == []
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


def test_rotate_and_translate_postprocess_require_explicit_frame_and_units() -> None:
    payload = _minimal_profile(
        postprocess=[
            {"type": "translate", "frame": "object", "xyz_m": [0.1, 0.2, 0.3]},
            {"type": "rotate", "frame": "tcp", "rpy_deg": [90.0, 0.0, -45.0]},
        ]
    )

    profile = parse_grasp_profiles(payload)["demo"]

    assert profile.postprocess[0].xyz_m == (0.1, 0.2, 0.3)
    assert profile.postprocess[1].rpy_rad == pytest.approx(
        (np.pi / 2.0, 0.0, -np.pi / 4.0)
    )
    assert profile.to_dict()["postprocess"][1] == {
        "type": "rotate",
        "frame": "tcp",
        "rpy_rad": [np.pi / 2.0, 0.0, -np.pi / 4.0],
    }


def test_explicit_closing_width_override_is_resolved() -> None:
    profile = parse_grasp_profiles(
        _minimal_profile(closing_width={"mode": "override", "width_m": 0.035})
    )["demo"]

    assert profile.closing_width == ClosingWidthSpec(mode="override", width_m=0.035)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({}, r"profiles\.demo\.parameters\.azimuth_deg"),
        (
            {
                "surface": ["bottom"],
                "azimuth_deg": {"values": [0.0]},
                "tip_y_deg": {"values": [0.0]},
                "depth_m": {"values": [0.01]},
                "frobnicate": {"values": [1.0]},
            },
            r"profiles\.demo\.parameters\.frobnicate",
        ),
        (
            {
                "surface": ["bottom"],
                "azimuth_deg": {"values": [0.0]},
                "tip_y_deg": {"values": [0.0]},
                "depth": {"values": [0.01]},
            },
            r"profiles\.demo\.parameters\.depth",
        ),
        (
            {
                "surface": ["bottom"],
                "azimuth_deg": {"values": [0.0]},
                "tip_y_deg": {"values": [0.0]},
                "depth_m": {"values": ["oops"]},
            },
            r"profiles\.demo\.parameters\.depth_m\.values\[0\]",
        ),
        (
            {
                "surface": ["sideways"],
                "azimuth_deg": {"values": [0.0]},
                "tip_y_deg": {"values": [0.0]},
                "depth_m": {"values": [0.01]},
            },
            r"profiles\.demo\.parameters\.surface\[0\]",
        ),
    ],
)
def test_generator_parameter_schema_rejects_missing_unknown_unit_type_and_enum(
    parameters, message
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_grasp_profiles(_minimal_profile(parameters=parameters))


def test_sorting_field_must_reference_a_resolved_parameter() -> None:
    sorting = [
        {"field": "not_declared_deg", "transform": "absolute", "order": "ascending"}
    ]

    with pytest.raises(ValueError, match=r"profiles\.demo\.sorting\[0\]\.field"):
        parse_grasp_profiles(_minimal_profile(sorting=sorting))

    string_sorting = [
        {"field": "surface", "transform": "absolute", "order": "ascending"}
    ]
    with pytest.raises(ValueError, match=r"profiles\.demo\.sorting\[0\]\.field"):
        parse_grasp_profiles(_minimal_profile(sorting=string_sorting))


def test_cuboidal_block_keeps_empty_defaults_and_validates_optional_formula_fields() -> None:
    empty = _minimal_profile(generator="cuboidal_block", parameters={})
    assert parse_grasp_profiles(empty)["demo"].parameters == {}

    configured = _minimal_profile(
        generator="cuboidal_block",
        parameters={
            "face": ["front", "top"],
            "rotate_z_90": {"values": [False, True]},
            "surface_parallel_offset_m": {"values": [0.0, 0.005]},
            "depth_m": {"values": [0.02]},
            "tip_x_deg": {"values": [0.0, 15.0]},
        },
    )
    profile = parse_grasp_profiles(configured)["demo"]

    assert profile.parameters["face"] == ("front", "top")
    assert profile.parameters["rotate_z_90"] == (False, True)
    assert profile.parameters["tip_x_rad"] == pytest.approx((0.0, np.pi / 12.0))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"profiles": {}, "version": 1}, r"version"),
        ({"profiles": []}, r"profiles"),
        (
            _minimal_profile(generator="sphere"),
            r"profiles\.demo\.generator",
        ),
        (
            _minimal_profile(mystery=True),
            r"profiles\.demo\.mystery",
        ),
        (
            _minimal_profile(pregrasp_opening_width_m=0.1),
            r"profiles\.demo\.pregrasp_opening_width_m",
        ),
        (
            _minimal_profile(parameters={"collision_margin_m": {"values": [0.01]}}),
            r"profiles\.demo\.parameters\.collision_margin_m",
        ),
        (
            _minimal_profile(
                closing_width={"mode": "geometry", "clearance_m": 0.0, "extra": 1}
            ),
            r"profiles\.demo\.closing_width\.extra",
        ),
        (
            _minimal_profile(closing_width={"mode": "geometry", "pregrasp_m": 0.1}),
            r"profiles\.demo\.closing_width\.pregrasp_m",
        ),
    ],
)
def test_unknown_or_cross_module_fields_fail_with_json_path(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        parse_grasp_profiles(payload)


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"type": "translate", "xyz_m": [0.0, 0.0, 0.0]}, r"postprocess\[0\]\.frame"),
        (
            {"type": "translate", "frame": "world", "xyz_m": [0.0, 0.0, 0.0]},
            r"postprocess\[0\]\.frame",
        ),
        (
            {"type": "translate", "frame": "tcp", "offset_m": [0.0, 0.0, 0.0]},
            r"postprocess\[0\]\.offset_m",
        ),
        (
            {"type": "rotate", "frame": "tcp", "rpy_rad": [0.0, 0.0, 0.0]},
            r"postprocess\[0\]\.rpy_rad",
        ),
        (
            {"type": "scale", "frame": "object", "xyz_m": [1.0, 1.0, 1.0]},
            r"postprocess\[0\]\.type",
        ),
    ],
)
def test_postprocess_is_strict_and_reports_step_path(step, message) -> None:
    with pytest.raises(ValueError, match=message):
        parse_grasp_profiles(_minimal_profile(postprocess=[step]))


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"profiles":{"demo":{"generator":"ring","generator":"cylinder"}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"generator.*重复"):
        load_grasp_profiles(path)
