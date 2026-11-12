"""场景高度调整必须同时作用于转盘 visual/collision，而不改变操作目标。"""

import xml.etree.ElementTree as ET

import pytest

from gene_agent.scene.objects import build_objects


def _positions(path):
    root = ET.parse(path).getroot()
    return {geom.get("name"): tuple(float(x) for x in geom.get("pos", "0 0 0").split())
            for geom in root.findall(".//geom") if geom.get("name")}


def test_turntable_offset_moves_visual_and_collision_together(tmp_path):
    baseline = build_objects(tmp_path / "base")
    adjusted = build_objects(tmp_path / "adjusted", turntable_height_offset_m=-.006)
    before = _positions(baseline["xml_path"])
    after = _positions(adjusted["xml_path"])
    moved = {name for name in before if "turntable" in name}
    assert "microwave_turntable_collision" in moved
    assert len(moved) >= 2
    for name in moved:
        assert after[name][:2] == pytest.approx(before[name][:2])
        assert after[name][2] == pytest.approx(before[name][2] - .006)
    assert after["scene_table"] == pytest.approx(before["scene_table"])
    assert adjusted["turntable_height_offset_m"] == -.006


def test_turntable_offset_rejects_nonfinite_or_large_value(tmp_path):
    for value in (float("nan"), -.05):
        with pytest.raises(ValueError, match="转盘高度"):
            build_objects(tmp_path / "bad", turntable_height_offset_m=value)


def test_floor_offset_moves_visual_and_collision_without_moving_turntable(tmp_path):
    baseline = build_objects(tmp_path / "base")
    adjusted = build_objects(tmp_path / "adjusted", microwave_floor_height_offset_m=-.005)
    before = _positions(baseline["xml_path"])
    after = _positions(adjusted["xml_path"])
    for name in ("microwave_cabinet_cavity_floor_visual",
                 "microwave_floor_collision"):
        assert after[name][:2] == pytest.approx(before[name][:2])
        assert after[name][2] == pytest.approx(before[name][2] - .005)
    assert after["microwave_turntable_collision"] == pytest.approx(
        before["microwave_turntable_collision"])
    assert adjusted["microwave_floor_height_offset_m"] == -.005
