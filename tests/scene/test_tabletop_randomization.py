"""桌面初态采样应可复现、无初始重叠，并保留拒绝原因。"""

import numpy as np
import pytest

from gene_agent.scene.tabletop_randomization import sample_tabletop_layout


def test_sample_tabletop_layout_is_seeded_and_keeps_clearance():
    nominal = {"brush": (.42, -.22, 0.), "cube": (.42, -.02, 0.),
               "cube_red": (.49, -.04, .3), "dustpan": (.45, .22, -1.57)}
    ranges = {"brush": (.01, .01, .08), "cube": (.02, .02, .5),
              "cube_red": (.02, .02, .5), "dustpan": (0., 0., 0.)}
    footprint = {"brush": (.10, .04), "cube": (.022, .022),
                 "cube_red": (.022, .022), "dustpan": (.16, .15)}
    one = sample_tabletop_layout(nominal, ranges, footprint,
                                 table_bounds_xy=(-.1, .86, -.5, .5), seed=7)
    two = sample_tabletop_layout(nominal, ranges, footprint,
                                 table_bounds_xy=(-.1, .86, -.5, .5), seed=7)
    assert one == two
    assert one["seed"] == 7
    assert one["attempts"] >= 1
    assert one["valid"]
    assert one["poses_xyyaw_rad"] != {k: list(v) for k, v in nominal.items()}
    for name, pose in one["poses_xyyaw_rad"].items():
        assert np.all(np.abs(np.asarray(pose) - nominal[name]) <=
                      np.asarray(ranges[name]) + 1e-12)


def test_sample_tabletop_layout_reports_impossible_distribution():
    with pytest.raises(ValueError, match="未找到有效初态"):
        sample_tabletop_layout(
            {"a": (0., 0., 0.), "b": (0., 0., 0.)},
            {"a": (0., 0., 0.), "b": (0., 0., 0.)},
            {"a": (.1, .1), "b": (.1, .1)},
            table_bounds_xy=(-1., 1., -1., 1.), seed=1, max_attempts=3)


def test_placement_excluded_region_uses_object_footprint():
    from gene_agent.scene.placement import sample_placement
    rules={"schema_version":1,"objects":{"bowl":{"mode":"fixed","frame":"world"}},
           "constraints":{"excluded_bounds_xy_m":[{"object":"bowl","label":"base","bounds_xy_m":[.1,.2,.1,.2]}]}}
    pose=np.eye(4);pose[:2,3]=[.05,.15]
    result=sample_placement({"bowl":pose},rules,seed=1,randomize=False,footprints={"bowl":[.12,.12]})
    assert not result["valid"]
    assert result["invalid_reasons"][0]["reason"]=="excluded_region"
