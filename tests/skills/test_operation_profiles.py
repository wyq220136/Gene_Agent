import numpy as np
import pytest


def test_environment_override_has_explicit_provenance():
    from gene_agent.skills.library.pick.brush import load_brush_pick_profile
    sim = load_brush_pick_profile('simulation')
    real = load_brush_pick_profile('realworld', overrides={'grasp_tilt_deg': -10.})
    assert sim.values['grasp_tilt_deg'] == -20.
    assert real.values['grasp_tilt_deg'] == -10.
    assert real.sources['grasp_tilt_deg'] == 'explicit_override'
    assert sim.values['lift_policy'] == 'world_z'
    assert real.values['lift_policy'] == 'pregrasp_direction'
    assert sim.sha256 != real.sha256


@pytest.mark.parametrize('overrides', [{'typo': 1}, {'open_width_m': float('nan')},
    {'approach_distance_m': -.1}, {'lift_policy': 'mystery'}])
def test_bad_profile_fails_before_planning(overrides):
    from gene_agent.skills.library.pick.brush import load_brush_pick_profile
    with pytest.raises(ValueError): load_brush_pick_profile('simulation', overrides=overrides)


def test_identical_effective_geometry_gives_identical_targets():
    from gene_agent.skills.library.pick.brush import load_brush_pick_profile
    from gene_agent.skills.library.pick.brush import brush_pick_targets
    concept = 'data/realworld/brush_cube_dustpan_0/concepts/brush.json'
    a = brush_pick_targets(np.eye(4), concept, load_brush_pick_profile('simulation'))
    b = brush_pick_targets(np.eye(4), concept, load_brush_pick_profile('realworld'))
    for name in a: np.testing.assert_allclose(a[name], b[name], atol=1e-9)


def test_simulation_rejects_unsupported_disabled_hold_check():
    from gene_agent.skills.library.pick.brush import load_brush_pick_profile
    with pytest.raises(ValueError, match='hold'):
        load_brush_pick_profile('simulation', overrides={'verify_hold_for_pick': False})
