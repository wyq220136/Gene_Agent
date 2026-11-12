from gene_agent.feedback.motion_diagnostics import summarize_motion_result


def test_nested_motion_result_preserves_ik_evidence():
    result = {'success': False, 'failure': {'kind': 'goal_ik_no_solution',
        'summary': 'no IK', 'details': {'ik_diagnostic': {
            'seed_count': 16, 'successful_seed_count': 0,
            'selected_seed_index': 2, 'seeds': [{'index': 2, 'position_error_m': .002, 'rotation_error_rad': .01}],
            'collision_check': {'self_collision': True, 'scene_collision': False}}}},
        'metrics': {'start_state_check': {'valid': True}}}
    summary = summarize_motion_result(result)
    assert summary.start_state_valid is True
    assert summary.self_collision is True
    assert summary.best_position_error_m == .002
    assert '候选' in summary.conclusion
    assert '全局' in ' '.join(summary.unknowns)


def test_legacy_missing_evidence_remains_unknown():
    summary = summarize_motion_result({'success': False, 'failure': {'kind': 'goal_ik_no_solution'}})
    assert summary.self_collision is None
    assert summary.best_position_error_m is None
    assert summary.unknowns
    assert summary.next_actions


def test_scene_probe_does_not_prove_global_unreachability():
    summary = summarize_motion_result({'success': False, 'failure': {'kind': 'goal_ik_no_solution',
        'details': {'ik_diagnostic': {'world_obstruction_probe': {
            'world_collision_disabled_for_probe': True, 'relaxed_ik_success': False}}}}})
    assert summary.world_disabled_still_failed is True
    assert '不可达' not in summary.conclusion
    assert '自碰撞' in summary.to_markdown()


def test_solver_rejection_does_not_assert_goal_ik_was_valid():
    summary = summarize_motion_result({'success': False, 'failure': {'kind': 'solver_rejected'}})
    assert '目标 IK 可用' not in summary.conclusion


def test_legacy_result_key_is_readable(tmp_path):
    import json
    from gene_agent.feedback.native_run import read_planning_failures
    (tmp_path/'events.jsonl').write_text(json.dumps({'kind':'planning','stage_id':'move','result':{'success':False,'failure':{'kind':'path_not_found'}}})+'\n')
    rows=list(read_planning_failures(tmp_path))
    assert rows[0]['summary']['kind']


def test_bundle_finder_ignores_actual_image_sidecar(tmp_path):
    import json
    from gene_agent.feedback.motion_diagnostics import find_diagnostic_bundles
    (tmp_path/'diagnostics').mkdir()
    for name,row in [('a.actual.json',dict(stage_id='align',mode='actual')),('a.json',dict(stage_id='align',schema_version=1,planning={},diagnostic={'status':'partial'}))]:
        (tmp_path/'diagnostics'/name).write_text(json.dumps(row))
    assert [p.name for p in find_diagnostic_bundles(tmp_path,stage='align')]==['a.json']
