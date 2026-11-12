"""在线 Bowl Viewer 必须保持规划轨迹、附着事件和场景版本事实。"""
import json
import numpy as np
import pytest


def report_fixture():
    def move(name, a, b, operation):
        return {'name': name, 'status': 'planned', 'target': {'frame': 'world_tcp', 'matrix': np.eye(4).tolist()},
                'details': {}, 'result': {'success': True, 'duration_s': .3,
                'trajectory': {'joint_names': ['j'], 'times_s': [0., .1], 'positions_rad': [[a], [b]]}}}
    return {'scene_revision': 2, 'instance_id': 'bowl', 'success': True,
            'visual_attachment': {'collision_snapshot_ref': 'attached.json'},
            'stage_contract': [{'name': 'open', 'operation': 'gripper_command'},
                {'name': 'pregrasp', 'operation': 'plan'}, {'name': 'approach', 'operation': 'linear_tcp'},
                {'name': 'close', 'operation': 'gripper_command'}, {'name': 'lift', 'operation': 'linear_tcp'}],
            'stages': [{'name': 'open', 'status': 'planned', 'details': {'width_m': .08}},
                move('pregrasp', 0., 1., 'plan'), move('approach', 1., 2., 'linear_tcp'),
                {'name': 'close', 'status': 'planned', 'details': {'width_m': .007}},
                move('lift', 2., 3., 'linear_tcp')]}


def snapshots(root):
    (root / 'collision').mkdir()
    (root / 'collision/collision_snapshot.json').write_text(json.dumps({'records': [{'obstacle_name': 'bowl'}]}))
    (root / 'attached.json').write_text(json.dumps({'records': [], 'transferred_objects': {'bowl': []}}))


def test_pick_attachment_follows_real_tcp_after_close_and_seek_restores(tmp_path):
    from gene_agent.visualization.microwave.bowl_session import object_visual_run
    from gene_agent.visualization.viser_sim.playback import VisualPlayback, VisualState
    snapshots(tmp_path)
    run = object_visual_run(report_fixture(), 'pick_bowl', tmp_path)
    assert run.success
    np.testing.assert_allclose(run.stages[-1].trajectory.positions_rad[:, 0], [2., 3.])
    assert run.stages[-1].attach_before == ('bowl',)
    assert run.stages[-1].collision_snapshot_ref == str(tmp_path / 'attached.json')
    def fk(names, q):
        pose = np.eye(4); pose[0, 3] = q[0]; return pose
    initial = VisualState(('j',), np.array([0.]), .08, {'bowl': np.eye(4)})
    playback = VisualPlayback(run, initial, fk)
    final = playback.sample(playback.duration_s).state
    assert final.attachments == ('bowl',)
    assert final.object_poses_world['bowl'][0, 3] == pytest.approx(1.)
    assert playback.sample(0).state.attachments == ()
    assert playback.sample(0).state.object_poses_world['bowl'][0, 3] == 0.


def test_failure_prefix_cannot_claim_attachment_or_invent_following_motion(tmp_path):
    from gene_agent.visualization.microwave.bowl_session import object_visual_run
    snapshots(tmp_path)
    report = report_fixture()
    report['success'] = False
    report['stages'][2].update(status='failed', result={'success': False, 'failure': {'category': 'goal_ik_no_solution'}})
    run = object_visual_run(report, 'pick_bowl', tmp_path)
    assert not run.success
    assert len(run.stages) == 3
    assert run.stages[-1].trajectory is None
    assert run.stages[-1].failure['category'] == 'goal_ik_no_solution'
    assert run.stages[-1].attach_after == ()


def test_insert_detach_preserves_actual_released_pose(tmp_path):
    from gene_agent.visualization.microwave.bowl_session import object_visual_run
    snapshots(tmp_path)
    released = np.eye(4); released[0, 3] = .42
    report = {'scene_revision': 2, 'success': True, 'stage_contract': [
        {'name': 'release', 'operation': 'gripper_command'}, {'name': 'detach', 'operation': 'attachment'}],
        'stages': [{'name': 'release', 'status': 'planned', 'details': {'width_m': .08}},
                   {'name': 'detach', 'status': 'planned', 'details': {}, 'object_pose_world': released.tolist()}]}
    run = object_visual_run(report, 'insert_bowl', tmp_path, initial_attachments=('bowl',))
    assert run.stages[0].attach_before == ('bowl',)
    assert run.stages[0].attach_after == ('bowl',)
    assert run.stages[1].attach_after == ()
    np.testing.assert_allclose(run.stages[1].object_pose_world, released)


def test_scene_edit_rebuilds_gt_candidates_collision_and_invalidates_plan(tmp_path):
    from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession
    from gene_agent.visualization.viser_sim.model import SceneEdits
    session = ObjectVisualSession(output_dir=tmp_path / 'online')
    old_pose = session.case.bowl_pose.copy()
    old_target = np.array(session.case.candidates[0]['T_world_tcp'])
    old_hash = session.case.snapshot['snapshot_sha256']
    changed = old_pose.copy(); changed[0, 3] += .015
    session.apply_edits(SceneEdits(object_poses_world={'bowl': changed}))
    assert session.case.revision == 1
    assert session.run is None and session.playback is None
    np.testing.assert_allclose(session.case.bowl_pose, changed, atol=1e-7)
    np.testing.assert_allclose(np.array(session.case.candidates[0]['T_world_tcp'])[:3, 3], old_target[:3, 3] + [.015, 0, 0])
    assert session.case.snapshot['snapshot_sha256'] != old_hash
    assert (session.case.output_dir / 'collision/collision_snapshot.json').is_file()


def test_online_app_edit_invalidates_old_timeline_and_retains_new_gt(tmp_path):
    import viser
    from gene_agent.visualization.viser_sim.object_app import ObjectVisualApp
    from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession
    server = viser.ViserServer(host='127.0.0.1', port=0)
    try:
        session = ObjectVisualSession(output_dir=tmp_path / 'ui')
        app = ObjectVisualApp(server, session=session)
        assert app.play_button.disabled
        from gene_agent.visualization.viser_sim.model import VisualStage
        app.scene.set_stage(VisualStage('old_scene', 'pick_bowl', 'attachment', True,
            collision_snapshot_ref=str(session.case.output_dir / 'collision/collision_snapshot.json')))
        app.scene.set_collision_visible(True)
        assert app.scene.renderer.collision_handles
        before = session.case.bowl_pose.copy()
        app.pose_fields['bowl'][0].value = float(before[0, 3] + .02)
        app._apply_edits()
        assert session.case.revision == 1
        assert session.case.bowl_pose[0, 3] == pytest.approx(before[0, 3] + .02)
        assert app.playback is None and app.play_button.disabled
        assert app.controller is session
        # 新 revision 尚未规划，不允许遗留上一版碰撞节点。
        assert not any(name.startswith('/pick_bowl/collision/')
                       for name in server.scene._handle_from_node_name)
    finally:
        server.stop()


def test_visual_mesh_exclusions_are_not_reported_as_allowed_contact(tmp_path):
    from gene_agent.visualization.microwave.bowl_session import object_visual_run
    snapshots(tmp_path)
    (tmp_path / 'collision/collision_snapshot.json').write_text(json.dumps({
        'records': [{'obstacle_name': 'microwave_proxy'}],
        'excluded': [{'body_name': 'microwave', 'reason': 'contact_masks_both_zero'},
                     {'body_name': 'right_link_1', 'reason': 'active_group_uses_original_robot_collision_spheres'}]}))
    run = object_visual_run(report_fixture(), 'pick_bowl', tmp_path)
    assert run.stages[0].collision_allowed == ()


def test_table_pose_size_and_drag_commit_share_rebuilt_collision(tmp_path):
    import viser
    from scipy.spatial.transform import Rotation
    from gene_agent.visualization.viser_sim.object_app import ObjectVisualApp
    from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession
    server = viser.ViserServer(host='127.0.0.1', port=0)
    try:
        session = ObjectVisualSession(output_dir=tmp_path / 'table-ui')
        app = ObjectVisualApp(server, session=session)
        original_bowl = session.case.bowl_pose.copy()
        app.drag_controls['bowl'].position = original_bowl[:3, 3] + [.01, -.01, 0.]
        app._pose_from_drag('bowl')
        app.pose_fields['table'][0].value += .02
        app.pose_fields['table'][5].value = 15.
        app.table_sizes[0].value = 1.2
        app.table_sizes[1].value = .7
        app._apply_edits()
        np.testing.assert_allclose(session.case.bowl_pose[:3, 3], original_bowl[:3, 3]+[.01,-.01,0.])
        geom = session.case.model.geom('scene_table')
        np.testing.assert_allclose(2*geom.size[:2], [1.2,.7])
        np.testing.assert_allclose(session.case.data.geom_xmat[geom.id].reshape(3,3),
                                   Rotation.from_euler('z',15,degrees=True).as_matrix(), atol=1e-7)
        record = next(r for r in session.case.snapshot['records'] if r['geom_name']=='scene_table')
        np.testing.assert_allclose(np.array(record['T_world_geom'])[:3,3],session.case.data.geom_xpos[geom.id])
    finally:
        server.stop()


def test_sync_plan_is_pending_until_tick_and_can_be_stopped_before_gpu(tmp_path):
    import viser
    from gene_agent.visualization.viser_sim.object_app import ObjectVisualApp
    from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession
    server = viser.ViserServer(host='127.0.0.1',port=0)
    try:
        session = ObjectVisualSession(output_dir=tmp_path/'sync-ui')
        app = ObjectVisualApp(server,session=session)
        app._start_plan()
        assert app.planning
        assert not (session.case.output_dir/'skill_pick_bowl_result.json').exists()
        app._stop()
        assert not app.planning
        assert session.run is None
    finally:
        server.stop()


def test_pick_offset_agrees_with_attachment_evidence(tmp_path):
    from gene_agent.visualization.microwave.bowl_session import object_visual_run
    from gene_agent.visualization.viser_sim.playback import VisualPlayback, VisualState
    snapshots(tmp_path)
    report = report_fixture()
    offset = np.eye(4); offset[0,3] = -2.
    report['visual_attachment']['T_tcp_bowl'] = offset.tolist()
    run = object_visual_run(report,'pick_bowl',tmp_path)
    def fk(names,q):
        pose=np.eye(4);pose[0,3]=q[0];return pose
    playback=VisualPlayback(run,VisualState(('j',),np.array([0.]),.08,{'bowl':np.eye(4)}),fk)
    for t in (playback.stage_start_s('pick_bowl/lift'),playback.duration_s):
        state=playback.sample(t).state
        np.testing.assert_allclose(np.linalg.inv(fk(state.joint_names,state.joint_positions_rad)) @ state.object_poses_world['bowl'],offset)


def test_replanning_discards_prior_timeline_before_cancellation(tmp_path):
    import viser
    from gene_agent.visualization.viser_sim.object_app import ObjectVisualApp
    from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession
    from gene_agent.visualization.viser_sim.model import VisualRun, VisualStage
    from gene_agent.visualization.viser_sim.playback import VisualPlayback
    server = viser.ViserServer(host='127.0.0.1',port=0)
    try:
        session=ObjectVisualSession(output_dir=tmp_path/'repeat-ui')
        app=ObjectVisualApp(server,session=session)
        run=VisualRun('pick_bowl',0,(VisualStage('pick_bowl/open','pick_bowl','gripper',True,gripper_width_m=.08),))
        session.run=run
        app.playback=VisualPlayback(run,session.initial,session.tcp_pose)
        app._start_plan()
        assert app.playback is None
        app._stop()
        assert not app.planning and app.play_button.disabled
    finally:server.stop()


def test_knowledge_view_does_not_keep_previous_planning_collision_snapshot(tmp_path):
    import viser
    from gene_agent.visualization.viser_sim.object_app import ObjectVisualApp
    from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession
    from gene_agent.visualization.viser_sim.model import VisualStage
    server=viser.ViserServer(host='127.0.0.1',port=0)
    try:
        session=ObjectVisualSession(output_dir=tmp_path/'knowledge-ui')
        app=ObjectVisualApp(server,session=session)
        app.scene.set_stage(VisualStage('pick_bowl/approach','pick_bowl','attachment',True,
            collision_snapshot_ref=str(session.case.output_dir/'collision/collision_snapshot.json')))
        app.scene.set_collision_visible(True)
        app._show_knowledge()
        assert not app.scene.renderer.collision_handles
        assert app.scene._shown_snapshot_ref is None
    finally:server.stop()


def test_bowl_task_previews_reproducible_sample_without_changing_planning_candidates(tmp_path):
    import viser
    from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession
    from gene_agent.visualization.viser_sim.object_task_app import ObjectTaskApp

    server = viser.ViserServer(host='127.0.0.1', port=0)
    try:
        session = ObjectVisualSession(output_dir=tmp_path / 'candidate-preview')
        original_ids = [item['candidate_id'] for item in session.case.candidates]
        app = ObjectTaskApp(server, session=session)
        assert not app.candidate.visible
        assert len(app._sample_indices) == 5
        assert len(set(app._sample_indices)) == 5
        first = tuple(app._sample_indices)
        app._next_candidate_sample()
        assert tuple(app._sample_indices) != first
        assert [item['candidate_id'] for item in session.case.candidates] == original_ids
    finally:
        server.stop()


def test_object_task_previews_completed_partition_before_planning_next(tmp_path):
    import viser
    from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession
    from gene_agent.visualization.viser_sim.object_task_app import ObjectTaskApp
    from gene_agent.visualization.viser_sim.model import VisualRun, VisualStage
    from gene_agent.visualization.viser_sim.playback import VisualPlayback

    server = viser.ViserServer(host='127.0.0.1', port=0)
    try:
        session = ObjectVisualSession(output_dir=tmp_path / 'progressive')
        app = ObjectTaskApp(server, session=session)
        stage = VisualStage('pick_bowl/open', 'pick_bowl', 'gripper', True,
                            gripper_width_m=.08, state_duration_s=.2)
        partial = VisualRun('microwave_object_insert', session.case.revision, (stage,))
        playback = VisualPlayback(partial, session.initial, session.tcp_pose)
        calls = []
        def step():
            calls.append('step')
            if len(calls) == 1:
                return {'done': False, 'message': '抓碗已规划',
                        'partial_run': partial, 'partial_playback': playback}
            return {'done': True, 'run': partial}
        session.plan_step = step
        app._start_plan()
        app.tick()
        assert calls == ['step']
        assert app.planning and app._previewing_prefix
        assert app.playback is playback
        app._last_tick -= 1.
        app.tick()
        assert calls == ['step']
        assert not app._previewing_prefix
        app.tick()
        assert calls == ['step', 'step']
        assert not app.planning
    finally:
        server.stop()
