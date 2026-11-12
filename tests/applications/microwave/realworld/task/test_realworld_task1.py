"""完整真机 Task 只编排独立子任务，并使用实测结果交接。"""

import json

from gene_agent.realworld.microwave import task as module


_ORDER = (
    "open_door_first", "after_open_transition", "pick_bowl", "insert_bowl",
    "before_close_first_transition", "close_door_first", "rotate_knob",
    "before_reopen_transition", "open_door_second", "after_reopen_transition", "extract_bowl",
    "place_bowl_on_table", "before_close_second_transition", "close_door_second",
)


def _options(tmp_path, execute=True):
    return dict(
        observe_collision_scene=lambda: {"observed": True},
        make_runtime=lambda snapshot: object(), planner_pool=object(),
        output_dir=tmp_path / "task1", execute=execute,
        subtask_options={key: {} for key in
                         ("door", "pick", "insert", "knob", "extract", "table")},
    )


def _passed():
    # 假观测模拟已声明的现场视觉检查；测试本身不提供真机成功证据。
    return {"status": "pass", "source": "vision"}


def _door_completed(part):
    skill_ids = ("handle_pull", "push_open") if part == "open" else ("push_close",)
    return {"status": "executed", "success": True, "skills": [
        {"skill_id": skill_id, "postcondition": _passed()} for skill_id in skill_ids
    ]}


def test_task_rejects_pass_without_check_source():
    """完整任务不能只凭下层报告里的 pass 字符串继续下一段。"""
    door = {"skills": [{"skill_id": "push_close",
                        "postcondition": {"status": "pass"}}]}
    _, door_verified = module._postcondition_evidence("door", door)
    _, pick_verified = module._postcondition_evidence(
        "pick", {"postcondition": {"status": "pass"}},
    )
    assert door_verified is False
    assert pick_verified is False
    _, simulated_verified = module._postcondition_evidence(
        "pick", {"postcondition": {
            "status": "pass", "source": "simulation_privileged_weld",
        }},
    )
    assert simulated_verified is False


def test_full_task_uses_same_subtask_functions_and_passes_actual_reports(
    tmp_path, monkeypatch,
):
    called = []
    pick = {"status": "executed", "pick_success": True, "marker": "actual-pick",
            "postcondition": _passed()}
    extract_grasp = {"status": "executed", "grasp_success": True,
                     "marker": "actual-extract-grasp"}

    def door(**kwargs):
        called.append(("door", kwargs["part"]))
        return _door_completed(kwargs["part"])

    def transition(**kwargs):
        if kwargs["output_dir"].name == "before_reopen_transition":
            assert kwargs["phase_id"] == "task/before_reopen/return_default_tcp"
            assert kwargs["motion_kind"] == "plan"
            assert kwargs["contact_policy"] == "avoid_door"
        called.append(("transition", kwargs["execute"]))
        return {"status": "executed", "success": True,
                "postcondition": {"status": "pass", "source": "observed_robot_and_door"}}

    def picked(**kwargs):
        called.append(("pick", kwargs["execute"]))
        return pick

    def inserted(**kwargs):
        assert kwargs["pick_report"] is pick
        called.append(("insert", kwargs["execute"]))
        return {"status": "executed", "insert_success": True,
                "postcondition": _passed()}

    def knob(**kwargs):
        called.append(("knob", kwargs["execute"]))
        return {"status": "executed", "rotation_success": True,
                "rotation_postcondition": _passed()}

    def extracted(**kwargs):
        called.append(("extract", kwargs["execute"]))
        return {"status": "executed", "extract_success": True,
                "grasp": extract_grasp, "postcondition": _passed()}

    def table(**kwargs):
        assert kwargs["grasp_report"] is extract_grasp
        called.append(("table", kwargs["execute"]))
        return {"status": "executed", "table_place_success": True,
                "postcondition": _passed()}

    for name, fn in (("run_observed_door_subtask", door),
                     ("run_observed_door_return_default_tcp", transition),
                     ("run_observed_bowl_pick", picked),
                     ("run_observed_bowl_insert", inserted),
                     ("run_observed_knob_skill", knob),
                     ("run_observed_bowl_extract", extracted),
                     ("run_observed_table_place", table)):
        monkeypatch.setattr(module, name, fn)
    result = module.run_observed_microwave_task1(**_options(tmp_path))
    assert result["success"] is True
    assert result["status"] == "executed"
    assert tuple(row["step_id"] for row in result["steps"]) == _ORDER
    assert called == [
        ("door", "open"), ("transition", True), ("pick", True),
        ("insert", True), ("transition", True), ("door", "close"),
        ("knob", True), ("transition", True), ("door", "open"),
        ("transition", True), ("extract", True), ("table", True), ("transition", True),
        ("door", "close"),
    ]
    saved = json.loads((tmp_path / "task1/task_result.json").read_text())
    assert saved["success"] is True


def test_no_knob_task_reuses_same_subtasks_and_keeps_reopen_transition(
    tmp_path, monkeypatch,
):
    """跳过旋钮不能跳掉后续开门前的安全过渡，也不能改用另一套子任务。"""
    calls = []

    def door(**kwargs):
        calls.append(("door", kwargs["part"]))
        return _door_completed(kwargs["part"])

    def transition(**kwargs):
        calls.append(("transition", kwargs["output_dir"].name))
        return {"status": "executed", "success": True,
                "postcondition": _passed()}

    def pick(**_kwargs):
        calls.append(("pick", None))
        return {"status": "executed", "pick_success": True,
                "postcondition": _passed()}

    def insert(**kwargs):
        assert kwargs["pick_report"]["pick_success"] is True
        calls.append(("insert", None))
        return {"status": "executed", "insert_success": True,
                "postcondition": _passed()}

    def extract(**_kwargs):
        calls.append(("extract", None))
        return {"status": "executed", "extract_success": True,
                "grasp": {"grasp_success": True}, "postcondition": _passed()}

    def table(**kwargs):
        assert kwargs["grasp_report"]["grasp_success"] is True
        calls.append(("table", None))
        return {"status": "executed", "table_place_success": True,
                "postcondition": _passed()}

    monkeypatch.setattr(module, "run_observed_door_subtask", door)
    monkeypatch.setattr(module, "run_observed_door_return_default_tcp", transition)
    monkeypatch.setattr(module, "run_observed_bowl_pick", pick)
    monkeypatch.setattr(module, "run_observed_bowl_insert", insert)
    monkeypatch.setattr(module, "run_observed_knob_skill",
                        lambda **_kwargs: (_ for _ in ()).throw(
                            AssertionError("跳过旋钮时不能调用旋钮 Skill")))
    monkeypatch.setattr(module, "run_observed_bowl_extract", extract)
    monkeypatch.setattr(module, "run_observed_table_place", table)

    result = module.run_observed_microwave_task1(
        **_options(tmp_path), include_knob=False,
    )

    assert result["success"] is True
    assert result["include_knob"] is False
    assert result["skipped_steps"] == ["rotate_knob"]
    assert tuple(row["step_id"] for row in result["steps"]) == tuple(
        step for step in _ORDER if step != "rotate_knob"
    )
    assert ("transition", "before_reopen_transition") in calls
    assert ("door", "open") in calls[7:]


def test_failed_subtask_stops_without_running_following_steps(tmp_path, monkeypatch):
    called = []

    def door(**kwargs):
        called.append(kwargs["part"])
        return _door_completed(kwargs["part"])

    def pick(**kwargs):
        called.append("pick")
        return {"status": "unknown", "pick_success": False,
                "first_failure": {"code": "pick_not_confirmed", "message": "证据不足"}}

    monkeypatch.setattr(module, "run_observed_door_subtask", door)
    monkeypatch.setattr(module, "run_observed_door_return_default_tcp",
                        lambda **kwargs: {"status": "executed", "success": True,
                                          "postcondition": _passed()})
    monkeypatch.setattr(module, "run_observed_bowl_pick", pick)
    result = module.run_observed_microwave_task1(**_options(tmp_path))
    assert called == ["open", "pick"]
    assert result["status"] == "unknown"
    assert result["first_failure"]["step_id"] == "pick_bowl"
    assert result["first_failure"]["code"] == "pick_not_confirmed"


def test_plan_only_validates_first_subtask_and_does_not_predict_future(
    tmp_path, monkeypatch,
):
    called = []

    def door(**kwargs):
        called.append((kwargs["part"], kwargs["execute"]))
        return {"status": "validated_prefix", "success": False,
                "first_failure": {"code": "not_executed", "message": "仅预检"}}

    monkeypatch.setattr(module, "run_observed_door_subtask", door)
    result = module.run_observed_microwave_task1(**_options(tmp_path, False))
    assert called == [("open", False)]
    assert result["success"] is False
    assert result["status"] == "validated_prefix"
    assert len(result["steps"]) == 1


def test_success_flag_without_postcondition_stops_complete_task(tmp_path, monkeypatch):
    """子任务误报成功但缺少现场后置证据时，不能继续运动。"""
    monkeypatch.setattr(module, "run_observed_door_subtask", lambda **kwargs: {
        "status": "executed", "success": True, "skills": [
            {"skill_id": "handle_pull", "postcondition": {"status": "pass"}},
            {"skill_id": "push_open", "postcondition": {"status": "unknown"}},
        ],
    })
    monkeypatch.setattr(module, "run_observed_door_return_default_tcp",
                        lambda **kwargs: (_ for _ in ()).throw(
                            AssertionError("后置条件未知时不能进入下一段")))

    result = module.run_observed_microwave_task1(**_options(tmp_path))

    assert result["success"] is False
    assert len(result["steps"]) == 1
    assert result["first_failure"]["step_id"] == "open_door_first"
    assert result["first_failure"]["code"] == "postcondition_unverified"


def test_failed_task_transition_stops_before_pick(tmp_path, monkeypatch):
    calls = []

    def door(**kwargs):
        calls.append("open")
        return _door_completed(kwargs["part"])

    def transition(**kwargs):
        calls.append("transition")
        return {"status": "failed", "success": False,
                "first_failure": {"code": "motion_planning_failed",
                                  "message": "无法安全撤离"}}

    monkeypatch.setattr(module, "run_observed_door_subtask", door)
    monkeypatch.setattr(module, "run_observed_door_return_default_tcp", transition)
    monkeypatch.setattr(module, "run_observed_bowl_pick",
                        lambda **kwargs: (_ for _ in ()).throw(AssertionError("不能抓碗")))
    result = module.run_observed_microwave_task1(**_options(tmp_path))
    assert calls == ["open", "transition"]
    assert result["first_failure"]["step_id"] == "after_open_transition"
    assert result["first_failure"]["code"] == "motion_planning_failed"


def test_close_transition_failure_keeps_insert_success_and_stops_door_close(
    tmp_path, monkeypatch,
):
    calls = []

    def door(**kwargs):
        calls.append(("door", kwargs["part"]))
        assert kwargs["part"] == "open"
        return _door_completed(kwargs["part"])

    def transition(**kwargs):
        calls.append(("transition", len([c for c in calls if c[0] == "transition"])))
        if len([c for c in calls if c[0] == "transition"]) == 2:
            return {"status": "failed", "success": False,
                    "first_failure": {"code": "path_not_found", "message": "放碗后撤离失败"}}
        return {"status": "executed", "success": True,
                "postcondition": _passed()}

    monkeypatch.setattr(module, "run_observed_door_subtask", door)
    monkeypatch.setattr(module, "run_observed_door_return_default_tcp", transition)
    monkeypatch.setattr(module, "run_observed_bowl_pick",
                        lambda **kwargs: {"status": "executed", "pick_success": True,
                                          "postcondition": _passed()})
    monkeypatch.setattr(module, "run_observed_bowl_insert",
                        lambda **kwargs: {"status": "executed", "insert_success": True,
                                          "postcondition": _passed()})
    result = module.run_observed_microwave_task1(**_options(tmp_path))
    assert result["first_failure"]["step_id"] == "before_close_first_transition"
    assert result["first_failure"]["code"] == "path_not_found"
    assert calls == [("door", "open"), ("transition", 0), ("transition", 1)]


def test_handle_grasp_force_is_only_passed_to_door_skill_not_task_transition(
    tmp_path, monkeypatch,
):
    """门把手力参数不能误传到仅负责回默认 TCP 的 Task 过渡段。"""
    calls = []

    def door(**kwargs):
        calls.append("door")
        assert kwargs["handle_grasp_force"] == .7
        return _door_completed(kwargs["part"])

    def transition(**kwargs):
        calls.append("transition")
        assert "handle_grasp_force" not in kwargs
        return {"status": "failed", "success": False,
                "first_failure": {"code": "stop_here", "message": "停在过渡段"}}

    monkeypatch.setattr(module, "run_observed_door_subtask", door)
    monkeypatch.setattr(module, "run_observed_door_return_default_tcp", transition)
    args = _options(tmp_path)
    args["subtask_options"]["door"]["handle_grasp_force"] = .7

    result = module.run_observed_microwave_task1(**args)

    assert calls == ["door", "transition"]
    assert result["first_failure"]["code"] == "stop_here"


def test_invalid_subtask_result_preserves_failure_report(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'run_observed_door_subtask', lambda **kwargs: [])
    result = module.run_observed_microwave_task1(**_options(tmp_path))
    assert result['success'] is False
    assert result['first_failure']['code'] == 'subtask_exception'
    assert len(result['steps']) == 1
    assert result['shared_flow']['end_state']['values']['physical_state_known'] is False


def test_task_does_not_mark_unavailable_or_partial_sdk_state_known(tmp_path, monkeypatch):
    for terminal in (
        {'state_id': 'unavailable:settle', 'source': 'unavailable', 'values': {'state_unavailable': True}},
        {'state_id': 'sdk_partial:release', 'source': 'realworld_sdk_partial',
         'values': {'joint_state_unknown': True, 'gripper_width_m': .08}},
    ):
        monkeypatch.setattr(module, 'run_observed_door_subtask', lambda **kwargs: {
            'status': 'failed', 'success': False, 'shared_flow': {'end_state': terminal}})
        result = module.run_observed_microwave_task1(**_options(tmp_path))
        values = result['shared_flow']['end_state']['values']
        assert values['terminal_state'] == terminal
        assert values['physical_state_known'] is False


def test_load_combo_reuses_pick_insert_and_observes_again(tmp_path, monkeypatch):
    calls = []
    frames = []
    pick = {'status': 'executed', 'pick_success': True,
            'postcondition': _passed(), 'actual_marker': 'new-pick'}
    def observe():
        frames.append(len(frames))
        return {'provenance': {'door_angle_rad': 1.57}}
    def picked(**kw):
        calls.append('pick')
        kw['observe_collision_scene']()
        return pick
    def inserted(**kw):
        calls.append('insert')
        kw['observe_collision_scene']()
        assert kw['pick_report'] is pick
        return {'status':'executed', 'insert_success': True, 'postcondition':_passed()}
    monkeypatch.setattr(module, 'run_observed_bowl_pick', picked)
    monkeypatch.setattr(module, 'run_observed_bowl_insert', inserted)
    args = _options(tmp_path)
    args['observe_collision_scene'] = observe
    result = module.run_observed_microwave_subtask(name='load_bowl',
        options=args.pop('subtask_options'), **args)
    assert calls == ['pick','insert']
    assert len(frames) >= 2
    assert result['success'] is True
    assert result['requested_scope'] == 'load_bowl'
    assert result['task_success'] is None
    assert result['operation_success'] is True


def test_load_combo_preflight_never_advances_to_insert(tmp_path, monkeypatch):
    calls = []
    def picked(**kw):
        calls.append('pick')
        return {'status':'validated_prefix', 'pick_success':False,
                'first_failure':{'code':'not_executed'}}
    monkeypatch.setattr(module, 'run_observed_bowl_pick', picked)
    monkeypatch.setattr(module, 'run_observed_bowl_insert', lambda **kw: calls.append('insert'))
    args = _options(tmp_path, execute=False)
    args['observe_collision_scene'] = lambda: {'provenance': {'door_angle_rad': 1.57}}
    result = module.run_observed_microwave_subtask(name='load_bowl',
        options=args.pop('subtask_options'), **args)
    assert calls == ['pick']
    assert result['status'] == 'validated_prefix'
    assert result['task_success'] is None


def test_combo_rejects_closed_door_before_motion(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(module, 'run_observed_bowl_pick', lambda **kw: calls.append('pick'))
    args = _options(tmp_path)
    args['observe_collision_scene'] = lambda: {'provenance': {'door_angle_rad': 0.}}
    result = module.run_observed_microwave_subtask(name='load_bowl',
        options=args.pop('subtask_options'), **args)
    assert not calls
    assert result['success'] is False
    assert result['first_failure']['code'] == 'precondition_failed'


def test_unload_combo_hands_actual_grasp_to_table_and_stops_on_unknown(tmp_path, monkeypatch):
    calls = []
    grasp = {'status': 'executed', 'grasp_success': True}
    def extract(**kw):
        calls.append('extract')
        return {'status':'executed', 'extract_success':True, 'grasp':grasp, 'postcondition':_passed()}
    def table(**kw):
        calls.append('table')
        assert kw['grasp_report'] is grasp
        return {'status':'unknown', 'table_place_success':False,
                'first_failure': {'code':'unconfirmed_support','message':'落稳未确认'}}
    monkeypatch.setattr(module, 'run_observed_bowl_extract', extract)
    monkeypatch.setattr(module, 'run_observed_table_place', table)
    args = _options(tmp_path)
    args['observe_collision_scene'] = lambda: {'provenance': {'door_angle_rad': 1.57}}
    result = module.run_observed_microwave_subtask(name='unload_bowl',
        options=args.pop('subtask_options'), **args)
    assert calls == ['extract','table']
    assert not result['success']
    assert result['first_failure']['step_id'] == 'place_bowl_on_table'
    assert result['first_failure']['code'] == 'unconfirmed_support'
