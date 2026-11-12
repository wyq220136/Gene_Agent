"""开门和关门子任务只组合已存在的单门 Skill。"""

from gene_agent.realworld.microwave import door as door_task


def _options(tmp_path, part, observe):
    return dict(
        part=part, observe_collision_scene=observe,
        make_runtime=lambda _hash: object(), planner_pool=object(),
        output_dir=tmp_path / "door", config_path="unused.json",
        execute=True, max_microwave_translation_m=.005,
        max_microwave_rotation_rad=.02, max_start_error_rad=.02,
        max_gripper_width_error_m=.005, max_door_angle_error_rad=.05,
    )


def test_open_subtask_reobserves_after_handle_before_push(tmp_path, monkeypatch):
    angles = iter((0., .43))
    received = []

    def skill(**kwargs):
        angle = kwargs["observe_collision_scene"]()
        received.append((kwargs["skill_id"], angle))
        return {"success": True, "status": "executed", "postcondition": {
            "status": "pass", "source": "observed_door_angle",
        }}

    monkeypatch.setattr(door_task, "run_observed_door_skill", skill)
    report = door_task.run_observed_door_subtask(
        **_options(tmp_path, "open", lambda: next(angles))
    )
    assert report["success"] is True
    assert received == [("handle_pull", 0.), ("push_open", .43)]
    assert [row["skill_id"] for row in report["skills"]] == ["handle_pull", "push_open"]
    assert (tmp_path / "door/door_subtask_result.json").is_file()


def test_handle_failure_does_not_start_push(tmp_path, monkeypatch):
    called = []

    def skill(**kwargs):
        called.append(kwargs["skill_id"])
        return {"success": False, "status": "failed", "first_failure": {
            "phase_id": "handle_pull/01_pregrasp", "code": "goal_ik_no_solution",
            "message": "无可达目标",
        }}

    monkeypatch.setattr(door_task, "run_observed_door_skill", skill)
    report = door_task.run_observed_door_subtask(
        **_options(tmp_path, "open", lambda: 0.)
    )
    assert called == ["handle_pull"]
    assert report["success"] is False
    assert report["first_failure"]["code"] == "goal_ik_no_solution"


def test_close_subtask_never_prepares_opening_skills(tmp_path, monkeypatch):
    called = []

    def skill(**kwargs):
        called.append(kwargs["skill_id"])
        return {"success": False, "status": "validated_prefix",
                "first_failure": {"code": "not_executed", "message": "只预检"}}

    monkeypatch.setattr(door_task, "run_observed_door_skill", skill)
    options = _options(tmp_path, "close", lambda: .5)
    options["execute"] = False
    report = door_task.run_observed_door_subtask(**options)
    assert called == ["push_close"]
    assert report["status"] == "validated_prefix"
    assert report["success"] is False


def test_unverified_handle_result_cannot_start_push(tmp_path, monkeypatch):
    called = []

    def skill(**kwargs):
        called.append(kwargs["skill_id"])
        return {"success": True, "status": "executed", "postcondition": {
            "status": "unknown", "source": "missing_angle",
        }}

    monkeypatch.setattr(door_task, "run_observed_door_skill", skill)
    report = door_task.run_observed_door_subtask(
        **_options(tmp_path, "open", lambda: 0.)
    )
    assert called == ["handle_pull"]
    assert report["success"] is False
    assert report["status"] == "unknown"
    assert report["first_failure"]["code"] == "door_postcondition_unverified"
