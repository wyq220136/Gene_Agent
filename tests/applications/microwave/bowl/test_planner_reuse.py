"""只替换昂贵 GPU 求解器，实际执行应用边界和有界 pool 的复用逻辑。"""
from types import SimpleNamespace
import json

import numpy as np
import pytest

import gene_agent.tasks.policies.microwave_task.planning.insert as transfer
from gene_agent.motion.curobo.session import CuroboPlannerPool


def test_sequential_stage_acquisition_is_lazy_and_restarts_sampling(monkeypatch):
    created = []

    class Backend:
        def __init__(self, **kwargs):
            self.cursor = 0
            self.closed = False
            created.append(self)

        def reset_seed(self):
            self.cursor = 0

        def fk(self, names, positions):
            self.cursor += 1
            return self.cursor

        def close(self):
            self.closed = True

    monkeypatch.setattr(transfer, "load_realman_motion_config", lambda *a, **k: {})
    snapshot = {"scene_model": {}, "scope": "test", "snapshot_sha256": "abc",
                "gripper_snapshot": {}}
    with CuroboPlannerPool(planner_factory=Backend) as pool:
        first = transfer._planner({}, snapshot, sphere_link_m=None,
                                  seed=123, revision=0, pool=pool)
        second = transfer._planner({}, snapshot, sphere_link_m=None,
                                   seed=123, revision=1, pool=pool)
        assert created == []  # 不能预先获取下一阶段而修改仍在使用的实例。
        state = SimpleNamespace(joint_names=("j1",), positions_rad=(0.,))
        assert first.fk(state) == 1
        assert first.fk(state) == 2
        assert second.fk(state) == 1  # 新阶段恢复原先冷构造的采样起点。
        assert len(created) == 1
        assert second.planner.planner_cache["cache_hit"]
        assert first.cache_evidence["cache_hit"] is False
        assert second.cache_evidence["cache_hit"] is True
    assert created[0].closed


def test_stage_pool_keeps_attachment_and_collision_topology_separate(monkeypatch):
    made = []

    class Backend:
        def __init__(self, **kwargs):
            made.append(kwargs)
        def reset_seed(self):
            pass
        def close(self):
            pass

    monkeypatch.setattr(transfer, "load_realman_motion_config", lambda *a, **k: {})
    base = {"scene_model": {}, "scope": "test", "snapshot_sha256": "a",
            "gripper_snapshot": {}}
    with CuroboPlannerPool(planner_factory=Backend) as pool:
        for scene, spheres in ((base, None), (base, [[0, 0, 0, .01]]),
                               ({**base, "scope": "contact"}, None)):
            transfer._planner({}, scene, sphere_link_m=spheres,
                              seed=123, revision=0, pool=pool).planner
    assert len(made) == 3


@pytest.mark.parametrize("saved_tcp", [True, False])
@pytest.mark.parametrize("interface", ["stage", "legacy"])
def test_insert_reuses_recorded_actual_tcp_and_accepts_legacy_reports(tmp_path, monkeypatch, saved_tcp, interface):
    """新增记录应消除FK实例；旧记录仍从相同approach终态求FK。"""
    actual = np.eye(4)
    actual[:3, 3] = [.2, .3, .4]
    end = {"joint_names": ["j1"], "positions_rad": [.25]}
    attachment = {"snapshot_sha256": "hash"}
    if saved_tcp:
        attachment["T_world_tcp_at_attach"] = actual.tolist()
    report = {"success": True, "instance_id": "bowl", "scene_revision": 0,
              "visual_attachment": attachment,
              "stages": [{"name": name, "result": {"goal_state": end}}
                         for name in ("lift", "approach")]}
    (tmp_path / "skill_pick_bowl_result.json").write_text(json.dumps(report))
    (tmp_path / "visual_attachment").mkdir()
    (tmp_path / "visual_attachment/collision_snapshot.json").write_text(
        json.dumps({"snapshot_sha256": "hash"}))
    fk_calls = []

    def fk(state):
        fk_calls.append(state.positions_rad)
        return actual

    monkeypatch.setattr(transfer, "_planner", lambda *a, **k: SimpleNamespace(fk=fk))
    monkeypatch.setattr(transfer, "_SessionStageAdapter", lambda *a, **k: SimpleNamespace(fk=fk))

    class ReachedSphereConversion(Exception):
        pass

    def check_tcp(snapshot, tcp, robot):
        np.testing.assert_array_equal(tcp, actual)
        raise ReachedSphereConversion

    monkeypatch.setattr(transfer, "_sphere_link_from_world", check_tcp)
    case = SimpleNamespace(output_dir=tmp_path, revision=0, metadata={"robot": {}})
    with pytest.raises(ReachedSphereConversion):
        transfer.run_insert_bowl_skill_planning(case, motion_interface=interface)
    assert fk_calls == ([] if saved_tcp else [(.25,)])
