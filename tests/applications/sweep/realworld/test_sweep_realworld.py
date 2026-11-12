"""真机只规划入口必须区分实测、预测和合成输入。"""
import json
import time

import numpy as np
import pytest

from gene_agent.robots.pika_frames import load_frame_profile
from gene_agent.realworld.sweep.observation import validate_snapshot


def profile_payload():
    return {"schema_version": 1, "status": "nominal_audit_only", "source": "test fixture",
            "T_link7_sdk_flange": np.eye(4).tolist(),
            "T_link7_pika_model": np.eye(4).tolist(),
            "T_pika_model_operation_tcp": np.eye(4).tolist(),
            "T_sdk_flange_reference_control": np.eye(4).tolist()}


def snapshot_payload():
    now = time.time()*1000
    return {"schema_version": 1, "source": "realworld_observation", "observed_at_unix_ms": now,
            "joint_names": [f"joint_{i}" for i in range(1,8)], "joint_positions_rad": [0]*7,
            "gripper_width_m": .04, "T_sdk_flange": np.eye(4).tolist(),
            "T_sdk_object": {name: np.eye(4).tolist() for name in ("brush", "cube", "dustpan")},
            "camera_frame_number": 1, "camera_serial": "fixture", "pose_frame_verified": True,
            "hold": {"confirmed": True, "source": "manual_cli", "confirmed_at_unix_ms": now}}


def test_named_frame_profile_composes_and_does_not_assume_tip(tmp_path):
    payload = profile_payload()
    payload["T_link7_sdk_flange"][2][3] = -.0172
    payload["T_pika_model_operation_tcp"][0][3] = .18
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload))
    result = load_frame_profile(path)
    np.testing.assert_allclose(result["T_sdk_flange_operation_tcp"][:3,3], [.18,0,.0172])
    assert result["status"] == "nominal_audit_only"
    payload["T_link7_pika_model"][0][0] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_frame_profile(path)


def test_real_snapshot_checks_freshness_hold_and_source():
    snapshot = snapshot_payload()
    validate_snapshot(snapshot, max_age_s=60)
    snapshot["hold"]["confirmed"] = False
    with pytest.raises(ValueError, match="夹持"):
        validate_snapshot(snapshot, max_age_s=60)
    snapshot = snapshot_payload()
    snapshot["observed_at_unix_ms"] -= 120000
    with pytest.raises(ValueError, match="过期"):
        validate_snapshot(snapshot, max_age_s=60)
    snapshot = snapshot_payload()
    snapshot["source"] = "synthetic_fixture"
    with pytest.raises(ValueError, match="synthetic"):
        validate_snapshot(snapshot, max_age_s=60)
    validate_snapshot(snapshot, max_age_s=60, allow_synthetic=True)


def test_snapshot_rejects_wrong_order_nonfinite_and_unverified_frame():
    for key, value in (("joint_names", [f"left_joint_{i}" for i in range(1,8)]),
                       ("joint_positions_rad", [float("nan")]*7),
                       ("pose_frame_verified", False), ("gripper_width_m", -.01)):
        snapshot = snapshot_payload()
        snapshot[key] = value
        with pytest.raises(ValueError):
            validate_snapshot(snapshot, max_age_s=60)


def test_chinese_planner_evidence_is_saved_under_ascii_locale(tmp_path):
    import os
    import subprocess
    import sys
    code=('from pathlib import Path; import sys; '
          'from gene_agent.realworld.sweep.planning import write_json; '
          'write_json(Path(sys.argv[1]), {"message":"\\u78b0\\u649e"})')
    result=subprocess.run([sys.executable,"-c",code,str(tmp_path/"evidence.json")],
        env={**os.environ,"LC_ALL":"C","PYTHONUTF8":"0","PYTHONCOERCECLOCALE":"0"},
        capture_output=True)
    assert result.returncode==0,result.stderr.decode(errors="replace")
    assert json.loads((tmp_path/"evidence.json").read_text(encoding="utf-8"))["message"]=="碰撞"


def test_stage_filter_drops_mount_exception_for_removed_table(tmp_path):
    from gene_agent.motion.curobo.scene import filter_collision_snapshot
    source={"snapshot_sha256":"fixture","scene_model":{"cuboid":{"table":{},"cube":{}},"mesh":{}},
        "records":[{"obstacle_name":"table","body_name":"world","geom_name":"scene_table"},
                   {"obstacle_name":"cube","body_name":"cube","geom_name":"cube_collision"}],
        "mounting_contact_policy":{"robot_links":["base_link"],"obstacle_names":["table"]}}
    stage=filter_collision_snapshot(source,tmp_path,keep_body_names=None,rejected_body_names=set(),
        rejected_geom_names={"scene_table"},attached_object_id=None,scope="test")
    assert stage["mounting_contact_policy"]["obstacle_names"]==[]
    assert source["mounting_contact_policy"]["obstacle_names"]==["table"]


def test_allowed_brush_cube_contact_does_not_allow_finger_cube_collision():
    import mujoco
    from gene_agent.realworld.sweep.planning import check_world_contacts
    chain=''.join(f'<body name="link_{i}"><inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/>'
                  f'<joint name="joint_{i}" axis="0 0 1"/>' for i in range(1,8))
    xml=('<mujoco><worldbody><body name="base_link">'+chain+
         '<site name="pika_tcp"/><body name="pika_gripper_left_link">'
         '<geom type="box" size=".01 .01 .01"/></body>'+ '</body>'*8+
         '<body name="brush"><freejoint/><geom type="box" size=".01 .01 .01"/></body>'
         '<body name="cube"><geom type="box" size=".01 .01 .01"/></body>'
         '</worldbody></mujoco>')
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    robot={"joint_groups":{"left":[f"joint_{i}" for i in range(1,8)]}}
    allowed=(("brush","cube"),("pika_gripper_left_link","brush"))
    positions=[[0]*7 for _ in range(8)]
    result=check_world_contacts(model,data,robot,positions,np.eye(4),allowed)
    assert not result["success"]
    assert any(set(row["roles"])=={"pika_gripper_left_link","cube"} for row in result["forbidden_contacts"])
    assert not any(set(row["roles"])=={"brush","cube"} for row in result["forbidden_contacts"])
    assert result["sample_count"]==1+max(row["sample"] for row in result["forbidden_contacts"])
    assert result["sample_count"]<len(positions)


@pytest.mark.parametrize("site_name",["sweep_long.example.json","sweep_old.example.json"])
def test_scene_adapter_keeps_old_assets_and_removes_simulation_weld(tmp_path,site_name):
    from pathlib import Path
    import mujoco
    from gene_agent.realworld.sweep.scene import read_site,build_observed_scene,ROOT
    from gene_agent.robots.pika_frames import model_frames,pose_from_mujoco
    from gene_agent.robots.realman_pika import build_pika_robot
    site=read_site(ROOT/"config/realworld"/site_name)
    snapshot=snapshot_payload();snapshot["source"]="synthetic_fixture"
    profile=load_frame_profile(site["frame_profile"])
    robot=build_pika_robot(tmp_path/"source_robot",frame_profile_path=site["frame_profile"])
    model=mujoco.MjModel.from_xml_path(robot["xml_path"]);data=mujoco.MjData(model)
    model_frames(model,data,robot,[0]*7,.04)
    snapshot["T_sdk_flange"]=(pose_from_mujoco(data.body("link_7"))@profile["T_link7_sdk_flange"]).tolist()
    source=Path(site["scene_template"]);before=source.read_bytes()
    built,state,metadata,grip,provenance=build_observed_scene(site,snapshot,tmp_path/"scene",allow_synthetic=True)
    assert built.neq==0
    assert state.time==0
    assert provenance["physics_stepped"] is False
    assert source.read_bytes()==before
    assert metadata["frame_profile"]["sha256"]==profile["sha256"]
    assert grip.shape==(4,4)
    with pytest.raises(ValueError,match="synthetic"):
        build_observed_scene(site,snapshot,tmp_path/"must_reject")


def test_old_brush_is_default_and_demo_uses_its_first_target(tmp_path):
    import importlib.util
    from pathlib import Path
    from gene_agent.realworld.sweep.scene import read_site,ROOT
    spec=importlib.util.spec_from_file_location("sweep_cli",ROOT/"tools/run/run_sweep_realworld.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.DEFAULT_SITE.name=="sweep_old.example.json"
    site=read_site(ROOT/"config/realworld/sweep_old.example.json")
    assert site["brush_variant"]=="old"
    assert site["objects"]["brush"]["body_name"]=="brush"
    run=ROOT/site["demo_source_run"]
    source=json.loads((run/"result.json").read_text(encoding="utf-8"))
    effective,snapshot=module.make_demo(site,run,tmp_path)
    assert effective["brush_variant"]=="old"
    assert snapshot["fixture_target_object"]==source["target_order"][0]
    np.testing.assert_allclose(snapshot["T_sdk_object"]["cube"],source["stages"]["lift"]["end_state"]["objects"][source["target_order"][0]]["T_world_object"])


@pytest.mark.parametrize("kind,selection",[("skill","sweep_cube"),("task","sweep_cube_into_dustpan")])
def test_existing_sweep_name_routes_by_record_type(tmp_path,monkeypatch,kind,selection):
    import sys
    from gene_agent.visualization import realworld_sweep as viewer
    (tmp_path/"result.json").write_text(json.dumps({"task":"sweep_realworld_plan_only","brush_variant":"old"}))
    calls=[]
    monkeypatch.setattr(viewer,"serve_preview",lambda run,**kwargs:calls.append(run))
    monkeypatch.setattr(sys,"argv",["viewer.py",f"--{kind}",selection,"--run",str(tmp_path)])
    assert viewer.dispatch_legacy_viewer(kind)
    assert calls==[tmp_path]
    (tmp_path/"result.json").write_text(json.dumps({"task":"sweep_cube_into_dustpan"}))
    assert viewer.dispatch_legacy_viewer(kind) is False


def test_explicit_operation_frame_rotation_matches_model_site(tmp_path):
    import mujoco
    from scipy.spatial.transform import Rotation
    from gene_agent.robots.realman_pika import build_pika_robot,CONFIG
    from gene_agent.robots.pika_frames import pose_from_mujoco
    payload=profile_payload()
    payload["T_link7_pika_model"][:]=np.eye(4).tolist()
    target=np.eye(4);target[:3,:3]=Rotation.from_euler("z",30,degrees=True).as_matrix()
    target[:3,3]=[.18,.01,0]
    payload["T_pika_model_operation_tcp"]=target.tolist()
    path=tmp_path/"profile.json";path.write_text(json.dumps(payload))
    default_before=CONFIG.read_bytes()
    robot=build_pika_robot(tmp_path/"robot",frame_profile_path=path)
    model=mujoco.MjModel.from_xml_path(robot["xml_path"]);data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    actual=np.linalg.inv(pose_from_mujoco(data.body("link_7")))@pose_from_mujoco(data.site("pika_tcp"))
    np.testing.assert_allclose(actual,target,atol=1e-7)
    np.testing.assert_allclose(robot["transforms"]["left"]["link_from_tcp"],target,atol=1e-7)
    from gene_agent.robots import measure_gripper_width
    assert measure_gripper_width(model,data,robot,"left")["width_m"]==pytest.approx(
        robot["grippers"]["left"]["width_range_m"][1],abs=1e-6)
    assert CONFIG.read_bytes()==default_before


def test_cli_does_not_offer_hardware_execution():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[4]/"tools/run/run_sweep_realworld.py"
    spec=importlib.util.spec_from_file_location("sweep_realworld_cli",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as caught:
        module.main(["--execute"])
    assert caught.value.code==2


def test_nonzero_sdk_model_residual_does_not_move_observed_brush(tmp_path):
    import mujoco
    from gene_agent.realworld.sweep.scene import read_site,build_observed_scene,ROOT
    from gene_agent.realworld.sweep.planning import _set_prediction
    from gene_agent.robots.pika_frames import model_frames,pose_from_mujoco
    from gene_agent.robots.realman_pika import build_pika_robot
    site=read_site(ROOT/"config/realworld/sweep_long.example.json")
    profile=load_frame_profile(site["frame_profile"])
    robot=build_pika_robot(tmp_path/"source_robot",frame_profile_path=site["frame_profile"])
    model=mujoco.MjModel.from_xml_path(robot["xml_path"]);data=mujoco.MjData(model)
    model_frames(model,data,robot,[0]*7,.04)
    observed_flange=pose_from_mujoco(data.body("link_7"))@profile["T_link7_sdk_flange"]
    observed_flange[2,3]+=.002
    snapshot=snapshot_payload();snapshot["source"]="synthetic_fixture"
    snapshot["T_sdk_flange"]=observed_flange.tolist()
    built,state,metadata,grip,provenance=build_observed_scene(site,snapshot,tmp_path/"scene",allow_synthetic=True)
    before=pose_from_mujoco(state.body("brush"))
    _set_prediction(built,state,metadata,[0]*7,grip)
    np.testing.assert_allclose(pose_from_mujoco(state.body("brush")),before,atol=1e-10)
    assert provenance["sdk_model_tcp_registration"]["applied_for_planning"] is True
    np.testing.assert_allclose(provenance["T_world_tcp_observed_from_sdk"],
                              observed_flange@profile["T_sdk_flange_operation_tcp"])
