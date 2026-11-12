"""人工持刷、单方块清扫的现场只规划入口；无运动执行选项。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_SITE=ROOT/"config/realworld/sweep_old.example.json"
sys.path.insert(0,str(ROOT))


def make_demo(site,source_run:Path,output:Path):
    """从已保存的抬刷状态制作测试输入，不声称是本次真机感知或夹持。"""
    import numpy as np
    from gene_agent.robots.pika_frames import load_frame_profile
    from gene_agent.realworld.sweep.planning import write_json
    source_run=source_run.resolve()
    result=json.loads((source_run/"result.json").read_text(encoding="utf-8"))
    state=result["stages"]["lift"]["end_state"]
    robot=state["robots"]["left"]
    site=copy.deepcopy(site)
    prepared=source_run/"task_scene"
    if (prepared/"scene.xml").is_file():
        site["scene_template"]=str(prepared/"scene.xml")
    else:
        site["scene_template"]=result["scene_xml_path"]
    for role,row in site["objects"].items():
        row["body_name"]=role
        if (prepared/f"concepts/{role}.json").is_file():
            row["concept_path"]=str(prepared/f"concepts/{role}.json")
            row["concept_part_aliases"]={}
    profile=load_frame_profile(site["frame_profile"])
    link=np.asarray(robot["T_world_tcp"])@np.linalg.inv(profile["T_link7_operation_tcp"])
    now=time.time()*1000
    first_target=result["target_order"][0]
    if first_target not in {"cube","cube_red","cube_blue"}:
        raise ValueError("历史样例的第一个目标不是已支持的方块")
    snapshot={"schema_version":1,"source":"synthetic_fixture","observed_at_unix_ms":now,
        "joint_names":[f"joint_{i}" for i in range(1,8)],"joint_positions_rad":robot["joint_positions_rad"],
        "T_sdk_flange":(link@profile["T_link7_sdk_flange"]).tolist(),
        "gripper_width_m":robot["gripper"]["width_m"],"pose_frame_verified":True,
        "camera_frame_number":0,"camera_serial":"synthetic_no_camera",
        "T_sdk_object":{role:state["objects"][source]["T_world_object"]
            for role,source in (("brush","brush"),("cube",first_target),("dustpan","dustpan"))},
        "hold":{"confirmed":True,"source":"synthetic_from_saved_simulation_hold_NOT_real_hold",
                "confirmed_at_unix_ms":now},
        "fixture_source":str(source_run/"result.json"),
        "fixture_target_object":first_target,
        "fixture_source_sha256":hashlib.sha256((source_run/"result.json").read_bytes()).hexdigest()}
    site["calibration_status"]="nominal_audit_only"
    site["calibration_source"]="synthetic fixture; identity SDK base hypothesis"
    for key in ("T_sdk_urdf_base","T_sdk_camera"):site[key]=np.eye(4).tolist()
    write_json(output/"demo_site.json",site);write_json(output/"observation.json",snapshot)
    return site,snapshot


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_SITE)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--snapshot",type=Path,help="同帧现场快照 JSON")
    modes.add_argument("--capture",action="store_true",help="显式连接相机/Pose 服务，只读机器人状态")
    modes.add_argument("--demo",action="store_true",help="明确标记的历史仿真输入，仅验证接线")
    parser.add_argument("--demo-run",type=Path,help="覆盖配置中指定的历史样例")
    parser.add_argument("--allow-synthetic",action="store_true",help="允许读取另存的 synthetic 快照")
    parser.add_argument("--held-brush-confirmed",action="store_true",help="操作者确认当前人工夹稳刷子；不驱动夹爪")
    parser.add_argument("--check-config",action="store_true")
    parser.add_argument("--scene-only",action="store_true",help="只构建和检查现场初态，不运行 GPU 规划")
    parser.add_argument("--output",type=Path,default=ROOT/"runs/realworld_sweep"/time.strftime("%Y%m%d-%H%M%S"))
    args=parser.parse_args(argv)
    from gene_agent.realworld.sweep.scene import read_site,build_observed_scene
    from gene_agent.realworld.sweep.planning import write_json,plan_sweep
    from gene_agent.robots.pika_frames import load_frame_profile
    site=read_site(args.config)
    if args.check_config:
        profile=load_frame_profile(site["frame_profile"])
        print(f"配置结构有效；brush={site.get('brush_variant','unspecified')}；site={site['calibration_status']}；Pika={profile['status']}。不连接设备，不代表现场标定通过。")
        return 0
    if not (args.snapshot or args.capture or args.demo):parser.error("请指定 --snapshot、--capture 或 --demo")
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    source_paths=["tools/run/run_sweep_realworld.py","gene_agent/robots/pika_frames.py",
        "gene_agent/robots/realman_pika.py","gene_agent/scene/rigid_geometry.py",
        "gene_agent/knowledge/sweep.py","gene_agent/skills/library/sweep/skill.py",
        "gene_agent/realworld/sweep/observation.py","gene_agent/realworld/sweep/scene.py",
        "gene_agent/realworld/sweep/planning.py","gene_agent/motion/curobo/scene.py",
        "gene_agent/motion/curobo/core.py","gene_agent/motion/curobo/session.py"]
    write_json(output/"code_sources_sha256.json",{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                                for name in source_paths})
    snapshot=None
    try:
        if args.demo:
            demo_source=args.demo_run or ROOT/site["demo_source_run"]
            site,snapshot=make_demo(site,demo_source,output)
        elif args.capture:
            if site["calibration_status"]!="site_verified" or load_frame_profile(site["frame_profile"])["status"]!="site_verified":
                raise ValueError("先完成现场标定并设置 site_verified；示例配置不能连接设备")
            from gene_agent.realworld.sweep.observation import capture_snapshot
            snapshot=capture_snapshot(site,output,held_brush_confirmed=args.held_brush_confirmed)
        else:snapshot=json.loads(args.snapshot.read_text(encoding="utf-8"))
        site["snapshot_source"]=snapshot["source"]
        write_json(output/"site_input.json",site);write_json(output/"observation.json",snapshot)
        model,data,robot,grip,provenance=build_observed_scene(site,snapshot,output/"scene",
                                                   allow_synthetic=args.demo or args.allow_synthetic)
        if args.scene_only:
            write_json(output/"result.json",{"schema_version":1,"task":"sweep_realworld_plan_only",
                "brush_variant":site.get("brush_variant","unspecified"),
                "source":snapshot["source"],"evidence_mode":"observed_scene_only","execution_enabled":False,
                "planning_complete":False,"task_success":None,"scene_xml_path":str(output/"scene/scene.xml"),
                "T_tcp_brush":grip.tolist(),"candidates":[]})
            print(f"现场初态已保存：{output}；没有调用运动规划或下发命令。")
            return 0
        report=plan_sweep(model,data,robot,grip,site,output)
        print(f"只规划完成={report['planning_complete']}；source={snapshot['source']}；没有执行；记录：{output}")
        return 0 if report["planning_complete"] else 2
    except Exception:
        (output/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        if not (output/"result.json").exists():
            write_json(output/"result.json",{"schema_version":1,"task":"sweep_realworld_plan_only",
                "brush_variant":site.get("brush_variant","unspecified"),
                "execution_enabled":False,"task_success":None,"planning_complete":False,
                "source":None if snapshot is None else snapshot.get("source"),
                "evidence_mode":"failed_plan_only","error":str(sys.exc_info()[1]),
                "scene_xml_path":str(output/"scene/scene.xml"),"candidates":[]})
        print(f"失败证据：{output/'traceback.txt'}",file=sys.stderr)
        raise


if __name__=="__main__":raise SystemExit(main())
