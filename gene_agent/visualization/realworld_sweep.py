"""清扫现场只规划记录的显示；没有运动或感知控制接口。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading

import mujoco
import numpy as np

from gene_agent.robots.pika_frames import pose_from_mujoco
from gene_agent.visualization.frame_axes import append_frame_axes


def load_preview(run:Path):
    run=Path(run).resolve()
    record_path=run/"result.json"
    if not record_path.exists():record_path=run/"execution_result.json"
    report=json.loads(record_path.read_text(encoding="utf-8"))
    executed=report.get("source_format")=="realworld_sweep_execution"
    if not executed and (report.get("task")!="sweep_realworld_plan_only" or report.get("execution_enabled") is not False):
        raise ValueError("不是清扫只规划记录")
    model=mujoco.MjModel.from_xml_path(report["scene_xml_path"]);data=mujoco.MjData(model)
    frames_path=run/("executed_frames.json" if executed else "preview_frames.json")
    frames=json.loads(frames_path.read_text(encoding="utf-8")) if frames_path.exists() else [{
        "stage":"initial_only","predicted":False,"qpos":np.load(run/"scene/initial_state.npz")["qpos"].tolist()}]
    if not frames or any(len(frame["qpos"])!=model.nq for frame in frames):
        raise ValueError("记录中的 qpos 与保存场景不一致")
    for frame in frames:frame.setdefault("predicted",executed)
    data.qpos[:]=frames[0]["qpos"];mujoco.mj_forward(model,data)
    return report,model,data,frames


def preview_axes(data):
    return {"TCP":pose_from_mujoco(data.site("pika_tcp")),
            "BRUSH":pose_from_mujoco(data.body("brush")),
            "CUBE":pose_from_mujoco(data.body("cube")),
            "PAN":pose_from_mujoco(data.body("dustpan"))}


def render_preview(run:Path):
    from PIL import Image,ImageDraw,ImageFont
    report,model,data,frames=load_preview(run)
    output=Path(run)/"previews";output.mkdir(exist_ok=True)
    model.vis.global_.offwidth=1200;model.vis.global_.offheight=900
    option=mujoco.MjvOption();option.geomgroup[3]=0
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",20)
    contact=max((i for i,f in enumerate(frames) if f["stage"]=="lower_to_contact"),default=0)
    for name,index,azimuth,elevation,distance,center in (
        ("initial_overview",0,125,-30,1.55,[.25,0,.24]),
        ("initial_top",0,90,-85,1.25,[.36,.08,.10]),
        ("contact_plan",contact,140,-20,.75,[.35,.1,.16]),
        ("final_plan",len(frames)-1,120,-25,1.1,[.35,.1,.18]),
    ):
        data.qpos[:]=frames[index]["qpos"];mujoco.mj_forward(model,data)
        camera=mujoco.MjvCamera();camera.azimuth=azimuth;camera.elevation=elevation
        camera.distance=distance;camera.lookat[:]=center
        with mujoco.Renderer(model,height=900,width=1200) as renderer:
            renderer.update_scene(data,camera=camera,scene_option=option)
            append_frame_axes(renderer.scene,preview_axes(data),axis_length_m=.04)
            image=Image.fromarray(renderer.render())
        canvas=Image.new("RGB",(1200,980),"#f2f4f6");canvas.paste(image,(0,0))
        draw=ImageDraw.Draw(canvas)
        draw.text((12,909),f"{name} | {frames[index]['stage']} | source={report['source']}",fill="black",font=font)
        caption=("SDK endpoint samples; brush pose is a rigid-grasp hypothesis, cube stays at first frame"
                 if report.get("source_format")=="realworld_sweep_execution" else
                 "PLAN PREVIEW ONLY: no robot execution, no cube motion, no task success claim")
        draw.text((12,943),caption,fill="#a02020",font=font)
        canvas.save(output/f"{name}.png")
    print(f"截图：{output}")


def serve_preview(run:Path,*,host="127.0.0.1",port=8092):
    import viser
    from gene_agent.visualization.pick_bowl.scene import PickBowlScene,_pose
    report,model,data,frames=load_preview(run)
    server=viser.ViserServer(host=host,port=port)
    if report.get("device_evidence_mode","real_sdk")!="real_sdk":
        server.gui.add_markdown("**离线假设备测试：未发送真机命令；这些端点不是实机测量。**")
    server.scene.set_up_direction("+z")
    scene=PickBowlScene(server,model,data,{}, {})
    support=report.get("support_adjustment")
    if support:
        from gene_agent.scene.rigid_geometry import visual_mesh_object
        original_handles=[]
        descriptions=[]
        for role,row in support["objects"].items():
            descriptions.append(f"{role}: Z +{row['shift_world_z_m']*1000:.1f} mm")
            mesh=visual_mesh_object(model,data,role)
            wxyz,position=_pose(np.asarray(row["T_world_object_observed"]))
            original_handles.append(server.scene.add_mesh_simple(f"/sweep_preview/raw_perception/{role}",
                vertices=np.asarray(mesh.vertices,dtype=np.float32),faces=np.asarray(mesh.faces,dtype=np.uint32),
                color=(170,60,210),opacity=.3,wxyz=wxyz,position=position,visible=False))
        server.gui.add_markdown("### 初始桌面支撑校正\n\n"+"；".join(descriptions)+
            "。机器人关节仍为原输入；物体校正是规划假设，未覆盖原始感知。")
        raw_toggle=server.gui.add_checkbox("Show raw perception meshes",initial_value=False)
        @raw_toggle.on_update
        def show_raw(event):
            for handle in original_handles:handle.visible=event.target.value
    # 实线来自轨迹逐样本 FK，不把目标 pose 连线当作机器人规划。
    path=[]
    for row in frames:
        data.qpos[:]=row["qpos"];mujoco.mj_forward(model,data)
        path.append(data.site("pika_tcp").xpos.copy())
    data.qpos[:]=frames[0]["qpos"];mujoco.mj_forward(model,data)
    executed=report.get("source_format")=="realworld_sweep_execution"
    if len(path)>1 and not executed:
        points=np.asarray(path,dtype=np.float32)
        trajectory=server.scene.add_line_segments("/sweep_preview/tcp_trajectory",
            points=np.stack((points[:-1],points[1:]),axis=1),colors=(230,100,30),line_width=3.)
        show_path=server.gui.add_checkbox("Show cuRobo TCP trajectory",initial_value=True)
        @show_path.on_update
        def path_visible(event):trajectory.visible=event.target.value
    # 拒绝轨迹独立显示，不接入有效前缀，也不用于下一个 Skill 的起点。
    rejected_samples=[]
    stages=list(report.get("pick",{}).get("stages",[]))
    for row in report.get("candidates",[]):stages.extend(row.get("stages",[]))
    for index,stage in enumerate(stages):
        check=stage.get("postcheck",stage.get("full_world_postcheck",{}))
        rejected_samples.extend(s for s in check.get("forbidden_contacts",[]) if "contact_position_world_m" in s)
        rejected_path=stage.get("rejected_preview_frames_path")
        if not rejected_path:continue
        diagnostic=json.loads(Path(rejected_path).read_text(encoding="utf-8"))
        line=[]
        for sample in diagnostic:
            data.qpos[:]=sample["qpos"];mujoco.mj_forward(model,data)
            line.append(data.site("pika_tcp").xpos.copy())
        if len(line)>1:
            line=np.asarray(line,dtype=np.float32)
            server.scene.add_line_segments(f"/sweep_preview/rejected/{index}",
                points=np.stack((line[:-1],line[1:]),axis=1),colors=(220,25,30),line_width=4.)
    if rejected_samples:
        server.scene.add_point_cloud("/sweep_preview/rejected_contacts",
            points=np.asarray([s["contact_position_world_m"] for s in rejected_samples]),
            colors=(255,0,0),point_size=.009)
        server.gui.add_markdown("**红线/红点：被拒绝的诊断轨迹与碰撞位置，不属于可接受规划。**")
    data.qpos[:]=frames[0]["qpos"];mujoco.mj_forward(model,data)
    point_path=Path(run)/"perception/scene_points.npz"
    if point_path.exists():
        cloud=np.load(point_path)
        point_handle=server.scene.add_point_cloud("/sweep_preview/observed_rgbd",
            points=cloud["points"],colors=cloud["colors"],point_size=.002,visible=False)
        show_cloud=server.gui.add_checkbox("Show first-frame RGB-D cloud",initial_value=False)
        @show_cloud.on_update
        def cloud_visible(event):point_handle.visible=event.target.value
    quality_path=Path(run)/"perception/registration_quality.json"
    if quality_path.exists():
        quality=json.loads(quality_path.read_text(encoding="utf-8"))
        warnings=[f"{name}: 中位残差 {row['median_depth_to_surface_m']*1000:.1f} mm"
                  for name,row in quality.items() if row.get("warning")]
        if warnings:
            server.gui.add_markdown("### 感知待核对\n\n"+"；".join(warnings)+
                "。请打开 RGB-D 点云对照；残差不是 GT pose 误差，规划完成也不代表感知正确。")
    target_handles=[]
    candidates=report.get("candidates",[])
    if candidates:
        chosen=report.get("selected_index")
        targets=candidates[report.get("preview_candidate_index",0) if chosen is None else chosen]["targets"]
        for name,pose in targets.items():
            wxyz,position=_pose(np.asarray(pose))
            target=server.scene.add_frame(f"/sweep_preview/targets/{name}",wxyz=wxyz,
                position=position,axes_length=.05,axes_radius=.0015,visible=False)
            server.scene.add_label(f"/sweep_preview/targets/{name}/label",f"target: {name}")
            target_handles.append(target)
    for name,pose in report.get("pick",{}).get("targets",{}).items():
        wxyz,position=_pose(np.asarray(pose))
        target=server.scene.add_frame(f"/sweep_preview/pick_targets/{name}",wxyz=wxyz,
            position=position,axes_length=.05,axes_radius=.0015,visible=False)
        target_handles.append(target)
    handles={name:server.scene.add_frame(f"/sweep_preview/frames/{name}",
        wxyz=_pose(pose)[0],position=_pose(pose)[1],axes_length=.04,axes_radius=.0015)
        for name,pose in preview_axes(data).items()}
    cube_pose=data.body("cube").xpos.copy()
    server.scene.add_label("/sweep_preview/cube_label","cube stays at observed position",position=cube_pose+[0,0,.04])
    description=("本记录为SDK离散实测关节端点；不是连续遥测。刷子按已确认的刚性夹持假设显示，"
                 "方块保持首帧位置，真实任务结果见独立人工检查记录。" if executed else
                 "没有机器人执行；时间轴是 cuRobo 规划 FK。桌面抓刷预览中的闭爪和夹持均是假设，"
                 "仅从预测抓取终点开始绑定刷子；方块保持观测初态。")
    server.gui.add_markdown(("### 清扫真机执行记录\n\n" if executed else "### 清扫真机接入 · 只规划预览\n\n")+
        f"刷子：**{ {'old':'原蓝色旧刷','long':'白柄长刷'}.get(report.get('brush_variant'),'以当前 mesh 为准')}**。\n\n"
        f"输入来源：**{report['source']}**。\n\n"
        f"机器人初态：**{report.get('robot_state_source','以 observation.json 为准')}**。\n\n"
        +description+"\n\n"+
        (f"执行状态：{report.get('status')}；任务结果：{report.get('task_success')}" if executed else
         f"请求范围：**{report.get('preview_scope','task')}**；范围内规划完成："
         f"**{report.get('requested_scope_complete',report.get('planning_complete',False))}**。"
         "未进行清扫成功验收。"))
    if not report.get("requested_scope_complete",report.get("planning_complete")) and not executed:
        from gene_agent.realworld.sweep.diagnostics import failure_summary
        server.gui.add_markdown("### 停止原因\n\n"+failure_summary(report))
    observation_path=Path(run)/"observation.json"
    if observation_path.exists():
        observation=json.loads(observation_path.read_text(encoding="utf-8"))
        actual=report.get("robot_state_source")=="sdk_read_only"
        description="拍照时 SDK 实测关节（不是持续实时遥测）" if actual else "假定关节；该历史记录没有实机姿态"
        joints=observation.get("joint_positions_rad",[])
        server.gui.add_markdown(description+"\n\n关节 rad：`"+", ".join(f"{v:.4f}" for v in joints)+"`")
    from gene_agent.visualization.operation_evidence import operation_evidence_markdown
    shared_description = operation_evidence_markdown(report)
    if shared_description: server.gui.add_markdown(shared_description)
    slider=server.gui.add_slider("Plan sample",min=0,max=max(1,len(frames)-1),step=1,initial_value=0)
    reset=server.gui.add_button("返回首帧机器人状态")
    @reset.on_click
    def reset_initial(_event):slider.value=0
    status=server.gui.add_markdown(f"阶段：{frames[0]['stage']}")
    show_targets=server.gui.add_checkbox("Show target TCP frames",initial_value=False)
    @show_targets.on_update
    def targets_visible(event):
        for target in target_handles:target.visible=event.target.value
    lock=threading.Lock()
    @slider.on_update
    def update(_event):
        with lock,server.atomic():
            row=frames[min(int(slider.value),len(frames)-1)]
            data.qpos[:]=row["qpos"];mujoco.mj_forward(model,data)
            scene.update_visuals(data)
            for name,pose in preview_axes(data).items():handles[name].wxyz,handles[name].position=_pose(pose)
            status.content=f"阶段：{row['stage']}；预测状态：{row['predicted']}"
    @server.on_client_connect
    def connected(client):
        client.camera.look_at=(.35,.05,.22);client.camera.position=(1.1,-.9,.85)
    print(f"只规划预览：http://{host}:{port}；Ctrl+C 退出",flush=True)
    try:threading.Event().wait()
    except KeyboardInterrupt:pass
    finally:server.stop()


def dispatch_legacy_viewer(kind:str)->bool:
    """保留既有 view_skill/view_task 命令；该分支不进入仿真实际回放。"""
    probe=argparse.ArgumentParser(add_help=False)
    probe.add_argument(f"--{kind}")
    probe.add_argument("--run",type=Path)
    args,_=probe.parse_known_args()
    selections=({"sweep_cube":"old","sweep_long_brush":"long"} if kind=="skill" else
                {"sweep_cube_into_dustpan":"old","sweep_long":"long"})
    selection=getattr(args,kind)
    if selection not in (None,"sweep_realworld",*selections):return False
    record={}
    if args.run is not None and (args.run/"result.json").is_file():
        record=json.loads((args.run/"result.json").read_text(encoding="utf-8"))
    elif args.run is not None and (args.run/"execution_result.json").is_file():
        record=json.loads((args.run/"execution_result.json").read_text(encoding="utf-8"))
    if (record.get("task")!="sweep_realworld_plan_only" and
        record.get("source_format")!="realworld_sweep_execution" and selection!="sweep_realworld"):return False
    variant=record.get("brush_variant")
    if selection in selections and variant in {"old","long"} and selections[selection]!=variant:
        raise ValueError("记录中的刷子型号与所选清扫 Skill/Task 不同")
    parser=argparse.ArgumentParser(description="清扫现场只规划预览，不是 MuJoCo 执行回放")
    # sweep_realworld 只兼容旧命令；新命令沿用同一个清扫 Skill/Task 名称。
    parser.add_argument(f"--{kind}",choices=(*selections,"sweep_realworld"))
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--host",default="127.0.0.1");parser.add_argument("--port",type=int,default=8092 if kind=="skill" else 8093)
    parser.add_argument("--render-only",action="store_true")
    args=parser.parse_args()
    if args.render_only:render_preview(args.run)
    else:serve_preview(args.run,host=args.host,port=args.port)
    return True
