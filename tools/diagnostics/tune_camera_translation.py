"""固定机器人和标定旋转/Y，只在 Viser 调整点云的 SDK X/Z 平移。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    import argparse, time, json
    import numpy as np
    import mujoco
    from gene_agent.visualization.realworld_scene import scene_bundle, load_observation
    from gene_agent.visualization.afford_scene import create_scene_server, _wxyz, add_coordinate_frames
    from gene_agent.visualization.pick_bowl.scene import PickBowlScene
    from gene_agent.realworld.microwave.perception_setup import load_camera_calibration
    from gene_agent.perception_modules.camera.calibration.translation_refinement import save_translation_candidate
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--camera-params",required=True)
    p.add_argument("--output",required=True,help="保存候选的新目录；只有点击保存才写文件")
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,default=8121)
    args=p.parse_args()
    calibration=load_camera_calibration(args.camera_params)
    with scene_bundle(args.run) as bundle:
        scene,model,data,rgbd,warning=load_observation(bundle)
        if rgbd is None:raise ValueError(warning)
        server=create_scene_server(host=args.host,port=args.port)
        try:
            world_sdk=np.asarray(scene["provenance"]["T_world_sdk"],float)
            sdk_camera=np.asarray(calibration["T_sdk_camera"],float)
            world_camera=world_sdk@sdk_camera
            geometry=PickBowlScene(server,model,data,scene.get("robot_metadata",{}),{})
            for gid,handle in geometry.visual_handles.items():
                body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[gid])) or ""
                handle.visible=body.startswith(("left_","right_","robot_body"))
            ids=np.flatnonzero(rgbd["valid"]);ids=ids[np.linspace(0,len(ids)-1,min(len(ids),100000),dtype=int)]
            cloud=server.scene.add_point_cloud("/measured_rgbd",points=rgbd["cloud"].reshape(-1,3)[ids],colors=rgbd["rgb"].reshape(-1,3)[ids],point_size=.002,
                position=world_camera[:3,3],wxyz=_wxyz(world_camera))
            frames=add_coordinate_frames(server,{"SDK":world_sdk,"camera":world_camera},axes_length=.12)
            server.gui.add_markdown("## SDK X/Z 平移核对\n\n机器人 mesh 来自原实测关节，保持固定；点云使用所选标定。"
                "旋转、SDK Y、内参及原始板中测量值均不改。此界面不发运动命令，不重新规划。"
                "截图中的水平/竖直方向不一定是 SDK X/Z，请看坐标轴。")
            x=server.gui.add_slider("SDK ΔX (mm)",min=-100.,max=100.,step=.5,initial_value=0.)
            z=server.gui.add_slider("SDK ΔZ (mm)",min=-100.,max=100.,step=.5,initial_value=0.)
            status=server.gui.add_markdown("未保存。请同时检查腕部和上臂，不能只贴合一个局部。")
            def update(_):
                delta=np.array([x.value,0,z.value])*.001
                cloud.position=world_camera[:3,3]+world_sdk[:3,:3]@delta
                frames[1].position=cloud.position
                status.content="SDK 附加平移（米）：`"+json.dumps(delta.tolist())+"`；旋转与 Y 不变。"
            x.on_update(update);z.on_update(update)
            reset=server.gui.add_button("归零（不写文件）")
            def zero(_):x.value=0.;z.value=0.;update(None)
            reset.on_click(zero)
            save=server.gui.add_button("另存平移候选")
            def persist(_):
                try:
                    path=save_translation_candidate(args.camera_params,[x.value*.001,0,z.value*.001],args.output)
                    status.content=f"候选已保存：`{path}`；原标定未覆盖。需新 PERCEPTION 和新 OUT 核对。"
                except Exception as error:status.content=f"未保存：{type(error).__name__}: {error}"
            save.on_click(persist)
            server.gui.add_image(rgbd["rgb"],label=f"冻结 RGB frame {scene['provenance']['source_frame_number']}")
            server.initial_camera.look_at=(.2,.2,0.);server.initial_camera.position=(-.8,-.7,.7)
            print(f"XZ translation viewer: http://{args.host}:{args.port}",flush=True)
            while True:time.sleep(.25)
        except KeyboardInterrupt:pass
        finally:server.stop()


if __name__=="__main__":main()
