"""随机样本的准确快照及只读三维查看；不再次采样或编辑场景。"""
import argparse
import json
from pathlib import Path
import threading

import mujoco
import numpy as np


def save_snapshot(case,folder,sample):
    folder=Path(folder)
    mujoco.mj_saveLastXML(str(folder/'view_scene.xml'),case.model)
    fields=('qpos','qvel','ctrl','act','mocap_pos','mocap_quat','eq_active')
    np.savez_compressed(folder/'view_state.npz',
        **{n:np.asarray(getattr(case.data,n)).copy() for n in fields})
    (folder/'view_meta.json').write_text(json.dumps({'task':case.task,'seed':sample['seed'],
        'valid':sample['valid'],'source':sample['source'],'time_s':float(case.data.time)},indent=2))


def load_snapshot(folder):
    folder=Path(folder)
    model=mujoco.MjModel.from_xml_path(str(folder/'view_scene.xml'))
    data=mujoco.MjData(model)
    with np.load(folder/'view_state.npz',allow_pickle=False) as state:
        for name in state.files:
            target=getattr(data,name)
            if target.shape!=state[name].shape:raise ValueError('样本模型与状态尺寸不一致')
            target[:]=state[name]
    metadata=json.loads((folder/'view_meta.json').read_text())
    data.time=metadata['time_s'];mujoco.mj_forward(model,data)
    return model,data,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args()
    document=json.loads((args.samples/'distribution.json').read_text())
    samples={str(s['seed']):s for s in document['samples']
             if (args.samples/f"seed_{s['seed']}"/'view_meta.json').exists()}
    if str(args.seed) not in samples:raise ValueError('该旧样本没有三维快照，请重新生成预览')
    import viser
    from gene_agent.visualization.pick_bowl.scene import PickBowlScene
    server=viser.ViserServer(host='127.0.0.1',port=args.port)
    server.scene.set_up_direction('+z')
    task=document['task'];lock=threading.RLock();renderer=None
    target=(.35,.25,.25) if task=='microwave' else (.36,-.03,.08)
    front=(-1.,1.4,.95) if task=='microwave' else (-.8,.9,.7)
    server.initial_camera.position=front;server.initial_camera.look_at=target
    server.initial_camera.up=(0,0,1);server.initial_camera.fov=.8
    server.gui.configure_theme(dark_mode=False,control_width='medium',show_logo=False)
    server.gui.add_markdown('## 随机样本三维查看\n左键拖动画面旋转，右键平移，滚轮缩放。切换样本会保留视角。物体保持采样快照。')
    selection=server.gui.add_dropdown('样本 seed',options=tuple(samples),initial_value=str(args.seed))
    status=server.gui.add_markdown('')
    def show(event=None):
        nonlocal renderer
        with lock:
            model,data,meta=load_snapshot(args.samples/f'seed_{selection.value}')
            if renderer is not None:
                for handle in renderer.visual_handles.values():handle.remove()
            renderer=PickBowlScene(server,model,data,{}, {})
            renderer.update_visuals(data)
            status.content=f"**seed {selection.value}** · "+('初态检查通过；任务尚未验证' if meta['valid'] else '无效初态，保留用于检查')
    selection.on_update(show)
    def camera(event,position):
        client=event.client
        if client:
            client.camera.position=position;client.camera.look_at=target
            client.camera.up=(0,0,1);client.camera.fov=.8
    server.gui.add_button('正面视角').on_click(lambda e:camera(e,front))
    server.gui.add_button('俯视视角').on_click(lambda e:camera(e,(target[0],target[1]+.01,target[2]+1.5)))
    show()
    @server.on_client_connect
    def connect(client):
        client.camera.position=front;client.camera.look_at=target;client.camera.up=(0,0,1);client.camera.fov=.8
    print(f'样本三维查看 http://127.0.0.1:{args.port}',flush=True)
    try:threading.Event().wait()
    finally:server.stop()


if __name__=='__main__':main()
