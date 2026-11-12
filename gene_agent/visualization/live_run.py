"""订阅落盘记录的只读窗口：运行中跟随，结束后保留全部已收到前缀。"""
from pathlib import Path
import argparse
from threading import RLock
from functools import wraps
import bisect
import json
import time
import numpy as np
from gene_agent.feedback.live_record import JsonlTail


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _locked(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        with self._lock:return method(self,*args,**kwargs)
    return call


class ActualStreamApp:
    """严格显示实际 qpos；不播放规划轨迹冒充执行，也不重新运行任务。"""
    def __init__(self,server,run):
        self._lock=RLock();self.user_paused=False
        self.server=server;self.run=Path(run);self.frames=[];self.times=[];self.model_name=None
        self.tail=JsonlTail(self.run/'live/frames.jsonl');self.playing=True;self.last_tick=time.monotonic()
        self.position_s=0.;self.renderer=None;self.last_index=-1
        server.scene.set_up_direction('+z')
        server.gui.configure_theme(dark_mode=False,control_width='large',show_logo=False)
        server.gui.add_markdown('## MuJoCo 实际执行 · 在线记录\n画面来自执行期间保存的实际 qpos。**暂停浏览不停止任务**；停止任务请用工作台作业按钮。')
        self.status=server.gui.add_markdown('等待实际采样。')
        self.follow=server.gui.add_checkbox('跟随新增实际记录',initial_value=True)
        self.timeline=server.gui.add_slider('已记录时间 (s)',min=0.,max=1.,step=.01,initial_value=0.)
        self.speed=server.gui.add_slider('播放速度',min=.1,max=3.,step=.1,initial_value=1.)
        server.gui.add_button('播放 / 暂停').on_click(lambda _:self.toggle_play())
        server.gui.add_button('回到起点').on_click(lambda _:self.rewind())
        self.camera=server.gui.add_dropdown('相机',options=('全景','接触近景','夹爪侧面'))
        self.camera.on_update(lambda _:self.set_camera())
        self.timeline.on_update(lambda _:self.seek(self.timeline.value))
        @server.on_client_connect
        def connected(client):self.set_camera(client)
        self.set_camera()

    @_locked
    def set_camera(self,client=None):
        center=np.array([.44,0.,.10])
        if self.renderer is not None:
            names=('brush','bowl')
            for name in names:
                try:center=np.array(self.data.body(name).xpos);break
                except KeyError:pass
        offset={'全景':[-.75,.75,.65],'接触近景':[-.34,.38,.28],'夹爪侧面':[-.36,.22,.10]}[self.camera.value]
        if self.camera.value=='全景':center=np.array([.44,0.,.15])
        self.server.initial_camera.position=tuple(center+offset);self.server.initial_camera.look_at=tuple(center)
        for c in ([client] if client else self.server.get_clients().values()):
            c.camera.position=tuple(center+offset);c.camera.look_at=tuple(center);c.camera.up_direction=(0.,0.,1.)

    @_locked
    def toggle_play(self):
        self.playing=not self.playing;self.user_paused=not self.playing

    @_locked
    def rewind(self):
        self.follow.value=False;self.playing=False;self.user_paused=True;self.timeline.value=0.;self.seek(0.)

    @_locked
    def seek(self,at):
        if not self.frames:return
        import mujoco
        from gene_agent.visualization.pick_bowl.scene import PickBowlScene
        self.position_s=float(at);index=max(0,bisect.bisect_right(self.times,at)-1)
        row=self.frames[index]
        if row['model']!=self.model_name:
            if self.renderer is not None:
                # PickBowlScene 使用固定 /world 前缀；清空由本窗口专有的旧场景。
                self.server.scene.reset()
            self.model=mujoco.MjModel.from_binary_path(str(self.run/'live'/row['model']))
            self.data=mujoco.MjData(self.model);self.model_name=row['model']
            self.data.qpos[:]=row['qpos'];self.data.time=row['sim_time_s']
            mujoco.mj_forward(self.model,self.data)
            self.renderer=PickBowlScene(self.server,self.model,self.data,{},{});self.renderer.set_collision_visible(False)
            self.set_camera()
        if len(row['qpos'])!=self.model.nq:raise ValueError('在线实际 qpos 与冻结模型长度不一致')
        self.data.qpos[:]=row['qpos'];self.data.time=row['sim_time_s'];mujoco.mj_forward(self.model,self.data)
        self.renderer.update_visuals(self.data);self.last_index=index

    @_locked
    def poll(self,manifest):
        incoming=self.tail.read()
        for row in incoming:
            # 流的顺序是执行顺序；跨模型的仿真时间重置不被误当成可排序的新事实。
            self.times.append((self.times[-1]+max(0.,row['sim_time_s']-self.frames[-1]['sim_time_s'])
                               if self.frames else 0.))
            self.frames.append(row)
        if self.frames:
            self.timeline.max=max(.01,self.times[-1])
            if self.last_index<0:self.seek(0.)
            if incoming and self.follow.value and not self.user_paused:self.playing=True
        self.manifest=manifest
        self.status.content=f"**状态：{manifest['status']}**；已保存 {len(self.frames)} 实际采样，{self.times[-1] if self.times else 0.:.2f} s。\n任务成功：{manifest.get('task_success','运行中 / 未知')}\n失败阶段：{manifest.get('failure_stage') or '无已记录失败'}\n仿真辅助：见原报告与逐帧 attachment，不代表真机夹持。"+ ('\n**异常：** '+manifest['error'] if manifest.get('error') else '')+ ('\n在线记录错误：'+manifest['recording_error'] if manifest.get('recording_error') else '')

    @_locked
    def tick(self):
        now=time.monotonic();dt=now-self.last_tick;self.last_tick=now
        if not self.playing or not self.frames:return
        at=min(self.times[-1],self.position_s+dt*self.speed.value)
        self.timeline.value=at;self.seek(at)
        if at>=self.times[-1] and (not self.follow.value or self.manifest['status']!='running'):self.playing=False


class LiveRunApp:
    """父界面先就绪；数据发布后建立一次模型，其后更新相同时间轴。"""
    def __init__(self,server,run,output,skill=None):
        self.server=server;self.run=Path(run);self.output=Path(output);self.app=None;self.skill=skill
        self.wait=server.gui.add_markdown('## 在线查看\n等待运行初态；规划/执行与此窗口相互独立。失败和取消后的已有记录仍可回放。')
        self.signature=None;self.next_poll=0.

    def tick(self):
        now=time.monotonic()
        if now>=self.next_poll:
            self.next_poll=now+.3
            path=self.run/'live/manifest.json'
            if path.exists():
                manifest=read(path)
                job=self.run.parent/'job.json'
                if job.exists():
                    state=read(job)['state']
                    if state in {'failed','completed','canceled','interrupted'} and manifest['status'] in {'running','planning','stopped'}:
                        manifest={**manifest,'status':state}
                if manifest['kind']=='planned':
                    planpath=self.run/'live/plan.json';signature=(planpath.stat().st_mtime_ns,manifest['status'])
                    if signature!=self.signature:
                        report=read(planpath)
                        if manifest['status']!=report['status']:report={**report,'status':manifest['status']}
                        if self.app is None:
                            from gene_agent.visualization.pure_planning.app import PurePlanningApp
                            self.wait.remove();self.app=PurePlanningApp(self.server,run=self.run,output=self.output,report=report,skill=self.skill)
                        else:self.app.update_report(report)
                        self.signature=signature
                elif manifest['kind']=='actual':
                    if self.skill:raise ValueError('实际在线流目前显示完整 Task；请去掉 --skill，结束后可用原 Skill 回放筛选')
                    if self.app is None:self.wait.remove();self.app=ActualStreamApp(self.server,self.run)
                    self.app.poll(manifest)
                else:raise ValueError('未知在线记录类型')
            elif self.run.parent.joinpath('job.json').exists():
                job=read(self.run.parent/'job.json')
                if job['state'] not in {'running','queued'}:self.wait.content=f"## 运行已{job['state']}\n尚未生成轨迹或实际采样。请查看作业日志，不能制造前缀画面。"
        if self.app:self.app.tick()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skill');p.add_argument('--task')
    p.add_argument('--port',type=int,default=8093);args=p.parse_args()
    import viser
    server=viser.ViserServer(host='127.0.0.1',port=args.port)
    try:
        app=LiveRunApp(server,args.run,args.output,skill=args.skill)
        print(f'在线查看：http://127.0.0.1:{server.get_port()}；仅订阅记录，不控制任务',flush=True)
        while True:app.tick();time.sleep(.03)
    except KeyboardInterrupt:pass
    finally:server.stop()

if __name__=='__main__':main()
