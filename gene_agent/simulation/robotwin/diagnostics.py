"""开发专用的原生接触/评分诊断；不提供给policy运行接口。"""
from copy import deepcopy


class RobotwinDiagnostics:
    """只读当前接触与原生规则记录，缺失数据明确unknown。"""
    def __init__(self,native_env):
        self.__env=native_env

    def capture_failure_image(self, output, *, stage_id, state_id):
        """只读已初始化的原生相机，不get_obs/step，不把内部信息返回策略。"""
        from pathlib import Path
        import json, time
        import numpy as np
        from PIL import Image
        started=time.monotonic();env=self.__env
        env._update_render();env.cameras.update_picture()
        frames=env.cameras.get_rgb()
        name='head_camera' if 'head_camera' in frames else next(iter(frames))
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.asarray(frames[name]['rgb'],dtype=np.uint8)).save(output)
        cameras=[]
        for camera in env.cameras.static_camera_list:
            if camera.name==name:
                cameras.append(dict(name=name,width=camera.width,height=camera.height,
                    intrinsic_cv=camera.get_intrinsic_matrix().tolist(),extrinsic_cv=camera.get_extrinsic_matrix().tolist()))
        row=dict(mode='actual',status='available',stage_id=stage_id,state_id=state_id,path=str(output),
                 source='native_readonly_render',camera_name=name,cameras=cameras,elapsed_s=time.monotonic()-started)
        output.with_suffix('.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
        return row

    def capture(self, *, stage_id, state_id):
        """接触为力冲量N*s；不编造原引擎未提供的力值。"""
        row=dict(stage_id=stage_id,state_id=state_id,source='robotwin_native_diagnostics',hypotheses=[])
        try:
            contacts=[]
            for pair in self.__env.scene.get_contacts():
                for point in pair.points:
                    contacts.append(dict(body_names=[b.entity.name for b in pair.bodies],
                        position_world_m=point.position.tolist(),normal_world=point.normal.tolist(),
                        impulse_Ns=point.impulse.tolist(),separation_m=float(point.separation)))
            row['contacts']=dict(status='available',values=contacts)
        except Exception as exc:
            row['contacts']=dict(status='unknown',error=f'{type(exc).__name__}: {exc}')
        row['observed_facts']=deepcopy({k:self.__env.info[k] for k in ('nail','tool_identity','task_setup') if k in self.__env.info})
        return row


def native_episode_identity(native_env):
    """只供记录器关联原案例/视频；seed绝不传入RuntimePort或Skill参数。"""
    from pathlib import Path
    directory=getattr(native_env,'eval_video_path',None)
    counter=getattr(native_env,'test_num',None)
    return dict(native_episode_ordinal=getattr(native_env,'ep_num',None),
                native_seed=getattr(native_env,'guidance_seed',None),
                native_video_ref=str(Path(directory)/f'episode{counter}.mp4') if directory is not None and counter is not None else None)
