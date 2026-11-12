"""规划worker的窄JSONL协议；stdout仅供协议，依赖日志进入stderr。"""
from contextlib import redirect_stdout
import importlib.util
import json
import os
import sys
import traceback


def _plain(value):
    """已有MotionResult中的数组显式序列化，不截断原诊断字段。"""
    if hasattr(value,'tolist'):return value.tolist()
    raise TypeError(f'不可序列化的规划字段: {type(value).__name__}')


def main():
    """一个worker维护一个模型的有界planner pool，关闭时释放。"""
    session=None;identity=None
    try:
        for line in sys.stdin:
            request=json.loads(line)
            if request['operation']=='close':break
            try:
                with redirect_stdout(sys.stderr):
                    if request['operation']=='describe':
                        import curobo
                        result=dict(protocol_version=1,pid=os.getpid(),python=sys.executable,
                            curobo_path=curobo.__file__,api_available=importlib.util.find_spec('curobo.kinematics') is not None,
                            loaded_engines=[n for n in ('mujoco','sapien') if n in sys.modules])
                    else:
                        from .session import CuroboPlanningSession
                        from ..model import JointState,PoseTarget,CollisionPolicy,CollisionAllowance
                        current=json.dumps(request['robot'],sort_keys=True)
                        if identity!=current:
                            if session:session.close()
                            session=CuroboPlanningSession(request['robot']);identity=current
                        raw=request['start'];start=JointState(tuple(raw['joint_names']),tuple(raw['positions_rad']))
                        cp=request.get('collision_policy')
                        policy=CollisionPolicy(cp['policy_id'],tuple(CollisionAllowance(**a) for a in cp['allowances'])) if cp else None
                        with session.stage(snapshot=request['snapshot'],start=start,group=request['group'],
                                scene_revision=request['scene_revision'],collision_policy=policy,**request.get('options',{})) as motion:
                            operation=request['operation']
                            if operation=='fk':result=motion.fk()
                            elif operation=='evaluate_ik':
                                targets=[PoseTarget(t['frame'],t['matrix']) for t in request['targets']]
                                result=[r.to_dict() for r in motion.evaluate_ik(targets,request_ids=request['request_ids'])]
                            else:
                                target=request['target'];target=PoseTarget(target['frame'],target['matrix'])
                                result=getattr(motion,operation)(target,request_id=request['request_id'],**request.get('request_options',{})).to_dict()
                response={'ok':True,'result':result}
            except Exception as error:
                response=dict(ok=False,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
            print(json.dumps(response,default=_plain,allow_nan=False),flush=True)
    finally:
        if session:session.close()


if __name__=='__main__':main()
