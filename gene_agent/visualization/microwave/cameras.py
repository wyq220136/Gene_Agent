"""微波炉录像机位；只决定可见性，不包含任务或动作逻辑。"""
from __future__ import annotations

import numpy as np


def camera_parameters(viewer, frame, name):
    model, data = viewer.model, viewer.data
    if name in {'overview', 'task'}:
        return {'lookat':[.38,.20,.16], 'distance':1.08, 'azimuth':-54., 'elevation':-24.}
    if name != 'detail':
        return viewer._camera(name)[1]
    robot = viewer.robot_metadata
    if robot is None or 'right' not in robot['grippers']:
        raise ValueError('微波炉近景需要保存的右臂夹爪元数据')
    pads = [np.asarray(data.geom(n).xpos) for n in robot['grippers']['right']['pad_geom_names']]
    fingers = np.mean(pads,axis=0)
    label = frame.stage_id
    if 'push_open' in label:
        # 逐帧侧向推导在门打开末端会沿手臂轴观察，整幅近景被腕部遮挡。
        # 该机位已经用25→90°操作的接触及末端实际帧检查。
        return {'lookat':fingers.tolist(), 'distance':.5,
                'azimuth':30.,'elevation':-25.}
    if 'knob' in label:
        target=np.asarray(data.site('knob_0_site').xpos)
        return {'lookat':(.55*target+.45*fingers).tolist(), 'distance':.42,
                'azimuth':-54.,'elevation':-17.}
    if any(token in label for token in ('pick','insert','extract','place')):
        target=np.asarray(data.body('bowl').xpos)
        return {'lookat':(.55*target+.45*fingers).tolist(), 'distance':.55,
                'azimuth':-20.,'elevation':-30.}
    handle=np.asarray(data.site('microwave_handle_grasp').xpos)
    side=np.cross(pads[0]-pads[1],[0.,0.,1.]);side[2]=0.
    outward=handle-np.asarray(data.body('microwave').xpos);outward[2]=0.
    if np.linalg.norm(side)<1e-8:side=outward.copy()
    if np.dot(side,outward)<0:side=-side
    norm=np.linalg.norm(side)
    if norm<1e-8:raise ValueError('微波炉近景侧向退化')
    side/=norm;side[2]=.25;side/=np.linalg.norm(side)
    return {'lookat':(.45*handle+.55*fingers).tolist(),
            'distance':.38,
            'azimuth':float(np.degrees(np.arctan2(-side[1],-side[0]))),
            'elevation':float(-np.degrees(np.arcsin(side[2])))}
