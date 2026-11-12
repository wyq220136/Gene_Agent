"""URDF 夹爪目标与路径绘制；只用于显示，不引入物理引擎。"""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from gene_agent.scene.visual_inputs import _geometry_rows


def pose_args(T):
    return dict(position=T[:3,3],wxyz=Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]])


def load_geometry(row):
    if row.get('mesh_path'):
        mesh=trimesh.load_mesh(row['mesh_path'],process=False)
        if isinstance(mesh,trimesh.Scene):mesh=mesh.to_geometry()
        mesh=mesh.copy();mesh.vertices*=np.asarray(row.get('scale',[1,1,1]))
        return mesh
    return trimesh.creation.box(extents=row['size_xyz_m'])


class TargetGripper:
    """使用与机器人同一 URDF 的夹爪外观，金色表示目标而非已到达。"""
    def __init__(self,server):
        self.server=server;self.handles=[];self.identity=None
        self.root=server.scene.add_frame('/target_gripper',show_axes=False,visible=False)

    def select(self,robot,pose,width):
        identity=(robot.group,round(float(width),5))
        if identity!=self.identity:
            for h in self.handles:h.remove()
            self.handles=[]
            saved=dict(robot.joint_configuration)
            try:
                # 夹爪自身的局部几何由源 URDF FK 得到，不把150/180mm等值硬编码到显示。
                robot.urdf.update_cfg({**saved,**robot.gripper_configuration(width)})
                tcp_local=robot.urdf.get_transform('link_7')@robot.link_from_tcp
                for i,row in enumerate(_geometry_rows(robot.urdf,robot.urdf_path)):
                    if not any(k in row['link_name'].lower() for k in ('gripper','finger','jaw','pika')):continue
                    mesh=load_geometry(row)
                    mesh.visual.face_colors=np.tile([245,190,55,135],(len(mesh.faces),1))
                    handle=self.server.scene.add_mesh_trimesh(f'/target_gripper/{i}',mesh,
                        **pose_args(np.linalg.inv(tcp_local)@row['T_object_visual']))
                    self.handles.append(handle)
            finally:robot.urdf.update_cfg(saved)
            self.identity=identity
        self.root.visible=True
        for k,v in pose_args(pose).items():setattr(self.root,k,v)

    def hide(self):self.root.visible=False


def line_segments(server,name,points,color=(65,139,206),width=3.):
    points=np.asarray(points,float)
    if len(points)<2:return None
    return server.scene.add_line_segments(name,points=np.stack([points[:-1],points[1:]],axis=1),
                                         colors=color,line_width=width)
