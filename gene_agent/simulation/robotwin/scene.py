"""从 SAPIEN 实际接触几何导出 cuRobo 场景；不需要 MuJoCo 中转。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.robots.aloha import make_aloha_gripper_snapshot, digest
from .runtime import pose_matrix


def _pose_list(matrix):
    """cuRobo obstacle pose 使用 xyz+wxyz。"""
    q=Rotation.from_matrix(matrix[:3,:3]).as_quat()
    return [*matrix[:3,3].tolist(),*q[[3,0,1,2]].tolist()]


class RobotwinSceneProvider:
    """组合根显式提供原环境；policy只得到RuntimePort，不得到本对象的env。"""
    def __init__(self,native_env,runtime,*,robot_metadata,output_dir, object_bodies=None):
        self.__object_bodies=dict(object_bodies or {})
        self.__env=native_env
        self.__runtime=runtime
        self.__robot=robot_metadata
        self.__output=Path(output_dir)
        self.__output.mkdir(parents=True,exist_ok=True)
        self.__visual_cache={}

    def snapshot(self, *, state, group, collision_policy, held_object=None):
        """当前组由源碰撞球检查，其余机器人和场景几何全部进入world。"""
        current=self.__runtime.observe(object_names=tuple(state.values.get('objects',{})),robot_groups=tuple(state.values.get('robots',{})))
        if current.state_id!=state.state_id:
            raise ValueError('规划场景请求state已过期')
        finger_contact=(held_object is None and collision_policy.policy_id=='grasp_finger_contact')
        if collision_policy.allowances and held_object is None and not finger_contact:
            raise ValueError('接触例外必须绑定明确持物link或声明指部抓取接触')
        held_body=None;attachment=None;transferred={}
        if held_object is not None:
            from gene_agent.knowledge.geometry.transforms import validate_pose
            oid=held_object['object_id'];held_body=self.__object_bodies.get(oid,oid)
            link_from_object=validate_pose(held_object['T_link_object'])
            attachment=dict(spheres_link=[],finger_links=[],object_id=oid)

        robot=self.__robot
        base_from_world=np.linalg.inv(np.asarray(robot['transforms'][group]['world_from_urdf_base']))
        active=set(robot['planning_assets'][group]['robot_config']['kinematics']['collision_link_names'])
        scene={'cuboid':{},'mesh':{}};records=[]
        if attachment is not None:
            import xml.etree.ElementTree as ET
            urdf=robot['planning_assets'][group]['robot_config']['kinematics'].get('urdf_path')
            if urdf:
                joints={j.get('name'):j for j in ET.parse(urdf).getroot().findall('joint')}
                attachment['finger_links']=[joints[n].find('child').get('link') for n in robot['grippers'][group]['joint_names']]

        for entity in self.__env.scene.entities:
            if entity.name in active:continue
            for ci,component in enumerate(entity.components):
                if not hasattr(component,'get_collision_shapes'):continue
                world_from_body=pose_matrix(component.get_entity_pose())
                for si,shape in enumerate(component.get_collision_shapes()):
                    name=f'{entity.name}__{ci}_{si}'
                    transform=base_from_world@world_from_body@pose_matrix(shape.local_pose)
                    row=dict(obstacle_name=name,body_name=entity.name,source='sapien_collision_shape',
                             shape_type=type(shape).__name__,approximation='original_collision')
                    if entity.name==held_body:
                        from gene_agent.motion.curobo.gripper import cover_bounds
                        if hasattr(shape,'half_size'):
                            bounds=np.array([-np.asarray(shape.half_size),np.asarray(shape.half_size)])
                        else:
                            mesh=self._mesh(shape)
                            if mesh is None:raise ValueError('不能抓持plane')
                            bounds=mesh.bounds
                        # 每个原convex shape的AABB小格外接球完整覆盖，不删细小部件。
                        local=link_from_object@pose_matrix(shape.local_pose)
                        for ball in cover_bounds(bounds,max_cell_m=.016):
                            center=local[:3,:3]@ball['center']+local[:3,3]
                            attachment['spheres_link'].append([*center.tolist(),ball['radius']])
                            attachment.setdefault('sphere_sources',[]).append(dict(source_body=held_body,source_shape_id=name))
                        row['transferred_to']='gene_held_tool';records.append(row)
                        transferred[held_object['object_id']]=dict(source_body=held_body,link='gene_held_tool')
                        continue
                    if hasattr(shape,'half_size'):

                        scene['cuboid'][name]=dict(dims=(2*np.asarray(shape.half_size)).tolist(),pose=_pose_list(transform))
                    else:
                        mesh=self._mesh(shape)
                        if 'Cylinder' in type(shape).__name__:
                            row['approximation']='circumscribed_cylinder_mesh_32sides'
                        if mesh is None:
                            # PhysX plane 的法向为局部X；以厚2cm的下半空间板覆盖桌面附近工作区。
                            center=np.eye(4);center[0,3]=-.01
                            scene['cuboid'][name]=dict(dims=[.02,100.,100.],pose=_pose_list(transform@center))
                            row['approximation']='plane_workspace_slab_100m'
                        else:
                            payload=mesh.export(file_type='obj').encode()
                            sha=hashlib.sha256(payload).hexdigest()
                            path=self.__output/f'{sha}.obj'
                            if not path.exists():path.write_bytes(payload)
                            scene['mesh'][name]=dict(file_path=str(path.resolve()),pose=_pose_list(transform),scale=[1.,1.,1.])
                            row['mesh_sha256']=sha
                    records.append(row)
        contact_policy={}
        if attachment is not None and not attachment['spheres_link']:
            raise ValueError(f'持物对象没有导出几何: {held_body}')
        if collision_policy.allowances:
            bodies={r['body_name'] for r in records if 'transferred_to' not in r}
            allowed=set();contact_links=set()
            valid_links=set(robot['grippers'][group].get('contact_links',())) if finger_contact else {'gene_held_tool'}
            for pair in collision_policy.allowances:
                body=self.__object_bodies.get(pair.second,pair.second) if finger_contact else pair.second
                if pair.first not in valid_links or body not in bodies:
                    raise ValueError(f'不支持或未知的接触对: {pair.first}, {pair.second}')
                allowed.add(body);contact_links.add(pair.first)
            # validator按这个link集合×唯一目标复核，工具仍完整保留在world中。
            if finger_contact and len(allowed)!=1:raise ValueError('一次抓取接触只能绑定一个工具')
            contact_policy=dict(robot_links=sorted(contact_links),obstacle_names=[r['obstacle_name'] for r in records if r['body_name'] in allowed],
                                reason=collision_policy.to_dict())
        gripper=make_aloha_gripper_snapshot(robot,group,state.values['robots'][group]['gripper']['joint_positions_m'])
        payload=dict(scene_model=scene,records=records,gripper_snapshot=gripper,collision_mode='full',
                     scope='原生全场景碰撞 + 本体/冻结非活动臂 + 当前组源碰撞球',active_group=group,
                     world_frame=robot.get('world_frame',state.values['world_frame']),base_from_world=base_from_world.tolist(),
                     episode_id=state.values['episode_id'],state_id=state.state_id,sim_time_s=state.values['sim_time_s'],
                     excluded=[],transferred_objects=transferred,mounting_contact_policy=contact_policy)
        if attachment is not None:payload['attachment']=attachment
        from .visual_snapshot import save_visual_snapshot
        try:
            payload['visual_snapshot']=save_visual_snapshot(self.__env,robot,group,self.__output,self.__visual_cache,
                state_id=state.state_id,held_object=held_object,object_bodies=self.__object_bodies)
        except Exception as exc:
            payload['visual_snapshot']=dict(state_id=state.state_id,objects=[],warnings=[f'Visual capture failed: {type(exc).__name__}: {exc}'])
        payload['snapshot_sha256']=digest(payload)
        after=self.__runtime.observe(object_names=tuple(state.values.get('objects',{})),robot_groups=tuple(state.values.get('robots',{})))
        if after.state_id!=state.state_id:raise ValueError('场景导出过程中state改变')
        (self.__output/f'{payload["snapshot_sha256"]}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
        return payload

    @staticmethod
    def _mesh(shape):
        """保留原三角形；首版仅支持实测到的mesh/box/plane/cylinder，其他形状明确拒绝。"""
        import trimesh
        name=type(shape).__name__
        if hasattr(shape,'vertices'):
            vertices=np.asarray(shape.vertices)*np.asarray(getattr(shape,'scale',[1,1,1]))
            return trimesh.Trimesh(vertices=vertices,faces=np.asarray(shape.triangles).reshape(-1,3),process=False)
        if 'Plane' in name:return None
        if 'Cylinder' in name:
            # PhysX轴为局部X；外接32边形各侧面与原圆柱相切，不能用内接圆柱漏碰撞。
            mesh=trimesh.creation.cylinder(radius=shape.radius/np.cos(np.pi/32),height=2*shape.half_length,sections=32)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
            return mesh
        raise ValueError(f'未支持的原生碰撞几何: {name}')
