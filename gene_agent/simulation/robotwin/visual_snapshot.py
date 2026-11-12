"""原生渲染几何的只读快照；与规划碰撞资产分离，失败不影响正式规划。"""
from pathlib import Path
import hashlib
import numpy as np
from .runtime import pose_matrix


def save_visual_snapshot(env, robot, group, output, cache, *, state_id, held_object=None, object_bodies=None):
    import trimesh
    from PIL import Image
    output = Path(output) / 'visual'
    output.mkdir(parents=True, exist_ok=True)
    rows, warnings = [], []
    for entity in env.scene.entities:
        for ci, component in enumerate(entity.components):
            if not hasattr(component, 'render_shapes'):
                continue
            for si, shape in enumerate(component.render_shapes):
                name = f'{entity.name}__visual_{ci}_{si}'
                try:
                    # 缓存持有shape引用，避免id复用；几何在一个provider生命周期内不变。
                    key = id(shape)
                    cached = cache.get(key)
                    if cached is None:
                        scene = trimesh.Scene()
                        parts = list(shape.parts) if not hasattr(shape, 'vertices') else [shape]
                        for pi, part in enumerate(parts):
                            mesh = trimesh.Trimesh(vertices=np.asarray(part.vertices), faces=np.asarray(part.triangles), process=False)
                            # SAPIEN box/sphere/cylinder返回单位网格，capsule返回已缩放顶点。
                            kind=type(shape).__name__
                            scale=getattr(shape, 'scale', [1., 1., 1.])
                            if kind=='RenderShapeBox':scale=shape.half_size
                            elif kind=='RenderShapeSphere':scale=[shape.radius]*3
                            elif kind=='RenderShapeCylinder':scale=[shape.half_length,shape.radius,shape.radius]
                            mesh.apply_scale(scale)
                            material = part.material
                            texture = getattr(material, 'base_color_texture', None)
                            image = Image.fromarray(np.asarray(texture.download())) if texture is not None else None
                            color = np.asarray(material.base_color).copy()
                            color[3] = 1.  # 显示要求不透明；不改变仿真材质。
                            pbr = trimesh.visual.material.PBRMaterial(baseColorFactor=color, baseColorTexture=image,
                                metallicFactor=float(material.metallic), roughnessFactor=float(material.roughness), alphaMode='OPAQUE')
                            mesh.visual = trimesh.visual.TextureVisuals(uv=np.asarray(part.vertex_uv), material=pbr)
                            scene.add_geometry(mesh, node_name=f'part_{pi}')
                        payload = scene.export(file_type='glb')
                        sha = hashlib.sha256(payload).hexdigest()
                        path = output / f'{sha}.glb'
                        if not path.exists():
                            path.write_bytes(payload)
                        cached = (shape, str(path.resolve()), sha)
                        cache[key] = cached
                    rows.append(dict(name=name, body_name=entity.name, file_path=cached[1], sha256=cached[2],
                        world_from_body=pose_matrix(entity.pose).tolist(), body_from_visual=pose_matrix(shape.local_pose).tolist(),
                        source_asset=getattr(shape, 'filename', None)))
                except Exception as exc:
                    warnings.append(f'{name}: {type(exc).__name__}: {exc}')
    manifest = dict(schema_version=1, state_id=state_id, objects=rows, warnings=warnings)
    try:
        kin = robot['planning_assets'][group]['robot_config']['kinematics']
        path = Path(kin['urdf_path'])
        # 原生articulation提供完整关节状态（包括夹爪/另一臂），不从setting推测。
        joints = {}
        for articulation in env.scene.get_all_articulations():
            joints.update((j.name, float(q)) for j, q in zip(articulation.get_active_joints(), articulation.get_qpos()))
        import xml.etree.ElementTree as ET
        root=ET.parse(path).getroot()
        links={kin['base_link']}
        while True:
            children={j.find('child').get('link') for j in root.findall('joint') if j.find('parent').get('link') in links}
            if children<=links:break
            links.update(children)
        required=[j.get('name') for j in root.findall('joint') if j.find('child').get('link') in links and j.get('type')!='fixed']
        manifest['active_robot'] = dict(urdf_path=str(path.resolve()), urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            links=sorted(links),base_link=kin['base_link'],required_joints=required,
            joint_positions=joints, world_from_base=robot['transforms'][group]['world_from_urdf_base'])
        if held_object:
            manifest['attachment'] = dict(body_name=(object_bodies or {}).get(held_object['object_id'], held_object['object_id']),
                link=robot['tool_links'][group], T_link_object=held_object['T_link_object'])
    except Exception as exc:
        warnings.append(f'Robot visual state unavailable: {type(exc).__name__}: {exc}')
    return manifest
