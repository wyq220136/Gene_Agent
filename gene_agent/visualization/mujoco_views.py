"""从同一物理状态渲染多个固定相机；不推进仿真。"""
from pathlib import Path
import json
import mujoco
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from gene_agent.feedback.state import state_id
from .style import apply_scene_style, camera_presets, configure_camera
from .frame_axes import append_frame_axes


COLLISION_PALETTE = {
    'robot': [1.0, 0.32, 0.04, 0.78],
    'object': [0.18, 0.82, 0.30, 0.78],
    'fixture': [0.10, 0.52, 0.98, 0.72],
    'environment': [0.62, 0.66, 0.72, 0.62],
}


def _collision_category(model, geom_id):
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or 'world'
    if body_name == 'robot_body' or body_name in {'base_link', *(f'link_{i}' for i in range(1,8))} or body_name.startswith(('left_', 'right_', 'pika_gripper_')):
        return 'robot'
    if body_name.startswith('microwave') or body_name in {'knob_0', 'knob_1'}:
        return 'fixture'
    if body_id == 0:
        return 'environment'
    return 'object'


def apply_collision_palette(model):
    result = {}
    for geom_id in range(model.ngeom):
        if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
            continue
        category = _collision_category(model, geom_id)
        model.geom_rgba[geom_id] = COLLISION_PALETTE[category]
        entry = result.setdefault(category, {
            'rgba': list(COLLISION_PALETTE[category]), 'geom_names': [], 'body_names': [],
        })
        entry['geom_names'].append(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f'unnamed_{geom_id}'
        )
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        ) or 'world'
        if body_name not in entry['body_names']:
            entry['body_names'].append(body_name)
    return {name: result[name] for name in COLLISION_PALETTE if name in result}


def _add_collision_legend(pixels, palette):
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image, 'RGBA')
    row_height, padding = 22, 8
    width = 142
    height = padding * 2 + row_height * len(palette)
    draw.rounded_rectangle((8, 8, 8 + width, 8 + height), radius=5, fill=(0, 0, 0, 165))
    for index, (name, entry) in enumerate(palette.items()):
        y = 8 + padding + index * row_height
        rgb = tuple(round(channel * 255) for channel in entry['rgba'][:3])
        draw.rectangle((16, y + 3, 30, y + 17), fill=(*rgb, 255))
        draw.text((38, y + 2), name, fill=(255, 255, 255, 255))
    return np.asarray(image)


def _diagnostic_panel(pixels, *, palette, frames, diagnosis, identifier):
    """把文字放到场景右侧；图像左侧像素保持原始渲染结果。"""
    height, width = pixels.shape[:2]
    panel_width = 300
    panel_height = max(height, 560 if diagnosis and 'target_minus_tcp_world_m' in diagnosis else height)
    canvas = Image.new('RGB', (width + panel_width, panel_height), (25, 31, 42))
    canvas.paste(Image.fromarray(pixels), (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
        small = ImageFont.truetype('DejaVuSans.ttf', 12)
    except OSError:
        font = small = ImageFont.load_default()
    x, y = width + 14, 12
    def line(value, *, color=(235, 238, 243), compact=False):
        nonlocal y
        draw.text((x, y), value, fill=color, font=small if compact else font)
        y += 20 if compact else 25
    line('MuJoCo debug view')
    line(f'state {identifier[:12]}', compact=True)
    y += 10
    if frames:
        line('Frame origins (if visible):')
        line(', '.join(frames), compact=True)
        if diagnosis.get('world_frame'):
            line('WORLD = ' + diagnosis['world_frame'], compact=True)
        line('X red   Y green   Z blue', compact=True)
        line('TCP pink   TARGET yellow', compact=True)
        line('Arrows show local +axes', compact=True)
        y += 10
    if diagnosis:
        def vector(title, field):
            if field in diagnosis:
                values = np.asarray(diagnosis[field], dtype=float) * 1000
                line(title, compact=True)
                line('x {:+.1f}  y {:+.1f}  z {:+.1f} mm'.format(*values), compact=True)
        line('Position deltas:')
        line('Object center - actual TCP', compact=True)
        vector('in WORLD:', 'object_minus_tcp_world_m')
        vector('in TCP:', 'object_minus_tcp_tcp_m')
        if 'target_minus_tcp_world_m' in diagnosis:
            y += 6
            line('Target origin - actual TCP', compact=True)
            vector('in WORLD:', 'target_minus_tcp_world_m')
            vector('in TCP:', 'target_minus_tcp_tcp_m')
        y += 8
    if palette:
        line('Collision geoms:')
        for name, entry in palette.items():
            rgb = tuple(round(channel * 255) for channel in entry['rgba'][:3])
            draw.rectangle((x, y + 2, x + 13, y + 15), fill=rgb)
            draw.text((x + 22, y), name, fill=(235, 238, 243), font=small)
            y += 20
    return np.asarray(canvas), {'scene_bounds_px': [0, 0, width, height],
                                'panel_bounds_px': [width, 0, width + panel_width, panel_height]}


def render_views(model, data, output_dir, target_body='bowl', width=960, height=720, cameras=None,
                 collision_overlay=False, site_axes=False, view_names=None,
                 contact_debug=False, debug_frames=None, frame_axis_length_m=.08,
                 frame_diagnostics=None, robot_metadata=None, active_group=None):
    """相机距离单位为米，角度为度；cameras 按视角名覆盖固定参数。

    调用方需在本函数复制 data 时停止写入。复制后所有渲染独立于原 data；
    不在模块导入时设置 MUJOCO_GL，离屏进程应自行选择 egl。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = mujoco.MjData(model)
    mujoco.mj_copyData(frozen, model, data)
    identifier = state_id(model, frozen)
    if frame_diagnostics is not None and frame_diagnostics.get('state_id') != identifier:
        raise ValueError('frame_diagnostics state_id 与渲染状态不一致')
    mujoco.mj_forward(model, frozen)
    presets = camera_presets(
        model, frozen, target_body, robot_metadata=robot_metadata,
        active_group=active_group,
    )
    selected = tuple(view_names) if view_names is not None else ('overview', 'top', 'side', 'contact')
    unknown = set(selected) - set(presets)
    if unknown:
        raise ValueError(f'所请求的相机不可用: {sorted(unknown)}')
    presets = {name: presets[name] for name in selected}
    for name, override in (cameras or {}).items():
        if name not in presets:
            raise ValueError(f'Unknown camera: {name}')
        if set(override) - {'lookat', 'distance', 'azimuth', 'elevation'}:
            raise ValueError(f'Unknown camera parameters: {name}')
        presets[name].update(override)
    result = {'schema_version': 1, 'state_id': identifier, 'sim_time': float(frozen.time), 'views': {}}
    # MuJoCo 默认离屏 framebuffer 可能小于请求尺寸；临时扩展后还原模型设置。
    old_width, old_height = model.vis.global_.offwidth, model.vis.global_.offheight
    old_contact_width = model.vis.scale.contactwidth
    old_contact_height = model.vis.scale.contactheight
    original_rgba = model.geom_rgba.copy()
    try:
        model.vis.global_.offwidth = max(old_width, width)
        model.vis.global_.offheight = max(old_height, height)
        if contact_debug:
            # MuJoCo 默认 marker 对桌面物体过大；缩小后仍可定位且不会遮住接触几何。
            model.vis.scale.contactwidth = .1
            model.vis.scale.contactheight = .04
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for name, params in presets.items():
                camera = configure_camera(mujoco.MjvCamera(), params)
                renderer.update_scene(frozen, camera=camera)
                Image.fromarray(renderer.render()).save(output_dir / f'{name}.png')
                result['views'][name] = {'path': f'{name}.png', 'camera': params,
                                         'state_id': identifier, 'sim_time': float(frozen.time),
                                         'width': width, 'height': height}
                needs_overlay = collision_overlay or site_axes or debug_frames or frame_diagnostics or (contact_debug and name in ('contact', 'contact_side'))
                if needs_overlay:
                    option = mujoco.MjvOption()
                    if contact_debug and name in ('contact', 'contact_side'):
                        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                    if collision_overlay:
                        # 仅改变显示颜色，不改变 contype/conaffinity 或物理几何。
                        # group 1/2 是 visual；降低不透明度使显示与接触代理边界都可辨。
                        visual = np.isin(model.geom_group, [1, 2])
                        model.geom_rgba[visual, 3] = .18
                        palette = apply_collision_palette(model)
                        option.geomgroup[3] = 1
                    else:
                        palette = {}
                    if site_axes:
                        option.frame = mujoco.mjtFrame.mjFRAME_SITE
                    renderer.update_scene(frozen, camera=camera, scene_option=option)
                    frame_metadata = (
                        append_frame_axes(
                            renderer.scene, debug_frames,
                            axis_length_m=frame_axis_length_m,
                        )
                        if debug_frames else None
                    )
                    overlay_name = f'{name}_overlay'
                    overlay_pixels = renderer.render()
                    image_layout = None
                    if debug_frames or frame_diagnostics:
                        overlay_pixels, image_layout = _diagnostic_panel(
                            overlay_pixels, palette=palette,
                            frames=tuple(debug_frames or ()), diagnosis=frame_diagnostics,
                            identifier=identifier,
                        )
                    elif collision_overlay:
                        overlay_pixels = _add_collision_legend(overlay_pixels, palette)
                    Image.fromarray(overlay_pixels).save(output_dir / f'{overlay_name}.png')
                    result['views'][overlay_name] = {
                        **result['views'][name], 'path': f'{overlay_name}.png',
                        'width': int(overlay_pixels.shape[1]),
                        'height': int(overlay_pixels.shape[0]),
                        'collision_overlay': bool(collision_overlay), 'site_axes': bool(site_axes),
                        'contact_points': bool(contact_debug and name in ('contact', 'contact_side')),
                        'contact_forces': False,
                        'collision_palette': palette,
                        'legend': bool(collision_overlay),
                        'debug_frames': frame_metadata,
                        'frame_diagnostics': {
                            key: frame_diagnostics[key] for key in (
                                'object_minus_tcp_world_m', 'object_minus_tcp_tcp_m',
                                'target_minus_tcp_world_m', 'target_minus_tcp_tcp_m',
                            ) if key in frame_diagnostics
                        } if frame_diagnostics else None,
                        'image_layout': image_layout,
                        'site_names': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or f'unnamed_{i}'
                                       for i in range(model.nsite)] if site_axes else [],
                    }
                    model.geom_rgba[:] = original_rgba
    finally:
        model.geom_rgba[:] = original_rgba
        model.vis.global_.offwidth, model.vis.global_.offheight = old_width, old_height
        model.vis.scale.contactwidth = old_contact_width
        model.vis.scale.contactheight = old_contact_height
    (output_dir / 'views.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


from .viewer import open_viewer
