"""基座XYZ裁剪检查：Open3D交互、RGB覆盖及可复现的无界面快照。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from ..storage import ROOT, resolve_calibration_path
from .geometry import (crop_mask, load_crop_config, load_frame, make_frame, overlay_rgb,
                       plane_diagnostics, validate_crop)

LIMITATION = '桌面平面和裁剪覆盖仅用于诊断；单平面不能验证XY平移及绕Z旋转，不能据此宣称绝对标定准确。'
DEFAULT_CROP = {'min': [-.5, -.6, -.05], 'max': [.9, .6, .10]}


def display_points(frame, kept, *, maximum=18000):
    """只抽稀绘图，不能将抽稀后的结果用于覆盖率/统计。"""
    valid = np.flatnonzero(frame.valid)
    kept_indices = np.flatnonzero(kept)
    rng = np.random.default_rng(0)
    shown = rng.choice(valid, min(maximum, len(valid)), replace=False)
    selected = rng.choice(kept_indices, min(maximum, len(kept_indices)), replace=False)
    xyz = frame.points.reshape(-1, 3)
    rgb = frame.rgb.reshape(-1, 3).astype(float)/255
    colors = rgb[shown].copy()
    colors[~kept.ravel()[shown]] = [.60, .60, .60]
    return xyz[shown], colors, xyz[selected], rgb[selected]


def box_edges(crop):
    lo, hi = validate_crop(crop)
    corners = np.array(list(product(*zip(lo, hi))))
    edges = [[i, j] for i in range(8) for j in range(i+1, 8) if np.count_nonzero(corners[i] != corners[j]) == 1]
    return corners, edges


def diagnostics(frame, crop, table_z):
    kept = crop_mask(frame.points, frame.valid, crop)
    total, count = int(frame.valid.sum()), int(kept.sum())
    return kept, {'coordinate_frame': f"robot_base:{frame.calibration.get('robot_group', 'unspecified')}",
        'units': 'm', 'crop': crop, 'source': frame.source,
        'camera_parameters': str(frame.calibration_path), 'camera_parameters_sha256': frame.calibration_sha256,
        'camera_serial': frame.calibration['serial'], 'camera_depth_limit_m': frame.max_depth,
        'valid_points_in_depth_range': total, 'retained_points': count,
        'retained_fraction': count/total, 'outside_xyz_crop_points': total-count,
        'depth_invalid_or_out_of_range_pixels': int(frame.valid.size-total),
        'statistics_use_full_resolution': True,
        'table_plane_candidate': plane_diagnostics(frame.points[kept], table_z=table_z),
        'limitation': LIMITATION}


def snapshot(frame, crop, path, report):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    kept = crop_mask(frame.points, frame.valid, crop)
    before, bc, after, ac = display_points(frame, kept, maximum=10000)
    fig = Figure(figsize=(14, 10), layout='constrained')
    FigureCanvasAgg(fig)
    corners, edges = box_edges(crop)
    combined = np.r_[before, corners, np.zeros((1, 3))]
    lo, hi = np.min(combined, axis=0), np.max(combined, axis=0)
    center = (lo+hi)/2
    radius = max(float((hi-lo).max())*.55, .15)
    for index, (points, colors, title) in enumerate(((before, bc, 'Before: kept RGB / removed gray'),
                                                    (after, ac, 'After: SDK base XYZ crop'))):
        ax = fig.add_subplot(2, 2, index+1, projection='3d')
        if len(points):
            ax.scatter(*points.T, c=colors, s=.5, rasterized=True)
        for i, j in edges:
            ax.plot(*corners[[i, j]].T, color='orange', linewidth=1)
        for a, color in enumerate(('red', 'green', 'blue')):
            end = np.eye(3)[a]*.15
            ax.plot([0, end[0]], [0, end[1]], [0, end[2]], color=color, linewidth=2)
            ax.text(*end, 'XYZ'[a], color=color)
        ax.set(xlabel='Base X (m)', ylabel='Base Y (m)', zlabel='Base Z (m)', title=title,
               xlim=[center[0]-radius, center[0]+radius], ylim=[center[1]-radius, center[1]+radius],
               zlim=[center[2]-radius, center[2]+radius])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(35, -110)
    for index, (image, title) in enumerate(((frame.rgb, 'Original RGB'),
            (overlay_rgb(frame, kept), 'Same pixel mask: removed dark / invalid or depth-cut purple'))):
        ax = fig.add_subplot(2, 2, index+3)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
    plane = report['table_plane_candidate']
    detail = (f"plane tilt {plane['tilt_deg']:.2f} deg | median base Z {plane['median_height_m']:.4f} m"
              if plane['available'] else 'No reliable near-horizontal plane candidate')
    fig.suptitle(f"SDK base crop | {report['retained_points']:,}/{report['valid_points_in_depth_range']:,} points | {detail}\n"
                 'Visual / plane diagnostics only; not absolute calibration accuracy', fontsize=12)
    fig.savefig(path, dpi=120)


def save_inspection(frame, crop, output, *, table_z=None):
    output = Path(output)
    if hashlib.sha256(frame.calibration_path.read_bytes()).hexdigest() != frame.calibration_sha256:
        raise ValueError('检查期间标定文件发生变化，请重新加载再保存')
    kept, report = diagnostics(frame, crop, table_z)
    if not kept.any():
        raise ValueError('裁剪范围内没有点，不能保存检查配置')
    output.mkdir(parents=True, exist_ok=True)
    config = {'format': 'pika_point_cloud_processing', 'schema_version': 1, 'units': 'm',
        'coordinate_frame': report['coordinate_frame'], 'camera_serial': report['camera_serial'],
        'camera_parameters_sha256': frame.calibration_sha256, 'crop': crop,
        'sampling': {'method': 'random', 'num_points': 2048, 'seed': 0, 'insufficient_points': 'keep_all_then_repeat'},
        'saved_at': datetime.now(timezone.utc).isoformat(), 'source': frame.source,
        'note': '2048点采样字段供下游使用；本检查的统计和RGB覆盖始终使用完整像素。'}
    # 实时单帧需保存RGB-D才能复现；已有会话原图仅引用，不复制整轮采集。
    if not Path(frame.source).is_file():
        np.savez_compressed(output/'capture.npz', rgb=frame.rgb, depth_m=frame.depth, K=frame.K)
        report['saved_capture'] = 'capture.npz'
    for name, payload in [('point_cloud_processing.json', config), ('inspection_report.json', report)]:
        (output/name).write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    snapshot(frame, crop, output/'inspection.png', report)
    print(f'检查记录：{output.resolve()}\n{LIMITATION}', flush=True)
    return report


class CropApp:
    """Open3D原生界面，无需额外PyQt或外部数据集框架。"""
    def __init__(self, frame, crop, output, table_z=None):
        import open3d as o3d
        from open3d.visualization import gui, rendering
        self.frame, self.crop, self.output, self.table_z = frame, crop, output, table_z
        self.syncing = False
        self.gui, self.o3d = gui, o3d
        self.window = gui.Application.instance.create_window('Calibration inspection | SDK base XYZ (meters)', 1500, 950)
        self.scenes = []
        for _ in range(2):
            scene = gui.SceneWidget()
            scene.scene = rendering.Open3DScene(self.window.renderer)
            scene.scene.set_background([.94, .95, .97, 1])
            scene.scene.show_axes(True)
            self.window.add_child(scene)
            self.scenes.append(scene)
        self.images = [gui.ImageWidget(o3d.geometry.Image(frame.rgb.copy())),
                       gui.ImageWidget(o3d.geometry.Image(frame.rgb.copy()))]
        for image in self.images:
            self.window.add_child(image)
        self.panel = gui.Vert(6, gui.Margins(10, 10, 10, 10))
        self.panel.add_child(gui.Label('SDK BASE XYZ CROP (meters)'))
        self.panel.add_child(gui.Label('Left: before / right: cropped'))
        self.panel.add_child(gui.Label('Bottom: original RGB / pixel mask'))
        self.panel.add_child(gui.Label(f"Camera: {frame.calibration['serial']}"))
        self.panel.add_child(gui.Label(f'Camera depth: 0 < Z < {frame.max_depth:g} m'))
        self.panel.add_child(gui.Label('Initial bounds are editable, not ground truth.'))
        self.controls = []
        full = frame.points[frame.valid]
        bounds = np.quantile(full, [.01, .99], axis=0)
        for axis, label in enumerate('XYZ'):
            for kind in ('min', 'max'):
                row = gui.Horiz(4)
                row.add_child(gui.Label(f'{label} {kind}'))
                slider = gui.Slider(gui.Slider.DOUBLE)
                edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                value = crop[kind][axis]
                minimum = float(min(bounds[0, axis]-.3, crop['min'][axis]-.3, -1))
                maximum = float(max(bounds[1, axis]+.3, crop['max'][axis]+.3, 1))
                slider.set_limits(minimum, maximum)
                edit.set_limits(-100, 100)
                edit.decimal_precision = 4
                slider.double_value = edit.double_value = value
                slider.set_on_value_changed(lambda v, a=axis, k=kind: self.change(a, k, v))
                edit.set_on_value_changed(lambda v, a=axis, k=kind: self.change(a, k, v))
                row.add_child(slider)
                row.add_child(edit)
                self.panel.add_child(row)
                self.controls.append((axis, kind, slider, edit))
        self.status = gui.Label('')
        self.panel.add_child(self.status)
        self.plane = gui.Label('Plane: click Analyze table')
        self.panel.add_child(self.plane)
        for label, callback in [('3D overview', lambda: self.set_view('overview')),
                                ('Top XY', lambda: self.set_view('top')),
                                ('Front XZ', lambda: self.set_view('front')),
                                ('Side YZ', lambda: self.set_view('side')),
                                ('Analyze table', self.analyze), ('Save config + report + image', self.save)]:
            button = gui.Button(label)
            button.set_on_clicked(callback)
            self.panel.add_child(button)
        self.panel.add_child(gui.Label('Plane / crop cannot prove absolute accuracy.'))
        self.window.add_child(self.panel)
        self.window.set_on_layout(self.layout)
        self.refresh()
        corners, _ = box_edges(crop)
        bbox = o3d.geometry.AxisAlignedBoundingBox(np.minimum(corners.min(0), 0), np.maximum(corners.max(0), .15))
        for scene in self.scenes:
            scene.setup_camera(60, bbox, bbox.get_center().astype(np.float32))
        self.set_view('overview')

    def layout(self, event):
        gui = self.gui
        r = self.window.content_rect
        width = max(100, r.width-350)
        top = int(r.height*.60)
        self.panel.frame = gui.Rect(r.x+width, r.y, 350, r.height)
        for i in range(2):
            x = r.x+int(width*i/2)
            self.scenes[i].frame = gui.Rect(x, r.y, int(width/2), top)
            self.images[i].frame = gui.Rect(x, r.y+top, int(width/2), r.height-top)

    def change(self, axis, kind, value):
        if self.syncing:
            return
        self.crop[kind][axis] = float(value)
        self.syncing = True
        try:
            for a, k, slider, edit in self.controls:
                if a == axis and k == kind:
                    slider.set_limits(min(slider.get_minimum_value, value-.01), max(slider.get_maximum_value, value+.01))
                    slider.double_value = edit.double_value = value
        finally:
            self.syncing = False
        self.refresh()

    def refresh(self):
        from open3d.visualization import rendering
        o3d = self.o3d
        try:
            kept = crop_mask(self.frame.points, self.frame.valid, self.crop)
            before, bc, after, ac = display_points(self.frame, kept)
            corners, edges = box_edges(self.crop)
            for scene, points, colors in zip(self.scenes, (before, after), (bc, ac)):
                for name in ('points', 'crop'):
                    if scene.scene.has_geometry(name):
                        scene.scene.remove_geometry(name)
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
                cloud.colors = o3d.utility.Vector3dVector(colors)
                material = rendering.MaterialRecord()
                material.shader = 'defaultUnlit'
                material.point_size = 2
                if len(points):
                    scene.scene.add_geometry('points', cloud, material)
                box = o3d.geometry.LineSet(o3d.utility.Vector3dVector(corners), o3d.utility.Vector2iVector(edges))
                box.paint_uniform_color([1, .55, 0])
                material = rendering.MaterialRecord()
                material.shader = 'unlitLine'
                material.line_width = 2
                scene.scene.add_geometry('crop', box, material)
            self.images[1].update_image(o3d.geometry.Image(overlay_rgb(self.frame, kept)))
            self.status.text = f'Kept {kept.sum():,} / {self.frame.valid.sum():,}\nAll pixels counted; 3D display subsampled.'
            self.plane.text = 'Bounds changed: click Analyze table.'
        except ValueError as exc:
            for scene in self.scenes:
                for name in ('points', 'crop'):
                    if scene.scene.has_geometry(name):
                        scene.scene.remove_geometry(name)
            self.images[1].update_image(o3d.geometry.Image(overlay_rgb(self.frame, np.zeros_like(self.frame.valid))))
            self.plane.text = 'Invalid crop: no current selection displayed.'
            self.status.text = f'INVALID BOUNDS: {exc}'

    def set_view(self, view):
        lo, hi = validate_crop(self.crop)
        center = (lo+hi)/2
        distance = max(float(np.linalg.norm(hi-lo)), .5)
        direction, up = {'overview': ([.6, -1, .9], [0, 0, 1]),
                         'top': ([0, 0, 1], [0, 1, 0]),
                         'front': ([0, -1, .1], [0, 0, 1]),
                         'side': ([1, 0, .1], [0, 0, 1])}[view]
        for scene in self.scenes:
            scene.scene.camera.look_at(center.astype(np.float32),
                (center+np.array(direction)*distance).astype(np.float32), np.array(up, np.float32))

    def analyze(self):
        try:
            _, report = diagnostics(self.frame, self.crop, self.table_z)
            plane = report['table_plane_candidate']
            if not plane['available']:
                self.plane.text = 'No near-horizontal plane candidate.'
            else:
                self.plane.text = (f"Plane tilt: {plane['tilt_deg']:.2f} deg\nMedian base Z: {plane['median_height_m']:.4f} m\n"
                                   f"Plane fit RMSE: {plane['rmse_mm']:.2f} mm")
                if plane['height_error_mm'] is not None:
                    self.plane.text += f"\nMeasured height error: {plane['height_error_mm']:.2f} mm"
        except ValueError as exc:
            self.status.text = str(exc)

    def save(self):
        try:
            save_inspection(self.frame, self.crop, self.output, table_z=self.table_z)
            self.status.text = f'SAVED to:\n{self.output}'
            self.analyze()
        except (ValueError, OSError) as exc:
            self.status.text = f'Not saved: {exc}'


def main(argv=None):
    p = argparse.ArgumentParser(description='标定后检查：机器人SDK base下XYZ裁剪、RGB覆盖和桌面平面诊断；不发送运动')
    p.add_argument('--data-dir', type=Path, default=ROOT/'data/calibration/latest/third_camera')
    p.add_argument('--camera-params', type=Path, help='显式相机参数，用于其他会话或--live')
    p.add_argument('--sample', type=int, default=1, help='保存会话中的板位，从1开始')
    p.add_argument('--max-depth', type=float, default=1.2, help='相机光轴深度上限，米；XYZ裁剪在基座下另算')
    p.add_argument('--crop-min', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    p.add_argument('--crop-max', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    p.add_argument('--crop-config', type=Path, help='重用检查配置，严格核对相机、基座和标定哈希')
    p.add_argument('--table-z', type=float, help='可选：独立测得的桌面在SDK base中的Z高度，米')
    p.add_argument('--live', action='store_true', help='从相机服务只读获取当前一帧，支持人工量尺方案')
    p.add_argument('--camera-endpoint', default='tcp://127.0.0.1:18083')
    p.add_argument('--output', type=Path, help='保存检查记录的目录，默认data/calibration/diagnostics/base_crop_时间')
    p.add_argument('--snapshot', action='store_true', help='无界面保存检查图、裁剪配置和报告后退出')
    args = p.parse_args(argv)
    if not np.isfinite(args.max_depth) or args.max_depth <= 0:
        raise ValueError('--max-depth必须是有限正米数')
    if args.table_z is not None and not np.isfinite(args.table_z):
        raise ValueError('--table-z必须是有限米数')
    if bool(args.crop_min is not None) != bool(args.crop_max is not None):
        raise ValueError('--crop-min和--crop-max需要一起指定')
    if args.crop_config and args.crop_min is not None:
        raise ValueError('请在--crop-config和显式XYZ边界中选择一种')
    data_dir = resolve_calibration_path(args.data_dir)
    if args.camera_params:
        args.camera_params = resolve_calibration_path(args.camera_params)
    if args.live:
        from gene_agent.perception_modules.camera import CameraClient
        parameters = args.camera_params or data_dir/'third_view_extrinsic.json'
        camera = CameraClient(args.camera_endpoint)
        try:
            packet = camera.get_frame()
        finally:
            camera.close()
        frame = make_frame(packet.rgb, packet.depth, packet.intrinsic_matrix, packet.serial, parameters,
            source=f'{args.camera_endpoint} frame {packet.frame_number}', max_depth=args.max_depth)
    else:
        frame = load_frame(data_dir, parameters_path=args.camera_params, sample=args.sample, max_depth=args.max_depth)
    crop = load_crop_config(args.crop_config, frame) if args.crop_config else {
        'min': list(DEFAULT_CROP['min'] if args.crop_min is None else args.crop_min),
        'max': list(DEFAULT_CROP['max'] if args.crop_max is None else args.crop_max)}
    validate_crop(crop)
    output = args.output or ROOT/'data/calibration/diagnostics'/f'base_crop_{datetime.now():%Y%m%d_%H%M%S}'
    print(f'基座XYZ裁剪检查：{frame.calibration_path}\n相机：{frame.calibration["serial"]}，来源：{frame.source}')
    print(f'初始XYZ范围（米）：{crop}\n紫色表示无效深度或超过{frame.max_depth:g}米；暗色表示基座XYZ裁剪排除。')
    print(LIMITATION)
    if args.snapshot:
        save_inspection(frame, crop, output, table_z=args.table_z)
        return 0
    from open3d.visualization import gui
    gui.Application.instance.initialize()
    app = CropApp(frame, crop, output, args.table_z)
    gui.Application.instance.run()
    del app
    return 0
