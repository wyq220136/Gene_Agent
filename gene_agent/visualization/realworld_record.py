"""静态观测 + 实测端点/unknown 证据面板；没有设备、规划和动作接口。"""
from __future__ import annotations

import json
from pathlib import Path


def load_static_scene(row):
    import mujoco
    import numpy as np
    model = mujoco.MjModel.from_xml_path(row['xml_path'])
    data = mujoco.MjData(model)
    qpos = np.asarray(row['snapshot']['qpos_snapshot'], dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError('保存的静态观测 qpos_snapshot 与 scene.xml 维度不一致或数值无效')
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return model, data


def report_markdown(name, report):
    return (f'### {name}\n\n报告状态：**{report.get("status", "unknown")}**\n\n'
        'SDK measured_end/observed_after 是离散实测端点；plan/target 是保存的规划。'
        'unavailable/unknown 表示证据缺失，不补成成功或连续运动。\n\n'
        '```json\n' + json.dumps(report, ensure_ascii=False, indent=2) + '\n```')


class RealworldRecordApp:
    def __init__(self, server, record):
        self.server, self.record = server, record
        self.scene = None
        self.model = self.data = None
        server.scene.set_up_direction('+z')
        server.gui.add_markdown('## 静态观测 + 实测端点/unknown 证据\n\n'
            '只读已保存报告和感知场景；不连接相机或 SDK。场景保持所选观测帧，'
            '不会把实测端点插值成 actual 轨迹，也不会用终点关节更新旧帧物体状态。')
        server.gui.add_markdown('\n\n'.join(record['missing']))
        if record.get('targets'):
            server.gui.add_markdown('### 保存的规划目标（world_tcp，米）\n\n'
                '这些是目标 pose，不是实际到达证据。\n\n```json\n' +
                json.dumps(record['targets'], ensure_ascii=False, indent=2) + '\n```')
        names = tuple(record['reports'])
        self.report_control = server.gui.add_dropdown('报告', options=names, initial_value=names[0])
        self.report_info = server.gui.add_markdown(report_markdown(names[0], record['reports'][names[0]]))
        self.report_control.on_update(self._show_report)
        self.scene_info = server.gui.add_markdown('没有保存的完整静态场景；当前只显示报告。')
        if record['scenes']:
            options = tuple(record['scenes'])
            self.scene_control = server.gui.add_dropdown('静态观测帧', options=options, initial_value=options[0])
            self.scene_control.on_update(self._show_scene)
            self._show_scene(None)

    def _show_report(self, event):
        name = self.report_control.value
        self.report_info.content = report_markdown(name, self.record['reports'][name])

    def _show_scene(self, event):
        from gene_agent.visualization.pick_bowl.scene import PickBowlScene
        if self.scene is not None:
            self.server.scene.reset()
            self.server.scene.set_up_direction('+z')
        row = self.record['scenes'][self.scene_control.value]
        self.model, self.data = load_static_scene(row)
        self.scene = PickBowlScene(self.server, self.model, self.data, {}, {})
        source = row['provenance']
        self.scene_info.content = ('**静态观测重建，非 MuJoCo 执行**：'
            f'frame={source.get("source_frame_number", "unknown")}；'
            f'timestamp_ms={source.get("source_timestamp_ms", "unknown")}；'
            f'collision_snapshot={source.get("collision_snapshot_sha256", "unknown")}')
        self.server.initial_camera.position = (-1.26, 1.69, 1.)
        self.server.initial_camera.look_at = (.16, .2, .1)

    def tick(self):
        pass


def save_static_snapshot(record, output, *, camera_name='task'):
    if not record['scenes']:
        raise ValueError('无法截图：缺少保存的 scene.xml、qpos_snapshot 和静态观测来源；可不加 --output 打开报告面板')
    import mujoco
    from PIL import Image, ImageDraw
    from gene_agent.visualization.style import camera_presets, configure_camera
    name = next(iter(record['scenes']))
    row = record['scenes'][name]
    model, data = load_static_scene(row)
    presets = camera_presets(model, data, 'bowl')
    if camera_name not in presets:
        raise ValueError(f'静态观测相机 {camera_name!r} 不存在；可选 {tuple(presets)}')
    camera = configure_camera(mujoco.MjvCamera(), presets[camera_name])
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    with mujoco.Renderer(model, width=960, height=720) as renderer:
        renderer.update_scene(data, camera=camera)
        image = Image.fromarray(renderer.render())
    ImageDraw.Draw(image).text((12, 12), 'STATIC observed scene | no execution trajectory', fill='red')
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    manifest = {'source_format': 'realworld_static_observation', 'source_run': record['root'],
        'source_observation': name, 'provenance': row['provenance'],
        'camera': presets[camera_name], 'image': str(output),
        'claim': '静态观测；没有恢复或插值机器人执行轨迹'}
    output.with_suffix('.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest
