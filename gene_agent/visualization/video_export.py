"""共享 Skill/Task 视频导出：只读时间线，离屏渲染，直接流式编码。"""
from __future__ import annotations

from bisect import bisect_left
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gene_agent.simulation.skill_viewer import SkillMujocoViewer, restore_replay_frame
from gene_agent.visualization.style import configure_camera


def resolve_ffmpeg() -> str:
    """Resolve and execute-check FFmpeg before any task depends on video export."""
    candidates = []
    configured = os.environ.get("GENE_AGENT_FFMPEG")
    if configured:
        candidates.append(configured)
    discovered = shutil.which("ffmpeg")
    if discovered:
        candidates.append(discovered)
    candidates.extend((
        str(Path(sys.executable).resolve().parent / "ffmpeg"),
        "/data1/yuquan/miniconda3/envs/capnet/bin/ffmpeg",
    ))
    candidates.extend(str(path) for path in sorted(
        Path("/data1/yuquan/miniconda3/pkgs").glob("ffmpeg-*/bin/ffmpeg"),
        reverse=True,
    ))
    checked = []
    for raw in dict.fromkeys(candidates):
        path = Path(raw).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            checked.append(f"{path}: missing or not executable")
            continue
        try:
            version = subprocess.run(
                [str(path), "-version"], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=10, check=False,
            )
            encoders = subprocess.run(
                [str(path), "-hide_banner", "-encoders"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            checked.append(f"{path}: {type(error).__name__}: {error}")
            continue
        if (version.returncode == 0 and encoders.returncode == 0
                and "libx264" in encoders.stdout):
            return str(path.resolve())
        checked.append(
            f"{path}: version_exit={version.returncode}, "
            f"encoders_exit={encoders.returncode}, libx264={'libx264' in encoders.stdout}"
        )
    raise FileNotFoundError(
        "no runnable ffmpeg found; checked: " + "; ".join(checked)
    )


def restore_render_state(model, data, frame):
    """仅用于独立渲染副本，不计算接触/受力，不产生可继续物理积分的状态。

    完整 qpos 已含夹爪和物体的实测仿真位置。几何、site、相机和灯光
    只需位置运动学；不调用 mj_step，也不依赖上一帧的控制/接触状态。
    """
    qpos = np.asarray(frame.qpos, dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError('视频完整 qpos 维度或数值无效')
    data.qpos[:] = qpos
    if frame.sim_time_s is not None:
        data.time = float(frame.sim_time_s)
    data.ncon = 0
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)
    if model.ntendon:
        mujoco.mj_tendon(model, data)


def frame_schedule(timeline, *, fps):
    """保持源时间间隔；没被固定帧率覆盖的短状态事件补一帧并明确记录。

    不逐段重新 select，否则段间稳定等待可能因局部时间线变短而丢失。
    重复帧只保持原记录状态，不做位置插值。
    """
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError('fps 必须为正整数')
    duration = float(timeline.duration_s)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('视频时间线时长无效')
    times = sorted({min(i / fps, duration)
                    for i in range(max(2, math.ceil(duration * fps) + 1))})
    for segment in timeline._segments:
        start, end = segment.start_s, segment.start_s + segment.duration_s
        i = bisect_left(times, start)
        if i == len(times) or times[i] >= end:
            times.append(start)
            times.sort()
    return sorted(set(times))


def _evidence(run, mode):
    for name in ('summary.json', 'task_result.json', 'result.json', 'report.json', 'insert_result.json'):
        path = run / name
        if path.is_file():
            row = json.loads(path.read_text(encoding='utf-8'))
            declared = row.get('evidence_mode') or row.get('compound_mode')
            if not declared and row.get('simulation_assistance'):
                declared = row['simulation_assistance']
            if not declared and (row.get('mode') == 'physical_assisted'
                                 or row.get('simulation_weld_after_contact')):
                declared = 'simulation_assisted'
            if not declared and row.get('fully_physical_pick_insert_success') is False:
                declared = 'not_fully_physical'
            if name == 'result.json' and row.get('pick_result_path') and (run / 'pick/result.json').is_file():
                pick_report = json.loads((run / 'pick/result.json').read_text(encoding='utf-8'))
                # 配置允许 weld 不表示本次前缀已激活；以实际操作报告为准。
                declared = ('simulation_assisted' if pick_report.get('simulation_attachment') else
                            'mujoco_contact' if pick_report.get('contact_success') else 'mujoco_execution_prefix')
                row = {**row, 'requested_scope': pick_report.get('requested_scope', 'pick'),
                       'physical_pick_success': pick_report.get('physical_pick_success')}
            operation_success = row.get('operation_success', row.get('goal_success', row.get('success')))
            terminal = {'cup_pour_balls_into_bowl_and_return': 'place',
                        'scoop_balls_into_other_bowl': 'pour'}.get(row.get('task'))
            scope = row.get('requested_scope', row.get('until_skill'))
            partial = ((row.get('requested_scope') not in (None, 'all', 'full'))
                       or (terminal is not None and scope is not None
                           and scope not in (terminal, 'all', 'full')))
            evidence = {'report_ref': name, 'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'evidence_mode': declared or mode,
                'task_success': None if partial else row.get('task_success', row.get('goal_success', row.get('success'))),
                'operation_success': operation_success,
                'requested_scope': scope if partial else 'full',
                'fully_physical_success': row.get('fully_physical_success', row.get('physical_pick_success'))}
            launch = run / 'launch.json'
            if launch.is_file():
                selection = json.loads(launch.read_text(encoding='utf-8'))
                if selection.get('requested_scope') not in (None, 'full'):
                    evidence.update(task_success=None, requested_scope=selection['requested_scope'])
                if selection.get('status') in {'interrupted', 'exception'}:
                    evidence['task_success'] = None
            return evidence
    return {'evidence_mode': mode, 'task_success': None}


def _camera(viewer, frame, name, profile):
    if profile == 'microwave_task1':
        from gene_agent.visualization.microwave.cameras import camera_parameters
        params = camera_parameters(viewer, frame, name)
        return configure_camera(mujoco.MjvCamera(), params), params
    if profile is not None:
        raise ValueError(f'未知视频相机 profile：{profile}')
    return viewer._camera(name)


def export_video(run, output, *, mode='actual', cameras=('task',), profile=None,
                 fps=12, width=640, height=480, stage=None, skill=None,
                 full_forward=False, preset='veryfast', threads=2):
    """一个 Renderer 导出横向多视角 MP4；成功后才替换目标视频与 manifest。

    full_forward 是同一采样/画质的计时诊断开关，不改视频时间线或事实来源。
    actual 不足时明确报错；只有显式 mode=planned 才允许运动学规划预览。
    """
    started = perf_counter()
    if mode not in {'actual', 'planned'}:
        raise ValueError('mode 只能为 actual 或 planned')
    if (not cameras or len(set(cameras)) != len(cameras)
            or min(width, height) <= 0 or width % 2 or height % 2 or threads <= 0):
        raise ValueError('相机不能空或重复；画面尺寸须为正偶数，threads须为正')
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.suffix.lower() != '.mp4':
        raise ValueError('视频输出必须为 .mp4')
    viewer = SkillMujocoViewer(run, skill=skill)
    viewer.select(stage, mode=mode)
    times = frame_schedule(viewer.timeline, fps=fps)
    evidence = _evidence(run, mode)
    viewer.model.vis.global_.offwidth = max(viewer.model.vis.global_.offwidth, width)
    viewer.model.vis.global_.offheight = max(viewer.model.vis.global_.offheight, height)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=output.stem + '.', suffix='.partial.mp4',
                                         dir=output.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    command = [resolve_ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width * len(cameras)}x{height}', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', preset, '-threads', str(threads), '-crf', '22',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    process = None
    frames, seen, camera_records, headers = [], [], {}, {}
    timing = dict(load=perf_counter()-started, restore=0., render=0., overlay=0., encode_write=0.)
    last_key, pixels, rendered_states = None, None, 0
    # Pillow 隐式默认字体会在每次 draw.text 时重新取得，长视频显式复用。
    font = ImageFont.load_default()
    claim = 'MUJOCO SAVED ACTUAL' if mode == 'actual' else 'PLANNED KINEMATIC PREVIEW - NO CONTACT CLAIM'
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE)
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        with mujoco.Renderer(viewer.model, width=width, height=height) as renderer:
            for video_index, time_s in enumerate(times):
                frame = viewer.timeline.seek(time_s)
                # 同一保存样本的重复显示直接复用画面，不重复 forward/render。
                key = (frame.stage_id, frame.state_id, frame.sim_time_s, frame.sample_time_s)
                if key != last_key:
                    tick = perf_counter()
                    if mode == 'actual' and frame.qpos is not None and not full_forward:
                        restore_render_state(viewer.model, viewer.data, frame)
                    else:
                        restore_replay_frame(viewer.model, viewer.data, frame, run,
                            robot_metadata=viewer.robot_metadata, active_group=viewer.active_group,
                            baseline_qpos=viewer.planned_base_qpos)
                    viewer.frame = frame
                    timing['restore'] += perf_counter()-tick
                    canvas = Image.new('RGB', (width*len(cameras), height))
                    for panel, name in enumerate(cameras):
                        camera, params = _camera(viewer, frame, name, profile)
                        camera_records.setdefault(name, params)
                        tick = perf_counter()
                        renderer.update_scene(viewer.data, camera=camera, scene_option=option)
                        rgb = renderer.render()
                        timing['render'] += perf_counter()-tick
                        tick = perf_counter()
                        image = Image.fromarray(rgb)
                        header_key = (name, frame.stage_id)
                        if header_key not in headers:
                            header = Image.new('RGB', (width-10, 60), (20, 28, 40))
                            draw = ImageDraw.Draw(header)
                            label = frame.stage_id[-max(12, width//6):]
                            draw.text((7, 5), f'{name} | {claim}', fill='white', font=font)
                            draw.text((7, 22), label, fill='white', font=font)
                            draw.text((112, 39), str(evidence['evidence_mode']),
                                      fill=(220,230,240), font=font)
                            headers[header_key] = header
                        image.paste(headers[header_key], (5, 5))
                        draw = ImageDraw.Draw(image)
                        stamp = frame.sim_time_s if frame.sim_time_s is not None else time_s
                        draw.text((12, 44), f't={stamp:.3f}s', fill=(220,230,240), font=font)
                        canvas.paste(image, (panel*width, 0))
                        timing['overlay'] += perf_counter()-tick
                    pixels = canvas.tobytes()
                    last_key = key
                    rendered_states += 1
                tick = perf_counter()
                process.stdin.write(pixels)
                timing['encode_write'] += perf_counter()-tick
                if frame.stage_id not in seen:
                    seen.append(frame.stage_id)
                frames.append({'video_frame': video_index, 'video_time_s': video_index/fps,
                    'source_time_s': time_s, 'sim_time_s': frame.sim_time_s,
                    'stage_id': frame.stage_id, 'state_id': frame.state_id,
                    'sample_quality': frame.sample_quality})
        tick = perf_counter()
        process.stdin.close()
        error = process.stderr.read().decode('utf-8', errors='replace')
        if process.wait() != 0:
            raise RuntimeError(f'FFmpeg 视频编码失败：{error}')
        timing['encode_flush'] = perf_counter()-tick
        timing['total'] = perf_counter()-started
        manifest = {'schema_version': 2, 'source_run': str(run), 'video': str(output),
            'mode': mode, 'claim': claim, **evidence,
            'stage': stage or 'all', 'skill': viewer.timeline.view.skill,
            'cameras': list(cameras), 'camera_profile': profile, 'initial_cameras': camera_records,
            'fps': fps, 'size_px': [width*len(cameras),height], 'frame_count': len(times),
            'rendered_state_count': rendered_states, 'renderer_count': 1,
            'source_duration_s': viewer.timeline.duration_s, 'duration_s': len(times)/fps,
            'sample_policy': 'nearest_saved_sample_no_interpolation; short stage events add a frame',
            'render_state_method': 'full_forward' if full_forward or mode=='planned' else 'kinematics_only',
            'contact_forces_restored': False, 'stages_seen': seen, 'frames': frames,
            'encoder': {'codec':'libx264','preset':preset,'crf':22,'threads':threads},
            'timing_s': timing, 'output_bytes': temporary.stat().st_size,
            'export_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        # 先完成序列化，失败不能损坏之前已交付的视频。
        text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
        manifest_path = temporary.with_suffix('.json')
        manifest_path.write_text(text, encoding='utf-8')
        temporary.replace(output)
        manifest_path.replace(output.with_suffix('.json'))
        return manifest
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)
        temporary.with_suffix('.json').unlink(missing_ok=True)
        raise
