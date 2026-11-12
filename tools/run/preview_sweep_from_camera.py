#!/usr/bin/env python3
"""一条命令：首帧 RGB-D → SAM3 → FoundationPose → 抓刷/清扫 cuRobo → Viser。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-param', type=Path, default=ROOT/'data/calibration/cam_param.json',
                        help='固定相机；extrinsic_matrix 必须为 T_sdk_camera，长度单位米')
    parser.add_argument('--serial', help='可选，必须与标定文件一致；默认读取标定中的 serial')
    parser.add_argument('--config', type=Path, default=ROOT/'config/realworld/sweep_old.example.json')
    parser.add_argument('--collision-mode', choices=['full', 'robot_only'])
    parser.add_argument('--capture-dir', type=Path, help='复用已有 rgb.png/depth.png/camera.json，不连接相机')
    parser.add_argument('--camera-endpoint', help='可选：复用已有 camera server；默认直接采一帧后释放相机')
    state_mode = parser.add_mutually_exclusive_group()
    state_mode.add_argument('--robot-ip', help='只读关节和法兰；默认标定文件/现场配置地址，再回退 169.254.128.19')
    state_mode.add_argument('--assume-standby', action='store_true', help='显式离线诊断：假定模板待机，不连接机械臂')
    parser.add_argument('--sdk-python', default='/home/ubuntu/miniconda3/envs/pika/bin/python')
    parser.add_argument('--frame-profile', type=Path, help='覆盖机器人/夹爪命名 frame；默认使用该设备配置')
    parser.add_argument('--gripper-width', type=float, default=.04, help='初始实测/假定开口（米）；不读取或驱动夹爪')
    parser.add_argument('--table-top-z', type=float, help='SDK base 下桌面高度（米），仍核对深度支持')
    parser.add_argument('--no-support-correction', action='store_true', help='关闭初始桌面支撑校正，严格使用原始感知 pose')
    parser.add_argument('--max-support-correction', type=float, default=.03, help='初始物体沿 world Z 向上校正上限（米，默认0.03）')
    parser.add_argument('--brush-prompt', default='a cleaning brush')
    parser.add_argument('--cube-prompt', default='a green cube')
    parser.add_argument('--dustpan-prompt', default='a dustpan')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/realworld_sweep'/time.strftime('camera_%Y%m%d_%H%M%S'))
    parser.add_argument('--no-viewer', action='store_true', help='只保存记录；之后可用 view_task.py 打开')
    parser.add_argument('--stop-after',choices=['pregrasp','approach','pick','task'],default='task',
                        help='只规划到指定阶段；默认完整任务')
    parser.add_argument('--port', type=int, default=8093)
    args = parser.parse_args(argv)

    import mujoco
    import numpy as np
    from gene_agent.robots.pika_frames import load_frame_profile, pose_from_mujoco
    from gene_agent.scene.rigid_geometry import visual_mesh_object
    from gene_agent.realworld.sweep.scene import read_site, build_observed_scene
    from gene_agent.realworld.sweep.first_frame import load_calibration, capture, perceive, estimate_table, ROLES
    from gene_agent.realworld.sweep.planning import write_json
    from gene_agent.realworld.sweep.table_preview import preview_table_task

    calibration = load_calibration(args.camera_param)
    if args.serial and args.serial != calibration['serial']:
        parser.error('--serial 与标定文件不一致')
    if not 0 <= args.gripper_width <= .0941:
        parser.error('--gripper-width 超出 Pika 模型开口范围')
    if not np.isfinite(args.max_support_correction) or not 0 < args.max_support_correction <= .1:
        parser.error('--max-support-correction 必须在 (0,0.1] 米')
    site = read_site(args.config)
    if args.collision_mode is not None:
        site['motion'] = {**site.get('motion', {}), 'collision_mode': args.collision_mode}
    site['tabletop_support_policy'] = {'enabled': not args.no_support_correction,
        'max_adjustment_m':args.max_support_correction, 'clearance_m':.001}
    from gene_agent.realworld.sweep.observation import resolve_preview_robot_ip
    try:
        robot_ip = resolve_preview_robot_ip(site, args.robot_ip, calibration=calibration, assume_standby=args.assume_standby,
                                            saved_capture=args.capture_dir is not None)
    except ValueError as exc:
        parser.error(str(exc))
    if robot_ip:
        site['devices']['robot_ip'] = robot_ip
    selected_profile = args.frame_profile or site.get('frame_profiles_by_robot_ip', {}).get(robot_ip)
    if selected_profile:
        profile_path = Path(selected_profile)
        site['frame_profile'] = str((ROOT/profile_path).resolve())
        load_frame_profile(site['frame_profile'])
    if site.get('brush_variant') != 'old':
        parser.error('本入口先支持原蓝色旧刷；白柄长刷请使用原有独立入口')
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = None
    try:
        write_json(output/'camera_calibration.json', calibration)
        write_json(output/'command.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        sources = ['tools/run/preview_sweep_from_camera.py', 'gene_agent/realworld/sweep/first_frame.py',
                   'gene_agent/realworld/sweep/table_preview.py', 'gene_agent/realworld/sweep/preview_motion.py',
                   'gene_agent/realworld/sweep/planning.py', 'gene_agent/realworld/sweep/scene.py',
                   'gene_agent/realworld/sweep/observation.py', 'gene_agent/realworld/sweep/diagnostics.py',
                   'gene_agent/scene/rigid_geometry.py', 'gene_agent/perception_modules/local_service.py',
                   'gene_agent/scene/tabletop_support.py',
                   'gene_agent/knowledge/sweep.py', 'gene_agent/skills/library/sweep/skill.py']
        write_json(output/'code_sources_sha256.json', {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources})
        template = mujoco.MjModel.from_xml_path(site['scene_template'])
        reference = mujoco.MjData(template)
        if template.nkey:
            mujoco.mj_resetDataKeyframe(template, reference, 0)
        mujoco.mj_forward(template, reference)
        meshes = {}
        (output/'assets').mkdir()
        # 旧仿真曾按用户要求把方块缩小一半；真机感知恢复有来源的原测量尺寸。
        scene_metadata_path = Path(site['scene_template']).with_suffix('.json')
        scene_metadata = json.loads(scene_metadata_path.read_text(encoding='utf-8')) if scene_metadata_path.exists() else {}
        for role in ROLES:
            meshes[role] = output/f'assets/{role}.obj'
            mesh = visual_mesh_object(template, reference, site['objects'][role]['body_name'])
            original_scale = float(scene_metadata.get('objects', {}).get(role, {}).get('scene_scale_factor', 1.))
            if not np.isfinite(original_scale) or original_scale <= 0:
                raise ValueError(f'{role} 原场景缩放记录无效')
            scale = 1/original_scale
            mesh.apply_scale(scale)
            site['objects'][role]['uniform_scale'] = scale
            site['objects'][role]['scale_source'] = f'undo explicit simulation scene_scale_factor={original_scale}'
            mesh.export(meshes[role])
            site['objects'][role]['T_perception_asset'] = np.eye(4).tolist()
            site['objects'][role]['alignment_source'] = 'compiled visual exported into original asset body frame'
        print(f'输出：{output}', flush=True)
        print(f'采集固定相机 {calibration["serial"]} 的一帧 RGB-D', flush=True)
        read_before = None
        if robot_ip:
            from gene_agent.realworld.sweep.observation import _read_flange
            device = {'robot_ip': robot_ip, 'sdk_python': args.sdk_python}
            print(f'只读机器人 {robot_ip} 的当前关节和法兰；读取失败即停止，不使用模板替代', flush=True)
            read_before = _read_flange(device)
        camera, rgb, depth = capture(calibration, output/'capture', reuse_capture=args.capture_dir,
                                     endpoint=args.camera_endpoint)
        if robot_ip:
            flange, q = _read_flange(device)
            if np.max(np.abs(q-read_before[1])) > .005:
                raise ValueError('拍照期间机械臂移动；请保持静止重新采集')
            robot_source = 'sdk_read_only'
            write_json(output/'robot_state.json', {'source': robot_source, 'robot_ip': robot_ip,
                'physical_robot_group': calibration.get('robot_group', 'unspecified'),
                'joint_positions_rad': q.tolist(), 'T_sdk_flange': flange.tolist(),
                'read_at_unix_ms': time.time()*1000, 'execution_enabled': False,
                'gripper_width_m': args.gripper_width, 'gripper_width_source': 'user_input_not_sdk_read'})
            print('首帧机器人关节（rad）：'+np.array2string(q, precision=4), flush=True)
        else:
            profile = load_frame_profile(site['frame_profile'])
            q = np.array([reference.qpos[template.joint(f'joint_{i}').qposadr[0]] for i in range(1, 8)])
            flange = np.asarray(site['T_sdk_urdf_base']) @ pose_from_mujoco(reference.body('link_7')) @ profile['T_link7_sdk_flange']
            robot_source = 'assumed_template_standby'
            print('机器人初态：显式 --assume-standby，假定模板待机（非实测）', flush=True)
        prompts = {r: getattr(args, r+'_prompt') for r in ROLES}
        poses, excluded = perceive(calibration, output/'capture', meshes, output/'perception', prompts)
        table, table_evidence = estimate_table(depth, excluded, calibration, site['table_size_xyz_m'], top_z=args.table_top_z)
        write_json(output/'perception/table_estimate.json', table_evidence)
        site['T_sdk_camera'] = calibration['T_sdk_camera']
        site['T_sdk_table'] = table.tolist()
        site['calibration_source'] = f'preview-only assumed correct: {args.camera_param.resolve()}'
        site['devices']['camera']['serial'] = calibration['serial']
        site['snapshot_source'] = 'first_frame_preview'
        site['robot_state_source'] = robot_source
        now = time.time()*1000
        snapshot = {'schema_version': 1, 'source': 'first_frame_preview', 'execution_enabled': False,
                    'initial_brush_state': 'on_table', 'robot_state_source': robot_source,
                    'joint_names': [f'joint_{i}' for i in range(1, 8)], 'joint_positions_rad': q.tolist(),
                    'T_sdk_flange': flange.tolist(), 'gripper_width_m': args.gripper_width,
                    'pose_frame_verified': True, 'observed_at_unix_ms': camera.get('captured_at_unix_ms', now),
                    'camera_serial': camera['serial'], 'camera_frame_number': camera['frame_number'],
                    'hold': {'confirmed': False, 'source': 'brush_observed_on_table'},
                    'T_sdk_object': {r: p.tolist() for r, p in poses.items()},
                    'input_mode': 'saved_capture' if args.capture_dir else 'new_camera_frame'}
        write_json(output/'site_input.json', site);write_json(output/'observation.json', snapshot)
        model, data, robot, _, provenance = build_observed_scene(site, snapshot, output/'scene',
                                                                allow_first_frame_preview=True)
        if provenance.get('support_adjustment'):
            for role, row in provenance['support_adjustment']['objects'].items():
                print(f'规划支撑校正 {role}：world Z +{row["shift_world_z_m"]*1000:.1f} mm（原观测已保留）', flush=True)
        # 保存带颜色的首帧世界点云，供用户直接对照 mesh，不把点云用作任务成功证据。
        v, u = np.indices(depth.shape);valid = (depth > .08) & (depth < 3)
        z = depth[valid][::8];k = np.asarray(calibration['intrinsic_matrix'])
        xyz = np.column_stack(((u[valid][::8]-k[0, 2])*z/k[0, 0], (v[valid][::8]-k[1, 2])*z/k[1, 1], z))
        t = np.linalg.inv(site['T_sdk_urdf_base']) @ np.asarray(calibration['T_sdk_camera'])
        np.savez_compressed(output/'perception/scene_points.npz', points=xyz@t[:3, :3].T+t[:3, 3], colors=rgb[valid][::8])
        print(f'感知进程已退出；开始 cuRobo 规划，阶段范围：{args.stop_after}', flush=True)
        report = preview_table_task(model, data, robot, site, output,stop_after=args.stop_after)
        report['total_duration_s'] = time.monotonic()-start
        report['camera_calibration_sha256'] = calibration['sha256']
        report['frame_profile_status'] = load_frame_profile(site['frame_profile'])['status']
        report['input_mode'] = snapshot['input_mode']
        write_json(output/'result.json', report)
        ready=report.get('requested_scope_complete',report['planning_complete'])
        print(f'请求范围={args.stop_after}，范围内规划完成={ready}；记录：{output}；未发送运动命令', flush=True)
        if not ready:
            from gene_agent.realworld.sweep.diagnostics import failure_summary
            print(failure_summary(report), flush=True)
    except Exception as exc:
        (output/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        if (output/'result.json').exists():
            report = json.loads((output/'result.json').read_text(encoding='utf-8'))
            report.update(status='pipeline_error', error=str(exc), planning_complete=False)
            write_json(output/'result.json', report)
        else:
            write_json(output/'pipeline_error.json', {'error': str(exc), 'execution_enabled': False})
        print(f'失败证据：{output}/traceback.txt', file=sys.stderr)
        raise
    if not args.no_viewer:
        from gene_agent.visualization.realworld_sweep import serve_preview
        serve_preview(output, port=args.port)
    return 0 if report.get('requested_scope_complete',report['planning_complete']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
