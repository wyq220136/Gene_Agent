"""固定机器人与双相机，仅移动棋盘格的终端标定向导，不发送运动指令。"""
from __future__ import annotations

from .storage import standard_camera_parameters, publish_result, default_run_dir, resolve_cli_paths, same_recorded_config

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from gene_agent.knowledge.geometry.transforms import validate_pose
from gene_agent.perception_modules.camera import CameraClient
from gene_agent.realworld.devices.arms.realman.connection import connect_realman_read_only, normalize_realman_ip
from .guided_fixed_board_extrinsic import _capture_window, preview_session
from .guided_third_view_crosscheck import _check_camera_health, _jsonable, _prompt_pair, _record_pair, _save_json
from .moving_board_extrinsic import check_fixed_robot, check_new_board, evaluate_moving_board
from .wrist_handeye import board_object_points


def _parser():
    p = argparse.ArgumentParser(description='固定机器人和双相机，只移动板：多板位第三相机基座外参标定')
    p.add_argument('--handeye', type=Path)
    p.add_argument('--ip', default='192.168.1.18')
    p.add_argument('--group', choices=('left', 'right'), default='right')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--wrist-endpoint', default='tcp://127.0.0.1:18082')
    p.add_argument('--third-endpoint', default='tcp://127.0.0.1:18083')
    p.add_argument('--third-serial')
    p.add_argument('--poses', type=int, default=8, help='总板位数，默认8')
    p.add_argument('--validation-poses', type=int, default=2, help='最后若干板位独立验证，不参与拟合，默认2')
    p.add_argument('--state-source', choices=('udp', 'tcp'), default='udp')
    p.add_argument('--sdk-timeout-ms', type=int, default=2000)
    p.add_argument('--udp-target-ip')
    p.add_argument('--udp-port', type=int, default=8888)
    p.add_argument('--udp-timeout', type=float, default=.5)
    p.add_argument('--connect-retries', type=int, default=30)
    p.add_argument('--connect-retry-interval', type=float, default=.5)
    p.add_argument('--max-joint-change-deg', type=float, default=.2)
    p.add_argument('--max-reprojection-px', type=float, default=2.)
    p.add_argument('--max-pair-gap-ms', type=float, default=250.)
    p.add_argument('--max-validation-mm', type=float, default=10.)
    p.add_argument('--max-validation-deg', type=float, default=3.)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-preview', action='store_true', help='终端回车取帧、y确认，替代实时OpenCV')
    p.add_argument('--visualize', action='store_true')
    p.add_argument('--preview-only', action='store_true', help='只读取保存数据预览，不连接设备')
    p.add_argument('--max-depth', type=float, default=1.2,
                   help='Open3D仅显示相机光轴深度小于此值的点，单位米，默认1.2；只影响预览')
    return p


def preview(data_dir, session, report, *, max_depth=1.2):
    # 只叠加同一个板位的配对帧，不能把不同板位的点云混叠后要求板重合。
    print('移板模式：仅显示第一板位的同时配对点云，两台相机均保留原始RGB颜色；检查同一板是否重合。')
    single = {**session, 'samples': session['samples'][:1]}
    candidate = {**report, 'T_sdk_board': report['boards'][0]['T_sdk_board']}
    print(f'预览深度范围：0 < Z < {max_depth:g} 米（各相机光轴深度）。')
    preview_session(data_dir, single, candidate, max_depth=max_depth)


def _evaluate(session):
    cfg = session['config']
    return evaluate_moving_board(session['samples'], session['reference_flange'], cfg['T_flange_wrist'],
        validation_poses=cfg['validation_poses'], max_mm=cfg['max_validation_mm'], max_deg=cfg['max_validation_deg'])


def save_results(data_dir, session, report, *, visual_confirmed):
    if not report['quality_passed']:
        raise ValueError('未通过独立板位检查，不能发布外参')
    _save_json(data_dir / 'third_view_extrinsic.json', standard_camera_parameters(data_dir, session['samples'][0]['third'], {
        'calibration_type': 'fixed_robot_moving_board_from_existing_handeye',
        'pose_convention': 'T_A_B maps B coordinates into A; translation in meter',
        'T_sdk_third': report['T_sdk_third'], 'T_third_sdk': report['T_third_sdk'],
        'T_sdk_wrist': report['T_sdk_wrist'], 'T_wrist_third': report['T_wrist_third'],
        'camera_serial': session['third_serial'], 'robot_group': session['config']['robot_group'],
        'handeye_file': session['config']['handeye_file'], 'source_session': 'session.json',
        'quality_report': 'quality_report.json', 'quality_passed': True,
        'open3d_visual_confirmed': visual_confirmed, 'communication': session['communication_actual'],
        'train_count': report['train_count'], 'validation_count': report['validation_count'],
        'accuracy_limit': report['note'],
    }))
    # 板移动了，每个板位必须独立保存，不能产出一个含糊的固定板外参。
    _save_json(data_dir / 'board_poses.json', {'pose_convention': 'T_A_B maps B into A; meter',
                                            'boards': report['boards']})
    publish_result(data_dir, 'third_camera')


def main(argv=None):
    args = _parser().parse_args(argv)
    if not np.isfinite(args.max_depth) or args.max_depth <= 0:
        raise ValueError('--max-depth必须为有限正数，单位米')
    resolve_cli_paths(args, "handeye", "data_dir")
    if args.preview_only:
        if args.data_dir is None:
            raise ValueError('--preview-only需要--data-dir')
        session = json.loads((args.data_dir / 'session.json').read_text())
        if session.get('calibration_type') != 'fixed_robot_moving_board':
            raise ValueError('该目录不是固定机器人移板标定')
        preview(args.data_dir, session, _evaluate(session), max_depth=args.max_depth)
        return 0
    if args.handeye is None:
        raise ValueError('请指定已有的--handeye')
    if args.resume and args.data_dir is None:
        raise ValueError('--resume需要明确指定--data-dir')
    if args.validation_poses < 1 or args.poses - args.validation_poses < 3:
        raise ValueError('总板位需要至少3个估计板位和1个独立验证板位')
    limits = [args.udp_timeout, args.max_joint_change_deg, args.max_reprojection_px,
              args.max_pair_gap_ms, args.max_validation_mm, args.max_validation_deg]
    if not np.isfinite(limits).all() or min(limits) <= 0:
        raise ValueError('采样与误差阈值必须是有限正数')
    if not args.no_preview or args.visualize:
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError('需要图形桌面；无桌面请加--no-preview并去掉--visualize')
    handeye = json.loads(args.handeye.read_text())
    if handeye.get('calibration_type') != 'eye_in_hand_fixed_board' or handeye.get('robot_group') != args.group:
        raise ValueError('手眼标定类型或机械臂group不一致')
    wrist = validate_pose(np.asarray(handeye['T_flange_camera']))
    args.ip = normalize_realman_ip(args.ip)
    provenance = handeye
    if not provenance.get('robot_sdk_ip'):
        source = args.handeye.parent / handeye.get('source_samples', 'samples.json')
        if not source.is_file():
            raise ValueError('手眼文件缺少robot_sdk_ip且来源samples.json不存在，无法核对机器人来源')
        provenance = json.loads(source.read_text())
    if (normalize_realman_ip(provenance.get('robot_sdk_ip', '')) != args.ip
            or provenance.get('robot_group') != args.group):
        raise ValueError('手眼标定来源机器人IP/group与当前不一致')
    pattern = tuple(handeye['pattern_size_inner_corners'])
    square = float(handeye['square_size_m'])
    board_object_points(pattern, square)
    if args.wrist_endpoint == args.third_endpoint or args.third_serial == handeye['camera_serial']:
        raise ValueError('需要两台不同相机、两个不同endpoint')
    args.data_dir = args.data_dir or default_run_dir("moving_board")
    config = {'handeye_file': str(args.handeye.resolve()), 'T_flange_wrist': wrist.tolist(),
        'robot_sdk_ip': args.ip, 'robot_group': args.group, 'port': args.port,
        'verified_handeye_robot_ip': args.ip,
        'poses': args.poses, 'validation_poses': args.validation_poses,
        'wrist_endpoint': args.wrist_endpoint, 'third_endpoint': args.third_endpoint,
        'wrist_serial': handeye['camera_serial'], 'pattern_size': pattern, 'square_size_m': square,
        'state_source': args.state_source, 'sdk_timeout_ms': args.sdk_timeout_ms,
        'udp_target_ip': args.udp_target_ip, 'udp_port': args.udp_port, 'udp_timeout': args.udp_timeout,
        'max_joint_change_deg': args.max_joint_change_deg, 'max_reprojection_px': args.max_reprojection_px,
        'max_pair_gap_ms': args.max_pair_gap_ms, 'max_validation_mm': args.max_validation_mm,
        'max_validation_deg': args.max_validation_deg}
    config['pattern_size'] = list(pattern)
    config = _jsonable(config)
    path = args.data_dir / 'session.json'
    if (args.data_dir / 'RUN_FAILED.json').exists():
        raise ValueError('目录已标失败，请新建目录')
    if (args.data_dir / 'third_view_extrinsic.json').exists():
        raise FileExistsError('已有正式结果；使用--preview-only预览或新建目录')
    if args.resume:
        session = json.loads(path.read_text())
        if session.get('calibration_type') != 'fixed_robot_moving_board' or not same_recorded_config(session['config'], config, ('handeye_file',)):
            raise ValueError('续采模式或参数不匹配')
        if args.third_serial and session['third_serial'] not in (None, args.third_serial):
            raise ValueError('续采第三相机序列号不匹配')
    else:
        if args.data_dir.exists() and any(args.data_dir.iterdir()):
            raise FileExistsError('输出目录非空，请换目录或加--resume')
        session = {'calibration_type': 'fixed_robot_moving_board', 'config': config,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'samples': [],
            'T_flange_wrist': wrist.tolist(), 'third_serial': args.third_serial}
    print('固定机器人移板标定：机械臂、手腕相机和第三相机全程不动，只移动棋盘格。')
    print('每个板位两相机都须看全板，停稳后空格取样，松开约1秒再按空格确认；Q退出。')
    print(f'共{args.poses}个板位：前{args.poses-args.validation_poses}个估计，后{args.validation_poses}个独立验证。')
    print(f'程序不发送运动；状态源{args.state_source}。已有手眼误差不能通过本模式独立检出。')
    with ExitStack() as stack:
        cameras = {}
        for name, endpoint in [('wrist', args.wrist_endpoint), ('third', args.third_endpoint)]:
            cam = CameraClient(endpoint)
            stack.callback(cam.close)
            _check_camera_health(cam, name='手腕' if name == 'wrist' else '第三视角', endpoint=endpoint)
            cameras[name] = cam
        robot = stack.enter_context(connect_realman_read_only(args.ip, group=args.group, port=args.port,
            state_source=args.state_source, sdk_timeout_ms=args.sdk_timeout_ms,
            udp_target_ip=args.udp_target_ip, udp_port=args.udp_port, udp_timeout_s=args.udp_timeout,
            connect_retries=args.connect_retries, connect_retry_interval_s=args.connect_retry_interval))
        flange, joints = robot.read_flange_pose()
        if args.resume:
            check_fixed_robot({'T_sdk_flange': flange, 'joints_rad': joints},
                              session['reference_flange'], session['reference_joints'])
        else:
            session['reference_flange'] = flange.tolist()
            session['reference_joints'] = joints.tolist()
        session['communication_actual'] = robot.communication_info
        args.data_dir.mkdir(parents=True, exist_ok=True)
        _save_json(path, session)
        print(f'输出目录：{args.data_dir}；保持相机与机器人不动，同一命令加--data-dir此目录 --resume可续采。')
        while len(session['samples']) < args.poses:
            i = len(session['samples'])
            purpose = '独立验证' if i >= args.poses - args.validation_poses else '估计'
            print(f'\n板位{i+1}/{args.poses}（{purpose}）：请移动板，和已有板位至少相差3cm或5°，然后固定停稳。')
            with _capture_window(not args.no_preview):
                pair = _prompt_pair(cameras, robot, args, stage=f'BOARD {i+1}/{args.poses}',
                    pattern_size=pattern, square_size=square, wrist_serial=handeye['camera_serial'],
                    third_serial=session['third_serial'], data_dir=args.data_dir)
            if pair is None:
                return 1
            # 机器人移动应终止本轮而非把新的相机位姿混入数据。
            check_fixed_robot(pair, session['reference_flange'], session['reference_joints'])
            try:
                check_new_board(pair, session['samples'])
            except ValueError as exc:
                print(f'未收录：{exc}', flush=True)
                continue
            for name in ('wrist', 'third'):
                reference = (session['samples'][0][name]['K'] if session['samples'] else
                             handeye.get('intrinsic_matrix') if name == 'wrist' else None)
                if reference is not None and not np.allclose(reference, pair[name]['K'], atol=1e-6, rtol=0):
                    raise ValueError(f'{name}内参与标定/此前板位不一致')
            metadata = _record_pair(args.data_dir, f'board_{i+1:03d}', pair)
            metadata['purpose'] = purpose
            session['samples'].append(metadata)
            session['third_serial'] = pair['third']['serial']
            _save_json(path, session)
            print(f'已收录板位{i+1}。')
    report = _evaluate(session)
    _save_json(args.data_dir / 'quality_report.json', report)
    for c in report['checks']:
        print(f"板位{c['index']} {'验证' if c['holdout'] else '估计'}：{c['translation_mm']:.2f} mm / {c['rotation_deg']:.2f}°")
    print(f"独立验证最大偏差：{report['validation_translation_max_mm']:.2f} mm / {report['validation_rotation_max_deg']:.2f}°。")
    if args.visualize:
        preview(args.data_dir, session, report, max_depth=args.max_depth)
    if not report['quality_passed']:
        print('未通过质量检查，保留原始样本和候选，不生成正式外参。')
        return 2
    if input('检查完成后输入 ACCEPT 保存正式外参；其他输入取消：').strip() != 'ACCEPT':
        return 1
    save_results(args.data_dir, session, report, visual_confirmed=args.visualize)
    print(f'已保存：{args.data_dir / "third_view_extrinsic.json"}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('已中断；原始数据已保留。')
        raise SystemExit(130)
