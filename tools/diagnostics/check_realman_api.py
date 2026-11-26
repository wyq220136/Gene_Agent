"""RealMan API 基础自检：默认只读，显式启用单关节/夹爪微动。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ip', required=True, help='一侧机械臂控制器 IPv4')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--group', required=True, choices=['left', 'right'])
    p.add_argument('--action', choices=['read', 'joint', 'gripper'], default='read')
    p.add_argument('--execute', action='store_true', help='明确允许所选微动')
    p.add_argument('--joint', type=int, default=7, choices=range(1, 8))
    p.add_argument('--delta-deg', type=float, default=0.5, help='有符号角度，绝对值 0.2～1 度')
    p.add_argument('--gripper-step', type=int, default=5, help='有符号 SDK 行程，绝对值 1～10；正值张开')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--samples', type=int, default=1, help='只读采样轮数，1～20；各轮间隔0.5秒')
    return p


def validate(args):
    import ipaddress
    ipaddress.IPv4Address(args.ip)
    if not 1 <= args.port <= 65535:
        raise ValueError('端口越界')
    if not 1 <= args.samples <= 20 or (args.action != 'read' and args.samples != 1):
        raise ValueError('samples 必须为1～20，多轮采样仅用于 read 模式')
    if args.action != 'read' and not args.execute:
        raise ValueError('运动模式必须显式提供 --execute')
    if args.action == 'read' and args.execute:
        raise ValueError('read 模式不接受 --execute')
    if not math.isfinite(args.delta_deg) or not 0.2 <= abs(args.delta_deg) <= 1:
        raise ValueError('delta-deg 绝对值必须为 0.2～1 度')
    if not 1 <= abs(args.gripper_step) <= 10:
        raise ValueError('gripper-step 绝对值必须为 1～10')


class Probe:
    def __init__(self, sdk, report, output):
        self.sdk, self.report, self.output = sdk, report, output

    def save(self):
        self.output.write_text(json.dumps(self.report, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')

    def call(self, name, *args):
        row = {'api': name, 'args': args, 'time_unix': time.time()}
        self.report['calls'].append(row)
        self.save()  # 下发之前记录意图，异常退出也能定位最后一个 API。
        try:
            value = getattr(self.sdk, name)(*args)
            row['raw'] = {'id': value.id} if name == 'rm_create_robot_arm' else value
            print(f'{name}: {value!r}', flush=True)
            return value
        except BaseException as exc:
            row['exception'] = repr(exc)
            raise
        finally:
            self.save()

    def query(self, name):
        value = self.call(name)
        if not isinstance(value, tuple) or len(value) != 2 or value[0] != 0:
            hint = '（SDK 数据解析失败；随附数值不可作为实测状态）' if isinstance(value, tuple) and value and value[0] == -3 else ''
            raise ValueError(f'{name} 返回失败或格式异常{hint}: {value!r}')
        return value[1]

    def joints(self):
        values = self.query('rm_get_joint_degree')
        if len(values) != 7 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f'七轴关节状态无效: {values!r}')
        return list(map(float, values))

    def gripper(self):
        state = self.query('rm_get_gripper_state')
        # SDK 返回成功不等于末端夹爪在线/可用。
        if not isinstance(state, dict):
            raise ValueError(f'夹爪状态格式无效: {state!r}')
        # API2 Python 的字段为 enable_state；旧 JSON 协议使用 enable。
        enable_key = 'enable_state' if 'enable_state' in state else 'enable'
        if 'enable_state' in state and 'enable' in state and state['enable_state'] != state['enable']:
            raise ValueError(f'夹爪使能字段冲突; raw={state!r}')
        for key, expected in [('status', 1), (enable_key, 1), ('error', 0)]:
            if state.get(key) != expected:
                raise ValueError(f'夹爪 {key}={state.get(key)!r}，期望 {expected}; raw={state!r}')
        position = state.get('actpos')
        if isinstance(position, bool) or not isinstance(position, (float, int)) or not math.isfinite(position) or not -10 <= position <= 1500:
            raise ValueError(f'夹爪 actpos 无效，反馈范围应为 -10～1500; raw={state!r}')
        return state

    def command(self, name, *args):
        # 接收反馈范围独立于指令范围，防止返程误下发1003等反馈值。
        if name == 'rm_set_gripper_position':
            target = args[0]
            if (isinstance(target, bool) or not isinstance(target, (int, float))
                    or not math.isfinite(target) or not 0 <= target <= 1000):
                raise ValueError(f'夹爪位置指令必须在 [0, 1000]: {target!r}')
        code = self.call(name, *args)
        if code != 0:
            raise RuntimeError(f'{name} 失败: {code!r}')

    def move_joint(self, target, start):
        self.command('rm_movej', target, 1, 0, 0, 0)  # 1% 速度，非阻塞；度制。
        deadline, stable = time.monotonic() + 10, 0
        while time.monotonic() < deadline:
            current = self.joints()
            if any(q < min(a, b) - .15 or q > max(a, b) + .15
                   for q, a, b in zip(current, start, target)):
                raise RuntimeError('实测关节超出微动区间，停止且不自动返程')
            stable = stable + 1 if max(abs(a - b) for a, b in zip(current, target)) <= .08 else 0
            if stable >= 3:
                return
            time.sleep(.1)
        raise TimeoutError('10秒内未稳定到位，停止且不自动返程')

    def move_gripper(self, target):
        self.command('rm_set_gripper_position', target, True, 3)
        for _ in range(3):
            state = self.gripper()
            if state.get('mode') not in (1, 2, 3) or abs(state['actpos'] - target) > 2:
                raise RuntimeError(f'夹爪未稳定到位，不自动返程: {state!r}')
            time.sleep(.1)


def run(args, sdk):
    validate(args)
    output = args.output or Path(__file__).resolve().parents[2] / 'runs/tmp/realman_api' / time.strftime('%Y%m%d-%H%M%S') / f'{args.group}-{time.time_ns()}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'group': args.group, 'ip': args.ip, 'action': args.action,
              'options': vars(args), 'status': 'running', 'calls': [], 'checks': {},
              'schema_version': 3, 'samples': [], 'metadata': {}}
    probe = Probe(sdk, report, output)
    connected, moving = False, False
    try:
        probe.call('rm_set_timeout', 2000)
        handle = probe.call('rm_create_robot_arm', args.ip, args.port, 3)
        if getattr(handle, 'id', -1) <= 0:
            raise RuntimeError(f'连接失败: handle_id={getattr(handle, "id", None)}')
        connected = True
        # 版本采集为辅助证据；缺少该接口不会覆盖主要状态检查结果。
        for name in ['rm_api_version', 'rm_get_arm_software_info']:
            if callable(getattr(sdk, name, None)):
                try:
                    report['metadata'][name] = probe.call(name)
                except Exception as exc:
                    report['metadata'][name] = {'error': repr(exc)}
        for sample_index in range(args.samples):
            checks = {}
            report['samples'].append({'index': sample_index, 'checks': checks})
            for label, query in [('joints', probe.joints), ('gripper', probe.gripper)]:
                try:
                    checks[label] = {'ok': True, 'value': query()}
                except Exception as exc:
                    checks[label] = {'ok': False, 'error': str(exc)}
            if sample_index == 0:
                report['checks'] = checks  # 保留旧报告入口，详细结果见 samples。
            probe.save()
            if sample_index + 1 < args.samples:
                time.sleep(.5)
        if not all(c['ok'] for sample in report['samples'] for c in sample['checks'].values()):
            raise RuntimeError('初始状态检查失败；详细原始返回见 calls，不发送运动')
        if args.action == 'joint':
            start = probe.joints()
            time.sleep(.2)
            if max(abs(a-b) for a,b in zip(start, probe.joints())) > .08:
                raise RuntimeError('机械臂当前不稳定，不发送运动')
            target = start.copy()
            target[args.joint - 1] += args.delta_deg
            moving = True
            probe.move_joint(target, start)
            probe.move_joint(start, target)
            moving = False
        elif args.action == 'gripper':
            state = probe.gripper()
            if state.get('mode') not in (1, 2, 3):
                raise RuntimeError('夹爪非空闲状态，拒绝微动')
            if not 0 <= state['actpos'] <= 1000:
                raise ValueError('起点反馈超出0～1000指令范围，无法精确往返，拒绝夹爪微动')
            start = int(round(state['actpos']))
            target = start + args.gripper_step
            if not 0 <= target <= 1000:
                raise ValueError('目标行程越界，请改变 gripper-step 的符号')
            probe.move_gripper(target)
            probe.move_gripper(start)
        report['status'] = 'passed'
    except BaseException as exc:
        report['status'], report['error'] = 'failed', repr(exc)
        if moving:
            for name in ['rm_set_arm_stop', 'rm_set_arm_delete_trajectory']:
                try:
                    probe.command(name)
                except BaseException as stop_exc:
                    report.setdefault('stop_errors', []).append(repr(stop_exc))
        if args.action == 'gripper':
            report['note'] = '夹爪异常时不自动反向；此工具不保证夹爪急停，请使用现场急停。'
    finally:
        if connected:
            try:
                probe.command('rm_delete_robot_arm')
            except BaseException as exc:
                report['status'], report['disconnect_error'] = 'failed', repr(exc)
        probe.save()
        print(f"{report['status']}: {output}", flush=True)
    return 0 if report['status'] == 'passed' else 1


def main():
    p = parser()
    args = p.parse_args()
    try:
        validate(args)
    except ValueError as exc:
        p.error(str(exc))
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    return run(args, RoboticArm(rm_thread_mode_e(2)))


if __name__ == '__main__':
    sys.exit(main())
