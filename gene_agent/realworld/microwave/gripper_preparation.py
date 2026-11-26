"""现场抓碗前的显式右夹爪初始化；不依赖初始场景或起点开口读取。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time


def prepare_right_gripper(gripper, *, output_dir):
    """首个操作直接发800，之后确认实测到位；不移动机械臂、不返程。

    由入口仅在实际执行抓碗/装碗时调用。通信建立和本地参数校验
    不属于现场状态检测；此函数发令前不读取任何夹爪反馈。
    """
    maximum = float(gripper.max_width_m)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError('夹爪最大开口配置无效')
    target = maximum * .8
    path = Path(output_dir) / 'right_gripper_preparation.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='command_pending', target_sdk_position=800,
                  target_width_m=target, group='right', readings=[],
                  tolerance_sdk_units=10, stability_sdk_units=5,
                  required_stable_samples=3, feedback_timeout_s=3.)

    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')

    save()
    try:
        print('右夹爪初始化：先下发800，再读取反馈确认；尚未进行场景检测。', flush=True)
        gripper.command_width_m(target, blocking=True)
        report['status'] = 'confirming'
        start = time.monotonic()
        while True:
            width = float(gripper.read_width_m())
            raw = getattr(gripper, 'last_reading', None)
            reading = dict(width_m=width, sdk_reading=raw, elapsed_s=time.monotonic()-start)
            report['readings'].append(reading)
            if not math.isfinite(width):
                raise ValueError('初始化反馈不是有限宽度')
            recent = [r['width_m'] for r in report['readings'][-3:]]
            if (len(recent) == 3 and all(abs(w-target) <= maximum*.01 for w in recent)
                    and max(recent)-min(recent) <= maximum*.005):
                report.update(status='confirmed', measured_end_width_m=width)
                print('右夹爪800初始化到位，开始后续现场检测。', flush=True)
                return report
            if time.monotonic()-start >= 3.:
                raise TimeoutError('右夹爪未稳定到800±10，停止后续检测和执行')
            save()
            time.sleep(.1)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
