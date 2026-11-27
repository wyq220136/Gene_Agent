"""所有现场任务执行前的显式右夹爪初始化；不依赖起点开口读取。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from gene_agent.realworld.gripper_state import invalidate_endpoint


def prepare_right_gripper(gripper, *, output_dir):
    """首个操作直接发800，之后确认实测到位；不移动机械臂、不返程。

    由入口在任意任务的实际执行启动时调用一次。通信建立和本地参数校验
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
                  tolerance_sdk_units=10, feedback_method="blocking_command_plus_single_actpos")

    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')

    save()
    try:
        print('右夹爪初始化：先下发800，再读取反馈确认；尚未进行场景检测。', flush=True)
        invalidate_endpoint(gripper)
        gripper.command_width_m(target, blocking=True)
        report['status'] = 'confirming'
        start = time.monotonic()
        width = float(gripper.read_width_m())
        raw = getattr(gripper, 'last_reading', None)
        report['readings'].append(dict(width_m=width, sdk_reading=raw, elapsed_s=time.monotonic()-start))
        if not math.isfinite(width):
            raise ValueError('初始化反馈不是有限宽度')
        if abs(width-target) > maximum*.01:
            raise ValueError('右夹爪阻塞命令返回后未到800±10，停止后续检测和执行')
        gripper.confirmed_endpoint_width_m = width
        report.update(status='confirmed', measured_end_width_m=width)
        print('右夹爪800初始化到位，开始后续现场检测。', flush=True)
        return report
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
