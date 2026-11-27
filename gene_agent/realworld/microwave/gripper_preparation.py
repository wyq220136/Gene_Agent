"""所有现场任务执行前的显式右夹爪初始化；不依赖起点开口读取。"""
from __future__ import annotations

import json
import math
from pathlib import Path

from gene_agent.realworld.gripper_state import invalidate_endpoint


def prepare_right_gripper(gripper, *, output_dir):
    """首个操作阻塞下发800，不追加开口读取或到位复核。

    由入口在任意任务的实际执行启动时调用一次。通信建立和本地参数校验
    不属于现场状态检测；此函数不读取任何夹爪反馈；后续场景独立读取实测状态。
    """
    maximum = float(gripper.max_width_m)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError('夹爪最大开口配置无效')
    target = maximum * .8
    path = Path(output_dir) / 'right_gripper_preparation.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='command_pending', target_sdk_position=800,
                  target_width_m=target, group='right', readings=[],
                  endpoint_verified=False, feedback_method="disabled",
                  measured_end_width_m=None)

    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')

    save()
    try:
        print('右夹爪初始化：阻塞下发800，不追加开口到位检查。', flush=True)
        invalidate_endpoint(gripper)
        gripper.command_width_m(target, blocking=True)
        # SDK 正常返回只证明命令完成；不能伪造实测800或发布已确认终态。
        report['status'] = 'command_completed'
        print('右夹爪800命令已返回，开始后续现场检测。', flush=True)
        return report
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
