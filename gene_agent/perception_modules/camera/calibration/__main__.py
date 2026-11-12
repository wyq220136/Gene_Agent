"""标定统一入口；只分发明确选择的流程，帮助页面不加载硬件依赖。"""
from __future__ import annotations

from importlib import import_module
import sys

WORKFLOWS = {
    'handeye': ('guided_wrist_handeye', '路线B 第一步：固定板、移动手腕，求手眼变换'),
    'moving-board': ('guided_moving_board_extrinsic', '路线B 第二步：固定机器人、移动板，求第三相机基座外参'),
    'manual': ('guided_single_arm_calibration', '路线A：人工对齐量尺，含板相机采集和基座外参组合'),
    'board-camera': ('guided_board_camera', '路线A 可选独立步骤：仅求板相机关系，或组合已知板基座关系'),
    'inspect': ('inspection.viewer', '标定后检查：基座XYZ裁剪、RGB覆盖与桌面平面诊断'),
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ('-h', '--help'):
        print('相机标定：python -m gene_agent.perception_modules.camera.calibration <步骤> [参数]')
        print('优先路线B：handeye → moving-board；已有handeye.json可直接进行第二步。')
        for name, (_, description) in WORKFLOWS.items():
            print(f'  {name:14s} {description}')
        print('  results        查看最近正式标定的固定入口与原始目录')
        print('每一步加 --help 查看参数；说明：gene_agent/perception_modules/camera/calibration/README.md')
        print('输出统一放在data/calibration/runs/<用途>/，诊断放diagnostics/，不散放data根目录。')
        return 0
    command = arguments.pop(0)
    if command == 'results':
        from .storage import show_results
        return show_results()
    if command not in WORKFLOWS:
        raise SystemExit(f'未知标定步骤 {command!r}；使用 --help 查看路线A/B')
    module = import_module(f'{__package__}.{WORKFLOWS[command][0]}')
    return module.main(arguments)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('已中断；已保存的采集数据保留。')
        raise SystemExit(130)
