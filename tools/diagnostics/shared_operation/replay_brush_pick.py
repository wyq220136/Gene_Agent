"""离线重放保存的首帧抓取规划；无相机、SDK、串口或真机执行路径。"""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-after', choices=['pregrasp', 'approach', 'pick'], default='approach')
    args = parser.parse_args(argv)
    import mujoco
    import numpy as np
    from gene_agent.realworld.sweep.table_preview import preview_table_task
    source, output = args.run.resolve(), args.output.resolve()
    record = json.loads((source/'result.json').read_text(encoding='utf-8'))
    site = json.loads((source/'site_input.json').read_text(encoding='utf-8'))
    if record.get('task') != 'sweep_realworld_plan_only':
        parser.error('--run 必须是现场首帧预览目录')
    output.mkdir(parents=True, exist_ok=False)
    for name in ['site_input.json', 'observation.json', 'camera_calibration.json']:
        if (source/name).exists(): shutil.copy2(source/name, output/name)
    # 模型资产仍可引用源run的绝对路径，不改写原始观测和历史输出。
    shutil.copytree(source/'scene', output/'scene')
    for name in ['perception', 'capture']:
        if (source/name).exists(): shutil.copytree(source/name, output/name)
    model = mujoco.MjModel.from_xml_path(str(output/'scene/scene.xml'))
    data = mujoco.MjData(model)
    data.qpos[:] = np.load(output/'scene/initial_state.npz')['qpos']
    mujoco.mj_forward(model, data)
    robot = json.loads((output/'scene/robot/robot_metadata.json').read_text(encoding='utf-8'))
    site['support_adjustment'] = record.get('support_adjustment')
    result = preview_table_task(model, data, robot, site, output, stop_after=args.stop_after)
    (output/'offline_replay.json').write_text(json.dumps({'source_run': str(source),
        'hardware_connected': False, 'observation_reused': True}, ensure_ascii=False, indent=2))
    print(f"范围={args.stop_after} 完成={result['requested_scope_complete']}；纯离线预测：{output}")
    return 0 if result['requested_scope_complete'] else 2


if __name__ == '__main__': raise SystemExit(main())
