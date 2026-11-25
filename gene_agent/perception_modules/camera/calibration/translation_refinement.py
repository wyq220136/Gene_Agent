"""只修改 SDK X/Z 平移的标定候选；保持原始测量和旋转，输出新目录。"""
from pathlib import Path
import json
import hashlib
import numpy as np


def save_translation_candidate(camera_params, delta_sdk, output):
    source = Path(camera_params).resolve()
    delta = np.asarray(delta_sdk, float)
    if delta.shape != (3,) or not np.isfinite(delta).all() or delta[1] != 0:
        raise ValueError("本工具只允许有限的 SDK X/Z 平移，Y 必须为零")
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError("候选目录已存在，请换新目录，不能覆盖标定或历史候选")
    camera = json.loads(source.read_text())
    board = json.loads((source.parent/"board_to_robot.json").read_text())
    pose = np.asarray(camera["extrinsic_matrix"], float)
    if not np.allclose(pose, camera["right_extrinsic_matrix"], atol=1e-12):
        raise ValueError("右臂外参字段不一致")
    corrected = pose.copy(); corrected[:3,3] += delta
    sdk_board = np.asarray(board["board_to_robot"], float); sdk_board[:3,3] += delta
    board["board_to_robot"] = sdk_board.tolist()
    board["robot_to_board"] = np.linalg.inv(sdk_board).tolist()
    board["translation_refinement_sdk_m"] = (np.asarray(board.get("translation_refinement_sdk_m", [0,0,0]))+delta).tolist()
    record = dict(source_camera_params=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  delta_sdk_m=delta.tolist(), axes="right SDK base, meters", rotation_changed=False,
                  sdk_y_changed=False, measured_board_tcp_changed=False, validation="manual candidate; new observation required")
    camera["extrinsic_matrix"] = corrected.tolist()
    camera["right_extrinsic_matrix"] = corrected.tolist()
    camera["board_to_robot"] = sdk_board.tolist()
    record["previous_calibration_correction"] = camera.get("calibration_correction")
    camera["calibration_correction"] = dict(report_ref="translation_refinement.json", realworld_validation=record["validation"])
    camera["translation_refinement"] = record
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("camera_to_board.json", "sdk_tcp_reading.json"):
        path = source.parent/name
        if path.is_file(): (destination/name).write_bytes(path.read_bytes())
    for name, value in (("cam_param.json",camera),("board_to_robot.json",board),("translation_refinement.json",record)):
        (destination/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return destination/"cam_param.json"
