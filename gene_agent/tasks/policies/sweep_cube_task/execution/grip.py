"""对激活时的 weld 基准检查累计持刷误差，不把暂时变形当作新夹持。"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def measure_weld_drift(model, data, samples, *, name="sim_attach_left_brush",
                       position_limit_m=.003, rotation_limit_rad=np.deg2rad(5.)):
    eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
    if eq < 0 or not data.eq_active[eq]:
        return None
    body1, body2 = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
    reference_pos = model.eq_data[eq, 3:6]
    reference_rot = Rotation.from_quat(model.eq_data[eq, 6:10][[1, 2, 3, 0]]).as_matrix()
    preview = mujoco.MjData(model)
    mujoco.mj_copyData(preview, model, data)
    preview.eq_active[:] = 0
    errors = []
    for index, sample in enumerate(samples):
        preview.qpos[:] = sample["qpos"]
        mujoco.mj_forward(model, preview)
        r1, r2 = (preview.xmat[b].reshape(3, 3) for b in (body1, body2))
        position = r1.T @ (preview.xpos[body2] - preview.xpos[body1])
        errors.append({"sample_index": index,
            "position_error_m": float(np.linalg.norm(position - reference_pos)),
            "rotation_error_rad": float(Rotation.from_matrix(reference_rot.T @ r1.T @ r2).magnitude())})
    if not errors:
        return None
    pos = max(r["position_error_m"] for r in errors)
    rot = max(r["rotation_error_rad"] for r in errors)
    return {"reference": "weld_activation_relative_body_pose", "eq_data": model.eq_data[eq].tolist(),
            "solref": model.eq_solref[eq].tolist(), "solimp": model.eq_solimp[eq].tolist(),
            "position_limit_m": position_limit_m, "rotation_limit_rad": float(rotation_limit_rad),
            "max_position_error_m": pos, "max_rotation_error_rad": rot,
            "success": bool(pos <= position_limit_m and rot <= rotation_limit_rad), "samples": errors}
