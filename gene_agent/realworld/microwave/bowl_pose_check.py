"""抓碗输入的一致性检查；不猜测/翻转 FoundationPose，不修改目标。"""
from pathlib import Path
import json
import numpy as np
from gene_agent.knowledge.geometry import validate_pose


class BowlPoseTableError(ValueError):
    code = "bowl_pose_table_inconsistent"

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(
            "碗感知与桌面坐标不一致：开口轴对桌面上方向投影 "
            f"{diagnostics['opening_axis_table_up_dot']:.4f}，碗沿中心低于桌面 "
            f"{-diagnostics['rim_height_above_table_m'] * 1000:.2f} mm；"
            "不能靠抬高 TCP 修正物体位姿。请核对相机外参、碗 mesh frame 和初始注册 pose，重新准备感知。")


def check_bowl_pose_table(bowl_pose_sdk, table_pose_sdk, thickness_m, *, table_band_m=3 * .005):
    """只拒绝两项同时矛盾的正放碗输入：开口朝下且整个中心碗沿在支撑面下。

    +Y 是当前真机 mesh 的碗口方向，Concept 的 Ring +Z 为 -Y。
    容许带沿用已有桌面拟合的 3×5mm 带，不把正常倾斜或单项高度抖动拒绝。
    此检查不能判定是标定、注册还是桌面估计错误，更不能自动翻转/平移物体。
    """
    bowl = validate_pose(bowl_pose_sdk, field_name="T_sdk_bowl")
    table = validate_pose(table_pose_sdk, field_name="T_sdk_table")
    thickness = float(thickness_m)
    band = float(table_band_m)
    if not np.isfinite([thickness, band]).all() or thickness <= 0 or band < 0:
        raise ValueError("桌面厚度和拟合带必须有限且有效")
    path = Path(__file__).resolve().parents[3] / "assets/objects/bowl_concept.json"
    entries = json.loads(path.read_text())["conceptualization"]
    ring = next(item["parameters"] for item in entries if item["template"] == "Ring")
    concept = np.asarray(ring["pose_matrix"], float)
    up = table[:3, 2]
    top = table[:3, 3] + up * thickness / 2
    rim = bowl @ concept
    opening = -rim[:3, 2]
    alignment = float(opening @ up)
    height = float((rim[:3, 3] - top) @ up)
    half_height = float(ring["size"][2]) / 2
    diagnostics = dict(code=BowlPoseTableError.code, T_sdk_bowl=bowl.tolist(),
        T_sdk_table=table.tolist(), table_top_sdk_m=top.tolist(),
        rim_center_sdk_m=rim[:3, 3].tolist(), opening_axis_sdk=opening.tolist(),
        opening_axis_table_up_dot=alignment, rim_height_above_table_m=height,
        rim_half_height_m=half_height, table_uncertainty_band_m=band,
        table_band_source="existing table fit: 3 * 0.005m",
        check_frame="right_sdk_base", pose_modified=False,
        rule="opening faces below support AND rim center below support by more than rim half height + fit band")
    diagnostics["valid"] = not (alignment < 0 and height < -(half_height + band))
    if not diagnostics["valid"]:
        raise BowlPoseTableError(diagnostics)
    return diagnostics
