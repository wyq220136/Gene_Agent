"""微波炉最终状态判据；放置需原操作提供的物理落稳证据。"""
import numpy as np


def table_place_postcondition(checks):
    """用原始接触与位姿证据复核落稳；不读取策略的 success 字段。"""
    return bool(checks["settled_table_support"]["supported"]
                and checks["final_table_support"]["supported"]
                and checks["settle_translation_m"] <= checks["maximum_allowed_translation_m"]
                and checks["retreat_translation_m"] <= checks["maximum_allowed_translation_m"]
                and checks["final_bowl_tilt_deg"] <= checks["maximum_allowed_tilt_deg"]
                and checks["simulation_attachment_released"])

def evaluate_final(door_angle, knob_angle, knob_target, bowl_on_table, attachments):
    return {
        "door_closed": bool(abs(door_angle) <= np.deg2rad(5.)),
        "knob_at_target": bool(abs(knob_angle - knob_target)
                               <= np.deg2rad(3.)),
        "bowl_on_table": bool(bowl_on_table),
        "no_simulation_attachment": not attachments,
        "door_angle_deg": float(np.degrees(door_angle)),
        "knob_angle_deg": float(np.degrees(knob_angle)),
    }
