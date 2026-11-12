"""把 Codex 接触事实归纳成抓取调试所需的明确类别。"""
from __future__ import annotations


def classify_grasp_contacts(contacts, gripper_descriptor: dict, *, object_body: str) -> dict:
    pads = set(gripper_descriptor.get("pad_geom_names", []))
    physical = set(gripper_descriptor.get("collision_geom_names", []))
    finger_pads = {
        side: gripper_descriptor.get("fingers", {}).get(side, {}).get("pad_geom_name")
        for side in ("positive", "negative")
    }
    relevant = []
    undesired = []
    sides = {side: False for side in finger_pads}
    for contact in contacts:
        geom_names = set(contact.get("geom_names", []))
        if object_body not in contact.get("body_names", ()) or not geom_names.intersection(physical):
            continue
        relevant.append(contact)
        gripper_geoms = geom_names.intersection(physical)
        if gripper_geoms - pads:
            undesired.append(contact)
        for side, pad_name in finger_pads.items():
            if pad_name and pad_name in geom_names:
                sides[side] = True
    bilateral = all(sides.values())
    return {
        "success": bilateral and not undesired,
        "bilateral_pad_contact": bilateral,
        "pad_sides": sides,
        "contact_count": len(relevant),
        "undesired_gripper_contacts": undesired,
        "contacts": relevant,
    }


__all__ = ["classify_grasp_contacts"]
