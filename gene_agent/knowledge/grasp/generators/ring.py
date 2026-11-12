"""Ring part-local 抓取公式。"""
from __future__ import annotations
import itertools
import numpy as np
from gene_agent.knowledge.geometry import RingConcept
from gene_agent.knowledge.grasp.model import GraspProfile
from .common import LocalGraspCandidate, LocalGraspRejection, rotation_xyz, values

def generate_ring_local(
    concept: RingConcept, profile: GraspProfile
) -> tuple[LocalGraspCandidate | LocalGraspRejection, ...]:
    """围绕圆环生成 TCP；``+X`` 是 opening，``+Z`` 是 approach。"""
    if not isinstance(concept, RingConcept):
        raise TypeError("ring generator 要求 RingConcept")
    axis = np.asarray(concept.axis, dtype=float)
    reference = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) <= 0.9 else np.array([0.0, 1.0, 0.0])
    basis_x = reference - np.dot(reference, axis) * axis
    basis_x /= np.linalg.norm(basis_x)
    basis = np.column_stack((basis_x, np.cross(axis, basis_x), axis))
    # 旧 profile 的 depth_m 仅作读取兼容；新配置统一用轴向偏移。
    # 两者本来在公式中直接相加，允许同时指定会产生重复控制。
    if "depth_m" in profile.parameters and "vertical_offset_m" in profile.parameters:
        raise ValueError("ring 的 depth_m 与 vertical_offset_m 不能同时指定")
    offset_name = "vertical_offset_m" if "vertical_offset_m" in profile.parameters else "depth_m"
    names = ("surface", offset_name, "azimuth_rad", "tip_x_rad", "tip_y_rad")
    sequences = (values(profile.parameters, names[0], ("top",)), values(profile.parameters, names[1], (concept.height_m / 2.0,)), values(profile.parameters, names[2], (0.0,)), values(profile.parameters, names[3], (0.0,)), values(profile.parameters, names[4], (0.0,)))
    clearance = profile.closing_width.clearance_m or 0.0
    width = concept.outer_radius_m - concept.inner_radius_m + 2.0 * clearance
    radius = (concept.outer_radius_m + concept.inner_radius_m) / 2.0
    output: list[LocalGraspCandidate | LocalGraspRejection] = []
    for indices in itertools.product(*(range(len(sequence)) for sequence in sequences)):
        surface, offset, azimuth, tip_x, tip_y = (sequence[index] for sequence, index in zip(sequences, indices))
        if surface not in {"top", "bottom"}:
            raise ValueError("ring surface 只支持 top 或 bottom")
        if offset_name == "depth_m" and float(offset) < 0.0:
            raise ValueError("旧 ring depth_m 必须是非负数")
        raw_angle = float(azimuth)
        parameter_indices = dict(zip(names, indices))
        # legacy Ring 的存在区间固定从局部方位角 0 开始，不存在额外 start angle。
        # 完整环按周期解释任意有限角；部分圆弧则检查 profile 中声明的原始方位角，
        # 避免负角被取模后错误落入另一端。0 弧没有任何可抓取方位，包括端点 0。
        full_circle = concept.angle_span_rad == 2.0 * np.pi
        inside_arc = (
            full_circle
            or (
                concept.angle_span_rad > 0.0
                and 0.0 <= raw_angle <= concept.angle_span_rad
            )
        )
        if not inside_arc:
            empty_arc = concept.angle_span_rad == 0.0
            output.append(
                LocalGraspRejection(
                    code="RING_AZIMUTH_OUT_OF_ARC",
                    message=(
                        "Ring 是 0 弧度空弧，不能生成抓取候选"
                        if empty_arc
                        else "ring azimuth_rad 位于局部圆弧 [0, angle_span_rad] 之外"
                    ),
                    context={
                        "azimuth_rad": raw_angle,
                        "angle_span_rad": concept.angle_span_rad,
                    },
                    parameter_indices=parameter_indices,
                )
            )
            continue
        angle = raw_angle % (2.0 * np.pi) if full_circle else raw_angle
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        tcp_z = np.array([0.0, 0.0, -1.0 if surface == "top" else 1.0])
        old_rotation = np.column_stack((np.cross(radial, tcp_z), radial, tcp_z))
        rotation = old_rotation @ rotation_xyz((0.0, 0.0, np.pi / 2.0)) @ rotation_xyz((float(tip_x), float(tip_y), 0.0))
        height = concept.height_m / 2.0 - float(offset) if surface == "top" else -concept.height_m / 2.0 + float(offset)
        pose = np.eye(4); pose[:3, :3] = basis @ rotation; pose[:3, 3] = basis @ (radial * radius + [0.0, 0.0, height])
        parameter_values = (surface, float(offset), float(azimuth), float(tip_x), float(tip_y))
        output.append(LocalGraspCandidate(pose, width, dict(zip(names, parameter_values)), parameter_indices))
    return tuple(output)

__all__ = ["generate_ring_local"]
