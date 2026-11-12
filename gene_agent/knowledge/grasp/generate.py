"""统一 grasp 生成入口：集中处理 frame 链、后处理和 proposal 封装。"""
from __future__ import annotations
import numpy as np
from gene_agent.knowledge.geometry import validate_pose
from .generators import GENERATOR_REGISTRY
from .generators.common import LocalGraspRejection, rotation_xyz
from .model import CandidateDiagnostic, GraspGenerationResult, GraspProfile, GraspProposal, build_candidate_id

def _apply_postprocess(T_object_tcp: np.ndarray, profile: GraspProfile) -> np.ndarray:
    pose = np.array(T_object_tcp, dtype=float, copy=True)
    for step in profile.postprocess:
        delta = np.eye(4)
        if step.type == "translate":
            delta[:3, 3] = step.xyz_m
        else:
            delta[:3, :3] = rotation_xyz(step.rpy_rad)
        # object frame 的变换左乘，tcp frame 的变换右乘，并严格保持声明顺序。
        pose = delta @ pose if step.frame == "object" else pose @ delta
    return pose

def generate_grasps(concept, object_pose, profile: GraspProfile, *, part_name: str | None = None) -> GraspGenerationResult:
    """从 typed Concept 与 resolved profile 生成确定性的世界坐标 TCP。"""
    if not isinstance(profile, GraspProfile):
        raise TypeError("profile 必须是 GraspProfile")
    resolved_part = profile.part_name or part_name
    if not resolved_part:
        raise ValueError("通用 profile 必须通过 part_name 指明目标 part")
    generator = GENERATOR_REGISTRY.get(profile.generator)
    if generator is None:
        raise ValueError(f"尚未注册 grasp generator: {profile.generator!r}")
    T_world_object = validate_pose(object_pose, field_name="object_pose")
    proposals = []
    diagnostics = []
    for candidate in generator(concept, profile):
        candidate_id = build_candidate_id(profile.name, profile.generator, candidate.parameter_indices)
        if isinstance(candidate, LocalGraspRejection):
            diagnostics.append(
                CandidateDiagnostic(
                    candidate_id,
                    candidate.code,
                    candidate.message,
                    candidate.context,
                )
            )
            continue
        T_object_tcp = _apply_postprocess(concept.pose_in_object @ candidate.tcp_pose_part, profile)
        width = profile.closing_width.width_m if profile.closing_width.mode == "override" else candidate.closing_width_m
        proposals.append(GraspProposal(candidate_id, profile.generator, resolved_part, T_world_object @ T_object_tcp, width, candidate.parameters))
    # 稳定排序保留参数枚举顺序与 candidate ID；按逆序应用规则实现多级排序。
    for rule in reversed(profile.sorting):
        if rule.field is not None:
            field = rule.field[:-4] + "_rad" if rule.field.endswith("_deg") else rule.field
            if any(field not in proposal.parameters for proposal in proposals):
                raise ValueError(f"sorting.field {rule.field!r} 不在候选参数中")
            key = lambda proposal, name=field: abs(float(proposal.parameters[name]))
        else:
            # TCP +Z 指向世界 -Z 的程度；1 表示完全从上向下接近。
            key = lambda proposal: -float(proposal.tcp_pose_world[2, 2])
        proposals.sort(key=key, reverse=rule.order == "descending")
    return GraspGenerationResult(tuple(proposals), tuple(diagnostics), profile)

__all__ = ["generate_grasps"]
