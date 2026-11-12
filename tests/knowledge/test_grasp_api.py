"""Ring/axial 统一入口的 frame、postprocess 与 part 契约。"""

import numpy as np
import pytest

from gene_agent.knowledge.geometry import CuboidConcept, CylinderConcept
from gene_agent.knowledge.grasp import ClosingWidthSpec, GraspProfile, PosePostprocess, generate_grasps

def _profile(*, part="knob", postprocess=()):
    return GraspProfile(name="api", generator="cylinder_axial", part_name=part, parameters={"surface": ("bottom",), "depth_m": (0.02,), "azimuth_rad": (0.0,), "tip_x_rad": (0.0,)}, closing_width=ClosingWidthSpec(mode="geometry", clearance_m=0.001), postprocess=postprocess)

def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return result

def test_world_object_and_object_part_are_each_applied_once() -> None:
    part_pose = _rz(np.pi / 2.0); part_pose[:3, 3] = [0.2, 0.0, 0.0]
    world_pose = _rz(-np.pi / 2.0); world_pose[:3, 3] = [1.0, 2.0, 3.0]
    concept = CylinderConcept([0.01, 0.01], 0.04, part_pose)
    proposal = generate_grasps(concept, world_pose, _profile()).proposals[0]
    np.testing.assert_allclose(proposal.tcp_pose_world, world_pose @ part_pose @ np.eye(4))

def test_object_and_tcp_postprocess_use_distinct_frames_in_declared_order() -> None:
    world_pose = _rz(np.pi / 2.0); world_pose[:3, 3] = [3.0, 4.0, 5.0]
    concept = CylinderConcept([0.01, 0.01], 0.04, np.eye(4))
    profile = _profile(postprocess=(PosePostprocess(type="translate", frame="object", xyz_m=(1.0, 0.0, 0.0)), PosePostprocess(type="rotate", frame="object", rpy_rad=(0.0, 0.0, np.pi / 2.0)), PosePostprocess(type="translate", frame="tcp", xyz_m=(0.0, 2.0, 0.0)), PosePostprocess(type="rotate", frame="tcp", rpy_rad=(np.pi / 2.0, 0.0, 0.0))))
    proposal = generate_grasps(concept, world_pose, profile).proposals[0]
    trans_object = np.eye(4); trans_object[:3, 3] = [1, 0, 0]
    rot_object = _rz(np.pi / 2.0)
    trans_tcp = np.eye(4); trans_tcp[:3, 3] = [0, 2, 0]
    rot_tcp = np.eye(4); rot_tcp[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    np.testing.assert_allclose(proposal.tcp_pose_world, world_pose @ rot_object @ trans_object @ trans_tcp @ rot_tcp, atol=1e-12)

def test_profile_part_takes_precedence_and_generic_profile_requires_part_name() -> None:
    concept = CylinderConcept([0.01, 0.01], 0.04, np.eye(4))
    assert generate_grasps(concept, np.eye(4), _profile(part="profile_part"), part_name="argument_part").proposals[0].part_name == "profile_part"
    assert generate_grasps(concept, np.eye(4), _profile(part=None), part_name="argument_part").proposals[0].part_name == "argument_part"
    with pytest.raises(ValueError, match="part_name"):
        generate_grasps(concept, np.eye(4), _profile(part=None))

def test_generator_and_concept_type_mismatch_is_a_clear_error() -> None:
    with pytest.raises(TypeError, match="cylinder_axial.*CylinderConcept"):
        generate_grasps(CuboidConcept([0.02, 0.03, 0.04], np.eye(4)), np.eye(4), _profile())
