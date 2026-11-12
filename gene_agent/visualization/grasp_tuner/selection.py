"""调参后尽量继续显示同一个几何抓取，而不是跳回候选 0。"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from gene_agent.knowledge.grasp import GraspProposal


def closest_candidate_index(candidates: Sequence[GraspProposal], previous: GraspProposal | None) -> int:
    """参数完全一致时优先沿用；否则按 TCP 平移和姿态距离选择近邻。

    候选 ID 含采样序号，修改采样范围后序号可能指向另一组参数，
    因此不能仅用 ID 延续选择。返回值只影响 Viser 显示，不参与抓取筛选。
    """
    if not candidates or previous is None:
        return 0
    old_parameters = previous.to_dict()["parameters"]
    for index, candidate in enumerate(candidates):
        if candidate.to_dict()["parameters"] == old_parameters:
            return index
    old_pose = previous.tcp_pose_world
    return min(
        range(len(candidates)),
        key=lambda index: (
            float(np.linalg.norm(candidates[index].tcp_pose_world[:3, 3] - old_pose[:3, 3]))
            + 0.02 * float(np.linalg.norm(candidates[index].tcp_pose_world[:3, :3] - old_pose[:3, :3])),
            index,
        ),
    )


def representative_candidate_indices(
    candidates: Sequence[GraspProposal], selected: int, count: int
) -> tuple[int, ...]:
    """从当前候选之外选少量分散姿态，供真实夹爪预览。

    位置和朝向都参与距离计算：同一抓取点的不同倾角也能被选中。
    每次从离已展示姿态最远的候选中取一个，序号仅用于稳定打破平局。
    结果只决定显示，不改变 Knowledge 的候选或优先级。
    """
    if not candidates or count <= 0:
        return ()
    selected = max(0, min(selected, len(candidates) - 1))
    positions = np.asarray([item.tcp_pose_world[:3, 3] for item in candidates])
    rotations = np.asarray([item.tcp_pose_world[:3, :3].reshape(9) for item in candidates])

    def distances(anchor: int) -> np.ndarray:
        return (np.linalg.norm(positions - positions[anchor], axis=1)
                + .015 * np.linalg.norm(rotations - rotations[anchor], axis=1))

    # 只保存每个候选到已展示集合的最小距离；不会构建 N×N 距离矩阵。
    nearest = distances(selected)
    nearest[selected] = -np.inf
    picked: list[int] = []
    for _ in range(min(count, len(candidates) - 1)):
        chosen = int(np.argmax(nearest))
        picked.append(chosen)
        nearest = np.minimum(nearest, distances(chosen))
        nearest[selected] = -np.inf
        nearest[picked] = -np.inf
    return tuple(picked)


__all__ = ["closest_candidate_index", "representative_candidate_indices"]
