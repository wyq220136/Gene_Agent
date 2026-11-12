"""不依赖 Viser server 的抓取调参 session 数据层。"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from gene_agent.knowledge.geometry import Concept, CylinderConcept, TrifoldConcept, load_concept
from gene_agent.knowledge.grasp import GraspProposal, load_grasp_profiles, parse_grasp_profiles


SUPPORTED_PROFILES = ("task_1_bowl_0", "task_1_knob_0", "trifold_handle")


@dataclass(frozen=True, slots=True)
class ObjectVisual:
    """渲染器所需的最小物体描述，不把 Viser 类型泄漏进纯数据层。"""

    kind: Literal["mesh", "cylinder", "trifold"]
    mesh_path: Path | None = None
    radius_xy_m: tuple[float, float] | None = None
    height_m: float | None = None
    pose_in_object: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class GraspTunerState:
    """一个 profile 对应的不可变场景状态；物体 pose 固定为 identity。"""

    profile_name: str
    concept: Concept
    object_pose_world: NDArray[np.float64]
    object_visual: ObjectVisual
    candidates: tuple[GraspProposal, ...]
    selected_index: int = 0

    @property
    def selected_candidate(self) -> GraspProposal:
        return self.candidates[self.selected_index]


def _profile_assets(root: Path, profile_name: str) -> tuple[Path, ObjectVisual]:
    """集中维护首批 profile 的 Concept 来源和显示资产。"""
    if profile_name == "task_1_bowl_0":
        return (
            root / "assets/objects/bowl_concept.json",
            ObjectVisual(
                kind="mesh",
                mesh_path=(root / "assets/objects/bowl_visual.obj").resolve(),
            ),
        )
    if profile_name == "task_1_knob_0":
        return root / "assets/objects/microwave_concept.json", ObjectVisual(kind="cylinder")
    if profile_name == "trifold_handle":
        return root / "assets/objects/microwave_concept.json", ObjectVisual(kind="trifold")
    supported = ", ".join(SUPPORTED_PROFILES)
    raise ValueError(f"未知 profile {profile_name!r}；当前只支持：{supported}")


def build_session_state(root: str | Path, profile_name: str, *, profile_document: dict | None = None, object_pose_world: np.ndarray | None = None) -> GraspTunerState:
    """从仓库 profile、typed Concept 和正式生成器构建 demo 状态。

    此函数刻意不导入 Viser，因此单元测试和 Knowledge 调试无需占用端口。
    单体实验台默认使用 identity；场景 tuner 则传入 MuJoCo GT 的物体 pose。
    """
    repository_root = Path(root).resolve()
    profile_path = repository_root / "data/task_1/knowledge/grasp_profiles.json"
    profiles = load_grasp_profiles(profile_path) if profile_document is None else parse_grasp_profiles(profile_document)
    if profile_name not in profiles:
        raise ValueError(f"{profile_path} 中不存在 profile {profile_name!r}")
    profile = profiles[profile_name]
    count = math.prod(len(values) for values in profile.parameters.values())
    if count > 20_000:
        raise ValueError(f"profile {profile_name!r} 将枚举 {count} 个候选，超过 Viser 调参上限 20000；请缩小参数范围")
    if profile.part_name is None and profile_name != "trifold_handle":
        raise ValueError(f"profile {profile_name!r} 必须声明 part")

    concept_path, visual = _profile_assets(repository_root, profile_name)
    resolved_part = profile.part_name or "handle"
    part = load_concept(concept_path, part_name=resolved_part)
    concept = part.concept
    if visual.kind == "cylinder":
        if not isinstance(concept, CylinderConcept):
            raise TypeError(f"profile {profile_name!r} 的显示要求 CylinderConcept")
        visual = ObjectVisual(
            kind="cylinder",
            radius_xy_m=tuple(float(value) for value in concept.radius_xy_m),
            height_m=concept.height_m,
            pose_in_object=np.array(concept.pose_in_object, copy=True),
        )

    # generate_grasps 在 Knowledge 中是唯一候选事实来源；demo 不复制任何抓取公式。
    from gene_agent.knowledge.grasp import generate_grasps

    object_pose_world = np.eye(4, dtype=float) if object_pose_world is None else np.array(object_pose_world, dtype=float, copy=True)
    result = generate_grasps(concept, object_pose_world, profile, part_name=resolved_part)
    candidates = tuple(result.proposals)
    if not candidates:
        raise ValueError(f"profile {profile_name!r} 没有生成任何候选")
    object_pose_world.setflags(write=False)
    return GraspTunerState(
        profile_name=profile_name,
        concept=concept,
        object_pose_world=object_pose_world,
        object_visual=visual,
        candidates=candidates,
    )


class EditableGraspSession:
    """保留 task 原始 JSON 规格，在纯数据层完成试算与显式保存。

    编辑失败不会改变当前候选。默认输出集中到不跟踪的开发配置目录，避免
    未经人工验收就覆盖运行输入。调用者可明确指定原路径覆盖。
    """

    def __init__(self, root: str | Path, profile_name: str, *, output_path: str | Path | None = None, object_pose_world: np.ndarray | None = None):
        self.root = Path(root).resolve()
        self.source_path = self.root / "data/task_1/knowledge/grasp_profiles.json"
        self.output_path = (Path(output_path).resolve() if output_path else
                            self.root / "archive/task_1/configs/grasp_profiles.tuned.json")
        self.profile_name = profile_name
        self.object_pose_world = np.eye(4) if object_pose_world is None else np.array(object_pose_world, dtype=float, copy=True)
        initial_path = self.output_path if self.output_path.is_file() else self.source_path
        self.document = json.loads(initial_path.read_text(encoding="utf-8"))
        self.state = build_session_state(self.root, profile_name, profile_document=self.document, object_pose_world=self.object_pose_world)

    @property
    def profile_json(self) -> dict:
        return copy.deepcopy(self.document["profiles"][self.profile_name])

    def regenerate(self, *, parameters: dict, postprocess: list, closing_width: dict | None = None) -> GraspTunerState:
        """原子地验证输入并生成候选；错误直接带字段路径返回 UI。"""
        draft = copy.deepcopy(self.document)
        profile = draft["profiles"][self.profile_name]
        profile["parameters"] = parameters
        profile["postprocess"] = postprocess
        if closing_width is not None:
            profile["closing_width"] = closing_width
        state = build_session_state(self.root, self.profile_name, profile_document=draft, object_pose_world=self.object_pose_world)
        self.document, self.state = draft, state
        return state

    def replace_profile(self, profile_json: dict) -> GraspTunerState:
        """从反馈快照恢复完整 profile；先验证再替换当前 session。"""
        draft = copy.deepcopy(self.document)
        draft["profiles"][self.profile_name] = copy.deepcopy(profile_json)
        state = build_session_state(self.root, self.profile_name, profile_document=draft, object_pose_world=self.object_pose_world)
        self.document, self.state = draft, state
        return state

    def save(self) -> Path:
        """仅保存已成功重生成的输入；写临时文件后原子替换目标。"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # 多个 profile 的编辑器可能同时打开。保存时只合并当前 profile，
        # 避免后一次保存用启动时的旧快照覆盖前一个 profile 的调参结果。
        document = copy.deepcopy(self.document)
        if self.output_path.is_file():
            latest = json.loads(self.output_path.read_text(encoding="utf-8"))
            latest["profiles"][self.profile_name] = document["profiles"][self.profile_name]
            parse_grasp_profiles(latest)
            document = latest
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output_path)
        self.document = document
        return self.output_path
