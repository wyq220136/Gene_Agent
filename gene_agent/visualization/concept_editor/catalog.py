"""从已有标注运行记录找 mesh 与 Concept，供编辑器按物体名启动。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ConceptEntry:
    category: str
    folder: Path
    mesh: Path
    concept: Path
    finished_at_utc: str

    @property
    def input_concept(self) -> Path:
        tuned = self.folder / "concept.tuned.json"
        return tuned if tuned.is_file() else self.concept

    @property
    def output_concept(self) -> Path:
        return self.folder / "concept.tuned.json"


def discover_concepts(root: Path) -> list[ConceptEntry]:
    """只收录 mesh 和最终 Concept 都存在的标注；不猜测坐标系。"""
    root = Path(root).resolve()
    records = []
    for base in (root / "runs/concept_annotation", root / "data/realworld"):
        if base.is_dir():
            records.extend(base.rglob("annotation_run.json"))
    entries = []
    for record in records:
        try:
            run = json.loads(record.read_text(encoding="utf-8"))
            mesh = Path(run["mesh"]).expanduser()
            if not mesh.is_absolute():
                mesh = root / mesh
            concept = record.parent / run.get("final_concept", "concept.json")
            if not mesh.is_file() or not concept.is_file():
                continue
            annotation = json.loads(concept.read_text(encoding="utf-8"))
            category = str(annotation.get("category") or record.parent.name)
            entries.append(ConceptEntry(
                category=category, folder=record.parent.resolve(), mesh=mesh.resolve(),
                concept=concept.resolve(),
                finished_at_utc=str(run.get("finished_at_utc", "")),
            ))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(entries, key=lambda row: (row.category, row.finished_at_utc,
                                             str(row.folder)), reverse=False)


def resolve_object(entries: list[ConceptEntry], query: str) -> ConceptEntry:
    """目录名优先；同类别取最近完成的一次标注。"""
    needle = query.strip().casefold()
    if not needle:
        raise ValueError("物体名不能为空")
    exact_run = [entry for entry in entries if entry.folder.name.casefold() == needle]
    exact_category = [entry for entry in entries if entry.category.casefold() == needle]
    matches = exact_run or exact_category
    if not matches:
        matches = [entry for entry in entries if needle in entry.category.casefold()
                   or needle in entry.folder.name.casefold()]
        categories = {entry.category for entry in matches}
        if len(categories) > 1:
            raise ValueError(f"'{query}' 匹配多个物体：{', '.join(sorted(categories))}；请用完整名称")
    if not matches:
        names = ", ".join(sorted({entry.category for entry in entries})) or "无"
        raise ValueError(f"找不到物体 '{query}'；可用：{names}")
    return max(matches, key=lambda row: (row.finished_at_utc, str(row.folder)))


__all__ = ["ConceptEntry", "discover_concepts", "resolve_object"]
