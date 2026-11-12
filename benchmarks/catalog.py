"""只读任务定义目录；不加载 policy、规划器或设备。"""
from pathlib import Path
import hashlib
import json

DIRECTORY = Path(__file__).resolve().parent

def list_benchmarks() -> tuple[str, ...]:
    return tuple(sorted(p.parent.name for p in DIRECTORY.glob("*_task/definition.json")))

def load_definition(benchmark_id: str) -> dict:
    if not benchmark_id.endswith("_task") or Path(benchmark_id).name != benchmark_id or "/" in benchmark_id or "\\" in benchmark_id:
        raise ValueError("benchmark_id 必须是目录内的 <name>_task 名称")
    path = DIRECTORY / benchmark_id / "definition.json"
    definition = json.loads(path.read_text(encoding="utf-8"))
    version = definition["version"]
    if (definition["benchmark_id"] != benchmark_id
            or definition.get("schema_version", 1) != 1
            or type(version) is not int or version < 1):
        raise ValueError("benchmark 标识或版本不匹配")
    return definition

def identity(benchmark_id: str) -> dict:
    definition = load_definition(benchmark_id)
    path = DIRECTORY / benchmark_id / "definition.json"
    source_paths = {path, path.parent / "scene.py"}
    evaluation = DIRECTORY.parent / (definition["evaluation"]["module"].replace(".", "/") + ".py")
    source_paths.add(evaluation)
    if benchmark_id == "cup_pour_task":
        source_paths.add(DIRECTORY.parent / "gene_agent/knowledge/grasp/cup_handle.py")
    if benchmark_id in {"cup_pour_task", "spoon_scoop_task"}:
        source_paths.add(DIRECTORY / "tabletop.py")
    return {"id": benchmark_id, "version": definition["version"],
            "definition_ref": str(path),
            "definition_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_sha256": {str(p.relative_to(DIRECTORY.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(source_paths)}}
