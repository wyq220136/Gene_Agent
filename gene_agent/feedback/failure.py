"""把一次失败冻结成 Codex 可读取且可恢复的小证据包。"""
from __future__ import annotations

import json
from pathlib import Path
import traceback

from gene_agent.feedback.state import save_state, state_id
from gene_agent.simulation.diagnostics import query_diagnostics
from gene_agent.simulation.runtime import MujocoRuntime
from gene_agent.visualization import render_views


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _suggested_queries(failure_type: str) -> list[str]:
    if failure_type == "planner_failed":
        return ["compare planned start with operational state", "inspect nearby collision geometry"]
    if failure_type in {"execution_timeout", "stale_plan"}:
        return ["inspect joint tracking and current state_id", "query contacts near the active robot"]
    return ["inspect target pose and local contacts", "inspect overview and task camera views"]


def capture_failure(
    runtime,
    output_dir,
    *,
    stage_id: str,
    failure_type: str,
    message: str,
    object_names=(),
    robot_groups=(),
    target_body: str = "bowl",
    exception: BaseException | None = None,
    details=None,
    cameras=None,
) -> dict:
    """保存同一冻结状态的 operational、诊断、恢复状态和调试视图。"""
    details = {} if details is None else details
    # 在创建目录前验证，避免不可序列化输入留下看似有效的半个 run。
    json.dumps(details, ensure_ascii=False, allow_nan=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    frozen = runtime.copy_data()
    frozen_runtime = MujocoRuntime(
        runtime.model,
        frozen,
        runtime.metadata,
        scene_revision=runtime.scene_revision,
    )
    identifier = state_id(runtime.model, frozen)
    evidence_errors = {}
    result = {
        "schema_version": 1,
        "stage_id": str(stage_id),
        "failure_type": str(failure_type),
        "message": str(message),
        "state_id": identifier,
        "sim_time_s": float(frozen.time),
        "details": details,
        "suggested_queries": _suggested_queries(str(failure_type)),
    }

    try:
        operational = frozen_runtime.query(
            object_names=object_names,
            robot_groups=robot_groups,
        )
        _write_json(output / "operational_observation.json", operational)
    except Exception as error:  # 每类证据独立降级，保留其余事实。
        evidence_errors["operational_observation"] = f"{type(error).__name__}: {error}"

    try:
        diagnostics = query_diagnostics(
            frozen_runtime,
            body_names=(target_body,) if target_body else (),
        )
        _write_json(output / "diagnostics.json", diagnostics)
    except Exception as error:
        evidence_errors["diagnostics"] = f"{type(error).__name__}: {error}"

    try:
        save_state(runtime.model, frozen, output / "state.npz")
    except Exception as error:
        evidence_errors["save_state"] = f"{type(error).__name__}: {error}"

    if exception is not None:
        try:
            text = "".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            )
            (output / "traceback.txt").write_text(text, encoding="utf-8")
        except Exception as error:
            evidence_errors["traceback"] = f"{type(error).__name__}: {error}"

    try:
        render_views(
            runtime.model,
            frozen,
            output / "images",
            target_body=target_body,
            view_names=("overview", "task", "side", "contact", "contact_side"),
            cameras=cameras,
            collision_overlay=True,
            contact_debug=True,
        )
    except Exception as error:
        evidence_errors["render_views"] = f"{type(error).__name__}: {error}"

    result["evidence_incomplete"] = bool(evidence_errors)
    result["evidence_errors"] = evidence_errors
    _write_json(output / "failure.json", result)
    return result
