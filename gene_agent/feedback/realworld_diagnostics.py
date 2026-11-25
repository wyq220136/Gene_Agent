"""真机规划证据与逐候选索引；读取和汇总均不连接设备。"""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import traceback
from uuid import uuid4

from .motion_diagnostics import summarize_motion_result, save_motion_diagnostic


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                         default=lambda v: v.tolist()) + "\n"
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return path


def candidate_inventory(candidates, plan=None, *, candidate_order=None, error=None):
    """生成、排序、评估和选中分别记录；没走到的阶段不能算失败路径。"""
    plan = plan or {}
    order = list(candidate_order or [c["candidate_id"] for c in candidates])
    rank = {name: i for i, name in enumerate(order)}
    attempts = plan.get("attempts", [])
    rows = []
    for generated_index, candidate in enumerate(candidates):
        name = candidate["candidate_id"]
        phases = []
        for index, attempt in enumerate(attempts):
            if attempt.get("candidate_id") != name:
                continue
            result = attempt.get("result") or {}
            summary = summarize_motion_result(result).to_dict() if result.get("failure") else None
            phases.append(dict(attempt_index=index, stage=attempt.get("stage"),
                request_id=(attempt.get("request") or {}).get("request_id") or attempt.get("request_id"),
                success=result.get("success"), summary=summary,
                failure=result.get("failure"), metrics=result.get("metrics", {}),
                postcheck_details={k: v for k, v in result.items() if k not in {
                    "success", "failure", "metrics", "request_id", "operation", "backend",
                    "planner_mode", "duration_s", "trajectory", "goal_state"}}))
        rejected = None
        for stage in ("final_grasp_ik", "pregrasp_ik", "pregrasp", "approach", "approach_contact_postcheck"):
            checked = [p for p in phases if p["stage"] == stage]
            if checked and all(p["success"] is False for p in checked):
                rejected = checked[0]
                break
        posture = [p for p in phases if p["stage"] == "pregrasp_posture_check" and p["success"] is False]
        bad_requests = {p["request_id"] for p in posture}
        viable = [p for p in phases if p["stage"] == "pregrasp_ik" and p["success"] is True
                  and p["request_id"] not in bad_requests]
        if rejected is None and posture and not viable:
            rejected = posture[0]
        chosen = name == plan.get("selected_candidate_id")
        prefix = {p["name"]: p["status"] for p in plan.get("stages", [])}
        prefix_planned = all(prefix.get(p) == "planned" for p in ("open", "pregrasp", "approach", "close"))
        status = ("selected" if chosen and prefix_planned else
                  "rejected" if rejected else "selected_prefix_incomplete" if chosen else
                  "evaluated_not_selected" if phases else
                  "not_evaluated" if name in rank else "not_in_selection")
        completed = {p["stage"] for p in phases}
        rows.append(dict(candidate_id=name, generated_index=generated_index,
            selection_index=rank.get(name), status=status,
            target_world_tcp=candidate.get("T_world_tcp"),
            parameters=candidate.get("parameters", {}), first_rejection=rejected,
            phases=phases, not_reached=[dict(stage=stage,
                reason=("earlier_stage_rejected" if rejected else
                        "selection_stopped_or_not_required"))
                for stage in ("final_grasp_ik", "pregrasp_ik", "pregrasp", "approach")
                if stage not in completed]))
    return dict(schema_version=1, canonical_stage_map="docs/01-task-skills.svg",
        task_stage="load_bowl/pick_bowl", generated_count=len(candidates),
        selection_count=len(order), selected_candidate_id=plan.get("selected_candidate_id"),
        candidate_order=order, counts=dict(Counter(r["status"] for r in rows)),
        candidates=rows, error=error,
        scope="每行是一个抓取候选；phase 是实际请求。未到达阶段不代表该路径已求解或失败。")


def write_candidate_inventory(output, candidates, plan=None, **kwargs):
    payload = candidate_inventory(candidates, plan, **kwargs)
    path = write_json(Path(output) / "candidate_diagnostics.json", payload)
    lines = ["# 抓碗候选筛选", payload["scope"],
             f"生成 {payload['generated_count']}；进入筛选 {payload['selection_count']}；状态 {payload['counts']}",
             "|候选|顺序|状态|首个拒绝阶段|原因|", "|---|---:|---|---|---|"]
    for row in payload["candidates"]:
        first = row["first_rejection"] or {}
        summary = first.get("summary") or {}
        reason = (summary.get("conclusion") or (first.get("failure") or {}).get("summary")
                  or first.get("postcheck_details", {}).get("reason")
                  or first.get("postcheck_details", {}).get("reason_code", ""))
        reason = str(reason).replace("|", "\\|").replace("\n", " ")
        lines.append(f"|{row['candidate_id']}|{row['selection_index']}|{row['status']}|{first.get('stage', '')}|{reason}|")
    path.with_suffix(".md").write_text("\n\n".join(lines[:3]) + "\n\n" + "\n".join(lines[3:]) + "\n", encoding="utf-8")
    return path


class RealworldRunDiagnostics:
    """一次真机运行的记录生命周期；旁路错误不会改写正式规划/执行结果。"""
    def __init__(self, output, *, perception_paths=(), options=None, metadata=None):
        self.root = Path(output).resolve()
        self.request_output_dir = self.root / "motion_requests"
        self.options = dict(options or {"mode": "on_failure", "budget_s": 5.})
        from gene_agent.motion.curobo.diagnostics import DiagnosticOptions
        DiagnosticOptions(**self.options)
        self.metadata = dict(metadata or {})
        self.perception_paths = tuple(perception_paths)
        self.errors = []
        self._request_dirs = {self.request_output_dir}
        self._initial_attempted = False

    def __enter__(self):
        from gene_agent.motion.recording import motion_recording
        from gene_agent.realworld.scene_archive import SceneArchive
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "events.jsonl").exists():
            raise ValueError("诊断事件目录已有记录，不能混入另一轮运行")
        self.archive = SceneArchive(self.root / "initial_scene", metadata=self.metadata)
        self.stack = ExitStack()
        self.stack.enter_context(motion_recording(self))
        try:
            write_json(self.root / "diagnostic_manifest.json", dict(schema_version=1,
                options=self.options, metadata=self.metadata,
                initial_scene_ref="initial_scene/manifest.json", events_ref="events.jsonl",
                scene_viewer=dict(entrypoint="tools/viewers/view_realworld_scene.py",
                    argv=["--run", str(self.root), "--host", "127.0.0.1", "--port", "8120"],
                    read_only=True, source="initial_scene/manifest.json")))
        except Exception:
            self.stack.close()
            raise
        return self

    def record_error(self, scope, error):
        self.errors.append(dict(scope=scope, error=f"{type(error).__name__}: {error}"))
        try:
            write_json(self.root / "diagnostic_errors.json", self.errors)
        except OSError:
            pass

    def capture_initial(self, scene):
        if self._initial_attempted:
            return
        self._initial_attempted = True
        try:
            self.archive.add_scene(scene, label="initial")
        except Exception as error:
            self.record_error("initial_scene", error)
        # 模型构建/导出失败时仍保存已有原始帧，不让现场证据随之丢失。
        try:
            self.archive.add_perception(scene, self.perception_paths)
            self.archive.save()
        except Exception as error:
            self.record_error("initial_perception", error)

    def wrap_observer(self, observe):
        def recorded():
            scene = observe()
            self.capture_initial(scene)
            return scene
        from gene_agent.realworld.microwave.operation_perception import forward_observation_hooks
        forward_observation_hooks(observe, recorded)
        return recorded

    def note_selection(self, scene, candidates, parameters, profile_path, seed):
        try:
            return self.archive.add_selection(scene, candidates, parameters, profile_path, seed)
        except Exception as error:
            self.record_error("selection_scene", error)
            return None

    def record(self, requests, results, *, snapshot, replay_ref=None, invocation=None):
        call_id = uuid4().hex
        if replay_ref:
            self._request_dirs.add(Path(replay_ref).resolve().parent)
        for index, (request, result) in enumerate(zip(requests, results)):
            value = result.to_dict()
            identity = f"{call_id}_{index:04d}"
            ref = write_json(self.root / "motion_results" / f"{identity}.json", value)
            event = dict(schema_version=1, episode_id=self.root.name,
                kind="planning", invocation_id=call_id, batch_index=index,
                batch_size=len(requests), invocation=invocation,
                stage_id=request.request_id, request_id=request.request_id,
                phase=("final_grasp_ik" if request.request_id.rsplit(":", 1)[-1] == "final_ik" else
                       "pregrasp_ik" if request.request_id.rsplit(":", 1)[-1].startswith("pregrasp_ik_") else
                       request.request_id.rsplit(":", 1)[-1]),
                candidate_id=request.request_id.rsplit(":", 1)[0] if ":" in request.request_id else None,
                operation=request.operation.value, state_id=request.scene.snapshot_sha256,
                scene_revision=request.scene.revision, group=request.group,
                target_world_tcp=request.target.matrix.tolist(),
                result_ref=str(ref.relative_to(self.root)),
                replay_request_ref=((str(Path(replay_ref).relative_to(self.root))
                    if Path(replay_ref).is_relative_to(self.root) else str(replay_ref)) if replay_ref else None),
                success=result.success, duration_s=result.duration_s)
            if not result.success:
                try:
                    summary = summarize_motion_result(result, snapshot)
                    event["summary"] = summary.to_dict()
                    detail = save_motion_diagnostic(self.root, dict(event, planning=value), snapshot)
                    event["diagnostic_ref"] = str(detail.relative_to(self.root))
                except Exception as error:
                    event["diagnostic_error"] = f"{type(error).__name__}: {error}"
                    self.record_error("motion_diagnostic", error)
            encoded = json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()

    def __exit__(self, typ, error, tb):
        self.stack.close()
        if error is not None:
            try:
                write_json(self.root / "diagnostic_exception.json", dict(
                    error=f"{typ.__name__}: {error}", traceback="".join(traceback.format_exception(typ, error, tb))))
            except Exception as recording_error:
                self.record_error("exception_recording", recording_error)
        try:
            if not self._initial_attempted:
                self.archive.add_perception(None, self.perception_paths)
            for directory in self._request_dirs:
                self.archive.add_requests(directory)
        except Exception as recording_error:
            self.record_error("archive_inputs", recording_error)
        self.archive.manifest["recording_errors"] = list(self.errors)
        for scope, finalize in (("archive_manifest", self.archive.save),
                                ("failure_summary", lambda: write_realworld_summary(self.root))):
            try:
                finalize()
            except Exception as recording_error:
                self.record_error(scope, recording_error)
        return False


def inspect_realworld_run(root, *, candidate=None, stage=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("运行目录不存在")
    events, errors = [], []
    path = root / "events.jsonl"
    if path.is_file():
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            try:
                event = json.loads(line)
            except ValueError:
                errors.append(f"events.jsonl:{index+1} 不完整，覆盖可能缺失")
                continue
            if stage and stage not in event.get("stage_id", "") and stage != event.get("phase"):
                continue
            if candidate and candidate != event.get("candidate_id"):
                continue
            events.append(event)
    inventories = []
    for path in sorted(root.rglob("candidate_diagnostics.json")):
        if "initial_scene" in path.relative_to(root).parts:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if candidate:
            value["candidates"] = [r for r in value["candidates"] if r["candidate_id"] == candidate]
        inventories.append(dict(ref=str(path.relative_to(root)), **value))
    if not inventories:
        # 旧日志只能提炼已保存的 attempts；未记录的候选参数和现场不补造。
        for path in sorted(root.rglob("candidate_selection.json")):
            if "initial_scene" in path.relative_to(root).parts:
                continue
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                errors.append(str(path.relative_to(root)) + " 不完整")
                continue
            attempts = saved.get("plan", {}).get("attempts", [])
            order = saved.get("candidate_order") or list(dict.fromkeys(
                row.get("candidate_id") for row in attempts if row.get("candidate_id")))
            targets = {row["candidate_id"]: (row.get("request") or {}).get("target", {}).get("matrix")
                       for row in attempts if row.get("stage") == "final_grasp_ik"}
            candidates = [{"candidate_id": name, "T_world_tcp": targets.get(name),
                           "parameters": {"source": "legacy_parameters_not_recorded"}} for name in order]
            inventory = candidate_inventory(candidates, saved.get("plan"), candidate_order=order)
            inventory.update(ref=str(path.relative_to(root)), legacy_extracted=True,
                             scope="从旧 attempts 提炼；没有记录的生成候选/参数/现场保持未知")
            if candidate:
                inventory["candidates"] = [row for row in inventory["candidates"] if row["candidate_id"] == candidate]
            inventories.append(inventory)
            for index, row in enumerate(attempts):
                request, result = row.get("request") or {}, row.get("result") or {}
                request_id = request.get("request_id") or row.get("request_id")
                if (candidate and row.get("candidate_id") != candidate) or (stage and stage not in str(row.get("stage", ""))):
                    continue
                events.append(dict(request_id=request_id, stage_id=row.get("stage"), success=result.get("success"),
                    result_ref=f"{path.relative_to(root)}#/plan/attempts/{index}/result",
                    diagnostic_ref=None, replay_request_ref=result.get("metrics", {}).get("replay_request_ref"),
                    summary=summarize_motion_result(result).to_dict() if result.get("failure") else {}))
    operation_failures = []
    terminal_reports = []
    for path in sorted(root.rglob("*_result.json")):
        if "initial_scene" in path.relative_to(root).parts:
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            errors.append(str(path.relative_to(root)) + " 终态报告不完整")
            continue
        if not isinstance(report, dict):
            continue
        if not isinstance(report.get("execute"), bool) and path.name != "entry_result.json":
            continue
        reference = str(path.relative_to(root))
        terminal_reports.append(dict(report_ref=reference, status=report.get("status"),
            execute=report.get("execute"), task_success=report.get("task_success"),
            success=report.get("success"), kind=report.get("subtask") or report.get("skill_id")))
        failure = report.get("first_failure") or report.get("failure")
        if failure:
            operation_failures.append(dict(report_ref=reference, status=report.get("status"),
                execute=report.get("execute"), first_failure=failure))
    exception_path = root / "diagnostic_exception.json"
    exception = json.loads(exception_path.read_text()) if exception_path.is_file() else None
    recording_path = root / "diagnostic_errors.json"
    recording_errors = json.loads(recording_path.read_text()) if recording_path.is_file() else []
    return dict(schema_version=1, root=str(root), events=events, inventories=inventories,
        exception=exception, recording_errors=recording_errors,
        terminal_reports=terminal_reports, operation_failures=operation_failures,
        task_result_coverage="available" if terminal_reports else "not_recorded; planning success is not task success",
        counts=dict(Counter("success" if e.get("success") is True else "failed" if e.get("success") is False else "unknown" for e in events)),
        errors=errors, initial_scene_ref="initial_scene/manifest.json",
        scope="只读已保存证据；没有执行的阶段不补造结果；有限搜索失败不是全局不可达证明。")


def write_realworld_summary(root):
    root = Path(root)
    payload = inspect_realworld_run(root)
    write_json(root / "failure_analysis.json", payload)
    lines = ["# 真机失败定位", payload["scope"], f"规划请求：{payload['counts']}",
        "初始感知/场景包：initial_scene/manifest.json",
        "候选完整清单：各 selection/candidate_diagnostics.json 和同名 .md",
        "检查：python tools/diagnostics/inspect_realworld_run.py --run " + str(root) + " --details",
        "离线现场：python tools/diagnostics/replay_realworld_scene.py --bundle " + str(root / "initial_scene"),
        "|请求/阶段|分类|后端误差（m/rad）|详细证据|", "|---|---|---|---|"]
    if not payload["terminal_reports"]:
        lines.insert(3, "未找到任务/子任务终态报告；不能从规划结果判断执行或任务成功。")
    for failure in payload["operation_failures"]:
        detail = failure["first_failure"]
        lines.insert(3, "首个操作拒绝：" + str(detail.get("code") or detail.get("kind"))
                     + "；" + str(detail.get("message") or detail.get("summary"))
                     + "；原报告：" + failure["report_ref"])
    if payload["exception"]:
        lines.insert(3, "运行异常：" + payload["exception"]["error"] + "；完整栈见 diagnostic_exception.json")
    for row in payload["events"]:
        if row.get("success"):
            continue
        summary = row.get("summary", {})
        reason = str(summary.get("conclusion", "未知")).replace("|", "\\|")
        lines.append(f"|{row['request_id']}|{reason}|{summary.get('best_position_error_m')}/{summary.get('best_rotation_error_rad')}|{row.get('diagnostic_ref')}|")
    table_start = next(index for index, line in enumerate(lines) if line.startswith("|请求/阶段|"))
    (root / "failure_analysis.md").write_text("\n\n".join(lines[:table_start]) + "\n\n" + "\n".join(lines[table_start:]) + "\n", encoding="utf-8")
    return root / "failure_analysis.json"
