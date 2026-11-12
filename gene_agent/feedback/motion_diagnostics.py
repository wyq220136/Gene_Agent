"""把 motion backend 的原始失败证据整理成人和 Codex 都能读懂的摘要。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _selected_seed(diagnostic: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = diagnostic.get("selected_seed_index")
    for seed in diagnostic.get("seeds") or ():
        if _mapping(seed).get("index") == selected:
            return _mapping(seed)
    return {}


def _object_label(body_name: str, geom_name: str) -> str:
    """把细碎 geom 汇总为用户真正需要处理的场景对象。"""
    body = str(body_name or "").lower()
    geom = str(geom_name or "").lower()
    if "table" in geom or body == "world":
        return "桌面"
    if body == "bowl" or "bowl" in body:
        return "碗"
    if "microwave_door" in body:
        return "微波炉门"
    if body.startswith("microwave") or body.startswith("knob_"):
        return "微波炉本体"
    if body.startswith("left_") or body.startswith("right_"):
        return "另一机械臂"
    if "robot_body" in body or "robot_body" in geom:
        return "机器人本体"
    return body_name or geom_name or "未知物体"


def _collision_groups(obstacles: tuple[str, ...], snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    records = {
        str(row.get("obstacle_name")): row
        for row in snapshot.get("records") or ()
        if isinstance(row, Mapping) and row.get("obstacle_name")
    }
    groups = []
    for obstacle in obstacles:
        row = records.get(obstacle, {})
        label = _object_label(str(row.get("body_name", "")), str(row.get("geom_name", obstacle)))
        if label not in groups:
            groups.append(label)
    return tuple(groups)


@dataclass(frozen=True)
class MotionFailureSummary:
    """一次阶段失败的稳定分类与可复核事实；不猜测后端没有提供的原因。"""

    kind: str
    conclusion: str
    raw_category: str
    raw_status: str | None
    raw_message: str | None
    detailed_reason_code: str | None = None
    start_state_valid: bool | None = None
    ik_seed_count: int | None = None
    ik_success_count: int | None = None
    local_candidate_count: int | None = None
    local_candidate_success_count: int | None = None
    best_position_error_m: float | None = None
    best_rotation_error_rad: float | None = None
    scene_collision: bool | None = None
    self_collision: bool | None = None
    joint_limit_violation: bool | None = None
    world_disabled_still_failed: bool | None = None
    collision_evidence_source: str | None = None
    colliding_obstacles: tuple[str, ...] = ()
    collision_groups: tuple[str, ...] = ()

    unknowns: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    diagnostic_facts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["colliding_obstacles"] = list(self.colliding_obstacles)
        result["collision_groups"] = list(self.collision_groups)
        return result

    def to_markdown(self) -> str:
        lines = [f"**规划失败：{self.conclusion}**"]
        lines.extend(f"- {fact}" for fact in self.diagnostic_facts)
        if self.start_state_valid is not None:
            lines.append(f"- 起点严格检查：{'通过' if self.start_state_valid else '失败'}")
        if self.ik_seed_count is not None:
            lines.append(f"- 严格 IK：{self.ik_success_count or 0}/{self.ik_seed_count} 个 seed 成功")
        if self.best_position_error_m is not None or self.best_rotation_error_rad is not None:
            position = ("未知" if self.best_position_error_m is None else
                        f"{self.best_position_error_m * 1000:.1f} mm")
            rotation = ("未知" if self.best_rotation_error_rad is None else
                        f"{self.best_rotation_error_rad * 180 / 3.141592653589793:.1f}°")
            lines.append(f"- 所选候选误差：位置 {position}，姿态 {rotation}")
        if self.local_candidate_count is not None:
            lines.append(
                f"- 局部旋转候选：{self.local_candidate_success_count or 0}/"
                f"{self.local_candidate_count} 个 IK 成功"
            )
        if self.detailed_reason_code:
            lines.append(f"- 细分原因码：`{self.detailed_reason_code}`")
        if (self.scene_collision is not None or self.self_collision is not None
                or self.joint_limit_violation is not None):
            lines.append(
                "- 碰撞复核：场景碰撞 "
                f"{'是' if self.scene_collision else '否' if self.scene_collision is False else '未提供'}，"
                f"自碰撞 {'是' if self.self_collision else '否' if self.self_collision is False else '未提供'}，"
                "关节限位 "
                f"{'违反' if self.joint_limit_violation else '未违反' if self.joint_limit_violation is False else '未提供'}"
            )
        if self.collision_evidence_source == "world_relaxed_ik_solution":
            lines.append("- 碰撞证据：关闭世界碰撞得到 IK 后，再用完整场景严格复核该解")
        if self.collision_groups:
            lines.append("- 碰撞对象组：" + "、".join(self.collision_groups))
        if self.colliding_obstacles:
            lines.append("- 原始碰撞代理：" + "、".join(self.colliding_obstacles))
        if self.world_disabled_still_failed is not None:
            lines.append(
                "- 关闭场景碰撞复核："
                + ("仍未找到解（此对照保留自碰撞检查）" if self.world_disabled_still_failed else "可以求解")
            )
        if self.raw_message:
            lines.append(f"- 后端原始信息：`{self.raw_message}`")
        lines.extend("- 未知：" + item for item in self.unknowns)
        lines.extend("- 下一步：" + item for item in self.next_actions)
        return "  \n".join(lines)


def summarize_motion_failure(
    result: Mapping[str, Any],
    collision_snapshot: Mapping[str, Any] | None = None,
) -> MotionFailureSummary:
    """从一次后端结果生成稳定摘要，保留“未知”而不臆测碰撞原因。"""
    category = str(result.get("failure_category") or "unknown")
    start = _mapping(result.get("start_state_check"))
    start_collision = _mapping(start.get("collision_diagnostic"))
    ik = _mapping(result.get("ik_diagnostic"))
    solver = _mapping(result.get("solver_diagnostic"))
    detail = _mapping(result.get("failure_detail"))
    reason_code = str(detail.get("reason_code")) if detail.get("reason_code") else None
    diagnostic = ik or solver
    collision = (_mapping(detail.get("collision_check"))
                 or _mapping(diagnostic.get("collision_check")) or start_collision)
    probe = _mapping(ik.get("world_obstruction_probe"))
    collision_evidence_source = None
    probe_collision = _mapping(probe.get("strict_collision_check"))
    if (
        category in {"goal_ik_no_solution", "ik_pose_not_converged", "ik_constraint_rejected"}
        and probe.get("world_collision_disabled_for_probe")
        and probe.get("relaxed_ik_success")
        and probe_collision
    ):
        # 原失败 seed 可能离目标还很远，因此它的“无碰撞”不能解释真正
        # 达到目标后的可行性。优先采用关闭 world 后得到的 IK 解，再把
        # 该解放回完整场景严格复核所得的碰撞证据。
        collision = probe_collision
        collision_evidence_source = "world_relaxed_ik_solution"
    obstacles = tuple(str(value) for value in collision.get("colliding_obstacles") or ())
    if not obstacles and collision_evidence_source is not None:
        obstacles = tuple(str(value) for value in probe.get("blocking_obstacles") or ())
    groups = _collision_groups(obstacles, _mapping(collision_snapshot))
    selected = _selected_seed(ik)
    attempts = _mapping(result.get("target_selection_attempts"))
    scene_collision = _optional_bool(collision.get("scene_collision"))
    self_collision = _optional_bool(collision.get("self_collision"))
    joint_limit = _optional_bool(collision.get("joint_limit_violation"))
    start_valid = _optional_bool(start.get("valid"))
    world_disabled_failed = None
    if probe.get("world_collision_disabled_for_probe"):
        world_disabled_failed = not bool(probe.get("relaxed_ik_success"))

    if reason_code == "goal_self_collision":
        kind = "goal_self_collision"
        conclusion = "目标关节姿态发生机器人自碰撞。"
    elif reason_code == "goal_scene_collision":
        kind = "goal_scene_collision"
        conclusion = "目标关节姿态与场景发生碰撞。"
    elif reason_code == "goal_joint_limit_violation":
        kind = "goal_joint_limit_violation"
        conclusion = "目标候选违反关节限位，未通过当前机械臂可操作空间约束。"
    elif reason_code == "goal_pose_unreachable_within_search":
        kind = "goal_pose_unreachable_within_search"
        conclusion = (
            "关闭场景碰撞并保留自碰撞检查后，目标 IK 在当前关节限位与有限种子搜索中仍未收敛；"
            "这不是全局不可达证明。"
        )
    elif reason_code == "goal_pose_not_converged":
        kind = "goal_ik_not_converged"
        conclusion = "目标 IK 位姿误差未收敛；当前证据不能确认碰撞或绝对不可达。"
    elif reason_code == "path_self_collision":
        kind = "path_self_collision"
        conclusion = "候选路径在记录的 waypoint 发生机器人自碰撞。"
    elif reason_code == "path_scene_collision":
        kind = "path_scene_collision"
        conclusion = "候选路径在记录的 waypoint 与场景发生碰撞。"
    elif reason_code == "path_joint_limit_violation":
        kind = "path_joint_limit_violation"
        conclusion = "候选路径中的记录 waypoint 违反关节限位。"
    elif reason_code == "path_optimization_failed_no_confirmed_collision":
        kind = "path_not_found"
        conclusion = "目标 IK 已有候选，但轨迹优化未返回完整可行路径；已检查证据未确认碰撞或关节限位。"
    elif reason_code in {"goal_multiple_constraints", "path_multiple_constraints"}:
        kind = reason_code
        conclusion = str(detail.get("human_summary") or "候选同时违反多项约束。")
    elif category == "start_collision":
        if self_collision and not scene_collision:
            kind = "start_self_collision"
            conclusion = "起始姿态发生机器人自碰撞。"
        elif scene_collision:
            kind = "start_scene_collision"
            conclusion = "起始姿态与场景发生碰撞。"
        else:
            kind = "start_invalid"
            conclusion = "起始姿态未通过严格状态检查，详细碰撞证据缺失。"
    elif category in {"goal_ik_no_solution", "ik_pose_not_converged", "ik_constraint_rejected"}:
        kind = "goal_ik_not_converged"
        if collision_evidence_source == "world_relaxed_ik_solution" and scene_collision:
            conclusion = "关闭场景碰撞后找到 IK；严格复核时目标解与场景物体碰撞。"
        elif collision_evidence_source == "world_relaxed_ik_solution" and self_collision:
            conclusion = "关闭场景碰撞后找到 IK；严格复核时目标解发生机器人自碰撞。"
        elif scene_collision or self_collision:
            conclusion = "目标 IK 未找到有效解；所检查的失败候选存在碰撞。"
        elif world_disabled_failed:
            conclusion = "目标 IK 未收敛；关闭场景碰撞后仍失败，当前没有证据指向场景碰撞。"
        else:
            conclusion = "目标 IK 未收敛；现有证据不能确认碰撞是原因。"
    elif category in {"goal_collision", "collision_postcheck_rejected"}:
        kind = "goal_collision"
        conclusion = "目标姿态或返回轨迹未通过碰撞复核。"
    elif category == "path_collision":
        kind = "path_collision"
        conclusion = "候选路径存在碰撞。"
    elif category in {"path_not_found", "solver_rejected"}:
        kind = "path_not_found"
        conclusion = "轨迹求解失败；目标可行性须依据独立检查，不能仅凭此状态判断。"
        if scene_collision is False and self_collision is False:
            conclusion += " 已检查的候选未发现碰撞证据。"
    elif category == "constraint_violation":
        kind = "path_constraint_violation"
        conclusion = "轨迹未通过 Skill 要求的路径约束。"
    else:
        kind = category
        conclusion = str(result.get("message") or "规划失败，当前证据不足以进一步归因。")

    return MotionFailureSummary(
        kind=kind,
        conclusion=conclusion,
        raw_category=category,
        raw_status=None if result.get("status") is None else str(result.get("status")),
        raw_message=None if result.get("message") is None else str(result.get("message")),
        detailed_reason_code=reason_code,
        start_state_valid=start_valid,
        ik_seed_count=None if ik.get("seed_count") is None else int(ik["seed_count"]),
        ik_success_count=(None if ik.get("successful_seed_count") is None
                          else int(ik["successful_seed_count"])),
        local_candidate_count=(None if attempts.get("candidate_count") is None
                               else int(attempts["candidate_count"])),
        local_candidate_success_count=(None if attempts.get("ik_success_count") is None
                                       else int(attempts["ik_success_count"])),
        best_position_error_m=(None if selected.get("position_error_m") is None
                               else float(selected["position_error_m"])),
        best_rotation_error_rad=(None if selected.get("rotation_error_rad") is None
                                 else float(selected["rotation_error_rad"])),
        scene_collision=scene_collision,
        self_collision=self_collision,
        joint_limit_violation=joint_limit,
        world_disabled_still_failed=world_disabled_failed,
        collision_evidence_source=collision_evidence_source,
        colliding_obstacles=obstacles,
        collision_groups=groups,
    )


def summarize_motion_result(result, snapshot=None) -> MotionFailureSummary:
    """适配公共MotionResult；只提炼已有证据，不导入引擎或启动求解。"""
    if hasattr(result, 'to_dict'):
        result = result.to_dict()
    failure = _mapping(result.get('failure'))
    if not failure:
        raw = dict(result)
    else:
        raw = {**_mapping(result.get('metrics')), **_mapping(failure.get('details')),
               'failure_category': _mapping(failure.get('details')).get('failure_category') or failure.get('kind'),
               'status': failure.get('backend_status'), 'message': failure.get('summary')}
    summary = summarize_motion_failure(raw, snapshot)
    from dataclasses import replace
    unknowns = ['本次候选检查不证明全局不可达，也不保证存在无碰撞路径。']
    if summary.best_position_error_m is None:
        unknowns.append('记录未提供候选位姿误差。')
    if summary.self_collision is None or summary.scene_collision is None:
        unknowns.append('碰撞复核有缺失项，不能按无碰撞处理。')
    actions = ['查看所选候选及同次规划快照，核对TCP、约束和碰撞几何。']
    if summary.scene_collision or summary.self_collision:
        actions = ['先查看碰撞双方与候选姿态；不能仅放宽IK误差或禁用碰撞。']
    facts=[]
    bundle=_mapping(raw.get('diagnostic'))
    if raw.get('diagnostic_error'):unknowns.append('旁路诊断失败：'+str(raw['diagnostic_error']))
    candidates=_mapping(bundle.get('candidates'))
    for key,label in [('nearest_checked','已检查中误差最小'),('nearest_checked_feasible','已检查且约束可行中误差最小')]:
        c=candidates.get(key)
        if c:
            facts.append(f"{label} seed#{c.get('seed_index')}：FK位置 {c['position_error_m']*1000:.2f} mm，转角 {c['rotation_error_rad']*180/3.141592653589793:.2f}°；约束可行={c['constraint_feasible']}；严格收敛={c['strict_pose_converged']}")
        elif candidates:unknowns.append(label+'：没有已验证候选；不能推断不存在。')
    collision=_mapping(bundle.get('collisions'))
    for pair in collision.get('pairs',[])[:3]:
        other=pair['second'];label=other.get('label') or f"{other.get('object_id','unknown')}/{other.get('mesh_id','unknown')}"
        facts.append(f"碰撞证据 [{bundle.get('configuration_ref','unknown')}]：{pair['first']['label']} ↔ {label}；允许接触={pair.get('allowed')}")
    if collision.get('truncated'):unknowns.append('碰撞对仅保留前20对，见pair_count。')
    b=bundle.get('budget')
    if b:
        facts.append(f"诊断耗时 {b['elapsed_s']:.2f}s / 软预算 {b['budget_s']:g}s；补充求解 {b['solver_calls']} 次")
        if b.get('skipped'):unknowns.append('预算/次数限制跳过：'+', '.join(b['skipped'][:5]))
    unknowns.extend(bundle.get('errors',[]))
    return replace(summary, unknowns=tuple(unknowns), next_actions=tuple(actions),diagnostic_facts=tuple(facts))


__all__ = ["MotionFailureSummary", "summarize_motion_failure", "summarize_motion_result"]


def find_diagnostic_bundles(root, *, stage=None):
    """排除actual图片sidecar；仅返回真正的规划诊断工件。"""
    import json
    from pathlib import Path
    rows=[]
    for path in sorted((Path(root)/'diagnostics').glob('*.json')):
        try:row=json.loads(path.read_text())
        except (OSError,ValueError):continue
        if 'planning' in row and 'diagnostic' in row and (stage is None or row.get('stage_id')==stage):rows.append(path)
    return rows


def validate_diagnostic_bundle(bundle, *, verify_environment=False, allow_code_change=False):
    """缺现场/依赖必须拒绝；复查时校验当前解释器及实际cuRobo来源。"""
    import hashlib,json,sys
    from pathlib import Path
    def sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    required=('replay_config','request','snapshot','environment','asset_hashes')
    if any(not bundle.get(k) for k in required):raise ValueError('缺少完整现场或依赖清单；只能读取摘要')
    replay=bundle['replay_config'];request=bundle['request'];snapshot=bundle['snapshot'];env=bundle['environment']
    for key in ('replay_config','request'):
        if sha(bundle[key])!=bundle.get(key+'_sha256'):raise ValueError(key+' hash不匹配')
    if sha(snapshot)!=bundle.get('snapshot_payload_sha256'):raise ValueError('snapshot payload hash不匹配')
    scene=request.get('scene',{})
    if not scene.get('snapshot_sha256') or not (scene['snapshot_sha256']==snapshot.get('snapshot_sha256')==replay.get('scene_snapshot_sha256')):
        raise ValueError('请求/快照/replay scene hash不一致')
    if not (scene.get('scope')==snapshot.get('scope')==replay.get('scene_scope')):
        raise ValueError('请求/快照/replay scope不一致')
    if replay.get('scene_model')!=snapshot.get('scene_model'):raise ValueError('保存的场景几何不一致')
    if not env.get('curobo_module') or env['curobo_module'] not in bundle['asset_hashes']:
        raise ValueError('缺少cuRobo依赖签名')
    changes=[]
    code_root=Path(__file__).resolve().parents[1]/'motion'
    for path,expected in {**bundle['asset_hashes'],**bundle.get('code_hashes',{})}.items():
        p=Path(path)
        actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        is_code=path in bundle.get('code_hashes',{}) or (p.suffix=='.py' and p.is_relative_to(code_root))
        if actual!=expected:
            if is_code and allow_code_change and actual is not None:
                changes.append(dict(path=path,recorded_sha256=expected,current_sha256=actual))
            else:
                raise ValueError(f"{'代码' if is_code else '复诊依赖'}缺失或hash变化: {path}")
    if bundle.get('dependency_hashes'):
        from gene_agent.motion.curobo.diagnostics import dependency_signature
        actual=dependency_signature(bundle['dependency_root'],fresh=True)
        if actual!=bundle['dependency_hashes']:
            raise ValueError('cuRobo依赖源码/配置hash变化，不能用代码变更选项替换依赖')
    # mesh/URDF不能只在配置里留下路径却缺少资产身份。
    geometry_files=[row['file_path'] for objects in replay.get('scene_model',{}).values()
                    for row in objects.values() if row.get('file_path')]
    kin=replay.get('robot_config',{}).get('kinematics',{})
    if kin.get('urdf_path'):geometry_files.append(kin['urdf_path'])
    if isinstance(kin.get('collision_spheres'),str):geometry_files.append(kin['collision_spheres'])
    for asset in geometry_files:
        if asset not in bundle['asset_hashes']:
            raise ValueError(f'几何资产缺少hash: {asset}')
    if verify_environment:
        import curobo
        if Path(curobo.__file__).resolve()!=Path(env['curobo_module']).resolve() or sys.version!=env.get('python') or Path(sys.executable).resolve()!=Path(env.get('executable','')).resolve():
            raise ValueError('当前Python/cuRobo来源与保存的现场不同；不能静默替换依赖')

    return changes


def save_motion_diagnostic(root, event, snapshot=None):
    """失败工件独立落盘；现场图缺失不拿规划预览代替。"""
    import hashlib
    import json
    from pathlib import Path
    root=Path(root);folder=root/'diagnostics';folder.mkdir(exist_ok=True)
    result=event['planning'];diagnostic=_mapping(_mapping(result.get('failure')).get('details')).get('diagnostic')
    summary=summarize_motion_result(result,snapshot)
    # 同一状态允许多个目标候选；各自保留独立失败工件，避免旧引用被覆盖。
    identity_fields={k:event.get(k) for k in ('episode_id','stage_id','state_id')}
    call_id=result.get('metrics',{}).get('planning_timing',{}).get('call_id')
    if call_id:identity_fields['call_id']=call_id
    if event.get('target_world_tcp') is not None:
        identity_fields['target_world_tcp']=event['target_world_tcp']
    identity=hashlib.sha256(json.dumps(identity_fields,sort_keys=True).encode()).hexdigest()[:16]
    path=folder/f'{identity}.json'
    payload=dict(schema_version=1,episode_id=event.get('episode_id'),stage_id=event['stage_id'],state_id=event.get('state_id'),
        target_world_tcp=event.get('target_world_tcp'),
        summary=summary.to_dict(),planning=result,diagnostic=diagnostic,
        media={'actual':{'status':'unknown','reason':'无精确现场帧映射；原视频见manifest'},
               'diagnostic':{'status':'deferred','reason':'离线查看/导出，无需重跑动作'}})
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False))
    path.with_suffix('.md').write_text(summary.to_markdown()+'\n证据：'+str(path)+'\n')
    return path
