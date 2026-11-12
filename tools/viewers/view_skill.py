"""查看门、Bowl、旋钮 Skill，或清扫 Skill 的 MuJoCo 实际回放。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    if '--follow' in sys.argv:
        sys.argv.remove('--follow')
        from gene_agent.visualization.live_run import main
        if '--output' not in sys.argv:
            from uuid import uuid4
            sys.argv+=['--output',str(Path('runs/tmp/visual_planning')/('live_'+uuid4().hex[:10]))]
        main()
        raise SystemExit(0)
    if '--run' in sys.argv:
        import json
        report=Path(sys.argv[sys.argv.index('--run')+1])/'result.json'
        if report.is_file() and json.loads(report.read_text(encoding='utf-8')).get('record_type')=='pure_visual_plan':
            from gene_agent.visualization.pure_planning.cli import main
            # 旧入口的 Task 语义名不参与纯规划回放筛选。
            if '--task' in sys.argv:
                index=sys.argv.index('--task');del sys.argv[index:index+2]
            main()
            raise SystemExit(0)
    import os
    if "--render-only" in sys.argv:os.environ.setdefault("MUJOCO_GL","egl")
    from gene_agent.visualization.realworld_sweep import dispatch_legacy_viewer
    from gene_agent.visualization.saved_run_cli import dispatch_saved_run
    if not dispatch_legacy_viewer("skill") and not dispatch_saved_run("skill"):
        from _visual_object_cli import main
        main()
