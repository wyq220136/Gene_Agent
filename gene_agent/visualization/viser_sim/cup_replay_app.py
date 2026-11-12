"""杯子 Task 的 Viser MuJoCo 实际状态回放。"""

from __future__ import annotations

from pathlib import Path

from gene_agent.feedback.tabletop.cup_run import CUP_SKILLS, CUP_TASKS, find_latest_successful_run, load_replay
from gene_agent.visualization.viser_sim.spoon_replay_app import SpoonReplayApp


def _status(report, row, relative, duration, run_dir):
    pour = report.get("skills", {}).get("pour", {})
    place = report.get("skills", {}).get("place", {})
    return (f"**{row['stage']}** · {relative:.2f}/{duration:.2f} s · "
            f"run: `{run_dir}`  \n"
            f"Task 成功：`{report['success']}`；杯内初始 "
            f"`{report.get('initial_cargo_count', '?')}` 球；进碗 "
            f"`{pour.get('transferred_count', '?')}` 球；溢出 "
            f"`{pour.get('spilled_count', '?')}` 球；"
            f"放回：`{place.get('success', '?')}`；"
            "夹持：`双指接触后仿真辅助`；画面：`实际 MuJoCo 状态`。")


class CupReplayApp(SpoonReplayApp):
    def __init__(self, server, *, initial_selection: str, run_dir: Path | None = None):
        super().__init__(
            server, initial_selection=initial_selection, run_dir=run_dir,
            replay_loader=load_replay, latest_run_finder=find_latest_successful_run,
            skill_labels=CUP_SKILLS, task_labels=CUP_TASKS,
            intro=("### 杯子倾倒 · MuJoCo 实际执行回放\n"
                   "从上向下抓杯把，倾倒后放回。时间轴读取执行时记录的 qpos。"
                   "双指接触杯把后才可启用仿真辅助夹持；球由 MuJoCo 接触运动。"),
            camera_look=[.34, -.02, .12],
            camera_position=[-.28, .56, .52],
            status_formatter=_status,
        )
