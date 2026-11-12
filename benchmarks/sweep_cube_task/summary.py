"""按固定 seed 汇总清扫任务的完整 MuJoCo 试验。"""

from __future__ import annotations

from collections import Counter


def summarize_evaluation(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("测评至少需要一个 seed")
    successes = sum(bool(row["success"]) for row in rows)
    failures = Counter(row.get("failure_stage") or "unknown_failure"
                       for row in rows if not row["success"])
    return {"episodes": len(rows), "successes": successes,
            "success_rate": successes / len(rows),
            "failure_counts": dict(failures), "runs": rows}
