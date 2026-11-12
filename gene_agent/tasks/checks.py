"""人工确认与共享检查类型的兼容入口。"""
from typing import Callable
from gene_agent.skills.contracts import CheckResult

def resolve_unknown(
    result: CheckResult, *, ask: Callable[[str], str] | None = None,
    question: str = "本阶段是否达到目标？[y/n/?] ",
) -> CheckResult:
    """只把自动检查的 `unknown` 交给人工；已有失败不能被改写。

    未提供交互入口或回答无法判断时，原样保留 `unknown`。
    调用方可注入终端 `input`，同时也便于在离线执行记录回答。
    """
    if result.status != "unknown" or ask is None:
        return result
    answer = ask(question).strip().lower()
    status = {"y": "pass", "yes": "pass", "n": "fail", "no": "fail"}.get(
        answer, "unknown")
    return CheckResult(
        status, "manual_terminal",
        {"automatic": result.to_dict(), "answer": answer, "question": question},
        "人工确认" if status != "unknown" else "人工未能确认",
    )
