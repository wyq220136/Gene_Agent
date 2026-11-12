"""按实际操作结果推进；后端负责观测、执行与原始证据。"""

from typing import Protocol


class CupOperations(Protocol):
    """同步操作能力；状态与前序证据由适配持有，不隐式 reset。

    操作返回 True 表示实际执行及该操作后置检查通过；False 必须保留失败阶段
    和实际终态。仿真精确颗粒身份检查不能冒称已有真机对应观测。
    """
    def pick(self) -> bool: ...
    def pour(self) -> bool: ...
    def place(self) -> bool: ...
    def record_exception(self, error: Exception) -> None: ...
    def finish(self, success: bool, scope: str) -> dict: ...


def run(operations: CupOperations, *, until_skill: str = "place") -> dict:
    """pick → pour → place；pick 确认持杯与载荷，pour 倾倒并检查转移，place 扶正、放置及释放。

    until_skill 选择从起点执行到哪一步，不是从该步恢复；失败/异常立即停止。
    finish 在成功或失败后读取实际终态并保存报告，返回 requested_scope、
    operation_success、task_success（局部为 None）、actual_final 和失败证据。
    调用方负责准备场景/设备、配置与输出目录；此函数不创建后端或授权运动。
    """
    if until_skill not in ('pick', 'pour', 'place'):
        raise ValueError("不支持的 until_skill")
    success = False
    try:
        success = operations.pick()
        if success and until_skill in ('pour', 'place'):
            success = operations.pour()
        if success and until_skill in ('place',):
            success = operations.place()
    except Exception as error:
        operations.record_exception(error)
        success = False
    return operations.finish(success, until_skill)
