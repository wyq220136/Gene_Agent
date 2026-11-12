"""Grasp profile 的确定性离散范围展开。"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
from typing import Any, Mapping, Sequence


_RANGE_KEYS = frozenset({"start", "stop", "step", "order"})
_VALUE_KEYS = frozenset({"values"})
_ORDERS = frozenset({"ascending", "descending", "center_out"})


def _finite_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} 必须是有限数值")
    return result


def _decimal(value: Any, *, path: str) -> Decimal:
    number = _finite_number(value, path=path)
    try:
        return Decimal(str(number))
    except InvalidOperation as error:  # pragma: no cover - finite float 已覆盖该分支
        raise ValueError(f"{path} 必须是有限数值") from error


def _validate_value(value: Any, *, path: str) -> float | str:
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{path} 字符串不能为空")
        return value
    return _finite_number(value, path=path)


def order_values(
    values: Sequence[float | str], order: str
) -> tuple[float | str, ...]:
    """按显式规则排序；center_out 等距时保持输入中的先后顺序。"""
    if order not in _ORDERS:
        raise ValueError(f"order 不支持 {order!r}")
    copied = tuple(values)
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in copied
    ):
        raise ValueError(f"{order} 排序只支持数值")
    numeric = tuple(float(value) for value in copied)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"{order} 排序只支持有限数值")
    if order == "ascending":
        return tuple(sorted(numeric))
    if order == "descending":
        return tuple(sorted(numeric, reverse=True))

    low = min(numeric)
    high = max(numeric)
    # 同号大数直接相加可能溢出为 Infinity；先各自除以 2 可保持有限中点。
    center = 0.0 if low <= 0.0 <= high else low / 2.0 + high / 2.0
    # Python 排序稳定；距离相同的值沿输入顺序排列，
    # 因此不受 set/dict 顺序影响。
    return tuple(
        value
        for _, value in sorted(
            enumerate(numeric), key=lambda item: (abs(item[1] - center), item[0])
        )
    )


def expand_sample_spec(
    spec: Mapping[str, Any],
    *,
    path: str,
    max_samples: int = 10_000,
) -> tuple[float | str, ...]:
    """展开 ``values`` 或可达终点的浮点闭区间。

    范围用十进制字符串语义计算，避免 ``0.1 + 0.2`` 的二进制误差
    影响端点。
    ``stop`` 必须由整数个 ``step`` 到达；函数不会静默截断到 stop 之前。
    """
    if not isinstance(spec, Mapping):
        raise ValueError(f"{path} 必须是参数采样对象")
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0:
        raise ValueError("max_samples 必须是正整数")

    keys = set(spec)
    if "values" in keys:
        unknown = keys - _VALUE_KEYS
        if unknown:
            key = sorted(unknown)[0]
            raise ValueError(f"{path}.{key} 是未知字段")
        values = spec["values"]
        if not isinstance(values, list):
            raise ValueError(f"{path}.values 必须是列表")
        if not values:
            raise ValueError(f"{path}.values 不能为空")
        if len(values) > max_samples:
            raise ValueError(f"{path} 超过最大样本数 {max_samples}")
        return tuple(
            _validate_value(value, path=f"{path}.values[{index}]")
            for index, value in enumerate(values)
        )

    unknown = keys - _RANGE_KEYS
    if unknown:
        key = sorted(unknown)[0]
        raise ValueError(f"{path}.{key} 是未知字段")
    for key in ("start", "stop", "step"):
        if key not in spec:
            raise ValueError(f"{path}.{key} 是必填字段")

    start = _decimal(spec["start"], path=f"{path}.start")
    stop = _decimal(spec["stop"], path=f"{path}.stop")
    step = _decimal(spec["step"], path=f"{path}.step")
    if step == 0:
        raise ValueError(f"{path}.step 不能为零")
    delta = stop - start
    if delta > 0 and step < 0 or delta < 0 and step > 0:
        raise ValueError(f"{path}.step 方向无法从 start 到达 stop")

    if delta == 0:
        count = 1
    else:
        steps = delta / step
        if steps != steps.to_integral_value():
            raise ValueError(f"{path}.stop 必须可由整数个 step 到达")
        count = int(steps) + 1
    if count > max_samples:
        raise ValueError(f"{path} 超过最大样本数 {max_samples}")

    values = tuple(float(start + step * index) for index in range(count))
    order = spec.get("order")
    if order is None:
        return values
    if not isinstance(order, str):
        raise ValueError(f"{path}.order 必须是字符串")
    try:
        return order_values(values, order)
    except ValueError as error:
        raise ValueError(f"{path}.order: {error}") from error
