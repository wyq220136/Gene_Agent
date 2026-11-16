"""Python 3.9 兼容工具；保留新版本标准库的检查语义。"""

from __future__ import annotations


def strict_zip(*iterables):
    """等价于 zip(..., strict=True)，长度不一致时拒绝静默截断。

    与内建 zip 一样，在调用时取得 iterator，消费时才逐项检查长度；
    不预先把轨迹或候选集合展开为列表。
    """
    iterators = tuple(iter(value) for value in iterables)
    return _strict_zip_rows(iterators)


def _strict_zip_rows(iterators):
    if not iterators:
        return
    sentinel = object()
    while True:
        row = []
        for index, iterator in enumerate(iterators):
            try:
                value = next(iterator)
            except StopIteration:
                if index:
                    previous = "argument 1" if index == 1 else f"arguments 1-{index}"
                    raise ValueError(f"zip() argument {index + 1} is shorter than {previous}") from None
                for other_index, other in enumerate(iterators[1:], start=2):
                    if next(other, sentinel) is not sentinel:
                        previous = "argument 1" if other_index == 2 else f"arguments 1-{other_index - 1}"
                        raise ValueError(f"zip() argument {other_index} is longer than {previous}") from None
                return
            row.append(value)
        yield tuple(row)
