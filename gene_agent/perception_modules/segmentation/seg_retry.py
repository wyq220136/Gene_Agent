"""分割失败重试的无模型通用逻辑。

具体的 ``set_image`` 和 prompt 调用由调用方通过 ``run_attempt`` 注入，
这样重试策略不会依赖 CUDA、SAM3 或 HTTP 客户端。
"""


def retry_failed_items_with_fresh_image(
    items,
    run_attempt,
    *,
    max_attempts=10,
    on_retry=None,
):
    """重复执行失败项，每次执行由调用方重新注册图片后再提交 prompt。

    ``run_attempt(pending_items)`` 必须完成一次完整尝试：重新调用
    ``set_image(image)``，然后只对 ``pending_items`` 使用原 prompt 分割，
    并返回本轮仍然失败的项。函数返回 ``(remaining, attempt_count)``。
    """
    max_attempts = int(max_attempts)
    if max_attempts < 1:
        raise ValueError("max_attempts 必须至少为 1")

    pending = list(items)
    attempt_count = 0
    while pending and attempt_count < max_attempts:
        attempt_count += 1
        if attempt_count > 1 and on_retry is not None:
            on_retry(attempt_count, max_attempts, pending)
        pending = list(run_attempt(pending))
    return pending, attempt_count
