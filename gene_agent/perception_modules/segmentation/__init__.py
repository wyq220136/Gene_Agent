"""Segmentation server 客户端接口。"""

from importlib import import_module

_EXPORTS = {
    "segment_image": "seg_client",
    "segment_task_objects": "task_segmentation",
    "llm_bbox_segment": "llm_bbox_seg",
    "llm_bbox_fallback_batch": "llm_bbox_seg",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
