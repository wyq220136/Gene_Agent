#!/usr/bin/env bash

# 通过已经启动的 camera server 和 seg_server 处理感知测试目录中的物体。
# 该脚本只负责检查服务并调用调试 Python 模块，不负责启动常驻服务。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TASK_DIR="${TASK_DIR:-data/perception_test}"
CONFIG="${CONFIG:-gene_agent/perception_modules/segmentation/category_config.py}"
CATEGORIES="${CATEGORIES:-all}"
CAMERA_ENDPOINT="${CAMERA_ENDPOINT:-tcp://127.0.0.1:18082}"
SEG_SERVER="${SEG_SERVER:-http://127.0.0.1:18080}"

echo "检查 camera server 和 seg_server..."
python -m gene_agent.perception_modules.segmentation.task_segmentation \
    --task-dir "${TASK_DIR}" \
    --config "${CONFIG}" \
    --camera-endpoint "${CAMERA_ENDPOINT}" \
    --seg-server "${SEG_SERVER}" \
    --categories ${CATEGORIES}
