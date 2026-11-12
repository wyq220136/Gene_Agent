#!/usr/bin/env bash
set -u

ROOT=/data1/yuquan/Gene_Agent_new
PY=/data1/yuquan/miniconda3/envs/gene_agent/bin/python
CONFIG=$ROOT/data/task_1/simulation/mujoco_task1_physical_varied_bowl_front.json
RUN_ROOT=${1:?usage: $0 RUN_ROOT}

mkdir -p "$RUN_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT"
cd "$ROOT"

printf 'batch=microwave_task1_varied_bowl_front20\nconfig=%s\nseed_range=123..142\nvideo=actual overview+detail\nstarted=%s\n' \
  "$CONFIG" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_ROOT/launcher.log"

for i in $(seq 1 20); do
  run_name=$(printf 'run_%03d' "$i")
  run_dir="$RUN_ROOT/$run_name"
  terminal_log="$RUN_ROOT/.${run_name}.terminal.log"
  seed=$((122 + i))
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '%s\t%s\t%s\t%s\n' "$run_name" "$seed" "$start" 'running' >> "$RUN_ROOT/batch_status.tsv"

  "$PY" "$ROOT/tools/run/run_microwave_task1_physical.py" \
    --config "$CONFIG" --output "$run_dir" --video --randomize --seed "$seed" \
    > "$terminal_log" 2>&1
  rc=$?
  end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  video_status=missing
  if [ -f "$run_dir/video/task.mp4" ]; then
    video_status=ok
  fi
  if [ -d "$run_dir" ] && [ -f "$terminal_log" ]; then
    mv "$terminal_log" "$run_dir/terminal.log"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_name" "$seed" "$start" "$end" "$rc" "$video_status" \
    >> "$RUN_ROOT/batch_completed.tsv"
done

printf 'completed=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/launcher.log"
