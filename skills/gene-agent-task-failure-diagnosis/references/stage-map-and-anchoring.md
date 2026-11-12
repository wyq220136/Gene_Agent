# Stage map and diagnostic anchoring

## Purpose

Use one stable task vocabulary across logs, reports, videos, fixes, and regressions. The stage map prevents the same stop from being described as `pick_insert`, `pick_bowl`, `load_bowl`, or a later stage depending on the reporter.

## Selection order

1. Use an existing canonical task SVG referenced by repository documentation.
2. For microwave Task1, always use `docs/01-task-skills.svg`. Do not regenerate, replace, rename, or reinterpret its seven stages.
3. If no map exists, reconstruct the executed flow from task orchestration source, operation/sub-task definitions, phase names in run artifacts, transition code, and final acceptance conditions.
4. Freeze the new map before writing the failure report. Reuse the same map for all later runs of that task until the task contract changes.

## Required map contents

A useful map records:

- ordered top-level stages and stable stage identifiers;
- operation or sub-task names used in code and logs;
- internal phases in execution order;
- motion semantics when relevant, such as ordinary planning, TCP-linear motion, segmented linear motion, or fallback policy;
- pass conditions for the operation;
- Task-owned transitions between operations;
- complete-task acceptance conditions;
- source files or notes used to derive the map.

Do not put failure counts, run-specific parameters, temporary hypotheses, or planned future changes into the canonical map.

## Mapping a failure

Report all four levels when available:

```text
Task stage -> operation/sub-task -> Skill -> internal phase
```

Example for the fixed microwave map:

```text
Stage 2 load_bowl -> pick_bowl -> Pick -> approach
Stage 2 load_bowl -> insert_bowl -> Insert -> preplace
Stage 4/5 Task transition -> before_reopen_transition -> return_default_tcp
```

The third example is not Stage 5 because Stage 5 has not begun.

## Creating a map for another task

Create a JSON specification and render it with the bundled script:

```json
{
  "title": "Example Task: stages, phases, and pass conditions",
  "subtitle": "Canonical executed flow; each stage consumes the previous actual state.",
  "version": "v1",
  "sources": [
    "gene_agent/tasks/policies/example_task/simulation.py",
    "runs/example_run/run_001/summary.json"
  ],
  "stages": [
    {
      "id": "1",
      "label": "Acquire object",
      "operation": "load_object",
      "phases": ["open", "pregrasp", "approach", "close", "lift"],
      "pass_conditions": ["bilateral contact", "object lifted and held"]
    },
    {
      "id": "2",
      "label": "Place object",
      "operation": "place_object",
      "phases": ["preplace", "place", "release", "settle", "retreat"],
      "pass_conditions": ["supported", "stable after retreat"]
    }
  ],
  "task_success": ["all stages passed", "no attachment remains"]
}
```

Render:

```bash
python3 skills/gene-agent-task-failure-diagnosis/scripts/render_stage_map.py \
  --input docs/example-task-skills.json \
  --output docs/example-task-skills.svg
```

After rendering, open the SVG and verify stage order, phase text, pass conditions, and complete-task conditions. The JSON is the editable source; the SVG is the visual anchor.

## Updating a map

Update the map only when the executed task contract changes. Record the new version and source paths. Do not update it merely because a run stopped early or a diagnostic label changed.
