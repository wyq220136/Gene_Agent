# Evidence routing

## Read order

Follow the narrowest path from task stop to root evidence:

```text
batch status or experiment summary
-> run/summary.json
-> failure.report_ref and traceback_ref
-> operation report.json or result.json
-> phase plan, execution, contact, and postcondition records
-> failure.json, diagnostics.json, collision_snapshot.json, state.npz
-> actual execution video and its timing metadata
-> trajectory PNG or PLY only for planner-geometry inspection
-> runtime source path from target generation through the rejecting check
```

Directory layouts vary. Search by field and phase name when a file is not at the expected path. Record a missing artifact as a logging gap; do not fabricate a result.

## Evidence roles

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Canonical stage SVG | Stable task vocabulary, order, pass conditions | Which run failed or why |
| `summary.json` | Stop operation, high-level status, evidence references | Root cause by itself |
| Planner result or `plan.json` | IK/path result, constraints, planner diagnostics | Physical execution or contact |
| Execution record | Commanded and actual joints, tracking error, whether execution occurred | Task success after contact/release |
| MuJoCo contact/state | Actual simulated state and contacts | Whether a contact is allowed by task semantics |
| Gene Agent postcondition | Why the operation was accepted or rejected | Which earlier target-generation choice caused it without further analysis |
| Actual video | Visible pose, motion, missed contact, slip, abnormal joint behavior | Exact collision object, threshold, or solver state without logs |
| Trajectory PNG/PLY | Planned geometry and relative scene layout | Actual executed motion |

## Failure-stack reconstruction

Build a traceback-like chain:

1. **Task-level symptom:** what failed to happen or what visibly went wrong.
2. **Operation/phase rejection:** which phase or postcondition stopped the run.
3. **Tool or numeric evidence:** solver result, collision pair, error threshold, contact state, angle, displacement, or traceback.
4. **Cause analysis:** which upstream target, state, transition, execution, or contact condition produced the rejection.
5. **Code ownership:** which live source function produced that target, transition, policy, or condition.

Do not jump from step 1 directly to a solver label. Do not stop at step 3 when video and actual-state evidence reveal an upstream target-generation problem. Before editing, confirm that the identified source function is consumed by the failing runtime path.

## Same-state comparisons

Prefer comparisons that preserve:

- seed and randomized object pose;
- full initial simulator state or verified pose fingerprint;
- planner backend and mode;
- collision-world snapshot and attachment state;
- target request;
- controller and physics parameters unless they are the single tested factor;
- operation and complete-task success criteria.

Use frozen-request replay for planner hypotheses. Use fresh-reset full-operation replay for execution, contact, and postcondition hypotheses. A local resumed state is useful for diagnosis but does not replace a complete-task regression.
