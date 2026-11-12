# 批量 IK 候选评估

`MotionPlanner.evaluate_candidates(requests)` 是同步、后端无关接口。输入只接受一批 `IK` 请求，要求相同的 robot group、具名起始关节状态、冻结场景引用、目标 frame、碰撞策略、seed 和 planner mode。候选 `request_id` 必须唯一，输出 tuple 严格保持输入顺序。空输入返回空 tuple；混用场景或起点立即报错，不会在后台分组或悄悄修改场景。

cuRobo 实现位于 `CuroboPlannerAdapter.evaluate_candidates` 和 `SynchronousPlanner.solve_ik_batch`。锁定夹爪关节在返回解中先按活动关节名称剔除，避免改变 batch/seed 维度。后者用一个 `GoalToolPose` 的 batch 维放入全部目标，构造 `max_batch_size=B` 的 IK solver，只调用一次 `solve_pose`。本地 cuRobo 对 `B>1` 进入 `SolveMode.BATCH`，Levenberg–Marquardt 种子计算也调用 `solve_batch`。同一候选数复用 solver 实例；单目标 `solve_ik` 仍使用自己的 solver，不改变既有行为。

批量求解沿用 16 个 IK seeds、自碰撞、冻结场景、原始 collision spheres 和安装接触策略。它不会规划路径，也不会自动对失败候选做第二次单目标求解。每个返回解都会经独立碰撞检查；安装接触策略另有严格复核。失败项保留位置/姿态误差、阈值、约束可行性，以及必要时的碰撞查询。`duration_s` 是一次 batch 墙钟耗时除以候选数，便于旧调用方求和；计时包含逐项安装接触复核和必要的失败诊断。整批实际耗时保存在低层字典 `batch_total_duration_s`，基准脚本另用墙钟测量。

## 复现基准

在 `curobo` 环境中运行：

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/diagnostics/check_motion_batch.py \
  --case runs/pick-bowl-v2-verified \
  --limit 40 --seed 123 --warmup-runs 1 --measure-runs 2 \
  --output runs/motion-batch/report.json
```

该脚本读取同一保存的碗场景、候选、起点和碰撞快照，为串行和 batch 分别创建 solver；记录冷启动、预热、热态墙钟时间、候选成功集合、逐项失败事实和 PyTorch GPU 峰值内存。它不运行轨迹规划或物理抓取。

本机 RTX 4090 的最近一次实测：40 个候选的热态中位时间，串行 **3.133 s**，batch **0.523 s**，加速 **5.98 倍**；PyTorch 峰值 allocated 分别 **18 MB** 和 **354 MB**。串行成功 30/40，与旧存档成功 ID 一致；batch 成功 27/40。仅串行成功的 `bowl_ring_01_00`、`bowl_ring_01_01`、`bowl_ring_04_03` 在 batch 中位置误差为 7.1、5.1、7.4 mm，超过 5 mm 阈值；姿态误差在阈值内，约束可行性为 True，未确认碰撞。两组使用相同的 16 seeds 参数和 seed 整数，但**候选差异的具体原因尚未确认**。相同 seed 整数不保证 batch 与串行内部每个候选的随机序列或优化路径逐位一致。前一次独立运行的串行结果是 **29/40**、batch **26/40**；虽然 seed 参数相同，单候选波动原因也未确认，不能断定它只由优化排序造成。

脚本还演示一条**显式上层策略**：只把最后一轮 batch 失败的 13 个候选交给新建 serial solver，每个重试一次。该次重试额外耗时 **2.018 s**，包括串行 solver 冷启动；与 batch 相加为 **2.537 s**，合并成功 **31/40**。恢复的候选与原串行成功集合并不完全一致，说明单次二次求解不是稳定召回保证。这只是性能/召回率试验，未写入 motion backend，也未成为 Skill 默认行为。若 Skill 需要采用，应显式决定重试预算并在反馈中记录成本与每个结果。

## 验收边界

- 单元测试：`tests/test_motion_batch.py` 检查次序、ID、失败诊断、空输入和混合上下文拒绝。
- GPU 基准：`runs/motion-batch/report.json` 保留固定场景、seed、逐次耗时、成功差集和失败详情。
- 目前只批量筛选目标 pose 可达性。`plan()` 仍对选中候选规划轨迹；路径失败、pre-grasp 和闭爪应由 Skill 阶段处理。

## GPU批量轨迹的收益评估（2026-10-30）

已用本机实际 `BatchMotionPlanner` 做独立实验，尚未接入正式 typed 接口。
从相同已保存的碗场景选择4/8/16个 pregrasp 候选；固定seed123、16 IK seeds、
4 trajectory seeds、1 attempt、1 finetune、CUDA Graph开启，双方均不启用
PRM graph seeding。每组先测一轮冷调用，再取三轮热调用中位数。
两边均检查完整离散轨迹及生产路径的严格安装/场景碰撞复核，并检查终点
5 mm / 0.05 rad。计时包含取结果和后检。

| 模式／候选数 | 串行全部 | 批量全部 | 串行找到首条成功 | 批量返回首条成功 |
|---|---:|---:|---:|---:|
| robot_only／4 | 189 ms | 122 ms | 94 ms | 122 ms |
| robot_only／8 | 391 ms | 232 ms | 98 ms | 232 ms |
| robot_only／16 | 677 ms | 480 ms | 98 ms | 480 ms |
| full／4 | 868 ms | 747 ms | 441 ms | 747 ms |
| full／8 | 1498 ms | 1582 ms | 419 ms | 1582 ms |
| full／16 | 3110 ms | 3164 ms | 444 ms | 3164 ms |

“串行找到首条”是同一逐候选测量的成功前缀耗时；batch必须等待整批求解和
后检。两边均使用库原生求解＋相同后检，未加入正式 wrapper 失败时额外的
归因诊断，因此不是正式接口调用延迟承诺。三轮热态并非场景分布成功率。

每个独立实验中串行 PyTorch 峰值 allocated 约129 MiB；batch4/8/16约
502/1003/1992 MiB。它包含本实验保留的公共状态及前一串行结果，不等同于
单个求解器的独占显存或 nvidia-smi 总占用；原始报告也保存增量及reserved。
单独构造约90–211 ms；首次全量调用串行2.17–5.89 s、batch1.19–5.31 s，
包含首次图捕获等准备。可见复用比每次启动一个batch求解器更关键。

两模式的末轮成功数量：串行4/8/16候选分别为3/7/12（robot_only）、
3/6/13（full）；batch分别4/8/13、4/7/14。成功集合有差异，seed参数相同
不保证逐候选的随机样本和优化路径相同，不能当作等价后端直接替换。

建议：当前任务通常按优先级找到第一条可行路径即停止，继续用复用后的
串行规划。需要评价大量候选并比较质量时，可显式考虑小批量；robot_only
全量筛选有1.4–1.7倍收益，full下完整后检使收益不稳定。此次仅评估，
没有改变正式候选顺序、失败处理、碰撞或成功条件。

临时复现入口为 `runs/tmp/curobo_bowl_batch/20261030/batch_experiment.py`
（`--mode full|robot_only --count 4|8|16`），原始逐轮候选结果在同目录
`batch/<mode>/<count>/summary.json`，汇总 `batch_summary.json`。
该临时脚本依赖保存场景，要求新建输出目录；不作为已承诺的长期接口。
