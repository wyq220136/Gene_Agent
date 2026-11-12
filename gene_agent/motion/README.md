# Skill 怎样调用运动规划

写新Skill从本页开始。推荐使用 `StageMotion`，它绑定本阶段的关节起点、场景、
seed和模式，返回已有 `MotionResult`。所有规划都不下发机械臂或夹爪命令。

编写前先读[规划代码常见错误](../../human_experience.md#编写规划代码时的性能经验)。应用层持有整个操作生命周期的 session，Skill 不创建 solver；真实执行后读取实际状态再建下一 stage。修改坐标、夹爪或持物模型时先验证受影响的 FK/碰撞，再扩大任务试验。

## 1. Skill只接收阶段接口和目标

```python
from gene_agent.motion import PoseTarget, StageMotion

def plan_approach(motion: StageMotion, target_world_tcp):
    return motion.plan_linear(
        PoseTarget("world_tcp", target_world_tcp),
        request_id="pick/approach",
    )
```

目标为 `T_world_tcp` 的4×4矩阵，长度用米、角度用弧度。Skill不创建solver，
不读取机器人型号配置。下表中的target均为 `PoseTarget("world_tcp", matrix)`。

| 需求 | 方法 |
|---|---|
| 绑定起点的TCP | `motion.fk()` |
| 单目标IK | `motion.solve_ik(target, request_id="ik")` |
| 批量IK | `motion.evaluate_ik(targets, request_ids=["a", "b"])` |
| 普通pose轨迹 | `motion.plan_to(target, request_id="move")` |
| 严格TCP直线 | `motion.plan_linear(target, request_id="approach")` |
| 限制TCP偏移/路程的旋转 | `motion.plan_pivot(target, request_id="turn", limits=PivotLimits(.01, .04))` |

`PivotLimits`从gene_agent.motion导入，两个限制均为米；pivot不支持近似IK组合。
批量IK保留输入顺序与失败项，只筛终点；普通规划不保证全局最短。
批量request_ids须与targets数量相同且互不重复；空批返回空元组。
直线失败不会自动换成普通轨迹。允许误差的阶段可显式配置原有策略：

```python
result = motion.plan_to(
    target, request_id="pick/pregrasp",
    approximate_ik_budget={
        "max_position_error_m": 0.01,
        "max_rotation_error_rad": 0.05,
    },
)
```

默认严格。只有预算内且通过原碰撞、约束和路径后检的候选才可接受。
检查 `result.metrics["approximate_ik"]` 中的实际目标和误差；关键抓紧、插入
终态不能仅为解决IK失败而放宽。[进阶规则](../../docs/developer/motion/motion_baseline.md)。

## 2. 应用层提供当前状态、场景与session

```python
from gene_agent.motion import JointState
from gene_agent.motion.curobo import CuroboPlanningSession

with CuroboPlanningSession(robot_metadata, seed=123) as session:
    with session.stage(
        snapshot=snapshot,
        start=JointState(joint_names, current_joints_rad),
        group=active_group,
        scene_revision=revision,
    ) as motion:
        result = plan_approach(motion, target_world_tcp)
```

输入来源：

- 仿真：robot_metadata来自机器人/场景构建，起点从当前runtime/MuJoCo读取，
  快照用 [export_collision_scene](curobo/scene.py) 从相同当前状态导出。
- 无MuJoCo的场景：沿用 [visual_inputs](../scene/visual_inputs.py) 的准备入口；
  从其机器人metadata、冻结状态及快照装配，不伪造实际接触。
- 真机：先完成 [现场规划](../realworld/planning.py) 的provenance校验。
  首帧、实测另一臂、坐标系和快照hash检查仍必需；新session不代替现场校验或读取SDK。

session默认CUDA Graph、容量4的pool，正常/异常退出均释放，精确更新兼容的场景及
夹爪几何。默认模式继承外层配置，未配置时为full；显式模式须与快照及有效外层配置
一致。robot_only保留自碰撞和机器人保护，跳过普通场景及物体规划attachment。
[模式说明](../../docs/user/motion/planning_modes.md)、[碰撞快照](../../docs/developer/motion/motion_scene.md)。

同一session或共享pool只允许一个活跃stage。退出后旧motion和其planner视图失效。
一个stage绑定一个冻结起点，多次调用表示同一起点的候选评估/重试，不会自动推进
到上一条规划的预测终点。真实执行后重新观测再建立下一阶段；纯预览可显式使用
`result.goal_state`继续，但须标为预测状态。门角、夹爪、attachment等变化后更新快照。

`restart_sampling=True`仅用于明确复现旧冷实例的阶段边界；默认沿用pool采样策略。
批次/重试之间不要随意重置。`collision_policy`声明预期接触，不代表任意允许表
都已支持；实际接触配置需要对应快照及应用/执行后检，后端仍拒绝不支持的策略。

## 3. 保存结果，交给执行器

```python
if not result.success:
    print(result.failure.kind, result.failure.summary)
else:
    trajectory = result.trajectory
    predicted_end = result.goal_state
    # 交给现有仿真/真机执行器，随后读取实际状态。
```

success只表示本次规划通过，不证明抓住、移动物体或完成任务；IK结果没有轨迹。
保存 `result.to_dict()`：失败诊断在failure.details，路径/碰撞/近似信息在metrics，
缓存证据在 `metrics["planner_cache"]`，其中setup_duration_s是获取/初始化耗时。
result.duration_s沿用后端规划计时，首次图捕获与热调用分开比较。

每次调用另有 `metrics["planning_timing"]`：call_id、call_wall_s、batch_size/index。
batch 多个结果共享一次调用，不能逐项累加 call_wall_s；同次 lease 的
`planner_cache.acquisition_id` 只计一次 setup。失败尝试也统计。
`duration_s`、调用 wall 和执行桥等待有包含关系，不能相加。

`ProcessPlanningSession` 指定 log_path 后，在同目录追加 `planning_calls.jsonl`，
记录请求往返和异常。需要独立调试成功/失败请求时，显式指定
`request_output_dir=<本轮目录>/motion_requests`；同进程 session 则在
`session.stage(..., request_output_dir=...)` 指定。此选项会写完整模型/资产身份，
批次只写一份现场，结果给出 `metrics.replay_request_ref`。记录出错保留原结果并标明原因。
使用[离线重规划与统计命令](../../docs/user/motion/curobo_diagnostics.md#计时与独立重规划)，无需重新执行仿真前缀。

已有MotionRequest可通过阶段内 `motion.planner` 协议视图调用，但必须符合冻结上下文。
CPU单元测试可用 `StageMotion(fake_planner, PlanningContext(...))` 手工装配，
PlanningContext从gene_agent.motion导入，字段为group/start/scene/seed/planner_mode。
旧多阶段Skill需要逐请求建立stage，不能整体塞入一个冻结起点；参考
[放入碗迁移](../tasks/policies/microwave_task/planning/insert.py)。旧低层接口继续可用。

## 4. 直接跑一次

```bash
PYTHONUTF8=1 PYTHONPATH=. PYTHONNOUSERSITE=1 \
  /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/diagnostics/motion_skill_example.py \
  --output runs/tmp/motion_example/my_run --collision-mode robot_only
```

输出目录需不存在。示例自建单臂Pika诊断模型，测试IK、普通轨迹、直线、pivot、
批量IK、误差预算及失败样例，结果和首次/热耗时写入result.json。
预算正例使用严格可解目标，只验证预算透传；近似接受/超预算拒绝的策略回归见
[test_approximate_ik_plan.py](../../tests/motion/test_approximate_ik_plan.py)。
固定诊断姿态不代表任务场景已验收；不连接任何设备。
契约见 [model.py](model.py)、[planner.py](planner.py)；后端细节与历史证据见
[motion_baseline.md](../../docs/developer/motion/motion_baseline.md)。
