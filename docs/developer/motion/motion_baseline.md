# 同步 cuRobo 基线

编写新Skill先读 [Motion使用入口](../../../gene_agent/motion/README.md)。本页保留
后端细节、迁移证据及性能历史；不需要先通读本页才能调用规划。

## 阶段调用入口验收（2026-10-30）

`StageMotion` / `PlanningContext` 负责请求绑定，`CuroboPlanningSession` 负责
装配及有界pool生命周期。stage结束后导出的motion和planner视图失效；共享pool
也禁止另一调用在活跃stage中切换solver。旧请求默认值、序列化和调用继续兼容。
放入碗入口默认使用新接口，可显式 `motion_interface="legacy"` 做旧路径对照。
当前仅迁移该实际流水线，真机准备函数的全部provenance检查原样保留。
其他任务的独立物理/真机验收完成前，不删除旧实现。

typed请求增加PIVOT_TCP与PivotLimits，调用原有pivot求解和后检。实际示例发现
零平移pivot会被原直线方向计算拒绝，现仅在pivot的零位移分支使用起点TCP Z轴
作为辅助方向；实际目标、碰撞及路程限额保持，普通直线的零方向拒绝保持。
新旧cache key统一显式/省略full与attachment参数位置的等价表示，完整球几何仍
参与身份，防止混用入口导致重复初始化。

固定场景/seed的完整纯规划：full新旧均7/7、15次构造；robot_only均7/7、8次构造。
碗的候选、目标、阶段和关节轨迹逐元素相同。整轮full69.24→70.78s，robot_only
36.37→35.94s；full新接口测量与CPU回归同时进行，不能从该时间差推导稳定性能变化。
另在同进程交错8轮、丢弃前两轮，对包含配置/获取/请求/规划的热耗时测量：
full旧65.13ms / 新65.27ms，robot_only旧65.12ms / 新64.90ms；均仅构造1个solver，
轨迹最大关节差9.98e-7rad。结论是未观察到明显额外开销，不承诺逐位确定或提速。

离线示例在full、robot_only下通过IK、普通规划、直线、零位移pivot、batch IK、
预算透传、不可达拒绝及独立场景碰撞负例。预算正例严格可解，不作为GPU近似接受
分支已验收的证据；已有策略正/负例CPU回归保留。扩大回归682项通过，之后缓存规范化
与pivot修正的Motion/Skill/碗专项281项通过。独立Agent仅读代码地图与README，约53秒
写出短Skill，3个CPU用例验证批量筛选、严格直线与失败保留。
最终连同微波炉现场适配的专项368项通过，记录在`final_check_verified.log`。

证据：`runs/tmp/motion_interface/20261030/`中的`comparison.json`、
`example_{full_final,robot_only}/result.json`、对应`interface_overhead.json`、
`full_regression.log`、`final_core_verified.log`及`agent_demo/cpu_verification.json`。
首次示例pivot失败记录与修正后记录均保留。本轮不含MuJoCo物理或真机运动。

## 现行接口与保行为优化（2026-10-30）

调用方优先使用 `MotionRequest → CuroboPlannerAdapter → MotionResult`。
`motion/planner.py` 是后端无关协议，`motion/model.py` 定义请求/结果，
`motion/curobo/adapter.py` 校验和转换 typed 接口，`core.py` 只负责底层求解。
原 `from gene_agent.motion.curobo.core import CuroboPlannerAdapter` 等入口仍可用。
场景导出及机器人配置准备沿用 `scene.py`、机器人适配；不要在 Task 中重新定义坐标变换。

| 需求 | 现有接口 | 语义与使用边界 |
|---|---|---|
| 单目标 IK | `adapter.solve_ik(request)`，operation=`IK` | 检查目标可达性和当前碰撞约束，不证明到达路径可行 |
| 普通轨迹 | `adapter.plan(request)`，operation=`PLAN` | 保持原轨迹优化；不保证全局最短，不自动选择另一抓取目标 |
| TCP 直线 | 同上，operation=`LINEAR_TCP` | 保留全部插值点 FK 检查：侧偏、倒退、越界、端点和姿态；失败不隐式改为关节路径 |
| 原位旋转/限制绕路 | `planner.plan_pivot_tcp(..., max_tcp_excursion_m=..., max_tcp_path_length_m=...)` | 已有低层接口，限制 TCP 活动范围和总路程；尚未纳入 typed operation |
| 批量 IK | `adapter.evaluate_candidates(requests)` / `planner.solve_ik_batch(...)` | 同起点/场景/seed/mode；保留顺序和失败项；不等于批量轨迹规划 |
| 多候选轨迹筛选 | `CuroboRuntimeBridge.plan_and_execute_candidates(...)` | 当前顺序检查完整路径并选择第一个成功候选；保持原顺序，不称为 GPU batch trajectory |
| 有限 IK 误差 | `MotionRequest.approximate_ik_budget` 或 `plan_with_approximate_ik(...)` | 显式阶段预算；严格失败后检查候选、求 FK、正常重规划、复核终点，执行桥另核验真实终态 |
| 场景碰撞模式 | `motion_configuration({'collision_mode':'full'/'robot_only'})` | 准备快照和获取 planner 均在同一作用域；robot_only 保留自碰撞及机器人保护，跳过普通场景/attachment |
| 批量路径 FK | `planner.fk_batch(joint_names, positions_NxJ)` | 返回 N×4×4 world TCP；全部采样一次 GPU FK；不提供 IK 或路径成功结论 |

所有输入关节必须具名，米/弧度；目标默认 `T_world_TCP`。
`CollisionPolicy` 是请求声明，不能凭允许表任意关闭场景对象；实际启用的
几何由阶段快照决定，typed adapter 会拒绝不兼容声明。每阶段从真实状态
构造新请求，不能缓存上一阶段的起点或误把预测状态当作实测。

### 推荐生命周期

```python
from gene_agent.motion import CuroboPlannerAdapter, PlannerMode
from gene_agent.motion.curobo.session import CuroboPlannerPool
from gene_agent.motion.curobo import SynchronousPlanner

with CuroboPlannerPool(planner_factory=SynchronousPlanner, max_entries=4) as pool:
    # 每阶段重新准备 settings/snapshot；pool 只复用兼容的求解器和图。
    lease = pool.acquire(settings, snapshot,
                         planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK,
                         random_seed=123)
    adapter = CuroboPlannerAdapter(lease.planner, group=group,
                                  scene_revision=revision)
    result = adapter.plan(request)  # request 的 scene/seed/mode 必须与实际一致
    cache_evidence = lease.to_dict()
```

pool 结构身份继续包含机器人、锁关节、场景几何、碰撞策略和 attachment。
兼容的动态 pad 球及 world pose 原位更新，更新失败丢弃实例重建。
`with` 在正常/异常退出均释放缓存；Task/Skill 自行管理同一 pool 的生命周期，
不要每个微小阶段重建。旋钮原来的每段冷构造现改为一个 Skill pool，且显式
`reset_seed()` 保持原冷构造的采样起点；其他调用方原有采样策略未改。
robot_only 的 grasp/pregrasp 筛选若共享实例，在首次 pregrasp 求解前再重置
一次；后续分块保持连续采样，避免前一轮 batch 消耗影响原先独立的筛选。

### 本轮保持行为的改动

- 直线和 pivot 后检用批量 FK 代替逐点 GPU 往返，终点直接复用最后一项。
  全部检查点、误差上限、碰撞后检仍保留。
- batch IK 按活动关节名 reorder 后再读取 batch/seed，避免 Pika 锁定指关节混入七轴解。
- typed 结果保留 collision mode、环境/attachment 开关、起点和路径碰撞证据，
  失败结果保留近似候选筛选/重规划尝试。
- batch 耗时包含全部后检查；近似规划耗时包含严格失败与全部替代尝试。
  world-obstruction 诊断 solver 在 finally 释放，保持原诊断流程。
- 直线指标新增 `actual_path_length_m`、`direct_distance_m`，便于看是否绕路。
  旧 `path_length_m` 的直距数值保留兼容；没有更改路径接受标准。

### 实测与未改动的边界

同场景、seed=123、同完整微波炉纯规划入口，禁止物理步进/渲染，两模式分别
进行一次前后对照。阶段成功均7/7，旋钮候选、目标、顺序与轨迹长度一致。

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| full 整轮纯规划 | 96.76 s | 75.32 s（减少22.2%） |
| robot_only 整轮纯规划 | 71.24 s | 47.57 s（减少33.2%） |
| full solver wrapper 构造数 | 28 | 16 |
| robot_only solver wrapper 构造数 | 26 | 13 |
| full 旋钮分区 | 28.48 s | 9.70 s |
| robot_only 旋钮分区 | 27.73 s | 6.14 s |
| 128点热 FK | 37.86 ms | 0.38 ms |
| 固定短直线热规划 | 83.52 ms | 70.30 ms |

128点 FK 数值与短直线关节轨迹完全一致；完整旋钮轨迹最大关节差分别
2.53e-5 / 2.11e-5 rad，两者全部原始检查通过。此处“保行为”指约束、目标、
候选选择、阶段和成功条件一致，不承诺 GPU 优化轨迹逐位一致。
证据在 `runs/tmp/curobo_review/20261030/{comparison.json,before,after,gpu}`。
相关 Motion/Skill/微波炉/清扫/真机离线测试共658项通过。
独立终审补充了共享实例采样序列回归，最终相关214项通过。该修正后的
robot_only 整轮复测54.89 s（比基线减少22.9%），仍7/7；旋钮5.78 s，
构造13次，候选/目标/阶段与上述最大关节差不变，证据在同目录
`reviewed/{comparison.json,robot_only/timing.json}`。两次优化后整轮耗时
47.57–54.89 s，尚非多轮统计分布，不能承诺固定延迟。
不代表完整 MuJoCo 物理或真机回归，也不外推为位置分布成功率。

本轮有意保留：普通 PLAN 不承诺最短；真正 GPU batch trajectory 尚未接入；
batch IK 与串行 IK 可能命中不同候选，不自动互换；exact-size batch solver
缓存尚无独立数量上限；局部旋转候选与名义近似目标的现有优先级未改。
这些涉及策略、采样或新能力，不能包装成等价性能优化。后续应分别验证。
抓碗/取碗等旧入口仍有重复构造；本轮仅迁移旋钮生命周期，没有宣称消除
所有 Task 初始化开销。后续按相同基线逐个迁移，避免同时改动多条调用链。

### 抓碗放入与取出入口的后续复用（2026-10-30）

`run_insert_bowl_skill_planning`、`run_extract_bowl_skill_planning` 默认在一次
Skill 调用内持有容量4的现有 pool，正常/异常退出均释放；可选传入 `pool`
时由调用方负责结束生命周期。`object_transfer._planner` 的内部阶段代理
首次实际调用才获取实例并 `reset_seed()`，避免预建下一阶段时切换仍在使用
的快照或采样状态。该代理只用于当前单向同步流水线，不能在后阶段启动后
回头使用前阶段 adapter；它不是任意交错调用的通用 session。

阶段缓存记录在报告 `planner_cache`，每阶段保存独立 lease 字典副本，
不能从共享 planner 读取已被后续阶段覆盖的命中状态。
抓碗报告新增 `visual_attachment.T_world_tcp_at_attach`，保存实际 approach
终点已经算过的 TCP；放入时直接恢复原 link_7 附着球，省去只用于 FK 的
实例。缺该字段的旧报告仍按原实际 approach 终态求 FK。

同场景、seed123、CUDA Graph开启、禁止物理与渲染，各模式一次顺序前后测量：

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| full 完整纯规划 | 73.17 s | 72.07 s |
| full wrapper 构造数 | 16 | 15 |
| robot_only 完整纯规划 | 43.77 s | 38.09 s |
| robot_only wrapper 构造数 | 13 | 8 |
| robot_only 抓碗＋放入分区 | 12.33 s | 9.46 s |
| robot_only 取出＋放下分区 | 8.28 s | 5.09 s |

两种模式前后均7/7。碗候选、阶段、目标和轨迹长度一致；full 碗轨迹关节
全等，robot_only 最大差6.01e-5 rad，原路径和碰撞复核均通过。full仅少一次
轻量FK实例，1.5%的整轮时间差不足以证明稳定提速；robot_only减少五次构造。
保留 full 下不同碰撞拓扑/attachment 的必要独立实例，以及抓取候选中交错
使用的 strict/contact 实例，没有为降低构造数合并它们。

证据：`runs/tmp/curobo_bowl_batch/20261030/comparison.json` 及
`{baseline_clean,after}/{full,robot_only}/timing.json`。第一次基线有短暂GPU
进程重叠，保留在 `before/` 并标为无效，表中使用重新顺序测量的 baseline_clean。
相关Motion/Skill/微波炉回归466项通过，最终碗/缓存专项60项通过。
这仍是纯规划验收，未重新执行物理任务或真机。

## 当前默认（2026-10-29）

正式规划默认使用 `PlannerMode.CUDA_GRAPH_BENCHMARK` 开启 CUDA Graph；`MotionRequest`、`SynchronousPlanner`、现场规划context及正式应用入口保持一致。枚举名保留以兼容历史记录，现已不是仅限benchmark使用。

`CuroboPlannerPool` 默认 `reuse_gripper_geometry=True`；`CuroboRuntimeBridge` 在自身生命周期内持有容量1的pool，逐段查询真实状态、导出当前碰撞世界及attachment，不缓存旧执行起点。`plan["planner_cache"]` 保存命中/获取耗时/重建原因。结束使用可调用 `bridge.close()` 或 `pool.clear()` 释放solver；门和真机Task已有显式清理流程。

动态夹爪同步的结构保护及测量见 [性能记录](microwave_planning_performance.md)。不兼容结构仍正常重建；使用新默认不代表所有任务的动作或真机结果已经验收。诊断可显式传 `planner_mode=PlannerMode.DEVELOPMENT`；需要对照动态夹爪缓存时传 `reuse_gripper_geometry=False`。`check_motion.py` 也默认开启图，关闭时用 `--planner-mode development`。图捕获的冷启动开销仍须单独报告。

推广默认后的完整微波炉验收：`runs/tmp/motion_defaults/20261029/task/summary.json`，fresh reset成功13/13，最终门关闭、旋钮45°、碗在桌面且无attachment；仍为混合仿真。记录中的模式全部为 `cuda_graph_benchmark`。该轮墙钟183.45s，与CPU回归同时运行，且新增了其他阶段的Graph模式，不作为与此前139.06s的受控速度对照。Motion/Skills专项187项通过。

边界：一次性直接构造planner的旧入口会使用新Graph默认，但不会自动跨实例共享pool；需要复用时应持有明确有界的pool。当前Bridge的“全部候选失败”早返回尚缺 `planner_cache` 诊断字段；部分旧应用未显式调用新增 `close()`，bridge仍按对象生命周期持有solver，可能在长continuation中延长驻留。尚未发现由此导致的OOM或永久泄漏，本轮未扩展为旧应用资源管理重构。

以下保留2026-10-04的独立诊断说明及历史结果；其中显式 `development` 示例用于关闭图的对照，不代表现行默认。

受众：开发 Agent。状态：2026-10-04 已实际执行 GPU FK、IK、单次完整规划和已知碰撞负例。
这是一组独立诊断实验，尚未接通任务场景的完整碰撞模型或 MuJoCo 执行。

## 接口和边界

```python
from gene_agent.motion import PlannerMode, load_realman_motion_config, SynchronousPlanner
settings = load_realman_motion_config("right", robot_metadata)
planner = SynchronousPlanner(
    **settings,
    scene_model={"cuboid": {"diagnostic_far_box": {
        "dims": [.1, .1, .1], "pose": [3, 3, 3, 1, 0, 0, 0]
    }}},
    scene_scope="独立单臂诊断；未包含另一臂、本体、桌面和任务物体",
    planner_mode=PlannerMode.DEVELOPMENT,
    random_seed=123,
)
T_world_tcp = planner.fk(joint_names, positions_rad)
result = planner.solve_ik(joint_names, positions_rad, target, target_frame="world_tcp")
result = planner.plan(joint_names, positions_rad, target, target_frame="world_tcp")
```

关节输入必须具名、完整且有限，内部按配置映射顺序，不按七轴长度猜测。
`scene_model` 使用 cuRobo 原生字典，几何 pose 已在该组 URDF base 表达，四元数为 wxyz；
当前模块不接受“world下的几何但不知道frame”的输入。
`scene_scope` 强制声明具体覆盖范围，未集成完整场景的结果不能拿去执行任务。
结果包含 success、status、joint_names、positions_rad、duration_s、scene_scope、raw_result/raw_error。
完整轨迹额外给出 interpolation_dt_s=0.025；错误保留 traceback，不笼统归类为 IK 失败。
接口同步执行一次；没有异步队列、碰撞关闭、姿态放宽、多重 fallback 或重试层。

坐标完全复用已验证机器人 metadata：
`T_base_link_goal = inv(T_world_base) @ T_world_TCP_goal @ inv(T_link_TCP)`。
FK 使用反方向组合。world 默认为 right SDK base；link_7 是规划 frame，不等同 SDK flange。
两臂 FK 验证都使用各自 metadata，没有把左臂隐式当作右臂。

## 配置保留和必要修复

仅读新仓库的 Realman URDF、碰撞球 YAML。保留 collision_link_names、collision_spheres、
self_collision_ignore 和 self_collision_buffer 原值，通过测试逐项比较。
只把 URDF 和 asset_root_path 解析到新仓库的绝对路径，没有依赖旧项目运行时。
不迁移 attachment 空链接、旧批量完整规划、nearest IK、临时关闭碰撞及 CUDA graph 恢复机制。
当前原生 MotionPlanner 默认内部 IK/TrajOpt 流程仍会使用后端自身的数值种子，这不是旧层重试。

固定版本：仓库 `third_partys/curobo` 提交 `78fd485fa82d9b9a063fb4985e371814587e666a`。
实际采用当前 `Kinematics`、`InverseKinematics`、`MotionPlanner` API。
发现 `KinematicsCfg.from_robot_yaml_file(dict)` 会原地改写传入配置并加入 robot_cfg 包装，
后续复用给 IK/Planner 导致 `RobotCfg.__init__ unexpected keyword argument robot_cfg`。
修复仅为向各后端传递独立 deepcopy，保留失败原始记录，不对版本做猜测兼容。

配置：self_collision_check=True，IK种子16，TrajOpt种子4，max_attempts=1，
随机种子显式传入；无额外姿态约束。`development` 模式固定关闭 CUDA Graph，
单次调用的 `duration_s` 包含首次后端对象构造和内核初始化，不能当纯求解热启动基准。

`cuda_graph_benchmark` 模式固定启用 CUDA Graph。验收 CLI 对同一固定 shape 和输入分别
记录 warmup 与 measurement，首次图捕获不会混入测量均值。该模式用于保留复现论文速度的
能力，并不改变请求语义，也不加入 retry 或 fallback。

## 已执行验证

运行环境：curobo conda、PyTorch 2.5.1+cu121、RTX 4090；具体版本和代码哈希写入报告。

```bash
PYTHONUTF8=1 PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  -m curobo.examples.getting_started.inverse_kinematics --mode single --test
PYTHONUTF8=1 PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/diagnostics/check_motion.py --output runs/motion-baseline
PYTHONUTF8=1 PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/diagnostics/check_motion.py --output runs/motion-cuda-graph \
  --planner-mode cuda_graph_benchmark --warmup-runs 2 --measure-runs 5
PYTHONUTF8=1 PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  -m pytest tests/motion/test_motion_contract.py -q
```

注意官方 `--test` 会执行全部测试，虽然命令带 `--mode single`。实际结果：single IK成功，
100目标批量93/100，单碰撞IK成功，增加障碍后成功，碰撞批量50/50。
官方 FK 已由本项目环境检查完成，见 environment.md；这里只把官方 IK 作为独立环境锚点。

Realman GPU FK：左右各5组（home与4组固定seed ±0.15rad扰动），对MuJoCo TCP，
最大位置差4.60e-8m，旋转矩阵元素最大差1.31e-7。

Realman目标：右臂记录home，将首关节增加0.08rad，FK生成目标TCP，故目标有明确关节空间见证。
净空正例场景：只有URDF base [3,3,3]处0.1m cuboid，加保留的机器人自身碰撞配置。
实际IK成功，FK复核TCP位置误差5.24e-8m。
完整规划一次成功，21个waypoint，首点关节最大误差2.28e-7rad、末TCP位置误差8.11e-8m。
这些数值为诊断位姿，不能解释为任务目标成功率。

2026-10-06 回归产物 `runs/motion-phase1-gpu` 再次通过。CUDA Graph 产物
`runs/motion-phase1-cuda-graph-final` 在同一 RTX 4090 上通过；加入独立离散碰撞复核后，
IK 五次稳定测量中位数约7.54ms，首次初始化和图捕获约1.33s。该数字只描述当前固定诊断
目标，不能外推到任务场景或批量吞吐。

已知碰撞负例：另建一个URDF base原点处4m边长cuboid，明确包住整个机械臂。
相同目标 IK 返回 solver_failed 且无Python异常；脚本要求这个负例确实被拒绝。
负例不是在失败后改小障碍，而是独立验证碰撞检查是否生效。
起点先由独立 `RobotCollisionChecker` 检查；成功 IK 解和完整离散轨迹也再次复核。
若 cuRobo 在没有返回可复核解时只给出 `solver_rejected`，适配器保守报告
`constraint_violation`，不虚构 IK 无解或碰撞类别，也不启动关闭碰撞的诊断 fallback。

6个CPU契约测试通过：任意关节数重排、重复/缺失/非有限/错误形状拒绝、冻结碰撞配置保持。
实验脚本还将IK和规划末端FK、轨迹首点误差作为成功门槛，而非仅检查后端success字段。

结构化证据：[motion_result.json](../../../runs/motion-baseline/motion_result.json)，
[运行日志](../../../runs/motion-baseline/run.log)，[官方IK日志](../../../runs/motion-baseline/official_ik.log)，
[配置原地变更失败记录](../../../runs/motion-baseline/config-mutation-failure.json)。
报告包含完整解析配置、原始solver结果、seed、目标见证、代码哈希与版本。

## 尚未验证

- scene_revision、完整桌面/碗/微波炉、另一臂和本体的同步碰撞快照。
- cuRobo碰撞球与MuJoCo凸包接触的一致性、真实物体几何覆盖与非相邻自碰撞ignore合理性。
- 当前轨迹在MuJoCo位置伺服下的跟踪/接触；任何真实机器人运行。
- 夹爪开合、attachment、抓碗或旋钮任务成功。

下一步先建立frame明确的完整场景快照适配，再按已知碰撞/净空样例校验，再接同步Skill。
不得把此独立诊断场景替换为任务场景中的静默空场景。

### ALOHA 与持物几何的精确复用（2026-11-03）

ALOHA 的实际指关节值可在源模型、源球、关节集合和其他锁关节不变时原位更新；每次仍验证快照哈希与实际锁定值。`held_object.spheres_link` 在球数、半径、顺序/来源、物体身份及 finger link 不变时可更新球心。pool 使用独立的精确 held 内容身份检测更新，不能只删结构 key 而沿用旧几何。

更新复用已有 `KinematicsParams` 同步路径，覆盖运动学、MotionPlanner、独立 IK、batch IK 和碰撞检查器，清除旧起点验证缓存并重置采样；任何部分更新异常都销毁实例并重建。不同接触策略仍是独立缓存项。`reuse_gripper_geometry=False` 同时关闭这一轮新增的持物中心复用，便于冷模型对照。

冻结 GPU 检查使用同一碰撞策略，三份实测夹爪/持物关系：各 solver 在三组关节状态下的球与 fresh 模型差值为0、碰撞有效性一致、球buffer地址不变，三次规划均通过。后两次配置获取与规划合计约1.48s；不等于真实场景更新或接触策略切换均能达到这一延迟。证据：`runs/tmp/novel_tool/hammer/performance/dynamic-geometry-gpu.json`。

正常原生初态300004的新reset完整任务原评分成功；观测/目标到轨迹交付（不含仿真执行、感知推理及此前候选筛选），approach 6.49→3.50s，lift 7.00→2.92s，align/strike仍约6.9s。后两阶段首次使用不同策略仍需冷初始化；当前不是低于1s的通用闭环规划器。完整20例成功率尚未对这一性能版本重测。

困难原生初态300012也从fresh reset完整成功，保留右臂自碰撞候选拒绝与左臂候选3选择；approach/lift交付延迟3.48/2.89s。仅两个开发初态验收，不更新先前19/20的正式成功率。

### 同几何 world 更新与阶段接触规则切换

`SynchronousPlanner.update_scene` 对相同障碍集合、几何和文件路径仅原位更新
world pose，覆盖 MotionPlanner、IK、batch IK、独立碰撞 checker 和局部接触
validator 的分区。几何或路径变化仍完整重载；缺少障碍或更新异常由 pool
丢弃实例后重建。夹爪/持物几何的更新沿用原有独立校验。

`update_contact_policy` 完整替换阶段规则、重建 validator 并清空起点缓存。
支持该能力的 planner 可在 pool 中复用 GPU 求解资源；不会合并不同阶段的
允许接触集合，策略变化会重置采样。显式 `reuse_contact_policy=False` 可用于
旧生命周期对照；第三方 factory 未声明支持时仍按策略隔离。
运行记录 `rebuild_reason=contact_policy_updated` 表明资源命中但规则已替换。

本轮性能和 GPU 正反例证据见临时运行目录
`runs/tmp/novel_tool/hammer/performance/`，完整验证结果见本轮交付报告；
临时数据不作为其他任务、任意策略或真机已验收的声明。

本轮两个 fresh reset 开发样本（300004/300012）均通过原 `check_success`。
日志中的规划求解+setup累计从48.81/88.05s降为32.42/66.53s，包含候选预检及
失败尝试，排除物理、场景快照、IPC。正常样本实际动作阶段已有观测/目标至
轨迹返回约0.82/1.92/1.90/1.95/1.97s（pregrasp/approach/lift/align/strike）；
pregrasp之前已有预检，不能把该值当启动到首动作时间。困难样本align仍有一次
失败，累计准备+求解9.88s。GPU策略切换六个碰撞/净空正反例与fresh模型一致；
这不是正式20次成功率，也未重跑微波炉任务。
